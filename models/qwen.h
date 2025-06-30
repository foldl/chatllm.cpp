#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::qwen
{
    namespace v1
    {
        struct Config : public BaseConfig
        {
            int seq_length;
            int rope_dim;
            int flags;
            float rotary_emb_base;
        };

        class ChatHistoryEncoder : public BaseHistoryEncoder
        {
        public:
            void append_sys_prompt(std::vector<int> &ids) const override;
            void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
            void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
            void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
            void append_user_opening(int round_idx, std::vector<int> &ids) const override;
        };

        extern ChatHistoryEncoder _chat_encoder;

        class Tokenizer : public BaseTokenizer
        {
        public:
            Tokenizer(const Config &config);

            Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                    BaseHistoryEncoder *qa_encoder = nullptr,
                    BaseHistoryEncoder *completion_encoder = nullptr);

            size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

            void encode(const std::string &text, std::vector<int> &ids) const override;

            bool is_special_id(int id) const override;

        public:
            void encode(const std::string &text, std::vector<int> &ids, bool add_eos) const;

        public:
            void encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const;

        protected:
            virtual size_t do_load(tokenizer::DataReader *buffer, int n_vocab);

        public:
            int im_start_token_id;
            int im_end_token_id;
            int nl_token_id;
        };

        class ConditionalGeneration : public BaseModelForConditionalGeneration
        {
        public:
            typedef Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int> ModelClass;
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_QWEN);

            void load(ModelLoader &loader) override;

        public:
            Config config;
        };
    }

    namespace v2
    {
        struct Config : public BaseConfig
        {
            int num_key_value_heads;
            int sliding_window;
            float rope_theta;
        };

        class Tokenizer : public v1::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);

            Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
                : v1::Tokenizer(config, encoder)
            {}

            size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        };

        class ConditionalGeneration : public BaseModelForConditionalGeneration
        {
        public:
            typedef Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int> ModelClass;
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_QWEN2, bool tie_embeddings = false);

        public:
            Config config;

        private:
            const bool tie_embeddings;
        };
    }

    namespace v2_tie
    {
        typedef v2::Config Config;
        typedef v2::Tokenizer Tokenizer;

        class ConditionalGeneration : public v2::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
                : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_QWEN2TIE, true)
            {}
        };
    }

    namespace v2_moe
    {
        struct Config : public BaseConfig
        {
            int num_key_value_heads;
            int moe_intermediate_size;
            int shared_expert_intermediate_size;
            int sliding_window;
            int num_experts_per_tok;
            int num_experts;
            int norm_topk_prob;
            float rope_theta;
        };

        typedef v2::Tokenizer Tokenizer;

        class ConditionalGeneration : public ModelProxy
        {
        public:
            ConditionalGeneration() = default;

            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        };
    }

    namespace audio_tower
    {
        struct Config
        {
            ggml::type dtype;
            int num_mel_bins;
            int encoder_layers;
            int encoder_attention_heads;
            int encoder_ffn_dim;
            int d_model;
            int scale_embedding;
            int max_source_positions;

            int audio_token_index;

            // mel features
            int chunk_length;
            int feature_size;
            int hop_length;
            int n_fft;
            int n_samples;
            int nb_max_frames;
            int sampling_rate;
        };

        class AudioSelfAttention : public BaseCachelessAttention
        {
        public:
            AudioSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length);
        };

        class AudioTransformer : public Block
        {
        public:
            typedef LMBlock1<LayerNorm, AudioSelfAttention, LayerNorm, TheBiasedGELUMLP> LayerBlock;

            AudioTransformer(InitContext *ctx, const Config &config, int lm_hidden_size);

            int64_t get_param_num(bool effective_only) const override;

            void load(const std::string &path, TensorLoader *loader) override;

            ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

            bool is_loaded(void) const;

        protected:
            Embedding embed_positions;
            Conv1D    conv1;
            Conv1D    conv2;
            LayerNorm layer_norm;
            Linear    multi_modal_projector;
            std::vector<std::unique_ptr<LayerBlock>> layers;
        protected:
            bool loaded;
        };

        class AudioEmbeddingGeneration
        {
        public:
            AudioEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);

            bool load(ModelLoader &loader);

            bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &json_config);

            void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);

        protected:
            bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &audio, std::vector<uint8_t> &buf);
        protected:
            std::unique_ptr<AudioTransformer> model;
            BackendContext backend_context;
            const size_t GRAPH_SIZE;
            InitContext _ctx; // weight context
            std::string model_gpu_layers;
            const int n_threads;
        public:
            Config config;
        };
    }

    namespace v2_audio
    {
        typedef v2::Config Config;

        class ChatHistoryEncoder : public v1::ChatHistoryEncoder
        {
        public:
            void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;

        public:
            const audio_tower::Config *aud_config = nullptr;
            bool aud_loaded = false;
        };

        class Tokenizer : public v2::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);

            void inject_audio_ids(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count);
        public:
            int audio_bos_token_id;
            int audio_eos_token_id;
        };

        class ExtendEmbedding
        {
        public:
            ExtendEmbedding(int extra_ids) : pad_arg(new BlockParams::PadEmbedding(extra_ids, extra_ids)) {}
        public:
            BlockParams::PadEmbedding *pad_arg = nullptr;
        };

        class ConditionalGeneration : public ExtendEmbedding, public v2::ConditionalGeneration
        {
        public:
            typedef v2::ConditionalGeneration Base;
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_QWEN2_AUDIO);

        public:
            bool load_more(const json::JSON &config) override;

            void load(ModelLoader &loader) override;

            void before_generate(const GenerationConfig &gen_config) override;
        public:
            audio_tower::AudioEmbeddingGeneration audio;
        };
    }

    namespace marco_o1
    {
        typedef v2::Config Config;

        class Tokenizer : public v2::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);
        };

        class ConditionalGeneration : public v2::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        };
    }

    namespace qwq
    {
        typedef v2::Config Config;

        class Tokenizer : public v2::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);
        };

        class ConditionalGeneration : public v2::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        };
    }

    namespace ds_r1_distill
    {
        struct Config : v2::Config
        {
            int tie;
        };

        class ChatHistoryEncoder : public BaseHistoryEncoder
        {
        public:
            void append_sys_prompt(std::vector<int> &ids) const override;
            void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
            void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
            void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        };

        class Tokenizer : public BaseTokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);
            Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                    BaseHistoryEncoder *qa_encoder = nullptr,
                    BaseHistoryEncoder *completion_encoder = nullptr);

            size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        public:
            int user_token_id;
            int assistant_token_id;
            int nl_token_id;
        };

        class ConditionalGeneration : public v2::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN);
        };
    }

    namespace v2_5_vl
    {
        const int MROPE_SECTION_MAX = 4;

        struct Config : v2::Config
        {
            int tie_word_embeddings;
            int mrope_section[MROPE_SECTION_MAX];
        };

        class Tokenizer : public v2::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);

            size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        public:
            int object_ref_start_token_id;
            int object_ref_end_token_id;
            int box_start_token_id;
            int box_end_token_id;
            int quad_start_token_id;
            int quad_end_token_id;
            int vision_start_token_id;
            int vision_end_token_id;
            int vision_pad_token_id;
            int image_pad_token_id;
            int video_pad_token_id;
        };

        class ConditionalGeneration : public v2::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        };
    }

    namespace v3
    {
        const int MAX_LAYERS = 128;

        struct Config : BaseConfig
        {
            int num_key_value_heads;
            int head_dim;
            float rope_theta;
            float yarn_scaling_factor;
            int yarn_scaling_original_max_position_embeddings;
            int decoder_sparse_step;
            int moe_intermediate_size;
            int num_experts_per_tok;
            int num_experts;
            int norm_topk_prob;
            int tie_word_embeddings;
            int layer_is_sparse[MAX_LAYERS];
        };

        typedef v2::Tokenizer Tokenizer;

        class ConditionalGeneration : public BaseModelForConditionalGeneration
        {
        public:
            typedef HeterogeneousModel ModelClass;
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
                ModelType type = ModelType::MODEL_TYPE_QWEN3, const bool skip_lm_head = false, int extra_tensors = 0);

        private:
            int get_sparse_layer_num();

            Block *create_layer(InitContext *ctx, int layer_index);

        public:
            Config config;
        };
    }

    namespace ds_r1_distill_v3
    {
        typedef v3::Config Config;

        class Tokenizer : public ds_r1_distill::Tokenizer
        {
        public:
            Tokenizer(BaseConfig config);
        };

        typedef v3::ConditionalGeneration ConditionalGeneration;
    }


    namespace v3_emb
    {
        typedef v3::Config Config;

        class Tokenizer : public v3::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);

            void encode_embedding(const std::string &text, std::vector<int> &ids, EmbeddingPurpose purpose) const override;

        public:
            std::string task;
        };

        class ConditionalGeneration : public v3::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
                ModelType type = ModelType::MODEL_TYPE_QWEN3_Embedding, const bool skip_lm_head = true, int extra_tensors = 0);

            void set_additional_args(const std::map<std::string, std::string> &args) override;
        };
    }

    namespace v3_ranker
    {
        typedef v3::Config Config;

        class Tokenizer : public v3_emb::Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);
            size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

            void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const override;
        public:
            int yes_token_id;
            int no_token_id;
        };

        class ConditionalGeneration : public v3_emb::ConditionalGeneration
        {
        public:
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

            void set_tokenizer(BaseTokenizer *tokenizer) override;
        protected:
            ggml::tensor *yes_no_ids = nullptr;
        };
    }
}