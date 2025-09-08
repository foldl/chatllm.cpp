#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"
#include "llama.h"

namespace chatllm::phi::v2
{
    class QAHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class Phi2Tokenizer : public BaseTokenizer
    {
    public:
        Phi2Tokenizer(const BaseConfig &config);

        Phi2Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

    public:
        std::vector<int> qa_terminate_seq1;
        std::vector<int> qa_terminate_seq2;
        std::vector<int> qa_terminate_seq3;
        std::vector<int> qa_terminate_seq4;
        int qa_seq_max_len;
        std::vector<int> chat_terminate_seq;
    };

    class Phi2ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<BaseConfig, Embedding, LayerNorm, Phi2Block, int, int, int, int, int> ModelClass;
    public:
        Phi2ConditionalGeneration() = default;
        Phi2ConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type);

    public:
        BaseConfig config;

    protected:
        bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override;
    };

    namespace v1
    {
        struct Config : public BaseConfig
        {
        };

        class Tokenizer : public Phi2Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);
        };

        class ConditionalGeneration : public Phi2ConditionalGeneration
        {
        public:
            ConditionalGeneration() = default;
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);

            void load(ModelLoader &loader) override;
        };
    }

    namespace v2
    {
        struct Config : public BaseConfig
        {
            int rope_dim;
            float rope_theta;
        };

        class Tokenizer : public Phi2Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config);
        };

        class ConditionalGeneration : public Phi2ConditionalGeneration
        {
        public:
            ConditionalGeneration() = default;
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

            void load(ModelLoader &loader) override;
        };
    }
}

namespace chatllm::phi::v3
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int original_max_position_embeddings;
        int sliding_window;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    public:
        bool add_bos = true;
    };

    class Phi3Tokenizer : public BaseTokenizer
    {
    public:
        Phi3Tokenizer(const BaseConfig &config);
        Phi3Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &msg, std::vector<int> &ids, int type_token_id, int end_token_id = -1);
    public:
        int system_token_id;
        int user_token_id;
        int assistant_token_id;
        int end_token_id;
        int nl_token_id;
        bool append_nl_after_end_tok;
    };

    typedef Phi3Tokenizer Tokenizer;

    template <int sliding_window_len> class Phi3SelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        Phi3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        Phi3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    template <int sliding_window_len> class Phi3Block : public LMBlock1<RMSNorm, Phi3SelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>
    {
    public:
        Phi3Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1<RMSNorm, Phi3SelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        Phi3Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, Phi3SelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    // https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/25
    const int SLIDING_WINDOW_LEN            =  2048;

    typedef Phi3Block<SLIDING_WINDOW_LEN> Phi3Block4k;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<Phi3Block4k>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI3);

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length);
    };
}

namespace chatllm::phi::v3_su
{
    const int MAX_FACTOR_LEN = 128;

    struct Config : public BaseConfig
    {
        int max_position_embeddings;
        int num_key_value_heads;
        int original_max_position_embeddings;
        int sliding_window;
        int rope_scaling;
        float rope_theta;
        float short_factor[MAX_FACTOR_LEN];
        float long_factor[MAX_FACTOR_LEN];
    };

    typedef v3::Phi3Tokenizer Tokenizer;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<Phi3SUBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI3_SU);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length, bool tie_lm_head = false);
    };
}

namespace chatllm::phi::v3_su2
{
    typedef v3_su::Config Config;

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
    };

    typedef v3_su::ConditionalGeneration ConditionalGeneration;
}

namespace chatllm::phi::v3_su3
{
    struct Config : public v3_su2::Config
    {
        float short_mscale;
        float long_mscale;
    };

    typedef v3_su2::Tokenizer Tokenizer;

    class ConditionalGeneration : public v3_su2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI3_SU3);
    };
}

namespace chatllm::phi::v3_moe
{
    struct Config : public v3_su3::Config
    {
        int num_experts_per_tok;
        int num_local_experts;
    };

    typedef v3_su3::Tokenizer Tokenizer;

    struct mixing_param
    {
        float jitter;
    };

    void _compute_forward_custom_sparsemixer(ggml::tensor * dst, const ggml::tensor * a, const ggml::tensor * b, const ggml::tensor * c, int ith, int nth, void *userdata);

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class Phi3SparseMoE : public BaseSparseMLP
    {
    public:
        Phi3SparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
        // TODO: WIP
#if (0)
        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            const int64_t hidden_size = hidden_states->ne[0];
            const int64_t qlen        = hidden_states->ne[1];
            const int n_expert = num_local_experts;

            ggml::tensor * scores = gate.forward(ctx, hidden_states); // [qlen, num_experts]

            ggml::tensor * selected_experts = ggml::new_tensor_2d(ctx, GGML_TYPE_I32, qlen, num_experts_per_tok);  // [qlen, num_experts_per_tok]
            ggml::tensor * weights = ggml::new_tensor_3d(ctx, ggml::type_of(scores), 1, num_experts_per_tok, qlen); // [1, num_experts_per_tok, qlen]

            selected_experts = ggml::map_custom3_inplace(ctx, selected_experts, weights, scores, _compute_forward_custom_sparsemixer, 1, param);

            return forward_with_experts(ctx, hidden_states, selected_experts, weights);
        }
#endif
    public:
        mixing_param *param;
    };

    class Phi3SUSelfAttentionBiased : public Phi3SUSelfAttention
    {
    public:
        Phi3SUSelfAttentionBiased(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length);
    };

    template<int num_local_experts, int num_experts_per_tok> class Phi3MoEBlock : public LMBlock1<LayerNorm, Phi3SUSelfAttentionBiased, LayerNorm,
                        Phi3SparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        Phi3MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<LayerNorm, Phi3SUSelfAttentionBiased, LayerNorm,
                       Phi3SparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, LayerNorm, Phi3MoEBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK>, int, int, int, int, int> ModelClass;
    public:
        _ConditionalGeneration() = default;

        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : Base(type, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + 2 + config.num_hidden_layers * (11 + 3 + 5);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_local_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            Base::transformer = new ModelClass(
                                &w_ctx_, config, true,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = Base::get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.config(&w_ctx_, config.original_max_position_embeddings, config.rope_theta,
                                 config.short_mscale,
                                 config.long_mscale,
                                 config.hidden_size / config.num_attention_heads / 2,
                                 config.short_factor,
                                 config.long_factor);
            }

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size()) << "corrupted model weights";
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w2.weight", transformer->layers[i].mlp.experts.down.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w1.weight", transformer->layers[i].mlp.experts.gate.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w3.weight", transformer->layers[i].mlp.experts.up.weight);

                loader.read_tensor(layer_prefix + "block_sparse_moe.gate.weight",
                                transformer->layers[i].mlp.gate.weight);

                loader.read_tensor(layer_prefix + "input_layernorm.weight",
                                transformer->layers[i].input_layernorm.weight);
                loader.read_tensor(layer_prefix + "input_layernorm.bias",
                                transformer->layers[i].input_layernorm.bias);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                transformer->layers[i].post_attention_layernorm.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",
                                transformer->layers[i].post_attention_layernorm.bias);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);
            loader.read_tensor("lm_head.bias",   dynamic_cast<Linear *>(transformer->lm_head)->bias);
        }

    public:
        Config config;
    };

    const int NUM_EXPERTS                   =  16;
    const int EXPERTS_PER_TOK               =  2;

    typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_PHI3_MOE> ConditionalGeneration;
}

namespace chatllm::phi::v4
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode_role(const std::string &msg, std::vector<int> &ids);

        void encode(const std::string &msg, std::vector<int> &ids, bool add_end_tok = true);

    public:
        int im_start_token_id;
        int im_sep_token_id;
        int im_end_token_id;
    };

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
    };
}

namespace chatllm::phi::v4_mini
{
    typedef v3_su::Config Config;

    class Tokenizer : public v3_su::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    class ConditionalGeneration : public v3_su::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI4_MINI);
    };
}