#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"
#include "../src/vision_process.h"

namespace chatllm::gemma::v1
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        float rope_theta;
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

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *chat_encoder);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_start, bool add_end);
    public:
        int start_of_turn_token_id;
        int end_of_turn_token_id;
        int nl_token_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<BaseConfig, Embedding, RMSNorm, GemmaBlock, int, int, int, int, int, int> ModelClass;

    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA);

        void load(ModelLoader &loader) override;

    public:
        BaseConfig config;
    };
}

namespace chatllm::gemma::v2
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        int query_pre_attn_scalar;
        int sliding_window;

        float rope_theta;
        float final_logit_soft_capping;
        float attn_logit_soft_capping;
    };

    typedef v1::Tokenizer Tokenizer;

    template <int sliding_window_len> class Gemma2SWASelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        Gemma2SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    template <int sliding_window_len> class Gemma2SWABlock : public LMBlock4<RMSNorm, Gemma2SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
    {
    public:
        Gemma2SWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock4<RMSNorm, Gemma2SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
                    (ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    const int SLIDING_WINDOW_LEN = 4096;

    typedef Gemma2SWABlock<SLIDING_WINDOW_LEN> Gemma2SWABlock4k;

    class Gemma2FullBlock : public LMBlock4<RMSNorm, GemmaSelfAttention, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
    {
    public:
        Gemma2FullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock4(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class TanhScaling: public Block
    {
    public:
        TanhScaling(float scale_pre, float scale_post) : scale_pre(scale_pre), scale_post(scale_post) {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    public:
        const float scale_pre;
        const float scale_post;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA2);

        void load(ModelLoader &loader) override;
    public:

        bool is_sliding(int layer_id)
        {
            return layer_id % 2;
        }

    public:
        BaseConfig config;
        TanhScaling logits_pp;
        TanhScaling attn_scores_pp;
    };
}

namespace chatllm::gemma::siglip
{
    struct Config
    {
        ggml::type dtype;
        int hidden_size;
        int image_size;
        int intermediate_size;
        int num_attention_heads;
        int num_channels;
        int num_hidden_layers;
        int patch_size;
        int mm_tokens_per_image;
        float image_mean[3];
        float image_std[3];
        bool vision_use_head;

        int max_num_crops;
        int min_crop_size;
        float min_ratio_to_activate;
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int num_channels, int embed_dim, int image_size, int patch_size);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int num_patches;
        const int num_positions;
        Conv2D patch_embedding;
        ggml::tensor *position_embedding;
    };

    typedef TheBiasedGELUMLP SigMLP;

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *vision_outputs) override;

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int mm_tokens_per_image;
        const int patches_per_image;
        const int tokens_per_side;
        const int kernel_size;
        Linear  input_projection;
        RMSNorm soft_emb_norm;
    };

    class SigSelfAttention : public BaseCachelessAttention
    {
    public:
        SigSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true)
        {
            causal = false;
        }
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, SigSelfAttention, LayerNorm, SigMLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        bool is_loaded(void) const
        {
            return loaded;
        }

    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        LayerNorm  post_layernorm;
        MultiModalProjector multi_modal_projector;
    protected:
        bool loaded;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);
        bool load(ModelLoader &loader);

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config);

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        std::unique_ptr<VisionTransformer> vis_model;
        BackendContext backend_context;
        const size_t GRAPH_SIZE;
        InitContext _ctx; // weight context
        const int n_threads;
        std::string model_gpu_layers;
    public:
        Config vis_config;
    };

}

namespace chatllm::gemma::v3
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        int query_pre_attn_scalar;
        int sliding_window;
        int sliding_window_pattern;

        float rope_local_base_freq;
        float rope_theta;
        float rope_scaling_factor;
    };

    class ChatHistoryEncoder : public v1::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    protected:
        bool append_image(const vision::image_pixels_t pixels, const int w, const int h, std::vector<int> &ids) const;
    public:
        const siglip::Config *vis_config = nullptr;
        int MAX_PATCH_NUM = 0;
        bool vit_loaded = false;
    };

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int boi_token_id = -1;
        int eoi_token_id = -1;
        bool do_pan_and_scan = false;
    };

    template <int sliding_window_len> class Gemma3SWASelfAttention : public QKNormedAttention<RMSNorm, SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        Gemma3SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<RMSNorm, SlidingWindowAttentionImpl<sliding_window_len>>
                (ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    template <int sliding_window_len> class Gemma3SWABlock : public LMBlock4<RMSNorm, Gemma3SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
    {
    public:
        Gemma3SWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock4<RMSNorm, Gemma3SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
                    (ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class Gemma3FullSelfAttention : public QKNormedAttention<RMSNorm, BaseAttention>
    {
    public:
        Gemma3FullSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<RMSNorm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    class Gemma3FullBlock : public LMBlock4<RMSNorm, Gemma3FullSelfAttention, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
    {
    public:
        Gemma3FullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock4(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    typedef Gemma3SWABlock<512> Gemma3SWABlock512;
    typedef Gemma3SWABlock<1024> Gemma3SWABlock1024;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA3);
        void load(ModelLoader &loader) override;

        bool load_more(const json::JSON &config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;

        void before_generate(const GenerationConfig &gen_config) override;
    public:

        bool is_sliding(int layer_id) const;
        int number_of_sliding_window_layer(void) const;

    public:
        BaseConfig config;
        const int sliding_window;
        const int sliding_window_pattern;
        siglip::VisualEmbeddingGeneration visual;
    };
}
