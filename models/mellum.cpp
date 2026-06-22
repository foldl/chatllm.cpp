#include "qwen.h"
#include "../src/chat_encoders.h"

namespace chatllm::mellum
{
    const int MAX_LAYERS = 128;

    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        int sliding_window;
        int moe_intermediate_size;
        int num_experts_per_tok;
        int num_experts;
        int norm_topk_prob;
        int tie_word_embeddings;
        float full_attention_rope_theta;
        float full_attention_factor;
        float full_attention_original_max_position_embeddings;
        float full_attention_beta_fast;
        float full_attention_beta_slow;
        float full_attention_attention_factor;
        float sliding_attention_rope_theta;
        int layer_is_swa[MAX_LAYERS];
        int layer_is_sparse[MAX_LAYERS];
    };

    typedef qwen::v2::Tokenizer Tokenizer;

    class QWen3SelfAttention : public QKNormedRoPEAttention<RMSNormInplace, BaseAttention>
    {
    public:
        QWen3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length);
    };

    template <int sliding_window_len> class SWASelfAttention : public QKNormedRoPEAttention<RMSNormInplace, SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        typedef RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>> BaseBase;
        typedef QKNormedRoPEAttention<RMSNormInplace, SlidingWindowAttentionImpl<sliding_window_len>> Base;
        SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class SparseMoE : public BaseSparseMLP
    {
    public:
        SparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, class SelfAttention> class MoEBlock : public LMBlock1<RMSNorm, SelfAttention, RMSNorm,
        SparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
    {
    public:
        typedef SparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK> MoEMLP;
    public:
        MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size,
                int num_kv_heads,
                int head_dim, int max_length)
            : LMBlock1<RMSNorm, SelfAttention, RMSNorm, MoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size,
            num_kv_heads, head_dim, max_length)
        {}
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = ModelType::MODEL_TYPE_MELLUM);

        void set_tokenizer(BaseTokenizer *tokenizer) override;
    private:
        int get_sparse_full_layer_num() const;
        int get_sparse_swa_layer_num() const;
        int get_dense_layer_num() const;

        Block *create_layer(InitContext *ctx, int layer_index);

    public:
        const Config config;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
        config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const int sparse_full_layers = get_sparse_full_layer_num();
        const int sparse_swa_layers  = get_sparse_swa_layer_num();
        const int dense_layers       = get_dense_layer_num();
        CHATLLM_CHECK(dense_layers == 0);

        const size_t num_tensors = (config.tie_word_embeddings ? 2 : 3)
                                    + sparse_full_layers * 15
                                    + sparse_swa_layers * (15 + 1);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        if (config.tie_word_embeddings)
        {
            transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                nullptr,
                [&](InitContext *ctx, int layer_index) {
                    return create_layer(ctx, layer_index);
                });
        }
        else
        {
            transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false),
                [&](InitContext *ctx, int layer_index) {
                    return create_layer(ctx, layer_index);
                });
        }

        w_ctx_.check_used_mem_size(true);
    }

    int ConditionalGeneration::get_sparse_full_layer_num() const
    {
        int num = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if ((config.layer_is_sparse[i] != 0) && (config.layer_is_swa[i] == 0))
                num++;
        }
        return num;
    }

    int ConditionalGeneration::get_sparse_swa_layer_num() const
    {
        int num = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if ((config.layer_is_sparse[i] != 0) && (config.layer_is_swa[i] != 0))
                num++;
        }
        return num;
    }

    int ConditionalGeneration::get_dense_layer_num() const
    {
        int num = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (config.layer_is_sparse[i] == 0)
                num++;
        }
        return num;
    }

    Block *ConditionalGeneration::create_layer(InitContext *ctx, int layer_index)
    {
        CHATLLM_CHECK(config.sliding_window == 1024);

        if (config.layer_is_sparse[layer_index])
        {
            CHATLLM_CHECK((config.num_experts_per_tok == 8) && (config.num_experts == 64)) << "todo: MoE param";

            if (config.layer_is_swa[layer_index])
            {
                auto layer = new MoEBlock<64, 8, SWASelfAttention<1024>>(ctx, config.hidden_size, config.num_attention_heads,
                    config.intermediate_size, config.moe_intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length);
                layer->attention.freq_base = config.sliding_attention_rope_theta;
                layer->mlp.norm_topk_prob = config.norm_topk_prob != 0;
                return layer;
            }
            else
            {
                auto layer = new MoEBlock<64, 8, qwen::v3::QWen3SelfAttention>(ctx, config.hidden_size, config.num_attention_heads,
                    config.intermediate_size, config.moe_intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length);
                layer->attention.freq_base = config.full_attention_rope_theta;
                layer->attention.n_ctx          = config.max_length;
                layer->attention.n_original_ctx = (int)(config.full_attention_original_max_position_embeddings);
                layer->attention.freq_scale     = 1 / config.full_attention_factor;
                layer->attention.beta_fast      = config.full_attention_beta_fast;
                layer->attention.beta_slow      = config.full_attention_beta_slow;
                layer->attention.ext_factor               = 1.0f;
                layer->attention.attn_factor              = config.full_attention_attention_factor;
                layer->mlp.norm_topk_prob = config.norm_topk_prob != 0;
                return layer;
            }
        }
        else
        {
            CHATLLM_CHECK(false) << "todo: dense block";
            return nullptr;
        }
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        BaseModelForConditionalGeneration::set_tokenizer(tokenizer);
        tokenizer->set_system_prompt("");
    }
}

namespace chatllm::mellum
{
    REGISTER_MODEL_LOADER(MELLUM,           mellum, 1);
}