#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"
#include "llama.h"

namespace chatllm::deepseek::v1
{
    struct Config : public llama::v2::Config
    {
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

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
        Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;
    };

    class ConditionalGeneration : public llama::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
    };
}

namespace chatllm::deepseek::coder
{
    struct Config : public llama::v2::Config
    {
        float rope_scaling;
        float rope_theta;
    };

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        Tokenizer(const llama::v2::Config &config, BaseHistoryEncoder *chat_encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

    public:
        int fim_hole_token_id;
        int fim_begin_token_id;
        int fim_end_token_id;
        int user_token_id;
        int assistant_token_id;
        int eot_token_id;
    };

    class ConditionalGeneration : public llama::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        void load(ModelLoader &loader) override;
    private:
        Config config;
    };
}

namespace chatllm::deepseek::v1_moe
{
    struct Config : public v1::Config
    {
        int num_key_value_heads;
        int first_k_dense_replace;
        int moe_intermediate_size;
        int moe_layer_freq;
        int n_routed_experts;
        int n_shared_experts;
        int norm_topk_prob;
        int num_experts_per_tok;

        float rope_theta;
    };

    typedef v1::Tokenizer Tokenizer;

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class DeepSeekSparseMoE : public BaseSparseMLP
    {
    public:
        DeepSeekSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<DeepSeekSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> DeepSeekMoEMLP;
        typedef LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, DeepSeekMoEMLP> DeepSeekMoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_DEEPSEEK_V1_MoE, int head_dim = -1)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3
                                + moe_layer_num * (12 + 4)
                                + dense_layer_num * 12;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK))
                << "unsupported MoE param";

            if (head_dim < 1)
                head_dim = config.hidden_size / config.num_attention_heads;
            if (head_dim != config.hidden_size / config.num_attention_heads)
                CHATLLM_CHECK(dense_layer_num == 0) << "customized head_dim";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new DeepSeekMoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                        config.num_key_value_heads,
                        head_dim,
                        config.max_length);
                    return layer;
                }
                else
                {
                    return new LlamaBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);

            Base::transformer = transformer;

            #define config_rope(attention)     do { \
                    attention.freq_base      = config.rope_theta;                           \
                } while (false)

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<DeepSeekMoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                    layer->mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    auto *layer = dynamic_cast<LlamaBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".mlp2.",              ".shared_experts."},
                {".mlp1.gate.",         ".gate."},
                {".mlp1.experts.",      ".experts."},
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:
        Config config;

        bool is_layer_moe(int layer_index)
        {
            return (layer_index >= config.first_k_dense_replace) && (layer_index % config.moe_layer_freq == 0);
        }

        int get_moe_layer_num()
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                    r++;
            }
            return r;
        }
    };

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    typedef ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK> ConditionalGeneration;
}

namespace chatllm::deepseek::v2_light
{
    class QProj : public Block
    {
    public:
        QProj(InitContext *ctx, int hidden_size, int num_attention_heads,
            int q_lora_rank, int rope_dim, int qk_nope_head_dim, bool use_bias);

        int64_t get_param_num(bool effective_only) const override;

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        void load(const std::string &path, TensorLoader *loader) override;
    public:
        Linear *d_q_proj, *u_q_proj;
        RMSNorm *norm;
        Linear *q_proj;
    };

    // for opt_speed == false:
    //      k_pe    -> K cache;
    //      kv_lora -> V cache.
    // FIXME: for opt_speed == false, GGML_TYPE_F32 is used because GGML complains
    class BaseMLAttention : public KVCacheAttention
    {
    public:
        BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias,
                      int cache_length);

        BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias);

        int64_t get_param_num(bool effective_only) const override;

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

        void load(const std::string &path, TensorLoader *loader) override;
    protected:
        virtual ggml::tensor *output_project(ComputeContext *ctx, ggml::tensor *scores);
        ggml::tensor *forward_speed(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past);
        ggml::tensor *cross_attention_speed(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k_nope, ggml::tensor *k_pe, ggml::tensor *v);

                                             ggml::tensor *forward_memory(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past);
        ggml::tensor *cross_attention_memory(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k_pe, ggml::tensor *kv_lora);

        ggml::tensor *get_k_pe_from_cache(ComputeContext *ctx, const int n_past, const int qlen);
        ggml::tensor *get_kv_lora_from_cache(ComputeContext *ctx, const int n_past, const int qlen);

        void save_lora_to_cache(ComputeContext *ctx, const int n_past, const int qlen,
            ggml::tensor *k_pe, ggml::tensor *kv_lora);
        ggml::tensor *cross_attention_after_pe_memory(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen0,
                                             ggml::tensor *query_layer, ggml::tensor *k_pe, ggml::tensor *kv_lora);
    public:
        const bool opt_speed;
        const int kv_lora_rank;
        const int rope_dim;
        const int qk_nope_head_dim;
        const int v_head_dim;
        Linear d_kv_proj, k_pe_proj, u_k_nope_proj, u_v_proj;
        QProj q_proj;
        Linear o_proj;
        RMSNorm kv_norm;
    };

    typedef RoPESelfAttention<BaseMLAttention> MLAttention;

    struct Config : public v1::Config
    {
        int num_key_value_heads;
        int first_k_dense_replace;
        int kv_lora_rank;
        int moe_intermediate_size;
        int moe_layer_freq;
        int n_group;
        int n_routed_experts;
        int n_shared_experts;
        int norm_topk_prob;
        int num_experts_per_tok;
        int qk_nope_head_dim;
        int qk_rope_head_dim;
        int original_max_position_embeddings;
        int v_head_dim;

        float beta_fast;
        float beta_slow;
        float factor;
        float mscale;
        float mscale_all_dim;
        float rope_theta;
        float routed_scaling_factor;
    };

    typedef v1::Tokenizer Tokenizer;

    typedef LMBlock1<RMSNorm, MLAttention, RMSNorm, SiLUMLP> DeepSeek2Block;

    static float yarn_get_mscale(float scale = 1.0f, float mscale = 1.0f)
    {
        if (scale <= 1.0f)
            return 1.0f;
        return 0.1f * mscale * logf(scale) + 1.0f;
    }

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class DeepSeekSparseMoE : public BaseSparseMLP
    {
    public:
        DeepSeekSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size, bool gate_use_bias)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, gate_use_bias)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<DeepSeekSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> DeepSeekMoEMLP;
        typedef LMBlock1<RMSNorm, MLAttention, RMSNorm, DeepSeekMoEMLP> DeepSeek2MoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config) : ConditionalGeneration0(config, runtime_config, MODEL_TYPE_DEEPSEEK_V2_LIGHT, -1)
        {}

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int q_lora_rank, BaseSparseMLP::ScoreFunc score_func = BaseSparseMLP::ScoreFunc::Softmax,
            bool gate_use_bias = false, bool always_scaling = false)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3
                                + moe_layer_num * (16 + 3 + (gate_use_bias ? 1 : 0))
                                + dense_layer_num * 15
                                + (q_lora_rank > 0 ? config.num_hidden_layers * 2 : 0) ;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK)
                            && (config.n_group == 1))
                << "unsupported MoE param";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new DeepSeek2MoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                        config.num_key_value_heads, config.max_length,
                        q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                        false, gate_use_bias);
                    auto sparse = dynamic_cast<BaseSparseMLP *>(&layer->mlp.mlp1);
                    sparse->score_func = score_func;
                    sparse->routed_scaling_factor = config.routed_scaling_factor;
                    sparse->always_scaling = always_scaling;
                    return layer;
                }
                else
                {
                    return new DeepSeek2Block(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length,
                                                q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                                false);
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);
            Base::transformer = transformer;

            float m = 1.0f;
            float attn_scaling_factor = -1.0f;
            if (config.original_max_position_embeddings > 0)
            {
                m = yarn_get_mscale(config.factor, config.mscale) / yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor = 1 / sqrtf((float)(config.qk_rope_head_dim + config.qk_nope_head_dim));
                float mscale = yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor *= mscale * mscale;
                m /= 1.0f + 0.1f * logf(config.factor);
            }

            #define config_rope(attention)     do { \
                    attention.rope_mode      = RoPEMode::Original;                          \
                    attention.freq_base      = config.rope_theta;                           \
                    if (config.original_max_position_embeddings > 0)                        \
                    {                                                                       \
                    attention.n_ctx          = config.max_length;                           \
                    attention.n_original_ctx = config.original_max_position_embeddings;     \
                    attention.freq_scale     = 1 / config.factor;                           \
                    attention.beta_fast      = config.beta_fast;                            \
                    attention.beta_slow      = config.beta_slow;                            \
                    attention.ext_factor               = 1.0f;                              \
                    attention.attn_factor              = m;                                 \
                    attention.attn_scaling_factor      = attn_scaling_factor;               \
                    }                                                                       \
                } while (false)

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    DeepSeek2MoEBlock *layer = dynamic_cast<DeepSeek2MoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                    layer->mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    DeepSeek2Block *layer = dynamic_cast<DeepSeek2Block *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".mlp2.",              ".shared_experts."},
                {".mlp1.gate.",         ".gate."},
                {".mlp1.experts.",      ".experts."},
                {".mlp1.gate_score_correction_bias",     ".gate.e_score_correction_bias"}
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:
        Config config;

        bool is_layer_moe(int layer_index)
        {
            return (layer_index >= config.first_k_dense_replace) && (layer_index % config.moe_layer_freq == 0);
        }

        int get_moe_layer_num()
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                    r++;
            }
            return r;
        }
    };

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    typedef ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK> ConditionalGeneration;
}

namespace chatllm::deepseek::v2
{
    struct Config : public v2_light::Config
    {
        int q_lora_rank;
        int topk_group;     // TODO: group_limited_greedy
    };

    typedef v1::Tokenizer Tokenizer;

    const int NUM_EXPERTS                   =  160;
    const int EXPERTS_PER_TOK               =  6;

    class ConditionalGeneration : public v2_light::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
    };
}

namespace chatllm::deepseek::v3_light
{
    struct Config : public v2_light::Config
    {
    };

    typedef v1::Tokenizer Tokenizer;

    class ConditionalGeneration : public v2_light::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_DEEPSEEK_V3_LIGHT);
    };
}