#include "qwen.h"

namespace chatllm::grove::moe
{
    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        float rope_theta;
        int moe_intermediate_size;
        int num_experts_per_tok;
        int num_experts;
        int num_experts_per_group;
        int small_experts_intermediate_size;
        float small_experts_weight;
    };

    typedef qwen::v3::Tokenizer Tokenizer;

    // TODO: optimization: same small expert might be calculated twice.
    class BigLittleGroupedSparseMoE : public BaseSparseMLP
    {
    public:
        BigLittleGroupedSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size, int num_local_experts, int num_experts_per_tok,
                       int group_size, int small_experts_intermediate_size);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

    protected:
        ggml::tensor *forward_with_experts(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts,
            ggml::tensor *weights) override;
    public:
        MultiMLP small_experts;
        const int group_size;
        const int small_experts_intermediate_size;
        float small_experts_weight;
    };

    BigLittleGroupedSparseMoE::BigLittleGroupedSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size, int num_local_experts, int num_experts_per_tok,
                       int group_size, int small_experts_intermediate_size)
        : BaseSparseMLP(ctx, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, ActFunc::SILU, false),
            small_experts(ctx, hidden_size, small_experts_intermediate_size, num_local_experts / group_size, num_experts_per_tok, ActFunc::SILU, false, group_size),
            group_size(group_size), small_experts_intermediate_size(small_experts_intermediate_size),
            small_experts_weight(0.5f)
    {
    }

    int64_t BigLittleGroupedSparseMoE::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += small_experts.get_param_num(effective_only);
        r += BaseSparseMLP::get_param_num(effective_only);
        return r;
    }

    void BigLittleGroupedSparseMoE::load(const std::string &path, TensorLoader *loader)
    {
        BaseSparseMLP::load(path, loader);

        small_experts.load(path + "chunk_experts.", loader);
    }

    // selected_experts: [qlen, num_experts_per_tok]
    // weights:          [1, num_experts_per_tok, qlen]
    ggml::tensor *BigLittleGroupedSparseMoE::forward_with_experts(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts,
            ggml::tensor *weights)
    {
        ggml::tensor * large_out = BaseSparseMLP::forward_with_experts(ctx, hidden_states, selected_experts, weights);
        ggml::tensor * small_out = BaseSparseMLP::forward_with_experts(ctx, hidden_states, selected_experts, weights,
            [this](ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *selected_experts)
            {
                return small_experts.forward(ctx, hidden_states, selected_experts);
            });

        ggml::tensor * r         = ggml::add(ctx, large_out, small_out);

        return r;
    }

    #define SMALL_EXPERTS_GROUP_SIZE                2
    #define SMALL_EXPERTS_INTERMEDIATE_SIZE         128

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class GroveSparseMoE : public BigLittleGroupedSparseMoE
    {
    public:
        GroveSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BigLittleGroupedSparseMoE(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, SMALL_EXPERTS_GROUP_SIZE, SMALL_EXPERTS_INTERMEDIATE_SIZE)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class GroveMoEBlock : public
        LMBlock1<RMSNorm, qwen::v3::QWen3SelfAttention, RMSNorm, GroveSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
    {
    public:
        typedef GroveSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK> MoEMLP;
    public:
        GroveMoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size,
                int num_kv_heads,
                int head_dim, int max_length)
            : LMBlock1<RMSNorm, qwen::v3::QWen3SelfAttention, RMSNorm, MoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size,
            num_kv_heads, head_dim, max_length)
        {}
    };

    typedef GroveMoEBlock<128, 8> GroveMoEBlock128_8;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, GroveMoEBlock128_8, int, int, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = ModelType::MODEL_TYPE_GROVE_MOE);
    };


    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * (14 + 1 + 3);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        CHATLLM_CHECK(config.num_experts_per_group == SMALL_EXPERTS_GROUP_SIZE);
        CHATLLM_CHECK(config.small_experts_intermediate_size == SMALL_EXPERTS_INTERMEDIATE_SIZE);

        transformer = new ModelClass(&w_ctx_, config, false, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                              config.moe_intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base = config.rope_theta;
            layer.mlp.small_experts_weight = config.small_experts_weight;
        }

        w_ctx_.check_used_mem_size(true);
    }

    REGISTER_MODEL_LOADER(GROVE_MOE,         moe, 1);
}
