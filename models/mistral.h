#include "llama.h"

namespace chatllm::mistral::mistral
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        int sliding_window;
        float rope_theta;
    };

    const int SLIDING_WINDOW_LEN            =  4096;

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const llama::v2::Config &config);
        Tokenizer(const llama::v2::Config &config, BaseHistoryEncoder *encoder);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int start_inst_token_id;
        int end_inst_token_id;
        int tool_calls_token_id;
        int start_avail_tools_token_id;
        int end_avail_tools_token_id;
        int start_tool_results_token_id;
        int end_tool_results_token_id;
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        ChunkInterceptor *get_interceptor(void) override;
    };
}

namespace chatllm::mistral::mixtral
{
    struct Config : public mistral::Config
    {
        int num_experts_per_tok;
        int num_local_experts;
    };

    class Tokenizer : public mistral::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
        Tokenizer(const Config &config, BaseHistoryEncoder *encoder);
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class MixtralSparseMoE : public BaseSparseMLP
    {
    public:
        MixtralSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template<int num_local_experts, int num_experts_per_tok, int sliding_window_len> class MixtralBlock : public LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm,
                        MixtralSparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        MixtralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm,
                       MixtralSparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, MixtralBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int> ModelClass;
    public:
        _ConditionalGeneration() = default;

        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : Base(type, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * (11 + 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_local_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            CHATLLM_CHECK((mistral::SLIDING_WINDOW_LEN == config.sliding_window) || (config.sliding_window <= 0))
                << "sliding_window (" << config.sliding_window << ") must equal to " << mistral::SLIDING_WINDOW_LEN;

            Base::transformer = new ModelClass(
                                &w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            Base::batch_input = false;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w2.weight", transformer->layers[i].mlp.experts_down.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w1.weight", transformer->layers[i].mlp.experts_gate.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w3.weight", transformer->layers[i].mlp.experts_up.weight);

                loader.read_tensor(layer_prefix + "block_sparse_moe.gate.weight",
                                transformer->layers[i].mlp.gate.weight);

                loader.read_tensor(layer_prefix + "input_layernorm.weight",
                                transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };

    const int NUM_EXPERTS                   =  8;
    const int EXPERTS_PER_TOK               =  2;

    typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_MIXTRAL> ConditionalGeneration;
}

namespace chatllm::mistral::mistral2
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        int head_dim;
        int sliding_window;
        float rope_theta;
    };

    class Tokenizer : public mistral::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<MistralBlock<mistral::SLIDING_WINDOW_LEN>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_MISTRAL2);
        ChunkInterceptor *get_interceptor(void) override;
    };
}