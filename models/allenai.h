#pragma  once
#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::allenai::moe
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int num_experts_per_tok;
        int num_experts;
        int norm_topk_prob;
        float rope_theta;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class OLSparseMoE : public BaseSparseMLP
    {
    public:
        OLSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    class OLSelfAttention : public RoPESelfAttention<BaseNormedAttention>
    {
    public:
        OLSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention<BaseNormedAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {}
    };

    template<int num_local_experts, int num_experts_per_tok> class OLMoEBlock : public LMBlock1<RMSNorm, OLSelfAttention, RMSNorm,
                        OLSparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        OLMoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, OLSelfAttention, RMSNorm,
                       OLSparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, OLMoEBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK>, int, int, int, int, int> ModelClass;
    public:
        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : Base(type, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * (11 + 2 + 2);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            Base::transformer = new ModelClass(
                                &w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            auto transformer = Base::get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.rope_mode    = RoPEMode::Original;
                attention.freq_base    = config.rope_theta;
                transformer->layers[i].mlp.norm_topk_prob = config.norm_topk_prob != 0;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "mlp.experts.", _NUM_EXPERTS, ".down_proj.weight", transformer->layers[i].mlp.experts.down.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "mlp.experts.", _NUM_EXPERTS, ".gate_proj.weight", transformer->layers[i].mlp.experts.gate.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "mlp.experts.", _NUM_EXPERTS, ".up_proj.weight",   transformer->layers[i].mlp.experts.up.weight);

                loader.read_tensor(layer_prefix + "mlp.gate.weight", transformer->layers[i].mlp.gate.weight);

                loader.read_tensor(layer_prefix + "input_layernorm.weight",
                                transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);

                loader.read_tensor(layer_prefix + "self_attn.q_norm.weight", transformer->layers[i].attention.q_norm.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_norm.weight", transformer->layers[i].attention.k_norm.weight);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };

    typedef _ConditionalGeneration<64, 8, MODEL_TYPE_OLMoE> ConditionalGeneration;
}

namespace chatllm::allenai::dense
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_theta;
    };

    typedef moe::Tokenizer Tokenizer;

    typedef LMBlock4<Identity,
                 moe::OLSelfAttention,
                 RMSNorm,
                 Identity,
                 SiLUMLP,
                 RMSNorm> OlmoBlock;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, OlmoBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_OLMo2);
        void load(ModelLoader &loader) override;

    public:
        Config config;
    };
}
