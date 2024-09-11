namespace v1
{
struct Config : public BaseConfig
{
    int num_key_value_heads;
    int num_experts;
    int num_selected_experts;
    float rope_theta;
    float output_multiplier_scale;
};

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : BaseTokenizer(config, nullptr)
    {
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
};

const int NUM_EXPERTS                   =  8;
const int EXPERTS_PER_TOK               =  2;

// make it easy to test with different number of experts.
#define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

class GrokBaseAttention : public BaseAttention
{
public:
    GrokBaseAttention() : BaseAttention() {}

    GrokBaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
             bool qkv_bias, bool o_bias)
        : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias)
    {}
protected:
    ggml::tensor *apply_pos_embedding_kq(ComputeContext *ctx, ggml::tensor *kq, int hidden_size, int qlen, ggml::tensor *past) const override
    {
        float max = 30.0f;
        ggml::tensor *r = kq;

        r = ggml::scale_inplace(ctx, r, 1.0f / max);
        r = ggml::inplace_act(ctx, ActFunc::Tanh, r);
        r = ggml::scale_inplace(ctx, r, max);
        return r;
  }
};

class GrokSelfAttention : public RoPESelfAttention<GrokBaseAttention>
{
public:
    GrokSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
        : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, false, false) {}
};

template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class GrokSparseMoE : public BaseSparseMLP
{
public:
    GrokSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
        : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::GELU, false)
    {
    }
};

template<int num_local_experts, int num_experts_per_tok> class GrokBlock : public LMBlock4<RMSNorm,
                 GrokSelfAttention,
                 RMSNorm,
                 RMSNorm,
                 GrokSparseMoE<num_local_experts, num_experts_per_tok>,
                 RMSNorm>
{
public:
    typedef LMBlock4<RMSNorm,
                 GrokSelfAttention,
                 RMSNorm,
                 RMSNorm,
                 GrokSparseMoE<num_local_experts, num_experts_per_tok>,
                 RMSNorm> GrokLMBlock;
    GrokBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
        : GrokLMBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
    {}
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<Config, Embedding, RMSNorm, GrokBlock<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, int, int, int, int, int> ModelClass;
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

    void load(ModelLoader &loader) override;

public:
    Config config;
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);
    return size;
}

ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_GROK_1, config, runtime_config, 4096 * 2), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 2 + config.num_hidden_layers * (12 + 3);
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    CHATLLM_CHECK((NUM_EXPERTS == config.num_experts) && (EXPERTS_PER_TOK == config.num_selected_experts))
        << "unsupported MoE param";

    transformer = new ModelClass(
                        &w_ctx_, config, nullptr,
                        config.hidden_size, config.num_attention_heads,
                        config.intermediate_size, config.num_key_value_heads, config.max_length);

    for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    auto transformer = get_typed_transformer<ModelClass>();

    loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

        loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "experts.", config.num_experts, ".w2.weight", transformer->layers[i].mlp.experts_down.weight);
        loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "experts.", config.num_experts, ".w1.weight", transformer->layers[i].mlp.experts_gate.weight);
        loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "experts.", config.num_experts, ".w3.weight", transformer->layers[i].mlp.experts_up.weight);

        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);

        loader.read_tensor(layer_prefix + "rms_norm.weight",
                           transformer->layers[i].pre_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "rms_norm_1.weight",
                           transformer->layers[i].post_attention_layernorm.weight);

        loader.read_tensor(layer_prefix + "rms_norm_2.weight",
                           transformer->layers[i].pre_mlp_layernorm.weight);
        loader.read_tensor(layer_prefix + "rms_norm_3.weight",
                           transformer->layers[i].post_mlp_layernorm.weight);

        loader.read_tensor(layer_prefix + "router.weight",
                           transformer->layers[i].mlp.gate.weight);
    }

    loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

    CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
        << "corrupted model weights";
}
}