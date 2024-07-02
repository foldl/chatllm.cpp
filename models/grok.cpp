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
    ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const override
    {
        float max = 30.0f;
        ggml_tensor *r = kq;

        r = ggml_scale_inplace(ctx->gctx.get(), r, 1.0f / max);
        r = ggml_tanh_inplace(ctx->gctx.get(), r);
        r = ggml_scale_inplace(ctx->gctx.get(), r, max);
        return r;
    }
};

class GrokSelfAttention : public RoPESelfAttention<GrokBaseAttention>
{
public:
    GrokSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
        : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, false, false) {}
};

template<int num_local_experts, int num_experts_per_tok> class GrokBlock : public LMBlock4<RMSNorm,
                 GrokSelfAttention,
                 RMSNorm,
                 RMSNorm,
                 SparseMoE<GELUMLP, num_local_experts, num_experts_per_tok>,
                 RMSNorm>
{
public:
    typedef LMBlock4<RMSNorm,
                 GrokSelfAttention,
                 RMSNorm,
                 RMSNorm,
                 SparseMoE<GELUMLP, num_local_experts, num_experts_per_tok>,
                 RMSNorm> GrokLMBlock;
    GrokBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
        : GrokLMBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
    {}
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<Model<Config, Embedding, RMSNorm,
    GrokBlock<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, int, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config);

    void load(ModelLoader &loader) override;

public:
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 244ull * 1024 * 1024;

    Config config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);
    return size;
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_GROK_1, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 2 + config.num_hidden_layers * (12 + config.num_experts * 3);
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    CHATLLM_CHECK((NUM_EXPERTS == config.num_experts) && (EXPERTS_PER_TOK == config.num_selected_experts))
        << "unsupported MoE param";

    GRAPH_SIZE = 4096 * 2;

    transformer = new Model<Config, Embedding, RMSNorm, GrokBlock<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, int, int, int, int, int>(
                        &w_ctx_, config, nullptr,
                        config.hidden_size, config.num_attention_heads,
                        config.intermediate_size, config.num_key_value_heads, config.max_length);

    for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

        for (int j = 0; j < config.num_experts; j++)
        {
            std::string prefix = layer_prefix + "experts." + std::to_string(j) + '.';
            loader.read_tensor(prefix + "w1.weight", transformer->layers[i].mlp.experts[j].gate_proj.weight);
            loader.read_tensor(prefix + "w2.weight", transformer->layers[i].mlp.experts[j].down_proj.weight);
            loader.read_tensor(prefix + "w3.weight", transformer->layers[i].mlp.experts[j].up_proj.weight);
        }

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

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}
}