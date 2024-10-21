struct Config : public BaseConfig
{
    int window_length;
    int num_buckets;
};

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : Tokenizer(config, nullptr)
    {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }
};

class AlphaGeoSelfAttention: public BaseAttention
{
public:
    AlphaGeoSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_buckets, int max_length)
        : BaseAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, hidden_size / num_attention_heads, max_length, false, false),
          rel_embedding(ggml::new_tensor_2d(ctx, ctx->dtype, num_attention_heads, num_buckets)),
          attention_scale(ggml::new_tensor_1d(ctx, ctx->dtype, num_attention_heads))
    {}

    int64_t get_param_num(bool effective_only) const override
    {
        int64_t r = BaseAttention::get_param_num(effective_only);
        r += ggml::nelements(rel_embedding);
        r += ggml::nelements(attention_scale);
        return r;
    }

    ggml::tensor *cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k, ggml::tensor *v) override
    {
        const int head_size = hidden_size / num_attention_heads;

        // [qlen, heads, head_size]
        ggml::tensor * key_layer = ggml::reshape_3d(ctx, k, head_size, num_kv_heads, qlen);
        key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, pos);

        // [qlen, heads, head_size]
        ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, head_size, num_attention_heads, qlen);
        query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, pos);

        ggml::tensor *attn_scores = cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

        return attn_scores;
    }

public:
    ggml::tensor *rel_embedding;
    ggml::tensor *attention_scale;
};

class AlphaGeoLMBlock : public Block
{
public:
    AlphaGeoLMBlock() = default;

    AlphaGeoLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                int max_length)
        : input_layernorm(ctx, hidden_size),
            attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
            mlp(ctx, hidden_size, intermediate_size) {}

    using Block::forward;
    ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
    {
        hidden_states = input_layernorm.forward(ctx, hidden_states);

        ggml::tensor *attn_outputs = attention.forward(ctx, hidden_states, n_past);

        hidden_states = post_attention_layernorm.forward(ctx, attn_outputs);

        hidden_states = mlp.forward(ctx, hidden_states);

        return hidden_states;
    }

    void shift_cache(int shift, int total) override
    {
        attention.shift_cache(shift, total);
    }

    int64_t get_param_num(bool effective_only) const override
    {
        int64_t r = 0;
        r += input_layernorm.get_param_num(effective_only);
        r += attention.get_param_num(effective_only);
        r += post_attention_layernorm.get_param_num(effective_only);
        r += mlp.get_param_num(effective_only);
        return r;
    }

    size_t get_cache_size(void) const override
    {
        return attention.get_cache_size();
    }

    void  set_cache_buffer(BackendBuffer *buffer) override
    {
        return attention.set_cache_buffer(buffer);
    }

public:
    LayerNorm input_layernorm;
    AlphaGeoSelfAttention attention;
    LayerNorm post_attention_layernorm;
    GLM2MLP mlp;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<Config, Embedding, LayerNorm, GLM2Block, int, int, int, int, int> TransformerClass;
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_ALPHAGEO_LM)
        : BaseModelForConditionalGeneration(type, config, runtime_config)
    {
        CHATLLM_CHECK(false) << "WIP. Bye-bye.";
    }

    void load(ModelLoader &loader) override
    {
    }

public:
    Config config;
};