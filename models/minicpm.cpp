struct Config : public BaseConfig
{
    int num_key_value_heads;
    float rope_scaling;
    float rope_theta;
    float scale_depth;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t load(const char *buffer, int n_vocab) override;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<Config, Embedding, RMSNorm, MiniCPMBlock, int, int, int, int, int>>
{
public:
    ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_MINICPM)
        : BaseModelForConditionalGeneration<
                                  Model<Config, Embedding, RMSNorm, MiniCPMBlock, int, int, int, int, int>>(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 2 + config.num_hidden_layers * 12;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = Model<Config, Embedding, RMSNorm, MiniCPMBlock, int, int, int, int, int>(&w_ctx_, config, nullptr,
                                                                                config.hidden_size, config.num_attention_heads,
                                                                                config.intermediate_size, config.num_key_value_heads, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer.layers[i].attention;
            attention.freq_base = config.rope_theta;
            attention.freq_scale = 1 / config.rope_scaling;
            attention.set_prec(ggml_prec::GGML_PREC_F32);

            transformer.layers[i].hidden_scaling = config.scale_depth;
        }
    }

    void load(ModelLoader &loader) override
    {
        loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight",   transformer.layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",   transformer.layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",     transformer.layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer.layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
        }
        loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
            << "corrupted model weights";
    }

public:
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 344ull * 1024 * 1024;

    Config config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);
    return size;
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    append_user(round_idx, user, ids);
    tokenizer->encode(ai, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    if (round_idx == 0)
    {
        ids.push_back(tokenizer->bos_token_id);
        oss_prompt << " ";
    }
    oss_prompt << "<用户>" << user
               << "<AI>";

    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}
