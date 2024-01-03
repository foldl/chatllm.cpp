struct Config : public BaseConfig
{
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
    Tokenizer(const Config &config) : BaseTokenizer::BaseTokenizer(config, &_chat_encoder) {};

    size_t load(const char *buffer, int n_vocab) override;

    int get_terminate_token_id(void) const override { return eoa_token_id; }

    bool is_special_id(int id) const override;

public:
    int eoa_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<Config, RMSNorm, InternLMBlock, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config);

    void load(ModelLoader &loader) override;

public:
    static constexpr size_t MEM_SIZE = 512ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 144ull * 1024 * 1024;

    Config config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::SentencePieceProcessor();
    size_t size = tp->Load(buffer, n_vocab);

    eoa_token_id = tp->PieceToId("<eoa>");

    return size;
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "<s><|User|>:" << user << "<eoh>\n<|Bot|>:" << ai << "<eoa>\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "<s><|User|>:" << user << "<eoh>\n<|Bot|>:";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == eoa_token_id) || (id == bos_token_id) || (id == eos_token_id);
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_INTERNLM, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 16;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = Model<Config, RMSNorm, InternLMBlock, int, int, int, int>(&w_ctx_, config, false,
                                                                            config.hidden_size, config.num_attention_heads,
                                                                            config.intermediate_size, config.max_length);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
        loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.bias",   transformer.layers[i].attention.o_proj.bias);

        loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer.layers[i].mlp.down_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer.layers[i].mlp.up_proj.weight);

        loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer.layers[i].post_attention_layernorm.weight);
    }
    loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("lm_head.weight", transformer.lm_head.weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}