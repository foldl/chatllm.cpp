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
    Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
    sys_prompt = R"""(You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.)""";
    }

    size_t load(const char *buffer, int n_vocab) override;

    void encode(const std::string &text, std::vector<int> &ids) const override;

    bool is_special_id(int id) const override;

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<Config, RMSNorm, LlamaBlock, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_LLAMA2);

    void load(ModelLoader &loader) override;

public:
    static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 244ull * 1024 * 1024;

    Config config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::SentencePieceProcessor();
    size_t size = tp->Load(buffer, n_vocab);

    pad_token_id = tp->PieceToId("<pad>");
    return size;
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
{
    if (add_bos)
        ids.push_back(bos_token_id);
    BaseTokenizer::encode(text, ids);
    if (add_eos)
        ids.push_back(eos_token_id);
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
{
    encode(text, ids, false, false);
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    if (round_idx == 0)
        oss_prompt << "<<SYS>>\n" << tok->get_system_prompt() << "\n<</SYS>>\n\n";

    oss_prompt << "[INST] " + user << "[/INST] " << ai;

    auto text = oss_prompt.str();
    tok->encode(text, ids, true, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    if (round_idx == 0)
        oss_prompt << "<<SYS>>\n" << tok->get_system_prompt() << "\n<</SYS>>\n\n";

    oss_prompt << "[INST] " + user << "[/INST] ";

    auto text = oss_prompt.str();
    tok->encode(text, ids, true, false);
}

bool Tokenizer::is_special_id(int id) const
{
    return id == pad_token_id;
}

ConditionalGeneration::ConditionalGeneration(const Config &config, ModelType type)
    : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 12;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = Model<Config, RMSNorm, LlamaBlock, int, int, int, int>(&w_ctx_, config, false,
                                                                            config.hidden_size, config.num_attention_heads,
                                                                            config.intermediate_size, config.max_length);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight",
                           transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer.layers[i].mlp.down_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer.layers[i].mlp.up_proj.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                           transformer.layers[i].post_attention_layernorm.weight);

        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
    }
    loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("lm_head.weight", transformer.lm_head.weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}