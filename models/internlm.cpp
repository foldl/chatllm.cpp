struct Config : public BaseConfig
{
};

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer() : add_bos_token(true), add_eos_token(false) {};

    size_t load(const char *buffer, int n_vocab) override;

    std::vector<int> encode(const std::string &text) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length, const bool incremental) const override;

    static std::string build_prompt(const std::vector<std::string> &history);

    int get_terminate_token_id(void) override { return eoa_token_id; }

private:
    static std::string preprocess(const std::string &text);

    static std::string postprocess(const std::string &text);

    bool is_special_id(int id) const;

public:
    sentencepiece::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int unk_token_id;
    int eoa_token_id;
    bool add_bos_token;
    bool add_eos_token;
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
    size_t size = sp.Load(buffer, n_vocab);

    unk_token_id = sp.PieceToId("<unk>");
    bos_token_id = sp.PieceToId("<s>");
    eos_token_id = sp.PieceToId("</s>");
    pad_token_id = sp.PieceToId("</s>");
    eoa_token_id = sp.PieceToId("<eoa>");

    return size;
}

std::vector<int> Tokenizer::encode(const std::string &text) const
{
    std::vector<int> ids;
    sp.Encode(text, &ids);
    if (add_bos_token)
        ids.insert(ids.begin(), {bos_token_id});
    if (add_eos_token)
        ids.push_back(eos_token_id);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const
{
    // filter out special tokens
    std::vector<int> normal_ids(ids);
    normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(),
                                    [this](int id)
                                    { return is_special_id(id); }),
                     normal_ids.end());

    std::string text;
    sp.Decode(normal_ids, &text);
    return text;
}

std::vector<int> Tokenizer::encode_history(const std::vector<std::string> &history, int max_length, const bool incremental) const
{
    std::string prompt;
    if (incremental)
    {
        std::ostringstream oss_prompt;
        oss_prompt << "<s><|User|>:" << history[history.size() - 1] << "<eoh>\n<|Bot|>:";
        prompt = oss_prompt.str();
    }
    else
        prompt = build_prompt(history);
    std::vector<int> input_ids = encode(prompt);
    int offset = add_bos_token ? 1 : 0;
    if ((int)input_ids.size() > max_length)
    {
        int num_drop = (int)input_ids.size() - max_length;
        input_ids.erase(input_ids.begin() + offset, input_ids.begin() + offset + num_drop);
    }
    return input_ids;
}

std::string Tokenizer::build_prompt(const std::vector<std::string> &history)
{
    CHATLLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::ostringstream oss_prompt;
    for (size_t i = 0; i + 1 < history.size(); i += 2)
        oss_prompt << "<s><|User|>:" << history[i] << "<eoh>\n<|Bot|>:" << history[i + 1] << "<eoa>\n";

    oss_prompt << "<s><|User|>:" << history[history.size() - 1] << "<eoh>\n<|Bot|>:";
    return oss_prompt.str();
}

bool Tokenizer::is_special_id(int id) const
{
    return id == eoa_token_id;
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_INTERNLM, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 15;
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
        loader.read_tensor(layer_prefix + "self_attn.q_proj.bias", transformer.layers[i].attention.q_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.k_proj.bias", transformer.layers[i].attention.k_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.bias", transformer.layers[i].attention.v_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.bias", transformer.layers[i].attention.o_proj.bias);

        loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer.layers[i].mlp.down_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer.layers[i].mlp.up_proj.weight);

        loader.read_tensor(layer_prefix + "input_layernorm.weight",
                           transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                           transformer.layers[i].post_attention_layernorm.weight);
    }
    loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("lm_head.weight", transformer.lm_head.weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}