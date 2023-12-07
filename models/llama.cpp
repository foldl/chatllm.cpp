struct Config : public BaseConfig
{
};

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : add_bos_token(false)
    {
        sys_prompt = R"""(You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.)""";

        bos_token_id = config.bos_token_id;
        eos_token_id = config.eos_token_id;
    }

    size_t load(const char *buffer, int n_vocab) override;

    std::vector<int> encode(const std::string &text) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length, const bool incremental) const override;

    static std::string build_prompt(const std::vector<std::string> &history);

    //int get_terminate_token_id(void) override { return 13; }

private:
    static std::string preprocess(const std::string &text);

    static std::string postprocess(const std::string &text);

    bool is_special_id(int id) const;

    std::vector<int> encode(const std::string &text, bool add_bos, bool add_eos) const;

public:
    tokenizer::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    bool add_bos_token;
    bool add_eos_token;
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
    size_t size = sp.Load(buffer, n_vocab);

    pad_token_id = sp.PieceToId("<pad>");
    return size;
}

std::vector<int> Tokenizer::encode(const std::string &text, bool add_bos, bool add_eos) const
{
    std::vector<int> ids;
    sp.Encode(text, &ids);
    if (add_bos)
        ids.insert(ids.begin(), {bos_token_id});
    if (add_eos)
        ids.push_back(eos_token_id);
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string &text) const
{
    return encode(text, add_bos_token, add_eos_token);
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

template <class T> std::vector<T> & append_vector(std::vector<T> &r, const std::vector<T> &items)
{
    r.insert(r.end(), items.begin(), items.end());
    return r;
}

std::string trim(std::string str, const char *spaces = " \t")
{
    str.erase(str.find_last_not_of(spaces) + 1);
    str.erase(0,str.find_first_not_of(spaces));
    return str;
}

std::vector<int> Tokenizer::encode_history(const std::vector<std::string> &history, int max_length, const bool incremental) const
{
    CHATLLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::vector<int> input_ids;

    if (incremental)
    {
        std::string user = trim(history[history.size() - 1]);
        return encode("[INST] " + user + "[/INST]", true, false);
    }

    int64_t start = (int64_t)history.size() - 2;
    size_t total_id_num = start >= 0 ? encode(history[start], true, true).size() : 0;
    while (start >= 1)
    {
        total_id_num += encode(history[start], true, true).size();
        total_id_num += encode(history[start - 1], true, true).size();
        if (total_id_num >= (size_t)max_length / 2)
            break;
        start -= 2;
    }
    start++;

    bool add_sys = true;

    for (; start <= (int64_t)history.size() - 3; start += 2)
    {
        std::string user = trim(history[start]);
        std::string answer = trim(history[start + 1]);
        if (add_sys)
        {
            user = "<<SYS>>\n" + sys_prompt + "\n<</SYS>>\n\n" + user;
            add_sys = false;
        }

        //std::cout << "[INST] " + user + "[/INST] " + answer + " " << std::endl;

        append_vector(input_ids, encode("[INST] " + user + "[/INST] " + answer + " ", true, true));
    }

    std::string user = trim(history[history.size() - 1]);
    if (add_sys)
    {
        user = "<<SYS>>\n" + sys_prompt + "\n<</SYS>>\n\n" + user;
        add_sys = false;
    }

    append_vector(input_ids, encode("[INST] " + user + "[/INST]", true, false));

    return input_ids;
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