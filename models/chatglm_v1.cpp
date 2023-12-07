// ===== ChatLLM-6B =====

struct Config : public BaseConfig
{
};

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config) {}

    size_t load(const char *buffer, int n_vocab) override;

    std::vector<int> encode(const std::string &text) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length, const bool incremental = false) const override;

    static std::string build_prompt(const std::vector<std::string> &history);

private:
    static std::string preprocess(const std::string &text);

    static std::string postprocess(const std::string &text);

public:
    tokenizer::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int mask_token_id;
    int gmask_token_id;
    int pad_token_id;
};

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
    Model<Config, LayerNorm, GLMBlock, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config);

    void load(ModelLoader &loader) override;

public:
    static constexpr size_t MEM_SIZE = 272ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 128ull * 1024 * 1024;
    Config config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    size_t size = sp.Load(buffer, n_vocab);

    bos_token_id = sp.PieceToId("<sop>");
    eos_token_id = sp.PieceToId("<eop>");
    mask_token_id = sp.PieceToId("[MASK]");
    gmask_token_id = sp.PieceToId("[gMASK]");
    pad_token_id = sp.PieceToId("<pad>");
    return size;
}

std::vector<int> Tokenizer::encode(const std::string &text) const
{
    std::string input = preprocess(text);
    std::vector<int> ids;
    sp.Encode(input, &ids);
    ids.insert(ids.end(), {gmask_token_id, bos_token_id});
    return ids;
}

std::vector<int> Tokenizer::encode_history(const std::vector<std::string> &history, int max_length, const bool incremental) const
{
    std::string prompt;
    if (incremental)
    {
        std::ostringstream oss_prompt;
        oss_prompt << "[Round " << history.size() / 2 << "]\n问：" << history[history.size() - 1] << "\n";
        prompt = oss_prompt.str();
    }
    else
        prompt = build_prompt(history);

    std::vector<int> input_ids = encode(prompt);
    if ((int)input_ids.size() > max_length)
    {
        // sliding window: always take the last max_length tokens
        input_ids.erase(input_ids.begin(), input_ids.end() - max_length);
    }
    return input_ids;
}

std::string Tokenizer::build_prompt(const std::vector<std::string> &history)
{
    CHATLLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::ostringstream oss_prompt;

    for (size_t i = 0; i < history.size(); i += 2)
    {
        oss_prompt << "[Round " << i / 2 << "]\n问：" << history[i] << "\n答：";
        if (i < history.size() - 1)
        {
            oss_prompt << history[i + 1] << "\n";
        }
    }
    return oss_prompt.str();
}

std::string Tokenizer::decode(const std::vector<int> &ids) const
{
    std::string text;
    sp.Decode(ids, &text);
    text = postprocess(text);
    return text;
}

std::string Tokenizer::preprocess(const std::string &text)
{
    std::string output;

    // newline token
    {
        static const std::regex newline_regex("\n");
        output = std::regex_replace(text, newline_regex, "<n>");
    }
    // tab token
    {
        static const std::regex tab_regex("\t");
        output = std::regex_replace(output, tab_regex, "<|tab|>");
    }
    // blank tokens
    {
        static const std::regex pattern(R"([ ]{2,80})");
        output = regex_replace(output, pattern, [](const std::smatch &sm)
            {
                std::ostringstream oss;
                oss << "<|blank_" << sm.str().size() << "|>";
                return oss.str();
            });
    }

    return output;
}

std::string Tokenizer::postprocess(const std::string &text)
{
    std::string output;

    // newline token
    {
        static const std::regex pattern(R"(<n>)");
        output = std::regex_replace(text, pattern, "\n");
    }
    // tab token
    {
        static const std::regex pattern(R"(<\|tab\|>)");
        output = std::regex_replace(output, pattern, "\t");
    }
    // blank tokens
    {
        static const std::regex pattern(R"(<\|blank_(\d+)\|>)");
        output = regex_replace(output, pattern,
                                [](const std::smatch &sm)
                                { return std::string(std::stoi(sm[1].str()), ' '); });
    }

    // replace punctuations
    // reference: https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
    {
        static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        static const std::vector<std::pair<std::wregex, std::wstring>> punct_map{
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]),)")), converter.from_bytes("$1，")},
            {std::wregex(converter.from_bytes(R"(,([\u4e00-\u9fff]))")), converter.from_bytes("，$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])!)")), converter.from_bytes("$1！")},
            {std::wregex(converter.from_bytes(R"(!([\u4e00-\u9fff]))")), converter.from_bytes("！$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]):)")), converter.from_bytes("$1：")},
            {std::wregex(converter.from_bytes(R"(:([\u4e00-\u9fff]))")), converter.from_bytes("：$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]);)")), converter.from_bytes("$1；")},
            {std::wregex(converter.from_bytes(R"(;([\u4e00-\u9fff]))")), converter.from_bytes("；$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])\?)")), converter.from_bytes("$1？")},
            {std::wregex(converter.from_bytes(R"(\?([\u4e00-\u9fff]))")), converter.from_bytes("？$1")},
        };
        std::wstring w_output = converter.from_bytes(output);
        for (const auto &punct_pair : punct_map)
        {
            w_output = std::regex_replace(w_output, punct_pair.first, punct_pair.second);
        }
        output = converter.to_bytes(w_output);
    }

    return output;
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_CHATGLM, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 15;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = Model<Config, LayerNorm, GLMBlock, int, int, int, int>(&w_ctx_, config, true,
                            config.hidden_size, config.num_attention_heads, config.num_hidden_layers,
                            config.max_length);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("transformer.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "transformer.layers." + std::to_string(i) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "input_layernorm.bias", transformer.layers[i].input_layernorm.bias);
        loader.read_tensor(layer_prefix + "attention.query_key_value.weight",
                            transformer.layers[i].attention.query_key_value.weight);
        loader.read_tensor(layer_prefix + "attention.query_key_value.bias",
                            transformer.layers[i].attention.query_key_value.bias);
        loader.read_tensor(layer_prefix + "attention.dense.weight", transformer.layers[i].attention.dense.weight);
        loader.read_tensor(layer_prefix + "attention.dense.bias", transformer.layers[i].attention.dense.bias);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                            transformer.layers[i].post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",
                            transformer.layers[i].post_attention_layernorm.bias);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer.layers[i].mlp.dense_h_to_4h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.bias", transformer.layers[i].mlp.dense_h_to_4h.bias);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.dense_4h_to_h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.bias", transformer.layers[i].mlp.dense_4h_to_h.bias);
    }
    loader.read_tensor("transformer.final_layernorm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("transformer.final_layernorm.bias", transformer.final_layernorm.bias);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";

#ifdef GGML_USE_CUBLAS
    auto transform_tensor_to_cuda = [](ggml_tensor *tensor)
    {
        tensor->backend = GGML_BACKEND_GPU;
        ggml_cuda_transform_tensor(tensor->data, tensor);
    };

    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        transform_tensor_to_cuda(transformer.layers[i].attention.query_key_value.weight);
        transform_tensor_to_cuda(transformer.layers[i].attention.dense.weight);
        transform_tensor_to_cuda(transformer.layers[i].mlp.dense_h_to_4h.weight);
        transform_tensor_to_cuda(transformer.layers[i].mlp.dense_4h_to_h.weight);
    }
#endif
}
