// ===== ChatLLM2-6B =====

struct Config : public BaseConfig
{
    int num_kv_heads;
};

class Tokenizer : public BaseTokenizer
{
public:
    size_t load(const char *buffer, int n_vocab) override;

    std::vector<int> encode(const std::string &text) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length, const bool incremental) const override;

    static std::string build_prompt(const std::vector<std::string> &history);

    bool is_special_id(int id) const;

public:
    sentencepiece::SentencePieceProcessor sp;
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
    Model<Config, RMSNorm, GLM2Block, int, int, int, int, int>>
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

    int special_id = sp.GetPieceSize();
    mask_token_id = special_id++;
    gmask_token_id = special_id++;
    smask_token_id = special_id++;
    sop_token_id = special_id++;
    eop_token_id = special_id++;

    return size;
}

std::vector<int> Tokenizer::encode(const std::string &text) const
{
    std::vector<int> ids;
    sp.Encode(text, &ids);
    ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
    return ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const
{
    // filter out special tokens
    std::vector<int> normal_ids(ids);
    normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id)
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
        oss_prompt << "[Round " << history.size() / 2 << "]\n问：" << history[history.size() - 1] << "\n";
        prompt = oss_prompt.str();
    }
    else
        prompt = build_prompt(history);

    std::vector<int> input_ids = encode(prompt);
    if ((int)input_ids.size() > max_length)
    {
        // sliding window: drop the least recent history while keeping the special prefix tokens
        int num_drop = (int)input_ids.size() - max_length;
        input_ids.erase(input_ids.begin() + 2, input_ids.begin() + 2 + num_drop);
    }
    return input_ids;
}

std::string Tokenizer::build_prompt(const std::vector<std::string> &history)
{
    CHATLLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::ostringstream oss_prompt;
    for (size_t i = 0; i < history.size(); i += 2)
    {
        oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << history[i] << "\n\n答：";
        if (i < history.size() - 1)
        {
            oss_prompt << history[i + 1] << "\n\n";
        }
    }
    return oss_prompt.str();
}

bool Tokenizer::is_special_id(int id) const
{
    return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
            id == eop_token_id;
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_CHATGLM2, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 9;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = Model<Config, RMSNorm, GLM2Block, int, int, int, int, int>(&w_ctx_, config, false,
                            config.hidden_size, config.num_attention_heads, config.num_kv_heads,
                            config.intermediate_size, config.max_length);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("transformer.embedding.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "transformer.encoder.layers." + std::to_string(i) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "self_attention.query_key_value.weight",
                            transformer.layers[i].attention.query_key_value.weight);
        loader.read_tensor(layer_prefix + "self_attention.query_key_value.bias",
                            transformer.layers[i].attention.query_key_value.bias);
        loader.read_tensor(layer_prefix + "self_attention.dense.weight", transformer.layers[i].attention.dense.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                            transformer.layers[i].post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer.layers[i].mlp.dense_h_to_4h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.dense_4h_to_h.weight);
    }
    loader.read_tensor("transformer.encoder.final_layernorm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("transformer.output_layer.weight", transformer.lm_head.weight);

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
    transform_tensor_to_cuda(lm_head.weight);
#endif
}