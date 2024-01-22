// ===== ChatLLM2-6B =====

struct Config : public BaseConfig
{
    int num_kv_heads;
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
    Tokenizer(const Config &config) : BaseTokenizer::BaseTokenizer(config, &_chat_encoder) {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder) : BaseTokenizer::BaseTokenizer(config, encoder) {}

    size_t load(const char *buffer, int n_vocab) override;

    void encode(const std::string &text, std::vector<int> &ids) const override;

    bool is_special_id(int id) const override;

public:
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
    Model<Config, Embedding, RMSNorm, GLM2Block, int, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_CHATGLM2);

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

    int special_id = tp->GetPieceSize();
    mask_token_id = special_id++;
    gmask_token_id = special_id++;
    smask_token_id = special_id++;
    sop_token_id = special_id++;
    eop_token_id = special_id++;

    return size;
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
{
    ids.insert(ids.end(), {gmask_token_id, sop_token_id}); // special prefix
    BaseTokenizer::encode(text, ids);
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "[Round " << round_idx + 1 << "]\n\n问：" << user << "\n\n答：" << ai << "\n\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "[Round " << round_idx + 1 << "]\n\n问：" << user << "\n\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

bool Tokenizer::is_special_id(int id) const
{
    return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
            id == eop_token_id;
}

ConditionalGeneration::ConditionalGeneration(const Config &config, ModelType type)
    : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 10;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = Model<Config, Embedding, RMSNorm, GLM2Block, int, int, int, int, int>(&w_ctx_, config, false,
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
    loader.read_tensor("transformer.output_layer.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

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