struct Config : public llama::Config
{
    int num_key_value_heads;
    float rope_scaling;
    float rope_theta;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public llama::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

    int get_terminate_token_id(void) const { return im_end_token_id; }

public:
    int im_start_token_id;
    int im_end_token_id;
    int im_sep_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<Config, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config);

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
    bos_token_id = tp->GetPieceSize();
    eos_token_id = bos_token_id + 1;
    im_start_token_id = 6;
    im_end_token_id = 7;
    im_sep_token_id = 8;
    return size;
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_user(round_idx, user, ids);

    tok->encode(ai, ids);
    ids.push_back(tok->im_end_token_id);
    tok->encode("\n", ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
    {
        ids.push_back(tok->im_start_token_id);
        tok->encode("system" + tok->get_system_prompt(), ids);
        ids.push_back(tok->im_end_token_id);
        tok->encode("\n", ids);
    }

    ids.push_back(tok->im_start_token_id);
    tok->encode("user\n" + user, ids);
    ids.push_back(tok->im_end_token_id);
    tok->encode("\n", ids);

    ids.push_back(tok->im_start_token_id);
    tok->encode("assistant\n", ids);
}

bool Tokenizer::is_special_id(int id) const
{
    return llama::Tokenizer::is_special_id(id)
            || (id == bos_token_id)
            || (id == eos_token_id)
            || (id == im_start_token_id)
            || (id == im_end_token_id)
            || (id == im_sep_token_id);
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(ModelType::MODEL_TYPE_YI, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 12;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;
    transformer = Model<Config, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int>(&w_ctx_, config, false,
                                                                            config.hidden_size, config.num_attention_heads,
                                                                            config.intermediate_size, config.num_key_value_heads, config.max_length);

    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        auto &attention = transformer.layers[i].attention;
        attention.freq_base = config.rope_theta;
        attention.freq_scale = 1 / config.rope_scaling;
    }

    if (transformer.get_param_num(false) > 20000000)
        GRAPH_SIZE = 4096;
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
    loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}