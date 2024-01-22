struct Config : public BaseConfig
{
    int seq_length;
    int rope_dim;
    int flags;
    float rotary_emb_base;
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
        sys_prompt = "You are a helpful assistant.";
    }

    size_t load(const char *buffer, int n_vocab) override;

    void encode(const std::string &text, std::vector<int> &ids) const override;

    bool is_special_id(int id) const override;

    int get_terminate_token_id(void) const override
    {
        return im_end_token_id;
    }

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_eos) const;

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const;

public:
    int im_start_token_id;
    int im_end_token_id;
    int nl_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int>>
{
public:
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
    tp = new tokenizer::BPEProcessor();
    size_t size = tp->Load(buffer, n_vocab);

    pad_token_id = eos_token_id = tp->GetPieceSize() + 0;
    im_start_token_id           = eos_token_id + 1;
    im_end_token_id             = eos_token_id + 2;

    std::vector<int> ids;
    tp->Encode("\n", &ids);
    nl_token_id = ids[0];

    return size;
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const
{
    if (add_im_start)
        ids.push_back(im_start_token_id);
    BaseTokenizer::encode(text, ids);
    if (add_im_end)
        ids.push_back(im_end_token_id);
    if (add_nl)
        ids.push_back(nl_token_id);
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
{
    encode(text, ids, false, false, false);
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    append_user(round_idx, user, ids);

    tok->encode(ai, ids, false, true, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    if (round_idx == 0)
    {
        oss_prompt << "system\n" << tok->get_system_prompt();
        tok->encode(oss_prompt.str(), ids, true, true, true);
    }

    oss_prompt.clear();
    oss_prompt << "user\n" << user;
    tok->encode(oss_prompt.str(), ids, true, true, true);

    oss_prompt << "assistant\n";
    tok->encode(oss_prompt.str(), ids, true, false, false);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id) || (id == im_start_token_id) || (id == im_end_token_id);
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration<
                                Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int>>(MODEL_TYPE_QWEN, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 16;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    // TODO: support of `use_dynamic_ntk` and `use_logn_attn`
    transformer = Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int>(&w_ctx_, config, false,
                                                                            config.hidden_size, config.num_attention_heads,
                                                                            config.intermediate_size, config.max_length);

    bool use_dynamic_ntk = (config.flags & 1) != 0;
    bool use_logn_attn   = (config.flags & 2) != 0;

    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        auto &layer = transformer.layers[i];
        auto att = dynamic_cast<QWenSelfAttention *>(&layer.attention);
        att->config(config.rope_dim, config.rotary_emb_base, config.seq_length,
                    use_dynamic_ntk, use_logn_attn);
    }
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("transformer.wte.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "transformer.h." + std::to_string(i) + '.';

        loader.read_tensor(layer_prefix + "attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
        loader.read_tensor(layer_prefix + "attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
        loader.read_tensor(layer_prefix + "attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
        loader.read_tensor(layer_prefix + "attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
        loader.read_tensor(layer_prefix + "attn.c_proj.weight", transformer.layers[i].attention.o_proj.weight);

        loader.read_tensor(layer_prefix + "ln_1.weight",        transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "ln_2.weight",        transformer.layers[i].post_attention_layernorm.weight);

        loader.read_tensor(layer_prefix + "mlp.c_proj.weight", transformer.layers[i].mlp.down_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.w1.weight",   transformer.layers[i].mlp.up_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.w2.weight", transformer.layers[i].mlp.gate_proj.weight);
    }
    loader.read_tensor("transformer.ln_f.weight", transformer.final_layernorm.weight);
    loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}