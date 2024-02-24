struct Config : public BaseConfig
{
    int num_key_value_heads;
    int head_dim;
    float rope_theta;
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
        : BaseTokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }

    size_t load(const char *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        pad_token_id = tp->PieceToId("<pad>");
        start_of_turn_token_id = tp->PieceToId("<start_of_turn>");
        end_of_turn_token_id   = tp->PieceToId("<end_of_turn>");
        nl_token_id = tp->PieceToId("\n");
        return size;
    }

    int get_terminate_token_id(void) const override { return end_of_turn_token_id; }

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_start, bool add_end)
    {
        if (add_start)
            ids.push_back(start_of_turn_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_end)
        {
            ids.push_back(end_of_turn_token_id);
            ids.push_back(nl_token_id);
        }
    }

public:
    int start_of_turn_token_id;
    int end_of_turn_token_id;
    int nl_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<BaseConfig, Embedding, RMSNorm, GemmaBlock, int, int, int, int, int, int>>
{
public:
    ConditionalGeneration(const Config &config, ModelType type = MODEL_TYPE_GEMMA)
        : BaseModelForConditionalGeneration<
                                  Model<BaseConfig, Embedding, RMSNorm, GemmaBlock, int, int, int, int, int, int>>(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 2 + config.num_hidden_layers * 12;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = Model<BaseConfig, Embedding, RMSNorm, GemmaBlock, int, int, int, int, int, int>(&w_ctx_, config,
                nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer.layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }

    void load(ModelLoader &loader) override
    {
        loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer.layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer.layers[i].mlp.up_proj.weight);
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
    static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 244ull * 1024 * 1024;

    BaseConfig config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_user(round_idx, user, ids);

    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    if (round_idx == 0)
        ids.push_back(tok->bos_token_id);

    oss_prompt << "user" << "\n" << user;
    tok->encode(oss_prompt.str(), ids, true, true);

    oss_prompt.str("");
    oss_prompt << "model" << "\n";
    tok->encode(oss_prompt.str(), ids, true, false);
}