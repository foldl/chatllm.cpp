struct Config : public BaseConfig
{
    int num_key_value_heads;
    float rope_scaling;
    float rope_theta;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
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
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    void encode(const std::string &text, std::vector<int> &ids) const override;

    bool is_special_id(int id) const override;

public:
    virtual void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
private:
    typedef BaseModelForConditionalGeneration Base;
    typedef Model<Config, Embedding, LayerNorm, OrionBlock, int, int, int, int, int> ModelClass;
public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_ORION)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 4 + config.num_hidden_layers * 14;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, false,
                                    config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, config.num_key_value_heads, config.max_length);
    }

    void load(ModelLoader &loader) override
    {
        auto transformer = get_typed_transformer<ModelClass>();

        loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "input_layernorm.bias",   transformer->layers[i].input_layernorm.bias);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight",   transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",   transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",     transformer->layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",   transformer->layers[i].post_attention_layernorm.bias);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
        }
        loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
        loader.read_tensor("model.norm.bias",   transformer->final_layernorm.bias);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

public:
    Config config;
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);
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

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->bos_token_id);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    oss_prompt << "Human: " << user << "\n\n";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, true);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    oss_prompt << "Assistant: ";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, true);
}


bool Tokenizer::is_special_id(int id) const
{
    return id == pad_token_id;
}