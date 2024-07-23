typedef llama::v3::Config Config;

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

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }
};

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids);
    ids.push_back(tok->eos_token_id);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    if (tok->get_system_prompt().size() > 0)
    {
        ids.push_back(tok->bos_token_id);
        tok->encode("system\n", ids);
        tok->encode(tok->get_system_prompt(), ids);
        ids.push_back(tok->eos_token_id);
    }
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss;

    ids.push_back(tok->bos_token_id);
    tok->encode("user\n", ids);
    tok->encode(user, ids);
    ids.push_back(tok->eos_token_id);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->bos_token_id);
    tok->encode("assistant\n", ids);
}

class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_SMOLLM)
        : ConditionalGeneration(config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration(const Config &config, ModelType type,
                        int num_key_value_heads, int max_length)
        : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, type, num_key_value_heads, max_length, 12, true)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
};