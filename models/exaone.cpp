typedef llama::v3_2::Config Config;

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
    Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "You are EXAONE model from LG AI Research, a helpful assistant.";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        return size;
    }
};

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss;
    oss << "[|system|]" << tok->get_system_prompt() << "[|endofturn|]\n";
    tok->encode(oss.str(), ids);
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss;
    oss << "[|assistant|]" << ai << "[|endofturn|]\n";
    tok->encode(oss.str(), ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss;
    oss << "[|user|]" << user << "\n";
    tok->encode(oss.str(), ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    tok->encode("[|assistant|]", ids);
}

class ConditionalGeneration : public llama::v3_2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v3_2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_EXAONE)
    {}
};