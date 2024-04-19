struct Config : public llama::v2::Config
{
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public llama::v2::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::v2::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;
};

class ConditionalGeneration : public llama::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::v2::ConditionalGeneration(config, MODEL_TYPE_DEEPSEEK)
    {
    }
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor2();
    size_t size = tp->Load(buffer, n_vocab);
    return size;
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
        oss_prompt << tok->get_system_prompt() << "\n\n";

    oss_prompt << "User: " << user << "\n\n"
               << "Assistant: " << ai;

    auto text = oss_prompt.str();
    tok->encode(text, ids, true, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
        oss_prompt << tok->get_system_prompt() << "\n\n";

    oss_prompt << "User: " << user << "\n\n"
               << "Assistant: ";

    auto text = oss_prompt.str();
    tok->encode(text, ids, true, false);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id);
}
