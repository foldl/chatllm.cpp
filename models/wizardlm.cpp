struct Config : public llama::Config
{
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
        sys_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi ASSISTANT: Hello.";
    }
};

class ConditionalGeneration : public llama::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::ConditionalGeneration(config, MODEL_TYPE_WIZARDLM)
    {
    }
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

    if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
        oss_prompt << tok->get_system_prompt() << " ";

    oss_prompt << "USER: " + user << " ASSISTANT:";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, false);
}
