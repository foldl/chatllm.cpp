// ===== ChatLLM2-6B =====

struct Config : public glm::v2::Config
{
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public glm::v2::Tokenizer
{
public:
    Tokenizer(const Config &config) : glm::v2::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "# language: Python";
    }
};

class ConditionalGeneration : public glm::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : glm::v2::ConditionalGeneration(config, MODEL_TYPE_CODEGEEX2)
    {
    }
};

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::string combined = tokenizer->get_system_prompt() + "\n" + user + "\n";
    tokenizer->encode(combined, ids);
}
