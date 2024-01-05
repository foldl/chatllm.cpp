namespace lm
{
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
}

namespace coder
{
    struct Config : public codellama::Config
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
        Tokenizer(const llama::Config &config)
            : llama::Tokenizer::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.";
        }
    };

    class ConditionalGeneration : public codellama::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, ModelType type)
            : codellama::ConditionalGeneration(config, type)
        {}

        ConditionalGeneration(const Config &config)
            : ConditionalGeneration(config, MODEL_TYPE_WIZARDCODER)
        {
        }
    };

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << ai << "\n\n";

        auto text = oss_prompt.str();

        append_user(round_idx, user, ids);
        tok->encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
            oss_prompt << tok->get_system_prompt() << "\n\n";

        oss_prompt << "### Instruction:\n" + user << "\n\n### Response:\n";

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }
}

namespace math
{
    struct Config : public mistral::Config
    {
    };

    class Tokenizer : public coder::Tokenizer
    {
    public:
        Tokenizer(const llama::Config &config)
            : coder::Tokenizer(config)
        {}
    };

    class ConditionalGeneration : public mistral::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config)
            : mistral::ConditionalGeneration(config, MODEL_TYPE_WIZARDMATH)
        {
        }
    };
}
