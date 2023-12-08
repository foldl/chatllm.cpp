struct Config : public llama::Config
{
};

class Tokenizer : public llama::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::Tokenizer::Tokenizer(config)
    {
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

protected:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

class ConditionalGeneration : public llama::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::ConditionalGeneration(config, MODEL_TYPE_DEEPSEEK)
    {
    }
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor();
    size_t size = tp->Load(buffer, n_vocab);
    printf("bos = %d\n", bos_token_id);
    return size;
}

void Tokenizer::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    if ((round_idx == 0) && (sys_prompt.size() > 0))
        oss_prompt << sys_prompt << "\n\n";

    oss_prompt << "User: " << user << "\n\n"
               << "Assistant: " << ai;

    auto text = oss_prompt.str();
    encode(text, ids, true, true);
}

void Tokenizer::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    if ((round_idx == 0) && (sys_prompt.size() > 0))
        oss_prompt << sys_prompt << "\n\n";

    oss_prompt << "User: " << user << "\n\n"
               << "Assistant: ";

    auto text = oss_prompt.str();
    encode(text, ids, true, false);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id);
}
