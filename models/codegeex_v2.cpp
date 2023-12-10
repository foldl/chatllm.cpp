// ===== ChatLLM2-6B =====

struct Config : public glm::v2::Config
{
};

class Tokenizer : public glm::v2::Tokenizer
{
public:
    Tokenizer(const Config &config) : glm::v2::Tokenizer::Tokenizer(config)
    {
        sys_prompt = "# language: Python";
    }

protected:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
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

void Tokenizer::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    return;
    std::string combined = sys_prompt + "\n" + user;
    glm::v2::Tokenizer::append_pair(round_idx, combined, ai, ids);
}

void Tokenizer::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::string combined = sys_prompt + "\n" + user + "\n";
    encode(combined, ids);
}
