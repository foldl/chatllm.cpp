// ===== ChatLLM2-6B =====

struct Config : public glm::v2::Config
{
};

class Tokenizer : public glm::v2::Tokenizer
{
public:
    Tokenizer(const Config &config) : glm::v2::Tokenizer::Tokenizer(config) {}

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

protected:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;

public:
    int system_token_id;
    int user_token_id;
    int assistant_token_id;
    int observation_token_id;
};

class ConditionalGeneration : public glm::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : glm::v2::ConditionalGeneration(config, MODEL_TYPE_CHATGLM3)
    {
    }
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    size_t size = glm::v2::Tokenizer::load(buffer, n_vocab);

    int token_id = eop_token_id + 1;
    system_token_id = token_id++;
    user_token_id   = token_id++;
    assistant_token_id  = token_id++;
    observation_token_id= token_id++;
    return size;
}

void Tokenizer::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    if ((round_idx == 0) && (sys_prompt.size() > 0))
    {
        ids.push_back(system_token_id);
        encode(sys_prompt, ids);
    }

    ids.push_back(user_token_id);
    encode(user, ids);
    ids.push_back(assistant_token_id);
    encode(ai, ids);
}

void Tokenizer::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    if ((round_idx == 0) && (sys_prompt.size() > 0))
    {
        ids.push_back(system_token_id);
        encode(sys_prompt, ids);
    }

    ids.push_back(user_token_id);
    encode(user, ids);
    ids.push_back(assistant_token_id);
}

bool Tokenizer::is_special_id(int id) const
{
    return glm::v2::Tokenizer::is_special_id(id)
        || (id == system_token_id)
        || (id == user_token_id)
        || (id == assistant_token_id)
        || (id == observation_token_id);
}
