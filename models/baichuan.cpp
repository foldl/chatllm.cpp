struct Config : public llama::Config
{
    int user_token_id;
    int assistant_token_id;
};

class Tokenizer : public llama::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::Tokenizer::Tokenizer(config),
        user_token_id(config.user_token_id),
        assistant_token_id(config.assistant_token_id)
    {
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

protected:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;

private:
    int user_token_id;
    int assistant_token_id;
};

class ConditionalGeneration : public llama::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::ConditionalGeneration(config, MODEL_TYPE_BAICHUAN)
    {
    }
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::SentencePieceProcessor();
    size_t size = tp->Load(buffer, n_vocab);
    return size;
}

void Tokenizer::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    if ((round_idx == 0) && (sys_prompt.size() > 0))
        encode(sys_prompt, ids, false, false);

    ids.push_back(user_token_id);
    encode(user, ids, false, false);
    ids.push_back(assistant_token_id);
    encode(user, ids, false, false);
}

void Tokenizer::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    if ((round_idx == 0) && (sys_prompt.size() > 0))
        encode(sys_prompt, ids, false, false);

    ids.push_back(user_token_id);
    encode(user, ids, false, false);
    ids.push_back(assistant_token_id);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id) || (id == user_token_id) || (id == assistant_token_id);
}
