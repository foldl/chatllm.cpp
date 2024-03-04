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
    Tokenizer(const Config &config) : glm::v2::Tokenizer::Tokenizer(config, &_chat_encoder) {}

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

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

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
    {
        ids.push_back(tok->system_token_id);
        tok->encode(tok->get_system_prompt(), ids);
    }

    ids.push_back(tok->user_token_id);
    tok->encode(user, ids);
    ids.push_back(tok->assistant_token_id);
    tok->encode(ai, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
    {
        ids.push_back(tok->system_token_id);
        tok->encode(tok->get_system_prompt(), ids);
    }

    ids.push_back(tok->user_token_id);
    tok->encode(user, ids);
    ids.push_back(tok->assistant_token_id);
}

bool Tokenizer::is_special_id(int id) const
{
    return glm::v2::Tokenizer::is_special_id(id)
        || (id == system_token_id)
        || (id == user_token_id)
        || (id == assistant_token_id)
        || (id == observation_token_id);
}
