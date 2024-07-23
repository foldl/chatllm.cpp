typedef llama::v3::Config Config;

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public llama::v2::Tokenizer
{
public:
    Tokenizer(const Config &config)
            : llama::v2::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
        resevered_0_token_id = 3;
        resevered_1_token_id = 4;
        unk_token_id = config.pad_token_id;
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        size_t size = llama::v2::Tokenizer::load(buffer, n_vocab);

        pad_token_id = unk_token_id;
        return size;
    }

public:
    int resevered_0_token_id;
    int resevered_1_token_id;
    int unk_token_id;
};

class ConditionalGeneration : public llama::v3::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::v3::ConditionalGeneration(config, ModelType::MODEL_TYPE_INDEX)
    {}
};

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    if (tokenizer->get_system_prompt().size() > 0)
    {
        ids.push_back(tokenizer->pad_token_id);
        tokenizer->encode(tokenizer->get_system_prompt(), ids);
    }

}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->resevered_0_token_id);
    tok->encode(user, ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->resevered_1_token_id);
}