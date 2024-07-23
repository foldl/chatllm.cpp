struct Config : public llama::v2::Config
{
    int num_key_value_heads;
    float rope_scaling;
    float rope_theta;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public llama::v2::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::v2::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

public:
    int instruct_token_id;
    int response_token_id;
};

class ConditionalGeneration : public llama::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::v2::ConditionalGeneration(config, MODEL_TYPE_TIGERBOT, config.num_key_value_heads, config.max_length)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
            attention.freq_scale = 1 / config.rope_scaling;
        }
    }
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    size_t size = llama::v2::Tokenizer::load(buffer, n_vocab);
    response_token_id = pad_token_id - 1;
    instruct_token_id = pad_token_id - 2;
    return size;
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);

    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->bos_token_id);
    ids.push_back(tok->instruct_token_id);
    tok->encode(user, ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->response_token_id);
}

bool Tokenizer::is_special_id(int id) const
{
    return llama::v2::Tokenizer::is_special_id(id)
            || (id == bos_token_id)
            || (id == eos_token_id)
            || (id == instruct_token_id)
            || (id == response_token_id);
}
