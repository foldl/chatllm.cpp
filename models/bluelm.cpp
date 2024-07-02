struct Config : public llama::v2::Config
{
    int num_key_value_heads;
    float rope_theta;
    float rope_scaling_factor;
    float rope_scaling_power;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public llama::v2::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::v2::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;
public:
    int sea_token_id;
    int seh_token_id;
    int ai_token_id;
    int human_token_id;
};

class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<BlueLMBlock>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, ModelType type = MODEL_TYPE_BLUELM)
        : ConditionalGeneration(config, type, config.num_attention_heads, config.max_length)
    {}

    ConditionalGeneration(const Config &config, ModelType type,
                          int num_key_value_heads, int max_length)
        : llama::v2::GenericConditionalGeneration<BlueLMBlock>(config, type, num_key_value_heads, max_length)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;

            if (config.rope_scaling_power > 0)
            {
                attention.rope_scaling_factor = config.rope_scaling_factor;
                attention.rope_scaling_power  = config.rope_scaling_power;
            }
        }
    }
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);

    sea_token_id = tp->PieceToId("[SEA]");
    seh_token_id = tp->PieceToId("[SEH]");
    ai_token_id    = tp->PieceToId("[|AI|]:");
    human_token_id = tp->PieceToId("[|Human|]:");
    return size;
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_user(round_idx, user, ids);
    tok->encode(user, ids, false, false);
}

void ChatHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->bos_token_id);
    ids.push_back(tok->human_token_id);
    tok->encode(user, ids, false, false);
    ids.push_back(tok->ai_token_id);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id) || (id == human_token_id) || (id == ai_token_id);
}