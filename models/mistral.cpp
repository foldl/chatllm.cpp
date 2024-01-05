struct Config : public llama::Config
{
    int num_key_value_heads;
    int sliding_window;
    float rope_theta;
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
        sys_prompt = "";
    }
};

class ConditionalGeneration : public llama::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config, ModelType type);

    ConditionalGeneration(const Config &config)
        : ConditionalGeneration(config, MODEL_TYPE_MISTRAL)
    {}
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

    oss_prompt << "[INST] " << user << " [/INST]";
    tok->encode(oss_prompt.str(), ids, true, false);
}

ConditionalGeneration::ConditionalGeneration(const Config &config, ModelType type)
    : llama::ConditionalGeneration(config, type,
            config.num_key_value_heads,
            config.sliding_window > 0 ? config.sliding_window : config.max_length)
{
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        auto &attention = transformer.layers[i].attention;
        attention.freq_base = config.rope_theta;
    }
}
