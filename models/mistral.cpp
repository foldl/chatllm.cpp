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

const int SLIDING_WINDOW_LEN            =  4096;

class Tokenizer : public llama::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : llama::Tokenizer::Tokenizer(config, encoder)
    {
        sys_prompt = "";
    }
};

class ConditionalGeneration : public llama::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config, ModelType type)
        : llama::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>(config, type,
            config.num_key_value_heads, config.max_length, 13)
    {
        CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == SLIDING_WINDOW_LEN))
            << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer.layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        batch_input = false;
    }

    ConditionalGeneration(const Config &config)
        : ConditionalGeneration(config, MODEL_TYPE_MISTRAL)
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

    oss_prompt << "[INST] " << user << " [/INST]";
    tok->encode(oss_prompt.str(), ids, true, false);
}
