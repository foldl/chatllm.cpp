namespace deepseek
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        float rope_scaling;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public deepseek_coder::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : deepseek_coder::Tokenizer::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = "";
            terminate_ids.emplace(bos_token_id);
        }
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_CODEFUSE_DEEPSEEK)
            : ConditionalGeneration(config, type, config.num_key_value_heads, config.max_length)
        {};

        ConditionalGeneration(const Config &config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, type, num_key_value_heads, max_length),
              config(config)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer.layers[i].attention;
                attention.freq_base = config.rope_theta;
                attention.freq_scale = 1 / config.rope_scaling;
            }

            if (transformer.get_param_num(false) > 20000000)
                GRAPH_SIZE = 4096;
        }

    private:
        Config config;
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

        if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
        {
            oss_prompt << "system\n"
                    << tok->get_system_prompt() << "\n";
            tok->encode(oss_prompt.str(), ids, true, false);
        }

        oss_prompt.str("");
        oss_prompt << "human\n" << user << "\n";
        tok->encode(oss_prompt.str(), ids, true, false);

        tok->encode("bot\n", ids, true, false);
    }
}