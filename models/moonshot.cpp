namespace moonlight
{
    struct Config : public deepseek::v3_light::Config
    {
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public qwen::v1::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : qwen::v1::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = "You are a helpful assistant.";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t size = do_load(buffer, n_vocab);
            tp->EnableReturnSpecialToken(true);

            CHATLLM_CHECK(163584 == bos_token_id) << "bos_token_id must be 163584, but " << bos_token_id;

            eos_token_id                = 163585;
            im_end_token_id             = 163586;
            im_middle_token_id          = 163601;
            im_user_token_id            = 163587;
            im_assistant_token_id       = 163588;
            im_system_token_id          = 163594;
            pad_token_id                = 163838;

            terminate_ids.insert(im_end_token_id);

            return size;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override
        {
            return encode(-1, text, ids, false);
        }

    public:
        void encode(int tag_token_id, const std::string &text, std::vector<int> &ids, bool add_end_tok = false) const
        {
            if (tag_token_id >= 0)
                ids.push_back(tag_token_id);
            BaseTokenizer::encode(text, ids);
            if (tag_token_id >= 0)
                ids.push_back(im_middle_token_id);
            if (add_end_tok)
                ids.push_back(im_end_token_id);
        }

    public:
        int im_system_token_id;
        int im_user_token_id;
        int im_assistant_token_id;
        int im_middle_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(-1, ai, ids, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(tok->im_system_token_id, "system", ids);
        tok->encode(-1, tok->get_system_prompt(), ids, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user_opening(round_idx, ids);
        tok->encode(-1, user, ids, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(tok->im_assistant_token_id, "assistant", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(tok->im_user_token_id, "user", ids);
    }

    class ConditionalGeneration : public deepseek::v3_light::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : deepseek::v3_light::ConditionalGeneration(config, runtime_config, MODEL_TYPE_MOONLIGHT)
        {
        }
    };
}