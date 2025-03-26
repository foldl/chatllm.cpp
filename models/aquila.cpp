namespace v2
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_theta;
        float rope_scaling_factor;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        enum SeparatorStyle
        {
            ADD_COLON_SINGLE,
            ADD_COLON_TWO,
            ADD_COLON_SPACE_SINGLE,
            NO_COLON_SINGLE,
            NO_COLON_TWO,
            ADD_NEW_LINE_SINGLE,
        };

        ChatHistoryEncoder()
            : sep_style(ADD_COLON_SINGLE)
        {
        }

        void setup(const std::string &t)
        {
            if (t == "aquila-chat")
            {
                def_sys_prompt = "A chat between a curious human and an artificial intelligence assistant. "
                                 "The assistant gives helpful, detailed, and polite answers to the human's questions.";
                role_tags = {"Human", "Assistant", "System"};
                sep_style = SeparatorStyle::ADD_COLON_SINGLE;
                seps= {"###", ""};
            }
            else if (t == "aquila-legacy")
            {
                def_sys_prompt = "A chat between a curious human and an artificial intelligence assistant. "
                                 "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n";
                role_tags = {"### Human: ", "### Assistant: ", "System"};
                sep_style = SeparatorStyle::NO_COLON_TWO;
                seps= {"\n", "</s>"};
            }
            else if (t == "aquila")
            {
                def_sys_prompt = "A chat between a curious human and an artificial intelligence assistant. "
                                 "The assistant gives helpful, detailed, and polite answers to the human's questions.";
                role_tags = {"Human", "Assistant", "System"};
                sep_style = SeparatorStyle::ADD_COLON_TWO;
                seps= {"###", "</s>"};
            }
            else if (t == "aquila-v1")
            {
                def_sys_prompt = "";
                role_tags = {"<|startofpiece|>", "<|endofpiece|>", ""};
                sep_style = SeparatorStyle::NO_COLON_TWO;
                seps= {"", "</s>"};
            }
            else
                CHATLLM_CHECK(false) << "unknown chat template: " << t;
        }

        void append_sys_prompt(std::vector<int> &ids) const override
        {
            //ids.push_back(tokenizer->bos_token_id);

            switch (sep_style)
            {
            case ADD_COLON_SINGLE:
            case ADD_COLON_TWO:
            case ADD_COLON_SPACE_SINGLE:
                tokenizer->encode(tokenizer->get_system_prompt() + seps[0], ids);
                break;
            case ADD_NEW_LINE_SINGLE:
                if (tokenizer->get_system_prompt().size() > 0)
                    tokenizer->encode(tokenizer->get_system_prompt() + seps[0], ids);
                break;
            case NO_COLON_SINGLE:
            case NO_COLON_TWO:
                tokenizer->encode(tokenizer->get_system_prompt(), ids);
                break;
            default:
                break;
            }
        }
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {
            append_ai_opening(round_idx, ids);
            append_msg(ai, ids);
        }

        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
        {
            append_role(0, ids);
            append_msg(user, ids);
        }

        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {
            append_role(1, ids);
        }

    private:
        void append_msg(const std::string &msg, std::vector<int> &ids) const
        {
            switch (sep_style)
            {
            case ADD_COLON_SINGLE:
            case ADD_COLON_SPACE_SINGLE:
            case ADD_NEW_LINE_SINGLE:
            case NO_COLON_SINGLE:
                tokenizer->encode(msg + seps[0], ids);
                break;
            case ADD_COLON_TWO:
            case NO_COLON_TWO:
                tokenizer->encode(msg + seps[1], ids);
                break;
            default:
                break;
            }
        }
        void append_role(int id, std::vector<int> &ids) const
        {
            switch (sep_style)
            {
            case ADD_COLON_SINGLE:
            case ADD_COLON_TWO:
            case ADD_COLON_SPACE_SINGLE:
                tokenizer->encode(role_tags[1] + ": ", ids);
                break;
            case ADD_NEW_LINE_SINGLE:
                tokenizer->encode(role_tags[1] + "\n", ids);
                break;
            case NO_COLON_SINGLE:
            case NO_COLON_TWO:
                tokenizer->encode(role_tags[1], ids);
                break;
            default:
                break;
            }
        }
    public:
        std::string def_sys_prompt;
        SeparatorStyle sep_style;
        std::vector<std::string> role_tags;
        std::vector<std::string> seps;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            // "AquilaChat2-7B": "aquila-v1",
            // "AquilaChat2-34B": "aquila-legacy",
            // "AquilaChat2-7B-16K": "aquila",
            // "AquilaChat2-34B-16K": "aquila"

            bool is_16k = config.max_length >= 16384;
            bool is_34b  = config.num_hidden_layers > 32;

            if (is_16k)
                _chat_encoder.setup("aquila");
            else
            {
                _chat_encoder.setup(is_34b ? "aquila-legacy" : "aquila-v1");
            }

            sys_prompt = _chat_encoder.def_sys_prompt;
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);
            tp->EnableReturnSpecialToken(true);

            return size;
        }
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_AQUILA2)
            : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {
            auto transformer = Base::get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.freq_base = config.rope_theta;
                if (config.rope_scaling_factor > 0)
                    attention.freq_scale = 1 / config.rope_scaling_factor;
            }
        }
    };
}