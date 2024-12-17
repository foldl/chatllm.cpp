namespace v3
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override
        {
            if (tokenizer->get_system_prompt().size() > 0)
            {
                std::ostringstream oss;
                oss << "'<|system|>\n" << tokenizer->get_system_prompt();
                tokenizer->encode(oss.str(), ids);
            }
        }
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "'<|assistant|>\n" << ai;
            tokenizer->encode(oss.str(), ids);
            ids.push_back(tokenizer->eos_token_id);
        }

        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "'<|user|>\n" << user;
            tokenizer->encode(oss.str(), ids);
        }

        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "'<|assistant|>\n";
            tokenizer->encode(oss.str(), ids);
        }
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr)
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);
            //tp->EnableReturnSpecialToken(true);
            int nl = tp->PieceToId("Ċ");
            printf("nl = %d\n", nl);
            tp->OverrideTokenDecoding(nl, "\n");

            return size;
        }
    };

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_FALCON3)
        {}
    };
}