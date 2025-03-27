namespace moe
{
    struct Config : public deepseek::v1_moe::Config
    {
        int head_dim;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : BaseTokenizer(config, &_chat_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    // '(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
                    "\\p{N}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
                    "\\s*[\\r\\n]",
                    "\\s+(?!\\S)",
                    "\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            role_open_token_id = tp->PieceToId("<role>");

            if (role_open_token_id >= 0)
                terminate_ids.insert(role_open_token_id);

            return size;
        }

    protected:
        int role_open_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);
        tokenizer->encode(ai, ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        tokenizer->encode("<role>HUMAN</role>", ids);
        tokenizer->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        tokenizer->encode("<role>ASSISTANT</role>", ids);
    }

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    class ConditionalGeneration : public deepseek::v1_moe::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : deepseek::v1_moe::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>(config, runtime_config, MODEL_TYPE_BAILINGMOE, config.head_dim)
        {}
    };
}