#include "llama.h"

namespace chatllm::megrez::chat
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
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
            sys_prompt = "你是Megrez-3B-Instruct，将针对用户的问题给出详细的、积极的回答。";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_role, bool add_turn_end) const;

    public:
        int role_start_token_id;
        int role_end_token_id;
        int turn_end_token_id;
        int function_start_token_id;
        int function_end_token_id;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        role_start_token_id           = tp->PieceToId("<|role_start|>");
        role_end_token_id             = tp->PieceToId("<|role_end|>");
        function_start_token_id       = tp->PieceToId("<|function_start|>");
        function_end_token_id         = tp->PieceToId("<|function_end|>");

        if (eos_token_id < 0)
            eos_token_id              = tp->PieceToId("<|eos|>");

        turn_end_token_id             = tp->PieceToId("<|turn_end|>");
        terminate_ids.insert(turn_end_token_id);

        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_role, bool add_turn_end) const
    {
        if (add_role)
            ids.push_back(role_start_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_role)
            ids.push_back(role_end_token_id);
        if (add_turn_end)
            ids.push_back(turn_end_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode("system", ids, true, false);
        tok->encode(tok->get_system_prompt(), ids, false, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        tok->encode("user", ids, true, false);
        tok->encode(user, ids, false, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("assistant", ids, true, false);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("user", ids, true, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == role_start_token_id) || (id == role_end_token_id);
    }

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_MEGREZ)
        {}
    };

    REGISTER_MODEL_LOADER(MEGREZ,                megrez::chat, 1);
}