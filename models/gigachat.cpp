typedef deepseek::v1_moe::Config Config;

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

    Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder)
        : BaseTokenizer(config, chat_encoder)
    {
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

public:
    int role_sep_token_id;
    int message_sep_token_id;
    int bracket_l_token_id;
    int bracket_r_token_id;
};

class ConditionalGeneration : public deepseek::v1_moe::ConditionalGeneration
{
public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : deepseek::v1_moe::ConditionalGeneration(config, runtime_config, MODEL_TYPE_GIGACHAT)
    {}
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor2(
        {
            // FIXME: different form llama3

            // llama3: (?i:'s|'t|'re|'ve|'m|'ll|'d)
            // [^\r\n\p{L}\p{N}]?\p{L}+
            //  \p{N}{1,3}
            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            // \s*[\r\n]+|\s+(?!\S)
            // \s+

            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
            // \p{N}{1,3}
            //  ?[^\s\p{L}\p{N}]+[\r\n/]*
            //  \s*[\r\n]+|\s+(?!\S)
            // \s+
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }
    );
    size_t size = tp->Load(buffer, n_vocab);

    role_sep_token_id       = tp->PieceToId("<|role_sep|>");
    message_sep_token_id    = tp->PieceToId("<|message_sep|>");
    bracket_l_token_id      = tp->PieceToId("[");
    bracket_r_token_id      = tp->PieceToId("]");

    terminate_ids.insert(message_sep_token_id);

    return size;
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->bos_token_id);
    tok->encode(tok->get_system_prompt(), ids);
    ids.push_back(tok->message_sep_token_id);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_user_opening(round_idx, ids);
    tok->encode(user, ids);
    ids.push_back(tok->message_sep_token_id);
    tok->encode("available functions", ids);
    ids.push_back(tok->role_sep_token_id);
    ids.push_back(tok->bracket_l_token_id);
    ids.push_back(tok->bracket_r_token_id);
    ids.push_back(tok->message_sep_token_id);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    tok->encode("assistant", ids);
    ids.push_back(tok->role_sep_token_id);
}

void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    tok->encode("user", ids);
    ids.push_back(tok->role_sep_token_id);
}