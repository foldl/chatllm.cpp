struct Config : public mistral::Config
{
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public mistral::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : mistral::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }

    size_t load(const char *buffer, int n_vocab) override
    {
        size_t r = mistral::Tokenizer::load(buffer, n_vocab);
        newline_token_id = tp->PieceToId("\n");
        tp->Encode("<|im_end|>", &chat_terminate_seq);
        return r;
    }

    int get_terminate_token_id(void) const override { return eos_token_id; }

    void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const override
    {
        if (add_bos)
            mistral::Tokenizer::encode("<|im_start|>", ids);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            mistral::Tokenizer::encode("<|im_end|>\n", ids);
    }

public:
    int newline_token_id;
    std::vector<int> chat_terminate_seq;
};

class ConditionalGeneration : public mistral::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config)
        : mistral::ConditionalGeneration(config, MODEL_TYPE_NEURALBEAGLE)
    {
    }

protected:
    bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override
    {
        if (output_ids.size() < 1) return false;

        Tokenizer *tokenizer = dynamic_cast<Tokenizer *>(this->tokenizer);
        if (match_output_sequence(output_ids, tokenizer->chat_terminate_seq))
        {
            pop_output = (int)tokenizer->chat_terminate_seq.size();
            return true;
        }

        keep_idx = ((int)output_ids.size()) - ((int)tokenizer->chat_terminate_seq.size()) + 1;
        return false;
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
    oss_prompt << "user\n"
               << user;
    tok->encode(oss_prompt.str(), ids, true, true);
    tok->encode("assistant\n", ids, true, false);
}
