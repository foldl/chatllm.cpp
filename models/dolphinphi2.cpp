class DolphinTokenizer : public BaseTokenizer
{
public:
    DolphinTokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer(config, encoder)
    {
    }

    size_t load(const char *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor();
        size_t size = tp->Load(buffer, n_vocab);
        bos_token_id        = tp->PieceToId("<|endoftext|>");
        im_start_token_id   = tp->GetPieceSize() - 1;
        im_end_token_id     = im_start_token_id - 1;
        eos_token_id        = im_end_token_id;
        return size;
    }

    bool is_special_id(int id) const override
    {
        return (id == pad_token_id);
    }

    int get_terminate_token_id(void) const override { return im_end_token_id; }

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_start, bool add_end) const
    {
        if (add_start)
            ids.push_back(im_start_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_end)
            ids.push_back(im_end_token_id);
    }
    int im_start_token_id;
    int im_end_token_id;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override
    {
        DolphinTokenizer *tok = dynamic_cast<DolphinTokenizer *>(tokenizer);
        append_user(round_idx, user, ids);
        tok->encode(ai, ids, false, true);
    }

    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
    {
        std::ostringstream oss_prompt;

        DolphinTokenizer *tok = dynamic_cast<DolphinTokenizer *>(tokenizer);
        if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
        {
            oss_prompt << "system" << std::endl << tok->get_system_prompt();
            tok->encode(oss_prompt.str(), ids, true, true);
        }
        oss_prompt.clear();
        oss_prompt << "user" << std::endl << user;
        tok->encode(oss_prompt.str(), ids, true, true);

        oss_prompt.clear();
        oss_prompt << "assistant" << std::endl;
        tok->encode(oss_prompt.str(), ids, true, false);
    }
};

static ChatHistoryEncoder _chat_encoder;

namespace v1
{
    struct Config : public phi2::v1::Config
    {
    };

    class Tokenizer : public DolphinTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : DolphinTokenizer(config, &_chat_encoder)
        {
            sys_prompt = "You are Dolphin, a helpful AI assistant.";
        }
    };

    class ConditionalGeneration : public phi2::v1::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config)
            : phi2::v1::ConditionalGeneration(config, MODEL_TYPE_DOLPHINPHI2)
        {}
    protected:
        bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override
        {
            return BaseModelForConditionalGeneration::is_output_terminated(output_ids, keep_idx, pop_output);
        }
    };
}