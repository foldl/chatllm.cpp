namespace dense
{
typedef llama::v2::Config Config;

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
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
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor3();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void encode(const std::string &text, std::vector<int> &ids) const override
    {
        encode(text, ids, false, false);
    }

public:
    virtual void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
    {
        if (add_bos)
            ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            ids.push_back(eos_token_id);
    }
};

class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_XVERSE)
        : ConditionalGeneration(config, type, config.num_attention_heads, config.max_length)
    {}

    ConditionalGeneration(const Config &config, ModelType type,
                        int num_key_value_heads, int max_length)
        : GenericConditionalGeneration<LlamaBlock>(config, type, num_key_value_heads, max_length)
    {}
};

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_user(round_idx, user, ids);
    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    //if (round_idx == 0)
    //    ids.push_back(tok->bos_token_id);

    oss_prompt << "Human: " << user
               << "\n\nAssistant: ";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, false);

    for (auto x : ids)
        printf("%d, ", x);
    printf("\n");
}
}