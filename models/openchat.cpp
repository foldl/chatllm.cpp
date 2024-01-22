typedef  mistral::Config Config;

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
        : mistral::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "GPT4";
    }

    int get_terminate_token_id(void) const override { return eos_token_id; }
};

class ConditionalGeneration : public mistral::ConditionalGeneration
{
public:
    ConditionalGeneration(const Config &config)
        : mistral::ConditionalGeneration(config, MODEL_TYPE_OPENCHAT)
    {
    }
};

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    // {{ bos_token }}
    // {% for message in messages %}
    //      {{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}
    // {% endfor %}
    // {% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_user(round_idx, user, ids);

    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss_prompt;

    if (0 == round_idx)
        ids.push_back(tok->bos_token_id);

    oss_prompt << tok->get_system_prompt() << " Correct User: " << user;
    tok->encode(oss_prompt.str(), ids, false, true);

    oss_prompt.clear();
    oss_prompt << tok->get_system_prompt() << " Correct Assistant: ";
    tok->encode(oss_prompt.str(), ids, false, false);
}