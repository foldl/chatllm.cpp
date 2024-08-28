struct Config : public glm::v2::Config
{
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public glm::v2::Tokenizer
{
public:
    Tokenizer(const Config &config) : glm::v2::Tokenizer::Tokenizer(config, &_chat_encoder)
    {
        user_name = "Human";
        bot_name = "CharacterGLM";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    void set_additional_args(const std::map<std::string, std::string> &args) override;
public:
    std::string user_name;
    std::string bot_name;
    std::string user_info;
    std::string bot_info;
};

class ConditionalGeneration : public glm::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : glm::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_CHARACTERGLM)
    {
    }
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    size_t size = glm::v2::Tokenizer::load(buffer, n_vocab);

    eos_token_id = tp->PieceToId("\n");
    return size;
}

void Tokenizer::set_additional_args(const std::map<std::string, std::string> &args)
{
    #define get_value(n) do {                       \
        auto it = args.find(#n);                    \
        if (it != args.end()) n = it->second;       \
    } while (0)

    get_value(user_name);
    get_value(bot_name);
    get_value(user_info);
    get_value(bot_info);

    const static std::regex r(R""([\r\n]+)"");
    user_info = std::regex_replace(user_info, r, " ");
    bot_info  = std::regex_replace(bot_info, r, " ");
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    append_ai_opening(round_idx, ids);
    std::ostringstream oss;
    oss << ai << "\n";
    tokenizer->encode(oss.str(), ids);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss;

    ids.insert(ids.end(), {tok->gmask_token_id, tok->sop_token_id}); // special prefix

    oss << "以下是一段" << tok->user_name << "和" << tok->bot_name << "之间的对话。\n";
    if (tok->bot_info.size() > 0)
    {
        oss << "关于"
            << tok->bot_name << "的信息："
            << tok->bot_info << "\n";
    }
    if (tok->user_info.size() > 0)
    {
        oss << "关于"
            << tok->user_name << "的信息："
            << tok->user_info << "\n";
    }
    tok->encode(oss.str(), ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss;
    oss << "[" << tok->user_name << "]" << user << "\n";
    tok->encode(oss.str(), ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss;
    oss << "[" << tok->bot_name << "]";
    tok->encode(oss.str(), ids);
}
