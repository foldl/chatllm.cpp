struct Config : public llama::Config
{
};

class Tokenizer : public llama::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::Tokenizer::Tokenizer(config)
    {
        sys_prompt = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.";
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

protected:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;

public:
    int fim_hole_token_id;
    int fim_begin_token_id;
    int fim_end_token_id;
    int user_token_id;
    int assistant_token_id;
    int eot_token_id;
};

class ConditionalGeneration : public llama::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config)
        : llama::ConditionalGeneration(config, MODEL_TYPE_DEEPSEEK_CODER)
    {
    }

    void load(ModelLoader &loader) override;
};

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor();
    size_t size = tp->Load(buffer, n_vocab);

    std::vector<int> ids;
    tp->Encode("<｜fim▁hole｜><｜fim▁begin｜><｜fim▁end｜><|User|><|Assistant|><|EOT|>", &ids);

    CHATLLM_CHECK(ids.size() == 6) << "tokenizer error";

    fim_hole_token_id = ids[0];
    fim_begin_token_id = ids[1];
    fim_end_token_id = ids[2];
    user_token_id = ids[3];
    assistant_token_id = ids[4];
    eot_token_id = ids[5];
    return size;
}

void Tokenizer::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    if (round_idx == 0)
        oss_prompt << sys_prompt;

    oss_prompt << "\n### Instruction:\n" << user
               << "\n### Response:\n" << ai << "\n<|EOT|>";

    auto text = oss_prompt.str();
    encode(text, ids, true, false);
}

void Tokenizer::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    if (round_idx == 0)
        oss_prompt << sys_prompt;

    oss_prompt << "\n### Instruction:\n" << user
               << "\n### Response:\n";

    auto text = oss_prompt.str();
    encode(text, ids, false, false);
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id) || (id == eot_token_id);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    llama::ConditionalGeneration::load(loader);

    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        auto &attention = transformer.layers[i].attention;
        attention.freq_base = 100000.f;
        attention.freq_scale = 4.0f;
    }
}