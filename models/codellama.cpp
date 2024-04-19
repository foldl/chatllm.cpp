struct Config : public llama::v2::Config
{
    float rope_theta;
};

class Tokenizer : public llama::v2::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::v2::Tokenizer::Tokenizer(config)
    {
        sys_prompt = "";
    }
};

class ConditionalGeneration : public llama::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config)
        : ConditionalGeneration(config, MODEL_TYPE_CODELLAMA)
    {
    }

    ConditionalGeneration(const Config &config, ModelType type)
        : llama::v2::ConditionalGeneration(config, type)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer.layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
};
