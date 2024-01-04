struct Config : public llama::Config
{
    float rope_theta;
};

class Tokenizer : public llama::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : llama::Tokenizer::Tokenizer(config)
    {
        sys_prompt = "";
    }
};

class ConditionalGeneration : public llama::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config)
        : ConditionalGeneration(config, MODEL_TYPE_CODELLAMA)
    {
    }

    ConditionalGeneration(const Config &config, ModelType type)
        : llama::ConditionalGeneration(config, type)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer.layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
};
