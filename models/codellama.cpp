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

    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : ConditionalGeneration(config, runtime_config, MODEL_TYPE_CODELLAMA)
    {
    }

    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : llama::v2::ConditionalGeneration(config, runtime_config, type)
    {
        auto transformer = Base::get_typed_transformer<ModelClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
};
