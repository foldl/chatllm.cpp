namespace readerlm
{
    typedef qwen::v2::Config Config;

    class Tokenizer : public qwen::v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : qwen::v2::Tokenizer(config)
        {
            sys_prompt = "You are an AI assistant developed by Jina AI.";
        }
    };

    class ConditionalGeneration : public qwen::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_READERLM2)
            : qwen::v2::ConditionalGeneration(config, runtime_config, type, true)
        {}
    };
}