typedef  openchat::Config Config;

typedef openchat::Tokenizer Tokenizer;

class ConditionalGeneration : public openchat::ConditionalGeneration
{
public:
    ConditionalGeneration(const Config &config)
        : openchat::ConditionalGeneration(config, MODEL_TYPE_STARLING)
    {
    }
};
