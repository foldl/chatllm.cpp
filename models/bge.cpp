namespace embedding
{
    typedef bce::embedding::Config Config;

    typedef bce::embedding::Tokenizer Tokenizer;

    class ConditionalGeneration : public bce::embedding::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : bce::embedding::ConditionalGeneration(config, MODEL_TYPE_BGE_M3)
        {}
    };
}

namespace ranker
{
    typedef bce::ranker::Config Config;

    typedef bce::ranker::Tokenizer Tokenizer;

    class ConditionalGeneration : public bce::ranker::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : bce::ranker::ConditionalGeneration(config, MODEL_TYPE_BGE_ReRanker_M3)
        {}
    };
}