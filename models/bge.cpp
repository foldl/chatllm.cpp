namespace embedding
{
    typedef bce::embedding::Config Config;

    typedef bce::embedding::Tokenizer Tokenizer;

    class ConditionalGeneration : public bce::embedding::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : bce::embedding::ConditionalGeneration(config, MODEL_TYPE_BGE_M3, MEM_SIZE, SCRATCH_SIZE)
        {}
    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 544ull * 1024 * 1024;
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
            : bce::ranker::ConditionalGeneration(config, MODEL_TYPE_BGE_ReRanker_M3, MEM_SIZE, SCRATCH_SIZE)
        {}
    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 544ull * 1024 * 1024;
    };
}