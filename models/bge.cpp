#include "bce.h"

namespace chatllm::bge::embedding
{
    typedef bce::embedding::Config Config;

    typedef bce::embedding::Tokenizer Tokenizer;

    class ConditionalGeneration : public bce::embedding::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : bce::embedding::ConditionalGeneration(config, runtime_config, MODEL_TYPE_BGE_M3)
        {}
    public:


    };
}

namespace chatllm::bge::ranker
{
    typedef bce::ranker::Config Config;

    typedef bce::ranker::Tokenizer Tokenizer;

    class ConditionalGeneration : public bce::ranker::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : bce::ranker::ConditionalGeneration(config, runtime_config, MODEL_TYPE_BGE_ReRanker_M3)
        {}
    public:


    };
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(BGE_M3,                bge::embedding, 1);
    REGISTER_MODEL_LOADER(BGE_ReRanker_M3,       bge::ranker, 1);
}