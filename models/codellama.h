#include "llama.h"

namespace chatllm::codellama
{
    struct Config : public llama::v2::Config
    {
        float rope_theta;
    };

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
    };

    class ConditionalGeneration : public llama::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);
    };
}