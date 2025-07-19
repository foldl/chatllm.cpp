#include "openchat.h"

namespace chatllm::starling
{
    typedef  openchat::Config Config;

    typedef openchat::Tokenizer Tokenizer;

    class ConditionalGeneration : public openchat::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : openchat::ConditionalGeneration(config, runtime_config, MODEL_TYPE_STARLING)
        {
        }
    };

    REGISTER_MODEL_LOADER(STARLING,              starling, 1);
}