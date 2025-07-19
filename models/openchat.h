#pragma once
#include "mistral.h"

namespace chatllm::openchat
{
    typedef  mistral::mistral::Config Config;

    class Tokenizer : public mistral::mistral::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
        Tokenizer(const Config &config, BaseHistoryEncoder *encoder);
    };

    class ConditionalGeneration : public mistral::mistral::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);
    };
}