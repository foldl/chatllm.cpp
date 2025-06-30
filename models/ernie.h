#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"

#include "llama.h"

namespace chatllm::ernie::dense
{
    struct Config : public chatllm::llama::v2::Config
    {
        int num_key_value_heads;
        int head_dim;
        int tie_word_embeddings;
        float rope_theta;
    };

    class Tokenizer : public chatllm::llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
    };

    class ConditionalGeneration : public chatllm::llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_ERNIE_DENSE);
    };
}