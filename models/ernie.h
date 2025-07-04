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
        Tokenizer(const BaseConfig &config);
    };

    class ConditionalGeneration : public chatllm::llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_ERNIE_DENSE);
    };
}

namespace chatllm::ernie::moe
{
    struct Config : public chatllm::llama::v2::Config
    {
        int num_key_value_heads;
        int tie_word_embeddings;
        int moe_num_experts;
        int moe_num_shared_experts;
        int moe_layer_start_index;
        int moe_intermediate_size;
        int moe_capacity[3];
        int moe_k;
        int moe_layer_interval;
        int use_correction_bias;

        float rope_theta;
    };

    typedef dense::Tokenizer Tokenizer;

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        void load(ModelLoader &loader);
    };
}
