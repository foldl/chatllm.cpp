#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::pangu::moe
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int moe_intermediate_size;
        int num_experts_per_tok;
        int num_experts;

        float rope_theta;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void encode_item(const char *tag, std::vector<int> &ids);
        void encode_item(const char *tag, const std::string &content, std::vector<int> &ids);
    public:
        int unused9_token_id;
        int unused10_token_id;
    };

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        void load(ModelLoader &loader);
    };
}
