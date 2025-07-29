#include "qwen.h"
#include "../src/models_priv.h"

namespace chatllm::jiutian
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int tie_word_embeddings;
        float rope_theta;
    };

    static qwen::v2::Config convert(const Config &config)
    {
        qwen::v2::Config r;
        *(BaseConfig *)&r = *(BaseConfig *)&config;
        r.num_key_value_heads = config.num_key_value_heads;
        r.sliding_window      = -1;
        r.rope_theta          = config.rope_theta;
        return r;
    }

    typedef qwen::v2::Tokenizer Tokenizer;

    class ConditionalGeneration : public qwen::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : qwen::v2::ConditionalGeneration(convert(config), runtime_config, MODEL_TYPE_JIUTIAN, config.tie_word_embeddings != 0)
        {}
    };

    REGISTER_MODEL_LOADER(JIUTIAN, jiutian, 1);
}