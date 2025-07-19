#include "llama.h"

namespace chatllm::yi
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        float rope_scaling;
        float rope_theta;
    };

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

    public:
        int im_start_token_id;
        int im_end_token_id;
        int im_sep_token_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_YI);

    public:
        Config config;
    };
}