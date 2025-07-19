#pragma  once
#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::bce::embedding
{
    struct Config : public BaseConfig
    {
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    protected:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos, int max_length) const;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, BCEFinalNorm, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
        void load(ModelLoader &loader) override;
        int get_text_embedding_dim(void) const override;
    public:
        Config config;
    };
}

namespace chatllm::bce::ranker
{
    struct Config : public embedding::Config
    {
    };

    class Tokenizer : public embedding::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const override;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaClassificationHead, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);

        void load(ModelLoader &loader) override;
    public:
        Config config;
    };
}