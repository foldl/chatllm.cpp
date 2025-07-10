#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"
#include "llama.h"

namespace chatllm::smol::lm
{
    typedef llama::v3::Config Config;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_SMOLLM);

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length);
    };
}

namespace chatllm::smol::vit
{
    struct VideoConfig
    {
        float fps;
        int max_frames;
        int longest_edge;
    };

    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;

        int scale_factor;
        int image_seq_len;
        int image_max_size;
        int image_size_per_split;

        float image_mean[3];
        float image_std[3];

        VideoConfig video;
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int image_max_size, int in_dim = 3);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int image_width, int image_height);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int  num_patches_per_side;
        const int  num_patches;
        Conv2D     patch_embedding;
        Embedding  position_embedding;
    };

    typedef TheBiasedGELUMLP MLP2;

    class ViTSelfAttention : public BaseCachelessAttention
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length);
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int image_width, int image_height);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    protected:
        ggml::tensor *pixel_shuffle(ComputeContext *ctx, ggml::tensor *image_features, int image_width, int image_height);
    public:
        const int hidden_size;
        const int scale_factor;
        const int patch_size;
        ggml::merge_patch_param merge_param;
        Linear    proj;
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP2> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int image_width, int image_height);
        bool is_loaded(void) const;
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        LayerNorm  post_layernorm;
        MultiModalProjector multi_modal_projector;
    protected:
        bool loaded;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);
        bool load(ModelLoader &loader);
        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config);
        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);
    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        std::unique_ptr<VisionTransformer> vis_model;
        BackendContext backend_context;
        const size_t GRAPH_SIZE;
        InitContext _ctx; // weight context
        std::string model_gpu_layers;
        const int n_threads;
    public:
        Config vis_config;
    };
}

namespace chatllm::smol::vlm
{
    typedef lm::Config Config;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);
        Tokenizer(const Config &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void inject_image(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count, int col_id, int row_id);
        void inject_global_image(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count);

    public:
        int end_of_utterance_token_id;
        int fake_token_around_image_token_id;
        int global_image_token_token_id;
        int nl_token_id;
        bool do_split;

        std::vector<std::vector<int>> row_col_token_ids;
    };

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int extra_ids) : pad_arg(new BlockParams::PadEmbedding(extra_ids, extra_ids)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        typedef llama::v2::GenericConditionalGeneration<LlamaBlock> Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_SMOL_VLM);

    public:
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;

        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void before_generate(const GenerationConfig &gen_config) override;
    public:
        vit::VisualEmbeddingGeneration visual;
    };
}

namespace chatllm::smol::lm3
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int no_rope_layer_interval;
        int tie_word_embeddings;
        float rope_theta;
    };

    class Tokenizer : public lm::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_SMOLLM3);

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length);
    };
}