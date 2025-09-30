#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::siglip
{
    const int CONFIG_GLOBAL_POOL_NONE           = 0;
    const int CONFIG_GLOBAL_POOL_MAP            = 1;


    struct Config
    {
        ggml::type dtype;
        int image_size;
        int patch_size;
        int width;
        int layers;
        int heads;
        float mlp_ratio;
        int global_pool;

        int lm_hidden_size;
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int image_width, int in_dim = 3);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int  num_patches_per_side;
        const int  num_patches;
        Conv2D     patch_embedding;
        Embedding  position_embedding;
    };

    typedef TheBiasedGELUMLP MLP2;

    class MLPProjector : public TheMLP
    {
    public:
        MLPProjector(InitContext *ctx, int in_dim, int out_dim, ActFunc act = ActFunc::GELU, bool bias = true);
    };

    class ViTSelfAttention : public BaseCachelessAttention
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length);
    };

    class AttentionPoolLatent : public BaseCachelessAttention
    {
    public:
        AttentionPoolLatent(InitContext *ctx,
            int embed_dim,
            int num_heads,
            int max_length,
            float mlp_ratio = 4.0f);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        ggml::tensor *latent;
        LayerNorm     norm;
        MLP2          mlp;
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int max_length);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int image_size;
        const int hidden_size;
        const int patch_size;
        LayerNorm norm;
        std::unique_ptr<AttentionPoolLatent> attn_pool;
        MLPProjector aligner;
    };

    class VisionTransformer : public DynamicBlock
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP2> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    public:
        const int max_length;
        PatchEmbedding embeddings;
        MultiModalProjector multi_modal_projector;
        std::vector<std::unique_ptr<LayerBlock>> layers;
    };

    class VisualEmbeddingGeneration : public BaseMediaProjectedEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config);
        bool load_model(Config vis_config);
        bool load(const std::string &path, ModelLoader &loader);
    protected:
        ggml::tensor *make_media_tensor(ComputeContext *ctx, const BaseTokenizer::MediaAsEmbeddingVector &media) override;
    public:
        Config config;
    };
}