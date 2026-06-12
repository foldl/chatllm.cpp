#pragma once

#include "../src/layers.h"
#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::kimi::vit
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int init_pos_emb_height;
        int init_pos_emb_width;
        int merge_kernel_size[2];

        int   in_token_limit;
        float image_mean[3];
        float image_std[3];
    };

    class Learnable2DInterpPosEmb : public Block
    {
    public:
        Learnable2DInterpPosEmb() {}
        Learnable2DInterpPosEmb(InitContext *ctx, int height, int width, int dim);

        int64_t get_param_num(bool effective_only) const override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        ggml::tensor *pos_emb;
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int pos_emb_height, int pos_emb_width, int in_dim = 3);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        Conv2D                  proj;
        Learnable2DInterpPosEmb pos_emb;
    };

    typedef TheBiasedGELUMLP MLP2;

    class ViTSelfAttention : public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length);
    protected:
        void before_forward(ComputeContext *ctx, const int n_past, const int qlen) override;

        ggml::tensor *apply_2d_rope(ComputeContext *ctx, ggml::tensor *hidden, int hidden_size, ggml::tensor *pos_w, ggml::tensor *pos_h) const;

        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override;
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override;
    public:
        ggml::tensor *pos_h;
        int grid_w = 0;
        int grid_h = 0;
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    protected:
        ggml::tensor *merge_patch(ComputeContext *ctx, ggml::tensor *x);
        ggml::merge_patch_param merge_param;
    public:
        const int hidden_size;
        LayerNorm pre_norm;
        Linear    linear_1;
        Linear    linear_2;
    };

    class VisionTransformer : public DynamicBlock
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP2> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size, Block *projector);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override;
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        LayerNorm  post_layernorm;
        std::unique_ptr<Block> multi_modal_projector;
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
        virtual DynamicBlock *do_create_transformer(InitContext *ctx, const Config &vis_config, const int lm_hidden_size);
    protected:
        std::unique_ptr<DynamicBlock> vis_model;
        BackendContext backend_context;
        const size_t GRAPH_SIZE;
        InitContext _ctx; // weight context
        std::string model_gpu_layers;
        const int n_threads;
    public:
        Config vis_config;
    };
}
