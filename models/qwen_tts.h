#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"
#include "../src/JSON.h"
#include "qwen.h"

namespace chatllm::qwen::tts
{
    class EuclideanCodebook : public Embedding
    {
    public:
        EuclideanCodebook(InitContext *ctx, int dim, int codebook_size)
            : Embedding(ctx, codebook_size, dim)
        {}

        void load(const std::string &path, TensorLoader *loader) override;
    };

    class VectorQuantization : public Block
    {
    public:
        VectorQuantization(InitContext *ctx,
            int dim,
            int codebook_size)
            : VectorQuantization(ctx, dim, codebook_size, dim)
        {}

        VectorQuantization(InitContext *ctx,
            int dim,
            int codebook_size,
            int codebook_dim);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        EuclideanCodebook       _codebook;
        std::unique_ptr<Block>  project_out;
    };

    class ResidualVectorQuantization : public Block
    {
    public:
        ResidualVectorQuantization(InitContext *ctx,
            int num_quantizers,
            int dim,
            int codebook_size)
            : ResidualVectorQuantization(ctx, num_quantizers, dim, codebook_size, dim)
        {}

        ResidualVectorQuantization(InitContext *ctx,
            int num_quantizers,
            int dim,
            int codebook_size,
            int codebook_dim);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        Sequential layers;
    protected:
        const int num_quantizers;
    };

    class Qwen3TTSTokenizerV2DecoderConfig
    {
    public:
        Qwen3TTSTokenizerV2DecoderConfig(
            int  codebook_size=2048,
            int  hidden_size=1024,
            int  latent_dim=1024,
            int  codebook_dim=512,
            int  max_position_embeddings=8000,
            float rope_theta=10000.0f,
            int  num_attention_heads=16,
            int  num_key_value_heads=16,
            bool attention_bias=false,
            int  sliding_window=72,
            int  intermediate_size=3072,
            ActFunc hidden_act=ActFunc::SILU,
            float layer_scale_initial_scale=0.01f,
            float rms_norm_eps=1e-5f,
            int  num_hidden_layers=8,
            int  num_quantizers=16,
            const std::vector<int> &upsample_rates={8, 5, 4, 3},
            const std::vector<int> &upsampling_ratios={2, 2},
            int  decoder_dim=1536,
            int head_dim=64);
        Qwen3TTSTokenizerV2DecoderConfig(const json::JSON &config);

    public:
        const int  codebook_size;
        const int  hidden_size;
        const int  latent_dim;
        const int  codebook_dim;
        const int  max_position_embeddings;
        const float rope_theta;
        const int  num_attention_heads;
        const int  num_key_value_heads;
        const bool attention_bias;
        const int  sliding_window;
        const int  intermediate_size;
        const ActFunc hidden_act;
        const float layer_scale_initial_scale;
        const float rms_norm_eps;
        const int  num_hidden_layers;
        const int  num_quantizers;
        const std::vector<int> upsample_rates;
        const std::vector<int> upsampling_ratios;
        const int  decoder_dim;
        const int  head_dim;
        std::vector<bool>      layer_type_sliding;
    };

    struct ResidualVectorQuantizerParam
    {
        int dimension = 128;
        int input_dimension = -1;
        int output_dimension = -1;
        int n_q = 8;
        bool q_dropout = false;
        int  bins = 1024;
        bool force_projection = false;
        ResidualVectorQuantizerParam(const Qwen3TTSTokenizerV2DecoderConfig &config);
    };

    class ResidualVectorQuantizer : public Block
    {
    public:
        ResidualVectorQuantizer(InitContext *ctx,
            const ResidualVectorQuantizerParam &param);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        ResidualVectorQuantization  vq;
        std::unique_ptr<Block>      output_proj;
    };

    class SplitResidualVectorQuantizer : public Block
    {
    public:
        SplitResidualVectorQuantizer(InitContext *ctx,
            const ResidualVectorQuantizerParam &param,
            int n_q_semantic = 1);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int n_q_semantic;
        std::unique_ptr<ResidualVectorQuantizer> rvq_first;
        std::unique_ptr<ResidualVectorQuantizer> rvq_rest;
    };

    class Qwen3TTSTokenizerV2CausalConvNet : public Block
    {
    public:
        Qwen3TTSTokenizerV2CausalConvNet(InitContext *ctx,
            int in_channels,
            int out_channels,
            int kernel_size,
            int dilation = 1,
            int stride = 1,
            int groups = 1);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int kernel_size;
        const int stride;
        const int padding;
        Conv1D conv;
    };

    class Qwen3TTSTokenizerV2CausalTransConvNet : public Block
    {
    public:
        Qwen3TTSTokenizerV2CausalTransConvNet(InitContext *ctx,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride = 1);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int left_pad;
        const int right_pad;
        ConvTransposed1D conv;
    };

    class Qwen3TTSTokenizerV2ConvNeXtBlock : public Block
    {
    public:
        Qwen3TTSTokenizerV2ConvNeXtBlock(InitContext *ctx, int dim);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const ActFunc   act;
        Qwen3TTSTokenizerV2CausalConvNet dwconv;
        LayerNorm       norm;
        Linear          pwconv1;
        Linear          pwconv2;
        ggml::tensor   *gamma;
    };

    class SnakeBeta: public Block
    {
    public:
        SnakeBeta(InitContext *ctx, int dim);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const float no_div_by_zero;
        ggml::tensor *alpha;
        ggml::tensor *beta;
    };

    class Qwen3TTSTokenizerV2DecoderDecoderResidualUnit : public Block
    {
    public:
        Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(InitContext *ctx, int dim = 16, int dilation = 1);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        SnakeBeta act1;
        SnakeBeta act2;
        Qwen3TTSTokenizerV2CausalConvNet conv2;
        Qwen3TTSTokenizerV2CausalConvNet conv1;
    };

    class Qwen3TTSTokenizerV2DecoderDecoderBlock : public Block
    {
    public:
        Qwen3TTSTokenizerV2DecoderDecoderBlock(InitContext *ctx,
            int in_dim,
            int out_dim,
            int upsample_rate);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        Sequential block;
    };

    class Qwen3TTSTokenizerV2DecoderAttention : public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        typedef RoPESelfAttention<BaseCachelessAttention> Base;
    public:
        Qwen3TTSTokenizerV2DecoderAttention(InitContext *ctx, int hidden_size,
            int num_attention_heads, int num_key_value_heads, int head_dim, int max_length);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;
    public:
        ggml::tensor *self_attn_layer_scale = nullptr;
    };

    class ScaledSiLUMLP : public SiLUMLP
    {
    public:
        ScaledSiLUMLP(InitContext *ctx, int hidden_size, int intermediate_size);
        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;
        int64_t get_param_num(bool effective_only) const override;
    public:
        ggml::tensor *scale;
    };

    class Qwen3TTSTokenizerV2DecoderTransformerLayer : public
        LMBlock1<RMSNorm, Qwen3TTSTokenizerV2DecoderAttention, RMSNorm, ScaledSiLUMLP>
    {
    public:
        typedef LMBlock1<RMSNorm, Qwen3TTSTokenizerV2DecoderAttention, RMSNorm, ScaledSiLUMLP> Base;
    public:
        Qwen3TTSTokenizerV2DecoderTransformerLayer(InitContext *ctx,
            const Qwen3TTSTokenizerV2DecoderConfig &config, const int layer_idx);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const bool is_sliding;
        ggml::tensor *self_attn_layer_scale;
    };

    class Qwen3TTSTokenizerV2DecoderTransformerModel : public Block
    {
    public:
        Qwen3TTSTokenizerV2DecoderTransformerModel(InitContext *ctx,
            const Qwen3TTSTokenizerV2DecoderConfig &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int   max_length;
        const int   sliding_window;
        Linear      input_proj;
        Linear      output_proj;
        Sequential  layers;
        RMSNorm     norm;
    protected:
        ggml::tensor *mask;
    };

    class Qwen3TTSTokenizerV2Decoder : public Block
    {
    public:
        Qwen3TTSTokenizerV2Decoder(InitContext *ctx,
            const Qwen3TTSTokenizerV2DecoderConfig &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        ggml::tensor *chunked_decode(ComputeContext *ctx, ggml::tensor *codes,
            const int chunk_size=300, const int left_context_size=25);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const Qwen3TTSTokenizerV2DecoderConfig config;
        const int total_upsample;
        Qwen3TTSTokenizerV2DecoderTransformerModel      pre_transformer;
        SplitResidualVectorQuantizer                    quantizer;
        Qwen3TTSTokenizerV2CausalConvNet                pre_conv;
        Sequential                                      upsample;
        Sequential                                      decoder;
    };

    class Qwen3TTSTokenizerV2Model : public Block
    {
    public:
        Qwen3TTSTokenizerV2Model(InitContext *ctx,
            const json::JSON &config);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        Qwen3TTSTokenizerV2Decoder decoder;
    };

    class Qwen3TTSDecoderLayer : public
        LMBlock1<RMSNorm, qwen::v3::QWen3SelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        typedef LMBlock1<RMSNorm, qwen::v3::QWen3SelfAttention, RMSNorm, SiLUMLP> Base;
    public:
        Qwen3TTSDecoderLayer(InitContext *ctx,
            const int hidden_size,
            const int num_attention_heads,
            const int intermediate_size,
            const int num_key_value_heads,
            const int max_length,
            const int head_dim,
            const float rms_norm_eps = 1e-5f);
    };

    class Qwen3TTSTalkerCodePredictorModel : public Block
    {
    public:
        Qwen3TTSTalkerCodePredictorModel(InitContext *ctx, const json::JSON &config, const int embedding_dim);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *inputs_embeds, int n_past) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        Embedding *get_codec_embedding(int index);
    public:
        RMSNorm     norm;
        Sequential  layers;
        Sequential  codec_embedding;
    };

    class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration : public Block
    {
    public:
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(InitContext *ctx,
            const json::JSON &config, const int embedding_dim);
        ggml::tensor *forward_finetune(ComputeContext *ctx, ggml::tensor *inputs_embeds);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *inputs_embeds, int n_past, int generation_step);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int num_code_groups;
        const int hidden_size;
        const int vocab_size;
        Qwen3TTSTalkerCodePredictorModel    model;
        Sequential                          lm_head;
        std::unique_ptr<Block>              small_to_mtp_projection;
    };

    class Qwen3TTSTalkerResizeMLP : public TheMLP
    {
    public:
        Qwen3TTSTalkerResizeMLP(InitContext *ctx, int hidden_size, int intermediate_size, int out_size, ActFunc act, bool bias):
            TheMLP(ctx, hidden_size, intermediate_size, out_size, act, bias)
        {}
        void load(const std::string &path, TensorLoader *loader) override;
    };
}