#pragma  once
#include "../src/models.h"
#include "../src/models_priv.h"
#include "llama.h"

namespace chatllm::orpheus::snac
{
    const int MAX_VQ_STRIDES = 4;
    const int MAX_RATES = 4;

    struct Config
    {
        int sampling_rate;
        int encoder_dim;
        int encoder_rates[MAX_RATES];
        int decoder_dim;
        int decoder_rates[MAX_RATES];
        int attn_window_size;
        int codebook_size;
        int codebook_dim;
        int vq_strides[MAX_VQ_STRIDES];
        int vq_stride_count;
        bool noise;
        bool depthwise;
    };

    class Noise : public Block
    {
    public:
        Noise(InitContext *ctx, int dim);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    protected:
        Conv1D linear;
    };

    class Snake1D: public Block
    {
    public:
        Snake1D(InitContext *ctx, int dim);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        ggml::tensor *alpha;
        ggml::tensor *alpha_reciprocal;
    };

    class ResidualUnit : public Block
    {
    public:
        ResidualUnit(InitContext *ctx, int dim, int dilation = 1, int groups = 1, int kernel_size = 7);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x) override;

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        Sequential block;
    };

    class DecoderBlock : public Sequential
    {
    public:
        DecoderBlock(InitContext *ctx, int input_dim, int output_dim, int stride, bool noise, int groups, int output_padding);
        DecoderBlock(InitContext *ctx, int input_dim = 16, int output_dim = 8, int stride = 1, bool noise = false, int groups = 1, bool auto_output_padding = true);

        void load(const std::string &path, TensorLoader *loader) override;
    };

    class Decoder : public Sequential
    {
    public:
        Decoder(InitContext *ctx, int input_channel, int channels,
                int rates_count, const int *rates,
                bool noise = false, bool depthwise = false,
                int attn_window_size = 32, bool auto_output_padding = true,
                int d_out = 1);
    };

    class VectorQuantize : public Block
    {
    public:
        VectorQuantize(InitContext *ctx,
                       int input_dim, int codebook_size, int codebook_dim, int stride = 1);
        ggml::tensor *dequantize(ComputeContext *ctx, ggml::tensor *embed_id);

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int codebook_size;
        const int codebook_dim;
        const int stride;
        Conv1D in_proj;
        Conv1D out_proj;
        Embedding codebook;
    };

    class ResidualVectorQuantize : public Sequential
    {
    public:
        ResidualVectorQuantize(InitContext *ctx,
            int input_dim,
            int codebook_size,
            int codebook_dim,
            int vq_strides_count,
            const int *vq_strides);

        ggml::tensor *dequantize(ComputeContext *ctx, const std::vector<ggml::tensor *> &embed_id);

    public:
        const int n_codebooks;
    };

    class Codec
    {
    public:
        Codec(InitContext *ctx, const Config &config);

        static int64_t number_of_static_tensors(const Config &config);

        void load(const std::string &path, ModelLoader &loader);
        void decode_frame(const GenerationConfig &gen_config, BackendContext &_context, const std::vector<int> &multiframe, const int64_t count, std::vector<float> &pcm_samples);
    protected:
        bool run_model(const GenerationConfig &gen_config, BackendContext &backend_context, std::vector<float> &pcm_samples);
    protected:
        const Config config;
        int frame_size;
        Decoder decoder;
        ResidualVectorQuantize quantizer;
        std::vector<std::vector<int>> codes;
        const size_t GRAPH_SIZE;
        std::vector<int> pyramid;
    };
}

namespace chatllm::orpheus::tts
{
    typedef llama::v3_2::Config Config;

    class Tokenizer : public llama::v3_2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;
    public:
        std::string voice;
        const int custom_token_start = 128256;
        const int custom_token_end = 156938;
    };

    class ConditionalGeneration : public llama::v3_2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        void load(ModelLoader &loader) override;

        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels) override;

        void set_additional_args(const std::map<std::string, std::string> &args) override;

        bool load_more(const json::JSON &config) override;
    protected:
        void reset_decoder(void);
        void decoder_push_llm_tok_id(const GenerationConfig &gen_config, int id, std::vector<float> &pcm_samples);
    protected:
        InitContext snac_ctx;
        snac::Config codec_config;
        std::unique_ptr<snac::Codec> codec;
        std::vector<int> vocoder_ids;
    };
}