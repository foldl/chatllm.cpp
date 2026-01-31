#pragma once
#include "qwen.h"

namespace chatllm::qwen::v3::audio_tower
{
    struct Config
    {
        ggml::type dtype;
        int num_hidden_layers;
        int num_attention_heads;
        int num_key_value_heads;
        int intermediate_size;
        int hidden_size;
        int max_source_positions;

        int n_window;
        int n_window_infer;
        int downsample_hidden_size;
        int conv_chunksize;

        // mel features
        int num_mel_bins;
        int chunk_length;
        int feature_size;
        int hop_length;
        int n_fft;
        int n_samples;
        int nb_max_frames;
        int sampling_rate;
    };

    class AudioSelfAttention : public BaseCachelessAttention
    {
    public:
        AudioSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length);
    protected:
        void allocate_pos_tensor(InitContext *ctx) override {}
        void prepare_pos_tensor(ComputeContext *ctx, const int n_past, const int qlen) override {}
    };

    class MultiModalProjector : public TheMLP
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size);
    };

    class AudioEmbedding : public Block
    {
    public:
        AudioEmbedding(InitContext *ctx, const Config &config);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    public:
        const int n_window;
        const int conv_chunksize;
        Conv2D    conv2d1;
        Conv2D    conv2d2;
        Conv2D    conv2d3;
        Linear    conv_out;
        ggml::tensor *positional_embedding;
    };

    class AudioTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, AudioSelfAttention, LayerNorm, TheBiasedGELUMLP> LayerBlock;

        AudioTransformer(InitContext *ctx, const Config &config, int lm_hidden_size);

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        bool is_loaded(void) const;
    public:
        int emb_vector_len = 0;
    protected:
        const int intermedia_size;
        AudioEmbedding          embed;
        LayerNorm               norm;
        MultiModalProjector     multi_modal_projector;
        std::vector<std::unique_ptr<LayerBlock>> layers;
    protected:
        bool loaded;
    };

    class AudioEmbeddingGeneration
    {
    public:
        AudioEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);

        bool load(ModelLoader &loader);

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &json_config);

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &audio, std::vector<uint8_t> &buf);
    protected:
        const int max_llm_tokens;
        std::unique_ptr<AudioTransformer> model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config config;
    };

    // Computes the output length of the convolutional layers and the output length of the audio encoder
    int get_feat_extract_output_lengths(int input_length);
    std::vector<int> get_feat_extract_output_lengths(const std::vector<int> &input_lengths);
    int pad_mel_len(int len);
}