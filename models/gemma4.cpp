#include "gemma.h"
#include <cstring>
#include "../src/audio_process.h"

namespace chatllm::gemma
{
    const int media_type_image = 0;
    const int media_type_audio = 1;

    class MultimodalEmbedder : public Block
    {
    public:
        MultimodalEmbedder(InitContext *ctx, int multimodal_hidden_size, int text_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *mm_outputs) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int       multimodal_hidden_size;
        const int       text_hidden_size;
        Linear          embedding_projection;
        RMSNormNoScale  embedding_pre_projection_norm;
    };

    MultimodalEmbedder::MultimodalEmbedder(InitContext *ctx, int multimodal_hidden_size, int text_hidden_size):
        multimodal_hidden_size(multimodal_hidden_size),
        text_hidden_size(text_hidden_size),
        embedding_projection(ctx, multimodal_hidden_size, text_hidden_size, false),
        embedding_pre_projection_norm(ctx, multimodal_hidden_size)
    {}

    ggml::tensor *MultimodalEmbedder::forward(ComputeContext *ctx, ggml::tensor *mm_outputs)
    {
        auto output = embedding_pre_projection_norm.forward(ctx, mm_outputs);
             output = embedding_projection.forward(ctx, output);
        return output;
    }

    int64_t MultimodalEmbedder::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embedding_projection.get_param_num(effective_only);
        r += embedding_pre_projection_norm.get_param_num(effective_only);
        return r;
    }

    void MultimodalEmbedder::load(const std::string &path, TensorLoader *loader)
    {
        embedding_pre_projection_norm.load(path + "embedding_pre_projection_norm.", loader);
                 embedding_projection.load(path + "embedding_projection.", loader);
    }
}

namespace chatllm::gemma::aud
{
    const int SUBSAMPLING_CH_NUM    = 2;

    struct Config
    {
        ggml::type dtype;
        int attention_chunk_size;
        int attention_context_left;
        int attention_context_right;
        int chunk_size_feed_forward;
        int conv_kernel_size;
        int hidden_size;
        int num_attention_heads;
        int num_hidden_layers;
        int output_proj_dims;
        int subsampling_conv_channels[SUBSAMPLING_CH_NUM];
        bool use_clipped_linears;

        float attention_invalid_logits_value;
        float attention_logit_cap;
        float gradient_clipping;
        float residual_weight;
        float rms_norm_eps;

        // features extractor params
        int feature_size;
        int fft_length;
        int frame_length;
        int hop_length;
        int sampling_rate;
        float max_frequency;
        float mel_floor;
        float min_frequency;
    };

    class AudioSubSampleConvProjectionLayer : public Block
    {
    public:
        AudioSubSampleConvProjectionLayer(InitContext *ctx, int in_channels, int out_channels, float norm_eps = 1e-6f):
            conv(ctx, in_channels, out_channels, 3, 2, 1, 1, 1, false),
            norm(ctx, out_channels, false, norm_eps)
        {
        }

        std::pair<ggml::tensor *, ggml::tensor *> forward2(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *mask);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += conv.get_param_num(effective_only);
            r += norm.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            conv.load(path + "conv.", loader);
            norm.load(path + "norm.", loader);
        }
    public:
        Conv2D      conv;
        LayerNorm   norm;
    };

    std::pair<ggml::tensor *, ggml::tensor *> AudioSubSampleConvProjectionLayer::forward2(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *mask)
    {
        if (mask)
        {
            hidden_states = ggml::mul(ctx, hidden_states, ggml::reshape(ctx, mask, 1, ggml::get_dim(mask, 0), ggml::get_dim(mask, 1)));
            mask          = ggml::reshape(ctx, mask, 2, ggml::get_dim(mask, 0) / 2, ggml::get_dim(mask, 1));
            mask          = ggml::view_3d(ctx, mask, 1, ggml::get_dim(mask, 1), ggml::get_dim(mask, 2),
                                          ggml::element_size(mask) * 2,
                                          ggml::element_size(mask) * 2 * ggml::get_dim(mask, 1),
                                          0);
            mask          = ggml::cont(ctx, mask);
            mask          = ggml::reshape(ctx, mask, ggml::get_dim(mask, 1), ggml::get_dim(mask, 2));
        }
        hidden_states = conv.forward (ctx, hidden_states);
        hidden_states = ggml::permute(ctx, hidden_states, 1, 2, 0, 3);
        hidden_states = norm.forward (ctx, hidden_states);
        hidden_states = ggml::permute(ctx, hidden_states, 2, 0, 1, 3);
        hidden_states = ggml::cont   (ctx, hidden_states);
        hidden_states = ggml::act    (ctx, ActFunc::RELU, hidden_states);
        return std::pair<ggml::tensor *, ggml::tensor *>(hidden_states, mask);
    }

    class AudioSubSampleConvProjection : public Block
    {
    public:
        AudioSubSampleConvProjection(InitContext *ctx, int hidden_size, const int *subsampling_conv_channels, float rms_norm_eps);
        std::pair<ggml::tensor *, ggml::tensor *> forward2(ComputeContext *ctx, ggml::tensor *input_features, ggml::tensor *input_features_mask);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_proj_linear.get_param_num(effective_only);
            r += layer0.get_param_num(effective_only);
            r += layer1.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            input_proj_linear.load(path + "input_proj_linear.", loader);
            layer0.load(path + "layer0.", loader);
            layer1.load(path + "layer1.", loader);
        }
    public:
        AudioSubSampleConvProjectionLayer   layer0;
        AudioSubSampleConvProjectionLayer   layer1;
        Linear                              input_proj_linear;
    };

    AudioSubSampleConvProjection::AudioSubSampleConvProjection(InitContext *ctx, int hidden_size, const int *subsampling_conv_channels, float rms_norm_eps):
        layer0(ctx, 1, subsampling_conv_channels[0], rms_norm_eps),
        layer1(ctx, subsampling_conv_channels[0], subsampling_conv_channels[1], rms_norm_eps),
        input_proj_linear(ctx, (subsampling_conv_channels[0] / 4) * subsampling_conv_channels[1], hidden_size, false)
    {
    }

    static int64_t calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
    }

    static int64_t calc_projected_length(int64_t mel_n_len)
    {
        const int kernel_size = 3;
        const int stride = 2;
        const int padding = 1;
        const int dilation = 1;
        const int64_t len1 = calc_conv_output_size(mel_n_len, kernel_size, stride, padding, dilation);
        const int64_t len2 = calc_conv_output_size(len1,      kernel_size, stride, padding, dilation);
        return len2;
    }

    std::pair<ggml::tensor *, ggml::tensor *> AudioSubSampleConvProjection::forward2(ComputeContext *ctx, ggml::tensor *input_features, ggml::tensor *input_features_mask)
    {
        auto r = layer0.forward2(ctx, input_features, input_features_mask);
             r = layer1.forward2(ctx, r.first,        r.second);
        auto hidden_states = r.first;
        const int seq_len  = ggml::get_dim(hidden_states, 1);
        const int batch    = ggml::get_dim(hidden_states, 3);
        hidden_states      = ggml::permute(ctx, hidden_states, 1, 2, 0, 3);
        hidden_states      = ggml::cont   (ctx, hidden_states);
        hidden_states      = ggml::reshape(ctx, hidden_states, ggml::nelements(hidden_states) / seq_len / batch, seq_len, batch);
        hidden_states      = input_proj_linear.forward(ctx, hidden_states);
        r.first = hidden_states;
        return r;
    }

    class AudioFeedForward : public Block
    {
    public:
        AudioFeedForward(InitContext *ctx, int hidden_size, float gradient_clipping, float residual_weight);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += ffw_layer_1.get_param_num(effective_only);
            r += ffw_layer_2.get_param_num(effective_only);
            r += pre_layer_norm.get_param_num(effective_only);
            r += post_layer_norm.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            ffw_layer_1.load(path + "ffw_layer_1.", loader);
            ffw_layer_2.load(path + "ffw_layer_2.", loader);
            pre_layer_norm.load(path + "pre_layer_norm.", loader);
            post_layer_norm.load(path + "post_layer_norm.", loader);
        }
    public:
        const float gradient_clipping;
        const float post_layer_scale;
        Linear   ffw_layer_1;
        Linear   ffw_layer_2;
        RMSNorm  pre_layer_norm;
        RMSNorm  post_layer_norm;
    };

    AudioFeedForward::AudioFeedForward(InitContext *ctx, int hidden_size, float gradient_clipping, float residual_weight):
        gradient_clipping(gradient_clipping), post_layer_scale(residual_weight),
        ffw_layer_1(ctx, hidden_size, hidden_size * 4, false),
        ffw_layer_2(ctx, hidden_size * 4, hidden_size, false),
        pre_layer_norm(ctx, hidden_size),
        post_layer_norm(ctx, hidden_size)
    {
        ffw_layer_1.enable_clipping();
        ffw_layer_2.enable_clipping();
    }

    ggml::tensor *AudioFeedForward::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        auto residual = hidden_states;
        hidden_states = ggml::clamp(ctx, hidden_states, -gradient_clipping, gradient_clipping);
        hidden_states = pre_layer_norm.forward(ctx, hidden_states);

        hidden_states = ffw_layer_1.forward(ctx, hidden_states);
        hidden_states = ggml::act(ctx, ActFunc::SILU, hidden_states);
        hidden_states = ffw_layer_2.forward(ctx, hidden_states);

        hidden_states = ggml::clamp(ctx, hidden_states, -gradient_clipping, gradient_clipping);
        hidden_states = post_layer_norm.forward(ctx, hidden_states);
        hidden_states = ggml::scale(ctx, hidden_states, post_layer_scale);
        hidden_states = ggml::add  (ctx, hidden_states, residual);

        return hidden_states;
    }

    class AudioCausalConv1D : public Conv1D
    {
    public:
        AudioCausalConv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
               int dilation = 1, int groups = 1, bool bias = true);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    private:
        const int left_pad;
    };

    AudioCausalConv1D::AudioCausalConv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding,
        int dilation, int groups, bool bias) :
        Conv1D(ctx, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
        left_pad((kernel_size - 1) * dilation + 1 - stride)
    {}

    ggml::tensor *AudioCausalConv1D::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        input = ggml::pad(ctx, input, left_pad, 0);
        input = Conv1D::forward(ctx, input);
        return input;
    }

    class AudioLightConv1D : public Block
    {
    public:
        AudioLightConv1D(InitContext *ctx, int hidden_size, int conv_kernel_size, ActFunc act, float rms_norm_rps, float gradient_clipping);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;

            r += conv_norm.get_param_num(effective_only);
            r += linear_end.get_param_num(effective_only);
            r += linear_start.get_param_num(effective_only);
            r += pre_layer_norm.get_param_num(effective_only);
            r += depthwise_conv1d.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            depthwise_conv1d.load(path + "depthwise_conv1d.", loader);
            pre_layer_norm.load(path + "pre_layer_norm.", loader);
            linear_start.load(path + "linear_start.", loader);
            linear_end.load(path + "linear_end.", loader);
            conv_norm.load(path + "conv_norm.", loader);
        }
    public:
        const ActFunc act;
        const float gradient_clipping;
        Linear              linear_start;
        Linear              linear_end;
        AudioCausalConv1D   depthwise_conv1d;
        RMSNorm             pre_layer_norm;
        RMSNorm             conv_norm;
    };

    AudioLightConv1D::AudioLightConv1D(InitContext *ctx, int hidden_size, int conv_kernel_size, ActFunc act, float rms_norm_rps, float gradient_clipping):
        act(act),
        gradient_clipping(gradient_clipping),
        linear_start(ctx, hidden_size, hidden_size * 2, false),
        linear_end(ctx, hidden_size, hidden_size, false),
        depthwise_conv1d(ctx, hidden_size, hidden_size, conv_kernel_size, 1, 0, 1, hidden_size, false),
        pre_layer_norm(ctx, hidden_size),
        conv_norm(ctx, hidden_size)
    {
        pre_layer_norm.eps = rms_norm_rps;
        conv_norm.eps = rms_norm_rps;
        linear_start.enable_clipping();
        linear_end.enable_clipping();
    }

    ggml::tensor *AudioLightConv1D::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        auto residual = hidden_states;

        hidden_states = pre_layer_norm.forward(ctx, hidden_states);
        hidden_states = linear_start.forward(ctx, hidden_states);
        hidden_states = ggml::glu(ctx, hidden_states, ActFunc::SIGMOID);

        hidden_states = ggml::transpose(ctx, hidden_states);
        hidden_states = depthwise_conv1d.forward(ctx, hidden_states);
        hidden_states = ggml::transpose(ctx, hidden_states);

        hidden_states = ggml::clamp(ctx, hidden_states, -gradient_clipping, gradient_clipping);
        hidden_states = conv_norm.forward(ctx, hidden_states);

        hidden_states = ggml::act(ctx, act, hidden_states);
        hidden_states = linear_end.forward(ctx, hidden_states);
        hidden_states = ggml::add(ctx, hidden_states, residual);

        return hidden_states;
    }

    class AudioAttention : public Block
    {
    public:
        AudioAttention(InitContext *ctx, const Config &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *position_embeddings, ggml::tensor *attention_mask);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;

            r += q_proj.get_param_num(effective_only);
            r += k_proj.get_param_num(effective_only);
            r += v_proj.get_param_num(effective_only);
            r += post.get_param_num(effective_only);
            r += relative_k_proj.get_param_num(effective_only);
            r += ggml::nelements(per_dim_scale);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            relative_k_proj.load(path + "relative_k_proj.", loader);
            q_proj.load(path + "q_proj.", loader);
            k_proj.load(path + "k_proj.", loader);
            v_proj.load(path + "v_proj.", loader);
            post.load(path + "post.", loader);
            loader->read_tensor(path + "per_dim_scale", per_dim_scale);
            loader->map_tensor_element(per_dim_scale, [this](float v) -> float {
                float t = chatllm::softplus(v);
                return t * q_scale;
            });
        }
    protected:
        ggml::tensor *_convert_to_block(ComputeContext *ctx, ggml::tensor *hidden_states);
        ggml::tensor *_extract_block_context(ComputeContext *ctx, ggml::tensor *hidden_states);
        ggml::tensor *_rel_shift(ComputeContext *ctx, ggml::tensor *hidden_states);
    public:
        const float q_scale;
        const float k_scale;
        const int chunk_size;
        const int max_past_horizon;
        const int max_future_horizon;
        const int num_heads;
        const int head_dim;
        Linear q_proj;
        Linear k_proj;
        Linear v_proj;
        Linear post;
        Linear relative_k_proj;
        ggml::tensor *per_dim_scale;
        v2::TanhScaling scaling;
    };

    AudioAttention::AudioAttention(InitContext *ctx, const Config &config):
        q_scale(1.0f / sqrtf((float)config.hidden_size) / logf(2)),
        k_scale(logf(1.0f + expf(1.0)) / logf(2)),
        chunk_size(config.attention_chunk_size),
        max_past_horizon(config.attention_context_left - 1),
        max_future_horizon(config.attention_context_right),
        num_heads(config.num_attention_heads),
        head_dim(config.hidden_size / config.num_attention_heads),
        q_proj(ctx, config.hidden_size, num_heads * head_dim, false),
        k_proj(ctx, config.hidden_size, num_heads * head_dim, false),
        v_proj(ctx, config.hidden_size, num_heads * head_dim, false),
          post(ctx, config.hidden_size, config.hidden_size, false),
        relative_k_proj(ctx, config.hidden_size, num_heads * head_dim, false),
        per_dim_scale(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, head_dim)),
        scaling(config.attention_logit_cap)
    {
        q_proj.enable_clipping();
        k_proj.enable_clipping();
        v_proj.enable_clipping();
          post.enable_clipping();
    }

    ggml::tensor *AudioAttention::_convert_to_block(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        const int seq_len    = ggml::get_dim(hidden_states, 2);
        const int num_blocks = (seq_len + chunk_size - 1) / chunk_size;
        const int pad = num_blocks * chunk_size - seq_len;
        if (0 == pad) return hidden_states;
        hidden_states = ggml::pad(ctx, hidden_states, 0, 0, 0, 0, 0, pad);
        hidden_states = ggml::reshape(ctx, hidden_states, ggml::get_dim(hidden_states, 0), ggml::get_dim(hidden_states, 1),
            chunk_size, num_blocks);
        return hidden_states;
    }

    ggml::tensor *AudioAttention::_extract_block_context(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        const int context_size = chunk_size + max_past_horizon + max_future_horizon;
        hidden_states = ggml::pad(ctx, hidden_states, 0, 0, 0, 0, max_past_horizon, max_future_horizon + chunk_size - 1);
        const int num_blocks = (ggml::get_dim(hidden_states, 2) - context_size) / chunk_size + 1;
        ggml::tensor *blocks = nullptr;
        for (int i = 0; i < num_blocks; i++)
        {
            const int start = i * chunk_size;
            auto part = ggml::view_4d(ctx, hidden_states, ggml::get_dim(hidden_states, 0), ggml::get_dim(hidden_states, 1),
                                      context_size, ggml::get_dim(hidden_states, 3),
                                      ggml::row_size(hidden_states),
                                      ggml::row_size(hidden_states) * ggml::get_dim(hidden_states, 1),
                                      ggml::row_size(hidden_states) * ggml::get_dim(hidden_states, 1) * ggml::get_dim(hidden_states, 2),
                                      ggml::row_size(hidden_states) * ggml::get_dim(hidden_states, 1) * start);
            blocks = blocks ? ggml::concat(ctx, blocks, part, 3) : part;
        }
        //blocks = ggml::permute(ctx, blocks, 2, 1, 0); // torch.movedim(hidden_states, -1, 2)
        return blocks;
    }

    ggml::tensor *AudioAttention::_rel_shift(ComputeContext *ctx, ggml::tensor *x)
    {
        const int context_size = chunk_size + max_past_horizon + max_future_horizon;

        const int position_length   = ggml::get_dim(x, 0);
        const int block_size        = ggml::get_dim(x, 1);
        const int num_blocks        = ggml::get_dim(x, 2);
        const int num_heads         = ggml::get_dim(x, 3);

        x = ggml::pad(ctx, x, 0, context_size + 1 - position_length);
        x = ggml::reshape(ctx, x, block_size * (context_size + 1), num_blocks, num_heads);
        x = ggml::view_3d(ctx, x, block_size * context_size, num_blocks, num_heads,
                        ggml::row_size(x),
                        ggml::row_size(x) * num_blocks,
                        0);
        x = ggml::reshape(ctx, x, context_size, block_size, num_blocks, num_heads);
        return x;
    }

    ggml::tensor *AudioAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *position_embeddings, ggml::tensor *attention_mask)
    {
        const int batch_size = ggml::get_dim(hidden_states, 2);
        const int seq_len    = ggml::get_dim(hidden_states, 1);

        CHATLLM_CHECK(batch_size == 1) << "we can't support 5 dims!";

        auto query_states = q_proj.forward(ctx, hidden_states);
        auto key_states   = k_proj.forward(ctx, hidden_states);
        auto value_states = v_proj.forward(ctx, hidden_states);

        query_states = ggml::reshape(ctx, query_states, head_dim, num_heads, seq_len, batch_size);
        key_states   = ggml::reshape(ctx, key_states,   head_dim, num_heads, seq_len, batch_size);
        value_states = ggml::reshape(ctx, value_states, head_dim, num_heads, seq_len, batch_size);

        key_states   = ggml::scale(ctx, key_states,   k_scale);
        query_states = ggml::mul  (ctx, query_states, per_dim_scale);

        query_states = _convert_to_block(ctx, query_states);
        key_states   = _extract_block_context(ctx, key_states);
        value_states = _extract_block_context(ctx, value_states);

        const int num_blocks = ggml::get_dim(query_states, 3);

        auto relative_key_states = relative_k_proj.forward(ctx, position_embeddings);
        relative_key_states = ggml::reshape(ctx, relative_key_states, head_dim, num_heads, ggml::get_dim(relative_key_states, 1));

        auto queries = ggml::permute(ctx, query_states, 0, 3, 1, 2);
        key_states   = ggml::permute(ctx, key_states,   0, 3, 1, 2);
        auto mat_ac  = ggml::mul_mat(ctx, key_states, queries);

        auto queries_flat   = ggml::reshape(ctx, queries, head_dim, -1, num_heads);
        relative_key_states = ggml::permute(ctx, relative_key_states, 0, 2, 1);
        auto mat_bd         = ggml::mul_mat(ctx, relative_key_states, queries_flat);
        mat_bd              = ggml::reshape(ctx, mat_bd, -1, chunk_size, num_blocks, num_heads);
        mat_bd              = _rel_shift(ctx, mat_bd);

        auto attn_weights   = ggml::add(ctx, mat_ac, mat_bd);
        attn_weights        = scaling.forward(ctx, attn_weights);
        attn_weights        = ggml::soft_max_ext(ctx, attn_weights, attention_mask);

        value_states        = ggml::permute (ctx, value_states, 1, 3, 0, 2);
        value_states        = ggml::cont    (ctx, value_states);
        auto attn_output    = ggml::mul_mat (ctx, value_states, attn_weights);
        attn_output         = ggml::permute (ctx, attn_output, 0, 2, 3, 1);
        attn_output         = ggml::cont    (ctx, attn_output);
        attn_output         = ggml::reshape (ctx, attn_output, -1, num_blocks * chunk_size);
        attn_output         = ggml::view_2d (ctx, attn_output, ggml::get_dim(attn_output, 0), seq_len,
                                                  ggml::row_size(attn_output), 0);

        attn_output         = post.forward(ctx, attn_output);

        return attn_output;
    }

    class AudioLayer : public Block
    {
    public:
        AudioLayer(InitContext *ctx, const Config &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *position_embeddings, ggml::tensor *attention_mask);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;

            r += feed_forward1.get_param_num(effective_only);
            r += feed_forward2.get_param_num(effective_only);
            r += self_attn.get_param_num(effective_only);
            r += lconv1d.get_param_num(effective_only);
            r += norm_pre_attn.get_param_num(effective_only);
            r += norm_post_attn.get_param_num(effective_only);
            r += norm_out.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            norm_post_attn.load(path + "norm_post_attn.", loader);
             feed_forward1.load(path + "feed_forward1.", loader);
             feed_forward2.load(path + "feed_forward2.", loader);
             norm_pre_attn.load(path + "norm_pre_attn.", loader);
                 self_attn.load(path + "self_attn.", loader);
                  norm_out.load(path + "norm_out.", loader);
                   lconv1d.load(path + "lconv1d.", loader);
        }
    public:
        const float             gradient_clipping;
        AudioFeedForward        feed_forward1;
        AudioFeedForward        feed_forward2;
        AudioAttention          self_attn;
        AudioLightConv1D        lconv1d;
        RMSNorm                 norm_pre_attn;
        RMSNorm                 norm_post_attn;
        RMSNorm                 norm_out;
    };

    AudioLayer::AudioLayer(InitContext *ctx, const Config &config):
        gradient_clipping(config.gradient_clipping),
        feed_forward1(ctx, config.hidden_size, config.gradient_clipping, config.residual_weight),
        feed_forward2(ctx, config.hidden_size, config.gradient_clipping, config.residual_weight),
        self_attn(ctx, config),
        lconv1d(ctx, config.hidden_size, config.conv_kernel_size, ActFunc::SILU, config.rms_norm_eps, config.gradient_clipping),
        norm_pre_attn(ctx, config.hidden_size),
        norm_post_attn(ctx, config.hidden_size),
        norm_out(ctx, config.hidden_size)

    {
    }

    ggml::tensor *AudioLayer::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *position_embeddings, ggml::tensor *attention_mask)
    {
        hidden_states = feed_forward1.forward(ctx, hidden_states);
        auto residual = hidden_states;

        hidden_states = ggml::clamp(ctx, hidden_states, -gradient_clipping, gradient_clipping);
        hidden_states = norm_pre_attn.forward(ctx, hidden_states);

        hidden_states = self_attn.forward(ctx, hidden_states, position_embeddings, attention_mask);

        hidden_states = ggml::clamp(ctx, hidden_states, -gradient_clipping, gradient_clipping);
        hidden_states = norm_post_attn.forward(ctx, hidden_states);

        hidden_states = ggml::add(ctx, hidden_states, residual);

        hidden_states = lconv1d.forward(ctx, hidden_states);
        hidden_states = feed_forward2.forward(ctx, hidden_states);

        hidden_states = ggml::clamp(ctx, hidden_states, -gradient_clipping, gradient_clipping);
        hidden_states = norm_out.forward(ctx, hidden_states);

        return hidden_states;
    }

    class AudioRelPositionalEncoding : public Block
    {
    public:
        AudioRelPositionalEncoding(InitContext *ctx, int hidden_size, int context_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;
    public:
        ggml::tensor *pos_embed;
    };

    class AudioModel : public DynamicBlock
    {
    public:
        AudioModel(InitContext *ctx, const Config &config, int llm_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_features);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        void before_eval(ComputeContext *ctx) override;
    public:
        const Config config;
        AudioSubSampleConvProjection    subsample_conv_projection;
        ggml::tensor                   *pos_emb;
        std::vector<AudioLayer>         layers;
        Linear                          out_proj;
        MultimodalEmbedder              embed_audio;
    protected:
        ggml::tensor                   *rt_attention_mask = nullptr;
        ggml::tensor                   *_convert_4d_mask_to_blocked_5d(ComputeContext *ctx, ggml::tensor *mask_4d);
    };

    AudioModel::AudioModel(InitContext *ctx, const Config &config, int llm_hidden_size):
        config(config),
        subsample_conv_projection(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config.hidden_size, config.subsampling_conv_channels, config.rms_norm_eps),
        pos_emb(ggml::new_tensor_2d(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), ggml::type_fallback(ctx->dtype, config.hidden_size), config.hidden_size, config.attention_context_left + config.attention_context_right)),
        out_proj(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.output_proj_dims),
        embed_audio(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.output_proj_dims, llm_hidden_size)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            ctx->move_to_layer(i);
            layers.emplace_back(ctx, config);
            layers[i].set_id(i);
        }
    }

    // active value: 1.0
    void fill_sliding_window_mask(ggml::tensor *mask, int window_left, int window_right = 0)
    {
        CHATLLM_CHECK(ggml::n_dims(mask) <= 2);
        const int seq_len = (int)ggml::get_dim(mask, 0);
        const int   q_len = (int)ggml::get_dim(mask, 1);
        const int offset  = seq_len - q_len;
        CHATLLM_CHECK(seq_len >= q_len);

        std::vector<float> data;
        data.resize(seq_len * q_len, 0.0f);

        for (int j = 0; j < q_len; j++)
        {
            const int n = j + offset;
            const int left  = n - window_left;
            const int right = n + window_right;
            for (int i = 0; i < seq_len; i++)
            {
                bool in_range = (left < i) && (i <= right);
                if (in_range) data[j * seq_len + i] = 1.0f;
            }
        }

        std::vector<uint8_t> buf;
        buf.resize(ggml::nbytes(mask));

        ggml::from_float(ggml::type_of(mask), data.data(), buf.data(), seq_len, ggml::nrows(mask));
        Backend::write_tensor_data(mask, buf.data());
    }

    void AudioModel::before_eval(ComputeContext *ctx)
    {
        fill_sliding_window_mask(rt_attention_mask, config.attention_context_left - 1, 0);
    }

    // Convert a standard 4D attention mask `[batch_size, 1, seq_len, seq_len]` to the 5D blocked format
    //    `[batch_size, 1, num_blocks, chunk_size, context_size]`
    ggml::tensor *AudioModel::_convert_4d_mask_to_blocked_5d(ComputeContext *ctx, ggml::tensor *mask_4d)
    {
        const int batch_size = (int)ggml::get_dim(mask_4d, 3);
        const int seq_len    = (int)ggml::get_dim(mask_4d, 1);

        CHATLLM_CHECK(batch_size == 1);

        const int chunk_size            = config.attention_chunk_size;
        const int max_past_horizon      = config.attention_context_left - 1;
        const int max_future_horizon    = config.attention_context_right;
        const int context_size          = chunk_size + max_past_horizon + max_future_horizon;

        const int num_blocks        = (seq_len + chunk_size - 1) / chunk_size;
        const int padded_seq_len    = num_blocks * chunk_size;
        const int pad_amount        = padded_seq_len - seq_len;

        mask_4d         = ggml::pad    (ctx, mask_4d, 0, pad_amount, 0, pad_amount, 0);
        auto mask_5d    = ggml::reshape(ctx, mask_4d, padded_seq_len, chunk_size, num_blocks);
        mask_5d         = ggml::pad    (ctx, mask_5d, max_past_horizon, max_future_horizon);

        ggml::tensor *r = nullptr;
        for (int i = 0, block_start = 0; i < num_blocks; i++, block_start += chunk_size)
        {
            auto part = ggml::view_3d(ctx, mask_5d, context_size, chunk_size, 1,
                                    ggml::row_size(mask_5d),
                                    ggml::row_size(mask_5d) * chunk_size,
                                    ggml::row_size(mask_5d) * chunk_size * i +
                                    ggml::element_size(mask_5d) * block_start);
            r = r ? ggml::concat(ctx, r, part, 2) : part;
        }
        return r;
    }

    ggml::tensor *AudioModel::forward(ComputeContext *ctx, ggml::tensor *input_features)
    {
        CHATLLM_CHECK(ggml::n_dims(input_features) <= 2);

        input_features = ggml::dup(ctx, input_features);

        auto r = subsample_conv_projection.forward2(ctx, input_features, nullptr);
        auto hidden_states = r.first;

        const int seq_len = ggml::get_dim(hidden_states, 1);
        rt_attention_mask = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, seq_len, seq_len);

        auto attention_mask = _convert_4d_mask_to_blocked_5d(ctx, rt_attention_mask);

        // mask: 0 -> -inf, 1 -> 0.0
        attention_mask = ggml::scale(ctx, attention_mask, 1.0f, -1.0f);
        attention_mask = ggml::scale(ctx, attention_mask, - config.attention_invalid_logits_value);

        for (int i = 0; i < (int)layers.size(); i++)
        {
            hidden_states = layers[i].forward(ctx, hidden_states, pos_emb, attention_mask);
        }

        hidden_states = out_proj.forward(ctx, hidden_states);
        hidden_states = embed_audio.forward(ctx, hidden_states);
        return hidden_states;
    }

    int64_t AudioModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += subsample_conv_projection.get_param_num(effective_only);
        r += embed_audio.get_param_num(effective_only);
        r += out_proj.get_param_num(effective_only);
        r += ggml::nelements(pos_emb);
        r += layers[0].get_param_num(effective_only) * layers.size();
        return r;
    }

    void AudioModel::load(const std::string &path, TensorLoader *loader)
    {
        subsample_conv_projection.load(path + "subsample_conv_projection.", loader);
        embed_audio.load(path, loader);
        out_proj.load(path + "output_proj.", loader);
        loader->read_tensor(path + "pos_embed.weight", pos_emb);
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layers[i].load(path + "layers." + std::to_string(i) + ".", loader);
        }
        _loaded = true;
    }

    class EmbeddingGeneration
    {
    public:
        EmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096 * 2);
        bool load(ModelLoader &loader);
        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config, const int max_projected_tokens);
        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);
        AudioModel *get_model(void) const { return aud_model.get(); }
        bool is_loaded() const;
    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        const int max_llm_tokens;
        std::unique_ptr<AudioModel> aud_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config _config;
    };

    EmbeddingGeneration::EmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "aud", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool EmbeddingGeneration::load(ModelLoader &loader)
    {
        if (aud_model.get())
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            aud_model->load("audio.", &loader);
            loader.pop_allocator_manager();
            return aud_model->is_loaded();
        }
        else
            return false;
    }

    bool EmbeddingGeneration::is_loaded() const
    {
        if (aud_model.get() == nullptr) return false;
        return aud_model->is_loaded();
    }

    bool EmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config, const int max_projected_tokens)
    {
        const auto _cfg = config["config.json"]["audio_config"];
        if (!_cfg.IsObject()) return false;

        _config.dtype = dtype;

        CHATLLM_CHECK(_cfg["model_type"].ToString() == "gemma4_audio");

        _config.attention_chunk_size        = (int)_cfg["attention_chunk_size"].ToInt();
        _config.attention_context_left      = (int)_cfg["attention_context_left"].ToInt();
        _config.attention_context_right     = (int)_cfg["attention_context_right"].ToInt();
        _config.chunk_size_feed_forward     = (int)_cfg["chunk_size_feed_forward"].ToInt();
        _config.conv_kernel_size            = (int)_cfg["conv_kernel_size"].ToInt();
        _config.hidden_size                 = (int)_cfg["hidden_size"].ToInt();
        _config.num_attention_heads         = (int)_cfg["num_attention_heads"].ToInt();
        _config.num_hidden_layers           = (int)_cfg["num_hidden_layers"].ToInt();
        _config.output_proj_dims            = (int)_cfg["output_proj_dims"].ToInt();

        for (int i = 0; i < SUBSAMPLING_CH_NUM; i++)
            _config.subsampling_conv_channels[i] = (int)_cfg["subsampling_conv_channels"][i].ToInt();

        _config.use_clipped_linears         = _cfg["use_clipped_linears"].ToBool();

        _config.attention_invalid_logits_value  = (float)_cfg["attention_invalid_logits_value"].ToFloat();
        _config.attention_logit_cap             = (float)_cfg["attention_logit_cap"].ToFloat();
        _config.gradient_clipping               = (float)_cfg["gradient_clipping"].ToFloat();
        _config.residual_weight                 = (float)_cfg["residual_weight"].ToFloat();
        _config.rms_norm_eps                    = (float)_cfg["rms_norm_eps"].ToFloat();

        const auto _pp_cfg = config["processor_config.json"]["feature_extractor"];
        if (!_pp_cfg.IsObject()) return false;

        _config.feature_size        = (int)_pp_cfg["feature_size"].ToFloat();
        _config.fft_length          = (int)_pp_cfg["fft_length"].ToFloat();
        _config.frame_length        = (int)_pp_cfg["frame_length"].ToFloat();
        _config.hop_length          = (int)_pp_cfg["hop_length"].ToFloat();
        _config.sampling_rate       = (int)_pp_cfg["sampling_rate"].ToFloat();
        _config.max_frequency       = (float)_pp_cfg["max_frequency"].ToFloat();
        _config.mel_floor           = (float)_pp_cfg["mel_floor"].ToFloat();
        _config.min_frequency       = (float)_pp_cfg["min_frequency"].ToFloat();

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 9 + 22 * _config.num_hidden_layers;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        aud_model.reset(new AudioModel(&_ctx, _config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

        return true;
    }

    void EmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if (!is_loaded() || (tok->media_emb.size() < 1)) return;

        for (auto &media : tok->media_emb)
        {
            if (media.type != media_type_audio) continue;
            run_model(gen_config, tok, dtype, media, buf);
        }
    }

    bool EmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &media, std::vector<uint8_t> &buf)
    {
        ggml::tensor *media_emb = nullptr;
        const auto make_graph = [this, &media_emb, &media](ComputeContext *ctx) -> ggml::tensor * {
            media_emb = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, media.width, _config.feature_size);
            auto emb = ggml::transpose(ctx, media_emb);
            auto r = aud_model->forward(ctx, emb);
            return r;
        };
        const auto write_input_data = [this, &media_emb, &media](ComputeContext *ctx) {
            Backend::write_tensor_data(media_emb, media.data.data(), 0, media.data.size() * sizeof(media.data[0]));
            aud_model->before_eval(ctx);
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        CHATLLM_CHECK((int)shape[1] == media.emb_vec_number);

        return true;
    }
}

namespace chatllm::gemma::vit
{
    struct Config
    {
        ggml::type dtype;
        int max_patches;
        int global_head_dim;
        int head_dim;
        int hidden_size;
        int intermediate_size;
        int num_attention_heads;
        int num_hidden_layers;
        int num_key_value_heads;
        int patch_size;
        int pooling_kernel_size;
        int position_embedding_size;

        float rope_theta;
        float image_mean[3];
        float image_std[3];

        bool standardize;
        bool use_clipped_linears;

        int video_max_soft_tokens;
        int video_max_num_frames;

        Config()
        {
            memset(this, 0, sizeof(*this));
        }
    };

    class VisionPatchEmbedder : public Block
    {
    public:
        VisionPatchEmbedder(InitContext *ctx, const Config &config):
            position_embedding_size(config.position_embedding_size),
            input_proj(ctx, 3 * config.patch_size * config.patch_size, config.hidden_size, false),
            position_embedding_table(ggml::new_tensor_3d(ctx, ggml::type_fallback(ctx->dtype, config.hidden_size), config.hidden_size, config.position_embedding_size, 2))
        {}

        void before_eval(ComputeContext *ctx) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *pixel_values, int width, int height);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_proj.get_param_num(effective_only);
            r += ggml::nelements(position_embedding_table);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            input_proj.load(path + "input_proj.", loader);
            loader->read_tensor(path + "position_embedding_table", position_embedding_table);
        }
    public:
        const int       position_embedding_size;
        Linear          input_proj;
        ggml::tensor   *position_embedding_table;
    protected:
        int width = 0;
        int height = 0;
        ggml::tensor   *rt_pos = nullptr;
    };

    void VisionPatchEmbedder::before_eval(ComputeContext *ctx)
    {
        std::vector<int> pos(width * height * 2);
        int *p_x = pos.data();
        int *p_y = pos.data() + width * height;
        int cnt = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                p_x[cnt] = j;
                p_y[cnt] = i;
                cnt += 1;
            }
        }
        Backend::write_tensor_data(rt_pos, pos.data());
    }

    ggml::tensor *VisionPatchEmbedder::forward(ComputeContext *ctx, ggml::tensor *pixel_values, int width, int height)
    {
        this->width  = width;
        this->height = height;
        rt_pos = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, height * width, 2);

        // Gemma4 applies no normalization and instead scales in model code
        pixel_values = ggml::scale(ctx, pixel_values, 2, -1);
        auto hidden_states = input_proj.forward(ctx, pixel_values);

        auto pos_emb_xy = ggml::get_rows(ctx, position_embedding_table, rt_pos);
        auto pos_emb_x  = ggml::view_2d(ctx, pos_emb_xy, ggml::get_dim(pos_emb_xy, 0), ggml::get_dim(pos_emb_xy, 1), ggml::row_size(pos_emb_xy), 0);
        auto pos_emb_y  = ggml::view_2d(ctx, pos_emb_xy, ggml::get_dim(pos_emb_xy, 0), ggml::get_dim(pos_emb_xy, 1), ggml::row_size(pos_emb_xy), ggml::row_size(pos_emb_xy) * ggml::get_dim(pos_emb_xy, 1));
        auto pos_emb    = ggml::add(ctx, pos_emb_x, pos_emb_y);

        hidden_states   = ggml::add(ctx, hidden_states, pos_emb);

        return hidden_states;
    }

    class VisionMLP : public BaseMLP
    {
    public:
        VisionMLP(InitContext *ctx, int hidden_size, int intermediate_size):
            BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU)
        {
        }
    };

    class TensorPosHelper
    {
    public:
        TensorPosHelper(int max_length)
            : max_length(max_length * 4)
        {
            v_pos.resize(this->max_length);
        }

        ggml::tensor *allocate_pos_tensor(InitContext *ctx)
        {
            ggml::tensor *r = ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length);
            ctx->get_allocator()->alloc(r);
            return r;
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen)
        {
            ggml::set_dim(pos, 0, length * 4);
            Backend::write_tensor_data(pos, v_pos.data(), 0, length * 2 * sizeof(v_pos[0]));
        }

        void prepare(int grid_h, int grid_w)
        {
            length = grid_h * grid_w;

            v_pos.clear();
            v_pos.resize(length * 2, 0);
            int *p_w = &v_pos[length * 0];
            int *p_h = &v_pos[length * 1];
            int index = 0;

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    p_w[index] = j;
                    p_h[index] = i;
                    index++;
                }
            }
        }
    public:
        const int max_length;
        int length;
        std::vector<int> v_pos;
    };

    class TensorPosHelper2D : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper2D(int max_length)
            : BaseTensorPosHelper(max_length * 4)
        {
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override
        {
            helper->prepare_pos_tensor(ctx, pos, n_past, qlen);
        }
    public:
        TensorPosHelper *helper;
    };

    class VisionAttention : public TensorPosHelperPrelude,  public QKNormedRoPEAttention<RMSNormInplace, BaseCachelessAttention>
    {
    public:
        typedef QKNormedRoPEAttention<RMSNormInplace, BaseCachelessAttention> Base;
        VisionAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
            TensorPosHelperPrelude(new TensorPosHelper2D(max_length)),
            Base(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
            head_size(head_dim)
        {
            TensorPosHelperPrelude::done();
            causal = false;
            CHATLLM_CHECK(rope_dim % 4 == 0);

            attn_scaling_factor = 1.0f;
        }

        void set_pos_helper(TensorPosHelper *helper)
        {
            TensorPosHelper2D *h = (TensorPosHelper2D *)pos_helper.get();
            h->helper = helper;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            const int hidden_size = o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            before_forward(ctx, n_past, qlen);

            ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpk = k_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpv = v_proj.forward(ctx, hidden_states);

            tmpv = ggml::reshape (ctx, tmpv, head_size,  num_kv_heads, ggml::get_dim(tmpv, 1), ggml::get_dim(tmpv, 2));
            tmpv = ggml::rms_norm(ctx, tmpv, 1e-6f);
            tmpv = ggml::reshape (ctx, tmpv, head_size * num_kv_heads, ggml::get_dim(tmpv, 2), ggml::get_dim(tmpv, 3));

            ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

            ggml::tensor *attn_output = o_proj.forward(ctx, scores);
            return attn_output;
        }
    protected:
        ggml::tensor *apply_2d_rope(ComputeContext *ctx, ggml::tensor *t, int hidden_size, int qlen) const
        {
            t = ggml::rope_2d_inplace(ctx, t, pos, freq_factors, rope_dim, n_original_ctx,
                                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            return t;
        }

        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            k = const_cast<VisionAttention *>(this)->k_layernorm.forward(ctx, k);
            k = apply_2d_rope(ctx, k, hidden_size, qlen);    // [qlen, heads, head_size]
            return k;
        }

        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            q = const_cast<VisionAttention *>(this)->q_layernorm.forward(ctx, q);
            q = apply_2d_rope(ctx, q, hidden_size, qlen);
            return q;
        }
    public:
        const int head_size;
        int grid_h = 0;
        int grid_w = 0;
    };

    typedef LMBlock4<RMSNorm, VisionAttention, RMSNorm, RMSNorm, VisionMLP, RMSNorm> VisionEncoderLayer;

    class VisionPooler : public Block
    {
    public:
        VisionPooler(InitContext *ctx, int hidden_size, int pooling_kernel_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int width, int height);
    public:
        const int   hidden_size;
        const float root_hidden_size;
        const int   pooling_kernel_size;
    };

    VisionPooler::VisionPooler(InitContext *ctx, int hidden_size, int pooling_kernel_size):
        hidden_size(hidden_size),
        root_hidden_size(sqrtf((float)hidden_size)),
        pooling_kernel_size(pooling_kernel_size)
    {
    }

    ggml::tensor *VisionPooler::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int width, int height)
    {
        CHATLLM_CHECK((width  % pooling_kernel_size) == 0);
        CHATLLM_CHECK((height % pooling_kernel_size) == 0);

        // re-arange patches
        hidden_states = ggml::reshape(ctx, hidden_states, hidden_size, width, height);
        hidden_states = ggml::permute(ctx, hidden_states, 2, 0, 1);
        auto r = ggml::avg_pool_2d(ctx, hidden_states, pooling_kernel_size, pooling_kernel_size);
        r = ggml::permute(ctx, r, 1, 2, 0);
        r = ggml::cont   (ctx, r);
        r = ggml::reshape(ctx, r, hidden_size, width / pooling_kernel_size * height / pooling_kernel_size);
        return r;
    }

    class Standardizer : public Block
    {
    public:
        Standardizer(InitContext *ctx, int hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        bool loaded;
        ggml::tensor *std_bias;
        ggml::tensor *std_scale;
    };

    Standardizer::Standardizer(InitContext *ctx, int hidden_size):
        loaded(false),
        std_bias(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F16, hidden_size)),
        std_scale(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F16, hidden_size))
    {}

    ggml::tensor *Standardizer::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        if (!loaded) return hidden_states;
        hidden_states = ggml::sub(ctx, hidden_states, std_bias);
        hidden_states = ggml::mul(ctx, hidden_states, std_scale);
        return hidden_states;
    }

    int64_t Standardizer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (!loaded) return r;
        r += ggml::nelements(std_bias);
        r += ggml::nelements(std_scale);
        return r;
    }

    void Standardizer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "std_bias")) return;
        loader->read_tensor(path + "std_bias", std_bias);
        loader->read_tensor(path + "std_scale", std_scale);
        loaded = true;
    }

    class VisionModel : public DynamicBlock
    {
    public:
        VisionModel(InitContext *ctx, const Config &config, const int text_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *pixel_values, int grid_w, int grid_h);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        VisionPatchEmbedder             patch_embedder;
        std::vector<VisionEncoderLayer> layers;
        VisionPooler                    pooler;
        MultimodalEmbedder              embed_vision;
        std::unique_ptr<TensorPosHelper> pos_helper;
        Standardizer                    standardizer;
    };

    VisionModel::VisionModel(InitContext *ctx, const Config &config, const int text_hidden_size):
        patch_embedder(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
        pooler(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.pooling_kernel_size),
        embed_vision(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, text_hidden_size),
        standardizer(ctx, config.hidden_size)
    {
        pos_helper.reset(new TensorPosHelper(config.max_patches));

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            ctx->move_to_layer(i);
            layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, config.num_key_value_heads, config.head_dim, config.position_embedding_size);
            layers[i].attention.set_pos_helper(pos_helper.get());
            layers[i].attention.freq_base = config.rope_theta;

            if (config.use_clipped_linears)
            {
                layers[i].attention.q_proj.enable_clipping();
                layers[i].attention.k_proj.enable_clipping();
                layers[i].attention.v_proj.enable_clipping();
                layers[i].attention.o_proj.enable_clipping();
                layers[i].mlp.enable_clipping();
            }
        }
    }

    ggml::tensor *VisionModel::forward(ComputeContext *ctx, ggml::tensor *pixel_values, int grid_w, int grid_h)
    {
        auto hidden_states = pixel_values;
        pos_helper->prepare(grid_h, grid_w);

        hidden_states = patch_embedder.forward(ctx, hidden_states, grid_w, grid_h);
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layers[i].attention.grid_h = grid_h;
            layers[i].attention.grid_w = grid_w;
            hidden_states = layers[i].forward(ctx, hidden_states, 0);
        }
        hidden_states = pooler.forward(ctx, hidden_states, grid_w, grid_h);
        hidden_states = standardizer.forward(ctx, hidden_states);
        hidden_states = embed_vision.forward(ctx, hidden_states);
        return hidden_states;
    }

    int64_t VisionModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (!is_loaded()) return r;
        r += patch_embedder.get_param_num(effective_only);
        r += embed_vision.get_param_num(effective_only);
        r += pooler.get_param_num(effective_only);
        r += layers[0].get_param_num(effective_only) * layers.size();
        r += standardizer.get_param_num(effective_only);
        return r;
    }

    void VisionModel::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "patch_embedder.input_proj.weight")) return;

        patch_embedder.load(path + "patch_embedder.", loader);
        embed_vision.load(path, loader);
        pooler.load(path + "pooler.", loader);
        standardizer.load(path, loader);
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layers[i].load(path + "blocks." + std::to_string(i) + '.', loader);
        }

        _loaded = true;
    }

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096);
        bool load(ModelLoader &loader);
        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config, const int max_projected_tokens);
        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);
        VisionModel *get_model(void) const { return vis_model.get(); }
        bool is_loaded() const;
    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        const int max_llm_tokens;
        std::unique_ptr<VisionModel> vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };

    VisualEmbeddingGeneration::VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "vis", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool VisualEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (vis_model.get())
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            vis_model->load("visual.", &loader);
            loader.pop_allocator_manager();
            return vis_model->is_loaded();
        }
        else
            return false;
    }

    bool VisualEmbeddingGeneration::is_loaded() const
    {
        if (vis_model.get() == nullptr) return false;
        return vis_model->is_loaded();
    }

    bool VisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config, const int max_projected_tokens)
    {
        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vis_config.dtype = dtype;

        vis_config.global_head_dim      = (int)vis_cfg["global_head_dim"].ToInt();
        vis_config.head_dim             = (int)vis_cfg["head_dim"].ToInt();
        vis_config.num_key_value_heads  = (int)vis_cfg["num_key_value_heads"].ToInt();
        vis_config.pooling_kernel_size  = (int)vis_cfg["pooling_kernel_size"].ToInt();
        vis_config.position_embedding_size  = (int)vis_cfg["position_embedding_size"].ToInt();
        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();

        vis_config.rope_theta           = (float)vis_cfg["rope_parameters"]["rope_theta"].ToFloat();
        vis_config.standardize          = vis_cfg["standardize"].ToBool();
        vis_config.use_clipped_linears  = vis_cfg["use_clipped_linears"].ToBool();

        vis_config.max_patches          = std::min(max_projected_tokens * vis_config.pooling_kernel_size * vis_config.pooling_kernel_size,
                                                   vis_config.position_embedding_size * vis_config.position_embedding_size);

        auto pp_cfg = config["processor_config.json"]["video_processor"];
        if (pp_cfg.IsObject())
        {
            auto image_mean = pp_cfg["image_mean"];
            auto image_std = pp_cfg["image_std"];
            CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
            CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

            vis_config.image_mean[0] = (float)image_mean[0].ToFloat();
            vis_config.image_mean[1] = (float)image_mean[1].ToFloat();
            vis_config.image_mean[2] = (float)image_mean[2].ToFloat();
            vis_config.image_std[0] = (float)image_std[0].ToFloat();
            vis_config.image_std[1] = (float)image_std[1].ToFloat();
            vis_config.image_std[2] = (float)image_std[2].ToFloat();

            vis_config.video_max_soft_tokens = (int)pp_cfg["max_soft_tokens"].ToInt();
            vis_config.video_max_num_frames  = (int)pp_cfg["num_frames"].ToInt();
        }
        else
            return false;

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + 14 * vis_config.num_hidden_layers + 2;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        vis_model.reset(new VisionModel(&_ctx, vis_config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

        return true;
    }

    void VisualEmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if (!is_loaded() || (tok->media_emb.size() < 1)) return;

        for (auto &image : tok->media_emb)
        {
            if (image.type != media_type_image) continue;
            run_model(gen_config, tok, dtype, image, buf);
        }
    }

    bool VisualEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
    {
        ggml::tensor *media_emb = nullptr;
        const auto make_graph = [this, &media_emb, &image](ComputeContext *ctx) -> ggml::tensor * {
            media_emb = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size * vis_config.patch_size * 3, image.grid_width * image.grid_height);
            auto r = vis_model->forward(ctx, media_emb, image.grid_width, image.grid_height);
            return r;
        };
        const auto write_input_data = [this, &media_emb, &image](ComputeContext *ctx) {
            Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            vis_model->patch_embedder.before_eval(ctx);
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }
}

namespace chatllm::gemma::v4
{
    const int MAX_LAYERS        = 128;

    struct Config : public BaseConfig
    {
        int attention_k_eq_v;
        int global_head_dim;
        int head_dim;
        int hidden_size_per_layer_input;
        int moe_intermediate_size;
        int num_experts;
        int num_global_key_value_heads;
        int num_key_value_heads;
        int num_kv_shared_layers;
        int sliding_window;
        int top_k_experts;
        int use_double_wide_mlp;
        int layer_is_swa[MAX_LAYERS];

        float final_logit_soft_capping;
        float full_attn_partial_rotary_factor;
        float full_attn_rope_theta;
        float swa_attn_rope_theta;
    };

    class ChatHistoryEncoder : public v1::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    protected:
        void append_image_piece(std::vector<int> &ids, const int w, const int h, const std::vector<uint8_t> &pixels) const;
        void append_image_piece(const ContentPiece &piece, std::vector<int> &ids) const;
        void append_audio_piece(const ContentPiece &piece, std::vector<int> &ids) const;
        void append_video_piece(const ContentPiece &piece, std::vector<int> &ids) const;
    public:
        const vit::Config *vis_config = nullptr;
        const aud::Config *aud_config = nullptr;
        bool  vit_loaded = false;
        bool  aud_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int boa_token_id = -1;
        int eoa_token_id = -1;
        int boi_token_id = -1;
        int eoi_token_id = -1;
        int boc_token_id = -1;
        int eoc_token_id = -1;
        int btc_token_id = -1;
        int etc_token_id = -1;
        int btr_token_id = -1;
        int etr_token_id = -1;
        double fps = 2.0;
    };

    Tokenizer::Tokenizer(const BaseConfig &config)
        : v1::Tokenizer(config, &_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        auto t = new tokenizer::BPEProcessor2();
        tp = t;
        size_t size = tp->Load(buffer, n_vocab);
        t->SetDecoderType(tokenizer::BPEProcessor2::DecoderType::Sequence);
        t->EnableReturnSpecialToken(true);

        int id = tp->PieceToId("<pad>");
        if (id >= 0) pad_token_id = id;
        start_of_turn_token_id  = tp->PieceToId("<|turn>");
        end_of_turn_token_id    = tp->PieceToId("<turn|>");
        nl_token_id             = tp->PieceToId("\n");

        boa_token_id            = tp->PieceToId("<|audio>");
        eoa_token_id            = tp->PieceToId("<audio|>");
        boi_token_id            = tp->PieceToId("<|image>");
        eoi_token_id            = tp->PieceToId("<image|>");
        boc_token_id            = tp->PieceToId("<|channel>");
        eoc_token_id            = tp->PieceToId("<channel|>");
        btc_token_id            = tp->PieceToId("<|tool_call>");
        etc_token_id            = tp->PieceToId("<tool_call|>");
        btr_token_id            = tp->PieceToId("<|tool_response>");
        etr_token_id            = tp->PieceToId("<tool_response|>");

        terminate_ids.insert(end_of_turn_token_id);

        return size;
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        const int b = media_type == "image" ? boi_token_id : boa_token_id;
        const int e = media_type == "image" ? eoi_token_id : eoa_token_id;
        ids.push_back(b);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(e);
    }

    class PerLayerEmbedding : public Block
    {
    public:
        PerLayerEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim, int embedding_dim_per_layer, int num_hidden_layers);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *input_embeds) override;
        ggml::tensor *get_input_for_layer(ComputeContext *ctx, int layer_id);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

    public:
        const int embedding_dim_per_layer;
        const int num_hidden_layers;
        const float per_layer_model_projection_scale;
        const float per_layer_input_scale;
        Embedding   embed_tokens_per_layer;
        Linear      per_layer_model_projection;
        RMSNorm     per_layer_projection_norm;
    protected:
        ggml::tensor *layer_id;
        ggml::tensor *per_layer_embs = nullptr;
    };

    class PerLayerEmbeddingBlock : public Block
    {
    public:
        PerLayerEmbeddingBlock(InitContext *ctx, int hidden_size, int embedding_dim_per_layer, ActFunc act = ActFunc::GELU);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds) override;

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

    public:
        ActFunc     act;
        Linear      per_layer_input_gate;
        Linear      per_layer_projection;
        RMSNorm     post_per_layer_input_norm;
    };

    PerLayerEmbedding::PerLayerEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim, int embedding_dim_per_layer, int num_hidden_layers) :
        embedding_dim_per_layer(embedding_dim_per_layer),
        num_hidden_layers(num_hidden_layers),
        per_layer_model_projection_scale(sqrtf(1.0f / embedding_dim)),
        per_layer_input_scale(sqrtf(0.5f)),
        embed_tokens_per_layer(ctx, ggml::type_fallback(ctx->dtype, embedding_dim_per_layer), num_embeddings, embedding_dim_per_layer * num_hidden_layers),
        per_layer_model_projection(ctx, embedding_dim, embedding_dim_per_layer * num_hidden_layers, false),
        per_layer_projection_norm(ctx, embedding_dim_per_layer),
        layer_id(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, num_hidden_layers))
    {
        ctx->get_allocator()->alloc(layer_id);
        std::vector<int32_t> ids;
        for (int i = 0; i < num_hidden_layers; i++) ids.push_back(i);
        Backend::write_tensor_data(layer_id, ids.data());
    }

    ggml::tensor *PerLayerEmbedding::forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *input_embeds)
    {
        per_layer_embs = embed_tokens_per_layer.forward(ctx, input_ids);
        per_layer_embs = ggml::reshape(ctx, per_layer_embs, embedding_dim_per_layer, num_hidden_layers, ggml::get_dim(per_layer_embs, 1), ggml::get_dim(per_layer_embs, 2));

        auto x = input_embeds;
        x = per_layer_model_projection.forward(ctx, x);
        x = ggml::scale  (ctx, x, per_layer_model_projection_scale);
        x = ggml::reshape(ctx, x, embedding_dim_per_layer, num_hidden_layers, ggml::get_dim(x, 1), ggml::get_dim(x, 2));
        x = per_layer_projection_norm.forward(ctx, x);

        per_layer_embs = ggml::add(ctx, per_layer_embs, x);
        per_layer_embs = ggml::scale(ctx, per_layer_embs, per_layer_input_scale);
        ggml::build_forward_expand(ctx, per_layer_embs);
        return per_layer_embs;
    }

    ggml::tensor *PerLayerEmbedding::get_input_for_layer(ComputeContext *ctx, int layer_id)
    {
        auto id = ggml::view_1d (ctx, this->layer_id, 1, layer_id * ggml::element_size(this->layer_id));
             id = ggml::repeat  (ctx, id, ggml::get_dim(id, 0), ggml::get_dim(per_layer_embs, 2));
        auto r  = ggml::get_rows(ctx, per_layer_embs, id);
             r  = ggml::reshape (ctx, r, ggml::get_dim(r, 0), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
        return r;
    }

    int64_t PerLayerEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += per_layer_model_projection.get_param_num(effective_only);
        r +=  per_layer_projection_norm.get_param_num(effective_only);
        r +=     embed_tokens_per_layer.get_param_num(effective_only);
        return r;
    }

    void PerLayerEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        per_layer_model_projection.load(path + "per_layer_model_projection.", loader);
         per_layer_projection_norm.load(path + "per_layer_projection_norm.", loader);
            embed_tokens_per_layer.load(path + "embed_tokens_per_layer.", loader);
    }

    PerLayerEmbeddingBlock::PerLayerEmbeddingBlock(InitContext *ctx, int hidden_size, int embedding_dim_per_layer, ActFunc act):
        act(act),
        per_layer_input_gate(ctx, hidden_size, embedding_dim_per_layer, false),
        per_layer_projection(ctx, embedding_dim_per_layer, hidden_size, false),
        post_per_layer_input_norm(ctx, hidden_size)
    {
    }

    ggml::tensor *PerLayerEmbeddingBlock::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds)
    {
        auto residual = hidden_states;
        hidden_states = per_layer_input_gate.forward(ctx, hidden_states);
        hidden_states = ggml::act(ctx, act, hidden_states);
        hidden_states = ggml::mul(ctx, hidden_states, layer_embeds);
        hidden_states = per_layer_projection.forward(ctx, hidden_states);
        hidden_states = post_per_layer_input_norm.forward(ctx, hidden_states);
        hidden_states = ggml::add(ctx, residual, hidden_states);
        return hidden_states;
    }

    int64_t PerLayerEmbeddingBlock::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += post_per_layer_input_norm.get_param_num(effective_only);
        r += per_layer_input_gate.get_param_num(effective_only);
        r += per_layer_projection.get_param_num(effective_only);
        return r;
    }

    void PerLayerEmbeddingBlock::load(const std::string &path, TensorLoader *loader)
    {
        post_per_layer_input_norm.load(path + "post_per_layer_input_norm.", loader);
             per_layer_input_gate.load(path + "per_layer_input_gate.", loader);
             per_layer_projection.load(path + "per_layer_projection.", loader);
    }

    template <class BaseAttn> class Gemma4Attention : public BaseAttn
    {
    public:
        using BaseAttn::BaseAttn;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = KVCacheAttention::get_param_num(effective_only);
            r += BaseAttn::q_proj.get_param_num(effective_only);
            if (nullptr == shared_attn)
            {
                r += BaseAttn::k_proj.get_param_num(effective_only);
                r += use_k_as_v ? 0 : BaseAttn::v_proj.get_param_num(effective_only);
            }
            r += BaseAttn::o_proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            KVCacheAttention::load(path, loader);
            BaseAttn::q_proj.load(path + "q_proj.", loader);
            BaseAttn::o_proj.load(path + "o_proj.", loader);
            if (nullptr == shared_attn)
            {
                BaseAttn::k_proj.load(path + "k_proj.", loader);
                if (!use_k_as_v)
                    BaseAttn::v_proj.load(path + "v_proj.", loader);
            }
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            const int hidden_size = BaseAttn::o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            BaseAttn::before_forward(ctx, n_past, qlen);

            ggml::tensor *tmpq = BaseAttn::q_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpk = shared_attn ? nullptr : BaseAttn::k_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpv = nullptr;

            if (nullptr == shared_attn)
            {
                const int head_size = BaseAttn::v_proj.out_features() / BaseAttn::num_kv_heads;
                tmpv = use_k_as_v ? tmpk : BaseAttn::v_proj.forward(ctx, hidden_states);
                tmpv = ggml::reshape (ctx, tmpv, head_size, BaseAttn::num_kv_heads, ggml::get_dim(tmpv, 1), ggml::get_dim(tmpv, 2));
                tmpv = ggml::rms_norm(ctx, tmpv, 1e-6f);
                tmpv = ggml::reshape (ctx, tmpv, head_size * BaseAttn::num_kv_heads, ggml::get_dim(tmpv, 2), ggml::get_dim(tmpv, 3));
            }

            ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

            ggml::tensor *attn_output = BaseAttn::o_proj.forward(ctx, scores);
            return attn_output;
        }

        using BaseAttn::apply_pos_embedding_k;
        using BaseAttn::apply_pos_embedding_q;

        ggml::tensor *cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k, ggml::tensor *v) override
        {
            const int head_size = hidden_size / BaseAttn::num_attention_heads;
            const int batch_size = ggml::get_dim(q, 2);

            // [qlen, heads, head_size]
            ggml::tensor * key_layer = nullptr;
            if (k)
            {
                key_layer = ggml::reshape_4d(ctx, k, head_size, BaseAttn::num_kv_heads, qlen, batch_size);
                key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, BaseAttn::pos);
            }

            // [qlen, heads, head_size]
            ggml::tensor * query_layer = ggml::reshape_4d(ctx, q, head_size, BaseAttn::num_attention_heads, qlen, batch_size);
            query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, BaseAttn::pos);

            ggml::tensor *attn_scores = BaseAttn::cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

            return attn_scores;
        }

        void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) override
        {
            if (shared_attn) return;
            BaseAttn::save_to_cache(ctx, n_past, qlen, k, v);
        }

        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            return shared_attn ? shared_attn->get_k_from_cache(ctx, hidden_size, n_past, qlen) : BaseAttn::get_k_from_cache(ctx, hidden_size, n_past, qlen);
        }

        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            return shared_attn ? shared_attn->get_v_from_cache(ctx, hidden_size, n_past, qlen) : BaseAttn::get_v_from_cache(ctx, hidden_size, n_past, qlen);
        }

    public:
        bool use_k_as_v = false;
        Gemma4Attention *shared_attn = nullptr;
    };

    class Gemma4TextRouter : public Block
    {
    public:
        Gemma4TextRouter(InitContext *ctx, int hidden_size, int num_experts, int num_expters_per_tok);
        std::pair<ggml::tensor *, ggml::tensor *> forward2(ComputeContext *ctx, ggml::tensor *hidden_states);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int num_experts;
        const int num_expters_per_tok;
        RMSNorm         norm;
        Linear          proj;
        ggml::tensor   *per_expert_scale;
    };

    class Gemma4MoE : public Block
    {
    public:
        Gemma4MoE(InitContext *ctx, int hidden_size, int moe_intermediate_size, int num_experts, int num_expters_per_tok);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int num_experts;
        const int num_expters_per_tok;
        Gemma4TextRouter router;
        MultiMLP         experts;
        RMSNorm          post_feedforward_layernorm_1;
        RMSNorm          post_feedforward_layernorm_2;
        RMSNorm          pre_feedforward_layernorm_2;
    };

    class CascadedMLP : public GELUMLP
    {
    public:
        using GELUMLP::GELUMLP;
        void init_moe(InitContext *ctx, int hidden_size, int moe_intermediate_size, int num_experts, int num_expters_per_tok);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *residual) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        std::unique_ptr<Gemma4MoE> moe;
    };

    Gemma4TextRouter::Gemma4TextRouter(InitContext *ctx, int hidden_size, int num_experts, int num_expters_per_tok):
        num_experts(num_experts), num_expters_per_tok(num_expters_per_tok),
        norm(ctx, hidden_size),
        proj(ctx, hidden_size, num_experts, false),
        per_expert_scale(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, num_experts))
    {
    }

    std::pair<ggml::tensor *, ggml::tensor *> Gemma4TextRouter::forward2(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        const int qlen = ggml::get_dim(hidden_states, 1);
        hidden_states = norm.forward(ctx, hidden_states);
        auto expert_scores = proj.forward(ctx, hidden_states);
        auto router_probabilities = ggml::soft_max(ctx, expert_scores);
        auto selected_experts = ggml::top_k(ctx, router_probabilities, num_expters_per_tok); // [qlen, num_experts_per_tok]

        ggml::tensor * weights = ggml::get_rows(ctx,
            ggml::reshape_3d(ctx, router_probabilities, 1, num_experts, qlen), selected_experts); // [1, num_experts_per_tok, qlen]

        weights = ggml::reshape_2d(ctx, weights, num_expters_per_tok, qlen);

        ggml::tensor * weights_sum = ggml::sum_rows(ctx, weights); // [1, n_tokens]

        weights = ggml::div(ctx, weights, weights_sum); // [num_experts_per_tok, n_tokens]
        weights = ggml::reshape_3d(ctx, weights, 1, num_expters_per_tok, qlen);

        auto per_expert_scale = this->per_expert_scale;
        per_expert_scale = ggml::reshape(ctx, per_expert_scale, 1, num_experts);
        per_expert_scale = ggml::repeat(ctx, per_expert_scale, 1, num_experts, ggml::get_dim(selected_experts, 1), ggml::get_dim(selected_experts, 2));
        auto routed_scaling_factor = ggml::get_rows(ctx, per_expert_scale, selected_experts);

        weights = ggml::mul(ctx, weights, routed_scaling_factor);

        auto r = std::pair<ggml::tensor *, ggml::tensor *>(weights, selected_experts);
        return r;
    }

    int64_t Gemma4TextRouter::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += norm.get_param_num(effective_only);
        r += proj.get_param_num(effective_only);
        r += ggml::nelements(per_expert_scale);
        return r;
    }

    Gemma4MoE::Gemma4MoE(InitContext *ctx, int hidden_size, int moe_intermediate_size, int num_experts, int num_expters_per_tok):
        num_experts(num_experts), num_expters_per_tok(num_expters_per_tok),
        router(ctx, hidden_size, num_experts, num_expters_per_tok),
        experts(ctx, hidden_size, moe_intermediate_size, num_experts, num_expters_per_tok, ActFunc::GELU, false),
        post_feedforward_layernorm_1(ctx, hidden_size),
        post_feedforward_layernorm_2(ctx, hidden_size),
        pre_feedforward_layernorm_2(ctx, hidden_size)
    {
    }

    ggml::tensor *Gemma4MoE::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        const int64_t hidden_size   = ggml::get_dim(hidden_states, 0);
        const int64_t qlen          = ggml::get_dim(hidden_states, 1);

        auto router = this->router.forward2(ctx, hidden_states);

        CPUMover mover(ctx, ctx->user_options.moe_on_cpu);

        auto selected_experts = router.second;
        auto weights = router.first;

        auto hidden_states_2 = pre_feedforward_layernorm_2.forward(ctx, hidden_states);

        hidden_states_2 = ggml::reshape_3d(ctx, hidden_states_2, hidden_size, 1, qlen);

        ggml::tensor * experts = this->experts.forward(ctx, hidden_states_2, selected_experts);
        experts                = ggml::mul(ctx, experts, weights);

        ggml::tensor * moe_out = nullptr;
        for (int i = 0; i < num_expters_per_tok; ++i)
        {
            ggml::tensor * cur_expert = ggml::view_2d(ctx, experts, hidden_size, qlen,
                                                    experts->nb[2], i * experts->nb[1]);

            moe_out = i == 0 ? cur_expert : ggml::add(ctx, moe_out, cur_expert);
        }

        hidden_states_2 = ggml::reshape(ctx, moe_out, hidden_size, qlen, ggml::get_dim(hidden_states, 2), ggml::get_dim(hidden_states, 3));
        hidden_states_2 = post_feedforward_layernorm_2.forward(ctx, hidden_states_2);

        return hidden_states_2;
    }

    int64_t Gemma4MoE::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += router.get_param_num(effective_only);
        r += experts.get_param_num(effective_only);
        r += post_feedforward_layernorm_1.get_param_num(effective_only);
        r += post_feedforward_layernorm_2.get_param_num(effective_only);
        r += pre_feedforward_layernorm_2.get_param_num(effective_only);
        return r;
    }

    void Gemma4MoE::load(const std::string &path, TensorLoader *loader)
    {
        router.load(path + "router.", loader);
        experts.load(path + "mlp.experts.", loader);
        post_feedforward_layernorm_1.load(path + "post_feedforward_layernorm_1.", loader);
        post_feedforward_layernorm_2.load(path + "post_feedforward_layernorm_2.", loader);
        pre_feedforward_layernorm_2.load(path + "pre_feedforward_layernorm_2.", loader);
    }

    void Gemma4TextRouter::load(const std::string &path, TensorLoader *loader)
    {
        loader->read_tensor(path + "scale", norm.weight);
        proj.load(path + "proj.", loader);
        loader->read_tensor(path + "per_expert_scale", per_expert_scale);

        const int hidden_size = ggml::get_dim(norm.weight, 0);
        const float scalar_root_size = (float)(1.0 / sqrt(hidden_size));
        loader->map_tensor_element(norm.weight, [scalar_root_size](float v) {
            return v * scalar_root_size;
        });
    }

    void CascadedMLP::init_moe(InitContext *ctx, int hidden_size, int moe_intermediate_size, int num_experts, int num_expters_per_tok)
    {
        moe.reset(new Gemma4MoE(ctx, hidden_size, moe_intermediate_size, num_experts, num_expters_per_tok));
    }

    ggml::tensor *CascadedMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *residual)
    {
        hidden_states = GELUMLP::forward(ctx, hidden_states);
        if (moe.get())
        {
            auto hidden_states_1 = moe->post_feedforward_layernorm_1.forward(ctx, hidden_states);
            auto hidden_states_flat = ggml::reshape(ctx, residual, ggml::get_dim(residual, 0), -1);
            auto hidden_states_2 = moe->forward(ctx, hidden_states_flat);
            hidden_states = ggml::add(ctx, hidden_states_1, hidden_states_2);
        }
        return hidden_states;
    }

    int64_t CascadedMLP::get_param_num(bool effective_only) const
    {
        int64_t r = GELUMLP::get_param_num(effective_only);
        if (moe.get()) r += moe->get_param_num(effective_only);
        return r;
    }

    void CascadedMLP::load(const std::string &path, TensorLoader *loader)
    {
        GELUMLP::load(path, loader);
        if (nullptr == moe.get()) return;
        moe->load(path, loader);
    }

    template <int sliding_window_len> class Gemma4SWASelfAttention : public QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>>
    {
    public:
        typedef RoPESelfAttention<Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>> BaseBase;
        typedef QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>> Base;
        Gemma4SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            Base::attn_scaling_factor = 1.0f;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += Base::get_param_num(effective_only);
            r -= Base::shared_attn != nullptr ? Base::k_layernorm.get_param_num(effective_only) : 0;
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            BaseBase::load(path, loader);
            Base::q_layernorm.load(path + "q_norm.", loader);
            if (nullptr == BaseBase::shared_attn)
                Base::k_layernorm.load(path + "k_norm.", loader);
        }
    };

    class BlockForward : public LMBlock4Forward
    {
    public:
        BlockForward(Block *pre_attention_layernorm,
                        Block *attention,
                        Block *post_attention_layernorm,
                        Block *pre_mlp_layernorm,
                        Block *mlp,
                        Block *post_mlp_layernorm,
                        Block *per_layer_emb,
                        float *layer_scalar,
                        int id):
            LMBlock4Forward(pre_attention_layernorm,
                attention,
                post_attention_layernorm,
                pre_mlp_layernorm,
                mlp,
                post_mlp_layernorm,
                id),
            layer_scalar(layer_scalar),
            per_layer_emb(per_layer_emb)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds, int n_past)
        {
            ggml::tensor *residual = hidden_states;

            hidden_states = pre_attention_layernorm->forward(ctx, hidden_states);
            hidden_states = attention->forward(ctx, hidden_states, n_past);
            hidden_states = post_attention_layernorm->forward(ctx, hidden_states);

            hidden_states = ggml::add(ctx, residual, hidden_states);
            residual = hidden_states;

            hidden_states = pre_mlp_layernorm->forward(ctx, hidden_states);
            hidden_states = mlp->forward(ctx, hidden_states, residual);
            hidden_states = post_mlp_layernorm->forward(ctx, hidden_states);

            hidden_states = ggml::add(ctx, residual, hidden_states);

            if (layer_embeds)
            {
                hidden_states = per_layer_emb->forward(ctx, hidden_states, layer_embeds);
            }
            hidden_states = ggml::scale(ctx, hidden_states, *layer_scalar);

            return hidden_states;
        }
    protected:
        float *layer_scalar;
        Block *per_layer_emb;
    };

    template <int sliding_window_len> class Gemma4DenseSWABlock : public LMBlock4<RMSNorm, Gemma4SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, CascadedMLP, RMSNorm>
    {
    public:
        typedef LMBlock4<RMSNorm, Gemma4SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, CascadedMLP, RMSNorm> Base;
        Gemma4DenseSWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds, int n_past) override
        {
            BlockForward eval(&(Base::pre_attention_layernorm),
                &(Base::attention),
                &(Base::post_attention_layernorm),
                &(Base::pre_mlp_layernorm),
                &(Base::mlp),
                &(Base::post_mlp_layernorm),
                per_layer, &layer_scalar,
                Base::get_id());
            hidden_states = eval.forward(ctx, hidden_states, layer_embeds, n_past);
            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += per_layer ? per_layer->get_param_num(effective_only) : 0;
            r += 1;
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            loader->read_scaler(path + "layer_scalar", &layer_scalar);
            if (per_layer)
                per_layer->load(path, loader);
        }

    public:
        PerLayerEmbeddingBlock *per_layer = nullptr;
        float layer_scalar = -1.0f;
    };

    class Gemma4FullSelfAttention : public QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<BaseAttention>>
    {
    public:
        typedef RoPESelfAttention<Gemma4Attention<BaseAttention>> BaseBase;
        typedef QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<BaseAttention>> Base;
        Gemma4FullSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            Base::attn_scaling_factor = 1.0f;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += Base::get_param_num(effective_only);
            r -= Base::shared_attn != nullptr ? Base::k_layernorm.get_param_num(effective_only) : 0;
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            BaseBase::load(path, loader);
            Base::q_layernorm.load(path + "q_norm.", loader);
            if (nullptr == BaseBase::shared_attn)
                Base::k_layernorm.load(path + "k_norm.", loader);
        }
    };

    class Gemma4DenseFullBlock : public LMBlock4<RMSNorm, Gemma4FullSelfAttention, RMSNorm, RMSNorm, CascadedMLP, RMSNorm>
    {
    public:
        typedef LMBlock4<RMSNorm, Gemma4FullSelfAttention, RMSNorm, RMSNorm, CascadedMLP, RMSNorm> Base;
        Gemma4DenseFullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds, int n_past) override
        {
            BlockForward eval(&(Base::pre_attention_layernorm),
                &(Base::attention),
                &(Base::post_attention_layernorm),
                &(Base::pre_mlp_layernorm),
                &(Base::mlp),
                &(Base::post_mlp_layernorm),
                per_layer, &layer_scalar,
                Base::get_id());
            hidden_states = eval.forward(ctx, hidden_states, layer_embeds, n_past);
            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += per_layer ? per_layer->get_param_num(effective_only) : 0;
            r += 1;
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            loader->read_scaler(path + "layer_scalar", &layer_scalar);
            if (per_layer)
                per_layer->load(path, loader);
        }
    public:
        PerLayerEmbeddingBlock *per_layer = nullptr;
        float layer_scalar = -1.0f;
    };

    class PerLayerBlocks
    {
    public:
        PerLayerBlocks(InitContext *ctx, const Config &config);
        int64_t get_param_num(bool effective_only) const;
    public:
        std::unique_ptr<PerLayerEmbedding>  per_layer_emb;
        std::vector<PerLayerEmbeddingBlock> per_layer_blocks;
    };

    class ModelClass : public PerLayerBlocks, public HeterogeneousModel
    {
    public:
        ModelClass(InitContext *ctx, const Config &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *per_layer_input_ids, int n_past) override;
        void load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids) override;
        int64_t get_param_num(bool effective_only) const override;
    protected:
        static Block *create_layer(InitContext *ctx, PerLayerBlocks *layer_blocks, const Config &config, int layer_index);
    public:
        const Config config;
    };

    template <class T> static void fix_proportional_rope(InitContext *ctx, const int head_dim, const int rope_dim, RoPESelfAttention<T> &attn)
    {
        const int n_dims = attn.rope_dim;
        attn.freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
        ctx->get_allocator()->alloc(attn.freq_factors);
        std::vector<float> freq_factors(n_dims / 2, 1.0f);

        for (int i = rope_dim / 2; i < n_dims / 2; i++)
        {
            freq_factors[i] = INFINITY;
        }
        Backend::write_tensor_data(attn.freq_factors, freq_factors.data());
    }

    template <int sliding_window> Gemma4DenseSWABlock<sliding_window> *create_swa_layer(InitContext *ctx, PerLayerBlocks *layer_blocks, const Config &config, int layer_index,
        const bool is_kv_shared_layer, const int intermediate_size)
    {
        auto r = new Gemma4DenseSWABlock<sliding_window>(ctx, config.hidden_size, config.num_attention_heads, intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
        r->attention.freq_base = config.swa_attn_rope_theta;

        static decltype(r->attention) *last_non_shared = nullptr;
        if (is_kv_shared_layer)
            r->attention.shared_attn = last_non_shared;
        else
            last_non_shared = &r->attention;
        if (config.hidden_size_per_layer_input > 0)
            r->per_layer = &layer_blocks->per_layer_blocks[layer_index];
        if (config.num_experts > 0)
            r->mlp.init_moe(ctx, config.hidden_size, config.moe_intermediate_size, config.num_experts, config.top_k_experts);
        return r;
    }

    Block *ModelClass::create_layer(InitContext *ctx, PerLayerBlocks *layer_blocks, const Config &config, int layer_index)
    {
        const bool is_kv_shared_layer = layer_index >= (config.num_hidden_layers - config.num_kv_shared_layers);
        BlockParams::DisableCache disable(is_kv_shared_layer);

        const int intermediate_size = is_kv_shared_layer && (config.use_double_wide_mlp != 0) ? config.intermediate_size * 2 : config.intermediate_size;

        if (config.layer_is_swa[layer_index])
        {
            switch (config.sliding_window)
            {
            case 512:
                return create_swa_layer<512>(ctx, layer_blocks, config, layer_index, is_kv_shared_layer,intermediate_size);
            case 1024:
                return create_swa_layer<1024>(ctx, layer_blocks, config, layer_index, is_kv_shared_layer, intermediate_size);
            default:
                CHATLLM_CHECK(false) << "unsupported sliding_window: " << config.sliding_window;
                break;
            }
        }
        else
        {
            const int head_dim = config.global_head_dim > 0 ? config.global_head_dim : config.head_dim;
            const int num_key_value_heads = config.attention_k_eq_v != 0 ? config.num_global_key_value_heads : config.num_key_value_heads;

            auto r = new Gemma4DenseFullBlock(ctx, config.hidden_size, config.num_attention_heads, intermediate_size,
                                        num_key_value_heads, head_dim, config.max_length);
            r->attention.freq_base = config.full_attn_rope_theta;
            r->attention.use_k_as_v = config.attention_k_eq_v != 0;

            fix_proportional_rope(ctx, head_dim, (int)(r->attention.rope_dim * config.full_attn_partial_rotary_factor), r->attention);

            static decltype(r->attention) *last_non_shared = nullptr;
            if (is_kv_shared_layer)
                r->attention.shared_attn = last_non_shared;
            else
                last_non_shared = &r->attention;
            if (config.hidden_size_per_layer_input > 0)
                r->per_layer = &layer_blocks->per_layer_blocks[layer_index];
            if (config.num_experts > 0)
                r->mlp.init_moe(ctx, config.hidden_size, config.moe_intermediate_size, config.num_experts, config.top_k_experts);
            return r;
        }
        CHATLLM_CHECK(false) << "bad";
        return nullptr;
    }

    int64_t ModelClass::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += PerLayerBlocks::get_param_num(effective_only);
        r += HeterogeneousModel::get_param_num(effective_only);
        return r;
    }

    PerLayerBlocks::PerLayerBlocks(InitContext *ctx, const Config &config)
    {
        if (config.hidden_size_per_layer_input > 0)
        {
            per_layer_emb.reset(new PerLayerEmbedding(ctx, config.vocab_size, config.hidden_size, config.hidden_size_per_layer_input, config.num_hidden_layers));
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                per_layer_blocks.emplace_back(ctx, config.hidden_size, config.hidden_size_per_layer_input);
            }
        }
    }

    int64_t PerLayerBlocks::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (per_layer_emb.get() != nullptr) r += per_layer_emb->get_param_num(effective_only);
        for (int i = 0; i < (int)per_layer_blocks.size(); i++)
            r += per_layer_blocks[i].get_param_num(effective_only);
        return r;
    }

    ModelClass::ModelClass(InitContext *ctx, const Config &config):
        PerLayerBlocks(ctx, config),
        HeterogeneousModel(ctx, config.num_hidden_layers, config.hidden_size,
            create_embedding<Embedding>(ctx, config),
            create_final_norm<RMSNorm>(ctx, config),
            nullptr,
            [this, &config](InitContext *ctx, int layer_index) -> Block * { return ModelClass::create_layer(ctx, this, config, layer_index); }),
        config(config)
    {
    }

    void ModelClass::load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids)
    {
        HeterogeneousModel::load(path, loader, layer_ids);
        if (per_layer_emb.get())
            per_layer_emb->load(path, loader);
    }

    ggml::tensor *ModelClass::forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *per_layer_input_ids, int n_past)
    {
        before_forward(ctx, input_ids, n_past);

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        ggml::tensor *hidden_states = word_embeddings->forward(ctx, input_ids);
        if (per_layer_emb.get())
        {
            per_layer_emb->forward(ctx, per_layer_input_ids, hidden_states);
        }

        for (auto &layer : layers)
        {
            ctx->move_to_layer(layer->get_id());
            if (layer_preprocess.get())
            {
                auto t = layer_preprocess->forward(this, ctx, hidden_states, layer->get_id());
                if (t) hidden_states = t;
            }

            auto layer_embeds = per_layer_emb.get() ? per_layer_emb->get_input_for_layer(ctx, layer->get_id()) : nullptr;

            hidden_states = layer->forward(ctx, hidden_states, layer_embeds, n_past);
        }

        last_hidden_state = hidden_states;

        ctx->move_to_layer(LayerAllocatorManager::Epilog);
        return final_steps->forward(this, ctx, input_ids, hidden_states);
    }

    class Prelude
    {
    public:
        Prelude(int n = 2048 * 10):
            pad_arg(new BlockParams::PadEmbedding(n, n))
        {
        }

        void done(void)
        {
            pad_arg.reset(nullptr);
        }
    protected:
        std::unique_ptr<BlockParams::PadEmbedding> pad_arg;
    };

    class ConditionalGeneration : public Prelude, public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA4);
        void load(ModelLoader &loader) override;

        void set_tokenizer(BaseTokenizer *tokenizer) override;
        bool load_more(const json::JSON &config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void before_generate(const GenerationConfig &gen_config) override;

        int64_t get_param_num(bool effective_only) const override;

    public:
        int tensor_num_of_layer(const Config &config, int layer_id) const;
        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output, const int batch_size,
                            std::function<ggml::tensor *(ComputeContext *, ggml::tensor *)> func_epilog) override;
    public:
        const Config config;
        const int max_projected_tokens;
        v2::TanhScaling logits_pp;
        vit::VisualEmbeddingGeneration  visual;
        aud::EmbeddingGeneration        audio;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        Prelude(),
        BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
        config(config),
        max_projected_tokens(BlockParams::get_padded_embedding_num()),
        logits_pp(1.0f / config.final_logit_soft_capping / sqrtf((float)config.hidden_size), config.final_logit_soft_capping),
        visual(runtime_config, pad_arg->get()),
        audio(runtime_config, pad_arg->get())
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        size_t num_tensors = 2;
        for (int i = 0; i < config.num_hidden_layers; i++) num_tensors += tensor_num_of_layer(config, i);
        if (config.hidden_size_per_layer_input)
        {
            num_tensors += 4 + 3 * config.num_hidden_layers;
        }
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config);
        get_typed_transformer<ModelClass>()->logits_pp = &logits_pp;

        batch_input = config.sliding_window - 1;
        w_ctx_.check_used_mem_size(true);

        Prelude::done();
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".pre_attention_layernorm.",   ".input_layernorm."},
            {".pre_mlp_layernorm.",         ".pre_feedforward_layernorm."},
            {".post_mlp_layernorm.",        ".post_feedforward_layernorm."},
            {".mlp.router.",                ".router."},
            {".mlp.mlp.experts.",           ".mlp.experts."},
            {".mlp.post_feedforward_layernorm_1.",                ".post_feedforward_layernorm_1."},
            {".mlp.post_feedforward_layernorm_2.",                ".post_feedforward_layernorm_2."},
            {".mlp.pre_feedforward_layernorm_2.",                 ".pre_feedforward_layernorm_2."},
        });

        BaseModelForConditionalGeneration::load(loader);

        _chat_encoder.vit_loaded  = visual.load(loader);
        _chat_encoder.aud_loaded  = audio.load(loader);
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        BaseModelForConditionalGeneration::set_tokenizer(tokenizer);
        if (visual.is_loaded())
        {
            _chat_encoder.vis_config = &visual.vis_config;
        }
        if (audio.is_loaded())
        {
            _chat_encoder.aud_config = &audio._config;
        }
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        BaseModelForConditionalGeneration::load_more(config);
        bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config, max_projected_tokens);
        if (!r) return r;
        r = audio.load_more(this->config.dtype, this->config.hidden_size, config, max_projected_tokens);
        return r;
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return;
        tok->fps    = utils::get_opt(args, "fps", tok->fps);
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = BaseModelForConditionalGeneration::get_param_num(effective_only);
        if (visual.is_loaded())
            r += visual.get_model()->get_param_num(effective_only);
        if (audio.is_loaded())
            r += audio.get_model()->get_param_num(effective_only);
        return r;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;

        CHATLLM_CHECK(tokenizer->get_image_total_emb_vectors() < max_projected_tokens) << "too many projected tokens.";

        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        audio.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);

        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    int ConditionalGeneration::tensor_num_of_layer(const Config &config, int layer_id) const
    {
        int r = 0;
        if (config.layer_is_swa[layer_id])
        {
            r = 17;
        }
        else
        {
            r = 17;
        }
        if (config.num_experts > 0)
        {
            r += 6 + 3 + 0;
        }
        return r;
    }

    bool ConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output, const int batch_size,
                            std::function<ggml::tensor *(ComputeContext *, ggml::tensor *)> func_epilog)
    {
        if (!initial_run)
        {
            initial_run = true;
            n_past = gen_config.max_length / transformer->get_reserved_batch_size() - ids_count;
            if (n_past < 0) n_past = 0;
            if (!before_initial_run(ids_count, gen_config, n_past))
                return false;
            n_past = 0;
        }

        before_run_model(input_ids, ids_count, gen_config, past);

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        std::vector<int> ids_for_layer;
        ids_for_layer.resize(ids_count * batch_size);
        for (int i = 0; i < ids_count * batch_size; i++)
        {
            ids_for_layer[i] = input_ids[i] < config.vocab_size ? input_ids[i] : config.pad_token_id;
        }

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);
        ggml::tensor *layer_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);

        ggml::tensor *r = get_typed_transformer<ModelClass>()->forward(&ctx, input_ids_tensor, layer_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (func_epilog)
        {
            r = func_epilog(&ctx, r);
        }
        else
        {
            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);
        }

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK((r->type == GGML_TYPE_F32) || (r->type == GGML_TYPE_I32)) << "output type must be float/int32: " << r->type;

        output.resize(ggml::nbytes(r) / sizeof(output[0]));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, input_ids);
        if (config.hidden_size_per_layer_input > 0)
            Backend::write_tensor_data(layer_ids_tensor, ids_for_layer.data());

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        transformer->before_eval(&ctx);
        ctx.compute();

        Backend::read_tensor_data(r, output.data());

        ctx.reset();

        return true;
    }

    void ChatHistoryEncoder::append_image_piece(std::vector<int> &ids, const int w, const int h, const std::vector<uint8_t> &pixels) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        const int patch_size = vis_config->patch_size;

        if ((w <= 0) || (h <= 0)) return;

        std::vector<float> scaled;
        vision::image_rescale(pixels, scaled);

        tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

        auto &image = tok->media_emb.back();
        image.type  = media_type_image;

        vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_PixelsLeftRightDown_ChannelsRGB);

        const int merge_length = vis_config->pooling_kernel_size * vis_config->pooling_kernel_size;
        image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

        const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
        tok->inject_media("image", ids, id_start, image.emb_vec_number);
    }

    void ChatHistoryEncoder::append_image_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {
        CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

        int w, h;
        std::vector<uint8_t> pixels;
        const int patch_size = vis_config->patch_size;

        vision::MaxPatchNum     param1(vis_config->max_patches);
        vision::MergeKernel     param2(vis_config->pooling_kernel_size, vis_config->pooling_kernel_size);

        vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
        if ((w <= 0) || (h <= 0)) return;

        append_image_piece(ids, w, h, pixels);
    }

    void ChatHistoryEncoder::append_video_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        const double fps = tok->fps;

        CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";
        auto video = new vision::VideoLoader(piece.content.c_str(), (float)fps, vis_config->video_max_num_frames);
        if (video->frames.size() < 1)
            return;

        std::ostringstream oss;

        for (size_t i = 0; i < video->frames.size(); i++)
        {
            oss.str(i > 0 ? " " : "");
            oss << utils::sec2ms(i / fps);
            tok->encode(oss.str(), ids, false, false);

            append_image_piece(ContentPiece(video->frames[i], ContentPiece::Image), ids);
        }
    }

    void ChatHistoryEncoder::append_audio_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        CHATLLM_CHECK(aud_loaded) << "Audio model not loaded";

        std::vector<float>          pcm_samples;
        std::vector<audio::mel>     mel_chunks;

        if (!audio::load(piece.content.c_str(), pcm_samples, aud_config->sampling_rate)) return;

        audio::mel_spectrogram_gemma_4(pcm_samples.data(), pcm_samples.size(),
            0,
            aud_config->sampling_rate,
            aud_config->feature_size,
            aud_config->frame_length,
            aud_config->fft_length,
            aud_config->hop_length,
            mel_chunks);

        auto &mel = mel_chunks[0];

        tok->media_emb.push_back({ .width = (int)mel.n_len, .emb_vec_number = (int)aud::calc_projected_length(mel.n_len), .data = {}});

        auto &media = tok->media_emb.back();
        media.type  = media_type_audio;
        media.data = std::move(mel.data);

        const int id_start = tok->get_image_total_emb_vectors() - media.emb_vec_number + tok->vocab_size;
        tok->inject_media("audio", ids, id_start, media.emb_vec_number);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        oss << "user\n";

        ids.push_back(tok->start_of_turn_token_id);

        auto add_and_flush = [&oss, &ids, tok](const std::string &s) {
            oss << s;
            std::string x = oss.str();
            tok->encode(x, ids, false, false);
            oss.str("");
        };

        for (auto &piece : user.pieces)
        {
            switch (piece.type)
            {
            case ContentPiece::Type::Text:
                oss << piece.content;
                break;
            case ContentPiece::Type::Image:
                add_and_flush("\n\n");
                append_image_piece(piece, ids);
                oss << "\n\n";
                break;
            case ContentPiece::Type::Audio:
                add_and_flush("\n\n");
                append_audio_piece(piece, ids);
                oss << "\n\n";
                break;
            case ContentPiece::Type::Video:
                add_and_flush("\n\n");
                append_video_piece(piece, ids);
                oss << "\n\n";
                break;
            default:
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
                break;
            }
        }

        tok->encode(oss.str(), ids, false, true);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(GEMMA4,                gemma::v4, 1);
}