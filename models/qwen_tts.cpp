#include "qwen_tts.h"

namespace chatllm::qwen::tts
{
    void EuclideanCodebook::load(const std::string &path, TensorLoader *loader)
    {
        loader->read_tensor(path + "embedding_sum", weight);
    }

    VectorQuantization::VectorQuantization(InitContext *ctx,
        int dim,
        int codebook_size,
        int codebook_dim):
        _codebook(ctx, dim, codebook_size),
        project_out(dim != codebook_dim ? (Block *)(new Linear(ctx, dim, codebook_dim)) : new Identity())
    {
    }

    ggml::tensor *VectorQuantization::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto quantized = _codebook.forward(ctx, input);
        quantized = project_out->forward(ctx, quantized);
        quantized = ggml::permute(ctx, quantized, 1, 0, 2);
        return quantized;
    }

    int64_t VectorQuantization::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += _codebook.get_param_num(effective_only);
        r += project_out->get_param_num(effective_only);
        return r;
    }

    void VectorQuantization::load(const std::string &path, TensorLoader *loader)
    {
        _codebook.load(path + "_codebook.", loader);
        project_out->load(path + "project_out.", loader);
    }

    ResidualVectorQuantization::ResidualVectorQuantization(InitContext *ctx,
        int num_quantizers,
        int dim,
        int codebook_size,
        int codebook_dim): Block(), num_quantizers(num_quantizers)
    {
        for (int i = 0; i < num_quantizers; i++)
        {
            layers.add_block(new VectorQuantization(ctx, dim, codebook_size, codebook_dim));
        }
    }

    ggml::tensor *ResidualVectorQuantization::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *quantized = nullptr;
        CHATLLM_CHECK(num_quantizers == ggml::get_dim(input, 2));
        CHATLLM_CHECK(1              == ggml::get_dim(input, 3));

        for (int i = 0; i < num_quantizers; i++)
        {
            auto code = ggml::view_2d(ctx, input, ggml::get_dim(input, 0), ggml::get_dim(input, 1),
                                      ggml::row_size(input),
                                      ggml::row_size(input) * ggml::get_dim(input, 1) * i);
            auto r = layers.blocks[i]->forward(ctx, code);
            quantized = quantized ? ggml::add(ctx, quantized, r) : r;
        }
        return quantized;
    }

    int64_t ResidualVectorQuantization::get_param_num(bool effective_only) const
    {
        return layers.get_param_num(effective_only);
    }

    void ResidualVectorQuantization::load(const std::string &path, TensorLoader *loader)
    {
        layers.load(path + "layers.", loader);
    }

    ResidualVectorQuantizerParam::ResidualVectorQuantizerParam(const Qwen3TTSTokenizerV2DecoderConfig &config)
    {
        dimension   = config.codebook_dim / 2;
        n_q         = config.num_quantizers;
        bins        = config.codebook_size;
        input_dimension  = config.codebook_dim;
        output_dimension = config.codebook_dim;
    }

    ResidualVectorQuantizer::ResidualVectorQuantizer(InitContext *ctx,
        const ResidualVectorQuantizerParam &param):
        Block(),
        vq(ctx, param.n_q, param.dimension, param.bins),
        output_proj(param.output_dimension == param.dimension && !param.force_projection ?
            (Block *)new Identity() :
            new Conv1D(ctx, param.dimension, param.output_dimension, 1, 1, 0, 1, 1, false))
    {}

    ggml::tensor *ResidualVectorQuantizer::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto codes = ggml::permute(ctx, input, 0, 2, 1);
        auto quantized = vq.forward(ctx, codes);
        quantized = output_proj->forward(ctx, quantized);
        return quantized;
    }

    int64_t ResidualVectorQuantizer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += output_proj->get_param_num(effective_only);
        r += vq.get_param_num(effective_only);
        return r;
    }

    void ResidualVectorQuantizer::load(const std::string &path, TensorLoader *loader)
    {
        vq.load(path + "vq.", loader);
        output_proj->load(path + "output_proj.", loader);
    }

    SplitResidualVectorQuantizer::SplitResidualVectorQuantizer(InitContext *ctx,
        const ResidualVectorQuantizerParam &param,
        int n_q_semantic):
        Block(),
        n_q_semantic(n_q_semantic)
    {
        {
            ResidualVectorQuantizerParam copy = param;
            copy.n_q = n_q_semantic;
            copy.force_projection = true;
            copy.q_dropout = false;
            rvq_first.reset(new ResidualVectorQuantizer(ctx, copy));
        }
        {
            ResidualVectorQuantizerParam copy = param;
            copy.n_q = param.n_q - n_q_semantic;
            copy.force_projection = true;
            copy.q_dropout = false;
            rvq_rest.reset(new ResidualVectorQuantizer(ctx, copy));
        }
    }

    ggml::tensor *SplitResidualVectorQuantizer::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto part = ggml::view_3d(ctx, input, ggml::get_dim(input, 0), n_q_semantic, ggml::get_dim(input, 2),
                                ggml::row_size(input),
                                ggml::row_size(input) * ggml::get_dim(input, 1), 0);
        auto quantized = rvq_first->forward(ctx, part);
        if (ggml::get_dim(input, 1) > n_q_semantic)
        {
            part = ggml::view_3d(ctx, input, ggml::get_dim(input, 0), ggml::get_dim(input, 1) - n_q_semantic, ggml::get_dim(input, 2),
                                ggml::row_size(input),
                                ggml::row_size(input) * ggml::get_dim(input, 1),
                                ggml::row_size(input) * n_q_semantic);
            auto rest = rvq_rest->forward(ctx, part);
            quantized = ggml::add(ctx, quantized, rest);
        }
        return quantized;
    }

    int64_t SplitResidualVectorQuantizer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += rvq_first->get_param_num(effective_only);
        r +=  rvq_rest->get_param_num(effective_only);
        return r;
    }

    void SplitResidualVectorQuantizer::load(const std::string &path, TensorLoader *loader)
    {
        rvq_first->load(path + "rvq_first.", loader);
         rvq_rest->load(path + "rvq_rest.", loader);
    }

    Qwen3TTSTokenizerV2CausalConvNet::Qwen3TTSTokenizerV2CausalConvNet(InitContext *ctx,
        int in_channels,
        int out_channels,
        int kernel_size,
        int dilation,
        int stride,
        int groups):
        Block(),
        stride(stride),
        kernel_size((kernel_size - 1) * dilation + 1),
        padding(this->kernel_size - this->stride),
        conv(ctx, in_channels, out_channels, kernel_size, stride, 0, dilation, groups)
    {
    }

    ggml::tensor *Qwen3TTSTokenizerV2CausalConvNet::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto _get_extra_padding_for_conv1d = [this](ggml::tensor *hidden_state)
        {
            int length = ggml::get_dim(hidden_state, 0);
            double n_frames = (double)(length - kernel_size + padding) / stride + 1;
            int ideal_length = ((int)std::ceil(n_frames) - 1) * stride + (kernel_size - padding);
            return ideal_length - length;
        };

        auto extra_padding = _get_extra_padding_for_conv1d(input);
        auto hidden_state = ggml::pad(ctx, input, padding, extra_padding);

        hidden_state = conv.forward(ctx, hidden_state);
        return hidden_state;
    }

    int64_t Qwen3TTSTokenizerV2CausalConvNet::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += conv.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTokenizerV2CausalConvNet::load(const std::string &path, TensorLoader *loader)
    {
        conv.load(path + "conv.", loader);
    }

    Qwen3TTSTokenizerV2CausalTransConvNet::Qwen3TTSTokenizerV2CausalTransConvNet(InitContext *ctx,
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride):
        Block(),
        left_pad(kernel_size - stride),
        right_pad(kernel_size - stride),
        conv(ctx, in_channels, out_channels, kernel_size, stride)
    {
    }

    ggml::tensor *Qwen3TTSTokenizerV2CausalTransConvNet::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        CHATLLM_CHECK(ggml::get_dim(input, 0) > left_pad + right_pad);

        auto hidden_state = conv.forward(ctx, input);
        hidden_state = ggml::view_4d(ctx, hidden_state,
            ggml::get_dim(hidden_state, 0) - left_pad - right_pad, ggml::get_dim(hidden_state, 1), ggml::get_dim(hidden_state, 2), ggml::get_dim(hidden_state, 3),
            ggml::row_size(hidden_state),
            ggml::row_size(hidden_state) * ggml::get_dim(hidden_state, 1),
            ggml::row_size(hidden_state) * ggml::get_dim(hidden_state, 1) * ggml::get_dim(hidden_state, 2),
            ggml::element_size(hidden_state) * left_pad);
        hidden_state = ggml::cont(ctx, hidden_state);
        return hidden_state;
    }

    int64_t Qwen3TTSTokenizerV2CausalTransConvNet::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += conv.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTokenizerV2CausalTransConvNet::load(const std::string &path, TensorLoader *loader)
    {
        conv.load(path + "conv.", loader);
    }

    Qwen3TTSTokenizerV2ConvNeXtBlock::Qwen3TTSTokenizerV2ConvNeXtBlock(InitContext *ctx, int dim):
        Block(),
        act(ActFunc::GELU),
        dwconv(ctx, dim, dim, 7, 1, 1, dim),
        norm(ctx, dim),
        pwconv1(ctx, dim, dim * 4),
        pwconv2(ctx, dim * 4, dim),
        gamma(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, dim))
    {
    }

    ggml::tensor *Qwen3TTSTokenizerV2ConvNeXtBlock::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto hidden_states = input;
        hidden_states = dwconv.forward(ctx, hidden_states);
        hidden_states = ggml::permute(ctx, hidden_states, 1, 0, 2);
        hidden_states = norm.forward(ctx, hidden_states);
        hidden_states = pwconv1.forward(ctx, hidden_states);
        hidden_states = ggml::act(ctx, act, hidden_states);
        hidden_states = pwconv2.forward(ctx, hidden_states);

        hidden_states = ggml::mul(ctx, hidden_states, gamma);

        hidden_states = ggml::permute(ctx, hidden_states, 1, 0, 2);

        hidden_states = ggml::add(ctx, hidden_states, input);
        return hidden_states;
    }

    int64_t Qwen3TTSTokenizerV2ConvNeXtBlock::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r +=  dwconv.get_param_num(effective_only);
        r +=    norm.get_param_num(effective_only);
        r += pwconv1.get_param_num(effective_only);
        r += pwconv2.get_param_num(effective_only);
        r += ggml::nelements(gamma);
        return r;
    }

    void Qwen3TTSTokenizerV2ConvNeXtBlock::load(const std::string &path, TensorLoader *loader)
    {
        pwconv1.load(path + "pwconv1.", loader);
        pwconv2.load(path + "pwconv2.", loader);
         dwconv.load(path + "dwconv.", loader);
           norm.load(path + "norm.", loader);
        loader->read_tensor(path + "gamma", gamma);
    }

    SnakeBeta::SnakeBeta(InitContext *ctx, int dim):
        Block(),
        no_div_by_zero(0.000000001f),
        alpha(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, dim)),
         beta(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, dim))
    {
    }

    ggml::tensor *SnakeBeta::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto alpha = ggml::reshape_3d(ctx, this->alpha, 1, ggml::get_dim(this->alpha, 0), 1);
        auto beta  = ggml::reshape_3d(ctx, this->beta,  1, ggml::get_dim(this->beta, 0), 1);

        alpha = ggml::mul(ctx, input, alpha);
        alpha = ggml::sin(ctx, alpha);
        alpha = ggml::square(ctx, alpha);
        alpha = ggml::mul(ctx, alpha, beta);

        auto hidden_states = ggml::add(ctx, input, alpha);
        return hidden_states;
    }

    int64_t SnakeBeta::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += ggml::nelements(alpha);
        r += ggml::nelements(beta);
        return r;
    }

    void SnakeBeta::load(const std::string &path, TensorLoader *loader)
    {
        std::vector<float> data(ggml::nelements(alpha));
        loader->read_tensor(path + "alpha", alpha);
        loader->read_tensor(path + "beta",  beta);

        Backend::read_tensor_data(alpha, data.data(), 0, data.size() * sizeof(float));
        for (int i = 0; i < (int)data.size(); i++)
        {
            data[i] = expf(data[i]);
        }
        Backend::write_tensor_data(alpha, data.data(), 0, data.size() * sizeof(float));

        Backend::read_tensor_data(beta, data.data(), 0, data.size() * sizeof(float));
        for (int i = 0; i < (int)data.size(); i++)
        {
            float t = expf(data[i]);
            t = 1.0f / (t + no_div_by_zero);
            data[i] = t;
        }
        Backend::write_tensor_data(beta, data.data(), 0, data.size() * sizeof(float));
    }

    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit::Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(InitContext *ctx, int dim, int dilation):
        Block(), act1(ctx, dim), act2(ctx, dim),
        conv1(ctx, dim, dim, 7, dilation),
        conv2(ctx, dim, dim, 1)
    {}

    ggml::tensor *Qwen3TTSTokenizerV2DecoderDecoderResidualUnit::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto residual = input;
        auto hidden_states = input;
        hidden_states =  act1.forward(ctx, hidden_states);
        hidden_states = conv1.forward(ctx, hidden_states);
        hidden_states =  act2.forward(ctx, hidden_states);
        hidden_states = conv2.forward(ctx, hidden_states);
        hidden_states = ggml::add(ctx, hidden_states, residual);
        return hidden_states;
    }

    int64_t Qwen3TTSTokenizerV2DecoderDecoderResidualUnit::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += conv1.get_param_num(effective_only);
        r += conv2.get_param_num(effective_only);
        r +=  act1.get_param_num(effective_only);
        r +=  act2.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTokenizerV2DecoderDecoderResidualUnit::load(const std::string &path, TensorLoader *loader)
    {
        conv1.load(path + "conv1.", loader);
        conv2.load(path + "conv2.", loader);
         act1.load(path + "act1.", loader);
         act2.load(path + "act2.", loader);
    }

    Qwen3TTSTokenizerV2DecoderDecoderBlock::Qwen3TTSTokenizerV2DecoderDecoderBlock(InitContext *ctx,
        int in_dim,
        int out_dim,
        int upsample_rate)
    {
        block.add_block(new SnakeBeta(ctx, in_dim));
        block.add_block(new Qwen3TTSTokenizerV2CausalTransConvNet(ctx, in_dim, out_dim, 2 * upsample_rate, upsample_rate));
        for (auto dilation : {1, 3, 9})
        {
            block.add_block(new Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(ctx, out_dim, dilation));
        }
    }

    ggml::tensor *Qwen3TTSTokenizerV2DecoderDecoderBlock::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        return block.forward(ctx, input);
    }

    int64_t Qwen3TTSTokenizerV2DecoderDecoderBlock::get_param_num(bool effective_only) const
    {
        return block.get_param_num(effective_only);
    }

    void Qwen3TTSTokenizerV2DecoderDecoderBlock::load(const std::string &path, TensorLoader *loader)
    {
        block.load(path + "block.", loader);
    }

    Qwen3TTSTokenizerV2DecoderAttention::Qwen3TTSTokenizerV2DecoderAttention(InitContext *ctx, int hidden_size,
        int num_attention_heads, int num_key_value_heads, int head_dim, int max_length):
        Base(ctx, hidden_size, num_attention_heads, num_key_value_heads, head_dim, max_length, false, false)
    {
        rope_mode = RoPEMode::Original;
    }

    ggml::tensor *Qwen3TTSTokenizerV2DecoderAttention::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        auto hidden_states = Base::forward(ctx, input, n_past);
        if (self_attn_layer_scale)
        {
            hidden_states = ggml::mul(ctx, hidden_states, self_attn_layer_scale);
        }
        return hidden_states;
    }

    ScaledSiLUMLP::ScaledSiLUMLP(InitContext *ctx, int hidden_size, int intermediate_size):
        SiLUMLP(ctx, hidden_size, intermediate_size),
        scale(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, hidden_size))
    {
    }

    ggml::tensor *ScaledSiLUMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        hidden_states = SiLUMLP::forward(ctx, hidden_states);
        hidden_states = ggml::mul(ctx, hidden_states, scale);
        return hidden_states;
    }

    int64_t ScaledSiLUMLP::get_param_num(bool effective_only) const
    {
        int64_t r = SiLUMLP::get_param_num(effective_only);
        r += ggml::nelements(scale);
        return r;
    }

    Qwen3TTSTokenizerV2DecoderTransformerLayer::Qwen3TTSTokenizerV2DecoderTransformerLayer(InitContext *ctx,
            const Qwen3TTSTokenizerV2DecoderConfig &config, const int layer_idx):
        Base(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_position_embeddings),
        is_sliding(config.layer_type_sliding[layer_idx]),
        self_attn_layer_scale(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size))
    {
        CHATLLM_CHECK(config.hidden_act == ActFunc::SILU);
        CHATLLM_CHECK(!config.attention_bias);
        attention.freq_base = config.rope_theta;
        attention.self_attn_layer_scale = self_attn_layer_scale;
        set_id(layer_idx);
        attention.set_id(layer_idx);
    }

    int64_t Qwen3TTSTokenizerV2DecoderTransformerLayer::get_param_num(bool effective_only) const
    {
        int64_t r = Base::get_param_num(effective_only);
        r += ggml::nelements(self_attn_layer_scale);
        return r;
    }

    void Qwen3TTSTokenizerV2DecoderTransformerLayer::load(const std::string &path, TensorLoader *loader)
    {
        Base::load(path, loader);
        loader->read_tensor(path + "self_attn_layer_scale.scale", self_attn_layer_scale);
        loader->read_tensor(path + "mlp_layer_scale.scale", mlp.scale);
    }

    std::vector<int> get_int_array(const json::JSON &config, const std::string &field)
    {
        std::vector<int> r;
        auto arr = config[field];
        if (!arr.IsArray()) return r;
        for (auto ele : arr.ArrayRange())
        {
            r.push_back((int)ele.ToInt());
        }
        return r;
    }

    Qwen3TTSTokenizerV2DecoderConfig::Qwen3TTSTokenizerV2DecoderConfig(const json::JSON &config):
        Qwen3TTSTokenizerV2DecoderConfig(
            (int)config["codebook_size"].ToInt(),
            (int)config["hidden_size"].ToInt(),
            (int)config["latent_dim"].ToInt(),
            (int)config["codebook_dim"].ToInt(),
            (int)config["max_position_embeddings"].ToInt(),
            (float)config["rope_theta"].ToFloat(),
            (int)config["num_attention_heads"].ToInt(),
            (int)config["num_key_value_heads"].ToInt(),
            (int)config["attention_bias"].ToInt(),
            (int)config["sliding_window"].ToInt(),
            (int)config["intermediate_size"].ToInt(),
            ActFunc::SILU,
            (float)config["layer_scale_initial_scale"].ToFloat(),
            (float)config["rms_norm_eps"].ToFloat(),
            (int)config["num_hidden_layers"].ToInt(),
            (int)config["num_quantizers"].ToInt(),
            get_int_array(config, "upsample_rates"),
            get_int_array(config, "upsampling_ratios"),
            (int)config["decoder_dim"].ToInt(),
            (int)config["head_dim"].ToInt(-1)
        )
    {
    }

    Qwen3TTSTokenizerV2DecoderConfig::Qwen3TTSTokenizerV2DecoderConfig(
            int  codebook_size,
            int  hidden_size,
            int  latent_dim,
            int  codebook_dim,
            int  max_position_embeddings,
            float rope_theta,
            int  num_attention_heads,
            int  num_key_value_heads,
            bool attention_bias,
            int  sliding_window,
            int  intermediate_size,
            ActFunc hidden_act,
            float layer_scale_initial_scale,
            float rms_norm_eps,
            int  num_hidden_layers,
            int  num_quantizers,
            const std::vector<int> &upsample_rates,
            const std::vector<int> &upsampling_ratios,
            int  decoder_dim,
            int  head_dim):
        codebook_size(codebook_size),
        hidden_size(hidden_size),
        latent_dim(latent_dim),
        codebook_dim(codebook_dim),
        max_position_embeddings(max_position_embeddings),
        rope_theta(rope_theta),
        num_attention_heads(num_attention_heads),
        num_key_value_heads(num_key_value_heads),
        attention_bias(attention_bias),
        sliding_window(sliding_window),
        intermediate_size(intermediate_size),
        hidden_act(hidden_act),
        layer_scale_initial_scale(layer_scale_initial_scale),
        rms_norm_eps(rms_norm_eps),
        num_hidden_layers(num_hidden_layers),
        num_quantizers(num_quantizers),
        upsample_rates(upsample_rates),
        upsampling_ratios(upsampling_ratios),
        decoder_dim(decoder_dim),
        head_dim(head_dim > 0 ? head_dim : hidden_size / num_attention_heads)
    {
        for (int i = 0; i < num_hidden_layers; i++) layer_type_sliding.push_back(true);
    }

    Qwen3TTSTokenizerV2DecoderTransformerModel::Qwen3TTSTokenizerV2DecoderTransformerModel(InitContext *ctx,
        const Qwen3TTSTokenizerV2DecoderConfig &config):
        Block(),
        max_length(config.max_position_embeddings),
        sliding_window(config.sliding_window),
        input_proj(ctx, config.latent_dim, config.hidden_size),
        output_proj(ctx, config.hidden_size, config.latent_dim),
        norm(ctx, config.hidden_size),
        mask(ggml::new_tensor_2d(ctx, GGML_TYPE_F32,  max_length,  max_length))
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            layers.add_block(new Qwen3TTSTokenizerV2DecoderTransformerLayer(ctx, config, i));
        }

        std::vector<float> v_sliding_mask;
        v_sliding_mask.resize(max_length * max_length, -INFINITY);
        float *data = v_sliding_mask.data();

        for (int i = 0; i < max_length; i++)
        {
            for (int j = 0; j < max_length; j++)
            {
                int pos = i * max_length + j;
                if ((i - config.sliding_window < j) && (j <= i))
                //if (j <= i)
                    data[pos] = 0.0f;
            }
        }

        ctx->get_allocator()->alloc(mask);
        Backend::write_tensor_data(mask, data);
    }

    ggml::tensor *Qwen3TTSTokenizerV2DecoderTransformerModel::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        const int qlen = (int)ggml::get_dim(input, 1);

        ggml::tensor *sub_mask = ggml::view_2d(ctx, mask, qlen, qlen, max_length * ggml::element_size(mask), 0);
        sub_mask = ggml::cont(ctx, sub_mask);

        for (int i = 0; i < (int)layers.blocks.size(); i++)
        {
            auto layer = dynamic_cast<Qwen3TTSTokenizerV2DecoderTransformerLayer *>(layers.blocks[i].get());
            layer->attention.mask = layer->is_sliding ? sub_mask : nullptr;
        }

        auto inputs_embeds = input_proj.forward(ctx, input);
        auto hidden_states = layers.forward(ctx, inputs_embeds, 0);
        hidden_states = norm.forward(ctx, hidden_states);
        hidden_states = output_proj.forward(ctx, hidden_states);
        return hidden_states;
    }

    int64_t Qwen3TTSTokenizerV2DecoderTransformerModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += output_proj.get_param_num(effective_only);
        r +=  input_proj.get_param_num(effective_only);
        r +=      layers.get_param_num(effective_only);
        r +=        norm.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTokenizerV2DecoderTransformerModel::load(const std::string &path, TensorLoader *loader)
    {
        output_proj.load(path + "output_proj.", loader);
         input_proj.load(path + "input_proj.", loader);
             layers.load(path + "layers.", loader);
               norm.load(path + "norm.", loader);
    }

    static int get_total_upsample(const Qwen3TTSTokenizerV2DecoderConfig &config)
    {
        int r = 1;
        for (int i = 0; i < (int)config.upsample_rates.size(); i++)
            r *= config.upsample_rates[i];
        for (int i = 0; i < (int)config.upsampling_ratios.size(); i++)
            r *= config.upsampling_ratios[i];
        return r;
    }

    Qwen3TTSTokenizerV2Decoder::Qwen3TTSTokenizerV2Decoder(InitContext *ctx,
        const Qwen3TTSTokenizerV2DecoderConfig &config):
        Block(),
        config(config),
        total_upsample(get_total_upsample(config)),
        pre_transformer(ctx, config),
        quantizer(ctx, config),
        pre_conv(ctx, config.codebook_dim, config.latent_dim, 3)
    {
        for (auto factor : config.upsampling_ratios)
        {
            auto seq = new Sequential(ctx);
            seq->add_block(new Qwen3TTSTokenizerV2CausalTransConvNet(ctx, config.latent_dim, config.latent_dim,
                                factor, factor));
            seq->add_block(new Qwen3TTSTokenizerV2ConvNeXtBlock(ctx, config.latent_dim));
            upsample.add_block(seq);
        }

        decoder.add_block(new Qwen3TTSTokenizerV2CausalConvNet(ctx, config.latent_dim, config.decoder_dim, 7));
        for (int i = 0; i < (int)config.upsample_rates.size(); i++)
        {
            int in_dim  = config.decoder_dim / (1 << i);
            int out_dim = config.decoder_dim / (1 << (i + 1));
            decoder.add_block(new Qwen3TTSTokenizerV2DecoderDecoderBlock(ctx, in_dim, out_dim, config.upsample_rates[i]));
        }
        int output_dim = config.decoder_dim / (1 << config.upsample_rates.size());
        decoder.add_block(new SnakeBeta(ctx, output_dim));
        decoder.add_block(new Qwen3TTSTokenizerV2CausalConvNet(ctx, output_dim, 1, 7));
    }

    ggml::tensor *Qwen3TTSTokenizerV2Decoder::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        CHATLLM_CHECK(ggml::get_dim(input, 1) == config.num_quantizers);

        auto hidden = quantizer.forward(ctx, input);
        hidden = pre_conv.forward(ctx, hidden);

        hidden = ggml::permute(ctx, hidden, 1, 0, 2);
        hidden = ggml::cont(ctx, hidden);

        hidden = pre_transformer.forward(ctx, hidden);
        hidden = ggml::permute(ctx, hidden, 1, 0, 2);

        hidden = upsample.forward(ctx, hidden);
        auto wav = decoder.forward(ctx, hidden);
        wav = ggml::clamp(ctx, wav, -1.0f, 1.0f);
        return wav;
    }

    ggml::tensor *Qwen3TTSTokenizerV2Decoder::chunked_decode(ComputeContext *ctx, ggml::tensor *codes,
            const int chunk_size, const int left_context_size)
    {
        ggml::tensor *wav = nullptr;
        codes = ggml::is_contiguous(codes) ? codes : ggml::cont(ctx, codes);
        const int64_t qlen = ggml::get_dim(codes, 1);
        int64_t start_index = 0;
        while (start_index < qlen)
        {
            const int64_t end_index = std::min(start_index + chunk_size, qlen);
            const int64_t context_size = std::min(start_index, (int64_t)left_context_size);
            auto chunk = ggml::view_2d(ctx, codes,
                            ggml::get_dim(codes, 0), (end_index - start_index + context_size),
                            ggml::row_size(codes), (start_index - context_size) * ggml::element_size(codes));
            auto wav_chunk = forward(ctx, chunk);

            const int64_t sample_cnt = ggml::get_dim(wav_chunk, 1);
            wav_chunk = ggml::view_2d(ctx, wav_chunk, ggml::get_dim(wav_chunk, 0), sample_cnt - (context_size * total_upsample),
                                    ggml::row_size(wav_chunk), context_size * total_upsample * ggml::element_size(wav_chunk));
            wav = wav ? ggml::concat(ctx, wav, wav_chunk, 1) : wav_chunk;

            start_index = end_index;
        }
        return wav;
    }

    int64_t Qwen3TTSTokenizerV2Decoder::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += pre_transformer.get_param_num(effective_only);
        r +=       quantizer.get_param_num(effective_only);
        r +=        pre_conv.get_param_num(effective_only);
        r +=        upsample.get_param_num(effective_only);
        r +=         decoder.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTokenizerV2Decoder::load(const std::string &path, TensorLoader *loader)
    {
        pre_transformer.load(path + "pre_transformer.", loader);
              quantizer.load(path + "quantizer.", loader);
               pre_conv.load(path + "pre_conv.", loader);
               upsample.load(path + "upsample.", loader);
                decoder.load(path + "decoder.", loader);
    }

    Qwen3TTSTokenizerV2Model::Qwen3TTSTokenizerV2Model(InitContext *ctx,
        const json::JSON &config):
        Block(),
        decoder(ctx, config["decoder_config"])
    {
    }

    int64_t Qwen3TTSTokenizerV2Model::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += decoder.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTokenizerV2Model::load(const std::string &path, TensorLoader *loader)
    {
        decoder.load(path + "decoder.", loader);
    }

    Qwen3TTSDecoderLayer::Qwen3TTSDecoderLayer(InitContext *ctx,
            const int hidden_size,
            const int num_attention_heads,
            const int intermediate_size,
            const int num_key_value_heads,
            const int max_length,
            const int head_dim,
            const float rms_norm_eps):
        Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads,
            head_dim > 0 ? head_dim : (hidden_size / num_attention_heads), max_length)
    {
    }

    Qwen3TTSTalkerCodePredictorModel::Qwen3TTSTalkerCodePredictorModel(InitContext *ctx, const json::JSON &config, const int embedding_dim):
        Block(),
        norm(ctx, (int)config["hidden_size"].ToInt())
    {
        const int hidden_size           = (int)config["hidden_size"].ToInt();
        const int num_attention_heads   = (int)config["num_attention_heads"].ToInt();
        const int intermediate_size     = (int)config["intermediate_size"].ToInt();
        const int num_key_value_heads   = (int)config["num_key_value_heads"].ToInt();
        const int max_length            = (int)config["max_position_embeddings"].ToInt();
        const int head_dim              = (int)config["head_dim"].ToInt();
        const int num_hidden_layers     = (int)config["num_hidden_layers"].ToInt();
        const int num_code_groups       = (int)config["num_code_groups"].ToInt();
        const int vocab_size            = (int)config["vocab_size"].ToInt();

        CHATLLM_CHECK(config["sliding_window"].IsNull());

        for (int i = 0; i < num_hidden_layers; i++)
        {
            ctx->move_to_layer(i);
            auto layer = new Qwen3TTSDecoderLayer(ctx, hidden_size, num_attention_heads, intermediate_size,
                num_key_value_heads, max_length, head_dim);

            auto allocator = ctx->get_allocator();
            auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
            layer->set_cache_buffer(buf);

            layers.add_block(layer);
        }

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        for (int i = 0; i < num_code_groups - 1; i++)
        {
            codec_embedding.add_block(new Embedding(ctx, vocab_size, embedding_dim));
        }
    }

    ggml::tensor *Qwen3TTSTalkerCodePredictorModel::forward(ComputeContext *ctx, ggml::tensor *inputs_embeds, int n_past)
    {
        auto hidden_states = inputs_embeds;
        hidden_states = layers.forward(ctx, hidden_states, n_past);
        hidden_states = norm.forward(ctx, hidden_states);
        return hidden_states;
    }

    int64_t Qwen3TTSTalkerCodePredictorModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += codec_embedding.get_param_num(effective_only);
        r +=          layers.get_param_num(effective_only);
        r +=            norm.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTalkerCodePredictorModel::load(const std::string &path, TensorLoader *loader)
    {
        codec_embedding.load(path + "codec_embedding.", loader);
                 layers.load(path + "layers.", loader);
                   norm.load(path + "norm.", loader);
    }

    Embedding *Qwen3TTSTalkerCodePredictorModel::get_codec_embedding(int index)
    {
        CHATLLM_CHECK((0 <= index) && (index < (int)codec_embedding.blocks.size()));
        return dynamic_cast<Embedding *>(codec_embedding.blocks[index].get());
    }

    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration::Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(InitContext *ctx,
        const json::JSON &config, const int embedding_dim):
        Block(),
        num_code_groups((int)config["num_code_groups"].ToInt()),
        hidden_size((int)config["hidden_size"].ToInt()),
        vocab_size((int)config["vocab_size"].ToInt()),
        model(ctx, config, embedding_dim)
    {
        for (int i = 0; i < num_code_groups - 1; i++)
        {
            lm_head.add_block(new Embedding(ctx, vocab_size, hidden_size));
        }

        if (hidden_size != embedding_dim)
        {
            small_to_mtp_projection.reset(new Linear(ctx, embedding_dim, hidden_size));
        }
        else
            small_to_mtp_projection.reset(new Identity());
    }

    ggml::tensor *Qwen3TTSTalkerCodePredictorModelForConditionalGeneration::forward(ComputeContext *ctx, ggml::tensor *inputs_embeds, int n_past, int generation_step)
    {
        if (ggml::type_of(inputs_embeds) == ggml::type::GGML_TYPE_I32)
        {
            inputs_embeds = model.codec_embedding.blocks[generation_step - 1]->forward(ctx, inputs_embeds);
        }
        inputs_embeds = small_to_mtp_projection->forward(ctx, inputs_embeds);
        auto hidden_states = model.forward(ctx, inputs_embeds, n_past);

        int64_t offset = ggml::row_size(hidden_states) * (ggml::get_dim(hidden_states, 1) - 1);
        auto sub = ggml::view_2d(ctx, hidden_states, ggml::get_dim(hidden_states, 0), 1, ggml::row_size(hidden_states), offset);
        auto logits = lm_head.blocks[generation_step]->forward(ctx, sub);

        return logits;
    }

    ggml::tensor *Qwen3TTSTalkerCodePredictorModelForConditionalGeneration::forward_finetune(ComputeContext *ctx, ggml::tensor *inputs_embeds)
    {
        inputs_embeds = small_to_mtp_projection->forward(ctx, inputs_embeds);
        auto hidden_states = model.forward(ctx, inputs_embeds, 0);
        const int hidden_size = ggml::get_dim(hidden_states, 0);

        ggml::tensor *logits = nullptr;
        for (int i = 1; i < num_code_groups; i++)
        {
            int64_t offset = ggml::row_size(hidden_states) * i;
            auto sub = ggml::view_2d(ctx, hidden_states, hidden_size, 1, ggml::row_size(hidden_states), offset);
            sub = lm_head.blocks[i - 1]->forward(ctx, sub);
            logits = logits ? ggml::concat(ctx, logits, sub, 1) : sub;
        }
        return logits;
    }

    int64_t Qwen3TTSTalkerCodePredictorModelForConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += small_to_mtp_projection->get_param_num(effective_only);
        r +=                 lm_head.get_param_num(effective_only);
        r +=                   model.get_param_num(effective_only);
        return r;
    }

    void Qwen3TTSTalkerCodePredictorModelForConditionalGeneration::load(const std::string &path, TensorLoader *loader)
    {
        small_to_mtp_projection->load(path + "small_to_mtp_projection.", loader);
                        lm_head.load(path + "lm_head.", loader);
                          model.load(path + "model.", loader);
    }

    void Qwen3TTSTalkerResizeMLP::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        fc0.load(path + "linear_fc1.", loader);
        fc1.load(path + "linear_fc2.", loader);
    }
}


namespace chatllm::qwen::v3_tts
{
    struct Config : v3::Config
    {
        int mrope_section[v2_5_vl::MROPE_SECTION_MAX];
        int text_hidden_size;
        int talker_vocab_size;
    };

    struct CodecTokIds
    {
        int codec_bos_id = 0;
        int codec_eos_token_id = 0;
        int codec_think_id = 0;
        int codec_nothink_id = 0;
        int codec_pad_id = 0;
        int codec_think_bos_id = 0;
        int codec_think_eos_id = 0;
        int tts_bos_token_id = 0;
        int tts_eos_token_id = 0;
        int tts_pad_token_id = 0;
    };

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
        void encode(const std::string &text, std::vector<int> &ids) const;
        void encode_instruct(std::vector<int> &ids) const;

        void add_tokens(const std::map<std::string, int> &added_tokens);
    protected:
        std::string _build_assistant_text(const std::string &text) const;
        std::string _build_ref_text(const std::string &text) const;
        std::string _build_instruct_text(const std::string &text) const;
    public:
        std::string lanuage = "auto";
        std::string instruct = "";
        std::string speaker = "vivian";
        CodecTokIds tok_ids;
    };

    Tokenizer::Tokenizer(const BaseConfig &config):
        v3::Tokenizer(config, nullptr)
    {}

    void Tokenizer::add_tokens(const std::map<std::string, int> &added_tokens)
    {
        for (auto kv : added_tokens)
        {
            tp->AddAddedToken(kv.first, kv.second);
        }
    }

    std::string Tokenizer::_build_assistant_text(const std::string &text) const
    {
        std::ostringstream oss;
        oss << "<|im_start|>assistant\n"
            << text
            << "<|im_end|>\n<|im_start|>assistant\n";
        return oss.str();
    }

    std::string Tokenizer::_build_ref_text(const std::string &text) const
    {
        std::ostringstream oss;
        oss << "<|im_start|>assistant\n"
            << text
            << "<|im_end|>\n";
        return oss.str();
    }

    std::string Tokenizer::_build_instruct_text(const std::string &text) const
    {
        std::ostringstream oss;
        oss << "<|im_start|>user\n"
            << text
            << "<|im_end|>\n";
        return oss.str();
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        auto t = _build_assistant_text(text);
        BaseTokenizer::encode(t, ids);
    }

    void Tokenizer::encode_instruct(std::vector<int> &ids) const
    {
        if (instruct.size() < 1) return;
        auto text = _build_instruct_text(instruct);
        BaseTokenizer::encode(text, ids);
    }

    class TalkerGeneration : public Block
    {
    public:
        TalkerGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096):
            eval(runtime_config, "aud", GRAPH_SIZE),
            _ctx(eval.get_backend_context()),
            code_block_size(16)
        {
            _ctx.cache_dtype = runtime_config.cache_type;
        }

        bool load_more(ggml::type dtype, const json::JSON &config);
        bool load(ModelLoader &loader);

        bool project_text(const GenerationConfig &gen_config, const int *input_text_ids, const int id_num,
            ggml::type dtype, std::vector<uint8_t> &buf);

        void code_predict(const GenerationConfig &gen_config, Sampler *sampler, std::vector<int> &sequences,
            const float *past_hidden, const float *last_id_hidden);

        void speech_decode(const GenerationConfig &gen_config, const std::vector<int> &encoded, std::vector<float> &decoded);
    protected:
        int code_predict(const GenerationConfig &gen_config, Sampler *sampler, const int *input_text_ids, const int id_num, int n_past, const int generation_step);
        int code_predict(const GenerationConfig &gen_config, Sampler *sampler, ggml::type dtype, const float *past_hidden, const float *last_id_hidden, int n_past, const int generation_step);

    public:
        // parts of Qwen3TTSTalkerForConditionalGeneration
        std::unique_ptr<tts::Qwen3TTSTalkerResizeMLP>                                    text_projection;
        std::unique_ptr<Linear>                                                          codec_head;
        std::unique_ptr<tts::Qwen3TTSTalkerCodePredictorModelForConditionalGeneration>   code_predictor;

        std::unique_ptr<tts::Qwen3TTSTokenizerV2Model>                                   speech_tokenizer;

        std::unique_ptr<Embedding>                                                       text_embedding;

    public:
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
        bool loaded = false;
        int embedding_dim = -1;
        const int code_block_size;
    };

    class ConditionalGeneration : public v2_5_vl::ExtendEmbedding, public v3::ConditionalGeneration
    {
    public:
        typedef v3::ConditionalGeneration Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = ModelType::MODEL_TYPE_QWEN3_TTS);
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void set_tokenizer(BaseTokenizer *tokenizer) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels) override;

    protected:
        void test(const GenerationConfig &gen_config, std::vector<int16_t> &audio);
        void generate_audio_codes(const GenerationConfig &gen_config,
            const std::vector<int> &input_ids,
            const std::vector<int> &instruct_ids,
            const std::vector<int> &ref_ids,
            const std::string &voice_clone_prompt,
            const int language_id,
            const int speaker_id,
            std::vector<int16_t> &audio);
        void inject_text_ids(const GenerationConfig &gen_config, const std::vector<int> &input_ids, std::vector<int> &mapped_ids);
        void inject_text_ids(const GenerationConfig &gen_config, const int *input_ids, const int num, std::vector<int> &mapped_ids);
        void map_text_ids(const GenerationConfig &gen_config, const std::vector<int> &input_ids, std::vector<int> &mapped_ids);
        void project_text_ids(const GenerationConfig &gen_config, const int *input_ids, const int num);
        void reset_token_ids(void);
        void prepare_ids(const GenerationConfig &gen_config,
            const std::vector<int> &input_ids,
            const std::vector<int> &instruct_ids,
            const std::vector<int> &ref_ids,
            const std::string &voice_clone_prompt,
            const int language_id,
            const int speaker_id);
        bool run_main_model(const GenerationConfig &gen_config,
                            std::vector<float> &output, std::vector<float> &last_hidden_states);
        int run_main_model(const GenerationConfig &gen_config, std::vector<float> &last_hidden_states, Sampler *sampler);
        int run_main_model(const GenerationConfig &gen_config, const int added_id, const std::vector<int> &code_block, std::vector<float> &last_hidden_states, Sampler *sampler);
    protected:
        const int talker_vocab_size;
        TalkerGeneration talker;
        std::map<std::string, int> codec_language_id;
        std::map<std::string, int> spk_id;
        std::map<std::string, int> spk_dialect_id;
        std::map<std::string, int> added_tokens;
        std::vector<int> talker_mapped_ids;
        std::vector<int> talker_mapped_added_ids;
        std::vector<int> trailing_text_ids;
        std::vector<float> talker_mapped_added_scale;
        int projected_token_num = 0;
        CodecTokIds codec_tok_ids;
        Embedding *embedding;
        std::vector<float>  last_hidden_states;
        std::set<int>       termination_token_ids;
        int tts_bos_token_id = -1;
        int tts_eos_token_id = -1;
        int tts_pad_token_id = -1;
        int trailing_text_next_index = 0;
    };

    bool TalkerGeneration::load_more(ggml::type dtype, const json::JSON &config)
    {
        bool r = false;

        const auto cfg = config["config.json"];
        if (cfg.IsNull()) return r;
        const auto p = cfg["talker_config"];
        if (p.IsNull()) return r;
        const auto q = config["speech_tokenizer-config.json"];
        if (q.IsNull()) return r;

        const int hidden_size       = (int)p["hidden_size"].ToInt();
        const int text_hidden_size  = (int)p["text_hidden_size"].ToInt();
        const int vocab_size        = (int)p["vocab_size"].ToInt();
        const int text_vocab_size   = (int)p["text_vocab_size"].ToInt();

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 400; // TODO: counting tensors (this is boring)
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        speech_tokenizer.reset(new tts::Qwen3TTSTokenizerV2Model(&_ctx, q));
         text_projection.reset(new tts::Qwen3TTSTalkerResizeMLP(&_ctx, text_hidden_size, text_hidden_size, hidden_size, ActFunc::SILU, true));
          text_embedding.reset(new Embedding(&_ctx, text_vocab_size, text_hidden_size));
          code_predictor.reset(new tts::Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(&_ctx, p["code_predictor_config"], hidden_size));
              codec_head.reset(new Linear(&_ctx, hidden_size, vocab_size, false));

        embedding_dim = hidden_size;

        // TODO: counting tensors
        //_ctx.check_used_mem_size(true);

        return true;
    }

    bool TalkerGeneration::load(ModelLoader &loader)
    {
        if (!codec_head.get()) return false;

        loader.push_allocator_manager(eval.get_layer_allocators());

        speech_tokenizer->load("", &loader);
         text_projection->load("talker.text_projection.", &loader);
          text_embedding->load("talker.model.text_embedding.", &loader);
          code_predictor->load("talker.code_predictor.", &loader);
              codec_head->load("talker.codec_head.", &loader);

        loader.pop_allocator_manager();

        loaded = true;
        return true;
    }

    void TalkerGeneration::code_predict(const GenerationConfig &gen_config, Sampler *sampler, std::vector<int> &sequences,
        const float *past_hidden, const float *last_id_hidden)
    {
        int id = code_predict(gen_config, sampler, ggml::type::GGML_TYPE_F32, past_hidden, last_id_hidden, 0, 0);
        sequences.push_back(id);
        for (int i = 0; i < 15 - 1; i++)
        {
            id = code_predict(gen_config, sampler, &id, 1, 2 + i, 1 + i);
            sequences.push_back(id);
        }
    }

    int TalkerGeneration::code_predict(const GenerationConfig &gen_config, Sampler *sampler, const int *input_text_ids, const int id_num, int n_past, const int generation_step)
    {
        ggml::tensor *input = nullptr;
        const auto make_graph = [this, &input, input_text_ids, id_num, n_past, generation_step](ComputeContext *ctx) -> ggml::tensor * {
            input = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, id_num);
            auto r = code_predictor->forward(ctx, input, n_past, generation_step);
            return r;
        };
        const auto write_input_data = [&input, input_text_ids, id_num](ComputeContext *ctx) {
            Backend::write_tensor_data(input, input_text_ids, 0, id_num * sizeof(input_text_ids[0]));
        };

        std::vector<int64_t> shape;
        std::vector<uint8_t> buf;
        eval.evaluate(gen_config, make_graph, write_input_data, ggml::type::GGML_TYPE_F32, shape, buf);

        CHATLLM_CHECK(buf.size() == sizeof(float) * code_predictor->vocab_size);
        int id = sampler->sampling((float *)buf.data(), code_predictor->vocab_size);
        return id;
    }

    int TalkerGeneration::code_predict(const GenerationConfig &gen_config, Sampler *sampler, ggml::type dtype, const float *past_hidden, const float *last_id_hidden, int n_past, const int generation_step)
    {
        ggml::tensor *input = nullptr;
        const auto make_graph = [this, &input, dtype, n_past, generation_step](ComputeContext *ctx) -> ggml::tensor * {
            input = ggml::new_tensor_2d(ctx, dtype, embedding_dim, 2);
            auto r = code_predictor->forward(ctx, input, n_past, generation_step);
            return r;
        };
        const auto write_input_data = [&input, past_hidden, last_id_hidden](ComputeContext *ctx) {
            auto size = ggml::nbytes(input) / 2;
            Backend::write_tensor_data(input, past_hidden,    0,    size);
            Backend::write_tensor_data(input, last_id_hidden, size, size);
        };

        std::vector<int64_t> shape;
        std::vector<uint8_t> buf;
        eval.evaluate(gen_config, make_graph, write_input_data, ggml::type::GGML_TYPE_F32, shape, buf);

        CHATLLM_CHECK(buf.size() == sizeof(float) * code_predictor->vocab_size);
        int id = sampler->sampling((float *)buf.data(), code_predictor->vocab_size);
        return id;
    }

    void TalkerGeneration::speech_decode(const GenerationConfig &gen_config, const std::vector<int> &encoded, std::vector<float> &decoded)
    {
        ggml::tensor *input = nullptr;
        const auto make_graph = [this, &encoded, &input](ComputeContext *ctx) -> ggml::tensor * {
            input = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, code_block_size, encoded.size() / code_block_size);
            auto r = ggml::transpose(ctx, input);
            r = speech_tokenizer->decoder.chunked_decode(ctx, r);
            return r;
        };
        const auto write_input_data = [&input, &encoded](ComputeContext *ctx) {
            Backend::write_tensor_data(input, encoded.data(), 0, ggml::nbytes(input));
        };

        std::vector<int64_t> shape;
        std::vector<uint8_t> buf;
        eval.evaluate(gen_config, make_graph, write_input_data, ggml::type::GGML_TYPE_F32, shape, buf);
        decoded.resize(buf.size() / sizeof(decoded[0]));
        memcpy(decoded.data(), buf.data(), buf.size());
    }

    bool TalkerGeneration::project_text(const GenerationConfig &gen_config, const int *input_text_ids, const int id_num,
            ggml::type dtype, std::vector<uint8_t> &buf)
    {
        ggml::tensor *input = nullptr;
        const auto make_graph = [this, &input, input_text_ids, id_num](ComputeContext *ctx) -> ggml::tensor * {
            input = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, id_num);
            auto r = text_embedding->forward(ctx, input);
            r = text_projection->forward(ctx, r);
            return r;
        };
        const auto write_input_data = [&input, input_text_ids, id_num](ComputeContext *ctx) {
            Backend::write_tensor_data(input, input_text_ids, 0, id_num * sizeof(input_text_ids[0]));
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type):
        v2_5_vl::ExtendEmbedding(10000),
        v3::ConditionalGeneration(config, runtime_config, type, true, 1, config.talker_vocab_size, config.hidden_size),
        talker_vocab_size(config.talker_vocab_size),
        talker(runtime_config)
    {
        delete pad_arg;
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        Base::set_tokenizer(tokenizer);
        (dynamic_cast<Tokenizer *>(tokenizer))->add_tokens(added_tokens);
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->lanuage  = utils::to_lower(utils::get_opt(args, "lanuage", tok->lanuage));
        tok->speaker  = utils::to_lower(utils::get_opt(args, "speaker", tok->speaker));
        tok->instruct =                 utils::get_opt(args, "instruct", tok->instruct);
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);

        {
            const auto cfg = config["tokenizer_config.json"];
            if (cfg.IsNull()) return false;
            const auto tok_cfg = cfg["added_tokens_decoder"];
            if (!tok_cfg.IsObject()) return false;
            for (auto &kv : tok_cfg.ObjectRange())
            {
                auto t = kv.second["content"].ToString();
                added_tokens[t] = (int)std::atoi(kv.first.c_str());
            };
        }

        const auto cfg = config["config.json"];
        if (cfg.IsNull()) return false;

        codec_tok_ids.tts_bos_token_id       = (int)cfg["tts_bos_token_id"].ToInt();
        codec_tok_ids.tts_eos_token_id       = (int)cfg["tts_eos_token_id"].ToInt();
        codec_tok_ids.tts_pad_token_id       = (int)cfg["tts_pad_token_id"].ToInt();

        const auto p = cfg["talker_config"];
        if (p.IsNull()) return false;

        codec_tok_ids.codec_bos_id       = (int)p["codec_bos_id"].ToInt();
        codec_tok_ids.codec_eos_token_id = (int)p["codec_eos_token_id"].ToInt();
        codec_tok_ids.codec_think_id     = (int)p["codec_think_id"].ToInt();
        codec_tok_ids.codec_nothink_id   = (int)p["codec_nothink_id"].ToInt();
        codec_tok_ids.codec_pad_id       = (int)p["codec_pad_id"].ToInt();
        codec_tok_ids.codec_think_bos_id = (int)p["codec_think_bos_id"].ToInt();
        codec_tok_ids.codec_think_eos_id = (int)p["codec_think_eos_id"].ToInt();

        for (auto kv : p["codec_language_id"].ObjectRange())
        {
            codec_language_id[kv.first] = (int)kv.second.ToInt();
        }
        for (auto kv : p["spk_id"].ObjectRange())
        {
            spk_id[kv.first] = (int)kv.second.ToInt();
        }
        for (auto kv : p["spk_is_dialect"].ObjectRange())
        {
            if (kv.second.IsString())
                spk_dialect_id[kv.first] = codec_language_id[kv.second.ToString()];
        }

        bool r = talker.load_more(this->config.dtype, config);

        termination_token_ids.emplace(codec_tok_ids.codec_eos_token_id);
        termination_token_ids.emplace(codec_tok_ids.codec_think_eos_id);

        return r;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {"talker.model.embed_tokens.",              "talker.model.codec_embedding."},
        });

        transformer->load("talker.model.", &loader, layer_ids);
        talker.load(loader);

        if (talker.loaded)
        {
            transformer->lm_head = talker.codec_head.get();
        }
    }

    void ConditionalGeneration::reset_token_ids(void)
    {
        talker_mapped_ids.clear();
        talker_mapped_added_ids.clear();
        talker_mapped_added_scale.clear();
        trailing_text_ids.clear();
        projected_token_num = 0;
        trailing_text_next_index = 0;
    }

    void ConditionalGeneration::map_text_ids(const GenerationConfig &gen_config, const std::vector<int> &input_ids, std::vector<int> &mapped_ids)
    {
        project_text_ids(gen_config, input_ids.data(), (int)input_ids.size());
        for (int i = 0; i < (int)input_ids.size(); i++, projected_token_num++)
        {
            mapped_ids.push_back(talker_vocab_size + projected_token_num);
        }
    }

    void ConditionalGeneration::project_text_ids(const GenerationConfig &gen_config, const int *input_ids, const int num)
    {
        std::vector<uint8_t> buf;
        if (num < 1) return;
        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        talker.project_text(gen_config, input_ids, num, ggml::type_of(emb->weight), buf);

        size_t offset = emb->get_base_nbytes() + projected_token_num * ggml::row_size(emb->weight);
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    void ConditionalGeneration::inject_text_ids(const GenerationConfig &gen_config, const int *input_ids, const int num, std::vector<int> &mapped_ids)
    {
        project_text_ids(gen_config, input_ids, num);

        for (int i = 0; i < num; i++, projected_token_num++)
        {
            mapped_ids.push_back(talker_vocab_size + projected_token_num);
        }
    }

    void ConditionalGeneration::inject_text_ids(const GenerationConfig &gen_config, const std::vector<int> &input_ids, std::vector<int> &mapped_ids)
    {
        inject_text_ids(gen_config, input_ids.data(), (int)input_ids.size(), mapped_ids);
    }

    void ConditionalGeneration::prepare_ids(const GenerationConfig &gen_config,
            const std::vector<int> &input_ids,
            const std::vector<int> &instruct_ids,
            const std::vector<int> &ref_ids,
            const std::string &voice_clone_prompt,
            const int language_id,
            const int speaker_id)
    {
        reset_token_ids();

        CHATLLM_CHECK(input_ids.size() >= 8);
        CHATLLM_CHECK(ref_ids.size() < 1) << "TODO: voice clone!";

        std::vector<int> mapped_ids;
        map_text_ids(gen_config, {codec_tok_ids.tts_bos_token_id, codec_tok_ids.tts_eos_token_id, codec_tok_ids.tts_pad_token_id}, mapped_ids);
        tts_bos_token_id = mapped_ids[0];
        tts_eos_token_id = mapped_ids[1];
        tts_pad_token_id = mapped_ids[2];

        inject_text_ids(gen_config, instruct_ids, talker_mapped_ids);

        // project _talker_input_embed_role: <|im_start|>assistant\n
        inject_text_ids(gen_config, input_ids.data(), 3, talker_mapped_ids);
        while (talker_mapped_added_ids.size() < talker_mapped_ids.size())
        {
            talker_mapped_added_ids.push_back(0);
            talker_mapped_added_scale.push_back(0.0f);
        }

        // codec_input_emebdding
        std::vector<int> codec_input_emebdding;
        {
            // codec_input_emebdding_0 (codec_prefill_list)
            if (language_id < 0)
            {
                codec_input_emebdding.push_back(codec_tok_ids.codec_nothink_id);
                codec_input_emebdding.push_back(codec_tok_ids.codec_think_bos_id);
                codec_input_emebdding.push_back(codec_tok_ids.codec_think_eos_id);
            }
            else
            {
                codec_input_emebdding.push_back(codec_tok_ids.codec_think_id);
                codec_input_emebdding.push_back(codec_tok_ids.codec_think_bos_id);
                codec_input_emebdding.push_back(language_id);
                codec_input_emebdding.push_back(codec_tok_ids.codec_think_eos_id);
            }

            // TODO: speaker_embed
            if (speaker_id >= 0)
            {
                codec_input_emebdding.push_back(speaker_id);
            }

            // codec_input_emebdding_1
            codec_input_emebdding.push_back(codec_tok_ids.codec_pad_id);
            codec_input_emebdding.push_back(codec_tok_ids.codec_bos_id);
        }

        std::vector<int> _talker_input_embed;
        for (int i = 0; i < (int)codec_input_emebdding.size() - 1; i++)
        {
            talker_mapped_ids.push_back(codec_input_emebdding[i]);
            talker_mapped_added_ids.push_back(tts_pad_token_id);
        }

        talker_mapped_added_ids.pop_back();
        talker_mapped_added_ids.push_back(tts_bos_token_id);

        inject_text_ids(gen_config, input_ids.data() + 3, 1, talker_mapped_ids);
        talker_mapped_added_ids.push_back(codec_input_emebdding.back());

        inject_text_ids(gen_config, input_ids.data() + 4, (int)input_ids.size() - 8, trailing_text_ids);
        trailing_text_ids.push_back(tts_eos_token_id);

        while (talker_mapped_added_scale.size() < talker_mapped_ids.size())
            talker_mapped_added_scale.push_back(1.0f);
    }

    int ConditionalGeneration::run_main_model(const GenerationConfig &gen_config, std::vector<float> &last_hidden_states, Sampler *sampler)
    {
        std::vector<float> logits;
        bool r = run_main_model(gen_config, logits, last_hidden_states);
        CHATLLM_CHECK(r);

        int id = sampler->sampling(logits.data(), talker_vocab_size);
        return id;
    }

    int ConditionalGeneration::run_main_model(const GenerationConfig &gen_config, const int added_id, const std::vector<int> &code_block, std::vector<float> &last_hidden_states, Sampler *sampler)
    {
        const int ids_count = (int)talker_mapped_ids.size();
        std::vector<float> logits;

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);

        std::vector<ggml::tensor *> input_ids;
        for (int i = 0; i < (int)code_block.size(); i++)
        {
            input_ids.push_back(ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, 1));
        }
        ggml::tensor *input_id_new          = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, 1);

        const auto custom_embedding = [this, input_ids, input_id_new, code_block](ComputeContext *ctx, ggml::tensor *input)
        {
            auto ids_main = input_ids[0];

            // last_id_hidden
            auto r = transformer->word_embeddings->forward(ctx, ids_main);

            for (int i = 1; i < (int)code_block.size(); i++)
            {
                auto ids = input_ids[i];
                auto emb = talker.code_predictor->model.get_codec_embedding(i - 1)->forward(ctx, ids);
                r        = ggml::add(ctx, r, emb);
            }

            auto pad_emb = transformer->word_embeddings->forward(ctx, input_id_new);
            auto output  = ggml::add(ctx, r, pad_emb);
            return output;
        };

        transformer->custom_embedding = custom_embedding;

        auto final_steps = dynamic_cast<LMFinalSteps *>(transformer->get_final_steps());

        ggml::tensor *r = transformer->forward(&ctx, nullptr, n_past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(ggml::type_of(r)                              == GGML_TYPE_F32) << "output type must be float: " << ggml::type_of(r);
        CHATLLM_CHECK(ggml::type_of(transformer->last_hidden_state) == GGML_TYPE_F32) << "output type must be float: " << ggml::type_of(transformer->last_hidden_state);

        logits.resize(ggml::nbytes(r) / sizeof(logits[0]));
        last_hidden_states.resize(ggml::nbytes(transformer->last_hidden_state) / sizeof(logits[0]));

        if (!ctx.allocate()) return false;

        for (int i = 0; i < (int)code_block.size(); i++)
        {
            Backend::write_tensor_data(input_ids[i],       &code_block[i]);
        }

        Backend::write_tensor_data(input_id_new,    &added_id);

        ctx.compute();

        Backend::read_tensor_data(r, logits.data());
        Backend::read_tensor_data(transformer->last_hidden_state, last_hidden_states.data());

        ctx.reset();
        n_past += 1;

        auto id = sampler->sampling(logits.data(), talker_vocab_size);
        return id;
    }

    bool ConditionalGeneration::run_main_model(const GenerationConfig &gen_config,
                            std::vector<float> &output, std::vector<float> &last_hidden_states)
    {
        const int ids_count = (int)talker_mapped_ids.size();

        CHATLLM_CHECK(talker_mapped_added_ids.size() == talker_mapped_ids.size());

        if (!initial_run)
        {
            initial_run = true;
            int past = gen_config.max_length / transformer->get_reserved_batch_size() - ids_count;
            if (past < 0) past = 0;
            if (!before_initial_run(ids_count, gen_config, past))
                return false;
        }

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);
        ggml::tensor *added_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);
        ggml::tensor *added_scale      = ggml::new_tensor_2d(&ctx, GGML_TYPE_F32, 1, ids_count);

        const auto custom_embedding = [this, added_ids_tensor, added_scale](ComputeContext *ctx, ggml::tensor *input)
        {
            auto emb = transformer->word_embeddings;
            auto output = emb->forward(ctx, input);
            auto added  = emb->forward(ctx, added_ids_tensor);
            added  = ggml::mul(ctx, added, added_scale);
            output = ggml::add(ctx, output, added);
            return output;
        };

        transformer->custom_embedding = custom_embedding;

        auto final_steps = dynamic_cast<LMFinalSteps *>(transformer->get_final_steps());

        ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, n_past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(ggml::type_of(r)                              == GGML_TYPE_F32) << "output type must be float: " << ggml::type_of(r);
        CHATLLM_CHECK(ggml::type_of(transformer->last_hidden_state) == GGML_TYPE_F32) << "output type must be float: " << ggml::type_of(transformer->last_hidden_state);

        output.resize(ggml::nbytes(r) / sizeof(output[0]));
        last_hidden_states.resize(ggml::nbytes(transformer->last_hidden_state) / sizeof(output[0]));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, talker_mapped_ids.data());
        Backend::write_tensor_data(added_ids_tensor, talker_mapped_added_ids.data());
        Backend::write_tensor_data(added_scale,      talker_mapped_added_scale.data());

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        ctx.compute();

        Backend::read_tensor_data(r, output.data());
        Backend::read_tensor_data(transformer->last_hidden_state, last_hidden_states.data());

        ctx.reset();
        n_past += ids_count;

        return true;
    }

    void ConditionalGeneration::test(const GenerationConfig &gen_config, std::vector<int16_t> &audio)
    {
        std::vector<int> codec_ids({1995, 1159, 355, 22, 1174, 1093, 625, 1814, 1058, 905, 1846, 1247, \
1677, 889, 812, 901, 1342, 353, 1713, 1919, 2011, 570, 446, 1267, \
361, 342, 294, 699, 732, 485, 2037, 927, 1766, 816, 410, 708, 260, \
446, 446, 466, 1067, 1542, 675, 173, 1239, 1518, 461, 1250, 872, \
1523, 1177, 1773, 1122, 1905, 6, 1455, 821, 1540, 1945, 347, 1482, \
887, 836, 419, 817, 603, 831, 1930, 707, 1280, 747, 465, 571, 840, \
265, 1462, 50, 661, 1402, 371, 1601, 1229, 1145, 986, 671, 1260, \
1299, 269, 840, 1046, 4, 980, 793, 339, 1296, 689, 1564, 294, 1307, \
1308, 292, 907, 523, 465, 630, 639, 694, 482, 287, 178, 1242, 1052, \
1890, 294, 820, 935, 602, 964, 1058, 1283, 947, 708, 1505, 518, 951, \
751, 1680, 203, 1398, 184, 1320, 229, 790, 1024, 1186, 1657, 1079, \
1248, 88, 131, 1629, 95, 1098, 1312, 1353, 452, 1499, 1569, 1598, \
1874, 1192, 935, 378, 679, 21, 110, 374, 1448, 703, 883, 1961, 844, \
49, 1948, 1402, 1939, 1600, 113, 1816, 1396, 1341, 1463, 1225, 1273, \
1581, 189});
        std::vector<float> pcm_samples;
        talker.speech_decode(gen_config, codec_ids, pcm_samples);

        for (size_t i = 0; i < pcm_samples.size(); i++)
            audio.push_back((int16_t)(pcm_samples[i] * 32767));
    }

    void ConditionalGeneration::generate_audio_codes(const GenerationConfig &gen_config,
            const std::vector<int> &input_ids,
            const std::vector<int> &instruct_ids,
            const std::vector<int> &ref_ids,
            const std::string &voice_clone_prompt,
            const int language_id,
            const int speaker_id,
            std::vector<int16_t> &audio)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        prepare_ids(gen_config, input_ids, instruct_ids, ref_ids, voice_clone_prompt, language_id, speaker_id);
        n_past = 0;
        n_past_offset = 0;

        std::vector<int>   block_ids;
        std::vector<float> last_id_hidden;
        std::vector<int>   codec_ids;
        std::unique_ptr<Sampler> sampler = std::unique_ptr<Sampler>(SamplerFactory::Create(gen_config));
        last_hidden_states.clear();
        last_id_hidden.resize(config.hidden_size);

        int last_id = -1;
        while ((gen_config.max_new_tokens <= 0) || (codec_ids.size() < (int)gen_config.max_new_tokens))
        {
            if (block_ids.size() > 0)
            {
                const int added_id = trailing_text_next_index < (int)trailing_text_ids.size() ? trailing_text_ids[trailing_text_next_index++] : tts_pad_token_id;
                last_id = run_main_model(gen_config, added_id, block_ids, last_hidden_states, sampler.get());
                block_ids.clear();
            }
            else
            {
                last_id = run_main_model(gen_config, last_hidden_states, sampler.get());
            }

            if (termination_token_ids.find(last_id) != termination_token_ids.end())
                break;
            if (last_id > talker.code_predictor->vocab_size)
                break;

            block_ids.push_back(last_id);

            auto emb = dynamic_cast<Embedding *>(transformer->word_embeddings);

            std::vector<uint8_t> buf;
            buf.resize(ggml::row_size(emb->weight));
            Backend::read_tensor_data(emb->weight, buf.data(), ggml::row_size(emb->weight) * last_id, ggml::row_size(emb->weight));
            ggml::to_float(ggml::type_of(emb->weight), buf.data(), last_id_hidden.data(), config.hidden_size, 1);

            talker.code_predict(gen_config, sampler.get(), block_ids, last_hidden_states.data(), last_id_hidden.data());

            codec_ids.insert(codec_ids.end(), block_ids.begin(), block_ids.end());
        }

        // decode
        std::vector<float> pcm_samples;
        talker.speech_decode(gen_config, codec_ids, pcm_samples);

        audio.resize(pcm_samples.size());
        for (size_t i = 0; i < pcm_samples.size(); i++)
            audio[i] = (int16_t)(32768 * pcm_samples[i]);
    }

    void ConditionalGeneration::speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        sample_rate = 24000;
        channels = 1;

        //test(gen_config, audio);
        //return;

        CHATLLM_CHECK(talker.loaded);

        std::vector<int> instruct_ids;
        std::vector<int> ref_ids;

        tok->encode_instruct(instruct_ids);

        int lang_id = codec_language_id.count(tok->lanuage) > 0 ? (codec_language_id.find(tok->lanuage))->second : -1;
        const int s_id    = spk_id.count(tok->speaker) > 0 ? (spk_id.find(tok->speaker))->second : -1;
        if ((tok->lanuage == "auto") || (tok->lanuage == "chinese"))
        {
            if (spk_dialect_id.count(tok->speaker) > 0)
                lang_id = (spk_dialect_id.find(tok->speaker))->second;
        }

        generate_audio_codes(gen_config, input_ids, instruct_ids,
            ref_ids, "",
            lang_id, s_id,
            audio);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(QWEN3_TTS,                    qwen::v3_tts, 1);
}