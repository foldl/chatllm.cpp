#include "qwen.h"
#include <string.h>
#include "../src/vision_process.h"

namespace chatllm
{
    const int MODEL_TYPE_QWEN3_5 = MODEL_TYPE_QWEN2_5_VL + 3;
}

namespace chatllm::qwen::v3_5
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int attn_output_gate;
        int linear_conv_kernel_dim;
        int linear_key_head_dim;
        int linear_num_key_heads;
        int linear_num_value_heads;
        int linear_value_head_dim;
        int head_dim;
        float rope_theta;
        int rope_dim;
        int mrope_sections[v2_5_vl::MROPE_SECTION_MAX];
        int moe_intermediate_size;
        int shared_expert_intermediate_size;
        int num_experts_per_tok;
        int num_experts;
        int tie_word_embeddings;
        int mtp_num_hidden_layers;
        float router_aux_loss_coef;
        int layer_is_la[v3::MAX_LAYERS];
    };

    typedef v3_vl::Tokenizer Tokenizer;

    class Prelude
    {
    public:
        Prelude(const Config &config):
            pos_helper(config.max_length, config.vocab_size),
            helper_prelude(new TensorPosHelperPrelude(&pos_helper))
        {
            old_rms_eps = BlockParams::Epsilon::rms_norm;
            BlockParams::Epsilon::rms_norm = 1e-6f;
        }

        void done(void)
        {
            BlockParams::Epsilon::rms_norm = old_rms_eps;
            helper_prelude.reset(nullptr);
        }
    protected:
        float old_rms_eps = 0.0f;
        qwen::v2_5_vl::TensorPosHelper3D pos_helper;
        std::unique_ptr<TensorPosHelperPrelude> helper_prelude;
    };

    class ConditionalGeneration : public Prelude, public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = (ModelType)MODEL_TYPE_QWEN3_5, int vocab_size = -1);
        void load(ModelLoader &loader) override;
        void before_run_model(const int *input_ids, const int ids_count, const GenerationConfig &gen_config, int past) override;
    private:
        int get_sparse_layer_num();
        int get_la_layer_num();

        Block *create_layer(InitContext *ctx, int layer_index);

    public:
        const Config config;
    private:
        std::vector<v2_5_vl::ImageGridSize> images_grid;
        int token_time = 0;
    };

    class QwenGatedAttention : public QKNormedRoPEAttention<RMSNorm, BaseAttention>
    {
    public:
        typedef QKNormedRoPEAttention<RMSNorm, BaseAttention> Base;
        QwenGatedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length);
        int64_t get_param_num(bool effective_only) const override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;
        void load(const std::string &path, TensorLoader *loader) override;
    protected:
        ggml::tensor *cross_attention_after_pe(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *query_layer, ggml::tensor *key_layer, ggml::tensor *v) override;
    public:
        Linear gate_proj;
    private:
        ggml::tensor *g = nullptr;
    };

    class RMSNormGated : public RMSNorm
    {
    public:
        RMSNormGated(InitContext *ctx, int normalized_shape, float eps);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, ggml::tensor *gate) override;
    };

    class CausalConv1D : public Conv1D
    {
    public:
        CausalConv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride = 1,
               int dilation = 1, int groups = 1, bool bias = true);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;
    protected:
        ggml::tensor *state;
    };

    CausalConv1D::CausalConv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride,
        int dilation, int groups, bool bias):
        Conv1D(ctx, in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias),
        state(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, kernel_size - 1, in_channels))
    {
        ctx->get_allocator()->alloc(state);
    }

    ggml::tensor *CausalConv1D::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        const int kernel_size = ggml::get_dim(state, 0) + 1;

        if (n_past < 1)
        {
            auto clear = ggml::scale(ctx, state, 0.0f);
            ggml::build_forward_expand(ctx, clear);
        }

        input = ggml::concat(ctx, state, input, 0);
        {
            const int hist_size = kernel_size - 1;
            const int offset = ggml::get_dim(input, 0) - hist_size;
            auto hist = ggml::view_3d(ctx, input, hist_size, ggml::get_dim(input, 1), ggml::get_dim(input, 2),
                ggml::row_size(input),
                ggml::row_size(input) * ggml::get_dim(input, 1),
                ggml::element_size(input) * offset);
            auto v = ggml::cpy(ctx, hist, state);
            ggml::build_forward_expand(ctx, v);
        }

        auto out = Conv1D::forward(ctx, input);
        out = ggml::act(ctx, ActFunc::SILU, out);

        return out;
    }

    class QwenGatedDeltaNet : public Block
    {
    public:
        QwenGatedDeltaNet(InitContext *ctx, int hidden_size, int conv_kernel_dim, int num_key_heads, int num_value_heads, int key_head_dim, int value_head_dim);
        int64_t get_param_num(bool effective_only) const override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;
        void load(const std::string &path, TensorLoader *loader) override;
    protected:
        ggml::tensor *causal_conv1d_update(ComputeContext *ctx, ggml::tensor *input, int n_past);
        ggml::tensor *recurrent_gated_delta_rule(ComputeContext *ctx, ggml::tensor *query, ggml::tensor *key, ggml::tensor *value,
            ggml::tensor *g, ggml::tensor *beta, bool use_qk_l2norm_in_kernel, const int chunk_size = 64);
        ggml::tensor *calc_beta(ComputeContext *ctx, ggml::tensor *input);
        ggml::tensor *calc_g(ComputeContext *ctx, ggml::tensor *input);
    public:
        const int head_v_dim;
        const int head_k_dim;
        const int key_dim;
        const int value_dim;
        const int num_v_heads;
        const int num_k_heads;
        ggml::tensor       *state;
        CausalConv1D        conv1d;
        ggml::tensor       *dt_bias;
        ggml::tensor       *A_log;
        RMSNormGated        norm;
        Linear              out_proj;
        Linear              in_proj_qkv;
        Linear              in_proj_z;
        Linear              in_proj_b;
        Linear              in_proj_a;
    };

    RMSNormGated::RMSNormGated(InitContext *ctx, int normalized_shape, float eps):
        RMSNorm(ctx, normalized_shape)
    {
        this->eps = eps;
    }

    ggml::tensor *RMSNormGated::forward(ComputeContext *ctx, ggml::tensor *input, ggml::tensor *gate)
    {
        auto hidden_states = RMSNorm::forward(ctx, input);
        auto             w = ggml::act(ctx, ActFunc::SILU, gate);
        hidden_states      = ggml::mul(ctx, hidden_states, w);
        return hidden_states;
    }

    QwenGatedAttention::QwenGatedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
        Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false),
        gate_proj(ctx, hidden_size, head_dim * num_attention_heads, nullptr, false)
    {
    }

    ggml::tensor *QwenGatedAttention::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        g = gate_proj.forward(ctx, input);
        g = ggml::sigmoid(ctx, g);
        auto r = Base::forward(ctx, input, n_past);
        return r;
    }

    ggml::tensor *QwenGatedAttention::cross_attention_after_pe(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *query_layer, ggml::tensor *key_layer, ggml::tensor *v)
    {
        auto attn_scores = Base::cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);
        attn_scores = ggml::mul(ctx, attn_scores, g);
        return attn_scores;
    }

    int64_t QwenGatedAttention::get_param_num(bool effective_only) const
    {
        int64_t r = Base::get_param_num(effective_only);
        r += gate_proj.get_param_num(effective_only);
        return r;
    }

    void QwenGatedAttention::load(const std::string &path, TensorLoader *loader)
    {
        Base::load(path, loader);
        gate_proj.load(path + "gate_proj.", loader);
    }

    QwenGatedDeltaNet::QwenGatedDeltaNet(InitContext *ctx, int hidden_size, int conv_kernel_dim, int num_key_heads, int num_value_heads, int key_head_dim, int value_head_dim):
        head_v_dim(value_head_dim), head_k_dim(key_head_dim),
        key_dim(key_head_dim * num_key_heads),
        value_dim(value_head_dim * num_value_heads),
        num_v_heads(num_value_heads), num_k_heads(num_key_heads),
        state(ggml::new_tensor_3d(ctx, ggml::type::GGML_TYPE_F32, head_v_dim, head_k_dim, num_k_heads)), // PERFORMANCE: F32 to make scale/out_proj happy
        conv1d(ctx, key_dim * 2 + value_dim, key_dim * 2 + value_dim, conv_kernel_dim, 1, 1, key_dim * 2 + value_dim, false),
        dt_bias(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, num_value_heads)),
        A_log(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, num_value_heads)),
        norm(ctx, head_v_dim, 1e-6f),
        out_proj(ctx, value_dim, hidden_size, false),
        in_proj_qkv(ctx, hidden_size, key_dim * 2 + value_dim, false),
        in_proj_z(ctx, hidden_size, value_dim, false),
        in_proj_b(ctx, hidden_size, num_value_heads, false),
        in_proj_a(ctx, hidden_size, num_value_heads, false)
    {
        ctx->get_allocator()->alloc(state);
    }

    int64_t QwenGatedDeltaNet::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += ggml::nelements(dt_bias);
        r += ggml::nelements(A_log);
        r += conv1d.get_param_num(effective_only);
        r += norm.get_param_num(effective_only);
        r += out_proj.get_param_num(effective_only);
        r += in_proj_qkv.get_param_num(effective_only);
        r += in_proj_a.get_param_num(effective_only);
        r += in_proj_b.get_param_num(effective_only);
        r += in_proj_z.get_param_num(effective_only);
        return r;
    }

    ggml::tensor *QwenGatedDeltaNet::calc_beta(ComputeContext *ctx, ggml::tensor *input)
    {
        auto b = in_proj_b.forward(ctx, input);
        auto beta = ggml::sigmoid(ctx, b);
        return beta;
    }

    ggml::tensor *QwenGatedDeltaNet::calc_g(ComputeContext *ctx, ggml::tensor *input)
    {
        auto a = in_proj_a.forward(ctx, input);
        auto g = ggml::add(ctx, a, dt_bias);
        g = ggml::softplus(ctx, g);
        g = ggml::mul(ctx, g, ggml::exp(ctx, A_log));
        g = ggml::neg(ctx, g);

        return g;
    }

    // return: [hidden_size, qlen, ...]
    ggml::tensor *QwenGatedDeltaNet::causal_conv1d_update(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        auto mixed_qkv = in_proj_qkv.forward(ctx, input);
        mixed_qkv = ggml::permute(ctx, mixed_qkv, 1, 0);

        mixed_qkv = conv1d.forward(ctx, mixed_qkv, n_past);
        mixed_qkv = ggml::permute(ctx, mixed_qkv, 1, 0);

        return mixed_qkv;
    }

    ggml::tensor *QwenGatedDeltaNet::recurrent_gated_delta_rule(ComputeContext *ctx, ggml::tensor *query, ggml::tensor *key, ggml::tensor *value,
            ggml::tensor *g, ggml::tensor *beta, bool use_qk_l2norm_in_kernel, const int chunk_size)
    {
        CHATLLM_CHECK(ggml::n_dims(key) <= 3);

        if (use_qk_l2norm_in_kernel)
        {
            query = ggml::norm_p2(ctx, query, 1e-6f);
            key   = ggml::norm_p2(ctx, key,   1e-6f);
        }

        auto reshape = [](ComputeContext *ctx, ggml::tensor *x) {
            auto t = ggml::permute(ctx, x, 0, 2, 1);
            t = ggml::cont(ctx, t);
            return t;
        };

        auto get_t = [](ComputeContext *ctx, ggml::tensor *x, int t) {
            auto r = ggml::view_2d(ctx, x, ggml::get_dim(x, 0), ggml::get_dim(x, 2),
                ggml::row_size(x) * ggml::get_dim(x, 1),
                ggml::row_size(x) * t);
            return r;
        };

        query = reshape(ctx, query);
        key   = reshape(ctx, key);
        value = reshape(ctx, value);
        // beta  = reshape(ctx, beta);
        // g     = reshape(ctx, g);

        const int k_head_dim        = (int)ggml::get_dim(key, 0);
        const int sequence_length   = (int)ggml::get_dim(key, 1);
        const int num_heads         = (int)ggml::get_dim(key, 2);
        const int v_head_dim        = (int)ggml::get_dim(value, 0);

        const float scale = 1.0f / sqrtf((float)ggml::get_dim(query, 0));
        query = ggml::scale(ctx, query, scale);

        auto last_state = state;
        std::vector<ggml::tensor *> attn_outs;

        for (int i = 0; i < sequence_length; i++)
        {
            auto q_t    = get_t(ctx, query, i);
            auto k_t    = get_t(ctx, key,   i);
            auto v_t    = get_t(ctx, value, i);
            auto g_t    = get_t(ctx, g,     i);
            auto beta_t = get_t(ctx, beta,  i);

            g_t     = ggml::exp(ctx, g_t);
            g_t     = ggml::reshape(ctx, g_t, 1, 1, ggml::get_dim(g_t,    0), ggml::get_dim(g_t, 1));
            beta_t  = ggml::reshape(ctx, beta_t, 1, ggml::get_dim(beta_t, 0), ggml::get_dim(beta_t, 1));
            k_t     = ggml::reshape(ctx, k_t,    1, ggml::get_dim(k_t,    0), ggml::get_dim(k_t,    1), ggml::get_dim(k_t,    2));
            q_t     = ggml::reshape(ctx, q_t,    1, ggml::get_dim(q_t,    0), ggml::get_dim(q_t,    1), ggml::get_dim(q_t,    2));

            last_state   = ggml::mul(ctx, last_state,   g_t);
            auto kv_mem  = ggml::mul(ctx, last_state,   k_t);
            kv_mem       = ggml::sum(ctx, kv_mem,       1);
            auto delta   = ggml::sub(ctx, v_t,          kv_mem);
            delta        = ggml::mul(ctx, delta,        beta_t);
            delta        = ggml::reshape(ctx, delta,    ggml::get_dim(delta, 0), 1, ggml::get_dim(delta, 1), ggml::get_dim(delta, 2));
            k_t          = ggml::repeat(ctx, k_t, ggml::get_dim(delta, 0));
            delta        = ggml::mul(ctx, k_t,          delta);
            last_state   = ggml::add(ctx, last_state,   delta);

            auto attn_t  = ggml::mul(ctx, last_state,   q_t);
            attn_t       = ggml::sum(ctx, attn_t,       1);
            attn_t       = ggml::reshape(ctx, attn_t,   ggml::get_dim(attn_t, 0), 1, ggml::get_dim(attn_t, 1), ggml::get_dim(attn_t, 2));

            attn_outs.push_back(attn_t);
        }

        ggml::build_forward_expand(ctx, ggml::cpy(ctx, last_state, state));

        ggml::tensor *core_attn_out = ggml::concat(ctx, attn_outs, 1);
        core_attn_out = ggml::permute(ctx, core_attn_out, 0, 2, 1, 3);
        core_attn_out = ggml::cont(ctx, core_attn_out);

        return core_attn_out;
    }

    ggml::tensor *QwenGatedDeltaNet::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        const int qlen = ggml::get_dim(input, 1);

        if (n_past < 1)
        {
            auto cleared = ggml::scale_inplace(ctx, state, 0.0f);
            ggml::build_forward_expand(ctx, cleared);
        }

        // continuous tensor: [hidden_size, qlen, ...]
        auto mixed_qkv = causal_conv1d_update(ctx, input, n_past);
        mixed_qkv = ggml::cont(ctx, mixed_qkv);

        auto query = ggml::view_4d(ctx, mixed_qkv, key_dim, ggml::get_dim(mixed_qkv, 1), ggml::get_dim(mixed_qkv, 2), ggml::get_dim(mixed_qkv, 3),
            ggml::row_size(mixed_qkv),
            ggml::row_size(mixed_qkv) * ggml::get_dim(mixed_qkv, 1),
            ggml::row_size(mixed_qkv) * ggml::get_dim(mixed_qkv, 1) * ggml::get_dim(mixed_qkv, 2),
            0);
        auto key = ggml::view_4d(ctx, mixed_qkv, key_dim, ggml::get_dim(mixed_qkv, 1), ggml::get_dim(mixed_qkv, 2), ggml::get_dim(mixed_qkv, 3),
            ggml::row_size(mixed_qkv),
            ggml::row_size(mixed_qkv) * ggml::get_dim(mixed_qkv, 1),
            ggml::row_size(mixed_qkv) * ggml::get_dim(mixed_qkv, 1) * ggml::get_dim(mixed_qkv, 2),
            ggml::element_size(mixed_qkv) * key_dim);
        auto value = ggml::view_4d(ctx, mixed_qkv, value_dim, ggml::get_dim(mixed_qkv, 1), ggml::get_dim(mixed_qkv, 2), ggml::get_dim(mixed_qkv, 3),
            ggml::row_size(mixed_qkv),
            ggml::row_size(mixed_qkv) * ggml::get_dim(mixed_qkv, 1),
            ggml::row_size(mixed_qkv) * ggml::get_dim(mixed_qkv, 1) * ggml::get_dim(mixed_qkv, 2),
            ggml::element_size(mixed_qkv) * (key_dim + key_dim));

        query = ggml::cont(ctx, query);
        key   = ggml::cont(ctx, key);
        value = ggml::cont(ctx, value);

        // [head_dim, n_heads, qlen, ...]
        query = ggml::reshape(ctx, query, head_k_dim, ggml::get_dim(query, 0) / head_k_dim, ggml::get_dim(query, 1), ggml::get_dim(query, 2));
        key   = ggml::reshape(ctx, key,   head_k_dim, ggml::get_dim(key,   0) / head_k_dim, ggml::get_dim(key,   1), ggml::get_dim(key,   2));
        value = ggml::reshape(ctx, value, head_v_dim, ggml::get_dim(value, 0) / head_v_dim, ggml::get_dim(value, 1), ggml::get_dim(value, 2));

        const int repeat = num_v_heads / num_k_heads;
        key   = ggml::repeat_interleave(ctx, key,   repeat, 0);
        query = ggml::repeat_interleave(ctx, query, repeat, 0);

        auto beta = calc_beta(ctx, input);
        auto g    = calc_g   (ctx, input);

        auto core_attn_out = recurrent_gated_delta_rule(ctx, query, key, value,
            g, beta, true);

        // reshape input data into 2D tensor
        core_attn_out = ggml::reshape_2d(ctx, core_attn_out, head_v_dim, ggml::nelements(core_attn_out) / head_v_dim);

        // gating
        auto z = in_proj_z.forward(ctx, input);
        z = ggml::reshape(ctx, z, head_v_dim, ggml::nelements(z) / head_v_dim);
        core_attn_out = norm.forward(ctx, core_attn_out, z);
        core_attn_out = ggml::reshape(ctx, core_attn_out, ggml::nelements(core_attn_out) / qlen, qlen);

        auto output = out_proj.forward(ctx, core_attn_out);

        return output;
    }

    void QwenGatedDeltaNet::load(const std::string &path, TensorLoader *loader)
    {
        in_proj_qkv.load(path + "in_proj_qkv.", loader);
             conv1d.load(path + "conv1d.", loader);
               norm.load(path + "norm.", loader);
           out_proj.load(path + "out_proj.", loader);
          in_proj_a.load(path + "in_proj_a.", loader);
          in_proj_b.load(path + "in_proj_b.", loader);
          in_proj_z.load(path + "in_proj_z.", loader);
        loader->read_tensor(path + "dt_bias", dt_bias);
        loader->read_tensor(path + "A_log", A_log);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int vocab_size):
        Prelude(config),
        BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
        config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const int sparse_layers = get_sparse_layer_num();
        const int la_layers = get_la_layer_num();
        const int full_layers = config.num_hidden_layers - la_layers;
        const int dense_layers = config.num_hidden_layers - sparse_layers;
        const size_t num_tensors = 3 + (config.tie_word_embeddings ? -1 : 0)
                                    + config.num_hidden_layers * 2
                                    + dense_layers * 3
                                    + sparse_layers * 8
                                    + la_layers * 11
                                    + full_layers * 10;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
            vocab_size <= 0 ? create_embedding<Embedding>(&w_ctx_, config) : create_embedding<Embedding>(&w_ctx_, vocab_size, config.hidden_size),
            create_final_norm<RMSNorm>(&w_ctx_, config),
            config.tie_word_embeddings ? (Block *)nullptr : create_lm_head(&w_ctx_, config, false),
            [&](InitContext *ctx, int layer_index) {
                return create_layer(ctx, layer_index);
            });

        w_ctx_.check_used_mem_size(true);
        Prelude::done();

        batch_input = 5;
    }

    int ConditionalGeneration::get_sparse_layer_num()
    {
        return config.num_experts > 0 ? config.num_hidden_layers : 0;
    }
    int ConditionalGeneration::get_la_layer_num()
    {
        int r = 0;
        for (int i = 0; i < v3::MAX_LAYERS; i++)
            r += config.layer_is_la[i];
        return r;
    }

    Block *ConditionalGeneration::create_layer(InitContext *ctx, int layer_index)
    {
        CHATLLM_CHECK(config.num_experts < 0) << "TODO: MoE";

        if (config.layer_is_la[layer_index])
        {
            typedef LMBlock1<RMSNorm, QwenGatedDeltaNet, RMSNorm, SiLUMLP> Layer;
            auto layer = new Layer(ctx, TypeLinearAttention(), config.hidden_size, config.intermediate_size,
                config.linear_conv_kernel_dim, config.linear_num_key_heads, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim);
            return layer;
        }
        else
        {
            typedef LMBlock1<RMSNorm, QwenGatedAttention, RMSNorm, SiLUMLP> Layer;
            auto layer = new Layer(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);
            layer->attention.mrope_sections = config.mrope_sections;
            layer->attention.rope_mode      = RoPEMode::IMROPE;
            layer->attention.rope_dim       = config.rope_dim;
            layer->attention.freq_base      = config.rope_theta;
            return layer;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".self_attn.A_log",                    ".linear_attn.A_log"},
            {".self_attn.conv1d.",                  ".linear_attn.conv1d."},
            {".self_attn.dt_bias",                  ".linear_attn.dt_bias"},
            {".self_attn.in_proj_a.",               ".linear_attn.in_proj_a."},
            {".self_attn.in_proj_b.",               ".linear_attn.in_proj_b."},
            {".self_attn.in_proj_z.",               ".linear_attn.in_proj_z."},
            {".self_attn.in_proj_qkv.",             ".linear_attn.in_proj_qkv."},
            {".self_attn.norm.",                    ".linear_attn.norm."},
            {".self_attn.out_proj.",                ".linear_attn.out_proj."},
        });
        BaseModelForConditionalGeneration::load(loader);
    }

    void ConditionalGeneration::before_run_model(const int *input_ids, const int ids_count, const GenerationConfig &gen_config, int past)
    {
        const int image_id_start = config.vocab_size;
        const int length = ids_count;

        // TODO:
        int token_n_inc = 1;

        if ((past == 0) && (n_past_offset == 0))
            token_time = 0;

        token_time = pos_helper.build_3d_pos(input_ids, length,
            images_grid, image_id_start, token_n_inc, token_time);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(QWEN3_5,                          qwen::v3_5, 1);
}
