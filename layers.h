#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "chat.h"

namespace chatllm
{

    class Block
    {
    public:
        Block() {}
        virtual ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input)
        {
            CHATLLM_THROW << "forward(ForwardContext *ctx, ggml_tensor *input): not implemented";
            return NULL;
        }
        virtual ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input, int n_past)
        {
            CHATLLM_THROW << "forward(ForwardContext *ctx, ggml_tensor *input, int n_past): not implemented";
            return NULL;
        }
        virtual void set_ctx(int n_ctx) const { }
        virtual void shift_cache(int shift, int total) { }
    };

    class ShiftPending
    {
    public:
        ShiftPending() : ShiftPending(0, 0) {}
        ShiftPending(int shift, int total) : shift(shift), total(total) {}
        void clear(void) { shift = 0; }
    public:
        int shift;
        int total;
    };

    class Embedding : public Block
    {
    public:
        Embedding() : weight(nullptr) {}
        Embedding(InitContext *ctx, int num_embeddings, int embedding_dim)
            : weight(ggml_new_tensor_2d(ctx->gctx.get(), ctx->dtype, embedding_dim, num_embeddings)) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override;

    public:
        ggml_tensor *weight;
    };

    class Linear : public Block
    {
    public:
        Linear() : weight(nullptr), bias(nullptr) {}
        Linear(InitContext *ctx, int in_features, int out_features, bool use_bias = true)
            : weight(ggml_new_tensor_2d(ctx->gctx.get(), ctx->dtype, in_features, out_features)),
              bias(use_bias ? ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, out_features) : nullptr) {}

        Linear(InitContext *ctx, int in_features, int out_features, ggml_tensor *weight, bool use_bias = true)
            : weight(weight != NULL ? weight : ggml_new_tensor_2d(ctx->gctx.get(), ctx->dtype, in_features, out_features)),
              bias(use_bias ? ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, out_features) : nullptr) {}

        int in_features() const { return weight->ne[0]; }
        int out_features() const { return weight->ne[1]; }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override;

    public:
        ggml_tensor *weight; // [out_features, in_features]
        ggml_tensor *bias;   // [out_features]
    };

    class LayerNorm : public Block
    {
    public:
        LayerNorm() : weight(nullptr), bias(nullptr) {}
        LayerNorm(InitContext *ctx, int normalized_shape)
            : weight(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, normalized_shape)),
              bias(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, normalized_shape)),
              eps(1e-5f) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override;

    public:
        ggml_tensor *weight; // [normalized_shape]
        ggml_tensor *bias;   // [normalized_shape]
        float eps;
    };

    class RMSNorm : public Block
    {
    public:
        RMSNorm() : weight(nullptr) {}
        RMSNorm(InitContext *ctx, int normalized_shape)
            : weight(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, normalized_shape)),
              eps(1e-5f) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override;

    public:
        ggml_tensor *weight;
        float eps;
    };

    class GLMMLP : public Block
    {
    public:
        GLMMLP() = default;
        GLMMLP(InitContext *ctx, int hidden_size)
            : dense_h_to_4h(ctx, hidden_size, 4 * hidden_size), dense_4h_to_h(ctx, 4 * hidden_size, hidden_size) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

    public:
        Linear dense_h_to_4h;
        Linear dense_4h_to_h;
    };

    class GLMSelfAttention : public Block
    {
    public:
        // TODO: kv cache type
        GLMSelfAttention() : num_attention_heads(0), n_ctx(0) {}
        GLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : query_key_value(ctx, hidden_size, 3 * hidden_size), dense(ctx, hidden_size, hidden_size),
              num_attention_heads(num_attention_heads),
              k_cache(ggml_new_tensor_3d(ctx->gctx.get(), GGML_TYPE_F16, hidden_size / num_attention_heads, max_length,
                                         num_attention_heads)),
              v_cache(ggml_new_tensor_3d(ctx->gctx.get(), GGML_TYPE_F16, max_length, hidden_size / num_attention_heads,
                                         num_attention_heads)),
              pos(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, max_length)),
              n_ctx(0),
              shift_pending()
        {
            k_cache->data = new char[ggml_nbytes(k_cache)];
            v_cache->data = new char[ggml_nbytes(v_cache)];
            pos->data = new char[ggml_nbytes(pos)]();
        }
        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        void set_ctx(int n_ctx) const override { n_ctx = n_ctx; };
        void shift_cache(int shift, int total) override
        {
            shift_pending = ShiftPending(shift, total);
        }

    public:
        Linear query_key_value;
        Linear dense;
        int num_attention_heads;
        ggml_tensor *k_cache; // [n_head, maxlen, head_size]
        ggml_tensor *v_cache; // [n_head, head_size, maxlen]
        ggml_tensor *pos;
        int n_ctx;
    private:
        ShiftPending shift_pending;
    };

    class GLMBlock : public Block
    {
    public:
        GLMBlock() : num_hidden_layers(0) {}
        GLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int num_hidden_layers, int max_length)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size),
              num_hidden_layers(num_hidden_layers) {}

        using Block::forward;
        virtual ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        void set_ctx(int n_ctx) const override { attention.set_ctx(n_ctx); };

    public:
        LayerNorm input_layernorm;
        GLMSelfAttention attention;
        LayerNorm post_attention_layernorm;
        GLMMLP mlp;
        int num_hidden_layers;
    };

    class GLM2SelfAttention : public Block
    {
    public:
        GLM2SelfAttention() : num_attention_heads(0), num_kv_heads(0) {}
        GLM2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : num_attention_heads(num_attention_heads), num_kv_heads(num_kv_heads),
              query_key_value(ctx, hidden_size, hidden_size + 2 * (hidden_size / num_attention_heads) * num_kv_heads),
              dense(ctx, hidden_size, hidden_size, false),
              k_cache(ggml_new_tensor_3d(ctx->gctx.get(), GGML_TYPE_F32, hidden_size / num_attention_heads, max_length,
                                         num_kv_heads)),
              v_cache(ggml_new_tensor_3d(ctx->gctx.get(), GGML_TYPE_F32, max_length, hidden_size / num_attention_heads,
                                         num_kv_heads)),
              pos(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, max_length)),
              shift_pending()
        {
            k_cache->data = new char[ggml_nbytes(k_cache)]();
            v_cache->data = new char[ggml_nbytes(v_cache)]();
            pos->data = new char[ggml_nbytes(pos)]();
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        void shift_cache(int shift, int total) override
        {
            shift_pending = ShiftPending(shift, total);
        }

    public:
        int num_attention_heads;
        int num_kv_heads;
        Linear query_key_value;
        Linear dense;
        ggml_tensor *k_cache; // [mqa_n_head, maxlen, head_size]
        ggml_tensor *v_cache; // [mqa_n_head, head_size, maxlen]
        ggml_tensor *pos;
    private:
        ShiftPending shift_pending;
    };

    class GLM2MLP : public Block
    {
    public:
        GLM2MLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : dense_h_to_4h(ctx, hidden_size, intermediate_size * 2, false),
              dense_4h_to_h(ctx, intermediate_size, hidden_size, false) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

    public:
        Linear dense_h_to_4h;
        Linear dense_4h_to_h;
    };

    // ChatGLM2 and InternLM share the same architecture within each layer
    template <class AttentionBlock, class MLPBlock> class LMBlock1 : public Block
    {
    public:
        LMBlock1() = default;
        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length, bool qkv_bias = true, bool o_bias = true)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
                  int max_length)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
        {
            ggml_tensor *residual = ggml_dup(ctx->gctx.get(), hidden_states);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = input_layernorm.forward(ctx, hidden_states);
            hidden_states = attention.forward(ctx, hidden_states, n_past);
            hidden_states = ggml_add_inplace(ctx->gctx.get(), hidden_states, residual);

            residual = ggml_dup(ctx->gctx.get(), hidden_states);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
            hidden_states = mlp.forward(ctx, hidden_states);
            hidden_states = ggml_add_inplace(ctx->gctx.get(), hidden_states, residual);

            return hidden_states;
        }

        void shift_cache(int shift, int total) override
        {
            attention.shift_cache(shift, total);
        }

    public:
        RMSNorm input_layernorm;
        AttentionBlock attention;
        RMSNorm post_attention_layernorm;
        MLPBlock mlp;
    };

    class GLM2Block : public LMBlock1<GLM2SelfAttention, GLM2MLP>
    {
    public:
        GLM2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
                  int max_length)
            : LMBlock1<GLM2SelfAttention, GLM2MLP>(ctx, hidden_size, num_attention_heads, num_kv_heads, intermediate_size, max_length) {}
    };

    class BaseSelfAttention : public Block
    {
    public:
        BaseSelfAttention() : num_attention_heads(0) {}
        BaseSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias = false, bool o_bias = false)
            : num_attention_heads(num_attention_heads),
              q_proj(ctx, hidden_size, hidden_size, nullptr, qkv_bias),
              k_proj(ctx, hidden_size, hidden_size, nullptr, qkv_bias),
              v_proj(ctx, hidden_size, hidden_size, nullptr, qkv_bias),
              o_proj(ctx, hidden_size, hidden_size, o_bias),
              k_cache(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F16, hidden_size * max_length)),
              v_cache(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F16, hidden_size * max_length)),
              pos(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, max_length)),
              max_length(max_length),
              freq_base(10000.0f),
              freq_scale(1.0f),
              ext_factor(0.0f),
              attn_factor(1.0f),
              beta_fast(0.0f),
              beta_slow(0.0f),
              shift_pending()
        {
            k_cache->data = new char[ggml_nbytes(k_cache)]();
            v_cache->data = new char[ggml_nbytes(v_cache)]();
            ggml_set_name(k_cache, "k_cache");
            ggml_set_name(v_cache, "v_cache");
            pos->data = new char[ggml_nbytes(pos)]();
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        void shift_cache(int shift, int total) override
        {
            shift_pending = ShiftPending(shift, total);
        }

    protected:
        // input & output: [qlen, heads, head_size]
        virtual ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const;
        virtual ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const;

        //
        virtual ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const { return kq; }

    public:
        int num_attention_heads;
        Linear q_proj, k_proj, v_proj;
        Linear o_proj;
        ggml_tensor *k_cache;
        ggml_tensor *v_cache;
        ggml_tensor *pos;
        int max_length;

        // rope param
        float freq_base;
        float freq_scale;
        float ext_factor;
        float attn_factor;
        float beta_fast;
        float beta_slow;

    private:
        ShiftPending shift_pending;
    };

    class InternLMSelfAttention : public BaseSelfAttention
    {
    public:
        InternLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, max_length, true, true) {}
    };

    class BaseMLP : public Block
    {
    public:
        BaseMLP() = default;
        BaseMLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : gate_proj(ctx, hidden_size, intermediate_size, false),
              down_proj(ctx, intermediate_size, hidden_size, false),
                up_proj(ctx, hidden_size, intermediate_size, false) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

    public:
        Linear gate_proj;
        Linear down_proj;
        Linear up_proj;
    };

    class InternLMBlock : public LMBlock1<InternLMSelfAttention, BaseMLP>
    {
    public:
        InternLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1<InternLMSelfAttention, BaseMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, max_length, false),
              max_length(max_length) {}
    private:
        int max_length;
    };

    class LlamaSelfAttention : public BaseSelfAttention
    {
    public:
        LlamaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, max_length, false, false) {}
    };

    class LlamaBlock : public LMBlock1<LlamaSelfAttention, BaseMLP>
    {
    public:
        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1<LlamaSelfAttention, BaseMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, max_length, false),
              max_length(max_length) {}
    private:
        int max_length;
    };
} // namespace chatllm
