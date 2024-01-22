#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "chat.h"

namespace chatllm
{
    void inspect_tensor(ggml_tensor *tensor, const char *msg,
        ggml_tensor *temp1 = nullptr, ggml_tensor *temp2 = nullptr, ggml_tensor *temp3 = nullptr, ggml_tensor *temp4 = nullptr, ggml_tensor *temp5 = nullptr);

    enum ActFunc
    {
        GELU,   // equivelent to `gelu_new`
        SILU,
    };

    class Block
    {
    public:
        Block(): prec(ggml_prec::GGML_PREC_DEFAULT), id(0) {}
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
        virtual void set_ctx(int n_ctx) { }
        virtual void shift_cache(int shift, int total) { }

        virtual void set_prec(ggml_prec prec)
        {
            this->prec = prec;
        }

        virtual void set_id(int id)
        {
            this->id = id;
        }

        virtual int64_t get_param_num(bool effective_only) const
        {
            return 0;
        }
    protected:
        ggml_prec prec;
        int id;
    };

    class Identity : public Block
    {
    public:
        Identity() {}
        Identity(InitContext *ctx, int a) {}
        Identity(InitContext *ctx, int a, int b) {}

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override
        {
            return input;
        }

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input, int n_past) override
        {
            return input;
        }
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

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml_nelements(weight);
        }

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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ggml_nelements(weight);
            if (bias) r += ggml_nelements(bias);
            return r;
        }

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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ggml_nelements(weight);
            if (bias) r += ggml_nelements(bias);
            return r;
        }

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

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml_nelements(weight);
        }

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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += dense_h_to_4h.get_param_num(effective_only);
            r += dense_4h_to_h.get_param_num(effective_only);
            return r;
        }

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

        void set_ctx(int n_ctx) override { this->n_ctx = n_ctx; };
        void shift_cache(int shift, int total) override
        {
            shift_pending = ShiftPending(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += query_key_value.get_param_num(effective_only);
            r += dense.get_param_num(effective_only);
            return r;
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

        void set_ctx(int n_ctx) override { attention.set_ctx(n_ctx); };

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_layernorm.get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            r += post_attention_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += query_key_value.get_param_num(effective_only);
            r += dense.get_param_num(effective_only);
            return r;
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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += dense_h_to_4h.get_param_num(effective_only);
            r += dense_4h_to_h.get_param_num(effective_only);
            return r;
        }

    public:
        Linear dense_h_to_4h;
        Linear dense_4h_to_h;
    };

    template <class InputNormBlock,
              class AttentionBlock,
              class PostNormBlock,
              class MLPBlock> class LMBlock1 : public Block
    {
    public:
        LMBlock1() = default;
        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_layernorm.get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            r += post_attention_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Block::set_id(id);

            input_layernorm.set_id(id);
            attention.set_id(id);
            post_attention_layernorm.set_id(id);
            mlp.set_id(id);
        }

    public:
        InputNormBlock input_layernorm;
        AttentionBlock attention;
        PostNormBlock post_attention_layernorm;
        MLPBlock mlp;
    };

    class GLM2Block : public LMBlock1<RMSNorm, GLM2SelfAttention, RMSNorm, GLM2MLP>
    {
    public:
        GLM2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
                  int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length) {}
    };

    class BaseAttention : public Block
    {
    public:
        BaseAttention() : num_attention_heads(0) {}

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias,
                      ggml_type cache_type, int cache_length)
            : num_attention_heads(num_attention_heads),
              num_kv_heads(num_kv_heads),
              q_proj(ctx, hidden_size, hidden_size, nullptr, qkv_bias),
              k_proj(ctx, hidden_size, hidden_size / (num_attention_heads / num_kv_heads), nullptr, qkv_bias),
              v_proj(ctx, hidden_size, hidden_size / (num_attention_heads / num_kv_heads), nullptr, qkv_bias),
              o_proj(ctx, hidden_size, hidden_size, o_bias),
              k_cache(ggml_new_tensor_1d(ctx->gctx.get(), cache_type, hidden_size / (num_attention_heads / num_kv_heads) * cache_length)),
              v_cache(ggml_new_tensor_1d(ctx->gctx.get(), cache_type, hidden_size / (num_attention_heads / num_kv_heads) * cache_length)),
              pos(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, max_length)),
              max_length(max_length),
              shift_pending(),
              attn_scaling(true),
              cache_length(cache_length)
        {
            k_cache->data = new char[ggml_nbytes(k_cache)]();
            v_cache->data = new char[ggml_nbytes(v_cache)]();
            ggml_set_name(k_cache, "k_cache");
            ggml_set_name(v_cache, "v_cache");
            pos->data = new char[ggml_nbytes(pos)]();
        }

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias,
                            GGML_TYPE_F16, max_length)
        {
        }

        void shift_cache(int shift, int total) override
        {
            shift_pending = ShiftPending(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += q_proj.get_param_num(effective_only);
            r += k_proj.get_param_num(effective_only);
            r += v_proj.get_param_num(effective_only);
            r += o_proj.get_param_num(effective_only);
            return r;
        }

    protected:
        // input & output: [qlen, heads, head_size]
        virtual ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const { return k; };
        virtual ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const { return q; };

        //
        virtual ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const { return kq; }

        virtual void before_forward(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen);

        virtual void save_to_cache(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v);

        // output: [heads, qlen, head_size]
        virtual ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen);

        // output: [heads, head_size, klen]
        virtual ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen);

        virtual ggml_tensor *cross_attention(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml_tensor *q, ggml_tensor *k, ggml_tensor *v);

        // k: [heads, qlen, head_size]
        // q: [heads, qlen, head_size]
        // v: [heads, head_size, klen]
        virtual ggml_tensor *calc_attn_scores(ForwardContext *ctx, int hidden_size, const int n_past, const int qlen,
                                              ggml_tensor *key_layer, ggml_tensor *query_layer, ggml_tensor *value_layer);

    public:
        int num_attention_heads;
        int num_kv_heads;
        Linear q_proj, k_proj, v_proj;
        Linear o_proj;
        ggml_tensor *k_cache;
        ggml_tensor *v_cache;
        ggml_tensor *pos;
        int max_length;

    protected:
        ShiftPending shift_pending;
        bool attn_scaling;
        int cache_length;
    };

    void fill_pos_vector(ggml_tensor *pos, int n_past, int qlen);

    // TODO: Optimize this !!! (after ggml support matrix with ring buffer?)
    // qlen must be 1.
    // This is just a proof of concept.
    template <int sliding_window_len> class BaseSlidingWindowAttentionRingCache : public BaseAttention
    {
    public:
        BaseSlidingWindowAttentionRingCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias, GGML_TYPE_F16, sliding_window_len),
              cache_offset(0),
              indices(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, sliding_window_len))
        {
            indices->data = new char[ggml_nbytes(indices)];
        }

    protected:
        void before_forward(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen) override
        {
            fill_pos_vector(pos, n_past, qlen);

            // shift cache
            if (shift_pending.shift > 0)
            {
                cache_offset += shift_pending.shift;
                shift_pending.clear();
            }
        }

        void save_to_cache(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) override
        {
            GGML_ASSERT(cache_length > qlen);

            const int write_offset = (cache_offset + n_past) % cache_length;
            const int empty = cache_length - write_offset;
            const int write_len = empty >= qlen ? qlen : empty;
            const int remain = qlen - write_len;

            {
                struct ggml_tensor * k_cache_view = ggml_view_1d(ctx->gctx.get(), k_cache, write_len * kv_hidden_size,
                        ggml_element_size(k_cache) * kv_hidden_size * write_offset);

                struct ggml_tensor * v_cache_view = ggml_view_1d(ctx->gctx.get(), v_cache, write_len * kv_hidden_size,
                        ggml_element_size(v_cache) * kv_hidden_size * write_offset);

                struct ggml_tensor * k_view = ggml_view_1d(ctx->gctx.get(), k, write_len * kv_hidden_size, 0);
                struct ggml_tensor * v_view = ggml_view_1d(ctx->gctx.get(), v, write_len * kv_hidden_size, 0);

                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_view, k_cache_view));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_view, v_cache_view));
            }

            if (remain > 0)
            {
                struct ggml_tensor * k_cache_view = ggml_view_1d(ctx->gctx.get(), k_cache, remain * kv_hidden_size,
                                            0);

                struct ggml_tensor * v_cache_view = ggml_view_1d(ctx->gctx.get(), v_cache, remain * kv_hidden_size,
                                            0);

                struct ggml_tensor * k_view = ggml_view_1d(ctx->gctx.get(), k, remain * kv_hidden_size,
                        write_len * kv_hidden_size * ggml_element_size(k));

                struct ggml_tensor * v_view = ggml_view_1d(ctx->gctx.get(), v, remain * kv_hidden_size,
                        write_len * kv_hidden_size * ggml_element_size(v));

                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_view, k_cache_view));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_view, v_cache_view));
            }

            {
                const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;
                const int start = (cache_offset + n_past + qlen - total) % cache_length;

                int32_t *p = (int32_t *)indices->data;
                for (int i = 0; i < total; i++)
                    p[i] = (start + i) % cache_length;
            }
        }

        // output: [heads, qlen, head_size]
        ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            const int repeat = num_attention_heads / num_kv_heads;
            const int kv_hidden_size = hidden_size / repeat;

            const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;

            ggml_tensor *indices_view = ggml_view_1d(ctx->gctx.get(), indices, total, 0);
            ggml_tensor *k_cache_view = ggml_view_2d(ctx->gctx.get(), k_cache, kv_hidden_size, cache_length,
                                            kv_hidden_size * ggml_element_size(k_cache), 0);
            ggml_tensor *key_layer = ggml_get_rows(ctx->gctx.get(), k_cache_view, indices_view);

            key_layer = ggml_reshape_3d(ctx->gctx.get(), key_layer, head_size, num_kv_heads, total);  // [qlen, heads, head_size]
            key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3);                         // [heads, qlen, head_size]

            return key_layer;
        }

        // output: [heads, head_size, klen]
        ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            const int repeat = num_attention_heads / num_kv_heads;
            const int kv_hidden_size = hidden_size / repeat;

            const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;

            ggml_tensor *indices_view = ggml_view_1d(ctx->gctx.get(), indices, total, 0);
            ggml_tensor *v_cache_view = ggml_view_2d(ctx->gctx.get(), v_cache, kv_hidden_size, cache_length,
                                            kv_hidden_size * ggml_element_size(v_cache), 0);
            ggml_tensor *value_layer  = ggml_get_rows(ctx->gctx.get(), v_cache_view, indices_view);

            value_layer = ggml_reshape_3d(ctx->gctx.get(), value_layer, head_size, num_kv_heads, total);  // [qlen, heads, head_size]
            value_layer = ggml_permute(ctx->gctx.get(), value_layer, 1, 2, 0, 3);                         // [heads, head_size, klen]
            value_layer = ggml_cont(ctx->gctx.get(), value_layer);

            return value_layer;
        }

        int cache_offset;
        ggml_tensor *indices;
    };

    template <int sliding_window_len> class BaseSlidingWindowAttentionFullCache : public BaseAttention
    {
    public:
        BaseSlidingWindowAttentionFullCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias, GGML_TYPE_F16, max_length),
              indices(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, 1)) // to ensure number of tensors are the same
        {
        }

    protected:

        // output: [heads, qlen, head_size]
        ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            const int repeat = num_attention_heads / num_kv_heads;
            const int kv_hidden_size = hidden_size / repeat;
            // TODO: It seems that qlen must be 1.
            int64_t len = n_past + qlen;
            int64_t offset = 0;
            if (len > sliding_window_len)
            {
                offset = len - sliding_window_len;
                len = sliding_window_len;
            }

            ggml_tensor *key_layer = nullptr;

            key_layer = ggml_view_1d(ctx->gctx.get(), k_cache, len * kv_hidden_size, offset * kv_hidden_size * ggml_element_size(k_cache));
            key_layer = ggml_reshape_3d(ctx->gctx.get(), key_layer, head_size, num_kv_heads, len);  // [qlen, heads, head_size]
            key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3);                       // [heads, qlen, head_size]

            return key_layer;
        }

        // output: [heads, head_size, klen]
        ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            int64_t len = n_past + qlen;
            int64_t offset = 0;
            if (len > sliding_window_len)
            {
                offset = len - sliding_window_len;
                len = sliding_window_len;
            }

            ggml_tensor * value_layer = ggml_view_3d(ctx->gctx.get(),
                            v_cache,
                            len, head_size, num_kv_heads,
                            cache_length * ggml_element_size(v_cache),
                            cache_length * ggml_element_size(v_cache) * head_size,
                            offset * ggml_element_size(v_cache)); // [heads, head_size, klen]
            return value_layer;
        }

        ggml_tensor *indices;
    };

    template <class BaseAttn> class BaseSelfAttention : public BaseAttn
    {
    public:
        BaseSelfAttention() : BaseAttention() {}
        BaseSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        BaseSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttn(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias),
              freq_base(10000.0f),
              freq_scale(1.0f),
              ext_factor(0.0f),
              attn_factor(1.0f),
              beta_fast(0.0f),
              beta_slow(0.0f),
              rope_dim(hidden_size / num_attention_heads)
        {
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
        {
            const int hidden_size = hidden_states->ne[0];
            const int qlen = hidden_states->ne[1];
            const int repeat = BaseAttn::num_attention_heads / BaseAttn::num_kv_heads;
            const int kv_hidden_size = hidden_size / repeat;

            BaseAttn::before_forward(ctx, kv_hidden_size, n_past, qlen);

            ggml_tensor *tmpq = BaseAttn::q_proj.forward(ctx, hidden_states);
            ggml_tensor *tmpk = BaseAttn::k_proj.forward(ctx, hidden_states);
            ggml_tensor *tmpv = BaseAttn::v_proj.forward(ctx, hidden_states);

            ggml_mul_mat_set_prec(tmpk, BaseAttn::prec);
            ggml_mul_mat_set_prec(tmpq, BaseAttn::prec);
            ggml_mul_mat_set_prec(tmpv, BaseAttn::prec);

            ggml_tensor *attn = BaseAttn::cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

            ggml_tensor *attn_output = BaseAttn::o_proj.forward(ctx, attn);

            return attn_output;
        }

    public:
        // rope param
        float freq_base;
        float freq_scale;
        float ext_factor;
        float attn_factor;
        float beta_fast;
        float beta_slow;
        int   rope_dim;

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return ggml_rope_custom_inplace(ctx->gctx.get(), k, past, rope_dim, 0, 0, 0,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);    // [qlen, heads, head_size]
        }
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return ggml_rope_custom_inplace(ctx->gctx.get(), q, past, rope_dim, 0, 0, 0,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);    // [qlen, heads, head_size];
        }
    };

    template <bool bias> class InternLMSelfAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        InternLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, bias, bias) {}
    };

    // This is **the** feed forward network (Multi-Layer Perceptron) in _Attention Is All You Need_.
    class TheMLP : public Block
    {
    public:
        TheMLP() = default;
        TheMLP(InitContext *ctx, int hidden_size, int intermediate_size, ActFunc act, bool bias)
            : fc0(ctx, hidden_size, intermediate_size, bias),
              fc1(ctx, intermediate_size, hidden_size, bias),
              act(act)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

        void set_prec(ggml_prec prec) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += fc0.get_param_num(effective_only);
            r += fc1.get_param_num(effective_only);
            return r;
        }

    public:
        Linear fc0;
        Linear fc1;
        ActFunc act;
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

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += gate_proj.get_param_num(effective_only);
            r += down_proj.get_param_num(effective_only);
            r += up_proj.get_param_num(effective_only);
            return r;
        }

    public:
        Linear gate_proj;
        Linear down_proj;
        Linear up_proj;
    };

    // TODO: this looks more OOP or generic, but due to difficulties in building computation graph,
    //       MLP (BaseMLP) has to be embedded into SparseMoE (calculation is done here too).
    // template<class Expert> class SparseMoE : public Block

    template<int num_local_experts, int num_experts_per_tok> class SparseMoE : public Block
    {
    public:
        typedef BaseMLP Expert;
        SparseMoE() = default;
        SparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : gate(ctx, hidden_size, num_local_experts, false)
        {
            for (int i = 0; i < num_local_experts; i++)
            {
                experts.emplace_back(Expert(ctx, hidden_size, intermediate_size));
                expert_gates.push_back(experts[i].gate_proj.weight);
                expert_downs.push_back(experts[i].down_proj.weight);
                expert_ups.push_back(experts[i].up_proj.weight);
            }
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override
        {
            const int n_tokens = hidden_states->ne[1];
            const int n_expert = num_local_experts;

            ggml_context * ggctx = ctx->gctx.get();

            ggml_tensor * logits = gate.forward(ctx, hidden_states); // [n_tokens, num_experts]

            ggml_tensor * probs = ggml_soft_max(ggctx, logits); // [n_tokens, num_experts]

            // select experts
            ggml_tensor * selected_experts = ggml_top_k(ggctx, probs, num_experts_per_tok); // [n_tokens, num_experts_per_tok]

            ggml_tensor * weights = ggml_get_rows(ggctx,
                    ggml_reshape_3d(ggctx, probs, 1, n_expert, n_tokens), selected_experts);

            weights = ggml_reshape_2d(ggctx, weights, num_experts_per_tok, n_tokens); // [n_tokens, num_experts_per_tok]

            ggml_tensor * weights_sum = ggml_sum_rows(ggctx, weights);

            weights = ggml_div(ggctx, weights, weights_sum); // [n_tokens, num_experts_per_tok]

            // compute expert outputs
            ggml_tensor * moe_out = nullptr;

            for (int i = 0; i < num_experts_per_tok; i++)
            {
                ggml_tensor * cur_expert;

                ggml_tensor * cur_up = ggml_mul_mat_id(ggctx, expert_ups.data(), n_expert, selected_experts, i, hidden_states);

                ggml_tensor * cur_gate = ggml_mul_mat_id(ggctx, expert_gates.data(), n_expert, selected_experts, i, hidden_states);

                cur_gate = ggml_silu(ggctx, cur_gate);

                cur_expert = ggml_mul(ggctx, cur_up, cur_gate); // [n_tokens, n_embd]

                cur_expert = ggml_mul_mat_id(ggctx, expert_downs.data(), n_expert, selected_experts, i, cur_expert); // [n_tokens, n_embd]

                cur_expert = ggml_mul(ggctx, cur_expert,
                        ggml_view_2d(ggctx, weights, 1, n_tokens, weights->nb[1], i * weights->nb[0]));

                if (i == 0) {
                    moe_out = cur_expert;
                } else {
                    moe_out = ggml_add(ggctx, moe_out, cur_expert);
                }
            }

            return moe_out;
        }

        void set_prec(ggml_prec prec) override
        {
            gate.set_prec(prec);
            for (auto &expert : experts)
            {
                // TODO: At present, `Expert.forward` is implemented in `forward`.
                expert.set_prec(prec);
            }
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += gate.get_param_num(effective_only);
            r += experts[0].get_param_num(effective_only) *
                    (effective_only ? num_experts_per_tok : experts.size());
            return r;
        }

    public:
        Linear gate;
        std::vector<Expert> experts;
        std::vector<ggml_tensor *> expert_gates;
        std::vector<ggml_tensor *> expert_ups;
        std::vector<ggml_tensor *> expert_downs;
    };

    template <bool bias> class InternLMBlock : public LMBlock1<RMSNorm, InternLMSelfAttention<bias>, RMSNorm, BaseMLP>
    {
    public:
        InternLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, InternLMSelfAttention<bias>, RMSNorm, BaseMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
        InternLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : InternLMBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_attention_heads, max_length)
        {}
    };

    class LlamaSelfAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        LlamaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        LlamaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    class LlamaBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, BaseMLP>
    {
    public:
        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class Phi2MLP : public TheMLP
    {
    public:
        Phi2MLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : TheMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU, true)
        {
            // set_prec(ggml_prec::GGML_PREC_F32);
        }
    };

    template <class InputNormBlock,
              class AttentionBlock,
              class MLPBlock> class LMBlock2 : public Block
    {
    public:
        LMBlock2() = default;

        LMBlock2(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length, bool qkv_bias = true, bool o_bias = true)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias),
              mlp(ctx, hidden_size, intermediate_size) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
        {
            ggml_tensor *residual = ggml_dup(ctx->gctx.get(), hidden_states);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = input_layernorm.forward(ctx, hidden_states);

            // TODO: why `hidden_state` is overwritten somewhere?
            ggml_tensor *dup_hidden = ggml_dup(ctx->gctx.get(), hidden_states);
            ggml_build_forward_expand(ctx->gf, dup_hidden);

            ggml_tensor *attn_outputs = attention.forward(ctx, dup_hidden, n_past);
            ggml_tensor *feed_forward_hidden_states = mlp.forward(ctx, dup_hidden);

            // CAUTION: MEMORY REUSED BETWEEN LAYERS
            ggml_tensor *r = ggml_add_inplace(ctx->gctx.get(), feed_forward_hidden_states, residual);
            r = ggml_add_inplace(ctx->gctx.get(), r, attn_outputs);

            return r;
        }

        void shift_cache(int shift, int total) override
        {
            attention.shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_layernorm.get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

    public:
        InputNormBlock input_layernorm;
        AttentionBlock attention;
        MLPBlock mlp;
    };

    // Let's name it CrossAttention as in [`modeling_phi.py`](). Take it easy.
    class Phi2CrossAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        Phi2CrossAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias, bool o_bias)
            : Phi2CrossAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        Phi2CrossAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
            rope_dim = 32;
            attn_scaling = false;
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;
    };

    class Phi2Block : public LMBlock2<LayerNorm, Phi2CrossAttention, Phi2MLP>
    {
    public:
        Phi2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock2(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, true, true)
        {}
    };

    #define SlidingWindowAttentionImpl BaseSlidingWindowAttentionFullCache

    template <int sliding_window_len> class MistralSelfAttention : public BaseSelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        MistralSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseSelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        MistralSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseSelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    template<int num_local_experts, int num_experts_per_tok, int sliding_window_len> class MixtralBlock : public LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm,
                        SparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        MixtralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm,
                       SparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class BaichuanSelfAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        BaichuanSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaichuanSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        BaichuanSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;

        ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const override;

    };

    class BaichuanBlock : public LMBlock1<RMSNorm, BaichuanSelfAttention, RMSNorm, BaseMLP>
    {
    public:
        BaichuanBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        BaichuanBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class QWenSelfAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        QWenSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length);

        void config(int rope_dim, float rope_freq_base, float seq_length, bool use_dynamic_ntk, bool use_logn_attn);

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;

    public:
        int seq_length;
        bool use_dynamic_ntk;
        bool use_logn_attn;
    protected:
        ggml_tensor *logn_list;
    };

    class QWenBlock : public LMBlock1<RMSNorm, QWenSelfAttention, RMSNorm, BaseMLP>
    {
    public:
        QWenBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}
    };

    class BlueLMSelfAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        BlueLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BlueLMSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        BlueLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
              rope_scaling_factor(1.0),
              rope_scaling_power(0.0),
              cached_hidden_size(0)
        {
        }

        void config(float rope_theta, float rope_scaling_factor, float rope_scaling_power);

    public:
        float rope_scaling_factor;
        float rope_scaling_power;

        int cached_hidden_size;

        // length: rope_dim/2
        std::vector<float> inv_freq;

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;

        void build_inv_freq_if_needed(int hidden_size);
    };

    class BlueLMBlock : public LMBlock1<RMSNorm, BlueLMSelfAttention, RMSNorm, BaseMLP>
    {
    public:
        BlueLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        BlueLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template <int sliding_window_len> class MistralBlock : public LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, BaseMLP>
    {
    public:
        MistralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, BaseMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        MistralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, BaseMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class StableLMAttention : public BaseSelfAttention<BaseAttention>
    {
    public:
        StableLMAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : StableLMAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        StableLMAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;
    };

    class StableLMBlock : public LMBlock1<LayerNorm, StableLMAttention, LayerNorm, BaseMLP>
    {
    public:
        StableLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };
} // namespace chatllm
