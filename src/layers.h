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
    [[ noreturn ]]
    void inspect_tensor(ggml_tensor *tensor, const char *msg,
        ggml_tensor *temp1 = nullptr, ggml_tensor *temp2 = nullptr, ggml_tensor *temp3 = nullptr, ggml_tensor *temp4 = nullptr, ggml_tensor *temp5 = nullptr);

    struct ggml_tensor * ggml_mul_mat_id(
        struct ggml_context * ctx,
        void  * as, int n_expert,
        struct ggml_tensor  * ids, int i,
        struct ggml_tensor  * b) ;

    struct alibi_ctx
    {
        int    n_past;
        int    n_head;
        float  bias_max;
    };

    enum ActFunc
    {
        GELU,   // equivelent to `gelu_new`
        SILU,
        Tanh,
        RELU,
        RELU2,  // square . relu
    };

    ggml_tensor *inplace_act(ggml_context *ctx, ActFunc act, ggml_tensor *input);

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

        virtual size_t get_cache_size(void) const { return 0; }
        virtual void  *set_cache_buffer(void *buffer) { return buffer; }
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

        Embedding(InitContext *ctx, int num_embeddings, int embedding_dim, int pos_max)
            : Embedding(ctx, num_embeddings, embedding_dim) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml_nelements(weight);
        }

    public:
        ggml_tensor *weight;
    };

    class VisualEmbedding  : public Embedding
    {
    public:
        VisualEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim)
            : Embedding(ctx, num_embeddings, embedding_dim) {}

        virtual ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *patches, int patches_per_row, ggml_tensor *text_input) = 0;
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

        int in_features() const { return (int)weight->ne[0]; }
        int out_features() const { return (int)weight->ne[1]; }

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

    class MultiLinear : public Block
    {
    public:
        MultiLinear() : weight(nullptr) {}
        MultiLinear(InitContext *ctx, int in_features, int out_features, int multi)
            : weight(ggml_new_tensor_3d(ctx->gctx.get(), ctx->dtype, in_features, out_features, multi)) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override { return nullptr; };

        virtual ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input, ggml_tensor *selected);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ggml_nelements(weight);
            return r;
        }

    public:
        ggml_tensor *weight; // [out_features, in_features, multi]
    };

    class LayerNorm : public Block
    {
    public:
        LayerNorm() : weight(nullptr), bias(nullptr) {}
        LayerNorm(InitContext *ctx, int normalized_shape)
            : LayerNorm(ctx, normalized_shape, true)
        {
        }

        LayerNorm(InitContext *ctx, int normalized_shape, bool use_bias)
            : weight(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, normalized_shape)),
              bias(use_bias ? ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, normalized_shape) : nullptr),
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

    class LayerNormNoBias : public LayerNorm
    {
    public:
        LayerNormNoBias() : LayerNorm() {}
        LayerNormNoBias(InitContext *ctx, int normalized_shape)
            : LayerNorm(ctx, normalized_shape, false)
        {
        }
    };

    class RobertaEmbedding : public Block
    {
    public:
        RobertaEmbedding() : word_weight(nullptr), position_weight(nullptr) {}
        RobertaEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim, int pos_max)
            : word_weight(ggml_new_tensor_2d(ctx->gctx.get(), ctx->dtype, embedding_dim, num_embeddings)),
              position_weight(ggml_new_tensor_2d(ctx->gctx.get(), ctx->dtype, embedding_dim, pos_max)),
              indices(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, pos_max)),
              ln(ctx, embedding_dim),
              pad_index(2)
        {
            indices->data = new char[ggml_nbytes(indices)];
            int32_t *p = (int32_t *)indices->data;
            for (int i = 0; i < pos_max; i++)
                p[i] = i;
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input, int n_past) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ln.get_param_num(effective_only);
            if (word_weight)        r += ggml_nelements(word_weight);
            if (position_weight)    r += ggml_nelements(position_weight);
            return r;
        }

    public:
        ggml_tensor *word_weight;
        ggml_tensor *position_weight;
        ggml_tensor *indices;
        LayerNorm    ln;
        int          pad_index;
    };

    class RMSNorm : public Block
    {
    public:
        RMSNorm() : weight(nullptr), inplace(true) {}
        RMSNorm(InitContext *ctx, int normalized_shape, bool inplace = true)
            : weight(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_F32, normalized_shape)),
              eps(1e-5f),
              inplace(inplace) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml_nelements(weight);
        }

    public:
        ggml_tensor *weight;
        float eps;
        const bool inplace;
    };

    class RMSNormNonInplace : public RMSNorm
    {
    public:
        RMSNormNonInplace(InitContext *ctx, int normalized_shape) : RMSNorm(ctx, normalized_shape, false) {}
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

    class GLMMLP : public TheMLP
    {
    public:
        GLMMLP() = default;
        GLMMLP(InitContext *ctx, int hidden_size)
            : GLMMLP(ctx, hidden_size, 4 * hidden_size) {}

        GLMMLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : TheMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU, true) {}
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
            pos->data = new char[ggml_nbytes(pos)]();
        }
        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        void set_ctx(int n_ctx) override { this->n_ctx = n_ctx; }
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

        size_t get_cache_size(void) const override
        {
            return ggml_nbytes(k_cache) + ggml_nbytes(v_cache);
        }

        void  *set_cache_buffer(void *buffer) override
        {
            uint8_t *b = (uint8_t *)buffer;
            k_cache->data = b; b += ggml_nbytes(k_cache);
            v_cache->data = b; b += ggml_nbytes(v_cache);
            return b;
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

        void set_ctx(int n_ctx) override { attention.set_ctx(n_ctx); }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_layernorm.get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            r += post_attention_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

        size_t get_cache_size(void) const override
        {
            return attention.get_cache_size();
        }

        void  *set_cache_buffer(void *buffer) override
        {
            return attention.set_cache_buffer(buffer);
        }

    public:
        LayerNorm input_layernorm;
        GLMSelfAttention attention;
        LayerNorm post_attention_layernorm;
        GLMMLP mlp;
        int num_hidden_layers;
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

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int head_dim, int max_length)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, mlp_intermediate_size1, mlp_intermediate_size2) {}


        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                        q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                        use_bias),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, mlp_intermediate_size1, mlp_intermediate_size2) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                        q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                        use_bias),
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

        size_t get_cache_size(void) const override
        {
            return attention.get_cache_size();
        }

        void  *set_cache_buffer(void *buffer) override
        {
            return attention.set_cache_buffer(buffer);
        }

    public:
        InputNormBlock input_layernorm;
        AttentionBlock attention;
        PostNormBlock post_attention_layernorm;
        MLPBlock mlp;
    };

    template <class PreAttnNormBlock,
              class AttentionBlock,
              class PostAttnNormBlock,
              class PreMLPNormBlock,
              class MLPBlock,
              class PostMLPNormBlock> class LMBlock4 : public Block
    {
    public:
        LMBlock4() = default;
        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : pre_attention_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length)
            : pre_attention_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int head_dim, int max_length)
            : pre_attention_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
        {
            ggml_tensor *residual = ggml_dup(ctx->gctx.get(), hidden_states);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = pre_attention_layernorm.forward(ctx, hidden_states);
            hidden_states = attention.forward(ctx, hidden_states, n_past);
            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);

            hidden_states = ggml_add_inplace(ctx->gctx.get(), hidden_states, residual);
            residual = ggml_cpy(ctx->gctx.get(), hidden_states, residual);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = pre_mlp_layernorm.forward(ctx, hidden_states);
            hidden_states = mlp.forward(ctx, hidden_states);
            hidden_states = post_mlp_layernorm.forward(ctx, hidden_states);

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
            r += pre_attention_layernorm.get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            r += post_attention_layernorm.get_param_num(effective_only);
            r += pre_mlp_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            r += post_mlp_layernorm.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Block::set_id(id);

            pre_attention_layernorm.set_id(id);
            attention.set_id(id);
            post_attention_layernorm.set_id(id);
            pre_mlp_layernorm.set_id(id);
            mlp.set_id(id);
            post_mlp_layernorm.set_id(id);
        }

        size_t get_cache_size(void) const override
        {
            return attention.get_cache_size();
        }

        void  *set_cache_buffer(void *buffer) override
        {
            return attention.set_cache_buffer(buffer);
        }

    public:
        PreAttnNormBlock pre_attention_layernorm;
        AttentionBlock attention;
        PostAttnNormBlock post_attention_layernorm;
        PreMLPNormBlock pre_mlp_layernorm;
        MLPBlock mlp;
        PostMLPNormBlock post_mlp_layernorm;
    };

    template <class InputNormBlock,
              class AttentionBlock,
              class PostNormBlock,
              class MLPBlock> class LMBlock3 : public LMBlock1<InputNormBlock, AttentionBlock, PostNormBlock, MLPBlock>
    {
    public:
        typedef LMBlock1<InputNormBlock, AttentionBlock, PostNormBlock, MLPBlock> Base;
        LMBlock3() = default;
        LMBlock3(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : Base::LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length),
              hidden_scaling(1.0f) {}

        LMBlock3(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length)
            : Base::LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length),
              hidden_scaling(1.0f) {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
        {
            ggml_context * ggctx = ctx->gctx.get();

            ggml_tensor *residual = ggml_dup(ggctx, hidden_states);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = Base::input_layernorm.forward(ctx, hidden_states);

            hidden_states = Base::attention.forward(ctx, hidden_states, n_past);
            hidden_states = ggml_scale_inplace(ggctx, hidden_states, hidden_scaling);
            hidden_states = ggml_add_inplace(ggctx, hidden_states, residual);

            residual = ggml_dup(ggctx, hidden_states);
            ggml_build_forward_expand(ctx->gf, residual);

            hidden_states = Base::post_attention_layernorm.forward(ctx, hidden_states);
            hidden_states = Base::mlp.forward(ctx, hidden_states);
            hidden_states = ggml_scale_inplace(ggctx, hidden_states, hidden_scaling);
            hidden_states = ggml_add_inplace(ggctx, hidden_states, residual);

            return hidden_states;
        }
    public:
        float hidden_scaling;
    };

    class CoreAttention : public Block
    {
    public:
        CoreAttention() : num_attention_heads(0), num_kv_heads(0), max_length(0) {}

        CoreAttention(InitContext *ctx, int num_attention_heads, int num_kv_heads, int max_length, ggml_type cache_type,
              int k_cache_ele_num, int v_cache_ele_num)
            : num_attention_heads(num_attention_heads),
              num_kv_heads(num_kv_heads),
              k_cache(k_cache_ele_num > 0 ? ggml_new_tensor_1d(ctx->gctx.get(), cache_type, k_cache_ele_num)
                                       : nullptr),
              v_cache(v_cache_ele_num > 0 ? ggml_new_tensor_1d(ctx->gctx.get(), cache_type, v_cache_ele_num)
                                       : nullptr),
              pos(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, max_length)),
              max_length(max_length),
              attn_scaling_factor(-1.0f),
              attn_scores_pp(nullptr),
              shift_pending(),
              attn_scaling(true),
              causal(true),
              last_attn_scores(nullptr)
        {
            if (k_cache_ele_num > 0)
            {
                ggml_set_name(k_cache, "k_cache");
            }
            if (v_cache_ele_num > 0)
            {
                ggml_set_name(v_cache, "v_cache");
            }
            pos->data = new char[ggml_nbytes(pos)]();
        }

        void shift_cache(int shift, int total) override
        {
            shift_pending = ShiftPending(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            if (attn_scores_pp)
                r += attn_scores_pp->get_param_num(effective_only);
            return r;
        }

        size_t get_cache_size(void) const override
        {
            size_t r = 0;
            if (k_cache)
                r += ggml_nbytes(k_cache);
            if (v_cache)
                r += ggml_nbytes(v_cache);
            return r;
        }

        void  *set_cache_buffer(void *buffer) override
        {
            uint8_t *b = (uint8_t *)buffer;
            if (k_cache)
            {
                k_cache->data = b;
                b += ggml_nbytes(k_cache);
            }
            if (v_cache)
            {
                v_cache->data = b;
                b += ggml_nbytes(v_cache);
            }
            return b;
        }

    protected:
        // k: [heads, qlen, head_size]
        // q: [heads, qlen, head_size]
        // v: [heads, head_size, klen]
        virtual ggml_tensor *calc_attn_scores(ForwardContext *ctx, int hidden_size, const int n_past, const int qlen,
                                              ggml_tensor *key_layer, ggml_tensor *query_layer, ggml_tensor *value_layer);

        // input & output: [qlen, heads, head_size]
        virtual ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const { return k; }
        virtual ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const { return q; }
        virtual ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const { return kq; }

        virtual void before_forward(ForwardContext *ctx, const int n_past, const int qlen);

        // k: [qlen, heads, head_size]
        // v: [qlen, hidden_size]
        virtual void save_to_cache(ForwardContext *ctx, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) = 0;

        // output: [heads, qlen, head_size]
        virtual ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) = 0;

        // output: [heads, head_size, klen]
        virtual ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) = 0;

        virtual ggml_tensor *cross_attention_after_pe(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml_tensor *query_layer, ggml_tensor *key_layer, ggml_tensor *v);

        virtual ggml_tensor *cross_attention(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml_tensor *q, ggml_tensor *k, ggml_tensor *v);

        // q & k: [qlen, heads, head_size]
        virtual ggml_tensor *cross_attention_3d(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml_tensor *query_layer, ggml_tensor *key_layer, ggml_tensor *v);

        ggml_tensor *get_last_attn_scores(void)
        {
            return last_attn_scores;
        }

    public:
        const int num_attention_heads;
        const int num_kv_heads;
        ggml_tensor *k_cache;
        ggml_tensor *v_cache;
        ggml_tensor *pos;
        const int max_length;
        float attn_scaling_factor;
        Block *attn_scores_pp;

    protected:
        ShiftPending shift_pending;
        bool attn_scaling;
        bool causal;
        ggml_tensor *last_attn_scores;
    };

    class KVCacheAttention : public CoreAttention
    {
    public:
        KVCacheAttention() : CoreAttention(), k_hidden_size(0), v_hidden_size(0), cache_length(0) {}

        KVCacheAttention(InitContext *ctx, int num_attention_heads, int num_kv_heads, int k_hidden_size, int v_hidden_size, int max_length,
                         ggml_type cache_type, int cache_length)
            : CoreAttention(ctx, num_attention_heads, num_kv_heads, max_length, cache_type,
                            k_hidden_size * cache_length,
                            v_hidden_size * cache_length),
              k_hidden_size(k_hidden_size),
              v_hidden_size(v_hidden_size),
              cache_length(cache_length)
        {
        }

    protected:
        virtual void before_forward(ForwardContext *ctx, const int n_past, const int qlen);

        // k: [qlen, heads, head_size]
        // v: [qlen, hidden_size]
        virtual void save_to_cache(ForwardContext *ctx, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v);

        // output: [heads, qlen, head_size]
        virtual ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen);

        // output: [heads, head_size, klen]
        virtual ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen);

    public:
        const int k_hidden_size;
        const int v_hidden_size;
        const int cache_length;
    };

    class BaseConsolidatedQKVAttention : public KVCacheAttention
    {
    public:
        BaseConsolidatedQKVAttention() : KVCacheAttention() {}

        BaseConsolidatedQKVAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseConsolidatedQKVAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, GGML_TYPE_F16, max_length)
        {}

        BaseConsolidatedQKVAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias,
                                     ggml_type cache_type, int cache_length)
            : KVCacheAttention(ctx, num_attention_heads, num_kv_heads,
                               head_dim * num_kv_heads,
                               head_dim * num_kv_heads,
                               max_length, cache_type, cache_length),
              query_key_value(ctx, hidden_size, hidden_size + 2 * (hidden_size / num_attention_heads) * num_kv_heads, qkv_bias),
              dense(ctx, hidden_size, hidden_size, o_bias)
        {
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = KVCacheAttention::get_param_num(effective_only);
            r += query_key_value.get_param_num(effective_only);
            r += dense.get_param_num(effective_only);
            return r;
        }

    public:
        Linear query_key_value;
        Linear dense;
    };

    class BaseAttention : public KVCacheAttention
    {
    public:
        BaseAttention() : KVCacheAttention() {}

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
                      bool qkv_bias, bool o_bias,
                      ggml_type cache_type, int cache_length)
            : KVCacheAttention(ctx, num_attention_heads, num_kv_heads, head_dim * num_kv_heads, head_dim * num_kv_heads, max_length, cache_type, cache_length),
              q_proj(ctx, hidden_size, head_dim * num_attention_heads, nullptr, qkv_bias),
              k_proj(ctx, hidden_size, head_dim * num_kv_heads, nullptr, qkv_bias),
              v_proj(ctx, hidden_size, head_dim * num_kv_heads, nullptr, qkv_bias),
              o_proj(ctx, head_dim * num_attention_heads, hidden_size, o_bias)
        {
        }

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      bool qkv_bias, bool o_bias,
                      ggml_type cache_type, int cache_length)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            cache_type, cache_length)
        {}

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            GGML_TYPE_F16, max_length)
        {}

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
             bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias,
                            GGML_TYPE_F16, max_length)
        {}

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = KVCacheAttention::get_param_num(effective_only);
            r += q_proj.get_param_num(effective_only);
            r += k_proj.get_param_num(effective_only);
            r += v_proj.get_param_num(effective_only);
            r += o_proj.get_param_num(effective_only);
            return r;
        }

        void set_prec(ggml_prec prec) override;

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

    public:
        Linear q_proj, k_proj, v_proj;
        Linear o_proj;
    };

    class BaseCachelessAttention : public BaseAttention
    {
    public:
        BaseCachelessAttention() = default;

        BaseCachelessAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
        {}

        BaseCachelessAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, GGML_TYPE_F16, 0),
              raw_k(nullptr),
              raw_v(nullptr)
        {}

    protected:
        void save_to_cache(ForwardContext *ctx, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) override;

        // output: [heads, qlen, head_size]
        ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override;

        // output: [heads, head_size, klen]
        ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override;
    private:
        ggml_tensor *raw_k;
        ggml_tensor *raw_v;
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
        void before_forward(ForwardContext *ctx, const int n_past, const int qlen) override
        {
            if (n_past == 0) cache_offset = 0;

            fill_pos_vector(pos, n_past, qlen);

            // shift cache
            if (shift_pending.shift > 0)
            {
                cache_offset += shift_pending.shift;
                shift_pending.clear();
            }
        }

        void save_to_cache(ForwardContext *ctx, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) override
        {
            GGML_ASSERT(cache_length > qlen);

            const int write_offset = (cache_offset + n_past) % cache_length;
            const int empty = cache_length - write_offset;
            const int write_len = empty >= qlen ? qlen : empty;
            const int remain = qlen - write_len;

            {
                struct ggml_tensor * k_cache_view = ggml_view_1d(ctx->gctx.get(), k_cache, write_len * k_hidden_size,
                        ggml_element_size(k_cache) * k_hidden_size * write_offset);

                struct ggml_tensor * v_cache_view = ggml_view_1d(ctx->gctx.get(), v_cache, write_len * v_hidden_size,
                        ggml_element_size(v_cache) * v_hidden_size * write_offset);

                struct ggml_tensor * k_view = ggml_view_1d(ctx->gctx.get(), k, write_len * k_hidden_size, 0);
                struct ggml_tensor * v_view = ggml_view_1d(ctx->gctx.get(), v, write_len * v_hidden_size, 0);

                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_view, k_cache_view));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_view, v_cache_view));
            }

            if (remain > 0)
            {
                struct ggml_tensor * k_cache_view = ggml_view_1d(ctx->gctx.get(), k_cache, remain * k_hidden_size,
                                            0);

                struct ggml_tensor * v_cache_view = ggml_view_1d(ctx->gctx.get(), v_cache, remain * v_hidden_size,
                                            0);

                struct ggml_tensor * k_view = ggml_view_1d(ctx->gctx.get(), k, remain * k_hidden_size,
                        write_len * k_hidden_size * ggml_element_size(k));

                struct ggml_tensor * v_view = ggml_view_1d(ctx->gctx.get(), v, remain * v_hidden_size,
                        write_len * v_hidden_size * ggml_element_size(v));

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

            const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;

            ggml_tensor *indices_view = ggml_view_1d(ctx->gctx.get(), indices, total, 0);
            ggml_tensor *k_cache_view = ggml_view_2d(ctx->gctx.get(), k_cache, k_hidden_size, cache_length,
                                            k_hidden_size * ggml_element_size(k_cache), 0);
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

            const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;

            ggml_tensor *indices_view = ggml_view_1d(ctx->gctx.get(), indices, total, 0);
            ggml_tensor *v_cache_view = ggml_view_2d(ctx->gctx.get(), v_cache, v_hidden_size, cache_length,
                                            v_hidden_size * ggml_element_size(v_cache), 0);
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
            : BaseSlidingWindowAttentionFullCache(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        BaseSlidingWindowAttentionFullCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, GGML_TYPE_F16, max_length),
              indices(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, 1)) // to ensure number of tensors are the same
        {
        }

    protected:

        // output: [heads, qlen, head_size]
        ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            const int repeat = num_attention_heads / num_kv_heads;
            // Note: `qlen` must be 1.
            int64_t len = n_past + qlen;
            int64_t offset = 0;
            if (len > sliding_window_len)
            {
                offset = len - sliding_window_len;
                len = sliding_window_len;
            }

            ggml_tensor *key_layer = nullptr;

            key_layer = ggml_view_1d(ctx->gctx.get(), k_cache, len * k_hidden_size, offset * k_hidden_size * ggml_element_size(k_cache));
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

    // TODO: debug this
    template <int sliding_window_len, int extra_len = 512> class BaseSlidingWindowAttentionPartialCache : public BaseAttention
    {
    public:
        BaseSlidingWindowAttentionPartialCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSlidingWindowAttentionPartialCache(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        BaseSlidingWindowAttentionPartialCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, GGML_TYPE_F16, sliding_window_len + extra_len),
              indices(ggml_new_tensor_1d(ctx->gctx.get(), GGML_TYPE_I32, 1)), // to ensure number of tensors are the same
              cache_offset(0)
        {
        }

    protected:

        void before_forward(ForwardContext *ctx, const int n_past, const int qlen) override
        {
            if (n_past == 0) cache_offset = 0;

            fill_pos_vector(pos, n_past, qlen);

            // shift cache
            if (shift_pending.shift > 0)
            {
                // do nothing
                shift_pending.clear();
            }
        }

        void save_to_cache(ForwardContext *ctx, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) override
        {
            CHATLLM_CHECK(qlen == 1)  << "qlen must be 1";

            {
                const int write_offset = cache_offset + n_past;
                const int empty = cache_length - write_offset;

                if (empty < qlen)
                {
                    int remain = sliding_window_len - qlen;
                    int shift = cache_length - remain;

                    struct ggml_tensor * k_cache_remain = ggml_view_1d(ctx->gctx.get(), k_cache, remain * k_hidden_size,
                                                ggml_element_size(k_cache) * k_hidden_size * shift);
                    struct ggml_tensor * k_cache_1d = ggml_view_1d(ctx->gctx.get(), k_cache, remain * k_hidden_size,
                                                0);

                    struct ggml_tensor * k_remain_dup = ggml_dup(ctx->gctx.get(), k_cache_remain);

                    struct ggml_tensor * v_cache_remain = ggml_view_2d(ctx->gctx.get(), v_cache, remain, v_hidden_size,
                                                cache_length * ggml_element_size(v_cache),
                                                shift * ggml_element_size(v_cache));
                    struct ggml_tensor * v_cache_2d =     ggml_view_2d(ctx->gctx.get(), v_cache, remain, v_hidden_size,
                                                cache_length * ggml_element_size(v_cache),
                                                0);

                    struct ggml_tensor * v_remain_dup = ggml_dup(ctx->gctx.get(), v_cache_remain);

                    ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_remain_dup, k_cache_1d));
                    ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_remain_dup, v_cache_2d));

                    cache_offset -= shift;
                }
            }

            const int write_offset = cache_offset + n_past;

            // compute the transposed [N, n_embd] V matrix
            struct ggml_tensor * Vcur = ggml_transpose(ctx->gctx.get(), v); // ggml_reshape_2d(ctx->gctx.get(), tmpv, kv_hidden_size, qlen));

            struct ggml_tensor * k_cache_view = ggml_view_1d(ctx->gctx.get(), k_cache, qlen * k_hidden_size,
                                        ggml_element_size(k_cache) * k_hidden_size * write_offset);

            struct ggml_tensor * v_cache_view = ggml_view_2d(ctx->gctx.get(), v_cache, qlen, v_hidden_size,
                    cache_length * ggml_element_size(v_cache), write_offset * ggml_element_size(v_cache));

            struct ggml_tensor * k_view = ggml_view_1d(ctx->gctx.get(), k, qlen * k_hidden_size, 0);

            // important: storing RoPE-ed version of K in the KV cache!
            ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_view, k_cache_view));
            ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), Vcur, v_cache_view));
        }

        // output: [heads, qlen, head_size]
        ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            int64_t len = n_past + qlen;
            if (len > sliding_window_len)
                len = sliding_window_len;
            int64_t offset = cache_offset + n_past + qlen - len;

            CHATLLM_CHECK(offset >= 0) << "offset must >= 0";

            ggml_tensor *key_layer = nullptr;

            key_layer = ggml_view_1d(ctx->gctx.get(), k_cache, len * k_hidden_size, offset * k_hidden_size * ggml_element_size(k_cache));
            key_layer = ggml_reshape_3d(ctx->gctx.get(), key_layer, head_size, num_kv_heads, len);  // [qlen, heads, head_size]
            key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3);                       // [heads, qlen, head_size]

            return key_layer;
        }

        // output: [heads, head_size, klen]
        ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            int64_t len = n_past + qlen;
            if (len > sliding_window_len)
                len = sliding_window_len;
            int64_t offset = cache_offset + n_past + qlen - len;

            ggml_tensor * value_layer = ggml_view_3d(ctx->gctx.get(),
                            v_cache,
                            len, head_size, num_kv_heads,
                            cache_length * ggml_element_size(v_cache),
                            cache_length * ggml_element_size(v_cache) * head_size,
                            offset * ggml_element_size(v_cache)); // [heads, head_size, klen]
            return value_layer;
        }

        ggml_tensor *indices;
        int cache_offset;
    };

    enum RoPEMode
    {
        Interleaved = 0,        // IQIQ......IQ
        Original = 2,           // II...IQQ...Q
        GLM = 4,
    };

    template <class BaseAttn> class RoPESelfAttention : public BaseAttn
    {
    public:
        RoPESelfAttention() : BaseAttention() {}
        RoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        RoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads,  max_length, qkv_bias, o_bias)
        {
        }

        RoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttn(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias),
              freq_base(10000.0f),
              freq_scale(1.0f),
              ext_factor(0.0f),
              attn_factor(1.0f),
              beta_fast(0.0f),
              beta_slow(0.0f),
              rope_dim(head_dim),
              rope_mode(RoPEMode::Interleaved)
        {
        }

        RoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias)
            : BaseAttn(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                      use_bias),
              freq_base(10000.0f),
              freq_scale(1.0f),
              ext_factor(0.0f),
              attn_factor(1.0f),
              beta_fast(0.0f),
              beta_slow(0.0f),
              rope_dim(rope_dim),
              n_ctx(0),
              n_original_ctx(0),
              rope_mode(RoPEMode::Original)
        {
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
        int   n_ctx;
        int   n_original_ctx;
        RoPEMode rope_mode;

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return ggml_rope_ext_inplace(ctx->gctx.get(), k, past, nullptr, rope_dim, rope_mode, n_original_ctx,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);    // [qlen, heads, head_size]
        }
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return ggml_rope_ext_inplace(ctx->gctx.get(), q, past, nullptr, rope_dim, rope_mode, n_original_ctx,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);    // [qlen, heads, head_size];
        }
    };

    class GLM2SelfAttention : public RoPESelfAttention<BaseConsolidatedQKVAttention>
    {
    public:
        GLM2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, false)
        {
            rope_dim = (hidden_size / num_attention_heads) / 2;
        }
    };

    class GLM2Block : public LMBlock1<RMSNorm, GLM2SelfAttention, RMSNorm, GLM2MLP>
    {
    public:
        GLM2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
                  int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length) {}
    };

    template <bool bias> class InternLMSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        InternLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, bias, bias) {}
    };

    class BaseMLP : public Block
    {
    public:
        BaseMLP() = default;
        BaseMLP(InitContext *ctx, int hidden_size, int intermediate_size, ActFunc act)
            : gate_proj(ctx, hidden_size, intermediate_size, false),
              down_proj(ctx, intermediate_size, hidden_size, false),
                up_proj(ctx, hidden_size, intermediate_size, false),
              act(act)
        {}

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
        const ActFunc act;
    };

    class SiLUMLP : public BaseMLP
    {
    public:
        SiLUMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::SILU)
        {}
    };

    class GELUMLP : public BaseMLP
    {
    public:
        GELUMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU)
        {}
    };

    void ggml_compute_forward_sigmoid(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata);

    template<class MLP, bool use_bias = false> class GatedMLP : public MLP
    {
    public:
        GatedMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  MLP(ctx, hidden_size, intermediate_size),
           gate(ctx, hidden_size, 1, use_bias)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override
        {
            ggml_tensor *scale = gate.forward(ctx, hidden_states);
            ggml_tensor *r     = MLP::forward(ctx, hidden_states);
            scale = ggml_map_custom1_inplace(ctx->gctx.get(), scale, ggml_compute_forward_sigmoid, 1, nullptr);
            r = ggml_mul_inplace(ctx->gctx.get(), r, scale);
            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += MLP::get_param_num(effective_only);
            r += gate.get_param_num(effective_only);
            return r;
        }
    public:
        Linear gate;
    };

    template<class MLP1, class MLP2> class CombinedMLP : public Block
    {
    public:
        CombinedMLP(InitContext *ctx, int hidden_size, int intermediate_size1, int intermediate_size2)
            :  mlp1(ctx, hidden_size, intermediate_size1),
               mlp2(ctx, hidden_size, intermediate_size2)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override
        {
            ggml_tensor *r1 = mlp1.forward(ctx, hidden_states);
            ggml_tensor *r2 = mlp2.forward(ctx, hidden_states);
            ggml_tensor *r = ggml_add(ctx->gctx.get(), r1, r2);
            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += mlp1.get_param_num(effective_only);
            r += mlp2.get_param_num(effective_only);
            return r;
        }
    public:
        MLP1 mlp1;
        MLP2 mlp2;
    };

    class BaseSparseMLP : public Block
    {
    public:
        BaseSparseMLP() = default;
        BaseSparseMLP(InitContext *ctx, int hidden_size, int intermediate_size, int num_local_experts, int num_experts_per_tok,
                  ActFunc act, bool gate_use_bias)
            : num_local_experts(num_local_experts), num_experts_per_tok(num_experts_per_tok),
              gate(ctx, hidden_size, num_local_experts, gate_use_bias),
              experts_gate(ctx, hidden_size, intermediate_size, num_local_experts),
              experts_down(ctx, intermediate_size, hidden_size, num_local_experts),
              experts_up  (ctx, hidden_size, intermediate_size, num_local_experts),
              act(act),
              norm_topk_prob(true)
        {
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

        void set_prec(ggml_prec prec) override
        {
            gate.set_prec(prec);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += experts_gate.get_param_num(effective_only);
            r += experts_down.get_param_num(effective_only);
            r += experts_up.get_param_num(effective_only);
            if (effective_only)
            {
                r /= num_local_experts / num_experts_per_tok;
            }
            r += gate.get_param_num(effective_only);

            return r;
        }

    public:
        const int num_local_experts;
        const int num_experts_per_tok;
        Linear gate;
        MultiLinear experts_gate;
        MultiLinear experts_down;
        MultiLinear experts_up;
        const ActFunc act;
        bool norm_topk_prob;
    };

    template <bool bias> class InternLMBlock : public LMBlock1<RMSNorm, InternLMSelfAttention<bias>, RMSNorm, SiLUMLP>
    {
    public:
        InternLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, InternLMSelfAttention<bias>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
        InternLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : InternLMBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_attention_heads, max_length)
        {}
    };

    class LlamaSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        LlamaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        LlamaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    class LlamaBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class RobertaPooler : public Block
    {
    public:
        RobertaPooler() : dense(), act(ActFunc::GELU) {}

        RobertaPooler(InitContext *ctx, int hidden_size)
            : dense(ctx, hidden_size, hidden_size, true),
              act(ActFunc::Tanh)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = dense.get_param_num(effective_only);
            return r;
        }

    public:
        Linear dense;
        ActFunc act;
    };

    class RobertaClassificationHead : public Block
    {
    public:
        RobertaClassificationHead() : dense(), act(ActFunc::GELU), out_proj() {}

        RobertaClassificationHead(InitContext *ctx, int hidden_size)
            : dense(ctx, hidden_size, hidden_size, true),
              act(ActFunc::Tanh),
              out_proj(ctx, hidden_size, 1)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += dense.get_param_num(effective_only);
            r += out_proj.get_param_num(effective_only);
            return r;
        }

    public:
        Linear dense;
        ActFunc act;
        Linear out_proj;
    };

    class BCEFinalNorm : public Block
    {
    public:
        BCEFinalNorm() {}

        BCEFinalNorm(InitContext *ctx, int hidden_size)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            return 0;
        }
    };

    class RobertaSelfAttention : public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        RobertaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RobertaSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length) {}

        RobertaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RobertaSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true) {}

        RobertaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
            causal = false;
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return k;
        }
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return q;
        }
    };

    class RobertaOutput : public Block
    {
    public:
        RobertaOutput() = default;

        RobertaOutput(InitContext *ctx, int hidden_size, bool use_bias = true)
            : RobertaOutput(ctx, hidden_size, hidden_size, use_bias)
        {}

        RobertaOutput(InitContext *ctx, int hidden_size, int intermediate_size, bool use_bias = true)
            : dense(ctx, intermediate_size, hidden_size, use_bias),
              norm(ctx, hidden_size)
        {}

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, ggml_tensor *attention_output);

    public:
        Linear dense;
        LayerNorm norm;
    };

    class RobertaMLP : public Block
    {
    public:
        RobertaMLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : RobertaMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU, true)
        {
        }

        RobertaMLP(InitContext *ctx, int hidden_size, int intermediate_size, ActFunc act, bool bias)
            : intermediate(ctx, hidden_size, intermediate_size, bias),
              output(ctx, hidden_size, intermediate_size, bias),
              act(act)
        {}

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) override;

    public:
        Linear intermediate;
        RobertaOutput output;
        ActFunc act;
    };

    class RobertaBlock : public Block
    {
    public:
        RobertaBlock() = default;

        RobertaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : RobertaBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, true, true)
        {}

        RobertaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length,
                bool qkv_bias, bool o_bias)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              output_layernorm(ctx, hidden_size)
        {
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override;

        void shift_cache(int shift, int total) override
        {
            attention.shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += post_attention_layernorm.get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            r += output_layernorm.get_param_num(effective_only);
            return r;
        }

    public:
        RobertaSelfAttention attention;
        LayerNorm post_attention_layernorm;
        RobertaMLP mlp;
        LayerNorm output_layernorm;
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

        size_t get_cache_size(void) const override
        {
            return attention.get_cache_size();
        }

        void  *set_cache_buffer(void *buffer) override
        {
            return attention.set_cache_buffer(buffer);
        }

    public:
        InputNormBlock input_layernorm;
        AttentionBlock attention;
        MLPBlock mlp;
    };

    // Let's name it CrossAttention as in [`modeling_phi.py`](). Take it easy.
    class Phi2CrossAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        Phi2CrossAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias, bool o_bias)
            : Phi2CrossAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        Phi2CrossAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
            rope_dim = 32;
            attn_scaling = false;
            rope_mode = RoPEMode::Original;
        }
    };

    class Phi2Block : public LMBlock2<LayerNorm, Phi2CrossAttention, Phi2MLP>
    {
    public:
        Phi2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock2(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, true, true)
        {}
    };

    #define SlidingWindowAttentionImpl BaseSlidingWindowAttentionPartialCache

    template <int sliding_window_len> class MistralSelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        MistralSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        MistralSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    class BaichuanSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        BaichuanSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaichuanSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        BaichuanSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
            alibi.bias_max = 8.0f;
            alibi.n_past = 0;
            alibi.n_head = num_attention_heads;
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;

        ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const override;

        alibi_ctx alibi;
    };

    class BaichuanBlock : public LMBlock1<RMSNorm, BaichuanSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        BaichuanBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        BaichuanBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class QWenSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        QWenSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length);

        void config(int rope_dim, float rope_freq_base, int seq_length, bool use_dynamic_ntk, bool use_logn_attn);

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

    class QWenBlock : public LMBlock1<RMSNorm, QWenSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        QWenBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}
    };

    class QWen2SelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        QWen2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, true, false)
        {
            rope_mode = RoPEMode::Original;
        }

        QWen2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : QWen2SelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length)
        {}
    };

    class QWen2Block : public LMBlock1<RMSNorm, QWen2SelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        QWen2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class BlueLMSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        BlueLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BlueLMSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        BlueLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
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

    class BlueLMBlock : public LMBlock1<RMSNorm, BlueLMSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        BlueLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        BlueLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template <int sliding_window_len> class MistralBlock : public LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>
    {
    public:
        MistralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        MistralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class StableLMAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        StableLMAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : StableLMAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        StableLMAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
            rope_mode = RoPEMode::Original;
        }
    };

    class StableLMBlock : public LMBlock1<LayerNorm, StableLMAttention, LayerNorm, SiLUMLP>
    {
    public:
        StableLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class OrionBlock : public LMBlock1<LayerNorm, LlamaSelfAttention, LayerNorm, SiLUMLP>
    {
    public:
        OrionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class MiniCPMBlock : public LMBlock3<RMSNorm, LlamaSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        MiniCPMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock3(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template <class Norm, class BaseAttn> class QKNormedAttention : public RoPESelfAttention<BaseAttn>
    {
    public:
        QKNormedAttention() : BaseAttn() {}

        QKNormedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention<BaseAttn>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias),
              k_layernorm(ctx, hidden_size / num_kv_heads),
              q_layernorm(ctx, hidden_size / num_attention_heads)
        {
            RoPESelfAttention<BaseAttn>::rope_mode = RoPEMode::Original;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += RoPESelfAttention<BaseAttn>::get_param_num(effective_only);
            r += k_layernorm.get_param_num(effective_only);
            r += q_layernorm.get_param_num(effective_only);
            return r;
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            k = const_cast<QKNormedAttention *>(this)->k_layernorm.forward(ctx, k);
            return RoPESelfAttention<BaseAttn>::apply_pos_embedding_k(ctx, k, hidden_size, qlen, past);    // [qlen, heads, head_size]
        }

        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            q = const_cast<QKNormedAttention *>(this)->q_layernorm.forward(ctx, q);
            return RoPESelfAttention<BaseAttn>::apply_pos_embedding_q(ctx, q, hidden_size, qlen, past);    // [qlen, heads, head_size];
        }

    public:
        Norm k_layernorm;
        Norm q_layernorm;
    };

    class PersimmonSelfAttention : public QKNormedAttention<LayerNorm, BaseAttention>
    {
    public:
        PersimmonSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : PersimmonSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length) {}

        PersimmonSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : QKNormedAttention<LayerNorm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true) {}
    };

    class PersimmonMLP : public TheMLP
    {
    public:
        PersimmonMLP() = default;
        PersimmonMLP(InitContext *ctx, int hidden_size)
            : PersimmonMLP(ctx, hidden_size, 4 * hidden_size) {}

        PersimmonMLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : TheMLP(ctx, hidden_size, intermediate_size, ActFunc::RELU2, true) {}
    };

    class PersimmonBlock : public LMBlock1<LayerNorm, PersimmonSelfAttention, LayerNorm, PersimmonMLP>
    {
    public:
        PersimmonBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}
    };

    class GemmaSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        GemmaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    class GemmaBlock : public LMBlock1<RMSNorm, GemmaSelfAttention, RMSNorm, GELUMLP>
    {
    public:
        GemmaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class CohereSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        CohereSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,  bool qkv_bias = false, bool o_bias = false)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
        }
    };

    class CohereBlock : public LMBlock2<LayerNormNoBias, CohereSelfAttention, SiLUMLP>
    {
    public:
        CohereBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock2(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, false, false) {}
    };

    class FuyuEmbedding : public VisualEmbedding
    {
    public:
        FuyuEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim)
            : VisualEmbedding(ctx, num_embeddings, embedding_dim),
              image_new_line_tok_id(0) {}

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *patches, int patches_per_row, ggml_tensor *text_input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += VisualEmbedding::get_param_num(effective_only);
            r += vision_embed.get_param_num(effective_only);
            return r;
        }

    public:
        Linear vision_embed;
        int image_new_line_tok_id;
    };

    class Phi3SUSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        Phi3SUSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
        }

        void config(int original_max_position_embeddings,
            float rope_theta, float scaling_factor, int factor_len, const float *short_factor, const float *long_factor);

        const float *get_inv_freq(int pos);

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override;
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override;

    public:
        int original_max_position_embeddings;
        float scaling_factor;
        std::vector<float> inv_freq_short;
        std::vector<float> inv_freq_long;
    };

    class Phi3SUBlock : public LMBlock1<RMSNorm, Phi3SUSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        Phi3SUBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template <int sliding_window_len, bool qvk_bias, bool o_bias> class StarCoder2SelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        StarCoder2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qvk_bias, o_bias)
        {
            RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>::rope_mode = RoPEMode::Original;
        }
    };

    class StarCoder2MLP : public TheMLP
    {
    public:
        StarCoder2MLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : TheMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU, true)
        {
        }
    };

    template <int sliding_window_len, bool qvk_bias = true, bool o_bias = true> class StarCoder2Block : public LMBlock1<LayerNorm, StarCoder2SelfAttention<sliding_window_len, qvk_bias, o_bias>, LayerNorm, StarCoder2MLP>
    {
    public:
        StarCoder2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<LayerNorm, StarCoder2SelfAttention<sliding_window_len, qvk_bias, o_bias>, LayerNorm, StarCoder2MLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };
} // namespace chatllm
