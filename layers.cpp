#include "layers.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <regex>
#include <string>
#include <functional>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

namespace chatllm
{

    ggml_tensor *Embedding::forward(ForwardContext *ctx, ggml_tensor *input)
    {
        ggml_tensor *output = ggml_get_rows(ctx->gctx.get(), weight, input);
        return output;
    }

    ggml_tensor *Linear::forward(ForwardContext *ctx, ggml_tensor *input)
    {
        // input: [seqlen, in_features]
        ggml_tensor *output = ggml_mul_mat(ctx->gctx.get(), weight, input); // [seqlen, out_features]
        if (bias)
        {
            output = ggml_add_inplace(ctx->gctx.get(), output, bias);
        }
        return output;
    }

    ggml_tensor *LayerNorm::forward(ForwardContext *ctx, ggml_tensor *input)
    {
        // input: [seqlen, normalized_shape]
        ggml_tensor *output = ggml_norm_inplace(ctx->gctx.get(), input, eps);
        output = ggml_mul_inplace(ctx->gctx.get(), output, weight);
        output = ggml_add_inplace(ctx->gctx.get(), output, bias);
        return output;
    }

    ggml_tensor *RMSNorm::forward(ForwardContext *ctx, ggml_tensor *input)
    {
        ggml_tensor *output = ggml_rms_norm_inplace(ctx->gctx.get(), input, eps);
        output = ggml_mul_inplace(ctx->gctx.get(), output, weight);
        return output;
    }

    ggml_tensor *GLMMLP::forward(ForwardContext *ctx, ggml_tensor *hidden_states)
    {
        ggml_tensor *output = dense_h_to_4h.forward(ctx, hidden_states);
        output = ggml_gelu_inplace(ctx->gctx.get(), output);
        output = dense_4h_to_h.forward(ctx, output);
        return output;
    }

    static void fill_pos_vector(ggml_tensor *pos, int n_past, int qlen)
    {
        int *p = (int *)pos->data;
        for (int i = 0; i < qlen; i++)
            p[i] = n_past + i;
        pos->ne[0] = qlen;
    }

    ggml_tensor *GLMSelfAttention::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past)
    {
        int hidden_size = hidden_states->ne[0];
        int qlen = hidden_states->ne[1];
        int head_size = hidden_size / num_attention_heads;
        int rope_dim = head_size / 2;
        fill_pos_vector(pos, n_past, qlen);

        if (shift_pending.shift > 0)
        {
            int remain = shift_pending.total - shift_pending.shift;
            if (remain > 0)
            {
                struct ggml_tensor * k_cache_remain = ggml_view_3d(ctx->gctx.get(), k_cache, head_size, remain, num_attention_heads, k_cache->nb[1], k_cache->nb[2],
                         shift_pending.shift * head_size * ggml_element_size(k_cache)); // [heads, remain, head_size]
                struct ggml_tensor * k_cache_dst    = ggml_view_3d(ctx->gctx.get(), k_cache, head_size, remain, num_attention_heads, k_cache->nb[1], k_cache->nb[2],
                         0); // [heads, remain, head_size]

                struct ggml_tensor * v_cache_remain = ggml_view_3d(ctx->gctx.get(), v_cache, remain, head_size, num_attention_heads, v_cache->nb[1], v_cache->nb[2],
                         shift_pending.shift * ggml_element_size(v_cache)); // [heads, head_size, remain]
                struct ggml_tensor * v_cache_dst    = ggml_view_3d(ctx->gctx.get(), v_cache, remain, head_size, num_attention_heads, v_cache->nb[1], v_cache->nb[2],
                         0); // [heads, head_size, remain]

                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_cache_remain, k_cache_dst));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_cache_remain, v_cache_dst));
            }

            shift_pending.clear();
        }

        ggml_tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, 3 * hidden]

        ggml_tensor *query_layer = ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen,
                                                3 * head_size * ggml_element_size(qkv), qkv->nb[1], 0);
        query_layer =
            ggml_rope_inplace(ctx->gctx.get(), query_layer, pos, rope_dim, 4, n_ctx); // [qlen, heads, head_size]
        query_layer = ggml_permute(ctx->gctx.get(), query_layer, 0, 2, 1, 3);            // [heads, qlen, head_size]

        ggml_tensor *key_layer =
            ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                         qkv->nb[1], head_size * ggml_element_size(qkv));
        key_layer = ggml_rope_inplace(ctx->gctx.get(), key_layer, pos, rope_dim, 4, n_ctx); // [qlen, heads, head_size]
        key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3);                      // [heads, qlen, head_size]

        ggml_tensor *value_layer = ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen,
                                                3 * head_size * ggml_element_size(qkv), qkv->nb[1],
                                                2 * head_size * ggml_element_size(qkv)); // [qlen, heads, head_size]
        value_layer = ggml_permute(ctx->gctx.get(), value_layer, 1, 2, 0, 3);            // [heads, head_size, qlen]

        // store key & value to cache
        ggml_tensor *k_cache_view =
            ggml_view_3d(ctx->gctx.get(), k_cache, head_size, qlen, num_attention_heads, k_cache->nb[1], k_cache->nb[2],
                         n_past * head_size * ggml_element_size(k_cache)); // [heads, qlen, head_size]
        ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), key_layer, k_cache_view));
        ggml_tensor *v_cache_view =
            ggml_view_3d(ctx->gctx.get(), v_cache, qlen, head_size, num_attention_heads, v_cache->nb[1], v_cache->nb[2],
                         n_past * ggml_element_size(v_cache)); // [heads, head_size, qlen]
        ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), value_layer, v_cache_view));

        key_layer = ggml_view_3d(ctx->gctx.get(), k_cache, head_size, n_past + qlen, num_attention_heads, k_cache->nb[1],
                                 k_cache->nb[2], 0); // [heads, klen, head_size]
        value_layer = ggml_view_3d(ctx->gctx.get(), v_cache, n_past + qlen, head_size, num_attention_heads, v_cache->nb[1],
                                   v_cache->nb[2], 0); // [heads, head_size, klen]

        ggml_tensor *attn_scores = ggml_mul_mat(ctx->gctx.get(), key_layer, query_layer); // [heads, qlen, klen]
        if (n_past == 0)
        {
            // build attention mask for context input
            ggml_tensor *inf = ggml_new_tensor_3d(ctx->gctx.get(), attn_scores->type, 1, qlen - 1, num_attention_heads);
            ggml_set_f32(inf, -INFINITY);
            ggml_tensor *masked_attn_scores = ggml_view_3d(
                ctx->gctx.get(), attn_scores, 1, qlen - 1, num_attention_heads, qlen * ggml_element_size(attn_scores),
                qlen * qlen * ggml_element_size(attn_scores), (qlen - 1) * ggml_element_size(attn_scores));
            ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), inf, masked_attn_scores));
        }
        attn_scores =
            ggml_scale_inplace(ctx->gctx.get(), attn_scores, ggml_new_f32(ctx->gctx.get(), 1.f / std::sqrt(head_size)));
        ggml_tensor *attn_probs = ggml_soft_max_inplace(ctx->gctx.get(), attn_scores); // [heads, qlen, klen]

        ggml_tensor *context_layer = ggml_mul_mat(ctx->gctx.get(), value_layer, attn_probs); // [heads, qlen, head_size]
        context_layer = ggml_reshape_2d(
            ctx->gctx.get(), ggml_cont(ctx->gctx.get(), ggml_permute(ctx->gctx.get(), context_layer, 0, 2, 1, 3)),
            hidden_size, qlen);

        ggml_tensor *attn_output = dense.forward(ctx, context_layer);
        return attn_output;
    }

    ggml_tensor *GLMBlock::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past)
    {
        ggml_tensor *alpha = ggml_new_f32(ctx->gctx.get(), std::sqrt(2.f * num_hidden_layers));

        ggml_tensor *attn_input = input_layernorm.forward(ctx, hidden_states);
        ggml_tensor *attn_output = attention.forward(ctx, attn_input, n_past);
        ggml_build_forward_expand(ctx->gf, attn_output);
        hidden_states =
            ggml_add_inplace(ctx->gctx.get(), ggml_scale_inplace(ctx->gctx.get(), attn_input, alpha), attn_output);

        ggml_tensor *mlp_input = post_attention_layernorm.forward(ctx, hidden_states);
        ggml_tensor *mlp_output = mlp.forward(ctx, mlp_input);
        ggml_build_forward_expand(ctx->gf, mlp_output);
        ggml_tensor *output =
            ggml_add_inplace(ctx->gctx.get(), ggml_scale_inplace(ctx->gctx.get(), mlp_input, alpha), mlp_output);
        return output;
    }

    ggml_tensor *GLM2SelfAttention::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past)
    {
        const int hidden_size = hidden_states->ne[0];
        const int qlen = hidden_states->ne[1];
        const int head_size = hidden_size / num_attention_heads;
        const int rope_dim = head_size / 2;
        const int mqa_scale = num_attention_heads / num_kv_heads;
        fill_pos_vector(pos, n_past, qlen);

        if (shift_pending.shift > 0)
        {
            int remain = shift_pending.total - shift_pending.shift;
            if (remain > 0)
            {
                struct ggml_tensor * k_cache_remain = ggml_view_3d(ctx->gctx.get(), k_cache, head_size, remain, num_kv_heads, k_cache->nb[1], k_cache->nb[2],
                         shift_pending.shift * head_size * ggml_element_size(k_cache)); // [kv_heads, remain, head_size]
                struct ggml_tensor * k_cache_dst    = ggml_view_3d(ctx->gctx.get(), k_cache, head_size, remain, num_kv_heads, k_cache->nb[1], k_cache->nb[2],
                         0); // [kv_heads, remain, head_size]

                struct ggml_tensor * v_cache_remain = ggml_view_3d(ctx->gctx.get(), v_cache, remain, head_size, num_kv_heads, v_cache->nb[1], v_cache->nb[2],
                         shift_pending.shift * ggml_element_size(v_cache)); // [kv_heads, head_size, qlen];
                struct ggml_tensor * v_cache_dst    = ggml_view_3d(ctx->gctx.get(), v_cache, remain, head_size, num_kv_heads, v_cache->nb[1], v_cache->nb[2],
                         0); // [kv_heads, head_size, qlen]

                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_cache_remain, k_cache_dst));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_cache_remain, v_cache_dst));
            }

            shift_pending.clear();
        }

        ggml_tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, hidden + 2 * kv_hidden]

        ggml_tensor *query_layer =
            ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen, head_size * ggml_element_size(qkv),
                         qkv->nb[1], 0); // [qlen, heads, head_size]
        query_layer = ggml_rope_inplace(ctx->gctx.get(), query_layer, pos, rope_dim, 0, 0);
        query_layer = ggml_view_4d(ctx->gctx.get(), query_layer, head_size, mqa_scale, num_kv_heads, qlen,
                                   query_layer->nb[1], query_layer->nb[1] * mqa_scale, query_layer->nb[2],
                                   0);                                        // [qlen, kv_heads, mqa_scale, head_size]
        query_layer = ggml_permute(ctx->gctx.get(), query_layer, 0, 2, 3, 1); // [kv_heads, mqa_scale, qlen, head_size]

        ggml_tensor *key_layer =
            ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_kv_heads, qlen, head_size * ggml_element_size(qkv),
                         qkv->nb[1], hidden_size * ggml_element_size(qkv)); // [qlen, kv_heads, head_size]
        key_layer = ggml_rope_inplace(ctx->gctx.get(), key_layer, pos, rope_dim, 0, 0);
        key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3); // [kv_heads, qlen, head_size]

        ggml_tensor *value_layer = ggml_view_3d(
            ctx->gctx.get(), qkv, head_size, num_kv_heads, qlen, head_size * ggml_element_size(qkv), qkv->nb[1],
            (hidden_size + head_size * num_kv_heads) * ggml_element_size(qkv)); // [qlen, kv_heads, head_size]
        value_layer = ggml_permute(ctx->gctx.get(), value_layer, 1, 2, 0, 3);   // [kv_heads, head_size, qlen]

        // store key & value to cache
        ggml_tensor *k_cache_view =
            ggml_view_3d(ctx->gctx.get(), k_cache, head_size, qlen, num_kv_heads, k_cache->nb[1], k_cache->nb[2],
                         n_past * head_size * ggml_element_size(k_cache)); // [kv_heads, qlen, head_size]
        ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), key_layer, k_cache_view));
        ggml_tensor *v_cache_view =
            ggml_view_3d(ctx->gctx.get(), v_cache, qlen, head_size, num_kv_heads, v_cache->nb[1], v_cache->nb[2],
                         n_past * ggml_element_size(v_cache)); // [kv_heads, head_size, qlen]
        ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), value_layer, v_cache_view));

        // concat key & value with past kv
        key_layer = ggml_view_4d(ctx->gctx.get(), k_cache, head_size, n_past + qlen, mqa_scale, num_kv_heads,
                                 k_cache->nb[1], 0, k_cache->nb[2],
                                 0); // [kv_heads, mqa_scale, klen, head_size]
        value_layer = ggml_view_4d(ctx->gctx.get(), v_cache, n_past + qlen, head_size, mqa_scale, num_kv_heads,
                                   v_cache->nb[1], 0, v_cache->nb[2],
                                   0); // [kv_heads, mqa_scale, head_size, klen]

        // flash attention
        ggml_tensor *context_layer = ggml_flash_attn(ctx->gctx.get(), query_layer, key_layer, value_layer,
                                                     true); // [mqa_scale, kv_heads, qlen, head_size]
        context_layer = ggml_reshape_2d(
            ctx->gctx.get(), ggml_cont(ctx->gctx.get(), ggml_permute(ctx->gctx.get(), context_layer, 0, 3, 1, 2)),
            hidden_size, qlen); // [qlen, hidden]

        ggml_tensor *attn_output = dense.forward(ctx, context_layer);
        return attn_output;
    }

    ggml_tensor *GLM2MLP::forward(ForwardContext *ctx, ggml_tensor *hidden_states)
    {
        ggml_tensor *output = dense_h_to_4h.forward(ctx, hidden_states);

        // swiglu activation
        ggml_tensor *x0 = ggml_view_2d(ctx->gctx.get(), output, output->ne[0] / 2, output->ne[1], output->nb[1], 0);
        ggml_tensor *x1 = ggml_view_2d(ctx->gctx.get(), output, output->ne[0] / 2, output->ne[1], output->nb[1],
                                       output->ne[0] / 2 * ggml_element_size(output));
        output = ggml_mul_inplace(ctx->gctx.get(), ggml_silu_inplace(ctx->gctx.get(), ggml_cont(ctx->gctx.get(), x0)), x1);

        output = dense_4h_to_h.forward(ctx, output);
        return output;
    }

    ggml_tensor *BaseMLP::forward(ForwardContext *ctx, ggml_tensor *hidden_states)
    {
        ggml_tensor *act = ggml_silu_inplace(ctx->gctx.get(), gate_proj.forward(ctx, hidden_states));
        ggml_tensor *proj = up_proj.forward(ctx, hidden_states);

        ggml_tensor *output = ggml_mul_inplace(ctx->gctx.get(), act, proj);
        output = down_proj.forward(ctx, output);
        return output;
    }

    ggml_tensor *BaseSelfAttention::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past)
    {
        const int hidden_size = hidden_states->ne[0];
        const int qlen = hidden_states->ne[1];
        const int head_size = hidden_size / num_attention_heads;
        fill_pos_vector(pos, n_past, qlen);

        // shift cache
        if (shift_pending.shift > 0)
        {
            int remain = shift_pending.total - shift_pending.shift;
            if (remain > 0)
            {
                struct ggml_tensor * k_cache_remain = ggml_view_1d(ctx->gctx.get(), k_cache, remain * hidden_size,
                                            ggml_element_size(k_cache) * hidden_size * shift_pending.shift);
                struct ggml_tensor * k_cache_1d = ggml_view_1d(ctx->gctx.get(), k_cache, remain * hidden_size,
                                            0);

                struct ggml_tensor * v_cache_remain = ggml_view_2d(ctx->gctx.get(), v_cache, remain, hidden_size,
                                            max_length * ggml_element_size(v_cache),
                                            shift_pending.shift * ggml_element_size(v_cache));
                struct ggml_tensor * v_cache_2d =     ggml_view_2d(ctx->gctx.get(), v_cache, remain, hidden_size,
                                            max_length * ggml_element_size(v_cache),
                                            0);

                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_cache_remain, k_cache_1d));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), v_cache_remain, v_cache_2d));
            }
            shift_pending.clear();
        }

        ggml_tensor *tmpq = q_proj.forward(ctx, hidden_states);
        ggml_tensor *tmpk = k_proj.forward(ctx, hidden_states);
        ggml_tensor *tmpv = v_proj.forward(ctx, hidden_states);

        // [qlen, heads, head_size]
        ggml_tensor * key_layer = ggml_reshape_3d(ctx->gctx.get(), tmpk, head_size, num_attention_heads, qlen);
        key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, pos);

        // [qlen, heads, head_size]
        ggml_tensor * query_layer = ggml_reshape_3d(ctx->gctx.get(), tmpq, head_size, num_attention_heads, qlen);
        query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, pos);

        // store key and value to memory
        {
            // compute the transposed [N, n_embd] V matrix
            struct ggml_tensor * Vcur = ggml_transpose(ctx->gctx.get(), tmpv); // ggml_reshape_2d(ctx->gctx.get(), tmpv, hidden_size, qlen));

            struct ggml_tensor * k_cache_view = ggml_view_1d(ctx->gctx.get(), k_cache, qlen * hidden_size,
                                        ggml_element_size(k_cache) * hidden_size * n_past);

            struct ggml_tensor * v_cache_view = ggml_view_2d(ctx->gctx.get(), v_cache, qlen, hidden_size,
                    max_length * ggml_element_size(v_cache), n_past * ggml_element_size(v_cache));

            struct ggml_tensor * k_view = ggml_view_1d(ctx->gctx.get(), key_layer, qlen * hidden_size, 0);

            // important: storing RoPE-ed version of K in the KV cache!
            ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_view, k_cache_view));
            ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), Vcur, v_cache_view));
        }

        query_layer = ggml_permute(ctx->gctx.get(), query_layer, 0, 2, 1, 3);                     // [heads, qlen, head_size]

        key_layer = ggml_view_1d(ctx->gctx.get(), k_cache, (n_past + qlen) * hidden_size, 0);
        key_layer = ggml_reshape_3d(ctx->gctx.get(), key_layer, head_size, num_attention_heads, n_past + qlen);  // [qlen, heads, head_size]
        key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3);                                        // [heads, qlen, head_size]

        ggml_tensor *attn_scores = ggml_mul_mat(ctx->gctx.get(), key_layer, query_layer); // [heads, qlen, klen]
        attn_scores = apply_pos_embedding_kq(ctx, attn_scores, hidden_size, qlen, pos);
        attn_scores = ggml_scale_inplace(ctx->gctx.get(), attn_scores, ggml_new_f32(ctx->gctx.get(), 1.f / std::sqrt(head_size)));

        // attn_masked = mask_past(attn_scores)
        struct ggml_tensor * attn_masked = ggml_diag_mask_inf_inplace(ctx->gctx.get(), attn_scores, n_past);

        // attn_probs = soft_max(attn_masked)
        struct ggml_tensor * attn_probs = ggml_soft_max_inplace(ctx->gctx.get(), attn_masked);

        ggml_tensor * value_layer = ggml_view_3d(ctx->gctx.get(),
                        v_cache,
                        n_past + qlen, head_size, num_attention_heads,
                        max_length * ggml_element_size(v_cache),
                        max_length * ggml_element_size(v_cache) * head_size,
                        0); // [heads, head_size, klen]

        ggml_tensor *context_layer = ggml_mul_mat(ctx->gctx.get(), value_layer, attn_probs); // [heads, qlen, head_size]
        context_layer = ggml_reshape_2d(
            ctx->gctx.get(),
            ggml_cont(ctx->gctx.get(), ggml_permute(ctx->gctx.get(), context_layer, 0, 2, 1, 3)),
            hidden_size, qlen);

        ggml_tensor *attn_output = o_proj.forward(ctx, context_layer);
        return attn_output;
    }

    ggml_tensor *BaseSelfAttention::apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor *past) const
    {
        const int rope_dim = hidden_size / num_attention_heads;
        return ggml_rope_inplace(ctx->gctx.get(), k, past, rope_dim, 0, 0);    // [qlen, heads, head_size]
    }

    ggml_tensor *BaseSelfAttention::apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor *past) const
    {
        const int rope_dim = hidden_size / num_attention_heads;
        return ggml_rope_inplace(ctx->gctx.get(), q, past, rope_dim, 0, 0);    // [qlen, heads, head_size];
    }

}
