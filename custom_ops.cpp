#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <functional>
#include "layers.h"

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

#define ggctx       (ctx->gctx.get())

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static float qwen_get_ntk_alpha(int true_seq_len, int seq_length)
{
    float context_value = log2f(true_seq_len / seq_length) + 1;
    float ntk_alpha = powf(2, ceil(context_value)) - 1;
    return MAX(ntk_alpha, 1.0);
}

static float qwen_get_scale(int true_seq_len, int seq_length, int rope_dim)
{
    float ntk_alpha = qwen_get_ntk_alpha(true_seq_len, seq_length);
    return powf(ntk_alpha, (float)rope_dim / (rope_dim - 2));
}

static void ggml_compute_forward_ntk_dynamic_rope_f32(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata)
{
    BlueLMSelfAttention *data = reinterpret_cast<BlueLMSelfAttention *>(userdata);

    const struct ggml_tensor *src0 = a;
    const struct ggml_tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const float p = (float)pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                for (int64_t i0 = 0; i0 < ne0; i0 += 2) {

                    float theta = p * data->inv_freq[i0 / 2];

                    float cos_theta = cosf(theta);
                    float sin_theta = sinf(theta);

                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x0 = src[0];
                    const float x1 = src[1];

                    dst_data[0] = x0*cos_theta - x1*sin_theta;
                    dst_data[1] = x0*sin_theta + x1*cos_theta;
                }
            }
        }
    }
}

static void ggml_compute_forward_ntk_dynamic_rope_f16(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata)
{
    BlueLMSelfAttention *data = reinterpret_cast<BlueLMSelfAttention *>(userdata);

    const struct ggml_tensor *src0 = a;
    const struct ggml_tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const float p = (float)pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                for (int64_t i0 = 0; i0 < ne0; i0 += 2) {

                    float theta = p * data->inv_freq[i0 / 2];

                    float cos_theta = cosf(theta);
                    float sin_theta = sinf(theta);

                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x0 = ggml_fp16_to_fp32(src[0]);
                    const float x1 = ggml_fp16_to_fp32(src[1]);

                    dst_data[0] = ggml_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                    dst_data[1] = ggml_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                }
            }
        }
    }
}

static void ggml_compute_forward_ntk_dynamic_rope(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata)
{
    switch (a->type)
    {
    case GGML_TYPE_F16:
        ggml_compute_forward_ntk_dynamic_rope_f16(dst, a, b, ith, nth, userdata);
        break;
    case GGML_TYPE_F32:
        ggml_compute_forward_ntk_dynamic_rope_f32(dst, a, b, ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}

static void ggml_compute_forward_ntk_mix_rope_f32(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata)
{
    BlueLMSelfAttention *data = reinterpret_cast<BlueLMSelfAttention *>(userdata);

    const struct ggml_tensor *src0 = a;
    const struct ggml_tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const float p = (float)pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                for (int64_t i0 = 0; i0 < ne0; i0 += 2) {

                    float theta = p * data->inv_freq[i0 / 2];

                    float cos_theta = cosf(theta);
                    float sin_theta = sinf(theta);

                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x0 = src[0];
                    const float x1 = src[1];

                    dst_data[0] = x0*cos_theta - x1*sin_theta;
                    dst_data[1] = x0*sin_theta + x1*cos_theta;
                }
            }
        }
    }
}

static void ggml_compute_forward_ntk_mix_rope_f16(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata)
{
    BlueLMSelfAttention *data = reinterpret_cast<BlueLMSelfAttention *>(userdata);

    const struct ggml_tensor *src0 = a;
    const struct ggml_tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const float p = (float)pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                for (int64_t i0 = 0; i0 < ne0; i0 += 2) {

                    float theta = p * data->inv_freq[i0 / 2];

                    float cos_theta = cosf(theta);
                    float sin_theta = sinf(theta);

                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x0 = ggml_fp16_to_fp32(src[0]);
                    const float x1 = ggml_fp16_to_fp32(src[1]);

                    dst_data[0] = ggml_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                    dst_data[1] = ggml_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                }
            }
        }
    }
}

static void ggml_compute_forward_ntk_mix_rope(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata)
{
    switch (a->type)
    {
    case GGML_TYPE_F16:
        ggml_compute_forward_ntk_mix_rope_f16(dst, a, b, ith, nth, userdata);
        break;
    case GGML_TYPE_F32:
        ggml_compute_forward_ntk_mix_rope_f32(dst, a, b, ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}