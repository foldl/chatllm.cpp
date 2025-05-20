#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <functional>
#include <random>
#include "layers.h"

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static float qwen_get_ntk_alpha(int true_seq_len, int seq_length)
{
    float context_value = log2f((float)true_seq_len / (float)seq_length) + 1;
    float ntk_alpha = powf(2, ceil(context_value)) - 1;
    return MAX(ntk_alpha, 1.0f);
}

static void ggml_compute_forward_mat_scale_f32(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int nr = (int)ggml::nrows(dst);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float * scales = (const float *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            //const auto p = pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                const float scale = scales[i1];

                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                          float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                    *dst_data = *src * scale;
                }
            }
        }
    }
}

static void ggml_compute_forward_mat_scale(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    switch (a->type)
    {
    case GGML_TYPE_F32:
        ggml_compute_forward_mat_scale_f32(dst, a, b, ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}

static void ggml_compute_forward_ntk_dynamic_rope_f32(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    QWenSelfAttention *data = reinterpret_cast<QWenSelfAttention *>(userdata);

    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = (int)ggml::nrows(dst);

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
            const auto p = pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                const float ntk_alpha = qwen_get_ntk_alpha(p, data->seq_length);
                const float base = data->freq_base * powf(ntk_alpha, (float)n_dims / ((float)n_dims - 2.0f));
                const float inv_freq_scale = powf(base, -2.0f / (float)n_dims);
                float inv_freq = 1.0f;

                for (int64_t ic = 0; ic < ne0; ic += 2, inv_freq *= inv_freq_scale) {
                    if (ic < n_dims) {
                        const int64_t ib = 0;

                        float theta = (float)p * inv_freq;
                        float cos_theta = cosf(theta);
                        float sin_theta = sinf(theta);

                        const int64_t i0 = ib*n_dims + ic/2;

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims/2];

                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                    } else {
                        const int64_t i0 = ic;

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_ntk_dynamic_rope_f16(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    QWenSelfAttention *data = reinterpret_cast<QWenSelfAttention *>(userdata);

    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = (int)ggml::nrows(dst);

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
            const auto p = pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                const float ntk_alpha = qwen_get_ntk_alpha(p, data->seq_length);
                const float base = data->freq_base * powf(ntk_alpha, (float)n_dims / ((float)n_dims - 2.0f));
                const float inv_freq_scale = powf(base, -2.0f / (float)n_dims);
                float inv_freq = 1.0f;

                for (int64_t ic = 0; ic < ne0; ic += 2, inv_freq *= inv_freq_scale) {
                    if (ic < n_dims) {
                        const int64_t ib = 0;

                        float theta = (float)p * inv_freq;
                        float cos_theta = cosf(theta);
                        float sin_theta = sinf(theta);

                        const int64_t i0 = ib*n_dims + ic/2;

                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = ggml_fp16_to_fp32(src[0]);
                        const float x1 = ggml_fp16_to_fp32(src[1]);

                        dst_data[0] = ggml_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                        dst_data[1] = ggml_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                    } else {
                        const int64_t i0 = ic;

                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_ntk_dynamic_rope(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
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

static void ggml_compute_forward_ntk_mix_rope_f32(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    BlueLMSelfAttention *data = reinterpret_cast<BlueLMSelfAttention *>(userdata);

    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = (int)ggml::nrows(dst);

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

static void ggml_compute_forward_ntk_mix_rope_f16(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    BlueLMSelfAttention *data = reinterpret_cast<BlueLMSelfAttention *>(userdata);

    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;

    const int nr = (int)ggml::nrows(dst);

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

static void ggml_compute_forward_ntk_mix_rope(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
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

static void build_ntk_mixed_inv_freq(int dim, std::vector<float> &inv_freq,
    int max_position_embeddings = 2048, float base = 10000.0, float k = 16, float b = 0.3)
{
    inv_freq.clear();
    float a = log(k) / powf((float)dim / 2, b);
    inv_freq.reserve(dim / 2);
    for (int i = 0; i < dim; i += 2)
    {
        float v = 1.0f / powf(base, (float)i / (float)dim) / expf(a * powf((float)i / 2 + 1, b));
        inv_freq.push_back(v);
    }
}


static void ggml_compute_forward_chatglm1_rope_f16(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    GLMSelfAttention *data = reinterpret_cast<GLMSelfAttention *>(userdata);

    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;
    int n_ctx  = data->n_ctx;

    const int nr = (int)ggml::nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(data->freq_base, -2.0f/n_dims);
    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                float theta_base = (float)p;

                theta_base = (float)MIN(p, n_ctx - 2);
                float block_theta = (float)MAX(p - (n_ctx - 2), 0);
                for (int64_t i0 = 0; i0 < ne0 / 4; i0++) {
                    const float cos_theta = cosf(theta_base);
                    const float sin_theta = sinf(theta_base) ;
                    const float cos_block_theta = cosf(block_theta);
                    const float sin_block_theta = sinf(block_theta);

                    theta_base *= theta_scale;
                    block_theta *= theta_scale;

                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                          ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data +  i3*nb3 + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x0 = ggml_fp16_to_fp32(src[0]);
                    const float x1 = ggml_fp16_to_fp32(src[n_dims/2]);
                    const float x2 = ggml_fp16_to_fp32(src[n_dims]);
                    const float x3 = ggml_fp16_to_fp32(src[n_dims/2*3]);

                    dst_data[0]          = ggml_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                    dst_data[n_dims/2]   = ggml_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                    dst_data[n_dims]     = ggml_fp32_to_fp16(x2*cos_block_theta - x3*sin_block_theta);
                    dst_data[n_dims/2*3] = ggml_fp32_to_fp16(x2*sin_block_theta + x3*cos_block_theta);
                }
            }
        }
    }
}

static void ggml_compute_forward_chatglm1_rope_f32(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    GLMSelfAttention *data = reinterpret_cast<GLMSelfAttention *>(userdata);

    const ggml::tensor *src0 = a;
    const ggml::tensor *src1 = b;

    GGML_TENSOR_UNARY_OP_LOCALS

    int n_dims = data->rope_dim;
    int n_ctx  = data->n_ctx;

    const int nr = (int)ggml::nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(data->freq_base, -2.0f/n_dims);
    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                float theta_base = (float)p;

                theta_base = (float)MIN(p, n_ctx - 2);
                float block_theta = (float)MAX(p - (n_ctx - 2), 0);
                for (int64_t i0 = 0; i0 < ne0 / 4; i0++) {
                    const float cos_theta = cosf(theta_base);
                    const float sin_theta = sinf(theta_base) ;
                    const float cos_block_theta = cosf(block_theta);
                    const float sin_block_theta = sinf(block_theta);

                    theta_base *= theta_scale;
                    block_theta *= theta_scale;

                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                          float * dst_data  = (float *)((char *)  dst->data +  i3*nb3 + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x0 = src[0];
                    const float x1 = src[n_dims/2];
                    const float x2 = src[n_dims];
                    const float x3 = src[n_dims/2*3];

                    dst_data[0]          = x0*cos_theta - x1*sin_theta;
                    dst_data[n_dims/2]   = x0*sin_theta + x1*cos_theta;
                    dst_data[n_dims]     = x2*cos_block_theta - x3*sin_block_theta;
                    dst_data[n_dims/2*3] = x2*sin_block_theta + x3*cos_block_theta;
                }
            }
        }
    }
}

static void ggml_compute_forward_chatglm1_rope(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
{
    switch (a->type)
    {
    case GGML_TYPE_F16:
        ggml_compute_forward_chatglm1_rope_f16(dst, a, b, ith, nth, userdata);
        break;
    case GGML_TYPE_F32:
        ggml_compute_forward_chatglm1_rope_f32(dst, a, b, ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}

static void ggml_compute_forward_randn_f32(ggml::tensor * dst, int ith, int nth, void * userdata)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const int64_t nb0 = dst->nb[0];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    const int nr = (int)ggml::nrows(dst);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    float * dst_data  = (float *)((char *)  dst->data +  i3*nb3 + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = distribution(gen);
                }
            }
        }
    }
}

static void ggml_compute_forward_randn_f16(ggml::tensor * dst, int ith, int nth, void * userdata)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const int64_t nb0 = dst->nb[0];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    const int nr = (int)ggml::nrows(dst);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data +  i3*nb3 + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = ggml_fp32_to_fp16(distribution(gen));
                }
            }
        }
    }
}

static void ggml_compute_forward_randn(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, void * userdata)
{
    switch (dst->type)
    {
    case GGML_TYPE_F16:
        ggml_compute_forward_randn_f16(dst, ith, nth, userdata);
        break;
    case GGML_TYPE_F32:
        ggml_compute_forward_randn_f32(dst, ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}

static void ggml_custom_compute_forward_randn(struct ggml_tensor * dst , int ith, int nth, void * userdata)
{
    switch (dst->type)
    {
    case GGML_TYPE_F16:
        ggml_compute_forward_randn_f16(dst, ith, nth, userdata);
        break;
    case GGML_TYPE_F32:
        ggml_compute_forward_randn_f32(dst, ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}

template <class T> static void ggml_compute_forward_flip_T_0(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth)
{
    const ggml::tensor *src0 = a;

    GGML_TENSOR_UNARY_OP_LOCALS

    const int nr = (int)ggml::nrows(dst);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    T * dst_data  = (T *)((char *)  dst->data +  i3*nb3  + i2*nb2   + i1*nb1   + i0*nb0);
                    T * src_data  = (T *)((char *) src0->data +  i3*nb03 + i2*nb02  + i1*nb01  + (ne0 - 1 - i0)*nb00);

                    dst_data[0] = src_data[0];
                }
            }
        }
    }
}

static void ggml_compute_forward_flip(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, void * userdata)
{
    const int dim = (int)(intptr_t)userdata;
    CHATLLM_CHECK(dim == 0) << "flip: only dim = 0 is supported";
    switch (a->type)
    {
    case GGML_TYPE_F16:
        ggml_compute_forward_flip_T_0<ggml_fp16_t>(dst, a, ith, nth);
        break;
    case GGML_TYPE_F32:
        ggml_compute_forward_flip_T_0<float>(dst, a, ith, nth);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}