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

static void ggml_custom_compute_forward_zeroes(struct ggml_tensor * dst , int ith, int nth, void * userdata)
{
    memset(dst->data, 0, ggml::nbytes(dst));
}

static void vect_mult_vect_4x4(float r[4], const float v[4], const float m[4][4])
{
    r[0] = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0] + v[3] * m[3][0];
    r[1] = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1] + v[3] * m[3][1];
    r[2] = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] + v[3] * m[3][2];
    r[3] = v[0] * m[0][3] + v[1] * m[1][3] + v[2] * m[2][3] + v[3] * m[3][3];
}

// ref: https://www.ece.mcmaster.ca/~xwu/interp_1.pdf
void ggml_compute_forward_bicubic_f32_1(ggml::tensor * dst, const ggml::tensor * src0, int ith, int nth, bool align_corners)
{
    const static float mat_ib[4][4] =
    {
        {-1.0f/6,  1.0f/2, -1.0f/2,  1.0f/6},
        { 1.0f/2, -1.0f  ,  1.0f/2,  0.0f  },
        {-1.0f/3, -1.0f/2,  1.0f  , -1.0f/6},
        {   0   ,  1     ,    0   ,    0   },
    };

    float f[4][4];
    float lx[4];
    float ly[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    const float sf0 = (float)ne0/src0->ne[0];
    const float sf1 = (float)ne1/src0->ne[1];

    const float pixel_offset = align_corners ? 0.0f : 0.5f;

    for (int64_t i3 = 0; i3 < ne3; i3++)
    {
        for (int64_t i2 = ith; i2 < ne2; i2 += nth)
        {
            for (int64_t i1 = 0; i1 < ne1; i1++)
            {
                float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;

                int64_t y0 = (int64_t)floorf(y);
                int64_t y_ = y0 - 1;
                int64_t y1 = y0 + 1;
                int64_t y2 = y0 + 2;
                float vy[4];

                y = y - floorf(y);

                y_ = std::max(int64_t(0), std::min(y_, ne01 - 1));
                y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));
                y2 = std::max(int64_t(0), std::min(y2, ne01 - 1));

                vy[0] = y * y * y;
                vy[1] = y * y;
                vy[2] = y;
                vy[3] = 1.0f;

                // y = transpose(y * invB)
                vect_mult_vect_4x4(ly, vy, mat_ib);

                for (int64_t i0 = 0; i0 < ne0; i0++)
                {
                    float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;

                    int64_t x0 = (int64_t)floorf(x);
                    int64_t x_ = x0 - 1;
                    int64_t x1 = x0 + 1;
                    int64_t x2 = x0 + 2;
                    float vx[4];

                    x = x - floorf(x);

                    x_ = std::max(int64_t(0), std::min(x_, ne00 - 1));
                    x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                    x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));
                    x2 = std::max(int64_t(0), std::min(x2, ne00 - 1));

                    vx[0] = x * x * x;
                    vx[1] = x * x;
                    vx[2] = x;
                    vx[3] = 1.0f;

                    // x = x * invB
                    vect_mult_vect_4x4(lx, vx, mat_ib);

                    f[0][0] = *(const float *)((const char *)src0->data + x_*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[1][0] = *(const float *)((const char *)src0->data + x0*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[2][0] = *(const float *)((const char *)src0->data + x1*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[3][0] = *(const float *)((const char *)src0->data + x2*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[0][1] = *(const float *)((const char *)src0->data + x_*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[1][1] = *(const float *)((const char *)src0->data + x0*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[2][1] = *(const float *)((const char *)src0->data + x1*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[3][1] = *(const float *)((const char *)src0->data + x2*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[0][2] = *(const float *)((const char *)src0->data + x_*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[1][2] = *(const float *)((const char *)src0->data + x0*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[2][2] = *(const float *)((const char *)src0->data + x1*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[3][2] = *(const float *)((const char *)src0->data + x2*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[0][3] = *(const float *)((const char *)src0->data + x_*nb00 + y2*nb01 + i2*nb02 + i3*nb03);
                    f[1][3] = *(const float *)((const char *)src0->data + x0*nb00 + y2*nb01 + i2*nb02 + i3*nb03);
                    f[2][3] = *(const float *)((const char *)src0->data + x1*nb00 + y2*nb01 + i2*nb02 + i3*nb03);
                    f[3][3] = *(const float *)((const char *)src0->data + x2*nb00 + y2*nb01 + i2*nb02 + i3*nb03);

                    // val = lx . F . ly
                    float t[4];
                    vect_mult_vect_4x4(t, lx, f);
                    const float val = t[0] * ly[0] + t[1] * ly[1] + t[2] * ly[2] + t[3] * ly[3];

                    float * y_dst = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                    *y_dst = val;
                }
            }
        }
    }
}

// ref: https://en.wikipedia.org/wiki/Bicubic_interpolation
void ggml_compute_forward_bicubic_f32_2(ggml::tensor * dst, const ggml::tensor * src0, int ith, int nth, bool align_corners)
{
    const static float mat_b[4][4] =
    {
        { 1.0f,  0.0f,  0.0f,  0.0f},
        { 0.0f,  0.0f,  1.0f,  0.0f},
        {-3.0f,  3.0f, -2.0f, -1.0f},
        { 2.0f, -2.0f,  1.0f,  1.0f},
    };

    float f[4][4];
    float lx[4];
    float ly[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    const float sf0 = (float)ne0/src0->ne[0];
    const float sf1 = (float)ne1/src0->ne[1];

    const float pixel_offset = align_corners ? 0.0f : 0.5f;

    for (int64_t i3 = 0; i3 < ne3; i3++)
    {
        for (int64_t i2 = ith; i2 < ne2; i2 += nth)
        {
            for (int64_t i1 = 0; i1 < ne1; i1++)
            {
                float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;

                int64_t y0 = (int64_t)floorf(y);
                int64_t y_ = y0 - 1;
                int64_t y1 = y0 + 1;
                int64_t y2 = y0 + 2;
                float vy[4];

                y = y - floorf(y);

                y_ = std::max(int64_t(0), std::min(y_, ne01 - 1));
                y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));
                y2 = std::max(int64_t(0), std::min(y2, ne01 - 1));

                vy[3] = y * y * y;
                vy[2] = y * y;
                vy[1] = y;
                vy[0] = 1.0f;

                // y = transpose(y * invB)
                vect_mult_vect_4x4(ly, vy, mat_b);

                for (int64_t i0 = 0; i0 < ne0; i0++)
                {
                    float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;

                    int64_t x0 = (int64_t)floorf(x);
                    int64_t x_ = x0 - 1;
                    int64_t x1 = x0 + 1;
                    int64_t x2 = x0 + 2;
                    float vx[4];

                    x = x - floorf(x);

                    x_ = std::max(int64_t(0), std::min(x_, ne00 - 1));
                    x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                    x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));
                    x2 = std::max(int64_t(0), std::min(x2, ne00 - 1));

                    vx[3] = x * x * x;
                    vx[2] = x * x;
                    vx[1] = x;
                    vx[0] = 1.0f;

                    // x = x * invB
                    vect_mult_vect_4x4(lx, vx, mat_b);

                    #define ff(_i0, _i1)   (*(const float *)((const char *)src0->data + _i0*nb00 + _i1*nb01 + i2*nb02 + i3*nb03))

                    f[0][0] = ff(x0, y0);
                    f[0][1] = ff(x0, y1);
                    f[1][0] = ff(x1, y0);
                    f[1][1] = ff(x1, y1);

                    f[0][2] = (ff(x0, y1) - ff(x0, y_)) / 2;  // fy(0, 0)
                    f[0][3] = (ff(x0, y2) - ff(x0, y0)) / 2;  // fy(0, 1)

                    f[1][2] = (ff(x1, y1) - ff(x1, y_)) / 2;  // fy(1, 0)
                    f[1][3] = (ff(x1, y2) - ff(x1, y0)) / 2;  // fy(1, 1)

                    f[2][0] = (ff(x1, y0) - ff(x_, y0)) / 2;  // fx(0, 0)
                    f[2][1] = (ff(x1, y1) - ff(x_, y1)) / 2;  // fx(0, 1)
                    f[3][0] = (ff(x2, y0) - ff(x0, y0)) / 2;  // fx(1, 0)
                    f[3][1] = (ff(x2, y1) - ff(x0, y1)) / 2;  // fx(1, 1)

                    f[2][2] = ((ff(x1, y1) - ff(x_, y1)) - (ff(x1, y_) - ff(x_, y_))) / 4;  // fxy(0, 0)
                    f[2][3] = ((ff(x1, y2) - ff(x_, y2)) - (ff(x1, y0) - ff(x_, y0))) / 4;  // fxy(0, 1)
                    f[3][2] = ((ff(x2, y1) - ff(x0, y1)) - (ff(x2, y_) - ff(x0, y_))) / 4;  // fxy(1, 0)
                    f[3][3] = ((ff(x2, y2) - ff(x0, y2)) - (ff(x2, y0) - ff(x0, y0))) / 4;  // fxy(1, 1)

                    #undef ff

                    // val = lx . F . ly
                    float t[4];
                    vect_mult_vect_4x4(t, lx, f);
                    const float val = t[0] * ly[0] + t[1] * ly[1] + t[2] * ly[2] + t[3] * ly[3];

                    float * y_dst = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                    *y_dst = val;
                }
            }
        }
    }
}

// ref: https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
// see: https://github.com/pytorch/pytorch/blob/1cce73b5f4806f266fbbcf3383057af5f2e3a0aa/aten/src/ATen/native/mps/kernels/UpSample.metal#L229

template <typename accscalar_t>
accscalar_t cubic_convolution1(accscalar_t x, accscalar_t A)
{
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename accscalar_t>
accscalar_t cubic_convolution2(accscalar_t x, accscalar_t A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename accscalar_t>
void get_cubic_upsampling_coefficients(accscalar_t coeffs[4], accscalar_t t)
{
    accscalar_t A = -0.75f;

    accscalar_t x1 = t;
    coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0f, A);
    coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);

    // opposite coefficients
    accscalar_t x2 = 1.0f - t;
    coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
    coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0f, A);
}

static void ggml_compute_forward_bicubic_f32(ggml::tensor * dst, const ggml::tensor * src0, int ith, int nth, bool align_corners)
{
    float f[4][4];
    float lx[4];
    float ly[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    const float sf0 = align_corners ? (float)(src0->ne[0] - 1) / (ne0 - 1) : (float)src0->ne[0] / ne0;
    const float sf1 = align_corners ? (float)(src0->ne[1] - 1) / (ne1 - 1) : (float)src0->ne[1] / ne1;

    const float pixel_offset = align_corners ? 0.0f : 0.5f;

    for (int64_t i3 = 0; i3 < ne3; i3++)
    {
        for (int64_t i2 = ith; i2 < ne2; i2 += nth)
        {
            for (int64_t i1 = 0; i1 < ne1; i1++)
            {
                float y = ((float)i1 + pixel_offset) * sf1 - pixel_offset;

                int64_t y0 = (int64_t)floorf(y);
                int64_t y_ = y0 - 1;
                int64_t y1 = y0 + 1;
                int64_t y2 = y0 + 2;

                y = y - floorf(y);

                y_ = std::max(int64_t(0), std::min(y_, ne01 - 1));
                y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));
                y2 = std::max(int64_t(0), std::min(y2, ne01 - 1));

                get_cubic_upsampling_coefficients<float>(ly, y);

                for (int64_t i0 = 0; i0 < ne0; i0++)
                {
                    float x = ((float)i0 + pixel_offset) * sf0 - pixel_offset;

                    int64_t x0 = (int64_t)floorf(x);
                    int64_t x_ = x0 - 1;
                    int64_t x1 = x0 + 1;
                    int64_t x2 = x0 + 2;

                    x = x - floorf(x);

                    x_ = std::max(int64_t(0), std::min(x_, ne00 - 1));
                    x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                    x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));
                    x2 = std::max(int64_t(0), std::min(x2, ne00 - 1));

                    get_cubic_upsampling_coefficients<float>(lx, x);

                    f[0][0] = *(const float *)((const char *)src0->data + x_*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[1][0] = *(const float *)((const char *)src0->data + x0*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[2][0] = *(const float *)((const char *)src0->data + x1*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[3][0] = *(const float *)((const char *)src0->data + x2*nb00 + y_*nb01 + i2*nb02 + i3*nb03);
                    f[0][1] = *(const float *)((const char *)src0->data + x_*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[1][1] = *(const float *)((const char *)src0->data + x0*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[2][1] = *(const float *)((const char *)src0->data + x1*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[3][1] = *(const float *)((const char *)src0->data + x2*nb00 + y0*nb01 + i2*nb02 + i3*nb03);
                    f[0][2] = *(const float *)((const char *)src0->data + x_*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[1][2] = *(const float *)((const char *)src0->data + x0*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[2][2] = *(const float *)((const char *)src0->data + x1*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[3][2] = *(const float *)((const char *)src0->data + x2*nb00 + y1*nb01 + i2*nb02 + i3*nb03);
                    f[0][3] = *(const float *)((const char *)src0->data + x_*nb00 + y2*nb01 + i2*nb02 + i3*nb03);
                    f[1][3] = *(const float *)((const char *)src0->data + x0*nb00 + y2*nb01 + i2*nb02 + i3*nb03);
                    f[2][3] = *(const float *)((const char *)src0->data + x1*nb00 + y2*nb01 + i2*nb02 + i3*nb03);
                    f[3][3] = *(const float *)((const char *)src0->data + x2*nb00 + y2*nb01 + i2*nb02 + i3*nb03);

                    float t[4];
                    vect_mult_vect_4x4(t, lx, f);
                    const float val = t[0] * ly[0] + t[1] * ly[1] + t[2] * ly[2] + t[3] * ly[3];

                    float * y_dst = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                    *y_dst = val;
                }
            }
        }
    }
}


static void ggml_custom_group_index_boost(struct ggml_tensor * dst , const struct ggml_tensor * src0, int ith, int nth, void * userdata)
{
    const int group_size = (int)(intptr_t)(userdata);

    CHATLLM_CHECK(ggml::type_of(src0) == ggml::type::GGML_TYPE_I32);

    GGML_TENSOR_UNARY_OP_LOCALS

    for (int64_t i3 = 0; i3 < ne3; i3++)
    {
        for (int64_t i2 = ith; i2 < ne2; i2 += nth)
        {
            for (int64_t i1 = 0; i1 < ne1; i1++)
            {
                const int inc = (int)i1 * group_size;
                for (int64_t i0 = 0; i0 < ne0; i0++)
                {
                    const int * x_src = (const int *)((char *)src0->data + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
                            int * y_dst = (      int *)((char *) dst->data + i0*nb0  + i1*nb1  + i2*nb2  + i3*nb3);
                    y_dst[0] = x_src[0] + inc;
                }
            }
        }
    }
}


extern "C" void test_interp(void)
{
    ggml::tensor dst;
    ggml::tensor src;

    /*
    float input[5][2][3] = {
        {{ 0.4474,  1.9418,  0.5986},
         {-0.6370, -0.1142,  0.8451}},

        {{ 1.4538,  0.9776,  0.4454},
         {-0.5263,  0.3359,  0.2385}},

        {{-0.2411,  0.2098, -0.5852},
         {-0.1465, -1.0969, -0.3429}},

        {{-0.0207, -0.1396, -1.7902},
         { 0.0193,  0.3782, -1.1898}},

        {{-1.2405, -0.9490, -0.0471},
         {-1.2282,  0.6968, -0.7653}}};
    */

    float input[1][2][3] = {
        {{ 10.0f, 20.0f, 30.f},
         { 30.0f, 40.0f, 30.f}}};

    const int result_w = 4;
    const int result_h = 4;

    ggml::init_tensor(&src, ggml::type::GGML_TYPE_F32, 3, 2, 1, 1);
    ggml::init_tensor(&dst, ggml::type::GGML_TYPE_F32, result_w, result_h, 1, 1);

    float output[1][result_h][result_w] = {0};

    dst.data = (void *)output;
    src.data = (void *)input;

    ggml_compute_forward_bicubic_f32(&dst, &src, 0, 1, false);

    for (int i = 0; i < 1; i++)
    {
        printf("[\n");
        for (int j = 0; j < result_h; j++)
        {
            printf("\t[\n\t\t");
            for (int k = 0; k < result_w; k++)
            {
                printf("%0.4f, ", output[i][j][k]);
            }
            printf("\n\t],\n");
        }
        printf("],\n");
    }

    exit(0);
}

static void ggml_compute_forward_bicubic(struct ggml_tensor * dst , int ith, int nth, void * userdata)
{
    switch (dst->type)
    {
    case GGML_TYPE_F32:
        CHATLLM_CHECK(ggml::type_of(dst->src[0]) == ggml::type::GGML_TYPE_F32);
        ggml_compute_forward_bicubic_f32(dst, dst->src[0], ith, nth, userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}

static void ggml_custom_merge_patch(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, const ggml::merge_patch_param * param)
{
    const int kernel_height = param->merge_kernel_size[0];
    const int kernel_width  = param->merge_kernel_size[1];
    const int new_height = param->grid_h / kernel_height;
    const int new_width  = param->grid_w / kernel_width;

    CHATLLM_CHECK(ggml::get_dim(src, 1) == (int64_t)param->grid_h * param->grid_w);

    const int64_t nr  = ggml::nrows(dst);
    const int64_t dr  = (nr + nth - 1)/nth;
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = nb1 * kernel_width;
    const int64_t nb3 = nb2 * kernel_height;
    const int64_t nb4 = nb3 * new_width;

    for (int64_t i4 = 0; i4 < new_height; i4++)
    {
        for (int64_t i3 = 0; i3 < new_width; i3++)
        {
            for (int64_t i2 = 0; i2 < kernel_height; i2++)
            {
                for (int64_t i1 = 0; i1 < kernel_width; i1++)
                {
                    const int64_t ir = (i2 + i4 * kernel_height) * param->grid_w + (i1 + i3 * kernel_width);
                    if (ir < ir0) continue;
                    if (ir > ir1) break;

                    const void *src_data  = (void *)((char *)  src->data + ir*nb1);
                          void *dst_data  = (void *)((char *)  dst->data + i4*nb4 + i3*nb3  + i2*nb2 + i1*nb1);
                    memcpy(dst_data, src_data, nb1);
                }
            }
        }
    }
}

static void ggml_custom_merge_patch(struct ggml_tensor * dst, int ith, int nth, void * userdata)
{
    const ggml::merge_patch_param *param = (const ggml::merge_patch_param *)userdata;

    const struct ggml_tensor * a = dst->src[0];
    CHATLLM_CHECK(ggml::is_contiguous(a));
    CHATLLM_CHECK(ggml::get_dim(a, 3) == 1);
    CHATLLM_CHECK(ggml::get_dim(a, 2) == 1);

    ggml_custom_merge_patch(dst, a, ith, nth, param);
}

struct xielu_param
{
    float alpha_p;
    float alpha_n;
    float beta;
    float eps;
};

static void ggml_custom_xielu_f32(struct ggml_tensor * dst , const struct ggml_tensor * src0, int ith, int nth, const xielu_param *param)
{
    const float alpha_p = param->alpha_p;
    const float alpha_n = param->alpha_n;
    const float beta    = param->beta;
    const float eps     = param->eps;

    const int64_t nr  = ggml::nrows(dst);
    const int64_t dr  = (nr + nth - 1)/nth;
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    GGML_TENSOR_UNARY_OP_LOCALS

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                          float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    const float x = *src;
                    if (x > 0.0f)
                    {
                        *dst_data = alpha_p * x * x + beta * x;
                    }
                    else
                    {
                        *dst_data = (std::expm1f(MIN(x, eps)) - x) * alpha_n + beta * x;
                    }
                }
            }
        }
    }
}

static void ggml_custom_xielu(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata)
{
    switch (dst->type)
    {
    case GGML_TYPE_F32:
        CHATLLM_CHECK(ggml::type_of(a) == ggml::type::GGML_TYPE_F32);
        ggml_custom_xielu_f32(dst, a, ith, nth, (xielu_param *)userdata);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}