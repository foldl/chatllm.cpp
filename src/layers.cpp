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
#include "backend.h"

#include "ggml-cpu.h"

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace chatllm
{

#include "custom_ops.cpp"

    ggml::type ggml::type_of(const ggml::tensor *a)
    {
        return a->type;
    }

    ggml::type ggml::type_of(const ggml::tensor &a)
    {
        return a.type;
    }

    ggml::type ggml::type_fallback(ggml::type type, int64_t last_dim)
    {
        auto block_size = ggml_blck_size(type);
        return (last_dim % block_size) != 0 ? ggml::type::GGML_TYPE_F16 : type;
    }

    ggml::type ggml::parse(const std::string &type)
    {
        if (type == "f32")
            return type::GGML_TYPE_F32;
        if ((type == "q8") || (type == "q8_0"))
            return type::GGML_TYPE_Q8_0;
        if (type == "q4_0")
            return type::GGML_TYPE_Q4_0;
        if (type == "q4_1")
            return type::GGML_TYPE_Q4_1;
        return type::GGML_TYPE_F16;
    }

    bool ggml::is_view_op(ggml::tensor *a)
    {
        enum ggml_op op = a->op;
        return ggml_backend_is_view_op(op);
    }

    const char *ggml::op_name(ggml::tensor *a)
    {
        return ggml_op_name(a->op);
    }

    bool ggml::maybe_inplace_op(ggml::tensor *a)
    {
        return a->view_src != nullptr;
    }

    ggml::tensor *ggml::new_tensor_1d(ComputeContext *ctx, ggml::type type, int64_t ne0)
    {
        ggml::tensor *tensor = ggml_new_tensor_1d(ctx->get_ctx(), type, ne0);
        ctx->cb_new_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::new_tensor_2d(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1)
    {
        ggml::tensor *tensor = ggml_new_tensor_2d(ctx->get_ctx(), type, ne0, ne1);
        ctx->cb_new_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::new_tensor_3d(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1, int64_t ne2)
    {
        ggml::tensor *tensor = ggml_new_tensor_3d(ctx->get_ctx(), type, ne0, ne1, ne2);
        ctx->cb_new_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::new_tensor_4d(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
    {
        ggml::tensor *tensor = ggml_new_tensor_4d(ctx->get_ctx(), type, ne0, ne1, ne2, ne3);
        ctx->cb_new_tensor(tensor);
        return tensor;
    }

    void ggml::set_input(ggml::tensor *a)
    {
        ggml_set_input(a);
    }

    void ggml::set_output(ggml::tensor *a)
    {
        ggml_set_output(a);
    }

    void ggml::set_name(ggml::tensor *tensor, const char *name)
    {
        ggml_set_name(tensor, name);
    }

    const char *ggml::get_name(tensor *tensor)
    {
        return ggml_get_name(tensor);
    }

    void ggml::fill(ggml::tensor *a, uint8_t value, size_t offset, size_t size)
    {
        if (size == 0) size = ggml::nbytes(a);
        ggml_backend_tensor_memset(a, value, offset, size);
    }

    float ggml::at(ggml::tensor *a, int64_t i, int64_t j, int64_t k, int64_t l)
    {
        auto p = (char *)a->data + l*a->nb[3] + k*a->nb[2] + j*a->nb[1] + i*a->nb[0];
        switch (ggml::type_of(a))
        {
        case ggml::type::GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(ggml_fp16_t *)p);
        case ggml::type::GGML_TYPE_F32:
            return *(float *)p;
        default:
            CHATLLM_CHECK(false) << "unsupported type";
            return 0.0f;
        }
    }

    ggml::tensor *ggml::init_tensor(ggml::tensor *tensor,
        ggml::type    type,
        int           n_dims,
        const int64_t * ne)
    {
        ggml::tensor *result = tensor;

        *result = ggml::tensor {
            /*.type         =*/ type,
            /*.buffer       =*/ NULL,
            /*.ne           =*/ { 1, 1, 1, 1 },
            /*.nb           =*/ { 0, 0, 0, 0 },
            /*.op           =*/ GGML_OP_NONE,
            /*.op_params    =*/ { 0 },
            /*.flags        =*/ 0,
            /*.src          =*/ { NULL },
            /*.view_src     =*/ NULL,
            /*.view_offs    =*/ 0,
            /*.data         =*/ NULL,
            /*.name         =*/ { 0 },
            /*.extra        =*/ NULL,
            /*.padding      =*/ { 0 },
        };

        for (int i = 0; i < n_dims; i++) {
            result->ne[i] = ne[i];
        }

        change_type(result, type);

        return result;
    }

    void ggml::change_type(ggml::tensor  *tensor, ggml::type type)
    {
        tensor->type = type;
        tensor->nb[0] = ggml_type_size(type);
        tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/ggml_blck_size(type));
        for (int i = 2; i < GGML_MAX_DIMS; i++) {
            tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
        }
    }

    size_t ggml::element_size(const ggml::tensor * tensor)
    {
        return ggml_element_size(tensor);
    }

    size_t ggml::nbytes(const ggml::tensor * tensor)
    {
        return ggml_nbytes(tensor);
    }

    int ggml::n_dims(const ggml::tensor * tensor)
    {
        return ggml_n_dims(tensor);
    }

    int ggml::get_dim(const ggml::tensor * tensor, int dim)
    {
        return (int)tensor->ne[dim];
    }

    ggml::tensor *ggml::inplace_act(ComputeContext *ctx, ActFunc act, ggml::tensor *input)
    {
        ggml::tensor *tensor = nullptr;
        switch (act)
        {
        case ActFunc::GELU:
            tensor = ggml_gelu_inplace(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::SILU:
            tensor = ggml_silu_inplace(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::Tanh:
            tensor = ggml_tanh_inplace(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::RELU:
            tensor = ggml_relu_inplace(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::RELU2:
            tensor = ggml_relu_inplace(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            tensor = ggml_sqr_inplace(ctx->get_ctx(), tensor);
            ctx->cb_op_tensor(tensor);
            break;
        default:
            CHATLLM_CHECK(false) << "not implemented act function: " << act;
            return NULL;
        }
        return tensor;
    }

    ggml::tensor *ggml::act(ComputeContext *ctx, ActFunc act, ggml::tensor *input)
    {
        ggml::tensor *tensor = nullptr;
        switch (act)
        {
        case ActFunc::GELU:
            tensor = ggml_gelu(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::SILU:
            tensor = ggml_silu(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::Tanh:
            tensor = ggml_tanh(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::RELU:
            tensor = ggml_relu(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            break;
        case ActFunc::RELU2:
            tensor = ggml_relu(ctx->get_ctx(), input);
            ctx->cb_op_tensor(tensor);
            tensor = ggml_sqr(ctx->get_ctx(), tensor);
            ctx->cb_op_tensor(tensor);
            break;
        default:
            CHATLLM_CHECK(false) << "not implemented act function: " << act;
            return NULL;
        }
        return tensor;
    }

    ggml::tensor *ggml::scale(ComputeContext *ctx, ggml::tensor *a, float s)
    {
        ggml::tensor *tensor = ggml_scale(ctx->get_ctx(), a, s);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::scale_inplace(ComputeContext *ctx, ggml::tensor *a, float  s)
    {
        ggml::tensor *tensor = ggml_scale_inplace(ctx->get_ctx(), a, s);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::abs(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_abs(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::clamp(ComputeContext *ctx, ggml::tensor *a, float min, float max)
    {
        ggml::tensor *tensor = ggml_clamp(ctx->get_ctx(), a, min, max);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::get_rows(ComputeContext *ctx, ggml::tensor  * a, ggml::tensor  * b)
    {
        ggml::tensor *tensor = ggml_get_rows(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::mul_mat(ComputeContext *ctx, ggml::tensor  * a, ggml::tensor  * b)
    {
        ggml::tensor *tensor = ggml_mul_mat(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::mul_mat_id(ComputeContext *ctx, ggml::tensor *as, ggml::tensor *b, ggml::tensor  *ids)
    {
        ggml::tensor *tensor = ggml_mul_mat_id(ctx->get_ctx(), as, b, ids);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::add_inplace(ComputeContext *ctx, ggml::tensor * a, ggml::tensor * b)
    {
        ggml::tensor *tensor = ggml_add_inplace(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::sub_inplace(ComputeContext *ctx, ggml::tensor * a, ggml::tensor * b)
    {
        ggml::tensor *tensor = ggml_sub_inplace(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::view_1d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, size_t offset)
    {
        ggml::tensor *tensor = ggml_view_1d(ctx->get_ctx(), a, ne0, offset);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::view_2d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset)
    {
        ggml::tensor *tensor = ggml_view_2d(ctx->get_ctx(), a, ne0, ne1, nb1, offset);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::view_3d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset)
    {
        ggml::tensor *tensor = ggml_view_3d(ctx->get_ctx(), a, ne0, ne1, ne2, nb1, nb2, offset);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::view_4d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset)
    {
        ggml::tensor *tensor = ggml_view_4d(ctx->get_ctx(), a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::reshape_1d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0)
    {
        ggml::tensor *tensor = ggml_reshape_1d(ctx->get_ctx(), a, ne0);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::reshape_2d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1)
    {
        ggml::tensor *tensor = ggml_reshape_2d(ctx->get_ctx(), a, ne0, ne1);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::reshape_3d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1, int64_t ne2)
    {
        ggml::tensor *tensor = ggml_reshape_3d(ctx->get_ctx(), a, ne0, ne1, ne2);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::reshape_4d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
    {
        ggml::tensor *tensor = ggml_reshape_4d(ctx->get_ctx(), a, ne0, ne1, ne2, ne3);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::repeat(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b)
    {
        ggml::tensor *tensor = ggml_repeat(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::repeat_interleave(ComputeContext *ctx, ggml::tensor *a, int repeat, int dim)
    {
        CHATLLM_CHECK(dim == 0) << "only works for dim=0 now";
        CHATLLM_CHECK(a->ne[3] == 1) << "too many dimensions";

        ggml::tensor *b = ggml::new_tensor_4d(ctx, ggml::type_of(a), repeat * a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
        ggml::tensor *r = ggml::repeat(ctx, a, b);
        r = reshape_4d(ctx, r, a->ne[0], repeat,  a->ne[1], a->ne[2]);
        r = permute(ctx, r, 1, 0, 2, 3);
        r = cont(ctx, r);
        r = reshape_3d(ctx, r, a->ne[0] * repeat, a->ne[1], a->ne[2]);

        return r;
    }

    ggml::tensor *ggml::permute(ComputeContext *ctx, ggml::tensor *a, int axis0, int axis1, int axis2, int axis3)
    {
        ggml::tensor *tensor = ggml_permute(ctx->get_ctx(), a, axis0, axis1, axis2, axis3);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::flip(ComputeContext *ctx, ggml::tensor *a, int dim)
    {
        ggml::tensor *tensor = ggml::map_custom1(ctx, a, ggml_compute_forward_flip, GGML_N_TASKS_MAX, (void *)(intptr_t)dim);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::norm(ComputeContext *ctx, ggml::tensor *a, float eps)
    {
        ggml::tensor *tensor = ggml_norm(ctx->get_ctx(), a, eps);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::norm_inplace(ComputeContext *ctx, ggml::tensor *a, float eps)
    {
        ggml::tensor *tensor = ggml_norm_inplace(ctx->get_ctx(), a, eps);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::rms_norm_inplace(ComputeContext *ctx, ggml::tensor *a, float eps)
    {
        ggml::tensor *tensor = ggml_rms_norm_inplace(ctx->get_ctx(), a, eps);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::rms_norm(ComputeContext *ctx, ggml::tensor *a, float eps)
    {
        ggml::tensor *tensor = ggml_rms_norm(ctx->get_ctx(), a, eps);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::simple_norm(ComputeContext *ctx, ggml::tensor *a, float eps)
    {
        int hidden_size = (int)a->ne[0];
        float scale = 1.0f / sqrtf((float)hidden_size);
        ggml::tensor *tensor = ggml::rms_norm(ctx, a, eps);
        tensor = ggml::scale(ctx, tensor, scale);
        return tensor;
    }

    ggml::tensor *ggml::rope(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, int n_dims, int mode)
    {
        ggml::tensor *tensor = ggml_rope(ctx->get_ctx(), a, b, n_dims, mode);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::rope_ext(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, ggml::tensor *c,
                                            int   n_dims, int   mode, int   n_ctx_orig,
                                            float freq_base, float freq_scale, float ext_factor,
                                            float attn_factor, float beta_fast, float beta_slow,
                                            const int *sections)
    {
        ggml::tensor *tensor = ggml_rope_multi(ctx->get_ctx(), a, b, c,
                                n_dims, (int *)sections, mode, n_ctx_orig,
                                freq_base, freq_scale, ext_factor,
                                attn_factor, beta_fast, beta_slow);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::rope_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, int n_dims, int mode)
    {
        ggml::tensor *tensor = ggml_rope_inplace(ctx->get_ctx(), a, b, n_dims, mode);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::rope_ext_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, ggml::tensor *c,
                                            int   n_dims, int   mode, int   n_ctx_orig,
                                            float freq_base, float freq_scale, float ext_factor,
                                            float attn_factor, float beta_fast, float beta_slow,
                                            const int *sections)
    {
        ggml::tensor *tensor = ggml_rope_ext_inplace(ctx->get_ctx(), a, b, c,
                                n_dims, mode, n_ctx_orig,
                                freq_base, freq_scale, ext_factor,
                                attn_factor, beta_fast, beta_slow);

                                #if (0)
        ggml::tensor *tensor = ggml_rope_multi_inplace(ctx->get_ctx(), a, b, c,
                                n_dims, (int *)sections, mode, n_ctx_orig,
                                freq_base, freq_scale, ext_factor,
                                attn_factor, beta_fast, beta_slow);
                                #endif
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::soft_max(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_soft_max(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::soft_max_inplace(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_soft_max_inplace(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml_tensor  *ggml::soft_max_ext(ComputeContext *ctx,  ggml::tensor *a,  ggml::tensor *mask, float scale, float max_bias)
    {
        ggml::tensor *tensor = ggml_soft_max_ext(ctx->get_ctx(), a, mask, scale, max_bias);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::sigmoid(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_sigmoid(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::diag_mask_inf(ComputeContext *ctx, ggml::tensor *a, int n_past)
    {
        ggml::tensor *tensor = ggml_diag_mask_inf(ctx->get_ctx(), a, n_past);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::diag_mask_inf_inplace(ComputeContext *ctx, ggml::tensor *a, int n_past)
    {
        ggml::tensor *tensor = ggml_diag_mask_inf_inplace(ctx->get_ctx(), a, n_past);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::mul(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b)
    {
        ggml::tensor *tensor = ggml_mul(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::div(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b)
    {
        ggml::tensor *tensor = ggml_div(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::sin(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_sin(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::cos(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_cos(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::square(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_sqr(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::sqrt(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_sqrt(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::log(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_log(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::tanh(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_tanh(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::randn_inplace(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml::map_custom1_inplace(ctx, a, ggml_compute_forward_randn, GGML_N_TASKS_MAX, NULL);
        return tensor;
    }

    ggml::tensor *ggml::sum_rows(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_sum_rows(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::mean(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_mean(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::top_k(ComputeContext *ctx, ggml::tensor *a, int k)
    {
        ggml::tensor *tensor = ggml_top_k(ctx->get_ctx(), a, k);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::mul_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b)
    {
        ggml::tensor *tensor = ggml_mul_inplace(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::cpy(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b)
    {
        ggml::tensor *tensor = ggml_cpy(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::dup(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_dup(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::cont(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_cont(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::transpose(ComputeContext *ctx, ggml::tensor *a)
    {
        ggml::tensor *tensor = ggml_transpose(ctx->get_ctx(), a);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::concat(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, int dim)
    {
        ggml::tensor *tensor = ggml_concat(ctx->get_ctx(), a, b, dim);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::add(ComputeContext *ctx, ggml::tensor * a, ggml::tensor * b)
    {
        ggml::tensor *tensor = ggml_add(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::sub(ComputeContext *ctx, ggml::tensor * a, ggml::tensor * b)
    {
        ggml::tensor *tensor = ggml_sub(ctx->get_ctx(), a, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::conv_2d(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *b)
    {
        ggml::tensor *tensor = ggml_conv_2d_sk_p0(ctx->get_ctx(), kernel, b);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::conv_1d(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding, int dilation)
    {
        ggml::tensor *tensor = ggml_conv_1d(ctx->get_ctx(), kernel, data, stride, padding, dilation);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::conv_1d_depthwise(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding, int dilation)
    {
        ggml::tensor *tensor = ggml_conv_1d_dw(ctx->get_ctx(), kernel, data, stride, padding, dilation);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::conv_transposed_1d( ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding, int dilation)
    {
        ggml::tensor *tensor = ggml_conv_transpose_1d(ctx->get_ctx(), kernel, data, stride, padding, dilation);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::map_custom1(ComputeContext *ctx, ggml::tensor *a, const ggml_custom1_op_t fun, int n_tasks, void *userdata)
    {
        ggml::tensor *tensor = ggml_map_custom1(ctx->get_ctx(), a, fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::map_custom1_inplace(ComputeContext *ctx, ggml::tensor *a, const ggml_custom1_op_t fun, int n_tasks, void *userdata)
    {
        ggml::tensor *tensor = ggml_map_custom1_inplace(ctx->get_ctx(), a, fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::map_custom2(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, const ggml_custom2_op_t fun, int n_tasks, void *userdata)
    {
        ggml::tensor *tensor = ggml_map_custom2(ctx->get_ctx(), a, b, fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::map_custom2_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, const ggml_custom2_op_t fun, int n_tasks, void *userdata)
    {
        ggml::tensor *tensor = ggml_map_custom2_inplace(ctx->get_ctx(), a, b, fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::map_custom3(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, ggml::tensor *c, const ggml_custom3_op_t fun, int n_tasks, void *userdata)
    {
        ggml::tensor *tensor = ggml_map_custom3(ctx->get_ctx(), a, b, c, fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::map_custom3_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, ggml::tensor *c, const ggml_custom3_op_t fun, int n_tasks, void *userdata)
    {
        ggml::tensor *tensor = ggml_map_custom3_inplace(ctx->get_ctx(), a, b, c, fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    ggml::tensor *ggml::custom(ComputeContext *ctx, ggml_custom_op_t fun, int n_tasks, void *userdata, std::vector<ggml::tensor *>inputs, ggml::type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
    {
        ggml::tensor *tensor = ggml_custom_4d(ctx->get_ctx(), type, ne0, ne1, ne2, ne3, inputs.data(), (int)inputs.size(), fun, n_tasks, userdata);
        ctx->cb_op_tensor(tensor);
        return tensor;
    }

    struct ggml_cgraph *ggml::new_graph_custom(ComputeContext *ctx, size_t size, bool grads)
    {
        return ggml_new_graph_custom(ctx->get_ctx(), size, grads);
    }

    void ggml::build_forward_expand(ComputeContext *ctx, ggml::tensor * tensor)
    {
        ggml_build_forward_expand(ctx->get_cgraph(), tensor);
    }

    void ggml::mul_mat_set_prec(ggml::tensor *a, ggml::prec prec)
    {
        ggml_mul_mat_set_prec(a, prec);
    }

    bool ggml::is_contiguous(const ggml::tensor *k)
    {
        return ggml_is_contiguous(k);
    }

    int64_t ggml::nrows(const ggml::tensor *tensor)
    {
        return ggml_nrows(tensor);
    }

    int64_t ggml::nelements(const ggml::tensor *tensor)
    {
        return ggml_nelements(tensor);
    }

    ggml::tensor *Sequential::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *output = input;
        for (auto &block : blocks)
        {
            output = block->forward(ctx, output);
        }
        return output;
    }

    ggml::tensor *Sequential::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        ggml::tensor *output = input;
        for (auto &block : blocks)
        {
            output = block->forward(ctx, output, n_past);
        }
        return output;
    }

    void Sequential::add_block(Block *block)
    {
        blocks.emplace_back(block);
    }

    void Sequential::add_block(std::unique_ptr<Block> block)
    {
        blocks.push_back(std::move(block));
    }

    int64_t Sequential::get_param_num(bool effective_only) const
    {
        int64_t total = 0;
        for (auto &block : blocks)
            total += block->get_param_num(effective_only);
        return total;
    }

    void Sequential::load(const std::string &path, TensorLoader *loader)
    {
        this->path = path;
        for (size_t i = 0; i < blocks.size(); i++)
        {
            std::string block_path = path + std::to_string(i) + ".";
            auto block = blocks[i].get();
            block->load(block_path, loader);
        }
    }

    Conv1D::Conv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding,
        int dilation, int groups, bool bias)
      : Conv1D(ctx, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias ? out_channels : 0)
    {}

    Conv1D::Conv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding,
        int dilation, int groups, int bias_dim)
     : weight(ggml::new_tensor_3d(ctx, ggml::type_fallback(ctx->dtype, kernel_size), kernel_size, in_channels / groups, out_channels)),
       bias(bias_dim > 0 ? ggml::new_tensor_1d(ctx, GGML_TYPE_F32, bias_dim) : nullptr),
       in_channels(in_channels),
       out_channels(out_channels),
       kernel_size(kernel_size),
       stride(stride),
       padding(padding),
       dilation(dilation),
       groups(groups)
    {
    }

    ggml::tensor *Conv1D::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *output = nullptr;
        if (groups == 1)
        {
            output = ggml::conv_1d(ctx, weight, input, stride, padding, dilation);
        }
        else if (groups == in_channels)
        {
            CHATLLM_CHECK((out_channels % in_channels) == 0) << "not implemented groups: " << groups;
            output = ggml::conv_1d_depthwise(ctx, weight, input, stride, padding, dilation);
        }
        else
        {
            CHATLLM_CHECK(false) << "not implemented groups: " << groups;
        }

        if (bias)
        {
            auto bias_view = ggml::view_2d(ctx, bias, 1, bias->ne[0], ggml::element_size(bias), 0);
            output = ggml::add(ctx, output, bias_view);
        }
        return output;
    }

    int64_t Conv1D::get_param_num(bool effective_only) const
    {
        int64_t total = ggml::nelements(weight);
        if (bias)
            total += ggml::nelements(bias);
        return total;
    }

    void Conv1D::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        loader->read_tensor(path + "weight", weight);
        if (bias)
            loader->read_tensor(path + "bias", bias);
    }

    ConvTransposed1D::ConvTransposed1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding,
        int output_padding, int dilation, int groups, bool bias)
        : Conv1D(ctx, out_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias ? out_channels : 0),
          output_padding(output_padding)
    {}

    ggml::tensor *ConvTransposed1D::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        // TODO: Optimization needed.
        // `ggml::conv_transposed_1d` is not fully functional. So we do it manually for some configurations.
        ggml::tensor *output = nullptr;
        if ((padding == 0) && (output_padding == 0) && (dilation == 1) && (groups == 1))
        {
            output = ggml::conv_transposed_1d(ctx, weight, input, stride, 0, 0);
        }
        else if (groups == 1)
        {
            int p = (kernel_size - 1) * (dilation + 1) / 2 - padding;
            CHATLLM_CHECK(output_padding == 0);

            ggml::tensor *fract_input = input;
            if (stride > 1)
            {
                CHATLLM_CHECK(ggml::get_dim(input, 3) == 1);
                auto view = ggml::reshape_4d(ctx, input, 1, ggml::get_dim(input, 0), ggml::get_dim(input, 1), ggml::get_dim(input, 2));
                auto zeros = ggml::new_tensor_4d(ctx, ggml::type_of(view), stride - 1, ggml::get_dim(input, 0), ggml::get_dim(input, 1), ggml::get_dim(input, 2));
                zeros = ggml::scale_inplace(ctx, zeros, 0.0);
				fract_input = ggml::concat(ctx, view, zeros, 0); // [stride, w[0], w[1], w[2]]

				fract_input = ggml::view_3d(ctx, fract_input, stride * (ggml::get_dim(input, 0) - 1) + 1, ggml::get_dim(input, 1), ggml::get_dim(input, 2),
                    ggml::element_size(input) * stride * ggml::get_dim(input, 0),
                    ggml::element_size(input) * stride * ggml::get_dim(input, 0) * ggml::get_dim(input, 1),
                    0);
                fract_input = ggml::cont(ctx, fract_input);
            }

            auto view = ggml::permute(ctx, weight, 0, 2, 1, 3);
            view =  ggml::flip(ctx, view, 0);

            output = ggml::conv_1d(ctx, view, fract_input, 1, p, 1);
        }
        else
        {
            CHATLLM_CHECK(false) << "not implemented groups: " << groups;
        }

		if (bias)
		{
			auto bias_view = ggml::view_2d(ctx, bias, 1, bias->ne[0], ggml::element_size(bias), 0);
			output = ggml::add(ctx, output, bias_view);
		}
        return output;
    }

    ggml::tensor *Unary::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        switch (op)
        {
        case Op::Sin:
            return ggml::sin(ctx, input);
        case Op::Cos:
            return ggml::cos(ctx, input);
        case Op::Square:
            return ggml::square(ctx, input);
        case Op::Sqrt:
            return ggml::sqrt(ctx, input);
        case Op::Log:
            return ggml::log(ctx, input);
        case Op::Tanh:
            return ggml::tanh(ctx, input);
        default:
            CHATLLM_CHECK(false) << "not implemented unary op: " << op;
            return nullptr;
        }
    }

    ggml::tensor *Unary::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        return forward(ctx, input);
    }

    ggml::tensor *Embedding::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *output = (ggml::n_dims(input) == 1) && (ggml::type::GGML_TYPE_I32 == input->type)
                                ? ggml::get_rows(ctx, weight, input)
                                : ggml::mul_mat(ctx, weight, input);
        return output;
    }

    ggml::tensor *Embedding::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        return forward(ctx, input);
    }

    void Embedding::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        loader->read_tensor(path + "weight", weight);
    }

    ggml::tensor *RobertaEmbedding::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        int qlen = (int)input->ne[0];

        ggml::tensor *idx = ggml::view_1d(ctx, indices, qlen, 0);

        ggml::tensor *output1 = ggml::get_rows(ctx, word_weight, input);
        ggml::tensor *output2 = ggml::get_rows(ctx, position_weight, idx);

        ggml::tensor *output = ggml::add_inplace(ctx, output1, output2);

        output = ln.forward(ctx, output);
        return output;
    }

    void RobertaEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        loader->read_tensor(path + "word_embeddings.weight", word_weight);
        loader->read_tensor(path + "position_embeddings.weight", position_weight);
        ln.load(path + "ln.", loader);
    }

    ggml::tensor *Linear::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        // input: [seqlen, in_features]
        ggml::tensor *output = ggml::mul_mat(ctx, weight, input); // [seqlen, out_features]
        ggml::mul_mat_set_prec(output, prec);
        if (bias)
        {
            output = ggml::add_inplace(ctx, output, bias);
        }
        return output;
    }

    void Linear::load(const std::string &path, TensorLoader *loader)
    {
        loader->read_tensor(path + "weight", weight);
        if (bias)
            loader->read_tensor(path + "bias", bias);
    }

    ggml::tensor *MultiLinear::forward(ComputeContext *ctx, ggml::tensor *input, ggml::tensor *selected)
    {
        return ggml::mul_mat_id(ctx, weight, input, selected);
    }

    ggml::tensor *LayerNorm::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        // input: [seqlen, normalized_shape]
        ggml::tensor *output = ggml::norm(ctx, input, eps);
        output = ggml::mul_inplace(ctx, output, weight);
        if (bias)
            output = ggml::add_inplace(ctx, output, bias);
        return output;
    }

    void LayerNorm::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        loader->read_tensor(path + "weight", weight);
        if (bias) loader->read_tensor(path + "bias", bias);
    }

    ggml::tensor *RMSNorm::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *output = inplace ? ggml::rms_norm_inplace(ctx, input, eps) : ggml::rms_norm(ctx, input, eps);
        output = inplace ? ggml::mul_inplace(ctx, output, weight) : ggml::mul(ctx, output, weight);
        return output;
    }

    void RMSNorm::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        loader->read_tensor(path + "weight", weight);
    }

    ggml::tensor *L2Norm::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *output = inplace ? ggml::rms_norm_inplace(ctx, input, eps) : ggml::rms_norm(ctx, input, eps);
        return output;
    }

    FIR2::FIR2(InitContext *ctx, int dim, int hidden_size)
        : weight(ggml::new_tensor_2d(ctx, GGML_TYPE_F32, 2, dim)),
          x0(ggml::new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size)),
          dim(dim)
    {
        ctx->get_allocator()->alloc(x0);
    }

    ggml::tensor *FIR2::forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
    {
        auto last_x = x0;
        if (n_past == 0)
        {
            last_x = ggml::scale_inplace(ctx, x0, 0.0f);
            ggml::build_forward_expand(ctx, last_x);
        }

        last_x = ggml::reshape_4d(ctx, last_x, input->ne[0], input->ne[1], input->ne[2], input->ne[3]);

        CHATLLM_CHECK(input->ne[1] == dim) << "ne[1] must == dim: " << input->ne[1] << ", " << dim;

        auto w0 = ggml::view_2d(ctx, weight, 1, dim, 2 * ggml::element_size(weight), 0);
        auto w1 = ggml::view_2d(ctx, weight, 1, dim, 2 * ggml::element_size(weight), ggml::element_size(weight));
        auto x0w = ggml::mul(ctx, last_x, w0);
        auto x1w = ggml::mul(ctx, input, w1);

        auto r = ggml::add(ctx, x0w, x1w);

        ggml::build_forward_expand(ctx, r);

        {
            last_x = ggml::cpy(ctx, input, last_x);
            ggml::build_forward_expand(ctx, last_x);
        }

        return r;
    }

    void FIR2::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        loader->read_tensor(path, weight);
    }

    ggml::tensor *RobertaPooler::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        int hidden_size = (int)hidden_states->ne[0];

        // We "pool" the model by simply taking the hidden state corresponding to the first token.
        ggml::tensor *first_token_tensor = ggml::view_2d(ctx, hidden_states, hidden_size, 1,
                                                      hidden_size * ggml::element_size(hidden_states), 0);
        ggml::tensor *output = dense.forward(ctx, first_token_tensor);
        output = ggml::inplace_act(ctx, act, output);
        return output;
    }

    void RobertaPooler::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        dense.load(path + "dense.", loader);
    }

    ggml::tensor *RobertaClassificationHead::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        int hidden_size = (int)hidden_states->ne[0];

        // We "pool" the model by simply taking the hidden state corresponding to the first token.
        ggml::tensor *first_token_tensor = ggml::view_2d(ctx, hidden_states, hidden_size, 1,
                                                      hidden_size * ggml::element_size(hidden_states), 0);
        ggml::tensor *output = dense.forward(ctx, first_token_tensor);
        output = ggml::inplace_act(ctx, act, output);
        output = out_proj.forward(ctx, output);
        output = ggml::sigmoid(ctx, output);
        return output;
    }

    void RobertaClassificationHead::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        dense.load(path + "dense.", loader);
        out_proj.load(path + "out_proj.", loader);
    }

    ggml::tensor *BCEFinalNorm::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        int hidden_size = (int)hidden_states->ne[0];
        ggml::tensor *first_token_tensor = ggml::view_1d(ctx, hidden_states, hidden_size, 0);
        ggml::tensor *output = ggml::simple_norm(ctx, first_token_tensor, eps);
        return output;
    }

    void MiniCPMMeanPooling::set_max_length(InitContext *ctx, int max_length)
    {
        std::vector<float> v_pos(max_length);
        pos = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, max_length);
        ctx->get_allocator()->alloc(pos);
        fill_pos_vector(ctx, v_pos, pos, 1, max_length);
    }

    ggml::tensor *MiniCPMMeanPooling::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        const int qlen = (int)hidden_states->ne[1];
        ggml::tensor *one_based_pos = ggml::view_1d(ctx, pos, qlen, ggml::element_size(pos));

        ggml::tensor *output = RMSNorm::forward(ctx, hidden_states);

        output = ggml::transpose(ctx, output);
        output = ggml::cont(ctx, output);
        output = ggml::mul(ctx, output, one_based_pos);
        output = ggml::mean(ctx, output);
        output = ggml::transpose(ctx, output);
        output = ggml::simple_norm(ctx, output, eps);
        return output;
    }

    void fill_pos_vector(ComputeContext *ctx, std::vector<int> &v_pos, ggml::tensor *pos, int n_past, int qlen)
    {
        for (int i = 0; i < qlen; i++)
            v_pos[i] = n_past + i;
        pos->ne[0] = qlen;
        Backend::write_tensor_data(pos, v_pos.data(), 0, qlen * sizeof(v_pos[0]));
    }

    void fill_pos_vector(ComputeContext *ctx, std::vector<float> &v_pos, ggml::tensor *pos, int n_past, int qlen)
    {
        for (int i = 0; i < qlen; i++)
            v_pos[i] = (float)(n_past + i);
        pos->ne[0] = qlen;
        Backend::write_tensor_data(pos, v_pos.data(), 0, qlen * sizeof(v_pos[0]));
    }

    ggml::tensor *GLMSelfAttention::apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const
    {
        return ggml::map_custom2_inplace(ctx, k, past, ggml_compute_forward_chatglm1_rope, GGML_N_TASKS_MAX, (void *)this);    // [qlen, heads, head_size]
    }
    ggml::tensor *GLMSelfAttention::apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const
    {
        return ggml::map_custom2_inplace(ctx, q, past, ggml_compute_forward_chatglm1_rope, GGML_N_TASKS_MAX, (void *)this);    // [qlen, heads, head_size]
    }

    ggml::tensor *GLMBlock::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        float alpha = sqrtf(2.f * (float)num_hidden_layers);

        ggml::tensor *attn_input = input_layernorm.forward(ctx, hidden_states);
        ggml::tensor *attn_output = attention.forward(ctx, attn_input, n_past);
        ggml::build_forward_expand(ctx, attn_output);
        hidden_states =
            ggml::add_inplace(ctx, ggml::scale_inplace(ctx, attn_input, alpha), attn_output);

        ggml::tensor *mlp_input = post_attention_layernorm.forward(ctx, hidden_states);
        ggml::tensor *mlp_output = mlp.forward(ctx, mlp_input);
        ggml::build_forward_expand(ctx, mlp_output);
        ggml::tensor *output =
            ggml::add_inplace(ctx, ggml::scale_inplace(ctx, mlp_input, alpha), mlp_output);
        return output;
    }

    ggml::tensor *BaseConsolidatedQKVAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int hidden_size = (int)hidden_states->ne[0];
        const int qlen = (int)hidden_states->ne[1];
        const int head_size = hidden_size / num_attention_heads;

        before_forward(ctx, n_past, qlen);

        ggml::tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, hidden + 2 * kv_hidden]

        ggml::tensor *tmpv =
            ggml::view_2d(ctx, qkv, head_size * num_kv_heads, qlen,
                        qkv->nb[1],
                        head_size * (num_attention_heads + num_kv_heads) * ggml::element_size(qkv)); // [qlen, kv_hidden]

        ggml::tensor *key_layer =
            ggml::view_3d(ctx, qkv, head_size, num_kv_heads, qlen, head_size * ggml::element_size(qkv),
                         qkv->nb[1], hidden_size * ggml::element_size(qkv)); // [qlen, kv_heads, head_size]

        ggml::tensor *query_layer =
            ggml::view_3d(ctx, qkv, head_size, num_attention_heads, qlen, head_size * ggml::element_size(qkv),
                         qkv->nb[1], 0); // [qlen, heads, head_size]

        ggml::tensor *scores = cross_attention_3d(ctx, hidden_size, n_past, qlen, query_layer, key_layer, tmpv);

        ggml::tensor *attn_output = dense.forward(ctx, scores);
        return attn_output;
    }

    void BaseConsolidatedQKVAttention::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        query_key_value.load(path + "query_key_value.", loader);
        dense.load(path + "dense.", loader);
    }

    ggml::tensor *GLM2MLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *output = dense_h_to_4h.forward(ctx, hidden_states);

        // swiglu activation
        ggml::tensor *x0 = ggml::view_2d(ctx, output, output->ne[0] / 2, output->ne[1], output->nb[1], 0);
        ggml::tensor *x1 = ggml::view_2d(ctx, output, output->ne[0] / 2, output->ne[1], output->nb[1],
                                       output->ne[0] / 2 * ggml::element_size(output));
        output = ggml::mul_inplace(ctx, ggml::inplace_act(ctx, ActFunc::SILU, ggml::cont(ctx, x0)), x1);

        output = dense_4h_to_h.forward(ctx, output);
        return output;
    }

    ggml::tensor *TheMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *intermediate = fc0.forward(ctx, hidden_states);
        intermediate = ggml::inplace_act(ctx, act, intermediate);
        ggml::tensor *output = fc1.forward(ctx, intermediate);
        return output;
    }

    void TheMLP::set_prec(ggml::prec prec)
    {
        Block::set_prec(prec);
        fc0.set_prec(prec);
        fc1.set_prec(prec);
    }

    void TheMLP::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        fc0.load(path + "fc0.", loader);
        fc1.load(path + "fc1.", loader);
    }

    ggml::tensor *BaseMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *act = ggml::inplace_act(ctx, this->act, gate_proj.forward(ctx, hidden_states));
        ggml::tensor *proj = up_proj.forward(ctx, hidden_states);

        ggml::tensor *output = ggml::mul_inplace(ctx, act, proj);
        output = down_proj.forward(ctx, output);
        return output;
    }

    void BaseMLP::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        up_proj.load(path + "up_proj.", loader);
        down_proj.load(path + "down_proj.", loader);
        gate_proj.load(path + "gate_proj.", loader);
    }

    ggml::tensor *CoreAttention::attn_scores_to_probs(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
        ggml::tensor *attn_scores)
    {
        const int head_size = hidden_size / num_attention_heads;

        if (attn_scaling)
        {
            if (attn_scaling_factor > 0)
                attn_scores = ggml::scale_inplace(ctx, attn_scores, attn_scaling_factor);
            else
                attn_scores = ggml::scale_inplace(ctx, attn_scores, 1.f / sqrtf((float)head_size));
        }

        if (attn_scores_pp)
            attn_scores = attn_scores_pp->forward(ctx, attn_scores);

        attn_scores = apply_pos_embedding_kq(ctx, attn_scores, hidden_size, qlen, pos);

        // attn_masked = mask_past(attn_scores)
        ggml::tensor * attn_masked = causal ? ggml::diag_mask_inf_inplace(ctx, attn_scores, n_past)
                                                  : attn_scores;

        // attn_probs = soft_max(attn_masked)
        ggml::tensor * attn_probs = ggml::soft_max_inplace(ctx, attn_masked);

        return attn_probs;
    }

    ggml::tensor *CoreAttention::calc_attn_scores(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
        ggml::tensor *key_layer, ggml::tensor *query_layer, ggml::tensor *value_layer)
    {
        const int head_size = hidden_size / num_attention_heads;

        // note auto-broadcasting in ggml_mul_mat for `repeat > 1`
        ggml::tensor *attn_scores = ggml::mul_mat(ctx, key_layer, query_layer); // [heads, qlen, klen]

        // default to F32 here
        ggml::mul_mat_set_prec(attn_scores, GGML_PREC_F32);

        // attn_probs = soft_max(attn_masked)
        ggml::tensor * attn_probs = attn_scores_to_probs(ctx, hidden_size, n_past, qlen, attn_scores);

        ggml::tensor *context_layer = ggml::mul_mat(ctx, value_layer, attn_probs); // [heads, qlen, head_size]
        last_attn_scores = ggml::reshape_2d(ctx,
            ggml::cont(ctx, ggml::permute(ctx, context_layer, 0, 2, 1, 3)),
            hidden_size, qlen);

        return last_attn_scores;
    }

    ggml::tensor *CoreAttention::cross_attention_after_pe(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *query_layer, ggml::tensor *key_layer, ggml::tensor *v)
    {
        const int head_size = hidden_size / num_attention_heads;

        if (!attn_scaling)
            query_layer = ggml::scale(ctx, query_layer, 1.f / sqrtf((float)head_size));

        // store key and value to memory
        save_to_cache(ctx, n_past, qlen, key_layer, v);

        query_layer = ggml::permute(ctx, query_layer, 0, 2, 1, 3);                     // [heads, qlen, head_size]

        key_layer = get_k_from_cache(ctx, hidden_size, n_past, qlen);

        ggml::tensor * value_layer = get_v_from_cache(ctx, hidden_size, n_past, qlen);

        ggml::tensor *attn_scores = calc_attn_scores(ctx, hidden_size, n_past, qlen, key_layer, query_layer, value_layer);
        return attn_scores;
    }

    ggml::tensor *CoreAttention::cross_attention_3d(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *query_layer, ggml::tensor *key_layer, ggml::tensor *v)
    {
        // [qlen, heads, head_size]
        key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, pos);

        // [qlen, heads, head_size]
        query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, pos);

        ggml::tensor *attn_scores = cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

        return attn_scores;
    }

    ggml::tensor *CoreAttention::cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k, ggml::tensor *v)
    {
        const int head_size = hidden_size / num_attention_heads;

        // [qlen, heads, head_size]
        ggml::tensor * key_layer = ggml::reshape_3d(ctx, k, head_size, num_kv_heads, qlen);
        key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, pos);

        // [qlen, heads, head_size]
        ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, head_size, num_attention_heads, qlen);
        query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, pos);

        ggml::tensor *attn_scores = cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

        return attn_scores;
    }

    void CoreAttention::allocate_pos_tensor(InitContext *ctx)
    {
        pos = ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length);
        v_pos.resize(max_length);
        ctx->get_allocator()->alloc(pos);
    }

    void CoreAttention::prepare_pos_tensor(ComputeContext *ctx, const int n_past, const int qlen)
    {
        fill_pos_vector(ctx, v_pos, pos, n_past, qlen);
    }

    void CoreAttention::before_forward(ComputeContext *ctx, const int n_past, const int qlen)
    {
        prepare_pos_tensor(ctx, n_past, qlen);
    }

    size_t CoreAttention::read_cache_data(void *buffer, size_t buffer_size) const
    {
        size_t r = 0;
        uint8_t *p = (uint8_t *)buffer;
        if (k_cache)
        {
            size_t s = ggml::nbytes(k_cache) <= buffer_size ? ggml::nbytes(k_cache) : buffer_size;
            Backend::read_tensor_data(k_cache, p, 0, s);
            r += s;
            buffer_size -= s;
            p += s;
        }
        if (v_cache && (buffer_size > 0))
        {
            size_t s = ggml::nbytes(v_cache) <= buffer_size ? ggml::nbytes(v_cache) : buffer_size;
            Backend::read_tensor_data(v_cache, p, 0, s);
            r += s;
        }
        return r;
    }

    size_t CoreAttention::write_cache_data(const void *buffer, size_t buffer_size)
    {
        size_t r = 0;
        const uint8_t *p = (const uint8_t *)buffer;
        if (k_cache)
        {
            size_t s = ggml::nbytes(k_cache) <= buffer_size ? ggml::nbytes(k_cache) : buffer_size;
            Backend::write_tensor_data(k_cache, p, 0, s);
            r += s;
            buffer_size -= s;
            p += s;
        }
        if (v_cache && (buffer_size > 0))
        {
            size_t s = ggml::nbytes(v_cache) <= buffer_size ? ggml::nbytes(v_cache) : buffer_size;
            Backend::write_tensor_data(v_cache, p, 0, s);
            r += s;
        }
        return r;
    }

    void KVCacheAttention::before_forward(ComputeContext *ctx, const int n_past, const int qlen)
    {
        CoreAttention::before_forward(ctx, n_past, qlen);

        // shift cache
        if (shift_pending.shift > 0)
        {
            int remain = shift_pending.total - shift_pending.shift;
            if (remain > 0)
            {
                ggml::tensor * k_cache_remain = ggml::view_1d(ctx, k_cache, remain * k_hidden_size,
                                            ggml::element_size(k_cache) * k_hidden_size * shift_pending.shift);
                ggml::tensor * k_cache_1d = ggml::view_1d(ctx, k_cache, remain * k_hidden_size,
                                            0);

                ggml::tensor * v_cache_remain = ggml::view_2d(ctx, v_cache, remain, v_hidden_size,
                                            cache_length * ggml::element_size(v_cache),
                                            shift_pending.shift * ggml::element_size(v_cache));
                ggml::tensor * v_cache_2d =     ggml::view_2d(ctx, v_cache, remain, v_hidden_size,
                                            cache_length * ggml::element_size(v_cache),
                                            0);

                ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_cache_remain, k_cache_1d));
                ggml::build_forward_expand(ctx, ggml::cpy(ctx, v_cache_remain, v_cache_2d));
            }
            shift_pending.clear();
        }
    }

    void KVCacheAttention::save_to_cache(ComputeContext *ctx, const int n_past, const int qlen,
        ggml::tensor *k, ggml::tensor *v)
    {
        // compute the transposed [N, n_embd] V matrix
        ggml::tensor * Vcur = ggml::transpose(ctx, v); // ggml::reshape_2d(ctx, tmpv, v_hidden_size, qlen));
        ggml::tensor * v_cache_view = ggml::view_2d(ctx, v_cache, qlen, v_hidden_size,
                cache_length * ggml::element_size(v_cache), n_past * ggml::element_size(v_cache));

        ggml::build_forward_expand(ctx, ggml::cpy(ctx, Vcur, v_cache_view));

        ggml::tensor * k_cache_view = nullptr;
        ggml::tensor * k_view = nullptr;
        if (ggml::is_contiguous(k))
        {
            k_cache_view = ggml::view_1d(ctx, k_cache, qlen * k_hidden_size,
                                    ggml::element_size(k_cache) * k_hidden_size * n_past);
            k_view = ggml::view_1d(ctx, k, qlen * k_hidden_size, 0);
        }
        else
        {
            // [qlen, heads, head_size]
            const int head_size = k_hidden_size / num_kv_heads;
            k_view = k;
            k_cache_view = ggml::view_1d(ctx, k_cache, qlen * k_hidden_size,
                                    ggml::element_size(k_cache) * k_hidden_size * n_past);
            k_cache_view = ggml::reshape_3d(ctx, k_cache_view, head_size, num_kv_heads, qlen);  // [qlen, heads, head_size]
        }

        // important: storing RoPE-ed version of K in the KV cache!
        ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_view, k_cache_view));

    }

    ggml::tensor *KVCacheAttention::get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen)
    {
        const int head_size = k_hidden_size / num_kv_heads;

        ggml::tensor *key_layer = nullptr;

        key_layer = ggml::view_1d(ctx, k_cache, (n_past + qlen) * k_hidden_size, 0);
        key_layer = ggml::reshape_3d(ctx, key_layer, head_size, num_kv_heads, n_past + qlen);  // [qlen, heads, head_size]
        key_layer = ggml::permute(ctx, key_layer, 0, 2, 1, 3);                                 // [heads, qlen, head_size]

        return key_layer;
    }

    ggml::tensor *KVCacheAttention::get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen)
    {
        const int head_size = v_hidden_size / num_kv_heads;

        ggml::tensor * value_layer = ggml::view_3d(ctx,
                        v_cache,
                        n_past + qlen, head_size, num_kv_heads,
                        cache_length * ggml::element_size(v_cache),
                        cache_length * ggml::element_size(v_cache) * head_size,
                        0); // [heads, head_size, klen]
        return value_layer;
    }

    void BaseAttention::set_prec(ggml::prec prec)
    {
        KVCacheAttention::set_prec(prec);
        q_proj.set_prec(prec);
        k_proj.set_prec(prec);
        v_proj.set_prec(prec);
    }

    ggml::tensor *BaseAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int hidden_size = o_proj.in_features();
        const int qlen = (int)hidden_states->ne[1];

        before_forward(ctx, n_past, qlen);

        ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);
        ggml::tensor *tmpk = k_proj.forward(ctx, hidden_states);
        ggml::tensor *tmpv = v_proj.forward(ctx, hidden_states);

        ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

        ggml::tensor *attn_output = o_proj.forward(ctx, scores);
        return attn_output;
    }

    void BaseAttention::load(const std::string &path, TensorLoader *loader)
    {
        KVCacheAttention::load(path, loader);
        q_proj.load(path + "q_proj.", loader);
        k_proj.load(path + "k_proj.", loader);
        v_proj.load(path + "v_proj.", loader);
        o_proj.load(path + "o_proj.", loader);
    }

    ggml::tensor *BaseNormedAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int hidden_size = o_proj.in_features();
        const int qlen = (int)hidden_states->ne[1];

        before_forward(ctx, n_past, qlen);

        ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);
        tmpq = q_norm.forward(ctx, tmpq);

        ggml::tensor *tmpk = k_proj.forward(ctx, hidden_states);
        tmpk = k_norm.forward(ctx, tmpk);

        ggml::tensor *tmpv = v_proj.forward(ctx, hidden_states);

        ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

        ggml::tensor *attn_output = o_proj.forward(ctx, scores);
        return attn_output;
    }

    void BaseNormedAttention::load(const std::string &path, TensorLoader *loader)
    {
        BaseAttention::load(path, loader);
        q_norm.load(path + "q_norm.", loader);
        k_norm.load(path + "k_norm.", loader);
    }

    void BaseCachelessAttention::save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v)
    {
        raw_k = k;
        raw_v = v;
    }

    ggml::tensor *BaseCachelessAttention::get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen)
    {
        // [qlen, heads, head_size] -> [heads, qlen, head_size]
        ggml::tensor *r = ggml::permute(ctx, raw_k, 0, 2, 1, 3);
        return r;
    }

    ggml::tensor *BaseCachelessAttention::get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen)
    {
        const int head_size = hidden_size / num_attention_heads;

        // [qlen, hidden_size] -> [heads, head_size, qlen]
        ggml::tensor *r = ggml::reshape_3d(ctx, raw_v, head_size, num_kv_heads, qlen);  // -> [qlen, heads, head_size]
        r = ggml::permute(ctx, r, 1, 2, 0, 3);   // [heads, head_size, qlen]
        r = ggml::cont(ctx, r);
        return r;
    }

    ALiBiSelfAttention::ALiBiSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
        : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
        mask(ggml::new_tensor_2d(ctx, GGML_TYPE_F32, max_length, max_length))
    {
        bias_max = 8.0f;
        scale = 1.0f / sqrtf(float(hidden_size / num_attention_heads));

        std::vector<float> v_mask;
        v_mask.resize(max_length * max_length);
        float *data = v_mask.data();

        for (int i = 0; i < max_length; i++)
        {
            for (int j = 0; j < max_length; j++)
            {
                int pos = i * max_length + j;
                if (j > i)
                    data[pos] = -INFINITY;
                else
                    data[pos] = -(float)std::abs(j - i);
            }
        }

        ctx->get_allocator()->alloc(mask);
        Backend::write_tensor_data(mask, data);
    }

    ggml::tensor *ALiBiSelfAttention::attn_scores_to_probs(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
        ggml::tensor *attn_scores)
    {
        ggml::tensor *sub_mask = ggml::view_2d(ctx, mask, n_past + qlen, qlen, max_length * ggml::element_size(mask), n_past * max_length * ggml::element_size(mask));
        sub_mask = ggml::cont(ctx, sub_mask);

        ggml::tensor *attn_probs = ggml::soft_max_ext(ctx, attn_scores, sub_mask, scale, bias_max);

        return attn_probs;
    }

    QWenSelfAttention::QWenSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
        : RoPESelfAttention(ctx, hidden_size, num_attention_heads, max_length, true, false),
            seq_length(0),
            use_dynamic_ntk(false),
            use_logn_attn(false),
            logn_list(ggml::new_tensor_1d(ctx, GGML_TYPE_F32, max_length))
    {
        ctx->get_allocator()->alloc(logn_list);
        logn_list_data.resize(max_length);
    }

    void QWenSelfAttention::config(int rope_dim, float rope_freq_base, int seq_length, bool use_dynamic_ntk, bool use_logn_attn)
    {
        this->rope_dim = rope_dim;
        this->freq_base = rope_freq_base;
        this->seq_length = seq_length;
        this->use_dynamic_ntk = use_dynamic_ntk;
        this->use_logn_attn = use_logn_attn;

        if (use_logn_attn)
        {
            float *p = (float *)logn_list_data.data();
            for (int i = 0; i < max_length; i++)
                p[i] = i > seq_length ? logf(float(i)) / logf((float)seq_length) : 1.0f;
            Backend::write_tensor_data(logn_list, logn_list_data.data());
        }
    }

    ggml::tensor *QWenSelfAttention::apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const
    {
        // [qlen, heads, head_size]
        return ggml::map_custom2(ctx, k, past, ggml_compute_forward_ntk_dynamic_rope, GGML_N_TASKS_MAX, const_cast<QWenSelfAttention *>(this));
    }

    ggml::tensor *QWenSelfAttention::apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const
    {
        // [qlen, heads, head_size];
        ggml::tensor *r = ggml::map_custom2(ctx, q, past, ggml_compute_forward_ntk_dynamic_rope, GGML_N_TASKS_MAX, const_cast<QWenSelfAttention *>(this));
        if (use_logn_attn)
        {
            const int *p = (const int *)past->data;
            int last_n = p[qlen - 1];
            if (last_n > seq_length)
            {
                ggml::tensor *scale = ggml::view_1d(ctx, logn_list, qlen, p[0] * ggml::element_size(logn_list));
                r = ggml::map_custom2(ctx, r, scale, ggml_compute_forward_mat_scale, GGML_N_TASKS_MAX, nullptr);
            }
        }
        return r;
    }

    void BlueLMSelfAttention::config(float rope_theta, float rope_scaling_factor, float rope_scaling_power)
    {
        this->freq_base = rope_theta;
        this->rope_scaling_factor = rope_scaling_factor;
        this->rope_scaling_power = rope_scaling_power;
    }

    void BlueLMSelfAttention::build_inv_freq_if_needed(int hidden_size)
    {
        if (cached_hidden_size != hidden_size)
        {
            cached_hidden_size = hidden_size;
            build_ntk_mixed_inv_freq(rope_dim, inv_freq, (int)((float)max_length / rope_scaling_factor), freq_base, rope_scaling_factor, rope_scaling_power);
        }
    }

    ggml::tensor *BlueLMSelfAttention::apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const
    {
        const_cast<BlueLMSelfAttention *>(this)->rope_dim = hidden_size / num_attention_heads;
        if (rope_scaling_power > 0.0)
        {
            const_cast<BlueLMSelfAttention *>(this)->build_inv_freq_if_needed(hidden_size);
            return ggml::map_custom2(ctx, k, past, ggml_compute_forward_ntk_mix_rope, GGML_N_TASKS_MAX, const_cast<BlueLMSelfAttention *>(this));
        }
        else
            return RoPESelfAttention::apply_pos_embedding_k(ctx, k, hidden_size, qlen, past);
    }

    ggml::tensor *BlueLMSelfAttention::apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const
    {
        const_cast<BlueLMSelfAttention *>(this)->rope_dim = hidden_size / num_attention_heads;
        if (rope_scaling_power > 0.0)
        {
            const_cast<BlueLMSelfAttention *>(this)->build_inv_freq_if_needed(hidden_size);
            return ggml::map_custom2(ctx, q, past, ggml_compute_forward_ntk_mix_rope, GGML_N_TASKS_MAX, const_cast<BlueLMSelfAttention *>(this));
        }
        else
            return RoPESelfAttention::apply_pos_embedding_q(ctx, q, hidden_size, qlen, past);
    }

    ggml::tensor *RobertaBlock::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        // CAUTION: MEMORY REUSED BETWEEN LAYERS

        ggml::tensor *attn_outputs = attention.forward(ctx, hidden_states, n_past);

        // see XLMRobertaSelfOutput
        ggml::tensor *sum = ggml::add(ctx, hidden_states, attn_outputs);
        ggml::tensor *attention_output = post_attention_layernorm.forward(ctx, sum);

        ggml::tensor *r = mlp.forward(ctx, attention_output);
        return r;
    }

    void RobertaBlock::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        attention.load(path + "attention.self.", loader);
        post_attention_layernorm.load(path + "post_attention_layernorm.", loader);
        mlp.load(path + "mlp.", loader);
        output_layernorm.load(path + "output_layernorm.", loader);
    }

    ggml::tensor *RobertaOutput::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *attention_output)
    {
        ggml::tensor *r = dense.forward(ctx, hidden_states);
        r = ggml::add_inplace(ctx, r, attention_output);
        r = norm.forward(ctx, r);
        return r;
    }

    void RobertaOutput::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        dense.load(path + "dense.", loader);
        norm.load(path + "ln.", loader);
    }

    ggml::tensor *RobertaMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *temp = intermediate.forward(ctx, hidden_states);
        temp = ggml::inplace_act(ctx, act, temp);
        temp = output.forward(ctx, temp, hidden_states);
        return temp;
    }

    void RobertaMLP::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        intermediate.load(path + "intermediate.dense.", loader);
        output.load(path + "output.", loader);
    }

    ggml::tensor *FuyuEmbedding::forward(ComputeContext *ctx, ggml::tensor *patches, int patches_per_row, ggml::tensor *text_input)
    {
        //ggml_get_rows
        return nullptr;
    }

    void Phi3SUSelfAttention::config(InitContext *ctx, int original_max_position_embeddings, float rope_theta, float short_scaling_factor, float long_scaling_factor, int factor_len, const float *short_factor, const float *long_factor)
    {
        if (freq_factors == nullptr)
        {
            rope_dim = factor_len * 2;
            freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, factor_len);
            ctx->get_allocator()->alloc(freq_factors);
        }

        const float *factors = nullptr;
        this->freq_base = rope_theta;
        if (max_length > original_max_position_embeddings)
        {
            attn_factor = long_scaling_factor;
            factors = long_factor;
        }
        else
        {
            attn_factor = short_scaling_factor;
            factors = short_factor;
        }

        CHATLLM_CHECK(rope_dim == factor_len * 2) << "factor_len mismatch!";

        Backend::write_tensor_data(freq_factors, factors);
    }

    ggml::tensor *BaseSparseMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        const int64_t qlen        = hidden_states->ne[1];
        const int n_expert = num_local_experts;

        ggml::tensor * logits = gate.forward(ctx, hidden_states); // [qlen, num_experts]

        CPUMover mover(ctx, ctx->user_options.moe_on_cpu);

        ggml::tensor * probs = nullptr;
        switch (score_func)
        {
        case ScoreFunc::Softmax:
            probs = ggml::soft_max(ctx, logits); // [qlen, num_experts]
            break;
        case ScoreFunc::Sigmoid:
            probs = ggml::sigmoid(ctx, logits);  // [qlen, num_experts]
            break;
        default:
            CHATLLM_CHECK(false) << "Invalid score_func!";
            break;
        }

        ggml::tensor * corrected_score = probs;
        if (gate_score_correction_bias)
        {
            corrected_score = ggml::add(ctx, corrected_score, gate_score_correction_bias);
        }

        // select experts
        ggml::tensor * selected_experts = ggml::top_k(ctx, corrected_score, num_experts_per_tok); // [qlen, num_experts_per_tok]

        ggml::tensor * weights = ggml::get_rows(ctx,
            ggml::reshape_3d(ctx, probs, 1, n_expert, qlen), selected_experts); // [1, num_experts_per_tok, qlen]

        if (norm_topk_prob) {
            weights = ggml::reshape_2d(ctx, weights, num_experts_per_tok, qlen);

            ggml::tensor * weights_sum = ggml::sum_rows(ctx, weights); // [1, n_tokens]

            weights = ggml::div(ctx, weights, weights_sum); // [num_experts_per_tok, n_tokens]
            weights = ggml::reshape_3d(ctx, weights, 1, num_experts_per_tok, qlen);

            if (always_scaling && (routed_scaling_factor > 0))
            {
                weights = ggml::scale(ctx, weights, routed_scaling_factor);
            }
        }
        else
        {
            if (routed_scaling_factor > 0.0f)
                weights = ggml::scale(ctx, weights, routed_scaling_factor);
        }


        return forward_with_experts(ctx, hidden_states, selected_experts, weights);
    }

    // selected_experts: [qlen, num_experts_per_tok]
    // weights:          [1, num_experts_per_tok, qlen]
    ggml::tensor *BaseSparseMLP::forward_with_experts(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts,
            ggml::tensor *weights)
    {
        const int64_t hidden_size = hidden_states->ne[0];
        const int64_t qlen        = hidden_states->ne[1];

        hidden_states = ggml::reshape_3d(ctx, hidden_states, hidden_size, 1, qlen);
        if (pre_weighting)
        {
            hidden_states = ggml::mul(ctx, hidden_states, weights);
        }
        ggml::tensor *gated = experts_gate.forward(ctx, hidden_states, selected_experts); // [n_ff, num_experts_per_tok, qlen]
        ggml::tensor *act = ggml::inplace_act(ctx, this->act, gated);
        ggml::tensor *up = experts_up.forward(ctx, hidden_states, selected_experts); // [n_ff, num_experts_per_tok, qlen]

        ggml::tensor *par = ggml::mul_inplace(ctx, up, act); // [n_ff, num_experts_per_tok, qlen]

        ggml::tensor * experts = experts_down.forward(ctx, par, selected_experts); // [hidden_size, num_experts_per_tok, qlen]

        if (!pre_weighting)
        {
            experts = ggml::mul(ctx, experts, weights);
        }

        ggml::tensor * moe_out = nullptr;
        for (int i = 0; i < num_experts_per_tok; ++i)
        {
            ggml::tensor * cur_expert = ggml::view_2d(ctx, experts, hidden_size, qlen,
                                                    experts->nb[2], i * experts->nb[1]);

            moe_out = i == 0 ? cur_expert : ggml::add(ctx, moe_out, cur_expert);
        }

        if (num_experts_per_tok == 1) {
            // avoid returning a non-contiguous tensor
            moe_out = ggml::cont(ctx, moe_out);
        }

        return moe_out;
    }

    void BaseSparseMLP::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        gate.load(path + "gate.", loader);

        loader->read_tensor(path + "experts_down.weight", path + "experts.", num_local_experts, ".down_proj.weight", experts_down.weight);
        loader->read_tensor(path + "experts_gate.weight", path + "experts.", num_local_experts, ".gate_proj.weight", experts_gate.weight);
        loader->read_tensor(path + "experts_up.weight",   path + "experts.", num_local_experts, ".up_proj.weight",   experts_up.weight);

        if (gate_score_correction_bias)
        {
            loader->read_tensor(path + "gate_score_correction_bias", gate_score_correction_bias);
        }
    }
}
