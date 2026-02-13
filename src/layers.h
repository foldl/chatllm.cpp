#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "chat.h"

#include "backend.h"

namespace chatllm
{
    void dump_weight_tensor(ggml::tensor *tensor);
    void print_tensor(ggml::tensor *tensor, int offset = 0, bool full = false);
    void print_tensor_shape(const char *info, ggml::tensor *tensor);
    void inspect_tensor(ggml::tensor *tensor, const char *format, ...);
    void clear_inspected_tensors(void);

    struct alibi_ctx
    {
        int    n_past;
        int    n_head;
        float  bias_max;
    };

    enum ActFunc
    {
        GELU,   // equivelent to `gelu_new`
        GELU_QUICK,
        SILU,
        Tanh,
        RELU,
        RELU2,  // square . relu
        SWISH,
    };

    namespace ggml
    {
        typedef ggml_prec prec;

        ggml::type type_fallback(ggml::type type, int64_t last_dim);

        ggml::tensor *new_tensor_1d(ComputeContext *ctx, ggml::type type, int64_t ne0);
        ggml::tensor *new_tensor_2d(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1);
        ggml::tensor *new_tensor_3d(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1, int64_t ne2);
        ggml::tensor *new_tensor_4d(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
        ggml::tensor *new_tensor_like(ComputeContext *ctx, ggml::type type, ggml::tensor *a);

        ggml::tensor *new_zeros(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);

        void set_input(ggml::tensor *a);
        void set_output(ggml::tensor *a);

        void fill(ggml::tensor *a, uint8_t value, size_t offset = 0, size_t size = 0);
        float  at(ggml::tensor *a, int64_t i, int64_t j, int64_t k, int64_t l);

        ggml::type    type_of(const ggml::tensor *a);

        ggml::tensor *cpy(ComputeContext *ctx, ggml::tensor *src, ggml::tensor *dst);

        ggml::tensor *dup(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *cont(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *cast(ComputeContext *ctx, ggml::tensor *a, ggml::type type);

        ggml::tensor *transpose(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *concat(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, int dim);

        ggml::tensor *pad(ComputeContext *ctx, ggml::tensor *a,
            int l0, int r0,
            int l1 = 0, int r1 = 0,
            int l2 = 0, int r2 = 0,
            int l3 = 0, int r3 = 0
        );

        // operators
        ggml::tensor *get_rows(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b);
        ggml::tensor *set_rows(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *indices, ggml::tensor *source);

        ggml::tensor *add(ComputeContext *ctx, ggml::tensor * a, ggml::tensor * b);
        ggml::tensor *add_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor * b);  // a = a + b
        ggml::tensor *add_id(ComputeContext *ctx, ggml::tensor *as, ggml::tensor *b, ggml::tensor *ids);

        ggml::tensor *sub(ComputeContext *ctx, ggml::tensor * a, ggml::tensor * b);
        ggml::tensor *sub_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor * b);

        // result = B * A^T
        ggml::tensor *mul_mat(ComputeContext *ctx, ggml::tensor *a, ggml::tensor  *b);
        ggml::tensor *mul_mat_id(ComputeContext *ctx, ggml::tensor *as, ggml::tensor *b, ggml::tensor *ids);

        ggml::tensor *mul(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b);
        ggml::tensor *mul_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b);

        ggml::tensor *div(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b);
        ggml::tensor *int_div(ComputeContext *ctx, ggml::tensor *a, int b);

        ggml::tensor *sum_rows(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *mean(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *sin(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *cos(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *square(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *sqrt(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *log(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *tanh(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *randn_inplace(ComputeContext *ctx, ggml::tensor *a); // standard normal distribution
        ggml::tensor *randn(ComputeContext *ctx, ggml::type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);

        ggml::tensor *top_k(ComputeContext *ctx, ggml::tensor *a, int k);
        ggml::tensor *ordering(ComputeContext *ctx, ggml::tensor *a, bool descending = false);

        ggml::tensor *view_1d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, size_t offset);
        ggml::tensor *view_2d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
        ggml::tensor *view_3d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
        ggml::tensor *view_4d(ComputeContext *ctx, ggml::tensor  *a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

        ggml::tensor *reshape_1d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0);
        ggml::tensor *reshape_2d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1);
        ggml::tensor *reshape_3d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1, int64_t ne2);
        ggml::tensor *reshape_4d(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
        ggml::tensor *reshape(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);
        ggml::tensor *flatten(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *repeat(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b);
        ggml::tensor *repeat(ComputeContext *ctx, ggml::tensor *a, int64_t ne0, int64_t ne1 = 0, int64_t ne2 = 0, int64_t ne3 = 0);
        ggml::tensor *repeat_interleave(ComputeContext *ctx, ggml::tensor *a, int repeat, int dim = 0);

        // axis0: target axis of axis 0 (differs from PyTorch)
        ggml::tensor *permute(ComputeContext *ctx, ggml::tensor *a, int axis0, int axis1, int axis2 = 2, int axis3 = 3);
        ggml::tensor *flip(ComputeContext *ctx, ggml::tensor *a, int dim = 0);

        enum InterpolateMode
        {
            Nearest = 0,
            Bilinear = 2,
            Bicubic = 3,
        };
        ggml::tensor *interpolate(ComputeContext *ctx, ggml::tensor *a, InterpolateMode mode, float scale_factor);
        ggml::tensor *interpolate(ComputeContext *ctx, ggml::tensor *a, InterpolateMode mode, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

        ggml::tensor *norm(ComputeContext *ctx, ggml::tensor *a, float eps);
        ggml::tensor *norm_inplace(ComputeContext *ctx, ggml::tensor *a, float eps);
        ggml::tensor *norm_p2(ComputeContext *ctx, ggml::tensor *a, float eps);
        ggml::tensor *group_norm(ComputeContext *ctx, ggml::tensor *a, int n_groups, float eps);
        ggml::tensor *rms_norm_inplace(ComputeContext *ctx, ggml::tensor *a, float eps);
        ggml::tensor *rms_norm(ComputeContext *ctx, ggml::tensor *a, float eps);
        ggml::tensor *simple_norm(ComputeContext *ctx, ggml::tensor *a, float eps); // p=2 normalization

        ggml::tensor *rope(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, int n_dims, int mode);
        ggml::tensor *rope_ext(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *pos, ggml::tensor *freq_factors,
                                            int   n_dims, int   mode, int   n_ctx_orig,
                                            float freq_base, float freq_scale, float ext_factor,
                                            float attn_factor, float beta_fast, float beta_slow,
                                            const int *sections = nullptr);
        ggml::tensor *rope_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, int n_dims, int mode);
        ggml::tensor *rope_ext_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *pos, ggml::tensor *freq_factors,
                                            int   n_dims, int   mode, int   n_ctx_orig,
                                            float freq_base, float freq_scale, float ext_factor,
                                            float attn_factor, float beta_fast, float beta_slow,
                                            const int *sections = nullptr);

        ggml::tensor *soft_max(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *soft_max_inplace(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *soft_max_ext(ComputeContext *ctx,  ggml::tensor *a,  ggml::tensor *mask, float scale, float max_bias);
        void          soft_max_attach_sinks(ggml::tensor *soft_max_result, ggml::tensor *sinks);

        ggml::tensor *sigmoid(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *diag_mask_inf(ComputeContext *ctx, ggml::tensor *a, int n_past);
        ggml::tensor *diag_mask_inf_inplace(ComputeContext *ctx, ggml::tensor *a, int n_past);

        ggml::tensor *inplace_act(ComputeContext *ctx, ActFunc act, ggml::tensor *input);
        ggml::tensor *act(ComputeContext *ctx, ActFunc act, ggml::tensor *input);

        ggml::tensor *scale(ComputeContext *ctx, ggml::tensor *a, float  s);
        ggml::tensor *scale_inplace(ComputeContext *ctx, ggml::tensor *a, float  s);
        // r = s * a + b
        ggml::tensor *scale(ComputeContext *ctx, ggml::tensor *a, float  s, float b);

        ggml::tensor *clamp(ComputeContext *ctx, ggml::tensor *a, float min, float max);
        ggml::tensor *avg_pool_2d(ComputeContext *ctx, ggml::tensor *a, int kernel_size, int stride, float padding = 0.0f);
        ggml::tensor *avg_pool_1d(ComputeContext *ctx, ggml::tensor *a, int kernel_size, int stride,   int padding = 0);

        ggml::tensor *abs(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *neg(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *sign(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *step(ComputeContext *ctx, ggml::tensor *a);
        ggml::tensor *exp(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *conv_1d(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding = 0, int dilation = 1);
        ggml::tensor *conv_1d_depthwise(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding, int dilation);
        ggml::tensor *conv_transposed_1d( ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding, int dilation);

        ggml::tensor *conv_2d(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data, int stride, int padding, int dilation);
        ggml::tensor *conv_2d(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data,
            int stride0, int stride1, int padding0, int padding1, int dilation0, int dilation1);
        ggml::tensor *conv_2d_depthwise(ComputeContext *ctx, ggml::tensor *kernel, ggml::tensor *data,
            int stride0, int stride1, int padding0, int padding1, int dilation0, int dilation1);

        // gate = gate.clamp(min=None, max=self.limit)
        // up   = up.clamp(min=-self.limit, max=self.limit)
        // glu  = gate * torch.sigmoid(gate * self.alpha)
        // result: ((up + 1) * glu)
        ggml::tensor *swiglu_oai(ComputeContext *ctx, ggml::tensor *gate, ggml::tensor *up, float alpha, float limit);

        ggml::tensor *xielu(ComputeContext *ctx, ggml::tensor *input, float alpha_n, float alpha_p, float beta, float eps);

        ggml::tensor *logsumexp(ComputeContext *ctx, ggml::tensor *a);

        // accept either probs or logits, but not both
        ggml::tensor *categorical_entropy(ComputeContext *ctx, ggml::tensor *probs, ggml::tensor *logits);

        ggml::tensor *map_custom1(ComputeContext *ctx, ggml::tensor *a, const ggml_custom1_op_t fun, int n_tasks, void *userdata);
        ggml::tensor *map_custom2(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, const ggml_custom2_op_t fun, int n_tasks, void *userdata);
        ggml::tensor *map_custom3(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, ggml::tensor *c, const ggml_custom3_op_t fun, int n_tasks, void *userdata);

        // Note: these `inplace` might not work when GPU is used
        ggml::tensor *map_custom1_inplace(ComputeContext *ctx, ggml::tensor *a, const ggml_custom1_op_t fun, int n_tasks, void *userdata);
        ggml::tensor *map_custom2_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, const ggml_custom2_op_t fun, int n_tasks, void *userdata);
        ggml::tensor *map_custom3_inplace(ComputeContext *ctx, ggml::tensor *a, ggml::tensor *b, ggml::tensor *c, const ggml_custom3_op_t fun, int n_tasks, void *userdata);

        // a quick helper since such casts is missing from ggml
        ggml::tensor *cast_int_to_i64(ComputeContext *ctx, ggml::tensor *a);

        ggml::tensor *custom(ComputeContext *ctx, ggml_custom_op_t fun, int n_tasks, void *userdata, std::vector<ggml::tensor *>inputs, ggml::type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);

        struct merge_patch_param
        {
            int grid_h;
            int grid_w;
            int merge_kernel_size[2];
        };
        // x ggml-shape: [hidden, all_patches]
        // r ggml-shape: [hidden, merge_kernel_size[0], merge_kernel_size[1], new_width, new_height]
        ggml::tensor *merge_patch(ComputeContext *ctx, ggml::tensor *x, const merge_patch_param *param);

        void mul_mat_set_prec(ggml::tensor *a, ggml::prec prec);
        bool is_contiguous(const ggml::tensor *a);
        bool is_view(const ggml::tensor *tensor);

        struct ggml_cgraph *new_graph_custom(ComputeContext *ctx, size_t size, bool grads);
        void build_forward_expand(ComputeContext *ctx, ggml::tensor * tensor);
    };

    // facility to pass extra params when constructing Blocks.
    // single thread assumed.
    class BlockParams
    {
    public:
        static int num_padding_embeddings; // number extra (padding) embedding
        class PadEmbedding
        {
        public:
            PadEmbedding(int n, int max) { BlockParams::num_padding_embeddings = n > max ? max : (n >= 0 ? n : 0); }
            ~PadEmbedding(void) { BlockParams::num_padding_embeddings = 0; }
            int get(void) const { return BlockParams::num_padding_embeddings; }
        };

        class OverrideKProjBiased
        {
        public:
            OverrideKProjBiased(bool biased);
            ~OverrideKProjBiased();
            static bool get(bool biased);
        protected:
            static bool active;
            static bool biased;
        };

        class DisableCache
        {
        public:
            DisableCache();
            ~DisableCache();
            static bool is_disabled(void);
        protected:
            static bool disabled;
        };

        class CoreAttentionUseSinks
        {
        public:
            CoreAttentionUseSinks(int size);
            ~CoreAttentionUseSinks();
            static int get();
        protected:
            static int size;
        };

        class MoE
        {
        public:
            static int num_experts;
            static int experts_per_tok;
        };

        class Epsilon
        {
        public:
            static float rms_norm;
        };
    };

    class PreludeCacheDisable
    {
    public:
        PreludeCacheDisable(void): disabler(new BlockParams::DisableCache())
        {
        }
    protected:
        BlockParams::DisableCache *disabler;
    };

    class Block
    {
    public:
        Block(): prec(ggml::prec::GGML_PREC_DEFAULT), id(0), reserved_batch_size(1), debug(false) {}
        virtual ~Block() {}
        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input)
        {
            CHATLLM_THROW << "forward(ComputeContext *ctx, ggml::tensor *input): not implemented";
            return NULL;
        }
        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past)
        {
            CHATLLM_THROW << "forward(ComputeContext *ctx, ggml::tensor *input, int n_past): not implemented";
            return NULL;
        }
        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w)
        {
            CHATLLM_THROW << "forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w): not implemented";
            return NULL;
        }
        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *_)
        {
            CHATLLM_THROW << "forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *_): not implemented";
            return NULL;
        }
        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input1, ggml::tensor *input2, int n_past)
        {
            CHATLLM_THROW << "forward(ComputeContext *ctx, ggml::tensor *input1, ggml::tensor *input2, int n_past): not implemented";
            return NULL;
        }
        virtual void set_ctx(int n_ctx) { }
        virtual void shift_cache(int shift, int total) { }

        virtual void set_prec(ggml::prec prec)
        {
            this->prec = prec;
        }

        virtual void set_id(int id)
        {
            this->id = id;
        }

        int get_id(void) const
        {
            return id;
        }

        virtual int64_t get_param_num(bool effective_only) const
        {
            return 0;
        }

        virtual void reserve_batch_size(int size)
        {
            reserved_batch_size = size;
        }

        virtual int get_reserved_batch_size(void) const
        {
            return reserved_batch_size;
        }

        virtual size_t get_cache_size(void) const { return 0; }
        virtual void   set_cache_buffer(BackendBuffer *buf) { }
        virtual size_t read_cache_data(void *buffer, size_t buffer_size) const { return 0; }
        virtual size_t write_cache_data(const void *buffer, size_t buffer_size) { return 0; }

        virtual void load(const std::string &path, TensorLoader *loader) { }

    protected:
        ggml::prec prec;
        int id;
        int reserved_batch_size;
    public:
        bool debug;
    };

    class Identity : public Block
    {
    public:
        Identity() {}
        Identity(InitContext *ctx, int a) {}
        Identity(InitContext *ctx, int a, int b) {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            return input;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override
        {
            return input;
        }
    };

    class Sequential : public Block
    {
    public:
        Sequential() {}
        Sequential(InitContext *ctx) {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;

        void add_block(Block * block);
        void add_block(std::unique_ptr<Block> block);

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        std::vector<std::unique_ptr<Block>> blocks;
        std::string path;
    };

    class ConvBase: public Block
    {
    public:
        ConvBase(InitContext *ctx, int64_t w_ne0, int64_t w_ne1, int64_t w_ne2, int64_t w_ne3,
            int64_t bias_dim);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        ggml::tensor *weight;
        ggml::tensor *bias;
    };

    class Conv1D: public ConvBase
    {
    public:
        Conv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
               int dilation = 1, int groups = 1, bool bias = true);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    protected:
        Conv1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding,
               int dilation, int groups, int bias_dim);
    protected:
        const int in_channels;
        const int out_channels;
        const int kernel_size;
        const int stride;
        const int padding;
        const int dilation;
        const int groups;
    };

    // Just like Torch:
    // when a Conv1d and a ConvTranspose1d are initialized with same parameters,
    // they are inverses of each other in regard to the input and output shapes.
    // So, here `in_channels` --> Conv1D --> `out_channels`
    class ConvTransposed1D : public Conv1D
    {
    public:
        ConvTransposed1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
               int output_padding = 0, int dilation = 1, int groups = 1, bool bias = true);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    protected:
        const int output_padding;
    };

    class Conv2D: public ConvBase
    {
    public:
        Conv2D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding = 0,
               int dilation = 1, int groups = 1, bool bias = true);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    protected:
        Conv2D(InitContext *ctx, int in_channels, int out_channels,
               int kernel_size0, int kernel_size1,
               int stride0, int stride1,
               int padding0, int padding1,
               int dilation0, int dilation1,
               int groups, int bias_dim);
    protected:
        const int in_channels;
        const int out_channels;
        const int kernel_size[2];
        const int stride[2];
        const int padding[2];
        const int dilation[2];
        const int groups;
    };

    class Conv3D: public ConvBase
    {
    public:
        Conv3D(InitContext *ctx, int in_channels, int out_channels,
               int kernel_size0, int kernel_size1, int kernel_size2,
               int stride0, int stride1, int stride2,
               int padding0, int padding1, int padding2,
               int dilation0, int dilation1, int dilation2,
               int groups, bool bias);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    protected:

    protected:
        const int in_channels;
        const int out_channels;
        const int kernel_size[3];
        const int stride[3];
        const int padding[3];
        const int dilation[3];
        const int groups;
    };

    class Unary : public Block
    {
    public:
        enum Op
        {
            Sin,
            Cos,
            Tanh,
            Square,
            Sqrt,
            Log,
        };
        Unary(InitContext *ctx, Op op): op(op) {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;
    public:
        const Op op;
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
        Embedding(InitContext *ctx, int num_embeddings, int embedding_dim)
            : Embedding(ctx, ctx->dtype, num_embeddings, embedding_dim)
        {
        }

        Embedding(InitContext *ctx, ggml::type dtype, int num_embeddings, int embedding_dim)
            : num_embeddings(num_embeddings), num_padded_embeddings(BlockParams::num_padding_embeddings),
              weight(ggml::new_tensor_2d(ctx, ggml::type_fallback(dtype, embedding_dim), embedding_dim,
                     num_embeddings + BlockParams::num_padding_embeddings))
        {
            ggml::set_input(weight);
        }

        Embedding(InitContext *ctx, int num_embeddings, int embedding_dim, int pos_max)
            : Embedding(ctx, num_embeddings, embedding_dim) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml::nelements(weight) / (num_embeddings + num_padded_embeddings) * num_embeddings;
        }

        // bytes without padding
        size_t get_base_nbytes(void);

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        const int num_embeddings;
        const int num_padded_embeddings;
        ggml::tensor *weight;
    };

    class Linear : public Block
    {
    public:
        Linear() : weight(nullptr), bias(nullptr) {}
        Linear(InitContext *ctx, int in_features, int out_features, bool use_bias = true)
            : weight(ggml::new_tensor_2d(ctx,  ggml::type_fallback(ctx->dtype, in_features), in_features, out_features)),
              bias(use_bias ? ggml::new_tensor_1d(ctx, GGML_TYPE_F32, out_features) : nullptr) {}

        Linear(InitContext *ctx, int in_features, int out_features, ggml::tensor *weight, bool use_bias = true)
            : weight(weight != NULL ? weight : ggml::new_tensor_2d(ctx, ggml::type_fallback(ctx->dtype, in_features), in_features, out_features)),
              bias(use_bias ? ggml::new_tensor_1d(ctx, GGML_TYPE_F32, out_features) : nullptr) {}

        int in_features() const { return (int)weight->ne[0]; }
        int out_features() const { return (int)weight->ne[1]; }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ggml::nelements(weight);
            if (bias) r += ggml::nelements(bias);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        ggml::tensor *weight; // [out_features, in_features]
        ggml::tensor *bias;   // [out_features]
    };

    class MultiLinear : public Block
    {
    public:
        MultiLinear() : weight(nullptr) {}
        MultiLinear(InitContext *ctx, int in_features, int out_features, int multi)
            : MultiLinear(ctx, in_features, out_features, multi, false)
        {}

        MultiLinear(InitContext *ctx, int in_features, int out_features, int multi, bool use_bias)
            : weight(ggml::new_tensor_3d(ctx, ggml::type_fallback(ctx->dtype, in_features), in_features, out_features, multi)),
              bias(use_bias ? ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, out_features, multi) : nullptr)
        {
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override { return nullptr; }

        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, ggml::tensor *selected);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += ggml::nelements(weight);
            r += ggml::nelements(bias);
            return r;
        }

    public:
        ggml::tensor *weight; // [in_features, out_features, multi]
        ggml::tensor *bias;   // [out_features, multi]
    };

    class GroupNorm : public Block
    {
    public:
        GroupNorm() : weight(nullptr), bias(nullptr), eps(1e-5f), num_groups(1) {}
        GroupNorm(InitContext *ctx, int num_groups, int normalized_shape)
            : GroupNorm(ctx, num_groups, normalized_shape, true)
        {
        }

        GroupNorm(InitContext *ctx, int num_groups, int normalized_shape, bool use_bias, float eps = 1e-5f)
            : weight(ggml::new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape)),
              bias(use_bias ? ggml::new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape) : nullptr),
              eps(eps), num_groups(num_groups)
        {
            CHATLLM_CHECK((normalized_shape % num_groups) == 0);
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ggml::nelements(weight);
            if (bias) r += ggml::nelements(bias);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        ggml::tensor *weight; // [normalized_shape]
        ggml::tensor *bias;   // [normalized_shape]
        const float eps;
        const int num_groups;
    };

    class LayerNorm : public GroupNorm
    {
    public:
        LayerNorm() : GroupNorm() {}
        LayerNorm(InitContext *ctx, int normalized_shape)
            : LayerNorm(ctx, normalized_shape, true)
        {
        }

        LayerNorm(InitContext *ctx, int normalized_shape, bool use_bias)
            : GroupNorm(ctx, normalized_shape, normalized_shape, use_bias) {}
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

    class FIR2 : public Block
    {
    public:
        FIR2(): weight(nullptr), dim(0) {}
        FIR2(InitContext *ctx, int dim, int hidden_size);

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int n_past) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            if (weight) r += ggml::nelements(weight);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;
    public:
        ggml::tensor *weight;
        ggml::tensor *x0;
        const int dim;
    };

    class RobertaEmbedding : public Block
    {
    public:
        RobertaEmbedding() : word_weight(nullptr), position_weight(nullptr) {}
        RobertaEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim, int pos_max)
            : word_weight(ggml::new_tensor_2d(ctx, ctx->dtype, embedding_dim, num_embeddings)),
              position_weight(ggml::new_tensor_2d(ctx, ctx->dtype, embedding_dim, pos_max)),
              indices(ggml::new_tensor_1d(ctx, GGML_TYPE_I32, pos_max)),
              ln(ctx, embedding_dim)
        {
            const int pad_index = 2;
            std::vector<int> v_indices;
            v_indices.resize(pos_max);
            for (int i = 0; i < pos_max; i++)
                v_indices[i] = pad_index + i;

            ctx->get_allocator()->alloc(indices);
            Backend::write_tensor_data(indices, v_indices.data());
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = ln.get_param_num(effective_only);
            r += ggml::nelements(word_weight);
            r += ggml::nelements(position_weight);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        ggml::tensor *word_weight;
        ggml::tensor *position_weight;
        ggml::tensor *indices;
        LayerNorm    ln;
    };

    class RMSNorm : public Block
    {
    public:
        RMSNorm() : weight(nullptr), inplace(false) {}
        RMSNorm(InitContext *ctx, int normalized_shape) : RMSNorm(ctx, normalized_shape, false) {}
    protected:
        RMSNorm(InitContext *ctx, int normalized_shape, bool inplace)
            : weight(ggml::new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape)),
              eps(BlockParams::Epsilon::rms_norm),
              inplace(inplace) {}
    public:
        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml::nelements(weight);
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        ggml::tensor *weight;
        float eps;
        const bool inplace;
    };

    class L2Norm : public Block
    {
    public:
        L2Norm() : inplace(false) {}
        L2Norm(InitContext *ctx, int normalized_shape)
            : L2Norm(ctx, normalized_shape, false) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

    protected:
        L2Norm(InitContext *ctx, int normalized_shape, bool inplace)
            : eps(1e-5f),
              inplace(inplace) {}

    public:
        float eps;
        const bool inplace;
    };

    class L2NormInplace : public L2Norm
    {
    public:
        L2NormInplace(InitContext *ctx, int normalized_shape)
            : L2Norm(ctx, normalized_shape, true) {}
    };

    class RMSNormInplace : public RMSNorm
    {
    public:
        RMSNormInplace(InitContext *ctx, int normalized_shape) : RMSNorm(ctx, normalized_shape, true) {}
    };

    // This is **the** feed forward network (Multi-Layer Perceptron) in _Attention Is All You Need_.
    class TheMLP : public Block
    {
    public:
        TheMLP() = default;
        TheMLP(InitContext *ctx, int hidden_size, int intermediate_size, ActFunc act, bool bias)
            : TheMLP(ctx, hidden_size, intermediate_size, hidden_size, act, bias)
        {}

        TheMLP(InitContext *ctx, int hidden_size, int intermediate_size, int out_size, ActFunc act, bool bias)
            : fc0(ctx, hidden_size, intermediate_size, bias),
              fc1(ctx, intermediate_size, out_size, bias),
              act(act)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        void set_prec(ggml::prec prec) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += fc0.get_param_num(effective_only);
            r += fc1.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        Linear fc0;
        Linear fc1;
        ActFunc act;
    };

    // a common setup
    class TheBiasedGELUMLP : public TheMLP
    {
    public:
        TheBiasedGELUMLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : TheMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU, true)
        {
        }

        TheBiasedGELUMLP(InitContext *ctx, int hidden_size, int intermediate_size, int out_size)
            : TheMLP(ctx, hidden_size, intermediate_size, out_size, ActFunc::GELU, true)
        {
        }
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

    class GLM2MLP : public Block
    {
    public:
        GLM2MLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : dense_h_to_4h(ctx, hidden_size, intermediate_size * 2, false),
              dense_4h_to_h(ctx, intermediate_size, hidden_size, false) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

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

    template <class AttentionBlock> class LMAttentionBlock : public Block
    {
        public:
        LMAttentionBlock() = default;
        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : attention(ctx, hidden_size, num_attention_heads, max_length) {}

        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length) {}

        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int head_dim, int max_length)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length) {}

        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length) {}

        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                        q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                        use_bias) {}

        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                        q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                        use_bias) {}

        LMAttentionBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length, bool qkv_bias, bool o_bias)
            : attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias) {}

        void shift_cache(int shift, int total) override
        {
            attention.shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Block::get_param_num(effective_only);
            r += attention.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Block::set_id(id);
            attention.set_id(id);
        }

        void reserve_batch_size(int size) override
        {
            Block::reserve_batch_size(size);
            attention.reserve_batch_size(size);
        }

        size_t get_cache_size(void) const override
        {
            return attention.get_cache_size();
        }

        void  set_cache_buffer(BackendBuffer *buffer) override
        {
            return attention.set_cache_buffer(buffer);
        }

        size_t read_cache_data(void *buffer, size_t buffer_size) const override
        {
            return attention.read_cache_data(buffer, buffer_size);
        }

        size_t write_cache_data(const void *buffer, size_t buffer_size) override
        {
            return attention.write_cache_data(buffer, buffer_size);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            attention.load(path + "self_attn.", loader);
        }

    public:
        AttentionBlock attention;
    };

    template <class InputNormBlock,
              class AttentionBlock,
              class PostNormBlock,
              class MLPBlock> class LMBlock1 : public LMAttentionBlock<AttentionBlock>
    {
    private:
        typedef LMAttentionBlock<AttentionBlock> Base;
    public:
        LMBlock1() = default;
        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, max_length),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size,
                  num_kv_heads, head_dim, max_length),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, mlp_intermediate_size) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size,
                  mlp_intermediate_size1, mlp_intermediate_size2,
                  num_kv_heads, head_dim, max_length),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, mlp_intermediate_size1, mlp_intermediate_size2) {}


        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias, bool gate_use_bias)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size,
                  mlp_intermediate_size1, mlp_intermediate_size2,
                  num_kv_heads, max_length,
                  q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim, use_bias),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, mlp_intermediate_size1, mlp_intermediate_size2, gate_use_bias) {}

        LMBlock1(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size,
                  num_kv_heads, max_length,
                  q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim, use_bias),
              input_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            ggml::tensor *residual = hidden_states;

            hidden_states = input_layernorm.forward(ctx, hidden_states);
            hidden_states = Base::attention.forward(ctx, hidden_states, n_past);
            if (Base::get_id() == 0)
            {
                //inspect_tensor(hidden_states, "attention");
            }

            if (scale_depth > 0.0f)
            {
                hidden_states = ggml::scale(ctx, hidden_states, scale_depth);
            }

            hidden_states = ggml::add(ctx, hidden_states, residual);
            residual = hidden_states;

            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
            last_result_post_attn_norm = hidden_states;

            hidden_states = mlp.forward(ctx, hidden_states);

            if (scale_depth > 0.0f)
            {
                hidden_states = ggml::scale(ctx, hidden_states, scale_depth);
            }

            hidden_states = ggml::add(ctx, hidden_states, residual);

            return hidden_states;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *hidden_states2, int n_past) override
        {
            ggml::tensor *residual = hidden_states;

            hidden_states = input_layernorm.forward(ctx, hidden_states);
            hidden_states = Base::attention.forward(ctx, hidden_states, n_past);
            if (Base::get_id() == 0)
            {
                //inspect_tensor(hidden_states, "attention");
            }

            if (scale_depth > 0.0f)
            {
                hidden_states = ggml::scale(ctx, hidden_states, scale_depth);
            }

            hidden_states = ggml::add(ctx, hidden_states, residual);
            residual = hidden_states;

            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
            last_result_post_attn_norm = hidden_states;

            hidden_states = mlp.forward(ctx, hidden_states, hidden_states2);

            if (scale_depth > 0.0f)
            {
                hidden_states = ggml::scale(ctx, hidden_states, scale_depth);
            }

            hidden_states = ggml::add(ctx, hidden_states, residual);

            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += input_layernorm.get_param_num(effective_only);
            r += post_attention_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Base::set_id(id);
            input_layernorm.set_id(id);
            post_attention_layernorm.set_id(id);
            mlp.set_id(id);
        }

        void reserve_batch_size(int size) override
        {
            Base::reserve_batch_size(size);
            post_attention_layernorm.reserve_batch_size(size);
            input_layernorm.reserve_batch_size(size);
            mlp.reserve_batch_size(size);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            input_layernorm.load(path + "input_layernorm.", loader);
            post_attention_layernorm.load(path + "post_attention_layernorm.", loader);
            mlp.load(path + "mlp.", loader);
        }

    public:
        InputNormBlock input_layernorm;
        PostNormBlock post_attention_layernorm;
        MLPBlock mlp;
        float scale_depth = -1.0;
    public:
        ggml::tensor *last_result_post_attn_norm = nullptr;
    };

    template <class PostNormBlock,
              class MLPBlock> class LMNoAttnBlock : public Block
    {
    public:
        LMNoAttnBlock() = default;
        LMNoAttnBlock(InitContext *ctx, int hidden_size, int intermediate_size)
            : post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            ggml::tensor *residual = ggml::dup(ctx, hidden_states);
            ggml::build_forward_expand(ctx, residual);

            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
            hidden_states = mlp.forward(ctx, hidden_states);

            hidden_states = ggml::add_inplace(ctx, hidden_states, residual);

            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += post_attention_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Block::set_id(id);
            post_attention_layernorm.set_id(id);
            mlp.set_id(id);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            post_attention_layernorm.load(path + "post_attention_layernorm.", loader);
            mlp.load(path + "mlp.", loader);
        }

    public:
        PostNormBlock post_attention_layernorm;
        MLPBlock mlp;
    };

    template <class PreAttnNormBlock,
              class AttentionBlock,
              class PostAttnNormBlock,
              class PreMLPNormBlock,
              class MLPBlock,
              class PostMLPNormBlock> class LMBlock4 : public LMAttentionBlock<AttentionBlock>
    {
    private:
        typedef LMAttentionBlock<AttentionBlock> Base;
    public:
        LMBlock4() = default;
        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, max_length),
              pre_attention_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length),
              pre_attention_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length),
              pre_attention_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        LMBlock4(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length, bool qkv_bias, bool o_bias)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, qkv_bias, o_bias),
              pre_attention_layernorm(ctx, hidden_size),
              post_attention_layernorm(ctx, hidden_size),
              pre_mlp_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size),
              post_mlp_layernorm(ctx, hidden_size) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            ggml::tensor *residual = hidden_states;

            hidden_states = pre_attention_layernorm.forward(ctx, hidden_states);
            hidden_states = Base::attention.forward(ctx, hidden_states, n_past);
            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);

            hidden_states = ggml::add(ctx, residual, hidden_states);
            residual = hidden_states;

            hidden_states = pre_mlp_layernorm.forward(ctx, hidden_states);
            hidden_states = mlp.forward(ctx, hidden_states);
            hidden_states = post_mlp_layernorm.forward(ctx, hidden_states);

            hidden_states = ggml::add(ctx, residual, hidden_states);

            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += pre_attention_layernorm.get_param_num(effective_only);
            r += post_attention_layernorm.get_param_num(effective_only);
            r += pre_mlp_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            r += post_mlp_layernorm.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Base::set_id(id);

            pre_attention_layernorm.set_id(id);
            post_attention_layernorm.set_id(id);
            pre_mlp_layernorm.set_id(id);
            mlp.set_id(id);
            post_mlp_layernorm.set_id(id);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            pre_attention_layernorm.load(path + "pre_attention_layernorm.", loader);
            post_attention_layernorm.load(path + "post_attention_layernorm.", loader);
            pre_mlp_layernorm.load(path + "pre_mlp_layernorm.", loader);
            mlp.load(path + "mlp.", loader);
            post_mlp_layernorm.load(path + "post_mlp_layernorm.", loader);
        }

    public:
        PreAttnNormBlock pre_attention_layernorm;
        PostAttnNormBlock post_attention_layernorm;
        PreMLPNormBlock pre_mlp_layernorm;
        MLPBlock mlp;
        PostMLPNormBlock post_mlp_layernorm;
    };

    class BaseTensorPosHelper
    {
    public:
        BaseTensorPosHelper(int max_length);
        virtual ggml::tensor *allocate_pos_tensor(InitContext *ctx);
        virtual void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen);
    protected:
        const int max_length;
        std::vector<int> v_pos;
    };

    class TensorPosHelperParam
    {
    public:
        static void set(BaseTensorPosHelper* helper);
        static BaseTensorPosHelper *get(int max_length);
    private:
        static BaseTensorPosHelper* helper;
    };

    class TensorPosHelperPrelude
    {
    public:
        TensorPosHelperPrelude(BaseTensorPosHelper *helper);
        void done();
    };

    class CoreAttention : public Block
    {
    public:
        CoreAttention() : def_pos_helper(0), num_attention_heads(0), num_kv_heads(0), max_length(0) {}

        CoreAttention(InitContext *ctx, int num_attention_heads, int num_kv_heads, int max_length)
            : CoreAttention(ctx, num_attention_heads, num_kv_heads, max_length, TensorPosHelperParam::get(max_length))
        {}

        CoreAttention(InitContext *ctx, int num_attention_heads, int num_kv_heads, int max_length, BaseTensorPosHelper *helper)
            : def_pos_helper(max_length),
              num_attention_heads(num_attention_heads),
              num_kv_heads(num_kv_heads),
              pos(nullptr),
              max_length(max_length),
              attn_scaling_factor(-1.0f),
              attn_scores_pp(nullptr),
              causal(true),
              shift_pending(),
              attn_scaling(true),
              last_attn_scores(nullptr),
              sinks(BlockParams::CoreAttentionUseSinks::get() > 0 ?
                    ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, BlockParams::CoreAttentionUseSinks::get())
                    : nullptr),
              pos_helper(helper ? helper : new BaseTensorPosHelper(max_length))
        {
            allocate_pos_tensor(ctx);
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
            r += ggml::nelements(sinks);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

    protected:
        virtual void allocate_pos_tensor(InitContext *ctx);
        virtual void prepare_pos_tensor(ComputeContext *ctx, const int n_past, const int qlen);

        // k: [heads, qlen, head_size]
        // q: [heads, qlen, head_size]
        // v: [heads, head_size, klen]
        virtual ggml::tensor *calc_attn_scores(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
                                            ggml::tensor *key_layer, ggml::tensor *query_layer, ggml::tensor *value_layer);
        virtual ggml::tensor *attn_scores_to_probs(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
                                            ggml::tensor *attn_scores);

        // input & output: [qlen, heads, head_size]
        // CAUTION: **inplace** operation is assumed.
        virtual ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const { return k; }
        virtual ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const { return q; }
        virtual ggml::tensor *apply_pos_embedding_kq(ComputeContext *ctx, ggml::tensor *kq, int hidden_size, int qlen, ggml::tensor *past) const { return kq; }

        virtual void before_forward(ComputeContext *ctx, const int n_past, const int qlen);

        // k: [qlen, heads, head_size]
        // v: [qlen, hidden_size]
        virtual void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) = 0;

        // output: [heads, qlen, head_size]
        virtual ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) = 0;

        // output: [heads, head_size, klen]
        virtual ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) = 0;

        virtual ggml::tensor *cross_attention_after_pe(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *query_layer, ggml::tensor *key_layer, ggml::tensor *v);

        virtual ggml::tensor *cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k, ggml::tensor *v);

        // q & k: [qlen, heads, head_size]
        virtual ggml::tensor *cross_attention_3d(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *query_layer, ggml::tensor *key_layer, ggml::tensor *v);

        ggml::tensor *get_last_attn_scores(void)
        {
            return last_attn_scores;
        }

    protected:
        BaseTensorPosHelper def_pos_helper;
    public:
        const int num_attention_heads;
        const int num_kv_heads;
        ggml::tensor *pos;
        ggml::tensor *mask = nullptr;
        const int max_length;
        float attn_scaling_factor;
        Block *attn_scores_pp;
        bool causal;

    protected:
        ShiftPending shift_pending;
        bool attn_scaling;
        ggml::tensor *last_attn_scores;
        ggml::tensor *sinks;
        std::unique_ptr<BaseTensorPosHelper> pos_helper;
    };

    class KVCacheAttention : public CoreAttention
    {
    public:
        KVCacheAttention() : CoreAttention(), k_hidden_size(0), v_hidden_size(0), cache_length(0) {}

        KVCacheAttention(InitContext *ctx, int num_attention_heads, int num_kv_heads, int k_hidden_size, int v_hidden_size, int max_length,
                         int cache_length)
            : CoreAttention(ctx, num_attention_heads, num_kv_heads, max_length),
              k_hidden_size(k_hidden_size),
              v_hidden_size(v_hidden_size),
              cache_length(BlockParams::DisableCache::is_disabled() ? 0 : cache_length),
              k_cache(nullptr), v_cache(nullptr), raw_k(nullptr), raw_v(nullptr)
        {
            if (cache_length > 0)
            {
                if (BlockParams::DisableCache::is_disabled())
                {
                    k_cache = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F16, 1, 1);
                    v_cache = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F16, 1, 1);
                }
                else
                {
                    k_cache = ggml::new_tensor_2d(ctx, ggml::type_fallback(ctx->cache_dtype, k_hidden_size), k_hidden_size, cache_length),
                    v_cache = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F16, cache_length, v_hidden_size);
                }

                ggml::set_name(k_cache, "k_cache");
                ggml::set_name(v_cache, "v_cache");
            }
            else;
        }

        size_t get_cache_size(void) const override
        {
            size_t r = 0;
            if (k_cache)
                r += ggml::nbytes(k_cache);
            if (v_cache)
                r += ggml::nbytes(v_cache);
            return r;
        }

        void  set_cache_buffer(BackendBuffer *buffer) override
        {
            size_t offset = 0;
            if (k_cache)
            {
                buffer->assign_to(k_cache, offset);
                offset += ggml::nbytes(k_cache);
            }
            if (v_cache)
            {
                buffer->assign_to(v_cache, offset);
            }
        }

        size_t read_cache_data(void *buffer, size_t buffer_size) const override;
        size_t write_cache_data(const void *buffer, size_t buffer_size) override;

    protected:
        virtual void before_forward(ComputeContext *ctx, const int n_past, const int qlen);

        // k: [batch, qlen, heads, head_size]
        // v: [batch, qlen, hidden_size]
        void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) override;

        // output: [batch, heads, qlen, head_size]
        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override;

        // output: [batch, heads, head_size, klen]
        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override;

    public:
        const int k_hidden_size;
        const int v_hidden_size;
        const int cache_length;
        ggml::tensor *k_cache;
        ggml::tensor *v_cache;
        int batch_size = 1;
    private:
        ggml::tensor *raw_k;
        ggml::tensor *raw_v;
    };

    class BaseConsolidatedQKVAttention : public KVCacheAttention
    {
    public:
        BaseConsolidatedQKVAttention() : KVCacheAttention() {}

        BaseConsolidatedQKVAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseConsolidatedQKVAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, max_length)
        {}

        BaseConsolidatedQKVAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias,
                                     int cache_length)
            : KVCacheAttention(ctx, num_attention_heads, num_kv_heads,
                               head_dim * num_kv_heads,
                               head_dim * num_kv_heads,
                               max_length, cache_length),
              query_key_value(ctx, hidden_size, hidden_size + 2 * (hidden_size / num_attention_heads) * num_kv_heads, qkv_bias),
              dense(ctx, hidden_size, hidden_size, o_bias)
        {
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = KVCacheAttention::get_param_num(effective_only);
            r += query_key_value.get_param_num(effective_only);
            r += dense.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

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
                      int cache_length)
            : KVCacheAttention(ctx, num_attention_heads, num_kv_heads, head_dim * num_kv_heads, head_dim * num_kv_heads, max_length, cache_length),
              q_proj(ctx, hidden_size, head_dim * num_attention_heads, nullptr, qkv_bias),
              k_proj(ctx, hidden_size, head_dim * num_kv_heads, nullptr, BlockParams::OverrideKProjBiased::get(qkv_bias)),
              v_proj(ctx, hidden_size, head_dim * num_kv_heads, nullptr, qkv_bias),
              o_proj(ctx, head_dim * num_attention_heads, hidden_size, o_bias)
        {
        }

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      bool qkv_bias, bool o_bias,
                      int cache_length)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            cache_length)
        {}

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            max_length)
        {}

        BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
             bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias,
                            max_length)
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

        void set_prec(ggml::prec prec) override;

        void set_id(int id) override
        {
            Block::set_id(id);

            q_proj.set_id(id);
            k_proj.set_id(id);
            v_proj.set_id(id);
            o_proj.set_id(id);
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        Linear q_proj, k_proj, v_proj;
        Linear o_proj;
    };

    class BaseNormedAttention : public BaseAttention
    {
    public:
        BaseNormedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias),
              q_norm(ctx, hidden_size),
              k_norm(ctx, hidden_size / num_attention_heads * num_kv_heads)
        {}

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = BaseAttention::get_param_num(effective_only);
            r += q_norm.get_param_num(effective_only);
            r += k_norm.get_param_num(effective_only);
            return r;
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        RMSNorm q_norm;
        RMSNorm k_norm;
    };

    class BaseCachelessAttention : public BaseAttention
    {
    public:
        BaseCachelessAttention() = default;

        BaseCachelessAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
        {}

        BaseCachelessAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, 0),
              raw_k(nullptr),
              raw_v(nullptr)
        {}

    protected:
        void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) override;

        // output: [heads, qlen, head_size]
        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override;

        // output: [heads, head_size, klen]
        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override;
    private:
        ggml::tensor *raw_k;
        ggml::tensor *raw_v;
    };

    void fill_pos_vector(ComputeContext *ctx, std::vector<int> &v_pos, ggml::tensor *pos, int n_past, int qlen);
    void fill_pos_vector(ComputeContext *ctx, std::vector<float> &v_pos, ggml::tensor *pos, int n_past, int qlen);

    // qlen must be 1.
    // This is just a proof of concept.
    template <int sliding_window_len> class BaseSlidingWindowAttentionRingCache : public BaseAttention
    {
    public:
        BaseSlidingWindowAttentionRingCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias, sliding_window_len),
              cache_offset(0),
              indices(ggml::new_tensor_1d(ctx, GGML_TYPE_I32, sliding_window_len))
        {
            v_indices.resize(sliding_window_len);

            ctx->get_allocator()->alloc(indices);
        }

    protected:
        void before_forward(ComputeContext *ctx, const int n_past, const int qlen) override
        {
            if (n_past == 0) cache_offset = 0;

            pos_helper->prepare_pos_tensor(ctx, pos, n_past, qlen);

            // shift cache
            if (shift_pending.shift > 0)
            {
                cache_offset += shift_pending.shift;
                shift_pending.clear();
            }
        }

        void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) override
        {
            GGML_ASSERT(cache_length > qlen);

            const int write_offset = (cache_offset + n_past) % cache_length;
            const int empty = cache_length - write_offset;
            const int write_len = empty >= qlen ? qlen : empty;
            const int remain = qlen - write_len;

            {
                ggml::tensor * k_cache_view = ggml::view_1d(ctx, k_cache, write_len * k_hidden_size,
                        ggml::row_size(k_cache) * write_offset);

                ggml::tensor * v_cache_view = ggml::view_1d(ctx, v_cache, write_len * v_hidden_size,
                        ggml::element_size(v_cache) * v_hidden_size * write_offset);

                ggml::tensor * k_view = ggml::view_1d(ctx, k, write_len * k_hidden_size, 0);
                ggml::tensor * v_view = ggml::view_1d(ctx, v, write_len * v_hidden_size, 0);

                ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_view, k_cache_view));
                ggml::build_forward_expand(ctx, ggml::cpy(ctx, v_view, v_cache_view));
            }

            if (remain > 0)
            {
                ggml::tensor * k_cache_view = ggml::view_1d(ctx, k_cache, remain * k_hidden_size,
                                            0);

                ggml::tensor * v_cache_view = ggml::view_1d(ctx, v_cache, remain * v_hidden_size,
                                            0);

                ggml::tensor * k_view = ggml::view_1d(ctx, k, remain * k_hidden_size,
                        write_len * k_hidden_size * ggml::element_size(k));

                ggml::tensor * v_view = ggml::view_1d(ctx, v, remain * v_hidden_size,
                        write_len * v_hidden_size * ggml::element_size(v));

                ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_view, k_cache_view));
                ggml::build_forward_expand(ctx, ggml::cpy(ctx, v_view, v_cache_view));
            }

            {
                const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;
                const int start = (cache_offset + n_past + qlen - total) % cache_length;

                for (int i = 0; i < total; i++)
                    v_indices[i] = (start + i) % cache_length;

                Backend::write_tensor_data(indices, v_indices.data(), 0, total * sizeof(v_indices[0]));
            }
        }

        // output: [heads, qlen, head_size]
        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            const int repeat = num_attention_heads / num_kv_heads;

            const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;

            ggml::tensor *indices_view = ggml::view_1d(ctx, indices, total, 0);
            ggml::tensor *k_cache_view = ggml::view_2d(ctx, k_cache, k_hidden_size, cache_length,
                                            ggml::row_size(k_cache), 0);
            ggml::tensor *key_layer = ggml::get_rows(ctx, k_cache_view, indices_view);

            key_layer = ggml::reshape_3d(ctx, key_layer, head_size, num_kv_heads, total);  // [qlen, heads, head_size]
            key_layer = ggml::permute(ctx, key_layer, 0, 2, 1, 3);                         // [heads, qlen, head_size]
            if (ggml::is_quantized(key_layer))
                key_layer = ggml::cont(ctx, key_layer);

            return key_layer;
        }

        // output: [heads, head_size, klen]
        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            const int repeat = num_attention_heads / num_kv_heads;

            const int total = n_past + qlen > cache_length ? cache_length : n_past + qlen;

            ggml::tensor *indices_view = ggml::view_1d(ctx, indices, total, 0);
            ggml::tensor *v_cache_view = ggml::view_2d(ctx, v_cache, v_hidden_size, cache_length,
                                            v_hidden_size * ggml::element_size(v_cache), 0);
            ggml::tensor *value_layer  = ggml::get_rows(ctx, v_cache_view, indices_view);

            value_layer = ggml::reshape_3d(ctx, value_layer, head_size, num_kv_heads, total);  // [qlen, heads, head_size]
            value_layer = ggml::permute(ctx, value_layer, 1, 2, 0, 3);                         // [heads, head_size, klen]
            value_layer = ggml::cont(ctx, value_layer);

            return value_layer;
        }

    public:
        int cache_offset;
        ggml::tensor *indices;
        std::vector<int> v_indices;
    };

    template <int sliding_window_len> class BaseSlidingWindowAttentionFullCache : public BaseAttention
    {
    public:
        BaseSlidingWindowAttentionFullCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSlidingWindowAttentionFullCache(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        BaseSlidingWindowAttentionFullCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, max_length),
              indices(ggml::new_tensor_1d(ctx, GGML_TYPE_I32, 1)) // to ensure number of tensors are the same
        {
        }

    protected:

        // output: [heads, qlen, head_size]
        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
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

            ggml::tensor *key_layer = nullptr;

            key_layer = ggml::view_1d(ctx, k_cache, len * k_hidden_size, offset * ggml::row_size(k_cache));
            key_layer = ggml::reshape_3d(ctx, key_layer, head_size, num_kv_heads, len);  // [qlen, heads, head_size]
            key_layer = ggml::permute(ctx, key_layer, 0, 2, 1, 3);                       // [heads, qlen, head_size]
            if (ggml::is_quantized(key_layer))
                key_layer = ggml::cont(ctx, key_layer);

            return key_layer;
        }

        // output: [heads, head_size, klen]
        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            int64_t len = n_past + qlen;
            int64_t offset = 0;
            if (len > sliding_window_len)
            {
                offset = len - sliding_window_len;
                len = sliding_window_len;
            }

            ggml::tensor * value_layer = ggml::view_3d(ctx,
                            v_cache,
                            len, head_size, num_kv_heads,
                            cache_length * ggml::element_size(v_cache),
                            cache_length * ggml::element_size(v_cache) * head_size,
                            offset * ggml::element_size(v_cache)); // [heads, head_size, klen]
            return value_layer;
        }

    public:
        ggml::tensor *indices;
    };

    template <int sliding_window_len, int extra_len = 512> class BaseSlidingWindowAttentionPartialCache : public BaseAttention
    {
    public:
        BaseSlidingWindowAttentionPartialCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSlidingWindowAttentionPartialCache(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        BaseSlidingWindowAttentionPartialCache(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, sliding_window_len + extra_len),
              indices(ggml::new_tensor_1d(ctx, GGML_TYPE_I32, 1)), // to ensure number of tensors are the same
              cache_offset(0)
        {
        }

    protected:

        void before_forward(ComputeContext *ctx, const int n_past, const int qlen) override
        {
            if (n_past == 0) cache_offset = 0;

            pos_helper->prepare_pos_tensor(ctx, pos, n_past, qlen);

            // shift cache
            if (shift_pending.shift > 0)
            {
                // do nothing
                shift_pending.clear();
            }
        }

        void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) override
        {
            CHATLLM_CHECK(qlen == 1)  << "qlen must be 1";

            {
                const int write_offset = cache_offset + n_past;
                const int empty = cache_length - write_offset;

                if (empty < qlen)
                {
                    int remain = sliding_window_len - qlen;
                    int shift = cache_length - remain;

                    ggml::tensor * k_cache_remain = ggml::view_1d(ctx, k_cache, remain * k_hidden_size,
                                                ggml::row_size(k_cache) * shift);
                    ggml::tensor * k_cache_1d = ggml::view_1d(ctx, k_cache, remain * k_hidden_size,
                                                0);

                    ggml::tensor * k_remain_dup = ggml::dup(ctx, k_cache_remain);

                    ggml::tensor * v_cache_remain = ggml::view_2d(ctx, v_cache, remain, v_hidden_size,
                                                cache_length * ggml::element_size(v_cache),
                                                shift * ggml::element_size(v_cache));
                    ggml::tensor * v_cache_2d =     ggml::view_2d(ctx, v_cache, remain, v_hidden_size,
                                                cache_length * ggml::element_size(v_cache),
                                                0);

                    ggml::tensor * v_remain_dup = ggml::dup(ctx, v_cache_remain);

                    ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_remain_dup, k_cache_1d));
                    ggml::build_forward_expand(ctx, ggml::cpy(ctx, v_remain_dup, v_cache_2d));

                    cache_offset -= shift;
                }
            }

            // patch n_past for memory estimation
            const int write_offset = cache_offset + n_past < cache_length ? cache_offset + n_past : cache_length - qlen;
            if (cache_offset + n_past >= cache_length) cache_offset = 0;

            // compute the transposed [N, n_embd] V matrix
            ggml::tensor * Vcur = ggml::transpose(ctx, v); // ggml::reshape_2d(ctx, tmpv, kv_hidden_size, qlen));

            ggml::tensor * k_cache_view = ggml::view_1d(ctx, k_cache, qlen * k_hidden_size,
                                        ggml::row_size(k_cache) * write_offset);

            ggml::tensor * v_cache_view = ggml::view_2d(ctx, v_cache, qlen, v_hidden_size,
                    cache_length * ggml::element_size(v_cache), write_offset * ggml::element_size(v_cache));

            ggml::tensor * k_view = ggml::view_1d(ctx, k, qlen * k_hidden_size, 0);

            // important: storing RoPE-ed version of K in the KV cache!
            ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_view, k_cache_view));
            ggml::build_forward_expand(ctx, ggml::cpy(ctx, Vcur, v_cache_view));
        }

        // output: [heads, qlen, head_size]
        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            int64_t len = n_past + qlen;
            if (len > sliding_window_len)
                len = sliding_window_len;
            int64_t offset = cache_offset + n_past + qlen - len;

            // patch memory estimation
            if (offset + len > cache_length)
                offset = 0;

            CHATLLM_CHECK(offset >= 0) << "offset must >= 0";

            ggml::tensor *key_layer = nullptr;

            key_layer = ggml::view_1d(ctx, k_cache, len * k_hidden_size, offset * ggml::row_size(k_cache));
            key_layer = ggml::reshape_3d(ctx, key_layer, head_size, num_kv_heads, len);  // [qlen, heads, head_size]
            key_layer = ggml::permute(ctx, key_layer, 0, 2, 1, 3);                       // [heads, qlen, head_size]
            if (ggml::is_quantized(key_layer))
                key_layer = ggml::cont(ctx, key_layer);

            return key_layer;
        }

        // output: [heads, head_size, klen]
        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            const int head_size = hidden_size / num_attention_heads;
            int64_t len = n_past + qlen;
            if (len > sliding_window_len)
                len = sliding_window_len;
            int64_t offset = cache_offset + n_past + qlen - len;

            // patch for memory estimation
            if (offset + len > cache_length)
                offset = 0;

            ggml::tensor * value_layer = ggml::view_3d(ctx,
                            v_cache,
                            len, head_size, num_kv_heads,
                            cache_length * ggml::element_size(v_cache),
                            cache_length * ggml::element_size(v_cache) * head_size,
                            offset * ggml::element_size(v_cache)); // [heads, head_size, klen]
            return value_layer;
        }

    public:
        ggml::tensor *indices;
        int cache_offset;
    };

    enum RoPEMode
    {
        Interleaved     = 0,                            // IQIQ......IQ
        Original        = GGML_ROPE_TYPE_NEOX,          // II...IQQ...Q (i.e. Su's Paper)
        MROPE           = GGML_ROPE_TYPE_MROPE,
        VISION          = GGML_ROPE_TYPE_VISION,
        IMROPE          = GGML_ROPE_TYPE_IMROPE,        // interleaved MROPE
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
              n_original_ctx(0),
              mrope_sections(nullptr),
              use_rope(true),
              freq_factors(nullptr),
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
              n_original_ctx(0),
              mrope_sections(nullptr),
              use_rope(true),
              freq_factors(nullptr),
              rope_mode(RoPEMode::Original)
        {
        }

        void setup_yarn(float rope_theta, int original_max_position_embeddings, float beta_fast, float beta_slow, float factor)
        {
            auto get_mscale = [](float scale, float mscale = 1.0f)
            {
                if (scale <= 1.0f) return 1.0f;
                return 0.1f * mscale * logf(scale) + 1.0f;
            };

            this->freq_base = rope_theta;
            this->n_original_ctx = original_max_position_embeddings;
            this->beta_slow = beta_slow;
            this->beta_fast = beta_fast;
            this->freq_scale  = 1 / factor;
            this->attn_factor = get_mscale(factor);
            this->ext_factor  = 1.0f;
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
        const int  *mrope_sections;
        bool  use_rope;
        ggml::tensor *freq_factors;
        RoPEMode rope_mode;

    protected:
        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            if (!use_rope) return k;
            return ggml::rope_ext_inplace(ctx, k, past, freq_factors, rope_dim, rope_mode, n_original_ctx,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, mrope_sections);    // [qlen, heads, head_size]
        }
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            if (!use_rope) return q;
            return ggml::rope_ext_inplace(ctx, q, past, freq_factors, rope_dim, rope_mode, n_original_ctx,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, mrope_sections);    // [qlen, heads, head_size];
        }
    };

    class GLMSelfAttention : public RoPESelfAttention<BaseConsolidatedQKVAttention>
    {
    public:
        GLMSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true)
        {
            rope_dim = (hidden_size / num_attention_heads) / 2;
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override;
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override;
    };

    class GLMBlock : public Block
    {
    public:
        GLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int num_hidden_layers, int max_length)
            : input_layernorm(ctx, hidden_size),
              attention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length),
              post_attention_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size),
              num_hidden_layers(num_hidden_layers) {}

        using Block::forward;
        virtual ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

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

        void  set_cache_buffer(BackendBuffer *buffer) override
        {
            return attention.set_cache_buffer(buffer);
        }

        size_t read_cache_data(void *buffer, size_t buffer_size) const override
        {
            return attention.read_cache_data(buffer, buffer_size);
        }

        size_t write_cache_data(const void *buffer, size_t buffer_size) override
        {
            return attention.write_cache_data(buffer, buffer_size);
        }

    public:
        LayerNorm input_layernorm;
        GLMSelfAttention attention;
        LayerNorm post_attention_layernorm;
        GLMMLP mlp;
        int num_hidden_layers;
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
            : BaseMLP(ctx, hidden_size, intermediate_size, act, false, false, false)
        {}

        BaseMLP(InitContext *ctx, int hidden_size, int intermediate_size, ActFunc act,
               bool gate_use_bias, bool down_use_bias, bool up_use_bias)
            : gate_proj(ctx, hidden_size, intermediate_size, gate_use_bias),
              down_proj(ctx, intermediate_size, hidden_size, down_use_bias),
                up_proj(ctx, hidden_size, intermediate_size, up_use_bias),
              act(act)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += gate_proj.get_param_num(effective_only);
            r += down_proj.get_param_num(effective_only);
            r += up_proj.get_param_num(effective_only);
            return r;
        }

        void set_prec(ggml::prec prec) override
        {
            Block::set_prec(prec);
            down_proj.set_prec(prec);
        }

        void load(const std::string &path, TensorLoader *loader) override;

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

    template<class MLP, bool use_bias = false> class GatedMLP : public MLP
    {
    public:
        GatedMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  MLP(ctx, hidden_size, intermediate_size),
           gate(ctx, hidden_size, 1, use_bias)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            ggml::tensor *scale = gate.forward(ctx, hidden_states);
            ggml::tensor *r     = MLP::forward(ctx, hidden_states);
            scale = ggml::sigmoid(ctx, scale);
            r = ggml::mul(ctx, r, scale);
            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += MLP::get_param_num(effective_only);
            r += gate.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            gate.load(path + "gate.", loader);
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

        CombinedMLP(InitContext *ctx, int hidden_size, int intermediate_size1, int intermediate_size2, bool mlp1_use_bias)
            :  mlp1(ctx, hidden_size, intermediate_size1, mlp1_use_bias),
               mlp2(ctx, hidden_size, intermediate_size2)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            ggml::tensor *r1 = mlp1.forward(ctx, hidden_states);
            ggml::tensor *r2 = mlp2.forward(ctx, hidden_states);
            ggml::tensor *r = ggml::add(ctx, r1, r2);
            return r;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *hidden_states2) override
        {
            ggml::tensor *r1 = mlp1.forward(ctx, hidden_states, hidden_states2);
            ggml::tensor *r2 = mlp2.forward(ctx, hidden_states);
            ggml::tensor *r = ggml::add(ctx, r1, r2);
            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += mlp1.get_param_num(effective_only);
            r += mlp2.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Block::set_id(id);
            mlp1.set_id(id);
            mlp2.set_id(id);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            mlp1.load(path + "mlp1.", loader);
            mlp2.load(path + "mlp2.", loader);
        }

    public:
        MLP1 mlp1;
        MLP2 mlp2;
    };

    class MultiMLP : public Block
    {
    public:
        MultiMLP(InitContext *ctx, int hidden_size, int intermediate_size, int num_local_experts, int num_experts_per_tok,
                  ActFunc act, bool use_bias, int group_size = 1);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts) override;

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

    public:
        MultiLinear gate;
        MultiLinear down;
        MultiLinear up;
        const ActFunc act;
        const int num_local_experts;
        const int num_experts_per_tok;
        const int group_size;
    };

    class GenericSparseMLP : public Block
    {
    public:
        enum ScoreFunc
        {
            Softmax,
            Sigmoid,
        };
        GenericSparseMLP() = default;
        GenericSparseMLP(InitContext *ctx, int hidden_size, int num_local_experts, int num_experts_per_tok,
                  bool gate_score_use_bias, bool grouped_max, bool router_scale, bool gate_use_bias);

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *hidden_states_for_gating) override;

        void set_prec(ggml::prec prec) override
        {
            gate.set_prec(prec);
        }

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

        void set_experts(MultiMLP *experts);
    public:
        const int num_local_experts;
        const int num_experts_per_tok;
        Linear gate;
        CPUMover *mover;        // when `+moe_on_cpu` is set, all things are done on CPU except for `gate`
        ggml::tensor *gate_score_correction_bias;
        ggml::tensor *group_indices;
        ggml::tensor *router_scale;
        bool norm_topk_prob;
        ScoreFunc score_func;
        float routed_scaling_factor;
        bool always_scaling;
        bool pre_weighting;
    protected:
        virtual ggml::tensor *forward_with_experts(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts,
            ggml::tensor *weights,
            std::function<ggml::tensor *(ComputeContext *ctx, ggml::tensor *hidden_states,
                ggml::tensor *selected_experts)> experts_forward);

        virtual ggml::tensor *select_experts(ComputeContext *ctx, ggml::tensor *corrected_score);

        virtual ggml::tensor *forward_with_experts(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts,
            ggml::tensor *weights);
        virtual ggml::tensor *calc_experts_outputs(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts);
    protected:
        MultiMLP *p_experts;    // shared pointer, not owned!
    };

    class GenericGroupedSparseMoE : public GenericSparseMLP
    {
    public:
        GenericGroupedSparseMoE(InitContext *ctx, int hidden_size, int num_experts, int experts_per_tok,
            bool gate_score_use_bias, bool grouped_max, bool router_scale, bool gate_use_bias)
            : GenericSparseMLP(ctx, hidden_size, num_experts, experts_per_tok, gate_score_use_bias, grouped_max, router_scale, gate_use_bias),
              n_group(-1), topk_group(-1)
        {
            score_func = ScoreFunc::Sigmoid;
            always_scaling = true;
        }
    protected:
        ggml::tensor *select_experts(ComputeContext *ctx, ggml::tensor *corrected_score) override;

    public:
        int n_group;
        int topk_group;
    };

    class BaseSparseMLP : public GenericSparseMLP
    {
    public:
        BaseSparseMLP() = default;
        BaseSparseMLP(InitContext *ctx, int hidden_size, int intermediate_size, int num_local_experts, int num_experts_per_tok,
                  ActFunc act, bool gate_score_use_bias, bool grouped_max = false, bool router_scale = false, bool experts_use_bias = false, bool gate_use_bias = false):
            GenericSparseMLP(ctx, hidden_size, num_local_experts, num_experts_per_tok, gate_score_use_bias, grouped_max, router_scale, gate_use_bias),
            experts(ctx, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, act, experts_use_bias)
        {
            set_experts(&experts);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = GenericSparseMLP::get_param_num(effective_only);
            r += experts.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            GenericSparseMLP::load(path, loader);
            experts.load(path + "experts.", loader);
        }
    public:
        MultiMLP experts;
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

        LlamaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    class FullBiasedSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        FullBiasedSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, max_length, true, true)
        {
            rope_mode = RoPEMode::Original;
        }

        FullBiasedSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true)
        {
            rope_mode = RoPEMode::Original;
        }

        FullBiasedSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, true, true)
        {
            rope_mode = RoPEMode::Original;
        }
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

        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class Llama31SelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        Llama31SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, max_length, false, false)
        {
            freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size / num_attention_heads / 2);
            ctx->get_allocator()->alloc(freq_factors);
        }

        Llama31SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
            freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size / num_attention_heads / 2);
            ctx->get_allocator()->alloc(freq_factors);
        }
    };

    class Llama31Block : public LMBlock1<RMSNorm, Llama31SelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        Llama31Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
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
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = dense.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

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
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += dense.get_param_num(effective_only);
            r += out_proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        Linear dense;
        ActFunc act;
        Linear out_proj;
    };

    class BCEFinalNorm : public Block
    {
    public:
        BCEFinalNorm() {}

        BCEFinalNorm(InitContext *ctx, int hidden_size):
            eps(1e-5f)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            return 0;
        }
    public:
        float eps;
    };

    class MiniCPMMeanPooling : public RMSNorm
    {
    public:
        MiniCPMMeanPooling() {}

        MiniCPMMeanPooling(InitContext *ctx, int normalized_shape)
            : RMSNorm(ctx, normalized_shape)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        void set_max_length(InitContext *ctx, int max_length);

    public:
        ggml::tensor *pos;
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
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            return k;
        }
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
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

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *attention_output) override;

        void load(const std::string &path, TensorLoader *loader) override;

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
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        void load(const std::string &path, TensorLoader *loader) override;

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
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

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

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        RobertaSelfAttention attention;
        LayerNorm post_attention_layernorm;
        RobertaMLP mlp;
        LayerNorm output_layernorm;
    };

    typedef TheBiasedGELUMLP Phi2MLP;

    template <class InputNormBlock,
              class AttentionBlock,
              class MLPBlock> class LMBlock2 : public LMAttentionBlock<AttentionBlock>
    {
    private:
        typedef LMAttentionBlock<AttentionBlock> Base;
    public:
        LMBlock2() = default;

        LMBlock2(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
                  int max_length, bool qkv_bias = true, bool o_bias = true)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads,
                   max_length, qkv_bias, o_bias),
              input_layernorm(ctx, hidden_size),
              mlp(ctx, hidden_size, intermediate_size) {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            ggml::tensor *residual = ggml::dup(ctx, hidden_states);
            ggml::build_forward_expand(ctx, residual);

            hidden_states = input_layernorm.forward(ctx, hidden_states);

            // TODO: why `hidden_state` is overwritten somewhere?
            ggml::tensor *dup_hidden = ggml::dup(ctx, hidden_states);
            ggml::build_forward_expand(ctx, dup_hidden);

            ggml::tensor *attn_outputs = Base::attention.forward(ctx, dup_hidden, n_past);
            ggml::tensor *feed_forward_hidden_states = mlp.forward(ctx, dup_hidden);

            ggml::tensor *r = ggml::add_inplace(ctx, feed_forward_hidden_states, residual);
            r = ggml::add_inplace(ctx, r, attn_outputs);

            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += input_layernorm.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            Base::set_id(id);
            input_layernorm.set_id(id);
            mlp.set_id(id);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            input_layernorm.load(path + "input_layernorm.", loader);
            mlp.load(path + "mlp.", loader);
        }

    public:
        InputNormBlock input_layernorm;
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

        MistralSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    class ALiBiSelfAttention : public BaseAttention
    {
    public:
        ALiBiSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : ALiBiSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length)
        {
        }

        ALiBiSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length);

    protected:
        ggml::tensor *attn_scores_to_probs(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
            ggml::tensor *attn_scores) override;

    public:
        float  bias_max;
        float  scale;
    };

    class BaichuanBlock : public LMBlock1<RMSNorm, ALiBiSelfAttention, RMSNorm, SiLUMLP>
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
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override;
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override;

    public:
        int seq_length;
        bool use_dynamic_ntk;
        bool use_logn_attn;
    protected:
        ggml::tensor *logn_list;
        std::vector<float> logn_list_data;
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

        QWen2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
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
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override;
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override;

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

        MistralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
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

    class MiniCPMBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        MiniCPMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template <class Norm, class BaseAttn> class QKNormedAttention : public RoPESelfAttention<BaseAttn>
    {
    public:
        QKNormedAttention() : BaseAttn() {}

        QKNormedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention<BaseAttn>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias),
              k_layernorm(ctx, head_dim),
              q_layernorm(ctx, head_dim),
              post_norm(false)
        {
            RoPESelfAttention<BaseAttn>::rope_mode = RoPEMode::Original;
        }

        QKNormedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention<BaseAttn>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias),
              k_layernorm(ctx, hidden_size / num_attention_heads),
              q_layernorm(ctx, hidden_size / num_attention_heads),
              post_norm(false)
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

        void load(const std::string &path, TensorLoader *loader) override
        {
            RoPESelfAttention<BaseAttn>::load(path, loader);
            k_layernorm.load(path + "k_norm.", loader);
            q_layernorm.load(path + "q_norm.", loader);
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            if (!RoPESelfAttention<BaseAttn>::use_rope) return k;

            if (!post_norm)
                k = const_cast<QKNormedAttention *>(this)->k_layernorm.forward(ctx, k);
            k = RoPESelfAttention<BaseAttn>::apply_pos_embedding_k(ctx, k, hidden_size, qlen, past);    // [qlen, heads, head_size]
            if (post_norm)
                k = const_cast<QKNormedAttention *>(this)->k_layernorm.forward(ctx, k);
            return k;
        }

        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            if (!RoPESelfAttention<BaseAttn>::use_rope) return q;

            if (!post_norm)
                q = const_cast<QKNormedAttention *>(this)->q_layernorm.forward(ctx, q);
            q = RoPESelfAttention<BaseAttn>::apply_pos_embedding_q(ctx, q, hidden_size, qlen, past);    // [qlen, heads, head_size];
            if (post_norm)
                q = const_cast<QKNormedAttention *>(this)->q_layernorm.forward(ctx, q);
            return q;
        }

    public:
        Norm k_layernorm;
        Norm q_layernorm;
        bool post_norm;
    };

    // FIXME: this seems problemic. LayerNorm is used in `apply_pos_embedding_xx`, while not `inplace`.
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

    class LongRoPEParam
    {
    public:
        LongRoPEParam(int rope_dim, float scaling_factor)
            : rope_dim(rope_dim), scaling_factor(scaling_factor)
        {}

        virtual const float *get_inv_freq(int max_pos) = 0;

        const int rope_dim;
        const float scaling_factor;
    };

    class Phi3SUSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        Phi3SUSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : Phi3SUSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
        }

        Phi3SUSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
        }

        void config(InitContext *ctx, int original_max_position_embeddings,
            float rope_theta, float short_scaling_factor, float long_scaling_factor, int factor_len, const float *short_factor, const float *long_factor);
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

    typedef TheBiasedGELUMLP StarCoder2MLP;

    template <int sliding_window_len, bool qvk_bias = true, bool o_bias = true> class StarCoder2Block : public LMBlock1<LayerNorm, StarCoder2SelfAttention<sliding_window_len, qvk_bias, o_bias>, LayerNorm, StarCoder2MLP>
    {
    public:
        StarCoder2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<LayerNorm, StarCoder2SelfAttention<sliding_window_len, qvk_bias, o_bias>, LayerNorm, StarCoder2MLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };
} // namespace chatllm
