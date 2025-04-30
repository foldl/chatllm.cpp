#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>

#include "ggml-backend.h"

namespace chatllm
{
    namespace ggml
    {
        typedef ggml_tensor tensor;
        typedef ggml_type   type;

        tensor *init_tensor(ggml::tensor  *tensor,
                            ggml::type     type,
                            int            n_dims,
                            const int64_t *ne);
        void change_type(ggml::tensor  *tensor, ggml::type type);

        size_t element_size(const ggml::tensor *tensor);
        size_t nbytes(const ggml::tensor *tensor);

        int n_dims(const ggml::tensor * tensor);

        type type_of(const ggml::tensor *tensor);
        type type_of(const ggml::tensor &tensor);
        type parse(const std::string &type);

        // why param type is `ggml::tensor`, but not `op`? `tensor` is an operation (a map) mathematically.
        bool is_view_op(ggml::tensor *a);
        const char *op_name(ggml::tensor *a);
        bool maybe_inplace_op(ggml::tensor *a);

        void set_name(tensor *tensor, const char *name);
        const char *get_name(tensor *tensor);

        void graph_dump_dot(ggml_backend_sched_t sched, const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);

        typedef bool (* need_observe_tensor_evaluation_callback)(ggml::tensor *tensor, void *user_data);
        typedef bool (* observe_tensor_evaluation_callback)(ggml::tensor *tensor, void *user_data);

        void log(enum ggml_log_level level, const char * format, ...);
    }

    class TensorLoader
    {
    public:
        virtual void read_tensor(const std::string &name, ggml::tensor *tensor) = 0;
        virtual void read_tensor(const std::string &name,
                        const std::string &layer_prefix, int num, const std::string &suffix,
                        ggml::tensor *tensor) = 0;
    };

    // Is `ggml_backend_buffer_type_t` a good name?
    typedef ggml_backend_buffer_type_t  ggml_backend_allocator;

    class LayerBufAllocator;

    class BackendBuffer
    {
    public:
        friend LayerBufAllocator;

        void *get_base(void);

        size_t get_size(void) const;

        bool is_host(void);

        ~BackendBuffer();

        void assign_to(ggml::tensor *tensor, size_t offset = 0);

    protected:
        BackendBuffer(ggml_backend_buffer_t buf);

        ggml_backend_buffer_t buf;
    };

    class Backend;

    class BackendBufAllocator
    {
    public:
        enum Usage
        {
            Matrix,
            Others,
            MAX
        };

        BackendBufAllocator(Backend *backend): total(), backend(backend) {}

        virtual BackendBuffer *alloc(size_t size, Usage usage = Usage::Others) = 0;
        virtual bool alloc(ggml::tensor *tensor) = 0;
        virtual bool alloc(ggml::tensor *tensor, Usage usage) = 0;
        virtual size_t get_alloc_size(ggml::tensor *tensor) = 0;
        virtual size_t get_alloc_size(ggml::tensor *tensor, Usage usage) = 0;

        virtual size_t get_alignment(Usage usage) const = 0;
        virtual size_t get_max_size(Usage usage) const = 0;

        virtual bool supported_by_backend(Backend *backend, ggml::tensor *tensor)
        {
            return false;
        }

        Backend *get_backend(void);

        virtual void show_info(void);

    protected:
        size_t total[Usage::MAX];
    public:
        Backend * const backend;
    };

    class LayerBufAllocator : public BackendBufAllocator
    {
    public:
        friend class BackendContext;

        LayerBufAllocator();
        LayerBufAllocator(ggml_backend_allocator alloc, Backend *backend);
        LayerBufAllocator(ggml_backend_allocator alloc_matrix, ggml_backend_allocator alloc_others, Backend *backend);

        BackendBuffer *alloc(size_t size, Usage usage = Usage::Others) override;
        bool alloc(ggml::tensor *tensor) override;
        bool alloc(ggml::tensor *tensor, Usage usage) override;
        size_t get_alloc_size(ggml::tensor *tensor) override;
        size_t get_alloc_size(ggml::tensor *tensor, Usage usage) override;

        bool supported_by_backend(Backend *backend, ggml::tensor *tensor) override;

        size_t get_alignment(Usage usage) const override;

        size_t get_max_size(Usage usage) const override;

        void free_all_buffers(void);

        void show_info(void) override;

        bool operator ==(const LayerBufAllocator &b);

    protected:
        Usage detect_usage(ggml::tensor *tensor);
        ggml_backend_allocator get_allocator(Usage usage);
        ggml_backend_allocator get_allocator(ggml::tensor *tensor);

    protected:
        ggml_backend_allocator alloc_matrix;
        ggml_backend_allocator alloc_others;
        std::vector<std::unique_ptr<BackendBuffer>> buffers;
    };

    class LayerAllocatorManager
    {
    public:
        enum MiscLayer
        {
            Prolog = -1,
            Epilog = -2,
        };

        void set_misc_layer_backend_mapping(int prolog, int epilog);

        void move_to_layer(int layer_id);

        int get_cur_layer(void) const;

        LayerBufAllocator *get_allocator(void);

        LayerBufAllocator *get_allocator(int layer_id);
        LayerBufAllocator *get_allocator(ggml::tensor *tensor);

        void register_tensor_allocator(ggml::tensor *tensor,  LayerBufAllocator *allocator);

        void override_to_cpu_only(bool flag);

    protected:
        int get_mapped_layer_id(int layer_id);
    public:
        std::vector<LayerBufAllocator> allocators;
    protected:
        int prolog_layer_backend_map_to_layer_id = -1;
        int epilog_layer_backend_map_to_layer_id = -1;
        int cur_layer = MiscLayer::Prolog;
        std::map<ggml::tensor *, LayerBufAllocator *> alloc_of_tensor;
        bool cpu_override = false;
    };

    class ComputeManager
    {
    public:
        enum DeviceType {
            // CPU device using system memory
            CPU,
            // GPU device using dedicated memory
            GPU,
            // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
            ACCEL
        };
        struct DeviceInfo
        {
            DeviceType type;
            std::string backend_name;
            std::string name;
            std::string description;
            size_t total_memory;
            size_t free_memory;
        };

        static void init(const std::string &ggml_dir = "");

        static std::string dev_type_to_str(DeviceType type);

        static ggml_backend_allocator get_default_allocator_cpu(bool host_buffer, int gpu_id);

        static ggml_backend_allocator get_default_allocator(ggml_backend_t backend);
        static int get_device_count(void);

        static ggml_backend_t init_backend_device(int index, const char *param = nullptr);

        static ggml_backend_allocator get_default_allocator_offload(int device);

        static size_t get_device_free_memory(int device, size_t *p_total = nullptr);

        static bool get_device_info(int device, DeviceInfo &info);
        static void get_devices_info(std::vector<DeviceInfo> &devs);

        static bool start_rpc_server(int device, const char *endpoints, size_t backend_mem = 0);
        static bool prepare_rpc_devices(const std::string &endpoints);

    protected:
        static ggml_backend_reg_t backend_rpc;
    };

    enum BufferType
    {
        Dedicated,  // only accessible by this backend
        Shared,     // accessible (at least) by this backend and host (CPU)
    };

    class Backend
    {
    public:

    public:
        Backend(ggml_backend_t backend, int n_layers, bool use_gpu);

        bool is_cpu(void) const;

        ggml_backend_allocator get_allocator(BufferType bt);

        void write_tensor_data_async(ggml::tensor * tensor, const void * data, size_t offset, size_t size);

        static void write_tensor_data(ggml::tensor * tensor, const void * data, size_t offset, size_t size);

        static void write_tensor_data(ggml::tensor * tensor, const void * data);

        void read_tensor_data_async(ggml::tensor * tensor, void * data, size_t offset, size_t size);

        static void read_tensor_data(ggml::tensor * tensor, void * data, size_t offset, size_t size);

        static void read_tensor_data(ggml::tensor * tensor, void * data);

        void synchronize(void);

    public:
        ggml_backend_t backend;
        const int n_layers;
        const bool use_gpu;
    private:
        bool _is_cpu;
    };

    class BackendContext
    {
    public:
        enum SplitMethod
        {
            None,
            Layer,
            Row,    // TODO: WIP. I can't figure out a _simple_ way for this yet.
        };

        struct gpu_cfg
        {
            int id;         // ignored for Metal
            int n_layers;
            bool prolog;
            bool epilog;

            void merge(const gpu_cfg &cfg)
            {
                if (cfg.n_layers > 0) n_layers = cfg.n_layers;
                if (cfg.prolog) prolog = true;
                if (cfg.epilog) epilog = true;
            }
        };

        BackendContext();

        void init(const std::vector<gpu_cfg> &gpu_cfgs, const int n_layers, const size_t graph_max_nodes_num, const int n_threads);

        ~BackendContext();

        bool reserve_memory(ggml_cgraph *gf);

        bool alloc_graph(ggml_cgraph *gf);

        void compute_graph(ggml_cgraph *gf);

        void reset();

        void dump_graph(ggml_cgraph *gf, const char *file_name);

        void set_abort_callback(struct llama_context *ctx, bool (*abort_callback)(void * data), void * abort_callback_data);

        void set_eval_observe_callback(ggml::need_observe_tensor_evaluation_callback need_observe_tensor_callback,
            ggml::observe_tensor_evaluation_callback observe_tensor_callback, void *user_data);

        void show_buffer_sizes(void);

        void synchronize(void);

        bool is_using_gpu(void) const;

    public:
        std::vector<Backend> backends;

        ggml_backend_t backend_cpu = nullptr;

        // memory buffers used to evaluate the model
        std::vector<uint8_t> buf_compute_meta;
        ggml_backend_sched_t sched = nullptr;

        // host buffer for the model output (logits and embeddings)
        ggml_backend_buffer_t buf_output = nullptr;

        LayerAllocatorManager layer_allocators;
        LayerBufAllocator host_allocator;

    protected:
        ggml_abort_callback abort_callback      = nullptr;
        void *              abort_callback_data = nullptr;

        std::vector<ggml_backend_t> gg_backends;
        std::vector<ggml_backend_buffer_type_t> gg_bufts;

    public:
        ggml::need_observe_tensor_evaluation_callback need_observe_tensor_callback = nullptr;
        ggml::observe_tensor_evaluation_callback      observe_tensor_callback = nullptr;
        void *                                        observe_tensor_callback_data = nullptr;
    };

    class ComputeContext
    {
    public:
        // additional user options
        struct UserOptions
        {
            bool moe_on_cpu = false;
        };

        ComputeContext(BackendContext *backend_context);

        virtual struct ggml_context *get_ctx() = 0;
        virtual ggml_cgraph *get_cgraph(void);

        virtual void cb_new_tensor(ggml::tensor *tensor);
        virtual void cb_op_tensor(ggml::tensor *tensor);

        virtual void move_to_layer(int layer_id);
        virtual void backend_cpu_override(bool flag);

        BackendBufAllocator *get_allocator(void);
        BackendBufAllocator *get_allocator(ggml::tensor *tensor);
        void register_tensor_allocator(ggml::tensor *tensor,  BackendBufAllocator *allocator);

        virtual Backend *get_backend(void);

        virtual void compute(void);

        virtual void synchronize(void);

        virtual bool allocate(void);

        virtual bool reserve_memory(void);

        virtual void reset(void);

        virtual size_t get_used_mem(void);
        virtual size_t get_mem_size(void);

        virtual bool is_using_gpu(void) const { return backend_context->is_using_gpu(); }

        BackendContext *get_backend_context(void) { return backend_context; }

    public:
        UserOptions user_options;

    protected:
        virtual ggml_backend_sched_t get_sched(void);

        BackendContext *backend_context;
    private:
        void set_backend_context(BackendContext *backend_context);
    };
}