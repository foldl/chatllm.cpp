#pragma once

#include "ggml-backend.h"

namespace chatllm
{
    // Is `ggml_backend_buffer_type_t` a good name?
    typedef ggml_backend_buffer_type_t  ggml_backend_allocator;

    class LayerBufAllocator;

    class BackendBuffer
    {
    public:
        friend LayerBufAllocator;

        void *get_base(void)
        {
            return ggml_backend_buffer_get_base(buf);
        }

        bool is_host(void)
        {
            return ggml_backend_buffer_is_host(buf);
        }

        ~BackendBuffer()
        {
            ggml_backend_buffer_free(buf);
        }

    protected:
        BackendBuffer(ggml_backend_buffer_t buf)
            : buf(buf)
        {}

        ggml_backend_buffer_t buf;
    };

    class BackendBufAllocator
    {
    public:
        enum Usage
        {
            Matrix,
            Others,
        };

        virtual BackendBuffer *alloc(size_t size, Usage usage = Usage::Others) = 0;
        virtual size_t get_alignment(Usage usage) const = 0;
        virtual size_t get_max_size(Usage usage) const = 0;

    protected:
        size_t total = 0;
    };

    class BackendContext;
    class Backend;

    class LayerBufAllocator : public BackendBufAllocator
    {
    public:
        friend BackendContext;

        LayerBufAllocator(): LayerBufAllocator(nullptr, nullptr, nullptr) {}
        LayerBufAllocator(ggml_backend_allocator alloc, Backend *backend): LayerBufAllocator(alloc, alloc, backend) {}
        LayerBufAllocator(ggml_backend_allocator alloc_matrix, ggml_backend_allocator alloc_others, Backend *backend)
            : alloc_matrix(alloc_matrix), alloc_others(alloc_others), backend(backend)
        {}

        BackendBuffer *alloc(size_t size, Usage usage = Usage::Others) override
        {
            total += size;
            ggml_backend_buffer_t buf = nullptr;
            switch (usage)
            {
            case Usage::Matrix:
                buf = ggml_backend_buft_alloc_buffer(alloc_matrix, size);
                break;
            case Usage::Others:
                buf = ggml_backend_buft_alloc_buffer(alloc_others, size);
                break;
            }
            CHATLLM_CHECK(buf) << __FUNCTION__ << "() failed to allocate buffer";

            auto r = new BackendBuffer(buf);
            buffers.emplace_back(r);
            return r;
        }

        size_t get_alignment(Usage usage) const override
        {
            switch (usage)
            {
            case Usage::Matrix:
                return ggml_backend_buft_get_alignment(alloc_matrix);
            case Usage::Others:
                return ggml_backend_buft_get_alignment(alloc_others);
            }
        }

        size_t get_max_size(Usage usage) const override
        {
            switch (usage)
            {
            case Usage::Matrix:
                return ggml_backend_buft_get_max_size(alloc_matrix);
            case Usage::Others:
                return ggml_backend_buft_get_max_size(alloc_others);
            }
        }

        void free_all_buffers(void)
        {
            total = 0;
            buffers.clear();
        }

        Backend *get_backend(void)
        {
            return backend;
        }

    protected:
        ggml_backend_allocator alloc_matrix;
        ggml_backend_allocator alloc_others;
        std::vector<std::unique_ptr<BackendBuffer>> buffers;
        Backend * const backend;
    };

    class ComputeContext
    {
    public:
        virtual struct ggml_context *get_ctx() = 0;
        virtual ggml_backend_sched_t get_sched(void) { return nullptr; }
        virtual ggml_cgraph *get_cgraph(void) { return nullptr; }
        virtual BackendBufAllocator *get_allocator(void) { return nullptr; }
        virtual void cb_new_tensor(ggml_tensor *tensor)
        {
            if (get_sched() && get_backend())
                ggml_backend_sched_set_tensor_backend(get_sched(), tensor, get_backend()->backend);
        }

        virtual void cb_op_tensor(ggml_tensor *tensor)
        {
            if (get_sched() && get_backend())
            {
                if (ggml_backend_supports_op(get_backend()->backend, tensor) || ggml_backend_offload_op(get_backend()->backend, tensor))
                    ggml_backend_sched_set_tensor_backend(get_sched(), tensor, get_backend()->backend);
            }
        }
    protected:
        virtual Backend *get_backend(void) { return nullptr; }
    };

    class ComputeManager
    {
    public:
        static ggml_backend_allocator get_default_allocator_cpu(bool host_buffer, bool use_gpu)
        {
            ggml_backend_allocator allocator = nullptr;

            if (use_gpu)
            {
        #if defined(GGML_USE_CUDA)
                // host buffers should only be used when data is expected to be copied to/from the GPU
                if (host_buffer) {
                    allocator = ggml_backend_cuda_host_buffer_type();
                }
        #elif defined(GGML_USE_SYCL)
                if (host_buffer) {
                    allocator = ggml_backend_sycl_host_buffer_type();
                }
        #elif defined(GGML_USE_CPU_HBM)
                allocator = ggml_backend_cpu_hbm_buffer_type();
        #elif defined(GGML_USE_VULKAN)
                if (host_buffer) {
                    allocator = ggml_backend_vk_host_buffer_type();
                }
        #endif
            }

            if (allocator == nullptr) {
                allocator = ggml_backend_cpu_buffer_type();
            }
            return allocator;
        }

        static ggml_backend_allocator get_default_allocator(ggml_backend_t backend)
        {
            return ggml_backend_get_default_buffer_type(backend);
        }

        static int get_device_count(void)
        {
            int count = 1;
        #if defined(GGML_USE_CUDA)
            count = ggml_backend_cuda_get_device_count();
        #elif defined(GGML_USE_SYCL)
            count = ggml_backend_sycl_get_device_count();
        #elif defined(GGML_USE_VULKAN)
            count = ggml_backend_vk_get_device_count();
        #elif defined(GGML_USE_CANN)
            return ggml_backend_cann_get_device_count();
        #endif
            return count;
        }

        static ggml_backend_t init_backend_device(int index)
        {
        #if defined(GGML_USE_CUDA)
            return ggml_backend_cuda_init(index);
        #elif defined(GGML_USE_SYCL)
            return ggml_backend_sycl_init(index);
        #elif defined(GGML_USE_VULKAN)
            return ggml_backend_vk_init(index);
        #elif defined(GGML_USE_CANN)
            return ggml_backend_cann_init(index);
        #endif
            return nullptr;
        }

        static ggml_backend_allocator get_default_allocator_offload(int gpu)
        {
            ggml_backend_allocator allocator = nullptr;

        #if defined(GGML_USE_METAL)
            allocator = ggml_backend_metal_buffer_type();
        #elif defined(GGML_USE_CUDA)
            allocator = ggml_backend_cuda_buffer_type(gpu);
        #elif defined(GGML_USE_VULKAN)
            allocator = ggml_backend_vk_buffer_type(gpu);
        #elif defined(GGML_USE_SYCL)
            allocator = ggml_backend_sycl_buffer_type(gpu);
        #elif defined(GGML_USE_CANN)
            allocator = ggml_backend_cann_buffer_type(gpu);
        #endif

            if (allocator == nullptr)
                allocator = get_default_allocator_cpu(true, true);
            return allocator;
        }

        static size_t get_device_free_memory(int device, size_t *p_total = nullptr)
        {
            size_t total = 0;
            size_t free = 0;
        #if defined(GGML_USE_CUDA)
            ggml_backend_cuda_get_device_memory(device, &free, &total);
        #elif defined(GGML_USE_SYCL)
            ggml_backend_sycl_get_device_memory(device, &free, &total);
        #elif defined(GGML_USE_VULKAN)
            ggml_backend_vk_get_device_memory(device, &free, &total);
        #elif defined(GGML_USE_CANN)
            ggml_backend_cann_get_device_memory(device, &free, &total);
        #else
            free = 1;
        #endif
            if (p_total) *p_total = total;
            return free;
        }

        static int get_max_devices(void)
        {
        #if defined(GGML_USE_METAL)
            return 1;
        #elif defined(GGML_USE_CUDA)
            return GGML_CUDA_MAX_DEVICES;
        #elif defined(GGML_USE_SYCL)
            return GGML_SYCL_MAX_DEVICES;
        #elif defined(GGML_USE_VULKAN)
            return GGML_VK_MAX_DEVICES;
        #elif defined(GGML_USE_CANN)
            return GGML_CANN_MAX_DEVICES;
        #else
            return 1;
        #endif
        }

        static bool is_gpu_offload_supported(void)
        {
        #if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)   || defined(GGML_USE_VULKAN) || \
            defined(GGML_USE_SYCL) || defined(GGML_USE_CANN)
            // Defined when llama.cpp is compiled with support for offloading model layers to GPU.
            return true;
        #else
            return false;
        #endif
        }
    };

    class Backend
    {
    public:
        Backend(ggml_backend_t backend, int n_layers, bool use_gpu)
            : backend(backend), n_layers(n_layers), use_gpu(use_gpu)
        {

        }

        bool is_cpu(void) const
        {
            return ggml_backend_is_cpu(backend);
        }

        ggml_backend_allocator get_allocator(void)
        {
            if (is_cpu())
            {
                // use host buffers for the CPU backend compute buffer
                return ComputeManager::get_default_allocator_cpu(true, use_gpu);
            }
            else
            {
                return ComputeManager::get_default_allocator(backend);
            }
        }

        void set_tensor_async(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size)
        {
            ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
        }

        void set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size)
        {
            ggml_backend_tensor_set(tensor, data, offset, size);
        }

        void get_tensor_async(struct ggml_tensor * tensor, void * data, size_t offset, size_t size)
        {
            ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        }

        void get_tensor(struct ggml_tensor * tensor, void * data, size_t offset, size_t size)
        {
            ggml_backend_tensor_get(tensor, data, offset, size);
        }

        void synchronize(void)
        {
            ggml_backend_synchronize(backend);
        }

    public:
        ggml_backend_t backend;
        const int n_layers;
        const bool use_gpu;
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
        };

        BackendContext()
        {
        }

        void init(const std::vector<gpu_cfg> &gpu_cfgs, const int n_layers, const int n_threads)
        {
            int n_gpu_layers = 0;
            for (auto &cfg : gpu_cfgs) n_gpu_layers += cfg.n_layers;
            const bool use_gpu = n_gpu_layers > 0;

            backend_cpu = ggml_backend_cpu_init();
            CHATLLM_CHECK(backend_cpu != nullptr) << __func__ << ": failed to initialize CPU backend";
            backends.emplace_back(backend_cpu, n_layers - n_gpu_layers, use_gpu);

            this->n_threads = n_threads;

            if (ComputeManager::is_gpu_offload_supported())
            {
                int dev_cnt = ComputeManager::get_device_count();
                for (int i = 0; i < dev_cnt; i++)
                {
                    size_t total = 0;
                    size_t free = ComputeManager::get_device_free_memory(0, &total);
                    printf("%d: total = %zd, free = %zd\n", i, total, free);
                }

                #if defined(GGML_USE_VULKAN)
                ggml_vk_print_devices_info();
                #endif
            }

        #if defined(GGML_USE_METAL)
            if (use_gpu)
            {
                CHATLLM_CHECK(gpu_cfgs.size() == 1) << __func__ << ": Metal backends number must be 1\n";

                backend_metal = ggml_backend_metal_init();
                CHATLLM_CHECK(backend_metal != nullptr) << __func__ << ": failed to initialize Metal backend\n";
                backends.emplace_back(backend_metal, n_gpu_layers, use_gpu);
            }
        #elif defined(GGML_USE_CUDA) ||  defined(GGML_USE_VULKAN) || defined(GGML_USE_SYCL) || defined(GGML_USE_CANN)
            {
                const int total = ComputeManager::get_device_count();
                for (auto cfg : gpu_cfgs)
                {
                    int device = cfg.id >= 0 ? cfg.id : 0;
                    ggml_backend_t backend = ComputeManager::init_backend_device(device);
                    CHATLLM_CHECK(backend != nullptr) << __func__ << ": failed to initialize backend: #" << device;
                    backends.emplace_back(backend, cfg.n_layers, use_gpu);
                }
            }
        #endif

        #ifdef GGML_USE_BLAS
            #error TODO
            backend_blas = ggml_backend_blas_init();
            if (backend_blas)
                backends.emplace_back(ctx->backend_blas, 0);
        #endif

            host_allocator.alloc_matrix = host_allocator.alloc_others = backends[0].get_allocator();

            int layer_id = 0;
            for (auto &backend : backends)
            {
                for (int i = 0; (i < backend.n_layers) && (layer_id < n_layers); i++, layer_id++)
                {
                    // TODO: matrix and others
                    layer_allocators.emplace_back(backend.get_allocator(), &backend);
                }
            }
        }

        ~BackendContext()
        {
            ggml_backend_sched_free(sched);

            for (auto backend : backends)
                ggml_backend_free(backend.backend);

            ggml_backend_buffer_free(buf_output);
        }
    public:
        std::vector<Backend> backends;
    #ifdef GGML_USE_METAL
        ggml_backend_t backend_metal = nullptr;
    #endif
    #ifdef GGML_USE_BLAS
        ggml_backend_t backend_blas = nullptr;
    #endif
        ggml_backend_t backend_cpu = nullptr;

        // memory buffers used to evaluate the model
        std::vector<uint8_t> buf_compute_meta;
        ggml_backend_sched_t sched = nullptr;

        // host buffer for the model output (logits and embeddings)
        ggml_backend_buffer_t buf_output = nullptr;

        std::vector<LayerBufAllocator> layer_allocators;
        LayerBufAllocator host_allocator;

        int n_threads;
    };
}