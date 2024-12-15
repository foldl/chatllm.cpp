#include <cstring>
#include <set>
#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
extern void ggml_vk_print_devices_info(void);
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#elif defined(GGML_USE_CANN)
#   include "ggml-cann.h"
#endif

#ifdef GGML_USE_BLAS
#  include "ggml-blas.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif

#include "backend.h"

#include "basics.h"

#include "ggml-cpu.h"

namespace chatllm
{
    void *BackendBuffer::get_base(void)
    {
        return ggml_backend_buffer_get_base(buf);
    }

    size_t BackendBuffer::get_size(void) const
    {
        return ggml_backend_buffer_get_size(buf);
    }

    bool BackendBuffer::is_host(void)
    {
        return ggml_backend_buffer_is_host(buf);
    }

    BackendBuffer::~BackendBuffer()
    {
        ggml_backend_buffer_free(buf);
    }

    void BackendBuffer::assign_to(ggml::tensor *tensor, size_t offset)
    {
        uint8_t *data = (uint8_t *)get_base() + offset;
        ggml_backend_tensor_alloc(buf, tensor, data);
    }

    BackendBuffer::BackendBuffer(ggml_backend_buffer_t buf)
        : buf(buf)
    {}

    void BackendBufAllocator::show_info(void)
    {
        printf("%30s allocated buffer size = (%8.2f, %8.2f) MiB\n", ggml_backend_name(backend->backend), total[0] / 1024.0 / 1024.0, total[1] / 1024.0 / 1024.0);
    }

    Backend *BackendBufAllocator::get_backend(void)
    {
        return backend;
    }

    LayerBufAllocator::LayerBufAllocator(): LayerBufAllocator(nullptr, nullptr, nullptr) {}
    LayerBufAllocator::LayerBufAllocator(ggml_backend_allocator alloc, Backend *backend): LayerBufAllocator(alloc, alloc, backend) {}
    LayerBufAllocator::LayerBufAllocator(ggml_backend_allocator alloc_matrix, ggml_backend_allocator alloc_others, Backend *backend)
        : BackendBufAllocator(backend), alloc_matrix(alloc_matrix), alloc_others(alloc_others)
    {
        CHATLLM_CHECK(alloc_matrix == alloc_others) << " TODO: alloc_matrix must be alloc_others now.";
    }

    void LayerBufAllocator::show_info(void)
    {
        BackendBufAllocator::show_info();
        printf("\tMatrix = %s, Others = %s\n", ggml_backend_buft_name(get_allocator(Usage::Matrix)), ggml_backend_buft_name(get_allocator(Usage::Others)));
    }

    BackendBuffer *LayerBufAllocator::alloc(size_t size, Usage usage)
    {
        total[usage] += size;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(get_allocator(usage), size);

        CHATLLM_CHECK(buf) << __FUNCTION__ << "() failed to allocate buffer";

        auto r = new BackendBuffer(buf);
        buffers.emplace_back(r);
        return r;
    }

    bool LayerBufAllocator::alloc(ggml::tensor *tensor)
    {
        BackendBuffer *buf = alloc(ggml::nbytes(tensor), detect_usage(tensor));
        if (nullptr == buf) return false;

        buf->assign_to(tensor);
        return true;
    }

    bool LayerBufAllocator::supported_by_backend(Backend *backend, ggml::tensor *tensor)
    {
        ggml_backend_allocator allocator = get_allocator(tensor); return false;
        return ggml_backend_supports_buft(backend->backend, allocator);
    }

    BackendBufAllocator::Usage LayerBufAllocator::detect_usage(ggml::tensor *tensor)
    {
        int dims = ggml::n_dims(tensor);
        return dims >= 2 ? Usage::Matrix : Usage::Others;
    }

    ggml_backend_allocator LayerBufAllocator::get_allocator(Usage usage)
    {
        switch (usage)
        {
        case Usage::Matrix:
            return alloc_matrix;
        case Usage::Others:
            return alloc_others;
        default:
            CHATLLM_CHECK(false);
            return nullptr;
        }
    }

    ggml_backend_allocator LayerBufAllocator::get_allocator(ggml::tensor *tensor)
    {
        return get_allocator(detect_usage(tensor));
    }

    size_t LayerBufAllocator::get_alignment(Usage usage) const
    {
        switch (usage)
        {
        case Usage::Matrix:
            return ggml_backend_buft_get_alignment(alloc_matrix);
        case Usage::Others:
            return ggml_backend_buft_get_alignment(alloc_others);
        default:
            CHATLLM_CHECK(0);
            return 0;
        }
    }

    size_t LayerBufAllocator::get_max_size(Usage usage) const
    {
        switch (usage)
        {
        case Usage::Matrix:
            return ggml_backend_buft_get_max_size(alloc_matrix);
        case Usage::Others:
            return ggml_backend_buft_get_max_size(alloc_others);
        default:
            CHATLLM_CHECK(0);
            return 0;
        }
    }

    void LayerBufAllocator::free_all_buffers(void)
    {
        memset(&total, 0, sizeof(total));
        buffers.clear();
    }

    bool LayerBufAllocator::operator ==(const LayerBufAllocator &b)
    {
        return (alloc_matrix == b.alloc_matrix)
            && (alloc_others == b.alloc_others)
            && (backend == b.backend);
    }


    void LayerAllocatorManager::set_misc_layer_backend_mapping(int prolog, int epilog)
    {
        prolog_layer_backend_map_to_layer_id = prolog;
        epilog_layer_backend_map_to_layer_id = epilog;
    }

    void LayerAllocatorManager::move_to_layer(int layer_id)
    {
        cur_layer = layer_id;
    }

    int LayerAllocatorManager::get_cur_layer(void) const
    {
        return cur_layer;
    }

    LayerBufAllocator *LayerAllocatorManager::get_allocator(void)
    {
        auto id = get_mapped_layer_id(cur_layer);
        return &allocators[id];
    }

    LayerBufAllocator *LayerAllocatorManager::get_allocator(int layer_id)
    {
        auto id = get_mapped_layer_id(layer_id);
        return &allocators[id];
    }

    LayerBufAllocator *LayerAllocatorManager::get_allocator(ggml::tensor *tensor)
    {
        return alloc_of_tensor[tensor];
    }

    void LayerAllocatorManager::register_tensor_allocator(ggml::tensor *tensor,  LayerBufAllocator *allocator)
    {
        alloc_of_tensor.insert_or_assign(tensor, allocator);
    }

    int LayerAllocatorManager::get_mapped_layer_id(int layer_id)
    {
        int id = layer_id;
        switch (id)
        {
        case MiscLayer::Prolog:
            id = prolog_layer_backend_map_to_layer_id;
            break;
        case MiscLayer::Epilog:
            id = epilog_layer_backend_map_to_layer_id;
            //if (id < 0) id = (int)allocators.size() - 2;
            break;
        default:
            break;
        }
        if ((id < 0) || (id >= (int)allocators.size()))
            id = (int)allocators.size() - 1;

        return id;
    }

    ggml_backend_allocator ComputeManager::get_default_allocator_cpu(bool host_buffer, bool use_gpu)
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

    ggml_backend_allocator ComputeManager::get_default_allocator(ggml_backend_t backend)
    {
        return ggml_backend_get_default_buffer_type(backend);
    }

    int ComputeManager::get_device_count(void)
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

    ggml_backend_t ComputeManager::init_backend_device(int index)
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

    ggml_backend_allocator ComputeManager::get_default_allocator_offload(int gpu)
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

    size_t ComputeManager::get_device_free_memory(int device, size_t *p_total)
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

    int ComputeManager::get_max_devices(void)
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

    bool ComputeManager::is_gpu_offload_supported(void)
    {
    #if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)   || defined(GGML_USE_VULKAN) || \
        defined(GGML_USE_SYCL) || defined(GGML_USE_CANN)
        // Defined when llama.cpp is compiled with support for offloading model layers to GPU.
        return true;
    #else
        return false;
    #endif
    }

    Backend::Backend(ggml_backend_t backend, int n_layers, bool use_gpu)
        : backend(backend), n_layers(n_layers), use_gpu(use_gpu)
    {

    }

    bool Backend::is_cpu(void) const
    {
        return ggml_backend_is_cpu(backend);
    }

    ggml_backend_allocator Backend::get_allocator(BufferType bt)
    {
        if (is_cpu() || (BufferType::Shared == bt) || !use_gpu)
        {
            // use host buffers for the CPU backend compute buffer
            return ComputeManager::get_default_allocator_cpu(true, use_gpu);
        }
        else
        {
            return ComputeManager::get_default_allocator(backend);
        }
    }

    void Backend::write_tensor_data_async(ggml::tensor * tensor, const void * data, size_t offset, size_t size)
    {
        ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
    }

    void Backend::write_tensor_data(ggml::tensor * tensor, const void * data, size_t offset, size_t size)
    {
        ggml_backend_tensor_set(tensor, data, offset, size);
    }

    void Backend::write_tensor_data(ggml::tensor * tensor, const void * data)
    {
        ggml_backend_tensor_set(tensor, data, 0, ggml::nbytes(tensor));
    }

    void Backend::read_tensor_data_async(ggml::tensor * tensor, void * data, size_t offset, size_t size)
    {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
    }

    void Backend::read_tensor_data(ggml::tensor * tensor, void * data, size_t offset, size_t size)
    {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }

    void Backend::read_tensor_data(ggml::tensor * tensor, void * data)
    {
        ggml_backend_tensor_get(tensor, data, 0, ggml::nbytes(tensor));
    }

    void Backend::synchronize(void)
    {
        ggml_backend_synchronize(backend);
    }

    BackendContext::BackendContext()
    {
    }

    void BackendContext::init(const std::vector<gpu_cfg> &gpu_cfgs, const int n_layers, const size_t graph_max_nodes_num)
    {
        int n_gpu_layers = 0;
        for (auto &cfg : gpu_cfgs) n_gpu_layers += cfg.n_layers;
        const bool use_gpu = n_gpu_layers > 0;

        buf_compute_meta.resize(ggml_tensor_overhead() * graph_max_nodes_num + ggml_graph_overhead_custom(graph_max_nodes_num, false));

        if (ComputeManager::is_gpu_offload_supported())
        {
            int dev_cnt = ComputeManager::get_device_count();
            for (int i = 0; i < dev_cnt; i++)
            {
                size_t total = 0;
                size_t free = ComputeManager::get_device_free_memory(0, &total);
                printf("%d: total = %zd, free = %zd\n", i, total, free);
            }
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
        if (use_gpu)
        {
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

        backend_cpu = ggml_backend_cpu_init();
        CHATLLM_CHECK(backend_cpu != nullptr) << __func__ << ": failed to initialize CPU backend";
        backends.emplace_back(backend_cpu, n_layers - n_gpu_layers, use_gpu);

        host_allocator.alloc_matrix = host_allocator.alloc_others = backends[backends.size() - 1].get_allocator(BufferType::Shared);

        int layer_id = 0;
        for (auto &backend : backends)
        {
            for (int i = 0; (i < backend.n_layers) && (layer_id < n_layers); i++, layer_id++)
            {
                // TODO: matrix and others
                layer_allocators.allocators.emplace_back(backend.get_allocator(BufferType::Dedicated), &backend);
            }
        }

        // a "faked" layer for CPU
        layer_allocators.allocators.emplace_back(host_allocator.alloc_matrix, &backends[backends.size() - 1]);

        gg_bufts.clear();
        gg_backends.clear();

        for (auto &b : backends)
        {
            gg_backends.push_back(b.backend);
            gg_bufts.push_back(b.get_allocator(BufferType::Dedicated));
        }
        sched = ggml_backend_sched_new(gg_backends.data(), gg_bufts.data(), (int)gg_backends.size(), graph_max_nodes_num, false);
    }

    BackendContext::~BackendContext()
    {
        ggml_backend_sched_free(sched);

        for (auto backend : backends)
            ggml_backend_free(backend.backend);

        ggml_backend_buffer_free(buf_output);
    }

    bool BackendContext::reserve_memory(ggml_cgraph *gf)
    {
        return ggml_backend_sched_reserve(sched, gf);
    }

    bool BackendContext::alloc_graph(ggml_cgraph *gf)
    {
        return ggml_backend_sched_alloc_graph(sched, gf);
    }

    static bool _backend_sched_eval_callback(ggml::tensor *t, bool ask, void *user_data)
    {
        auto *p = reinterpret_cast<BackendContext *>(user_data);
        if (ask)
            return p->need_observe_tensor_callback(t, p->observe_tensor_callback_data);
        else
            return p->observe_tensor_callback(t, p->observe_tensor_callback_data);
    }

    void BackendContext::compute_graph(ggml_cgraph *gf, int n_threads)
    {
    #ifdef GGML_USE_METAL
        if (ggml_backend_is_metal(backend_metal))
        {
            ggml_backend_metal_set_n_cb(backend_metal, n_threads);
        }
    #endif

        if (backend_cpu != nullptr)
        {
            ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
            ggml_backend_cpu_set_abort_callback(backend_cpu, abort_callback, abort_callback_data);
        }

    #ifdef GGML_USE_BLAS
        if (backend_blas != nullptr)
        {
            ggml_backend_blas_set_n_threads(backend_blas, n_threads);
        }
    #endif

        if (observe_tensor_callback)
            ggml_backend_sched_set_eval_callback(sched, _backend_sched_eval_callback, this);
        else
            ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);

        ggml_backend_sched_graph_compute_async(sched, gf);
        ggml_backend_sched_synchronize(sched);
    }

    void BackendContext::reset()
    {
        ggml_backend_sched_reset(sched);
    }

    void BackendContext::set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data)
    {
        this->abort_callback      = abort_callback;
        this->abort_callback_data = abort_callback_data;
    }

    void BackendContext::set_eval_observe_callback(ggml::need_observe_tensor_evaluation_callback need_observe_tensor_callback,
            ggml::observe_tensor_evaluation_callback observe_tensor_callback, void *user_data)
    {
        this->need_observe_tensor_callback = need_observe_tensor_callback;
        this->observe_tensor_callback      = observe_tensor_callback;
        this->observe_tensor_callback_data = user_data;
    }

    void BackendContext::show_buffer_sizes(void)
    {
        for (size_t i = 0; i < layer_allocators.allocators.size(); i++)
        {
            printf("layer #%d", (int)i);
            layer_allocators.allocators[i].show_info();
        }

        for (size_t i = 0; i < gg_backends.size(); i++)
        {
            ggml_backend_buffer_type_t buft = gg_bufts[i];
            size_t size = ggml_backend_sched_get_buffer_size(sched, gg_backends[i]);
            printf("%s: %30s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft), size / 1024.0 / 1024.0);
        }
    }

    void BackendContext::synchronize(void)
    {
        for (auto &backend : backends)
            backend.synchronize();
    }

    ComputeContext::ComputeContext(BackendContext *backend_context) : backend_context(backend_context)
    {
    }

    ggml_cgraph *ComputeContext::get_cgraph(void)
    {
        return nullptr;
    }

    void ComputeContext::cb_new_tensor(ggml::tensor *tensor)
    {
        if (get_backend())
            ggml_backend_sched_set_tensor_backend(get_sched(), tensor, get_backend()->backend);
        register_tensor_allocator(tensor, backend_context->layer_allocators.get_allocator());
    }

    void ComputeContext::cb_op_tensor(ggml::tensor *tensor)
    {
        if (get_backend() && ggml_backend_supports_op(get_backend()->backend, tensor))
        {
            //struct ggml_backend_buffer *buffer = tensor->buffer;
            //if (buffer && ggml_backend_supports_buft(get_backend()->backend, ggml_backend_buffer_get_type(buffer)))
            ggml_backend_sched_set_tensor_backend(get_sched(), tensor, get_backend()->backend);
        }
    }

    ggml_backend_sched_t ComputeContext::get_sched(void)
    {
        return backend_context->sched;
    }

    void ComputeContext::move_to_layer(int layer_id)
    {
        backend_context->layer_allocators.move_to_layer(layer_id);
    }

    BackendBufAllocator *ComputeContext::get_allocator(void)
    {
        return backend_context->layer_allocators.get_allocator();
    }

    BackendBufAllocator *ComputeContext::get_allocator(ggml::tensor *tensor)
    {
        return backend_context->layer_allocators.get_allocator(tensor);
    }

    void ComputeContext::register_tensor_allocator(ggml::tensor *tensor,  BackendBufAllocator *allocator)
    {
        backend_context->layer_allocators.register_tensor_allocator(tensor, dynamic_cast<LayerBufAllocator *>(allocator));
    }

    Backend *ComputeContext::get_backend(void)
    {
        return dynamic_cast<LayerBufAllocator *>(get_allocator())->get_backend();
    }

    void ComputeContext::compute(int n_threads)
    {
        backend_context->compute_graph(get_cgraph(), n_threads);
    }

    void ComputeContext::synchronize(void)
    {
        ggml_backend_sched_synchronize(get_sched());
    }

    bool ComputeContext::allocate(void)
    {
        return backend_context->alloc_graph(get_cgraph());
    }

    bool ComputeContext::reserve_memory(void)
    {
        return backend_context->reserve_memory(get_cgraph());
    }

    void ComputeContext::reset(void)
    {
        backend_context->reset();
    }

    size_t ComputeContext::get_used_mem(void)
    {
        return ggml_used_mem(get_ctx());
    }

    size_t ComputeContext::get_mem_size(void)
    {
        return ggml_get_mem_size(get_ctx());
    }

    void ComputeContext::set_backend_context(BackendContext *backend_context)
    {
        this->backend_context = backend_context;
    }
}