#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    1
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0
#define GGML_RPC_MAX_SERVERS       16

typedef struct ggml_backend_rpc_set_tensor_from_obj_ctx * ggml_backend_rpc_set_tensor_from_obj_ctx_t;
typedef void (*ggml_backend_rpc_set_tensor_from_obj_callback_t)(ggml_backend_rpc_set_tensor_from_obj_ctx_t ctx, void * data_block, size_t size);
typedef void (*ggml_backend_rpc_set_tensor_from_obj_handler_t)(const char * object_id, size_t object_offset, size_t size, ggml_backend_rpc_set_tensor_from_obj_callback_t callback, ggml_backend_rpc_set_tensor_from_obj_ctx_t ctx);

// these API can be get from `ggml_backend_reg_get_proc_address`
typedef void (*ggml_backend_rpc_add_device_t)(const char * endpoint);
typedef void (*ggml_backend_rpc_start_server_t)(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);
typedef void (*ggml_backend_rpc_register_set_tensor_from_obj_handler_t)(ggml_backend_rpc_set_tensor_from_obj_handler_t handler);
typedef void (*ggml_backend_rpc_tensor_from_file_register_file_t)(const char * fid, const char * file_name);

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);

GGML_BACKEND_API ggml_backend_dev_t ggml_backend_rpc_add_device(const char * endpoint);

GGML_BACKEND_API void ggml_backend_rpc_tensor_from_file_register_file(const char * fid, const char * file_name);
GGML_BACKEND_API void ggml_backend_rpc_register_set_tensor_from_obj_handler(ggml_backend_rpc_set_tensor_from_obj_handler_t handler);

#ifdef  __cplusplus
}
#endif
