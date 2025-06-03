#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cinttypes>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <cstring>
#include <fstream>

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

#define GGML_RPC_COMPRESS_FLAG  true
#define GGML_RPC_LZ4_ACC        200
#include "lz4.h"

struct compression_header
{
    uint32_t size;
    uint32_t compressed_size;
};

// cross-platform socket
struct socket_t {
    sockfd_t fd;
    socket_t(sockfd_t fd) : fd(fd) {}
    ~socket_t() {
        GGML_PRINT_DEBUG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

#define RPC_PROTO_VERSION       1

// RPC commands
enum rpc_cmd {
    RPC_CMD_HELLO = 0,
    RPC_CMD_ALLOC_BUFFER,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_FROM_OBJ,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_COUNT,
};

struct rpc_msg_hello_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
     int32_t proto_version;
        char description[GGML_MAX_NAME];
};

struct rpc_msg_get_alloc_size_req {
    rpc_tensor tensor;
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_from_obj_req {
    rpc_tensor tensor;
    char       obj_id[GGML_MAX_NAME];
    size_t     obj_offset;
    size_t     offset;
    size_t     size;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_graph_compute_rsp {
    uint8_t result;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    std::string name;
    size_t alignment;
    size_t max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

static std::shared_ptr<socket_t> make_socket(sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<socket_t>(fd);
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

static std::shared_ptr<socket_t> socket_connect(const char * host, int port) {
    struct sockaddr_in addr;
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock_ptr = make_socket(sockfd);
    if (sock_ptr == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (connect(sock_ptr->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return sock_ptr;
}

static std::shared_ptr<socket_t> socket_accept(sockfd_t srv_sockfd) {
    auto client_socket_fd = accept(srv_sockfd, NULL, NULL);
    auto client_socket = make_socket(client_socket_fd);
    if (client_socket == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return client_socket;
}

static std::shared_ptr<socket_t> create_server_socket(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock = make_socket(sockfd);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        fprintf(stderr, "Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        fprintf(stderr, "Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return sock;
}

static bool send_data(sockfd_t sockfd, const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t n = send(sockfd, (const char *)data + bytes_sent, (int)(size - bytes_sent), 0);
        if (n < 0) {
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool recv_data(sockfd_t sockfd, void * data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sockfd, (char *)data + bytes_recv, (int)(size - bytes_recv), 0);
        if (n <= 0) {
            return false;
        }
        bytes_recv += n;
    }
    return true;
}

static bool send_msg(sockfd_t sockfd, const void * msg, size_t msg_size, bool compress = false) {
    std::vector<uint8_t> compressed;
    if (compress) {
        compressed.resize(LZ4_compressBound((int)msg_size) + sizeof(compression_header));
        int s = LZ4_compress_fast((const char *)msg, (char *)compressed.data() + sizeof(compression_header), (int)msg_size, (int)compressed.size(), GGML_RPC_LZ4_ACC);
        compressed.resize(s + sizeof(compression_header));

        compression_header *header = (compression_header *)compressed.data();
        header->compressed_size = s;
        header->size = (uint32_t)msg_size;

        msg = compressed.data();
        msg_size = compressed.size();
    }

    if (!send_data(sockfd, &msg_size, sizeof(msg_size))) {
        return false;
    }
    return send_data(sockfd, msg, msg_size);
}

static bool recv_msg(sockfd_t sockfd, void * msg, size_t msg_size) {
    uint64_t size;
    if (!recv_data(sockfd, &size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return recv_data(sockfd, msg, msg_size);
}

static bool recv_msg(sockfd_t sockfd, std::vector<uint8_t> & input, bool decompress = false) {
    uint64_t size;
    std::vector<uint8_t> compressed;
    std::vector<uint8_t> *p = decompress ? &compressed : &input;
    if (!recv_data(sockfd, &size, sizeof(size))) {
        return false;
    }
    try {
        p->resize(size);
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    if (!recv_data(sockfd, p->data(), size)) {
        return false;
    }

    if (!decompress) {
        return true;
    }

    compression_header *header = (compression_header *)p->data();
    try {
        input.resize(header->size);
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    const int decompress_size = LZ4_decompress_safe((const char *)p->data() + sizeof(compression_header), (char *)input.data(), header->compressed_size, header->size);
    GGML_ASSERT(decompress_size == header->size);

    return true;
}

static bool recv_msg(sockfd_t sockfd, void * msg, size_t msg_size, bool decompress) {
    if (decompress) {
        std::vector<uint8_t> compressed;
        if (!recv_msg(sockfd, compressed, false)) {
            return false;
        }
        GGML_ASSERT(compressed.size() > sizeof(compression_header));

        compression_header *header = (compression_header *)compressed.data();
        GGML_ASSERT(header->size == msg_size);

        const int decompress_size = LZ4_decompress_safe((const char *)compressed.data() + sizeof(compression_header), (char *)msg, header->compressed_size, header->size);
        GGML_ASSERT(header->size == decompress_size);
        return true;
    }
    else {
        return recv_msg(sockfd, msg, msg_size);
    }
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(const std::shared_ptr<socket_t> & sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size,
                         const bool compress_cmd = false,
                         const bool decompress_output = false) {
    uint8_t cmd_byte = cmd;
    if (!send_data(sock->fd, &cmd_byte, sizeof(cmd_byte))) {
        return false;
    }

    if (!send_msg(sock->fd, input, input_size, compress_cmd)) {
        return false;
    }
    // TODO: currently the output_size is always known, do we need support for commands with variable output size?
    // even if we do, we can skip sending output_size from the server for commands with known output size
    if (!recv_msg(sock->fd, output, output_size, decompress_output)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// information about RPC server used by client
struct rpc_client_server_t
{
    std::shared_ptr<socket_t> socket;
    std::string description;
    uint64_t free_mem;
    uint64_t total_mem;
};

static void ggml_backend_rpc_hello(const std::shared_ptr<socket_t> & sock, rpc_msg_hello_rsp &rsp) {
    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, nullptr, 0, &rsp, sizeof(rsp));
    GGML_ASSERT(status);
    GGML_ASSERT(rsp.proto_version == RPC_PROTO_VERSION);
}

static std::shared_ptr<rpc_client_server_t> get_server(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::shared_ptr<rpc_client_server_t>> servers;
    static bool initialized = false;

    auto it = servers.find(endpoint);
    if (it != servers.end()) {
        if (auto server = it->second) {
            return server;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
#ifdef _WIN32
    if (!initialized) {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            return nullptr;
        }
        initialized = true;
    }
#else
    GGML_UNUSED(initialized);
#endif
    auto sock = socket_connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    GGML_PRINT_DEBUG("[%s] connected to %s, sockfd=%d\n", __func__, endpoint.c_str(), sock->fd);

    rpc_msg_hello_rsp hello;
    ggml_backend_rpc_hello(sock, hello);

    auto server = std::make_shared<rpc_client_server_t>(rpc_client_server_t{sock, std::string(hello.description), hello.free_mem, hello.total_mem});
    servers[endpoint] = server;
    return server;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    auto server = get_server(endpoint);
    if (server) {
        return server->socket;
    }
    return nullptr;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    GGML_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx->remote_ptr;
    } else {
        result.buffer = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = (uint32_t)tensor->ne[i];
        result.nb[i] = (uint32_t)tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data = reinterpret_cast<uint64_t>(tensor->data);
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        return status ? GGML_STATUS_SUCCESS : GGML_STATUS_ALLOC_FAILED;
    }
    else {
        return GGML_STATUS_FAILED;
    }
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size(), nullptr, 0, GGML_RPC_COMPRESS_FLAG);
    GGML_ASSERT(status);
}

static void ggml_backend_rpc_set_tensor_from_object(ggml_tensor * tensor, const char * object_id, size_t object_offset, size_t offset, size_t size) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_set_tensor_from_obj_req request = {};
    request.tensor = serialize_tensor(tensor);
    strncpy(request.obj_id, object_id, sizeof(request.obj_id) - 1);
    request.obj_offset  = object_offset;
    request.offset      = offset;
    request.size        = size;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_FROM_OBJ, &request, sizeof(request), nullptr, 0);
    GGML_ASSERT(status);
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size, false, GGML_RPC_COMPRESS_FLAG);
    GGML_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // check if src and dst are on the same server
    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
    if (src_ctx->sock != dst_ctx->sock) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_copy_tensor_req request;
    request.src = serialize_tensor(src);
    request.dst = serialize_tensor(dst);
    rpc_msg_copy_tensor_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    return response.result;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    GGML_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, nullptr, 0, &response, sizeof(response));
    GGML_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, nullptr, 0, &response, sizeof(response));
    GGML_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // See comments in init_tensor.
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request;

        request.tensor = serialize_tensor(tensor);

        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        GGML_ASSERT(status);

        return response.alloc_size;
    } else {
        return ggml_nbytes(tensor);
    }
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = (uint32_t)tensors.size();
    int output_size = sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + sizeof(n_nodes) + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t));
    *out_ntensors = n_tensors;
    rpc_tensor * out_tensors = (rpc_tensor *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    std::vector<uint8_t> input;
    serialize_graph(cgraph, input);
    rpc_msg_graph_compute_rsp response;
    auto sock = get_socket(rpc_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(), &response, sizeof(response), GGML_RPC_COMPRESS_FLAG);
    GGML_ASSERT(status);
    return (enum ggml_status)response.result;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        fprintf(stderr, "Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock);
    size_t max_size = get_max_size(sock);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_rpc_add_device(endpoint),
        /* .context = */ buft_ctx
    };
    buft_map[endpoint] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint) {
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rpc_guid(),
        /* .interface = */ ggml_backend_rpc_interface,
        /* .device    = */ ggml_backend_rpc_add_device(endpoint),
        /* .context   = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total) {
    auto server = get_server(endpoint);
    *free = server ? server->free_mem : 0;
    *total = server ? server->total_mem : 0;
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(ggml_backend_t backend) : backend(backend) {}
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    void alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    void get_alignment(rpc_msg_get_alignment_rsp & response);
    void get_max_size(rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor(const rpc_msg_set_tensor_from_obj_req & request);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);

private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    ggml_backend_t backend;
    std::unordered_set<ggml_backend_buffer_t> buffers;
};

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);

    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ggml_free(ctx);
        return false;
    }

    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backend);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft,tensor);

    ggml_free(ctx);
    return true;
}

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.proto_version = RPC_PROTO_VERSION;
    memset(response.description, 0, sizeof(response.description));
    auto dev = ggml_backend_get_device(backend);
    if (dev) {
        auto description = ggml_backend_dev_description(dev);
        strncpy(response.description, description, sizeof(response.description) - 1);
    }
}

void rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n", __func__, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        GGML_LOG_ERROR("[%s] size: %" PRIu64 " -> failed\n", __func__, request.size);
    }
}

void rpc_server::get_alignment(rpc_msg_get_alignment_rsp & response) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    GGML_PRINT_DEBUG("[%s] alignment: %lu\n", __func__, alignment);
    response.alignment = alignment;
}

void rpc_server::get_max_size(rpc_msg_get_max_size_rsp & response) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    GGML_PRINT_DEBUG("[%s] max_size: %lu\n", __func__, max_size);
    response.max_size = max_size;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);
    return true;
}

static bool rpc_set_tensor_from_object(ggml_tensor * tensor, const rpc_msg_set_tensor_from_obj_req & req);

bool rpc_server::set_tensor(const rpc_msg_set_tensor_from_obj_req & request) {
    const rpc_tensor * in_tensor = &request.tensor;
    uint64_t offset = request.offset;
    const size_t size = request.size;

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    rpc_set_tensor_from_object(tensor, request);
    ggml_free(ctx);
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ggml_free(ctx);
        return false;
    }

    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        GGML_LOG_ERROR("Null buffer for tensor passed to init_tensor function\n");
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ggml_free(ctx);
        return false;
    }

    ggml_free(ctx);
    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ggml_free(ctx);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_PRINT_DEBUG("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        ggml_free(ctx);
        return false;
    }

    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n",
                     __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ggml_free(ctx);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    const rpc_tensor * tensor = tensor_ptrs.at(id);
    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)(input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors));
    GGML_PRINT_DEBUG("[%s] n_nodes: %u, n_tensors: %u\n", __func__, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    response.result = status;
    ggml_free(ctx);
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(ggml_backend_t backend, sockfd_t sockfd, size_t free_mem, size_t total_mem) {
    rpc_server server(backend);
    while (true) {
        uint8_t cmd;
        if (!recv_data(sockfd, &cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            fprintf(stderr, "Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                rpc_msg_hello_rsp response;
                if (!recv_msg(sockfd, nullptr, 0)) {
                    return;
                }
                server.hello(response);
                response.free_mem  = free_mem;
                response.total_mem = total_mem;
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                server.alloc_buffer(request, response);
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                server.get_alloc_size(request, response);
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                if (!recv_msg(sockfd, nullptr, 0)) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                server.get_alignment(response);
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                if (!recv_msg(sockfd, nullptr, 0)) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                server.get_max_size(response);
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sockfd, input, GGML_RPC_COMPRESS_FLAG)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                if (!send_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_FROM_OBJ: {
                rpc_msg_set_tensor_from_obj_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                if (!server.set_tensor(request)) {
                    return;
                }
                if (!send_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sockfd, response.data(), response.size(), GGML_RPC_COMPRESS_FLAG)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sockfd, input, GGML_RPC_COMPRESS_FLAG)) {
                    return;
                }
                rpc_msg_graph_compute_rsp response;
                if (!server.graph_compute(input, response)) {
                    return;
                }
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                fprintf(stderr, "Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }
#ifdef _WIN32
    {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", res);
            return;
        }
    }
#endif
    auto server_socket = create_server_socket(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = socket_accept(server_socket->fd);
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection, free_mem=%zu, total_mem=%zu\n", free_mem, total_mem);
        fflush(stdout);
        rpc_serve_client(backend, client_socket->fd, free_mem, total_mem);
        printf("Client connection closed\n");
        fflush(stdout);
    }
#ifdef _WIN32
    WSACleanup();
#endif
}

// device interface

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    std::string name;
};

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    auto server = get_server(ctx->endpoint.c_str());
    if (server) {
        return server->description.c_str();
    }

    return ctx->name.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), free, total);

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str());

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str());

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static std::unordered_map<std::string, ggml_backend_dev_t> dev_map;
static std::vector<ggml_backend_dev_t> dev_list;

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    return "RPC";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    return dev_list.size();

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index < dev_list.size()) {
        return dev_list[index];
    }
    return nullptr;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_device") == 0) {
        return (void *)ggml_backend_rpc_add_device;
    }
    else if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    else if (std::strcmp(name, "ggml_backend_rpc_register_set_tensor_from_obj_handler") == 0) {
        return (void *)ggml_backend_rpc_register_set_tensor_from_obj_handler;
    }
    else if (std::strcmp(name, "ggml_backend_rpc_tensor_from_file_register_file") == 0) {
        return (void *)ggml_backend_rpc_tensor_from_file_register_file;
    }
    else if (std::strcmp(name, "ggml_backend_tensor_set_from_object") == 0) {
        return (void *)ggml_backend_rpc_set_tensor_from_object;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

ggml_backend_dev_t ggml_backend_rpc_add_device(const char * endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dev_map.find(endpoint) != dev_map.end()) {
        return dev_map[endpoint];
    }

    ggml_backend_rpc_device_context * ctx = new ggml_backend_rpc_device_context {
        /* .endpoint = */ endpoint,
        /* .name     = */ "RPC[" + std::string(endpoint) + "]",
    };

    ggml_backend_dev_t dev = new ggml_backend_device {
        /* .iface   = */ ggml_backend_rpc_device_i,
        /* .reg     = */ ggml_backend_rpc_reg(),
        /* .context = */ ctx,
    };

    dev_map[endpoint] = dev;
    dev_list.push_back(dev);

    return dev;
}

struct rpc_tensor_from_obj_handler {
    ggml_backend_rpc_set_tensor_from_obj_handler_t handler;
    std::unordered_map<std::string, std::string> fid2name;
    rpc_tensor_from_obj_handler(ggml_backend_rpc_set_tensor_from_obj_handler_t handler)
        : handler(handler) {
    }
};

static struct rpc_tensor_from_obj_handler * ggml_backend_rpc_get_tensor_from_obj_handler(void);

struct ggml_backend_rpc_set_tensor_from_obj_ctx {
    ggml_tensor * tensor;
    const rpc_msg_set_tensor_from_obj_req * req;
    size_t written_size;
    bool error;
};

static void rpc_set_tensor_from_obj_write_tensor_cb(ggml_backend_rpc_set_tensor_from_obj_ctx_t ctx, void * data_block, size_t size) {
    ggml_backend_tensor_set(ctx->tensor, data_block, ctx->req->offset + ctx->written_size, size);
    ctx->written_size += size;
}

static bool rpc_set_tensor_from_object(ggml_tensor * tensor, const rpc_msg_set_tensor_from_obj_req & req) {
    ggml_backend_rpc_set_tensor_from_obj_ctx ctx;
    ctx.tensor       = tensor;
    ctx.req          = &req;
    ctx.written_size = 0;
    ctx.error        = false;
    auto obj_handler = ggml_backend_rpc_get_tensor_from_obj_handler();
    if (!obj_handler->handler) {
        return false;
    }

    obj_handler->handler(req.obj_id, req.obj_offset, req.size, rpc_set_tensor_from_obj_write_tensor_cb, &ctx);

    return ctx.error;
}

static void rpc_set_tensor_from_file_handler(const char * object_id, size_t object_offset, size_t size,
    ggml_backend_rpc_set_tensor_from_obj_callback_t callback, ggml_backend_rpc_set_tensor_from_obj_ctx_t ctx)
{
    const size_t BLOCK_SIZE = 1024 * 1024 * 2;
    auto obj_handler = ggml_backend_rpc_get_tensor_from_obj_handler();
    auto filename = obj_handler->fid2name.find(object_id);
    GGML_ASSERT(filename != obj_handler->fid2name.end());

    std::ifstream file(filename->second, std::ios::binary);
    GGML_ASSERT(file);

    // Seek to the specified offset
    file.seekg(object_offset, std::ios::beg);

    std::vector<uint8_t> buffer;
    buffer.resize(BLOCK_SIZE);

    while (size > 0) {
        size_t block_size = size > BLOCK_SIZE ? BLOCK_SIZE : size;
        file.read((char *)buffer.data(), block_size);
        GGML_ASSERT(file);

        callback(ctx, buffer.data(), block_size);
        size -= block_size;
    }

    // Close the file
    file.close();
}

static struct rpc_tensor_from_obj_handler * ggml_backend_rpc_get_tensor_from_obj_handler(void) {
    static rpc_tensor_from_obj_handler obj_handler(rpc_set_tensor_from_file_handler);
    return &obj_handler;
}

void ggml_backend_rpc_tensor_from_file_register_file(const char * fid, const char * file_name) {
    auto obj_handler = ggml_backend_rpc_get_tensor_from_obj_handler();
    obj_handler->fid2name.emplace(std::string(fid), std::string(file_name));
}

void ggml_backend_rpc_register_set_tensor_from_obj_handler(ggml_backend_rpc_set_tensor_from_obj_handler_t handler) {
    auto obj_handler = ggml_backend_rpc_get_tensor_from_obj_handler();
    obj_handler->handler = handler;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
