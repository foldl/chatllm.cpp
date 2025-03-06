message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                        )

target_sources(ggml-rpc PRIVATE lz4.c)

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()
