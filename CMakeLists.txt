cmake_minimum_required(VERSION 3.12)
project(ChatLLM.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 20)

if (MSVC)
    add_compile_options("/utf-8")
    add_compile_options("/D_CRT_SECURE_NO_WARNINGS")
    add_compile_options("/wd4996")
    add_compile_options("/wd4722")
endif ()

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third_party/ggml)

#if (GGML_CUBLAS)
#    add_compile_definitions(GGML_USE_CUBLAS)
#endif ()

if (GGML_CLBLAST)
    add_compile_definitions(GGML_USE_CLBLAST)
endif ()

if (GGML_PERF)
    add_compile_definitions(GGML_PERF)
endif ()

add_library(chatllm STATIC chat.cpp vectorstore.cpp layers.cpp tokenizer.cpp models.cpp)
target_link_libraries(chatllm PRIVATE ggml)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE chatllm)
