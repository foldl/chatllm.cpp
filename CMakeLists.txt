cmake_minimum_required(VERSION 3.12)
project(ChatLLM.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third_party/ggml)

set(IC_BUILD_TESTS OFF CACHE BOOL "Build isocline without tests")
include_directories(third_party/isocline/include third_party/isocline/src)
add_subdirectory(third_party/isocline)

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
endif ()

if (GGML_PERF)
    add_compile_definitions(GGML_PERF)
endif ()

add_library(chatllm STATIC chat.cpp layers.cpp tokenizer.cpp models.cpp)
target_link_libraries(chatllm PRIVATE ggml)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE chatllm isocline)

# GoogleTest
option(CHATGLM_ENABLE_TESTING "chatllm: enable testing" OFF)
if (CHATGLM_ENABLE_TESTING)
    enable_testing()

    # ref: https://github.com/google/googletest/blob/main/googletest/README.md
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Specify the commit you depend on and update it regularly.
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Now simply link against gtest or gtest_main as needed. Eg
    add_executable(chatllm_test chatllm_test.cpp)
    target_link_libraries(chatllm_test PRIVATE chatllm gtest_main)
    add_test(NAME chatllm_test COMMAND chatllm_test)
endif ()

option(CHATGLM_ENABLE_PYBIND, "chatllm: enable python binding" OFF)
if (CHATGLM_ENABLE_PYBIND)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(chatllm ggml PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    add_subdirectory(third_party/pybind11)
    pybind11_add_module(_C chatllm_pybind.cpp)
    target_link_libraries(_C PRIVATE chatllm)
endif ()

# lint
set(CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/chatllm.h
    ${PROJECT_SOURCE_DIR}/chatllm.cpp
    ${PROJECT_SOURCE_DIR}/chatllm_test.cpp
    ${PROJECT_SOURCE_DIR}/chatllm_pybind.cpp)
set(PY_SOURCES
    ${PROJECT_SOURCE_DIR}/chatllm_cpp/__init__.py
    ${PROJECT_SOURCE_DIR}/convert.py
    ${PROJECT_SOURCE_DIR}/setup.py
    ${PROJECT_SOURCE_DIR}/examples/cli_chat.py
    ${PROJECT_SOURCE_DIR}/examples/web_demo.py
    ${PROJECT_SOURCE_DIR}/tests/test_convert.py
    ${PROJECT_SOURCE_DIR}/tests/test_chatllm_cpp.py)
add_custom_target(lint
    COMMAND clang-format -i ${CPP_SOURCES}
    COMMAND isort ${PY_SOURCES}
    COMMAND black ${PY_SOURCES} --line-length 120)
