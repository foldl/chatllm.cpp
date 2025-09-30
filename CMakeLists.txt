cmake_minimum_required(VERSION 3.12)
project(ChatLLM.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 20)

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/D_CRT_SECURE_NO_WARNINGS>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/wd4996>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/wd4722>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/wd4996>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/wd4722>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/MP>")
endif ()

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

# third-party libraries
include_directories(ggml/include/ggml ggml/src)
add_subdirectory(ggml)

#if (GGML_CUBLAS)
#    add_compile_definitions(GGML_USE_CUBLAS)
#endif ()

if (GGML_CLBLAST)
    add_compile_definitions(GGML_USE_CLBLAST)
endif ()

set(core_files src/backend.cpp
    src/chat.cpp
    src/vectorstore.cpp
    src/layers.cpp
    src/tokenizer.cpp
    src/models.cpp
    src/unicode.cpp
    src/unicode-data.cpp
    src/vision_process.cpp
    src/audio_process.cpp
    models/adept.cpp
    models/allenai.cpp
    models/alphageo.cpp
    models/apertus.cpp
    models/apriel.cpp
    models/aquila.cpp
    models/baichuan.cpp
    models/bailing.cpp
    models/bce.cpp
    models/bge.cpp
    models/bluelm.cpp
    models/chatglm.cpp
    models/characterglm.cpp
    models/codegeex.cpp
    models/codefuse.cpp
    models/codellama.cpp
    models/cohere.cpp
    models/decilm.cpp
    models/deepseek.cpp
    models/dolphinphi2.cpp
    models/ernie.cpp
    models/exaone.cpp
    models/falcon.cpp
    models/gemma.cpp
    models/gigachat.cpp
    models/gpt.cpp
    models/granite.cpp
    models/groq.cpp
    models/grok.cpp
    models/grove.cpp
    models/hermes.cpp
    models/hunyuan.cpp
    models/index.cpp
    models/instella.cpp
    models/internlm.cpp
    models/janus.cpp
    models/jina.cpp
    models/jiutian.cpp
    models/llama.cpp
    models/m_a_p.cpp
    models/megrez.cpp
    models/minicpm.cpp
    models/mistral.cpp
    models/moonshot.cpp
    models/neuralbeagle.cpp
    models/numinamath.cpp
    models/orpheus.cpp
    models/openchat.cpp
    models/orion.cpp
    models/oute.cpp
    models/pangu.cpp
    models/phi.cpp
    models/qwen.cpp
    models/reka.cpp
    models/seed.cpp
    models/siglip.cpp
    models/smol.cpp
    models/solar.cpp
    models/stablelm.cpp
    models/starcoder.cpp
    models/starling.cpp
    models/telechat.cpp
    models/tigerbot.cpp
    models/wizard.cpp
    models/xverse.cpp
    models/yi.cpp
    models/zhinao.cpp
    )

add_library(libchatllm SHARED EXCLUDE_FROM_ALL src/main.cpp ${core_files})
target_link_libraries(libchatllm PRIVATE ggml)
target_compile_definitions(libchatllm PUBLIC CHATLLM_SHARED_LIB)
SET_TARGET_PROPERTIES(libchatllm PROPERTIES PREFIX "")
set_target_properties(libchatllm
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "../bindings"
)

add_executable(main src/main.cpp ${core_files})
target_link_libraries(main PRIVATE ggml)
