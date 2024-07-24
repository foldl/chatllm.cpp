# ChatLLM.cpp

[中文版](README_zh.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![](./docs/demo.gif)

Inference of a bunch of models from less than 1B to more than 300B, for real-time chatting with [RAG](./docs/rag.md) on your computer (CPU),
pure C++ implementation based on [@ggerganov](https://github.com/ggerganov)'s [ggml](https://github.com/ggerganov/ggml).

| [Supported Models](./docs/models.md) | [Download Quantized Models](./docs/quick_start.md#download-quantized-models) |

**What's New:**

* 2024-07-23: Llama 3.1
* 2024-07-23: Llama-3-Groq with tool calling
* 2024-07-17: Mistral Nemo
* 2024-07-17: SmolLM
* 2024-07-14: [ggml updated](https://github.com/ggerganov/ggml/tree/3e7e5e26f90fecf4f7c2808df7d94454630b219c)
* 2024-07-12: NuminaMath
* 2024-07-11: Meta-AI multi-token prediction models
* 2024-07-05: CodeGeeX4
* 2024-07-04: InternLM 2.5 with tool calling
* 2024-07-03: Phi3 mini (June 2024 Update)
* 2024-07-01: LLM-Compiler
* 2024-06-28: Gemma-2
* 2024-06-23: Index
* 2024-06-17: DeepSeek-Coder-V2 (Instruct & Base)
* 2024-06-15: [Tool calling](./docs/tool_calling.md)
* 2024-06-07: Qwen2
* 2024-06-06: GLM-4
* 2024-06-03: XVERSE
* 2024-06-01: Codestral
* 2024-05-30: MAP-Neo
* 2024-05-29: [ggml](https://github.com/ggerganov/ggml) is forked instead of submodule
* 2024-05-26: Aya-23 from Cohere
* 2024-05-24: DeepSeek-V2 Light
* 2024-05-23: Mistral v0.3
* 2024-05-22: Phi3-medium
* 2024-05-15: StarCoder2
* 2024-05-14: [OpenAI API](./docs/binding.md#openai-compatible-api), CodeGemma Base & Instruct supported
* 2024-05-13: Yi 1.5 Chat models
* 2024-05-08: [Layer shuffling](./docs/fun.md)

## Features

* [x] Accelerated memory-efficient CPU inference with int4/int8 quantization, optimized KV cache and parallel computing;
* [x] Use OOP to address the similarities between different _Transformer_ based models;
* [x] Streaming generation with typewriter effect;
* [x] Continuous chatting (content length is virtually unlimited)

    Two methods are available: _Restart_ and _Shift_. See `--extending` options.

* [x] [Retrieval Augmented Generation](./docs/rag.md) (RAG) 🔥

* [x] [LoRA](./docs/models.md#lora-models);
* [x] Python/JavaScript/C [Bindings](./docs/binding.md), web demo, and more possibilities.

## Quick Start

As simple as `python chatllm.py -i -m :model_id`. [Check it out](./docs/quick_start.md).

## Usage

### Preparation

Clone the ChatLLM.cpp repository into your local machine:

```sh
git clone --recursive https://github.com/foldl/chatllm.cpp.git && cd chatllm.cpp
```

If you forgot the `--recursive` flag when cloning the repository, run the following command in the `chatllm.cpp` folder:

```sh
git submodule update --init --recursive
```

### Quantize Model

**Some quantized models can be downloaded [on demand](./docs/quick_start.md#download-quantized-models).**

Install dependencies of `convert.py`:

```sh
pip install -r requirements.txt
```

Use `convert.py` to transform models into quantized GGML format. For example, to convert the _fp16_ base model to q8_0 (quantized int8) GGML model, run:

```sh
# For models such as ChatLLM-6B, ChatLLM2-6B, InternLM, LlaMA, LlaMA-2, Baichuan-2, etc
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin

# For some models such as CodeLlaMA, model type should be provided by `-a`
# Find `-a ...` option for each model in `docs/models.md`.
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a CodeLlaMA
```

Use `-l` to specify the path of the LoRA model to be merged, such as:

```sh
python3 convert.py -i path/to/model -l path/to/lora/model -o quantized.bin
```

Note: Appropriately, only HF format is supported (with a few exceptions); Format of the generated `.bin` files is different from the one (GGUF) used by `llama.cpp`.

### Build

In order to build this project you have several different options.

- Using `make`:

  Prepare for using `make` on Windows:

  1. Download the latest fortran version of [w64devkit](https://github.com/skeeto/w64devkit/releases).
  2. Extract `w64devkit` on your pc.
  3. Run `w64devkit.exe`, then `cd` to the `chatllm.cpp` folder.

  ```sh
  make
  ```

  The executable is `./obj/main`.

- Using `CMake`:

  ```sh
  cmake -B build
  # On Linux, WSL:
  cmake --build build -j
  # On Windows with MSVC:
  cmake --build build -j --config Release
  ```

  The executable is `./build/obj/main`.

### Run

Now you may chat with a quantized model by running:

```sh
./build/bin/main -m chatglm-ggml.bin                            # ChatGLM-6B
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m llama2.bin  --seed 100                      # Llama-2-Chat-7B
# Hello! I'm here to help you with any questions or concerns ....
```

To run the model in interactive mode, add the `-i` flag. For example:

```sh
# On Windows
.\build\bin\Release\main -m model.bin -i

# On Linux (or WSL)
rlwrap ./build/bin/main -m model.bin -i
```

In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

## Acknowledgements

* This project is started as refactoring of [ChatGLM.cpp](https://github.com/li-plus/chatglm.cpp), without which, this project could not be possible.

* Thank those who have released their the model sources and checkpoints.

## Note

This project is my hobby project to learn DL & GGML, and under active development. PRs of features won't
be accepted, while PRs for bug fixes are warmly welcome.