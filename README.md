# ChatLLM.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![](./images/demo.gif)

Pure C++ implementation of several models for real-time chatting on your computer (CPU),
based on [@ggerganov](https://github.com/ggerganov)'s [ggml](https://github.com/ggerganov/ggml):

* LlaMA-like:
    * [x] All LlaMA-1 models
    * [x] LlaMA-2: [Chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), etc
    * [x] CodeLlaMA: [Instruct-7B](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
    * [x] DeepSeek: [Chat-7B](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat), [Coder-7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
    * [x] Baichuan-2: [Chat-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [Chat-13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
    * [x] Yi: [Chat-6B](https://huggingface.co/01-ai/Yi-6B-Chat)
    * [x] WizardLM: [LM 7B](https://huggingface.co/WizardLM/WizardLM-7B-V1.0), [LM 13B](https://huggingface.co/WizardLM/WizardLM-13B-V1.2), [Coder Python-7B](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0), [Math 7B](https://huggingface.co/WizardLM/WizardMath-7B-V1.1)
    * [x] Mistral: [Instruct-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)

        Note: Sliding-window attention is not implemented yet.

* ChatGLM:
    * [x] ChatGLM: [6B](https://huggingface.co/THUDM/chatglm-6b)
    * [x] ChatGLM2 family: [ChatGLM2 6B](https://huggingface.co/THUDM/chatglm2-6b), [CodeGeeX2 6B](https://huggingface.co/THUDM/codegeex2-6b), [ChatGLM3 6B](https://huggingface.co/THUDM/chatglm3-6b)

        Note on CodeGeeX2: Code completion only, no context. Use system prompt to specify language, e.g. `-s "# language: python"`.

* InternLM
    * [x] [Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), [Chat-7B v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1_1)

* Phi
    * [x] [Phi-2](https://huggingface.co/microsoft/phi-2)

        Note: `--temp 0` is recommended. Don't forget to try `--format qa`.

## Features

* [x] Pure C++ implementation based on [ggml](https://github.com/ggerganov/ggml);
* [x] Accelerated memory-efficient CPU inference with int4/int8 quantization, optimized KV cache and parallel computing;
* [x] Use OOP to address the similarities between different _Transformer_ based models;
* [x] Streaming generation with typewriter effect;
* [x] Continuous chatting (content length is virtually unlimited)

    Two methods are available: _Restart_ and _Shift_. See `--extending` options.

* [ ] LoRA;
* [ ] Python binding, web demo, and more possibilities.

**Preparation**

Clone the ChatLLM.cpp repository into your local machine:

```sh
git clone --recursive https://github.com/foldl/chatllm.cpp.git && cd chatllm.cpp
```

If you forgot the `--recursive` flag when cloning the repository, run the following command in the `chatllm.cpp` folder:

```sh
git submodule update --init --recursive
```

**Quantize Model**

Use `convert.py` to transform models into quantized GGML format. For example, to convert the _fp16_ base model to q8_0 (quantized int8) GGML model, run:

```sh
# DeepSeek LLM Chat models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a DeepSeek

# DeepSeek Coder models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a DeepSeekCoder

# CodeLlaMA models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a CodeLlaMA

# Yi models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a Yi

# WizardCoder models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a WizardCoder

# WizardLM models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a WizardLM

# WizardMath models
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a WizardMath

# For other models, such as ChatLLM-6B, ChatLLM2-6B, InternLM, LlaMA, LlaMA-2, Baichuan-2, etc
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin
```

Note: Only HF format is supported; Format of the generated `.bin` files is different from the one (GGUF) used by `llama.cpp`.

**Build & Run**

Compile the project using CMake:

```sh
cmake -B build
# On Linux, WSL:
cmake --build build -j
# On Windows with MSVC:
cmake --build build -j --config Release
```

Now you may chat with a quantized model by running:

```sh
./build/bin/main -m chatglm-ggml.bin                            # ChatLLM-6B
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m chatglm2-ggml.bin --top_p 0.8 --temp 0.8    # ChatLLM2-6B
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m llama2.bin  --seed 100                      # Llama-2-Chat-7B
# Hello! I'm here to help you with any questions or concerns ....
```

To run the model in interactive mode, add the `-i` flag. For example:

```sh
./build/bin/main -m model.bin -i
```

In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

## Acknowledgements

* This project is started as refactoring of [ChatGLM.cpp](https://github.com/li-plus/chatglm.cpp), without which, this project could not be possible.

* Thank those who have released their the model sources and checkpoints.
