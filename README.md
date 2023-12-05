# ChatLLM.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Pure C++ implementation of several models for real-time chatting on your computer (CPU),
based on [@ggerganov](https://github.com/ggerganov)'s [ggml](https://github.com/ggerganov/ggml):

* [x] [ChatLLM-6B](https://github.com/THUDM/ChatLLM-6B)
* [x] [ChatLLM2-6B](https://github.com/THUDM/ChatLLM2-6B)
* [ ] [InternLM]
* [x] LlaMA-like: [Llama-2-Chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat), etc

## Features

* [x] Pure C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp);
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
# For ChatLLM-6B
python3 convert.py -i THUDM/chatllm-6b -t q8_0 -o chatllm-ggml.bin
# For ChatLLM2-6B
python3 convert.py -i THUDM/chatllm2-6b -t q8_0 -o chatllm2-ggml.bin
```

Note: Only HF format is supported; GGML format is different from the one used by `llama.cpp`.

**Build & Run**

Compile the project using CMake:

```sh
cmake -B build
cmake --build build -j
```

Now you may chat with the quantized ChatLLM-6B model by running:

```sh
./build/bin/main -m chatllm-ggml.bin                            # ChatLLM-6B
# 你好👋！我是人工智能助手 ChatLLM-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m chatllm2-ggml.bin --top_p 0.8 --temp 0.8    # ChatLLM2-6B
# 你好👋！我是人工智能助手 ChatLLM2-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m llama2.bin  --seed 100                      # Llama-2-Chat-7B
# Hello! I'm here to help you with any questions or concerns ....
```

To run the model in interactive mode, add the `-i` flag. For example:

```sh
./build/bin/main -m chatllm-ggml.bin -i
```

In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

## Acknowledgements

* This project is started as refactoring of [ChatGLM.cpp](https://github.com/li-plus/chatglm.cpp), without which, this project could not be possible.

* Thank those who have released their the model sources and checkpoints:

    - [@THUDM](https://github.com/THUDM) for the amazing [ChatLLM-6B](https://github.com/THUDM/ChatLLM-6B) and [ChatLLM2-6B](https://github.com/THUDM/ChatLLM2-6B);
    - InternLM
    - LLaMA
