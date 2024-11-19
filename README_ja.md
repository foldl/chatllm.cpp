# ChatLLM.cpp

[English](README.md) | [中文版](README_zh.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![](./docs/demo.gif)

あなたのコンピュータ（CPU）でリアルタイムチャットを行うための、[RAG](./docs/rag.md) 対応の多くのモデル（1B未満から300B以上）を推論するための純粋なC++実装です。[@ggerganov](https://github.com/ggerganov) の [ggml](https://github.com/ggerganov/ggml) に基づいています。

| [サポートされているモデル](./docs/models.md) | [量子化モデルのダウンロード](./docs/quick_start.md#download-quantized-models) |

**新着情報:**

* 2024-11-01: Granite, [生成ステアリング](./docs/fun.md#generation-steering)
* 2024-09-29: LlaMA 3.2
* 2024-09-22: Qwen 2.5
* 2024-09-13: OLMoE
* 2024-09-11: MiniCPM3
* 2024-07-14: [ggml 更新](https://github.com/ggerganov/ggml/tree/3e7e5e26f90fecf4f7c2808df7d94454630b219c)
* 2024-06-15: [ツール呼び出し](./docs/tool_calling.md)
* 2024-05-29: [ggml](https://github.com/ggerganov/ggml) はサブモジュールではなくフォークされました
* 2024-05-14: [OpenAI API](./docs/binding.md#openai-compatible-api), CodeGemma Base & Instruct 対応
* 2024-05-08: [レイヤーシャッフル](./docs/fun.md#layer-shuffling)

## 特徴

* [x] int4/int8 量子化、最適化された KV キャッシュ、並列計算によるメモリ効率の高い CPU 推論の加速
* [x] OOP を使用して、異なる _Transformer_ ベースのモデル間の類似性に対処
* [x] タイプライター効果を伴うストリーミング生成
* [x] 継続的なチャット（コンテンツの長さは事実上無制限）

    2つの方法が利用可能です：_Restart_ と _Shift_。`--extending` オプションを参照してください。

* [x] [RAG](./docs/rag.md) (Retrieval Augmented Generation) 🔥

* [x] [LoRA](./docs/models.md#lora-models)
* [x] Python/JavaScript/C/Nim [バインディング](./docs/binding.md)、ウェブデモ、その他の可能性

## クイックスタート

`python chatllm.py -i -m :model_id` と入力するだけで簡単に始められます。[詳細はこちら](./docs/quick_start.md)。

## 使用方法

### 準備

ChatLLM.cpp リポジトリをローカルマシンにクローンします：

```sh
git clone --recursive https://github.com/foldl/chatllm.cpp.git && cd chatllm.cpp
```

リポジトリをクローンする際に `--recursive` フラグを忘れた場合は、`chatllm.cpp` フォルダで以下のコマンドを実行します：

```sh
git submodule update --init --recursive
```

### モデルの量子化

**一部の量子化モデルは[オンデマンドでダウンロード](./docs/quick_start.md#download-quantized-models)できます。**

`convert.py` の依存関係をインストールします：

```sh
pip install -r requirements.txt
```

`convert.py` を使用してモデルを量子化された GGML 形式に変換します。例えば、_fp16_ ベースモデルを q8_0（量子化 int8）GGML モデルに変換するには、以下のコマンドを実行します：

```sh
# ChatLLM-6B、ChatLLM2-6B、InternLM、LlaMA、LlaMA-2、Baichuan-2 などのモデルの場合
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin

# CodeLlaMA などの一部のモデルの場合、モデルタイプは `-a` で指定する必要があります
# 各モデルの `-a ...` オプションは `docs/models.md` に記載されています。
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a CodeLlaMA
```

マージする LoRA モデルのパスを指定するには、`-l` を使用します。例えば：

```sh
python3 convert.py -i path/to/model -l path/to/lora/model -o quantized.bin
```

注意：適切に、HF 形式のみがサポートされています（いくつかの例外を除く）。生成された `.bin` ファイルの形式は `llama.cpp` で使用されている GGUF とは異なります。

### ビルド

このプロジェクトをビルドするには、いくつかの異なるオプションがあります。

- `make` を使用：

  Windows で `make` を使用する準備：

  1. 最新バージョンの [w64devkit](https://github.com/skeeto/w64devkit/releases) をダウンロードします。
  2. `w64devkit` を解凍します。
  3. `w64devkit.exe` を実行し、`chatllm.cpp` フォルダに `cd` します。

  ```sh
  make
  ```

  実行ファイルは `./obj/main` です。

- `CMake` を使用してビルド：

  ```sh
  cmake -B build
  # Linux、WSL で：
  cmake --build build -j
  # Windows で MSVC を使用：
  cmake --build build -j --config Release
  ```

  実行ファイルは `./build/obj/main` です。

### 実行

量子化モデルとチャットするには、以下のコマンドを実行します：

```sh
./build/bin/main -m chatglm-ggml.bin                            # ChatGLM-6B
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m llama2.bin  --seed 100                      # Llama-2-Chat-7B
# Hello! I'm here to help you with any questions or concerns ....
```

インタラクティブモードでモデルを実行するには、`-i` フラグを追加します。例えば：

```sh
# Windows で：
.\build\bin\Release\main -m model.bin -i

# Linux（または WSL）で：
rlwrap ./build/bin/main -m model.bin -i
```

インタラクティブモードでは、チャット履歴が次のラウンドの会話のコンテキストとして使用されます。

`./build/bin/main -h` を実行して、他のオプションを探索してください！

## 謝辞

* このプロジェクトは [ChatGLM.cpp](https://github.com/li-plus/chatglm.cpp) のリファクタリングとして始まりました。これがなければ、このプロジェクトは実現しなかったでしょう。

* モデルのソースとチェックポイントを公開してくれた方々に感謝します。

## 注意

このプロジェクトは私の趣味のプロジェクトであり、DL & GGML を学ぶためのもので、現在積極的に開発中です。機能の PR は受け付けませんが、バグ修正の PR は大歓迎です。
