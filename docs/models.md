# Supported Models

## Chat/Instruct Models

* LlaMA-like (`LlamaForCausalLM`):
    * [x] All LlaMA-1 models
    * [x] LlaMA-2: [Chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), etc
    * [x] LlaMA-3: [Instruct-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Instruct-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), other derivations such as [Llama3-8B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat)
    * [x] CodeLlaMA: [Instruct-7B](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) (`-a CodeLlaMA`)
    * [x] DeepSeek: [Chat-7B](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) (`-a DeepSeek`) , [Coder-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) (`-a DeepSeekCoder`), [Coder-Instruct-1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) (`-a DeepSeekCoder`) 🔥
    * [x] Yi: (`-a Yi`)
        - v1: [Chat-6B](https://huggingface.co/01-ai/Yi-6B-Chat), [Chat-34B](https://huggingface.co/01-ai/Yi-34B-Chat)
        - v1.5: [Chat-6B](https://huggingface.co/01-ai/Yi-1.5-6B-Chat),
        [Chat-9B](https://huggingface.co/01-ai/Yi-1.5-9B-Chat),
        [Chat-34B](https://huggingface.co/01-ai/Yi-1.5-34B-Chat),
        [Chat-9B-16K](https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K),
        [Chat-34B-16K](https://huggingface.co/01-ai/Yi-1.5-34B-Chat-16K)
    * [x] WizardLM: [LM 7B](https://huggingface.co/WizardLM/WizardLM-7B-V1.0) (`-a WizardLM`), [LM 13B](https://huggingface.co/WizardLM/WizardLM-13B-V1.2) (`-a WizardLM`), [Coder Python-7B](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0) (`-a WizardCoder`)
    * [x] TigerBot: [Chat-7B](https://huggingface.co/TigerResearch/tigerbot-7b-chat), [Chat-13B](https://huggingface.co/TigerResearch/tigerbot-13b-chat-v5) (`-a TigerBot`)
    * [x] CodeFuse-DeepSeek: [33B](https://huggingface.co/codefuse-ai/CodeFuse-DeepSeek-33B) (`-a CodeFuseDeepSeek`)

    For other models that using `LlamaForCausalLM` architecture, for example, [aiXcoder-7B](https://huggingface.co/aiXcoder/aixcoder-7b-base), try `-a Yi`.

* Baichuan (`BaichuanForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [Chat-13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

* ChatGLM (`ChatGLMModel`):
    * [x] ChatGLM: [6B](https://huggingface.co/THUDM/chatglm-6b)
    * [x] ChatGLM2 family: [ChatGLM2 6B](https://huggingface.co/THUDM/chatglm2-6b), [CodeGeeX2 6B](https://huggingface.co/THUDM/codegeex2-6b), [ChatGLM3 6B](https://huggingface.co/THUDM/chatglm3-6b)

        Tip on CodeGeeX2: Code completion only, no context. Use system prompt to specify language, e.g. `-s "# language: python"`.

    * [x] CharacterGLM: [6B](https://huggingface.co/thu-coai/CharacterGLM-6B) (`-a CharacterGLM`)

        Note: Use additional key-value pair arguments to specify characters, `--kv user_name "..." bot_name "..." user_info "..." bot_info "..."`.

* InternLM (`InternLMForCausalLM`, `InternLM2ForCausalLM`)
    * [x] v1: [Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), [Chat-7B v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1_1), [Chat-20B](https://huggingface.co/internlm/internlm-chat-20b)
    * [x] v2: [Chat-7B](https://huggingface.co/internlm/internlm2-chat-7b), [Chat-20B](https://huggingface.co/internlm/internlm2-chat-20b)

* Mistral (`MistralForCausalLM`, `MixtralForCausalLM`)
    * [x] Mistral: [Instruct-7B-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Instruct-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

    * [x] OpenChat: [3.5](https://huggingface.co/openchat/openchat-3.5-1210/) (`-a OpenChat`) 🔥

        Tip: Use system prompt to select modes: `-s GPT4` (default mode), `-s Math` (mathematical reasoning mode).

    * [x] Starling: [7B-beta](https://huggingface.co/Nexusflow/Starling-LM-7B-beta) (`-a Starling`)

        Note: This is based on OpenChat, and is fully compatible with OpenChat GPT4 mode.

    * [x] WizardLM: [Math 7B](https://huggingface.co/WizardLM/WizardMath-7B-V1.1) (`-a WizardMath`)

    * [x] Mixtral: [Instruct-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 🔥, [Instruct-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)

        Three implementations of sliding-window attention (see `SlidingWindowAttentionImpl`):

        - Full cache: more RAM is needed.
        - Partial cache: less RAM is needed, and faster than ring cache (**default**).
        - Ring cache (i.e. rolling cache): least RAM, but current implementation is *naive* (slow). 💣

        Note: precision of these implementations differs, which causes different results.

    * [x] NeuralBeagle14: [7B](https://huggingface.co/mlabonne/NeuralBeagle14-7B) (`-a NeuralBeagle`)

    * [x] WizardLM-2: WizardLM-2-8x22B (official link is gone) (`-a WizardLM-2-MoE`)

        Note: For `MixtralForCausalLM` models, `--experts ...` is supported to select a subset of experts when converting.
        For example, `--experts 0,1,2,3` selects the first 4 experts.

* Phi (`PhiForCausalLM`, `Phi3ForCausalLM`)
    * [x] [Phi-2](https://huggingface.co/microsoft/phi-2/tree/eb8bbd1d37d258ea74fb082c53346d33056a83d4)

        Tip: `--temp 0` is recommended. Don't forget to try `--format qa`.

    * [x] [Dolphin Phi-2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2/tree/a084bb141f99f67e8ff56a654e29ddd53a0b4d7a) (`-a DolphinPhi2`) 🐬

    * [x] Phi-3 Mini: [Instruct-4k](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [Instruct-128k](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

    * [x] Phi-3 Medium: [Instruct-4k](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct), [Instruct-128k](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)

* QWen (`QWenLMHeadModel`, `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM`)
    * [x] v1: [Chat-7B](https://huggingface.co/Qwen/Qwen-7B-Chat), [Chat-14B](https://huggingface.co/Qwen/Qwen-14B-Chat), [QAnything-7B](https://huggingface.co/netease-youdao/Qwen-7B-QAnything)
    * [x] v1.5: [Chat-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat), [Chat-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat), [Chat-4B](https://huggingface.co/Qwen/Qwen1.5-4B-Chat), [Chat-7B](https://huggingface.co/Qwen/Qwen1.5-7B-Chat), [Chat-14B](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), [CodeQwen-Chat-7B](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat) (-a `CodeQwen`)
    * [x] v1.5 MoE: [Chat-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)

        Note: At present, to run this MoE model, manual update of `GGML_MAX_SRC` in `ggml.h` to a value >= 62 is required:

        ```
        #define GGML_MAX_SRC            62
        ```

        [Here is why](https://github.com/ggerganov/llama.cpp/pull/6074#issuecomment-2027419314).

* BlueLM (`BlueLMForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/vivo-ai/BlueLM-7B-Chat), [Chat-7B 32K](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-32K)

* Orion (`OrionForCausalLM`)
    * [x] [Chat-14B](https://huggingface.co/OrionStarAI/Orion-14B-Chat)

* MiniCPM (`MiniCPMForCausalLM`)
    * [x] [DPO-2B](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16), [SFT-2B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16),
          [SFT-1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)🔥
    * [x] [2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k) (Note: `--temp 0` is recommended.)
    * [x] [MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B)

* Adept Persimmon (`PersimmonForCausalLM`)
    * [x] [Chat-8B](https://huggingface.co/adept/persimmon-8b-chat)

* Gemma (`GemmaForCausalLM`)
    * [x] v1.0 [Instruct-2B](https://huggingface.co/google/gemma-2b-it), [Instruct-7B](https://huggingface.co/google/gemma-7b-it)
    * [x] v1.1 [Instruct-2B](https://huggingface.co/google/gemma-1.1-2b-it), [Instruct-7B](https://huggingface.co/google/gemma-1.1-7b-it)
    * [x] CodeGemma v1.1: [Instruct-7B](https://huggingface.co/google/codegemma-1.1-7b-it)

* Cohere (`CohereForCausalLM`)
    * [x] [C4AI Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)

* Zhinao (`ZhinaoForCausalLM`)
    * [x] [Chat-7B-4K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-4K), [Chat-7B-32K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-32K), [Chat-7B-360K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-360K)

* DeepSeek (`DeepseekV2ForCausalLM`)
    * [x] [V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)

    Note: To run this model on CPU, some _customization_ on `ggml` is needed:
    - Update `GGML_MAX_SRC` (see above)
    - Fix this [issue](https://github.com/ggerganov/ggml/issues/835)

## Base Models

Please use `--format completion` for these models.

* LlaMA-like (`LlamaForCausalLM`):
    * [x] DeepSeek: [Coder-Base-1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) (`-a DeepSeekCoder`), [Coder-Base-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) (`-a DeepSeekCoder`)

* Mistral (`MistralForCausalLM`, `MixtralForCausalLM`)
    * [x] Mistral: [Base-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Base-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

* Gemma (`GemmaForCausalLM`)
    * [x] CodeGemma v1.1: [Base-2B](https://huggingface.co/google/codegemma-1.1-2b), [Base-7B](https://huggingface.co/google/codegemma-1.1-7b)

* Grok-1
    * [x] [Base](https://huggingface.co/xai-org/grok-1)

        About [Grok-1](./docs/grok.md).

* StarCoder (`Starcoder2ForCausalLM`)
    * [x] [Base-3B](https://huggingface.co/bigcode/starcoder2-7b), [Base-7B](https://huggingface.co/bigcode/starcoder2-7b), [Base-15B](https://huggingface.co/bigcode/starcoder2-15b)

* Stable-LM (`StableLMEpochModel`)
    * [x] [Code-3B](https://huggingface.co/stabilityai/stable-code-3b)

## RAG Models

* Text Embedding (`XLMRobertaModel`)
    * [x] [BCE-Embedding](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
    * [x] [BGE-M3](https://huggingface.co/BAAI/bge-m3) (`-a BGE-M3`)

        Note: Only dense embedding is implemented.

* QA Ranking (`XLMRobertaForSequenceClassification`)
    * [x] [BCE-ReRanker](https://huggingface.co/maidalun1020/bce-reranker-base_v1)
    * [x] [BGE-ReRanker-M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (`-a BGE-Reranker-M3`)


## LoRA Models

These LoRA models have been tested:

* [Llama-3-Chinese-8B-Instruct](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-lora)