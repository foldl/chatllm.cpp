# Supported Models

## Chat/Instruct Models

* Adept Persimmon (`PersimmonForCausalLM`)
    * [x] [Chat-8B](https://huggingface.co/adept/persimmon-8b-chat)

* Apriel (`AprielForCausalLM`)
    * [x] [Instruct-5B](https://huggingface.co/ServiceNow-AI/Apriel-5B-Instruct/tree/a9a4831718a2fad437f25ace0d0259953fcaaa26)

* Aquila (`AquilaForCausalLM`)
    * [x] [Chat2-7B](https://huggingface.co/BAAI/AquilaChat2-7B/tree/9905960de19ea9e573c0dc3fbdf54d4ddcc610d3), [Chat2-34B](https://huggingface.co/BAAI/AquilaChat2-34B/commit/5c7990b198c94b63dfbfa022462b9cf672dbcfa0), [Chat2-7B-16K](https://huggingface.co/BAAI/AquilaChat2-7B-16K/commit/fb46d48479d05086ccf6952f19018322fcbb54cd), [Chat2-34B-16K](https://huggingface.co/BAAI/AquilaChat2-34B-16K/tree/9f19774f3e7afad2fc3d51fe308eac5a2d88c8b1)

* Baichuan (`BaichuanForCausalLM`, `BaichuanM1ForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [Chat-13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
    * [x] M1: [Instruct-14B](https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct)
    * [x] Fine-tunings: [Med-R1](https://modelscope.cn/models/wangrongsheng/Med-R1/files) (Tip: `--set chat_template im`)

* BlueLM (`BlueLMForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/vivo-ai/BlueLM-7B-Chat), [Chat-7B 32K](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-32K)

* ChatGLM (`ChatGLMModel`, `Glm4ForCausalLM`):
    * [x] ~~ChatGLM: [6B](https://huggingface.co/THUDM/chatglm-6b)~~
    * [x] ChatGLM2 family: [ChatGLM2 6B](https://huggingface.co/THUDM/chatglm2-6b), [CodeGeeX2 6B](https://huggingface.co/THUDM/codegeex2-6b), [ChatGLM3 6B](https://huggingface.co/THUDM/chatglm3-6b)

        Tip on CodeGeeX2: Code completion only, no context. Use system prompt to specify language, e.g. `-s "# language: python"`.

    * [x] CharacterGLM: [6B](https://huggingface.co/thu-coai/CharacterGLM-6B) (`-a CharacterGLM`)

        Note: Use additional key-value pair arguments to specify characters, `--kv user_name "..." bot_name "..." user_info "..." bot_info "..."`.

    * [x] GLM-4: [Chat-9B-128k](https://huggingface.co/THUDM/glm-4-9b-chat), [Chat-9B-1M](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
    * [x] CodeGeeX4: [9B](https://huggingface.co/THUDM/codegeex4-all-9b) (`-a CodeGeeX4`)
    * [x] GLM-4-0414: [](https://huggingface.co/THUDM/GLM-4-9B-0414/tree/645b8482494e31b6b752272bf7f7f273ef0f3caf),
    [GLM-Z1-9B-0414](https://huggingface.co/THUDM/GLM-Z1-9B-0414/commit/66907f8ddffdcdcafb7e8c9e96b7366449176c1a),
    [GLM-4-32B-0414](https://huggingface.co/THUDM/GLM-4-32B-0414/tree/dc842812afac55db3a8aca4484ca6f3c60bd472f),
    [GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414/tree/93593a027f03a597582d1db7e84894a0a8fa027e),
    [GLM-Z1-Rumination-32B-0414](https://huggingface.co/THUDM/GLM-Z1-Rumination-32B-0414/tree/6ae9ac6152a7d85761409d6d6ef201d9e8e8aeb1)

* Cohere (`CohereForCausalLM`)
    * [x] [C4AI Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
    * [x] [Aya-23-8B](https://huggingface.co/CohereForAI/aya-23-8B), [Aya-23-35B](https://huggingface.co/CohereForAI/aya-23-35B) (`-a Aya-23`, fully compatible with Command-R)
    * [x] [C4AI Command R7B](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024)

* DeciLM (`DeciLMForCausalLM`)
    * Nemotron: [Llama-3.3-Nemotron-Super-49B-v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/tree/1a2cb800fde0e2c100cf576215d7be3a5bffe717)

* DeepSeek (`DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`)
    * [x] v1: [Chat-16B](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/tree/eefd8ac7e8dc90e095129fe1a537d5e236b2e57c)

    * [x] v2: [Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat) (💣 not tested), [Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)

    * [x] Coder v2: [Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) (💣 not tested), [Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)

    * [x] Moonlight: [Instruct-16B](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) (`-a Moonlight`)

    * [x] GigaChat: [Instruct-20B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct/tree/5105af38a6a174b06a2bc25719c5ad5ce680a207) (`-a GigaChat`)

    Two optimization modes are defined: speed (default) and memory. See `BaseMLAttention`.

* EXAONE (`ExaoneForCausalLM`)
    * [x] v3.5: [Instruct-2.4B](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct), [Instruct-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct), [Instruct-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct)
    * [x] Deep: [2.4B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B/tree/b9e0d963cc9be39abce33381f40a8da4324cf4bb), [7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B/tree/19948cbbd0e9afb0f7b5a918eb7e2eb18341e076), [32B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-32B/tree/dfa797fc8d8ae6ecc0e5f7a450317cc1433b2545)

* Gemma (`GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`)
    * [x] v1.0: [Instruct-2B](https://huggingface.co/google/gemma-2b-it), [Instruct-7B](https://huggingface.co/google/gemma-7b-it)
    * [x] v1.1: [Instruct-2B](https://huggingface.co/google/gemma-1.1-2b-it), [Instruct-7B](https://huggingface.co/google/gemma-1.1-7b-it)
    * [x] CodeGemma v1.1: [Instruct-7B](https://huggingface.co/google/codegemma-1.1-7b-it)
    * [x] v2: [Instruct-2B](https://huggingface.co/google/gemma-2-2b-it), [Instruct-9B](https://huggingface.co/google/gemma-2-9b-it), [Instruct-27B](https://huggingface.co/google/gemma-2-27b-it)
    * [x] v3: [Instruct-1B](https://huggingface.co/google/gemma-3-1b-it/tree/9b99be88fdd7a2496bf644baade44348ad736c95)

    Note: Only download `tokenizer.model` and DO NOT download `tokenizer.json` when converting.

* Granite (`GraniteForCausalLM`, `GraniteMoeForCausalLM`)
    * [x] v3.0: [Instruct-1B-A400M](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct), [Instruct-3B-A800M](https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct)
    * [x] v3.1: [Instruct-1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct), [Instruct-3B-A800M](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
    * [x] v3.2: [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct/tree/0276d996f60d5eb0b376b6d06622042d4ef3eb4b)

* HunYuan (`HunYuanForCausalLM`)
    * [x] Dense: [Instruct-7B](https://huggingface.co/tencent/Hunyuan-7B-Instruct)

* Instella (`InstellaForCausalLM`)
    * [x] [Instruct-3B](https://huggingface.co/amd/Instella-3B-Instruct)

* InternLM (`InternLMForCausalLM`, `InternLM2ForCausalLM`)
    * [x] v1: [Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), [Chat-7B v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1_1), [Chat-20B](https://huggingface.co/internlm/internlm-chat-20b)
    * [x] v2: [Chat-1.8B](https://huggingface.co/internlm/internlm2-chat-1_8b), [Chat-7B](https://huggingface.co/internlm/internlm2-chat-7b), [Chat-20B](https://huggingface.co/internlm/internlm2-chat-20b), [Math-Plus-1.8B](https://huggingface.co/internlm/internlm2-math-plus-1_8b), [Math-Plus-7B](https://huggingface.co/internlm/internlm2-math-plus-7), [Math-Plus-20](https://huggingface.co/internlm/internlm2-math-plus-20b)
    * [x] v2.5: [Chat-1.8B](https://huggingface.co/internlm/internlm2_5-1_8b-chat), [Chat-7B](https://huggingface.co/internlm/internlm2_5-7b-chat), [Chat-7B-1M](https://huggingface.co/internlm/internlm2_5-7b-chat-1m), [Chat-20B](https://huggingface.co/internlm/internlm2_5-20b-chat)
    * [x] v3: [Instruct-8B](https://huggingface.co/internlm/internlm3-8b-instruct)

* Ling (`BailingMoeForCausalLM`)
    * [x] [Lite](https://huggingface.co/inclusionAI/Ling-lite/tree/a80ae6c479251f1ae33dda517ab83cdc6a312f99), [Coder-Lite](https://huggingface.co/inclusionAI/Ling-Coder-lite/tree/4a8647acf9d3855d599adaaaf4bf6ca14239d2ab)

* LlaMA-like (`LlamaForCausalLM`, `Llama4ForConditionalGeneration`):
    * [x] All LlaMA-1 models
    * [x] LlaMA-2: [Chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), etc
    * [x] LlaMA-3: [Instruct-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Instruct-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), other derivations such as [Llama3-8B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat)
    * [x] LlaMA-3.1: [Instruct-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), [Instruct-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
    * [x] LlaMA-3.2: [Instruct-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [Instruct-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
    * [x] CodeLlaMA: [Instruct-7B](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) (`-a CodeLlaMA`)
    * [x] LLM-Compiler: [7B](https://huggingface.co/facebook/llm-compiler-7b), [7B-FTD](https://huggingface.co/facebook/llm-compiler-7b-ftd), [13B](https://huggingface.co/facebook/llm-compiler-13b), [13B-FTD](https://huggingface.co/facebook/llm-compiler-13b-ftd)
    * [x] DeepSeek: [Chat-7B](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) (`-a DeepSeek`) , [Coder-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) (`-a DeepSeekCoder`), [Coder-Instruct-1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) (`-a DeepSeekCoder`) 🔥
    * [x] Yi: (`-a Yi`)
        - v1: [Chat-6B](https://huggingface.co/01-ai/Yi-6B-Chat), [Chat-34B](https://huggingface.co/01-ai/Yi-34B-Chat)
        - v1.5: [Chat-6B](https://huggingface.co/01-ai/Yi-1.5-6B-Chat),
        [Chat-9B](https://huggingface.co/01-ai/Yi-1.5-9B-Chat),
        [Chat-34B](https://huggingface.co/01-ai/Yi-1.5-34B-Chat),
        [Chat-9B-16K](https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K),
        [Chat-34B-16K](https://huggingface.co/01-ai/Yi-1.5-34B-Chat-16K)
        - Coder: [Chat-1.5B](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat),
        [Chat-9B](https://huggingface.co/01-ai/Yi-Coder-9B-Chat)
    * [x] WizardLM: [LM 7B](https://huggingface.co/WizardLM/WizardLM-7B-V1.0) (`-a WizardLM`), [LM 13B](https://huggingface.co/WizardLM/WizardLM-13B-V1.2) (`-a WizardLM`), [Coder Python-7B](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0) (`-a WizardCoder`)
    * [x] TigerBot: [Chat-7B](https://huggingface.co/TigerResearch/tigerbot-7b-chat), [Chat-13B](https://huggingface.co/TigerResearch/tigerbot-13b-chat-v5) (`-a TigerBot`)
    * [x] CodeFuse-DeepSeek: [33B](https://huggingface.co/codefuse-ai/CodeFuse-DeepSeek-33B) (`-a CodeFuseDeepSeek`)
    * [x] MAP-Neo: [Instruct-7B](https://huggingface.co/m-a-p/neo_7b_instruct_v0.1) (`-a MAP-Neo`)
    * [x] Index: [Chat-1.9B](https://huggingface.co/IndexTeam/Index-1.9B-Chat), [Character-1.9B](https://huggingface.co/IndexTeam/Index-1.9B-Character), [Chat-1.9B-32K](https://huggingface.co/IndexTeam/Index-1.9B-32K)
    * [x] NuminaMath: [7B-TIR](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
    * [x] SmolLM: (`-a SmolLM`)
        - v1: [Instruct-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)
        - v2: [Instruct-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
    * [x] Groq: [Llama-3-Groq-8B-Tool-Use](https://huggingface.co/Groq/Llama-3-Groq-8B-Tool-Use) (`-a Llama-3-Groq-8B-Tool-Use`)
    * [x] Megrez: [Instruct-3B](https://huggingface.co/Infinigence/Megrez-3B-Instruct) (`-a Megrex`)
    * [x] Falcon: (`-a Falcon3`)
        - v3: [Instruct-1B](https://huggingface.co/tiiuae/Falcon3-1B-Instruct), [Instruct-3B](https://huggingface.co/tiiuae/Falcon3-3B-Instruct), [Instruct-7B](https://huggingface.co/tiiuae/Falcon3-7B-Instruct), [Instruct-10B](https://huggingface.co/tiiuae/Falcon3-10B-Instruct)
    * [x] DeepSeek-R1-Distill-LlaMA: [8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), [70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) (`-a DeepSeek-R1-Distill-LlaMA`)
    * [x] DeepHermes-3: [Llama-3-8B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) (Use `-s ...` to enable thinking)
    * [x] Watt-tool: [8B](https://huggingface.co/watt-ai/watt-tool-8B), [70B](https://huggingface.co/watt-ai/watt-tool-70B)
    * [x] Reke-Flash: [Flash-3](https://huggingface.co/RekaAI/reka-flash-3/tree/69cea64942e4db4809b757ae2b0d312b4b610263) (`-a Reka-Flash-3`)
    * [x] Nemotron: [Llama-3.1-Nemotron-Nano-8B](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/tree/42f62a403ee352e019834442673256e3fe3de275)
    * [x] LlaMA-4: [Scout-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/tree/7dab2f5f854fe665b6b2f1eccbd3c48e5f627ad8), [Maverick-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/tree/f0ee6477b90b7a6aaefd2cfdf1b4a05d36184137)
    * [x] Seed-Coder: [Instruct-8B](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct/tree/7b934eb8f2ce8f40191fa26d12236eb8bc3a77aa),
    [Reasoning-8B](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Reasoning/tree/47f18f877ea7d6aa5c0d87a474f30d420b22bd98) (`--name Seed-Coder`)

    For other models that using `LlamaForCausalLM` architecture, for example, [aiXcoder-7B](https://huggingface.co/aiXcoder/aixcoder-7b-base), try `-a Yi`.

* MiniCPM (`MiniCPMForCausalLM`, `MiniCPM3ForCausalLM`)
    * [x] [DPO-2B](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16), [SFT-2B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16),
          [SFT-1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)🔥
    * [x] [2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k) (Note: `--temp 0` is recommended.)
    * [x] [MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B)
    * [x] v3: [4B](https://huggingface.co/openbmb/MiniCPM3-4B)
    * [x] v4: [0.5B](https://huggingface.co/openbmb/BitCPM4-0.5B/tree/fcad2c603edb0663a36e56999016cbf2d7644ea1), [8B](https://huggingface.co/openbmb/MiniCPM4-8B/tree/cd838a273dde346b7c319d443f41ecd31a71f1b6), [8B-Survey](https://huggingface.co/openbmb/MiniCPM4-Survey/tree/f3e7ca37096dbedbdd48f6bacb29513b64e78667), [8B-MCP](https://huggingface.co/openbmb/MiniCPM4-MCP/commit/4a6cefeea3115ca8fc6b03e1879e912718ba6487)

* Mistral (`MistralForCausalLM`, `MixtralForCausalLM`)
    * [x] Mistral: [Instruct-7B-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Instruct-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
    * [x] Small: [Instruct-24B](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

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

    * [x] Codestral: [22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1)
    * [x] Mistral-Nemo: [Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
    * [x] Small: [Instruct-24B](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
    * [x] DeepHermes-3-Mistral: [24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview/tree/48072dc6c0594a3198eb862c13613c4ab1119009) (`-a DeepHermes-3-Mistral`. Default: Thinking model.)
    * [x] Small-3.1: [Instruct-24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/9d6393981603defaeab8131cf223412c4fdcb99a)

        Note: Please download `config.json` & `tokenizer.json` from [here](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/tree/20b2ed1c4e9af44b9ad125f79f713301e27737e2).

    * [x] Devstral: [Small-2505](https://huggingface.co/mistralai/Devstral-Small-2505/tree/a6e97eaf3bfe308cb5396675a716147b2ced37c8)

* Olm (`OlmoeForCausalLM`, `Olmo2ForCausalLM`)
    * [x] OLMoE: [Instruct-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)
    * [x] OLM-2: [Instruct-7B](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct), [Instruct-13B](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct), [Instruct-32B](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct/tree/5942a2f5e0bc38c2a5f5200cec2ea236d5984547)

* Orion (`OrionForCausalLM`)
    * [x] [Chat-14B](https://huggingface.co/OrionStarAI/Orion-14B-Chat)

* Phi (`PhiForCausalLM`, `Phi3ForCausalLM`)
    * [x] [Phi-2](https://huggingface.co/microsoft/phi-2/tree/eb8bbd1d37d258ea74fb082c53346d33056a83d4)

        Tip: `--temp 0` is recommended. Don't forget to try `--format qa`.

    * [x] [Dolphin Phi-2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2/tree/a084bb141f99f67e8ff56a654e29ddd53a0b4d7a) (`-a DolphinPhi2`) 🐬

    * [x] Phi-3: [Mini-Instruct-4k](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [Mini-Instruct-128k](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct), [Medium-Instruct-4k](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct), [Medium-Instruct-128k](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)

    * [x] Phi-3.5: [Mini-Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), [MoE-Instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

    * [x] Phi-4: [Instruct](https://huggingface.co/microsoft/phi-4), [Mini-Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

* QWen (`QWenLMHeadModel`, `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM`)
    * [x] v1: [Chat-7B](https://huggingface.co/Qwen/Qwen-7B-Chat), [Chat-14B](https://huggingface.co/Qwen/Qwen-14B-Chat), [QAnything-7B](https://huggingface.co/netease-youdao/Qwen-7B-QAnything)
    * [x] v1.5: [Chat-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat), [Chat-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat), [Chat-4B](https://huggingface.co/Qwen/Qwen1.5-4B-Chat), [Chat-7B](https://huggingface.co/Qwen/Qwen1.5-7B-Chat), [Chat-14B](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), [CodeQwen-Chat-7B](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat) (`-a CodeQwen`)
    * [x] v1.5 MoE: [Chat-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)
    * [x] v2: [Instruct-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct), [Instruct-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), [Instruct-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct), [Instruct-72B](https://huggingface.co/Qwen/Qwen2-72B-Instruct)
    * [x] v2 MoE: [Instruct-57B-A14B](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct) (💣 not tested)
    * [x] v2.5: [Instruct-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Instruct-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Instruct-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Instruct-14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [Instruct-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), [Instruct-72B](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
    * [x] v2.5-Coder: [Instruct-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), [Instruct-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
    * [x] v2.5-Math: [Instruct-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct), [Instruct-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct), [Instruct-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct)
    * [x] [Marco-o1](https://huggingface.co/AIDC-AI/Marco-o1) (`-a Marco-o1`)
    * [x] QwQ: [32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview), [32B](https://huggingface.co/Qwen/QwQ-32B) (`-a QwQ`)
    * [x] [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2) (`-a ReaderLM-v2`)
    * [x] DeepSeek-R1-Distill-QWen: [1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), [32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B), [DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) (`-a DeepSeek-R1-Distill-QWen`)
    * [x] DeepSeek-R1-0528-Qwen3: [8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/tree/6e8885a6ff5c1dc5201574c8fd700323f23c25fa)  (`-a DeepSeek-R1-Distill-QWen3`)
    * [x] Skywork-OR1: [Math-7B](https://huggingface.co/Skywork/Skywork-OR1-Math-7B/tree/33731701b9dc7d5510effd440d5b1518be393f14), [7B-Preview](https://huggingface.co/Skywork/Skywork-OR1-7B-Preview/tree/23ba8a3362d4c679fccbb506a51365ce191398c2), [32B-Preview](https://huggingface.co/Skywork/Skywork-OR1-32B-Preview/tree/ba6cba290ca145dc6a13bd806f0a141e4b63bd4a)
    * [x] OlympicCoder: [7B](https://huggingface.co/open-r1/OlympicCoder-7B/tree/097f57223ba8fee3130ce7f739dabae3dd0ad0b9), [32B](https://huggingface.co/open-r1/OlympicCoder-32B/tree/34113aee9d255591a1fa75b60d1e3422e82c3b1f)
    * [x] v3: [235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/04127485f4b84439c34689fdee54a616531bf00d) (💣 not tested),
        [30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/4c446470ba0aec43e22ac1128f9ffd915f338ba3),
        [32B](https://huggingface.co/Qwen/Qwen3-32B/tree/30b8421510892303dc5ddd6cd0ac90ca2053478d),
        [14B](https://huggingface.co/Qwen/Qwen3-14B/tree/231c69a380487f6c0e52d02dcf0d5456d1918201),
        [8B](https://huggingface.co/Qwen/Qwen3-8B/tree/2069b3fae1114555f3c020c81410e51fa0f656f2),
        [4B](https://huggingface.co/Qwen/Qwen3-4B/tree/82d62bb073771e7a1ea59435f548908540217d1f),
        [1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/d3e258980a49b060055ea9038dad99d75923f7c4),
        [0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/6130ef31402718485ca4d80a6234f70d9a4cf362)
    * [x] MiMo: [7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL/tree/8c8a4968100fd7a4f0319c54438184bbeb767c82)

* Solor (`SolarForCausalLM`)
    * [x] [Pro](https://huggingface.co/upstage/solar-pro-preview-instruct/tree/dd4bcf7006df9b1ce3f87711e702e4063832aae3)

* TeleChat (`TeleChat2ForCausalLM`)
    * [x] v2: [3B](https://huggingface.co/Tele-AI/TeleChat2-3B), [7B](https://huggingface.co/Tele-AI/TeleChat2-7B), [115B](https://huggingface.co/Tele-AI/TeleChat2-115B)

* XVERSE (`XverseForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/xverse/XVERSE-7B-Chat), [Chat-13B](https://huggingface.co/xverse/XVERSE-13B-Chat), [Chat-65B](https://huggingface.co/xverse/XVERSE-65B-Chat)

    Note: Tokenizer's behavior is not 100% identical.

* Zhinao (`ZhinaoForCausalLM`)
    * [x] [Chat-7B-4K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-4K), [Chat-7B-32K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-32K), [Chat-7B-360K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-360K)

## Text Base Models

Please use `--format completion` for these models.

* AlphaGeometry-LM (`-a AlphaGeometry-LM`)
    * [x] [geometry.757](https://bit.ly/alphageometry)

* DeepSeek (`DeepseekV2ForCausalLM`)
    * [x] [Coder-V2-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base) (💣 not tested), [Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)

* Gemma (`GemmaForCausalLM`)
    * [x] CodeGemma v1.1: [Base-2B](https://huggingface.co/google/codegemma-1.1-2b), [Base-7B](https://huggingface.co/google/codegemma-1.1-7b)

* Grok-1
    * [x] [Base](https://huggingface.co/xai-org/grok-1)

        About [Grok-1](grok.md).

* LlaMA-like (`LlamaForCausalLM`):
    * [x] DeepSeek: [Coder-Base-1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) (`-a DeepSeekCoder`), [Coder-Base-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) (`-a DeepSeekCoder`)
    * [x] Seed-Coder: [Base-8B](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base/tree/44d3e28414b052f1fc8f9b58740edd2d8f2c2c31) (`--name Seed-Coder`)

* Mistral (`MistralForCausalLM`, `MixtralForCausalLM`)
    * [x] Mistral: [Base-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Base-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

* Stable-LM (`StableLMEpochModel`)
    * [x] [Code-3B](https://huggingface.co/stabilityai/stable-code-3b)

* StarCoder (`Starcoder2ForCausalLM`)
    * [x] [Base-3B](https://huggingface.co/bigcode/starcoder2-7b), [Base-7B](https://huggingface.co/bigcode/starcoder2-7b), [Base-15B](https://huggingface.co/bigcode/starcoder2-15b)

## TTS Models

* Orpheus TTS
    * [x] 3B: [EN](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft/tree/4206a56e5a68cf6cf96900a8a78acd3370c02eb6), [ZH](https://huggingface.co/canopylabs/3b-zh-ft-research_release/commit/29d016d6d0e5a2688267d3b3e432b7e23f043876), etc

        [SNAC-24kHz](https://huggingface.co/mlx-community/snac_24khz/tree/556af1cd3b1c5f2d294f6aa9bb886245d7b716ac) is used as codec.
        Use these additional command line options when converting: `--name Orpheus-TTS -a Orpheus-TTS --snac_model /path/to/snac_24kHz`

        Use `--set voice XX` to select voice `XX`, such as `tara`. [More info](https://github.com/canopyai/Orpheus-TTS?tab=readme-ov-file#prompting).

* OuteTTS:
    * [x] 1.0: [1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B/commit/911e296ce01148a01f3af9329163b0d298ac33a1), [0.6B](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B/tree/e7bcd87b0ca47fd8c46317c8f745a5e4e19c7b5c)

        [DAC.speech.v1.0 1.5kbps](https://huggingface.co/ibm-research/DAC.speech.v1.0/commits/main) is used as codec.
        Use these additional command line options when converting: `--name OuteTTS -a OuteTTS --dac_model /path/to/dac`

        Use `--set speaker /path/to/speaker.json` to select a speaker profile. [More info](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md#creating-custom-speaker-profiles).

## Multimodal Models

* Fuyu (`FuyuForCausalLM`)
    * [x] Base: [8B](https://huggingface.co/adept/fuyu-8b/commit/f41defefdb89be0d28cac19d94ce216e37cb6be5)

* Gemma (`Gemma3ForConditionalGeneration`)
    * [x] v3: [Instruct-4B](https://huggingface.co/google/gemma-3-4b-it/tree/dbd91bbaf64a0e591f4340ce8b66fd1dba9ab6bd), [Instruct-12B](https://huggingface.co/google/gemma-3-12b-it/tree/7553b6f39c33dc229bfbfe3831f7bcdbb6b738c7), [Instruct-27B](https://huggingface.co/google/gemma-3-27b-it/tree/dfb98f29ff907e391ceed2be3834ca071ea260f1)
    * [x] MedGemma: [Instruct-4B](https://huggingface.co/google/medgemma-4b-it/commit/698f7911b8e0569ff4ebac5d5552f02a9553063c)

    Note: Only download `tokenizer.model` and DO NOT download `tokenizer.json` when converting. Use `--set do-pan-and-scan 1` to enable _Pan and Scan_.

* Kimi (`KimiVLForConditionalGeneration`)
    * [x] VL: [A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/tree/7a3c132a7b0f1f1677f5a72f258bd3afded7d357), [A3B-Thinking](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking/commit/16681d8ac24e505088698e4e34ea494dd6e24400)

* Qwen (`Qwen2AudioForConditionalGeneration`)
    * [x] Qwen2-Audio: [7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/tree/0a095220c30b7b31434169c3086508ef3ea5bf0a)

* SmolVLM2 (`SmolVLMForConditionalGeneration`)
    * [x] [2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/tree/482adb537c021c86670beed01cd58990d01e72e4)

    Note: Use `--set do-split 1` to enable _Split_.

## RAG Models

* Text Embedding (`XLMRobertaModel`)
    * [x] [BCE-Embedding](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
    * [x] [BGE-M3](https://huggingface.co/BAAI/bge-m3) (`-a BGE-M3`)
    * [x] [MiniCPM-Embedding-Light](https://huggingface.co/openbmb/MiniCPM-Embedding-Light)

        Note: Only dense embedding is implemented.

    * Qwen-3 Embedding: [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/b22da495047858cce924d27d76261e96be6febc0), [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B/tree/636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff), [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B/commit/4e423935c619ae4df87b646a3ce949610c66241c)

* QA Ranking (`XLMRobertaForSequenceClassification`)
    * [x] [BCE-ReRanker](https://huggingface.co/maidalun1020/bce-reranker-base_v1)
    * [x] [BGE-ReRanker-M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (`-a BGE-Reranker-M3`)
    * [x] [MiniCPM-Reranker-Light](https://huggingface.co/openbmb/MiniCPM-Reranker-Light)
    * [x] Qwen-3 Reranker: [0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/ad4c588e592307dad69ff0fabc1b3ca5ea8e9f76), [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B/tree/57906229d41697e4494d50ca5859598cf86154a1), [8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B/tree/d678ef8b29dd0eb9d784473da5d5169b21ec948a)

## LoRA Models

These LoRA models have been tested:

* [Llama-3-Chinese-8B-Instruct](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-lora)

## Special Models

* [Meta-AI multi-token prediction models](https://huggingface.co/facebook/multi-token-prediction) checkpoints

    Download at least one multi-token prediction checkpoint (such as 7B_1T_4).
    Assume it is stored at /path/to/llama-multi-predict/7B_1T_4. Make sure `tokenizer.model` is downloaded to /path/to/llama-multi-predict.

    To convert it with `-a llama-multi-token-prediction-ckpt`:

    ```sh
    python convert.py -i /path/to/llama-multi-predict/7B_1T_4 -o llama-multi.bin -a llama-multi-token-prediction-ckpt
    ```

    This is a base model, and remember to use `--format completion`.

    Tip: Use `--kv n_future_tokens N` to change number of future tokens, N = [1, 4].