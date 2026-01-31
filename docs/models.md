# Supported Models

## Chat/Instruct Models

* Adept Persimmon (`PersimmonForCausalLM`)
    * [x] [Chat-8B](https://huggingface.co/adept/persimmon-8b-chat)

* Apertus (`ApertusForCausalLM`)
    * [x] [8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509/tree/9579a2de0fd74118ba3f3714cdc13585607762a8), [70B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509/tree/69eeb773ba2dadf22ab9d2dbebd5771bdfdcc1e8)

        Note: Use `--set enable-thinking 1` to enable thinking.

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

* ChatGLM (`ChatGLMModel`, `Glm4ForCausalLM`, `Glm4MoeLiteForCausalLM`):
    * [x] ~~ChatGLM: [6B](https://huggingface.co/THUDM/chatglm-6b)~~
    * [x] ChatGLM2 family: [ChatGLM2 6B](https://huggingface.co/THUDM/chatglm2-6b), [CodeGeeX2 6B](https://huggingface.co/THUDM/codegeex2-6b), [ChatGLM3 6B](https://huggingface.co/THUDM/chatglm3-6b)

        Tip on CodeGeeX2: Code completion only, no context. Use system prompt to specify language, e.g. `-s "# language: python"`.

    * [x] CharacterGLM: [6B](https://huggingface.co/thu-coai/CharacterGLM-6B) (`-a CharacterGLM`)

        Note: Use additional key-value pair arguments to specify characters, `--kv user_name "..." bot_name "..." user_info "..." bot_info "..."`.

    * [x] GLM-4: [Chat-9B-128k](https://huggingface.co/THUDM/glm-4-9b-chat), [Chat-9B-1M](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
    * [x] CodeGeeX4: [9B](https://huggingface.co/THUDM/codegeex4-all-9b) (`-a CodeGeeX4`)
    * [x] GLM-4: [GLM-4-0414](https://huggingface.co/THUDM/GLM-4-9B-0414/tree/645b8482494e31b6b752272bf7f7f273ef0f3caf),
    [GLM-Z1-9B-0414](https://huggingface.co/THUDM/GLM-Z1-9B-0414/commit/66907f8ddffdcdcafb7e8c9e96b7366449176c1a),
    [GLM-4-32B-0414](https://huggingface.co/THUDM/GLM-4-32B-0414/tree/dc842812afac55db3a8aca4484ca6f3c60bd472f),
    [GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414/tree/93593a027f03a597582d1db7e84894a0a8fa027e),
    [GLM-Z1-Rumination-32B-0414](https://huggingface.co/THUDM/GLM-Z1-Rumination-32B-0414/tree/6ae9ac6152a7d85761409d6d6ef201d9e8e8aeb1)
    * [x] 4.7-Flash: (https://huggingface.co/zai-org/GLM-4.7-Flash/tree/279ecdf8ee35f17f1939f95d6b113d8b806a7b2b)

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

* ERNIE (`Ernie4_5_ForCausalLM`, `Ernie4_5_MoeForCausalLM`)
    * [x] Non-thinking: [0.3B](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT/tree/c163aa422d265f995b024d1322d91c4e3cb52ec8), [A3B](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/tree/b24b8917f5379129992dad46c279683c7b845c96)
    * [x] Thinking: [A3B](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking/tree/78d7a200cddb8132b074adffcd5aa2ef3361b0ae)

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

    * [x] Rnj-1: [Intruct](https://huggingface.co/EssentialAI/rnj-1-instruct/tree/2360f0368eec9bcf6d51aec66f6341503a6934f0)

* GPT (`GptOssForCausalLM`)
    * [x] OSS: [20B](https://huggingface.co/openai/gpt-oss-20b/tree/cbf31f62664d4b1360b3a78427f7b3c3ed8f0fa8), [120B](https://huggingface.co/openai/gpt-oss-120b/tree/bc75b44b8a2a116a0e4c6659bcd1b7969885f423)

    Note: Q4_1/Q4_0 quantization won't work. Use Q8 instead.

* Granite (`GraniteForCausalLM`, `GraniteMoeForCausalLM`)
    * [x] v3.0: [Instruct-1B-A400M](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct), [Instruct-3B-A800M](https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct)
    * [x] v3.1: [Instruct-1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct), [Instruct-3B-A800M](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
    * [x] v3.2: [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct/tree/0276d996f60d5eb0b376b6d06622042d4ef3eb4b)

* GroveMoE (`GroveMoeForCausalLM`)
    * [x] [Inst](https://huggingface.co/inclusionAI/GroveMoE-Inst/tree/b3441abf1f3ed166e58c005ef2c528c584b55764)

* HunYuan (`HunYuanForCausalLM`, `HunYuanDenseV1ForCausalLM`)
    * [x] ~~Dense: [Instruct-7B](https://huggingface.co/tencent/Hunyuan-7B-Instruct)~~ (lost)
    * [x] Dense: [0.5B-Instruct](https://huggingface.co/tencent/Hunyuan-0.5B-Instruct/tree/9ec1774c379d7dde3f2d7ddd3286cde88949e181),
    [1.8B-Instruct](https://huggingface.co/tencent/Hunyuan-1.8B-Instruct/tree/21ab9fd367ee99ba8001d34a182252ddb2ed255c),
    [4B-Instruct](https://huggingface.co/tencent/Hunyuan-4B-Instruct/tree/3a419720cb283ece18dc6baac1b2484418cf525f),
    [7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct/tree/e256110382dc42f4e2f4d97afc9f8bea5a907a4a)
    * [x] MoE: [A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct/tree/202c9758065873e0ac7c80211e6275593f165442)
    * [x] MT1.5: [1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B/tree/dbad03788f49709801014c95d481a514c272ca52), [7B](https://huggingface.co/tencent/HY-MT1.5-7B/tree/f1aa7278ccb1825edaf806c7e40d1febcbf2cbbd)

* Instella (`InstellaForCausalLM`)
    * [x] [Instruct-3B](https://huggingface.co/amd/Instella-3B-Instruct)

* InternLM (`InternLMForCausalLM`, `InternLM2ForCausalLM`)
    * [x] v1: [Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), [Chat-7B v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1_1), [Chat-20B](https://huggingface.co/internlm/internlm-chat-20b)
    * [x] v2: [Chat-1.8B](https://huggingface.co/internlm/internlm2-chat-1_8b), [Chat-7B](https://huggingface.co/internlm/internlm2-chat-7b), [Chat-20B](https://huggingface.co/internlm/internlm2-chat-20b), [Math-Plus-1.8B](https://huggingface.co/internlm/internlm2-math-plus-1_8b), [Math-Plus-7B](https://huggingface.co/internlm/internlm2-math-plus-7), [Math-Plus-20](https://huggingface.co/internlm/internlm2-math-plus-20b)
    * [x] v2.5: [Chat-1.8B](https://huggingface.co/internlm/internlm2_5-1_8b-chat), [Chat-7B](https://huggingface.co/internlm/internlm2_5-7b-chat), [Chat-7B-1M](https://huggingface.co/internlm/internlm2_5-7b-chat-1m), [Chat-20B](https://huggingface.co/internlm/internlm2_5-20b-chat)
    * [x] v3: [Instruct-8B](https://huggingface.co/internlm/internlm3-8b-instruct)

* Jiutian (`JiutianForCausalLM`)
    * [x] [Math-8B](https://huggingface.co/JT-LM/JT-Math-8B-Instruct/tree/00a347fdae86ddd9e616aa0771492c6aff735697),
    [Math-8B-Thinking](https://huggingface.co/JT-LM/JT-Math-8B-Thinking/tree/87d8db3e39c65fa123c59a97266a3ec02ebf6bd6),
    [Coder-8B-Instruct](https://huggingface.co/JT-LM/JT-Coder-8B-Instruct/tree/9160d51e9acaae266cfef8493ea25d15e7ed6904),
    [DA-8B](https://huggingface.co/JT-LM/JT-DA-8B/commit/8bd5bb1a76305dcc777786b65c239b362cee808e)

* Ling/Ring (`BailingMoeForCausalLM`)
    * [x] [Lite](https://huggingface.co/inclusionAI/Ling-lite/tree/a80ae6c479251f1ae33dda517ab83cdc6a312f99), [Coder-Lite](https://huggingface.co/inclusionAI/Ling-Coder-lite/tree/4a8647acf9d3855d599adaaaf4bf6ca14239d2ab)
    * [x] v1.5: [Ling-lite-1.5-2507](https://huggingface.co/inclusionAI/Ling-lite-1.5-2507/tree/6656efdc763a77102207fc66b176e4c5d07a316b), [Ring-lite2507](https://huggingface.co/inclusionAI/Ring-lite-2507/commit/8cf0ec244871c90102b353cef3568e061fd2504f)
    * [x] v2: [Ling-mini-2.0](https://huggingface.co/inclusionAI/Ling-mini-2.0/tree/56c261e07b78d95dad61336fcbdb21ef4fdbcabe), [Ring-mini-2.0](https://huggingface.co/inclusionAI/Ring-mini-2.0/tree/d4eac003b34b59b733f05039a876616d840a37d6)

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
    * [x] Reke-Flash: [Flash-3](https://huggingface.co/RekaAI/reka-flash-3/tree/69cea64942e4db4809b757ae2b0d312b4b610263), [Flash-3.1](https://huggingface.co/RekaAI/reka-flash-3.1/tree/dcf8d6ee3219ca177b519cab47777d42864b54bd) (`-a Reka-Flash-3`)
    * [x] Nemotron: [Llama-3.1-Nemotron-Nano-8B](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/tree/42f62a403ee352e019834442673256e3fe3de275)
    * [x] LlaMA-4: [Scout-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/tree/7dab2f5f854fe665b6b2f1eccbd3c48e5f627ad8), [Maverick-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/tree/f0ee6477b90b7a6aaefd2cfdf1b4a05d36184137)
    * [x] Seed-Coder: [Instruct-8B](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct/tree/7b934eb8f2ce8f40191fa26d12236eb8bc3a77aa),
    [Reasoning-8B](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Reasoning/tree/47f18f877ea7d6aa5c0d87a474f30d420b22bd98) (`--name Seed-Coder`)
    * [x] Nanbeige4: [3B-Thinking](https://huggingface.co/Nanbeige/Nanbeige4-3B-Thinking-2511/tree/396a1fb7a5e1a87941b321b4da85bc510b424d00)

    For other models that using `LlamaForCausalLM` architecture, for example, [aiXcoder-7B](https://huggingface.co/aiXcoder/aixcoder-7b-base), try `-a Yi`.

    If there are both `tokenizer.model` and `tokenizer.json`, only download `tokenizer.model`.

* Megrez (`MegrezMoeForCausalLM`)
    * [x] (3x7B-A3B)[https://huggingface.co/Infinigence/Megrez2-3x7B-A3B/tree/3ffc3b7c0ffc0f0b27d71fba2a97dcc14c797bb4]

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
    * [x] Small-3.1: [Instruct-24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/73ce7c62b904fa83d7cb018e44c3bc06feed4d81)

    * [x] Devstral: [Small-2505](https://huggingface.co/mistralai/Devstral-Small-2505/tree/a6e97eaf3bfe308cb5396675a716147b2ced37c8), [Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507/tree/3178e9e2d8880880098af656c1fe223927ce74f8)

        Note: Please download `tokenizer.json` from [here](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/73ce7c62b904fa83d7cb018e44c3bc06feed4d81).

* Olm (`OlmoeForCausalLM`, `Olmo2ForCausalLM`)
    * [x] OLMoE: [Instruct-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)
    * [x] OLM-2: [Instruct-7B](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct), [Instruct-13B](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct), [Instruct-32B](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct/tree/5942a2f5e0bc38c2a5f5200cec2ea236d5984547)

* Ouro (`OuroForCausalLM`)
    * [x] [2.6B-Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/tree/a2d3a54cea06168ba371ec3e089734f715824d5b)

        Note: additional options supported (`--set ...`)
        * `total_ut_steps`: default 4
        * `exit_threshold`: default 1.0

* Orion (`OrionForCausalLM`)
    * [x] [Chat-14B](https://huggingface.co/OrionStarAI/Orion-14B-Chat)

* Pangu (`PanguProMoEForCausalLM`)
    * [x] MoE: [Pro-MoE](https://gitcode.com/ascend-tribe/pangu-pro-moe-model/tree/15e45a97fa314d86804f93f7faba107b43f8d25c)
    * [x] Embedded: [7B](https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model/tree/754817a9fc1cc4df2687709b758448f80c2dd64c), [1B](https://ai.gitcode.com/ascend-tribe/openpangu-embedded-1b-model/tree/75dd659167a45d6577555d405edb75e0b88215c2)

* Phi (`PhiForCausalLM`, `Phi3ForCausalLM`)
    * [x] [Phi-2](https://huggingface.co/microsoft/phi-2/tree/eb8bbd1d37d258ea74fb082c53346d33056a83d4)

        Tip: `--temp 0` is recommended. Don't forget to try `--format qa`.

    * [x] [Dolphin Phi-2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2/tree/a084bb141f99f67e8ff56a654e29ddd53a0b4d7a) (`-a DolphinPhi2`) 🐬

    * [x] Phi-3: [Mini-Instruct-4k](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [Mini-Instruct-128k](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct), [Medium-Instruct-4k](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct), [Medium-Instruct-128k](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)

    * [x] Phi-3.5: [Mini-Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), [MoE-Instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

    * [x] Phi-4: [Instruct](https://huggingface.co/microsoft/phi-4), [Mini-Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

* QWen (`QWenLMHeadModel`, `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3ForCausalLM`)
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
        [0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/6130ef31402718485ca4d80a6234f70d9a4cf362),
        [30B-A3B-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/01ee60ece4bb0c7a758003e9f45e4a9059b20594),
        [30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/tree/4a8a1645504d39f8c2b9eacfd6d72dac693d3488),
        [4B-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/eb25fbe4f35f7147763bc24445679d1c00588d89),
        [4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/tree/768f209d9ea81521153ed38c47d515654e938aea)
    * [x] v3-Coder: [30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/tree/d4add6db775e2ab1471d0dfcd5cd80cc4c541ad7)
    * [x] MiMo: [7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL/tree/8c8a4968100fd7a4f0319c54438184bbeb767c82)
    * [x] Confucius3-Math: [14B](https://huggingface.co/netease-youdao/Confucius3-Math/tree/62621490d5dccf5fea997be9df62dd8dc017f777) (`-a DeepSeek-R1-Distill-QWen`)
    * [x] Jan-Nano: [4B](https://huggingface.co/Menlo/Jan-nano/tree/5f4e450c127322db9477400890a0dd951c9f6ab7)
    * [x] Baichuan-M2: [32B](https://huggingface.co/baichuan-inc/Baichuan-M2-32B/tree/ebd3cf50aa02061a941f69e78d365d5620c5b5ac)
    * [x] MiroThinker-v1.5: [30B](https://huggingface.co/miromind-ai/MiroThinker-v1.5-30B/tree/fc29738d08b1b000af7af8f27c84caabeae8fb1d)

* Seed (`SeedOssForCausalLM`)
    * [x] OSS: [36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct/tree/6f42c8b5bf8f3f687bd6fb28833da03a19867ce8)

        Note: Use `--set thinking_budget N` to set `thinking_budget`. Default: -1.

* SmolLM-3 (`SmolLM3ForCausalLM`)
    * [x] [3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/tree/297fd6336cf21656d5f9d30a1db612ceeca67619)

* Solor (`SolarForCausalLM`)
    * [x] [Pro](https://huggingface.co/upstage/solar-pro-preview-instruct/tree/dd4bcf7006df9b1ce3f87711e702e4063832aae3)

* TeleChat (`TeleChat2ForCausalLM`)
    * [x] v2: [3B](https://huggingface.co/Tele-AI/TeleChat2-3B), [7B](https://huggingface.co/Tele-AI/TeleChat2-7B), [115B](https://huggingface.co/Tele-AI/TeleChat2-115B)
    * [x] v2.5 [35B](https://huggingface.co/Tele-AI/TeleChat2.5-35B/commit/e53676611f3c5072f7696a359132eaf456272151), [115B](https://huggingface.co/Tele-AI/TeleChat2-115B/tree/8be654fe28bfe60fca4cd483297167a6e570f93b)

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

* Maya1

    * [maya1] (https://huggingface.co/maya-research/maya1/tree/fbd30e2b3ec92d2e227df20005a73e172bc5d2de)

        [SNAC-24kHz](https://huggingface.co/mlx-community/snac_24khz/tree/556af1cd3b1c5f2d294f6aa9bb886245d7b716ac) is used as codec.
        Use these additional command line options when converting: `--name Maya1 -a Maya1 --snac_model /path/to/snac_24kHz`

        Use `--set voice XX` to describe the voice. [More info](https://huggingface.co/maya-research/maya1/blob/fbd30e2b3ec92d2e227df20005a73e172bc5d2de/prompt.txt).

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
    * [x] MedGemma: [Instruct-4B](https://huggingface.co/google/medgemma-4b-it/commit/698f7911b8e0569ff4ebac5d5552f02a9553063c), [Instruct-27B](https://huggingface.co/google/medgemma-27b-it/tree/2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f)
    * [x] TranslateGemma: [4B](https://huggingface.co/google/translategemma-4b-it/tree/4eded5e6860cc44d717912be26cc02d37606cf4f), [12B](https://huggingface.co/google/translategemma-12b-it/tree/94fb76d93138708b0deaab35e45af0cf5036e02a), [27B](https://huggingface.co/google/translategemma-27b-it/tree/006f953135eb646ad0c92bbde4d96ef7a023140c)

    Note: Only download `tokenizer.model` and DO NOT download `tokenizer.json` when converting. Use `--set do-pan-and-scan 1` to enable _Pan and Scan_.
    Use `--name TranslateGemma` when converting TranslateGemma models to activate translation support: specify language codes in prompts like
    `/en->zh Hello world.`. Source language code can be set to `auto` when translating texts. Default language codes can be configured by `--set ID code`,  such as `--set source-language-code zh` and `--set target-language-code en`.

* GLM (`Glm4vForConditionalGeneration`)
    * [x] v4: [4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash/tree/411bb4d77144a3f03accbf4b780f5acb8b7cde4e)

    Support additional options (Use `--set X Y` to change values) like `Kimi`.

* Janus (`MultiModalityCausalLM`)
    * [x] Pro: [1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B/tree/960ab33191f61342a4c60ae74d8dc356a39fafcb), [7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B/tree/5c3eb3fb2a3b61094328465ba61fcd4272090d67)

    Note: Use `--set parallel-size N` to generate `N` images in a single run (default: 2); `--set gen-head-temperature T` to set temperature of `gen-head` (default: 1.0). Add prefix "/gen " to prompts to generate images.

* Kimi (`KimiVLForConditionalGeneration`)
    * [x] VL: [A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/tree/7a3c132a7b0f1f1677f5a72f258bd3afded7d357), [A3B-Thinking](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking/commit/16681d8ac24e505088698e4e34ea494dd6e24400), [A3B-Thinking-2506](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/tree/f124f44fb6ab5778cfac5117e3902ef03e860ad4)

    Additional options (Use `--set X Y` to change values):
    * `video_max_frames`: default 20.
    * `native_resolution`: use native resolution or not, default: `false` (This seems sensitive to quantization, so defaults to `false`).
    * `fps`: Default 1.0.

* Mistral (`Mistral3ForConditionalGeneration`)
    * [x] Ministral-3: [3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/tree/e904b5a798e9397c0fd04e063a2aa90355653ffe),
    [3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512/tree/039f888eb54340b5e9870721f3c249fbc809b8e8),
    [8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16/tree/bde2b3370dbf8ad77ceab25a5a43bc9013cda350),
    [8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512/tree/f511871f6402ba68dadfb42a94a7a7e13499fd65)
    * [x] Devstral-Small-2: [24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/tree/8d27a0d2120f1563c11dc91d494e99f9678ecf79)

* Qwen (`Qwen2AudioForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`, `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`)
    * [x] Qwen2-Audio: [7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/tree/0a095220c30b7b31434169c3086508ef3ea5bf0a)
    * [x] Qwen2.5-VL: [3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3), [7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5)
    * [x] MiMo-VL: [7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL/tree/460c34be0c6cfe79b6b311647ae9112784f80b73), [7B-RL-2508](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508/tree/4bfb270765825d2fa059011deb4c96fdd579be6f)
    * [x] Dolphin: [v2](https://huggingface.co/ByteDance/Dolphin-v2/tree/c37c62768c644bb594da4283149c627765aa80f3)
    * [x] Qwen3-VL: [2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203),
    [4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17),
    [A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c), etc

* SmolVLM2 (`SmolVLMForConditionalGeneration`)
    * [x] [2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/tree/482adb537c021c86670beed01cd58990d01e72e4)

    Note: Use `--set do-split 1` to enable _Split_.

* Step-VL (`StepVLForConditionalGeneration`)
    * [x] v3: [10B](https://huggingface.co/stepfun-ai/Step3-VL-10B/tree/1028f3d962d9d06039885737cf1c224752f78a70)

    Additional options (Use `--set X Y` to change values):
    * `do-pan-and-scan`: default 1 (i.e. true). Set to `0` to use only a global view to reduce the compute.
    * `native-resolution`: default 0 (i.e. false). This model can support native resolution _mathematically_ without pan and scan. (for experiment only)

## OCR Models

* dots.ocr (`DotsOCRForCausalLM`)
    * [x] [3B](https://huggingface.co/rednote-hilab/dots.ocr/tree/ba670c5dcf03ff4e02015558c95b4042f5dce069)

    Note: Prompt for OCR: _{{image:...}}Extract the text content from this image_. [Here](https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py)
    are other prompts for OCR. Use `+single-turn` to discard history automatically.

* Nanonets-OCR2 (`Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`)
    * [x] OCR2: [3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B/tree/d0368059ad151ce9e38f526890cfd4f27b28be65), [1.5B](https://huggingface.co/nanonets/Nanonets-OCR2-1.5B-exp/tree/306a9b2a65672a3dbebd9bce9a9373a9a18674a2)

## ASR Models

* GLM-ASR (`GlmAsrForConditionalGeneration`)
    * [x] [Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512/tree/8172188f128059b57f09686dda5b36edff1606dd)

* Qwen3-ASR (`Qwen3ASRForConditionalGeneration`)
    * [x] [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0), [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5)

## RAG Models

### Text Embedding

Note: Only dense embedding is implemented.

* Roberta (`XLMRobertaModel`)
    * [x] [BCE-Embedding](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
    * [x] [BGE-M3](https://huggingface.co/BAAI/bge-m3) (`-a BGE-M3`)

* MiniCPM (`MiniCPMModel`)
    * [x] [MiniCPM-Embedding-Light](https://huggingface.co/openbmb/MiniCPM-Embedding-Light)

* Qwen-3 Embedding  (`Qwen3ForCausalLM`)
    * [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/b22da495047858cce924d27d76261e96be6febc0), [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B/tree/636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff), [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B/commit/4e423935c619ae4df87b646a3ce949610c66241c) (`-a Qwen3-Embedding`)

        Note: use `--set task ...` to specify task/instruction.

### Text Ranking

* Roberta (`XLMRobertaForSequenceClassification`)
    * [x] [BCE-ReRanker](https://huggingface.co/maidalun1020/bce-reranker-base_v1)
    * [x] [BGE-ReRanker-M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (`-a BGE-Reranker-M3`)

* MiniCPM (`MiniCPMModel`)
    * [x] [MiniCPM-Reranker-Light](https://huggingface.co/openbmb/MiniCPM-Reranker-Light)

* Qwen-3 Reranker  (`Qwen3ForCausalLM`)
    * [x] [0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/ad4c588e592307dad69ff0fabc1b3ca5ea8e9f76), [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B/tree/57906229d41697e4494d50ca5859598cf86154a1), [8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B/tree/d678ef8b29dd0eb9d784473da5d5169b21ec948a) (`-a Qwen3-Reranker`)

        Note: use `--set task ...` to specify task/instruction.

### Multi-modal Embedding

* Qwen3-VL Embedding (`Qwen3VLForConditionalGeneration`)
    * : [2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/5d57766937d093eb57ad6317c145e59e1c2833ac),
    [8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/tree/93dc85c0aaf1836e8ed35366f06e70625138817b) (`-a Qwen3-VL-Embedding`)

        Note: use `--set task ...` to specify task/instruction.

### Multi-modal Ranking

* Qwen3-VL Reranker (`Qwen3VLForConditionalGeneration`)
    * : [2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/tree/00002d1a4279cfce9b8361e3936d86e522672ebc),
    [8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B/tree/f0c49f76deac23a9d80e9a52e60054ea127d44f0) (`-a Qwen3-VL-Reranker`)

        Note: use `--set task ...` to specify task/instruction.

## LoRA Models

These LoRA models have been tested:

* [Llama-3-Chinese-8B-Instruct](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-lora)

## Special Models

Tips for diffusion LLMs, they are very sensitive to sampling parameters. The results may be completely unacceptable with improper parameters.

* LLaDA (`LLaDA2MoeModelLM`)
    * [x] [mini-preview](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview/tree/d25d3b2ac0b966b64da11d6c791f8bf4bc31e90c), [mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini/tree/d699e90dd1bd154d65128d9447f3570f4dac44f4)

        Supported options (`--set OPTION VALUE`):
        - `block_length`: default 32
        - `steps`: default 32
        - `minimal_topk`: default 1
        - `threshold`: default 0.95

* WeDLM (`WeDLMForCausalLM`)
    * [x] [8B-Instruct](https://huggingface.co/tencent/WeDLM-8B-Instruct/tree/c1e0373ec7e11bc27321a548ea54ea9728b0d9c0)

        Supported options (`--set OPTION VALUE`):
        - `block_size`: default 16

            When set to <= 1, it falls back to auto regressive decoding.
        - `accept_algo`: default 2
            - 0: entropy algo: https://github.com/Tencent/WeDLM/blob/d4481cab821044b8ebd5f78bc37f23787a6275ed/wedlm/engine/sampler.py#L169
            - 1: prob    algo: https://huggingface.co/tencent/WeDLM-8B-Instruct/blob/main/modeling_wedlm.py#L694
            - 2: custom  algo: sampling + prob
        - `threshold`: default 0.7

            For algo 0, tokens are accepted if entropy is less than threshold; for others, tokens are accepted when probability (or condidence level) is larger than this.
        - `pos_penalty_factor`: default 0.02 (used by entropy algo)

* [Meta-AI multi-token prediction models](https://huggingface.co/facebook/multi-token-prediction) checkpoints

    Download at least one multi-token prediction checkpoint (such as 7B_1T_4).
    Assume it is stored at /path/to/llama-multi-predict/7B_1T_4. Make sure `tokenizer.model` is downloaded to /path/to/llama-multi-predict.

    To convert it with `-a llama-multi-token-prediction-ckpt`:

    ```sh
    python convert.py -i /path/to/llama-multi-predict/7B_1T_4 -o llama-multi.bin -a llama-multi-token-prediction-ckpt
    ```

    This is a base model, and remember to use `--format completion`.

    Tip: Use `--kv n_future_tokens N` to change number of future tokens, N = [1, 4].
