# Supported Models

## Chat/Instruct Models

* LlaMA-like (`LlamaForCausalLM`):
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
    * [x] DeepHermes-3: [Llama-3-8B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) (Remember to user `-s ...`)
    * [x] Watt-tool: [8B](https://huggingface.co/watt-ai/watt-tool-8B), [70B](https://huggingface.co/watt-ai/watt-tool-70B)

    For other models that using `LlamaForCausalLM` architecture, for example, [aiXcoder-7B](https://huggingface.co/aiXcoder/aixcoder-7b-base), try `-a Yi`.

* Baichuan (`BaichuanForCausalLM`, `BaichuanM1ForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [Chat-13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
    * [x] M1: [Instruct-14B](https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct)

* ChatGLM (`ChatGLMModel`):
    * [x] ChatGLM: [6B](https://huggingface.co/THUDM/chatglm-6b)
    * [x] ChatGLM2 family: [ChatGLM2 6B](https://huggingface.co/THUDM/chatglm2-6b), [CodeGeeX2 6B](https://huggingface.co/THUDM/codegeex2-6b), [ChatGLM3 6B](https://huggingface.co/THUDM/chatglm3-6b)

        Tip on CodeGeeX2: Code completion only, no context. Use system prompt to specify language, e.g. `-s "# language: python"`.

    * [x] CharacterGLM: [6B](https://huggingface.co/thu-coai/CharacterGLM-6B) (`-a CharacterGLM`)

        Note: Use additional key-value pair arguments to specify characters, `--kv user_name "..." bot_name "..." user_info "..." bot_info "..."`.

    * [x] GLM-4: [Chat-9B-128k](https://huggingface.co/THUDM/glm-4-9b-chat), [Chat-9B-1M](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
    * [x] CodeGeeX4: [9B](https://huggingface.co/THUDM/codegeex4-all-9b) (`-a CodeGeeX4`)


* InternLM (`InternLMForCausalLM`, `InternLM2ForCausalLM`)
    * [x] v1: [Chat-7B](https://huggingface.co/internlm/internlm-chat-7b), [Chat-7B v1.1](https://huggingface.co/internlm/internlm-chat-7b-v1_1), [Chat-20B](https://huggingface.co/internlm/internlm-chat-20b)
    * [x] v2: [Chat-1.8B](https://huggingface.co/internlm/internlm2-chat-1_8b), [Chat-7B](https://huggingface.co/internlm/internlm2-chat-7b), [Chat-20B](https://huggingface.co/internlm/internlm2-chat-20b), [Math-Plus-1.8B](https://huggingface.co/internlm/internlm2-math-plus-1_8b), [Math-Plus-7B](https://huggingface.co/internlm/internlm2-math-plus-7), [Math-Plus-20](https://huggingface.co/internlm/internlm2-math-plus-20b)
    * [x] v2.5: [Chat-1.8B](https://huggingface.co/internlm/internlm2_5-1_8b-chat), [Chat-7B](https://huggingface.co/internlm/internlm2_5-7b-chat), [Chat-7B-1M](https://huggingface.co/internlm/internlm2_5-7b-chat-1m), [Chat-20B](https://huggingface.co/internlm/internlm2_5-20b-chat)
    * [x] v3: [Instruct-8B](https://huggingface.co/internlm/internlm3-8b-instruct)

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
    * [x] [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) (`-a QwQ`)
    * [x] [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2) (`-a ReaderLM-v2`)
    * [x] DeepSeek-R1-Distill-QWen: [1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), [32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B), [DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) (`-a DeepSeek-R1-Distill-QWen`)

* BlueLM (`BlueLMForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/vivo-ai/BlueLM-7B-Chat), [Chat-7B 32K](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-32K)

* Orion (`OrionForCausalLM`)
    * [x] [Chat-14B](https://huggingface.co/OrionStarAI/Orion-14B-Chat)

* MiniCPM (`MiniCPMForCausalLM`, `MiniCPM3ForCausalLM`)
    * [x] [DPO-2B](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16), [SFT-2B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16),
          [SFT-1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)🔥
    * [x] [2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k) (Note: `--temp 0` is recommended.)
    * [x] [MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B)
    * [x] [4B](https://huggingface.co/openbmb/MiniCPM3-4B)


* Adept Persimmon (`PersimmonForCausalLM`)
    * [x] [Chat-8B](https://huggingface.co/adept/persimmon-8b-chat)

* Gemma (`GemmaForCausalLM`)
    * [x] v1.0: [Instruct-2B](https://huggingface.co/google/gemma-2b-it), [Instruct-7B](https://huggingface.co/google/gemma-7b-it)
    * [x] v1.1: [Instruct-2B](https://huggingface.co/google/gemma-1.1-2b-it), [Instruct-7B](https://huggingface.co/google/gemma-1.1-7b-it)
    * [x] CodeGemma v1.1: [Instruct-7B](https://huggingface.co/google/codegemma-1.1-7b-it)
    * [x] v2: [Instruct-2B](https://huggingface.co/google/gemma-2-2b-it), [Instruct-9B](https://huggingface.co/google/gemma-2-9b-it), [Instruct-27B](https://huggingface.co/google/gemma-2-27b-it)

* Cohere (`CohereForCausalLM`)
    * [x] [C4AI Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
    * [x] [Aya-23-8B](https://huggingface.co/CohereForAI/aya-23-8B), [Aya-23-35B](https://huggingface.co/CohereForAI/aya-23-35B) (`-a Aya-23`, fully compatible with Command-R)
    * [x] [C4AI Command R7B](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024)

* Zhinao (`ZhinaoForCausalLM`)
    * [x] [Chat-7B-4K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-4K), [Chat-7B-32K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-32K), [Chat-7B-360K](https://huggingface.co/qihoo360/360Zhinao-7B-Chat-360K)

* DeepSeek (`DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`)
    * [x] v2: [Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat) (💣 not tested), [Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)

    * [x] Coder v2: [Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) (💣 not tested), [Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)

    * [x] Moonlight: [Instruct-16B](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) (`-a Moonlight`)

    Two optimization modes are defined: speed (default) and memory. See `BaseMLAttention`.

* XVERSE (`XverseForCausalLM`)
    * [x] [Chat-7B](https://huggingface.co/xverse/XVERSE-7B-Chat), [Chat-13B](https://huggingface.co/xverse/XVERSE-13B-Chat), [Chat-65B](https://huggingface.co/xverse/XVERSE-65B-Chat)

    Note: Tokenizer's behavior is not 100% identical.

* AllenAI (`OlmoeForCausalLM`, `Olmo2ForCausalLM`)
    * [x] OLMoE: [Instruct-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)
    * [x] OLM-2: [Instruct-7B](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct), [Instruct-13B](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)

* Granite (`GraniteForCausalLM`, `GraniteMoeForCausalLM`)
    * [x] v3.0: [Instruct-1B-A400M](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct), [Instruct-3B-A800M](https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct)
    * [x] v3.1: [Instruct-1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct), [Instruct-3B-A800M](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct), [Instruct-8B](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
    * [x] v3.2: [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct), [Instruct-2B](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct)

* EXAONE (`ExaoneForCausalLM`)
    * [x] v3.5: [Instruct-2.4B](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct), [Instruct-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct), [Instruct-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct)

* TeleChat (`TeleChat2ForCausalLM`)
    * [x] v2: [3B](https://huggingface.co/Tele-AI/TeleChat2-3B), [7B](https://huggingface.co/Tele-AI/TeleChat2-7B), [115B](https://huggingface.co/Tele-AI/TeleChat2-115B)

* HunYuan (`HunYuanForCausalLM`)
    * [x] Dense: [Instruct-7B](https://huggingface.co/tencent/Hunyuan-7B-Instruct)

## Base Models

Please use `--format completion` for these models.

* LlaMA-like (`LlamaForCausalLM`):
    * [x] DeepSeek: [Coder-Base-1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) (`-a DeepSeekCoder`), [Coder-Base-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) (`-a DeepSeekCoder`)

* DeepSeek (`DeepseekV2ForCausalLM`)
    * [x] [Coder-V2-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base) (💣 not tested), [Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)

* Mistral (`MistralForCausalLM`, `MixtralForCausalLM`)
    * [x] Mistral: [Base-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Base-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

* Gemma (`GemmaForCausalLM`)
    * [x] CodeGemma v1.1: [Base-2B](https://huggingface.co/google/codegemma-1.1-2b), [Base-7B](https://huggingface.co/google/codegemma-1.1-7b)

* Grok-1
    * [x] [Base](https://huggingface.co/xai-org/grok-1)

        About [Grok-1](grok.md).

* StarCoder (`Starcoder2ForCausalLM`)
    * [x] [Base-3B](https://huggingface.co/bigcode/starcoder2-7b), [Base-7B](https://huggingface.co/bigcode/starcoder2-7b), [Base-15B](https://huggingface.co/bigcode/starcoder2-15b)

* Stable-LM (`StableLMEpochModel`)
    * [x] [Code-3B](https://huggingface.co/stabilityai/stable-code-3b)

* AlphaGeometry-LM (`-a AlphaGeometry-LM`)
    * [x] [geometry.757](https://bit.ly/alphageometry)

## RAG Models

* Text Embedding (`XLMRobertaModel`)
    * [x] [BCE-Embedding](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
    * [x] [BGE-M3](https://huggingface.co/BAAI/bge-m3) (`-a BGE-M3`)
    * [x] [MiniCPM-Embedding-Light](https://huggingface.co/openbmb/MiniCPM-Embedding-Light)

        Note: Only dense embedding is implemented.

* QA Ranking (`XLMRobertaForSequenceClassification`)
    * [x] [BCE-ReRanker](https://huggingface.co/maidalun1020/bce-reranker-base_v1)
    * [x] [BGE-ReRanker-M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (`-a BGE-Reranker-M3`)
    * [x] [MiniCPM-Reranker-Light](https://huggingface.co/openbmb/MiniCPM-Reranker-Light)

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