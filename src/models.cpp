#include "models.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <regex>
#include <string>
#include <functional>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <numbers>
#include <unordered_map>

#include "layers.h"
#include "JSON.h"
#include "vision_process.h"
#include "audio_process.h"
#include "models_priv.h"

#include "../models/llama.h"
#include "../models/qwen.h"
#include "../models/hunyuan.h"
#include "../models/ernie.h"
#include "../models/pangu.h"
#include "../models/smol.h"

json::JSON json::JSON::_null = json::JSON();

namespace chatllm
{
    static ForwardContext *dbg_ctx = nullptr;
    static std::unordered_map<ggml::tensor *, std::string> inspected_set;
    static ggml::tensor *dbg_w = nullptr;

    void set_dbg_ctx(ForwardContext *c)
    {
        dbg_ctx = c;
    }

    void print_tensor_shape(const char *info, ggml::tensor *tensor)
    {
        printf("%s: shape of %s (%p): [%zd, %zd, %zd, %zd] [%zd, %zd, %zd, %zd]\n",
            info, tensor->name, tensor->data,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    }

    void print_tensor(ggml::tensor *tensor, int offset, const bool full)
    {
        const int PRINT_CNT = 64;
        print_tensor_shape("\n", tensor);

        std::vector<uint8_t > data;
        data.resize(ggml::nbytes(tensor));
        Backend::read_tensor_data(tensor, data.data());

        switch (tensor->type)
        {
        case GGML_TYPE_F32:
            {
                float * p = (float *)data.data();
                const size_t n = ggml::nbytes(tensor) / sizeof(float);
                for (size_t i = 0; i < n; i++)
                {
                    if (!full && ((PRINT_CNT < i) && (i < n - PRINT_CNT))) continue;
                    float t = p[i];
                    //t = ggml_fp16_to_fp32(ggml_fp32_to_fp16(t));
                    printf("[%3d] = %+3.18f\n", (int)i, t);
                    //printf("[%3d] = %08x\n", (int)i, *(uint32_t *)(p + i));
                }
            }
            break;
        case GGML_TYPE_F16:
            {
                ggml_fp16_t * p = (ggml_fp16_t *)data.data();
                const size_t n = ggml::nbytes(tensor) / sizeof(ggml_fp16_t);
                for (size_t i = 0; i < n; i++)
                {
                    if (!full && ((PRINT_CNT < i) && (i < n - PRINT_CNT))) continue;

                    printf("[%3d] = %+3.18f\n", (int)i,  ggml_fp16_to_fp32(p[i]));
                }
            }
            break;
        case GGML_TYPE_Q8_0:
            {
                #define QK8_0 32
                typedef struct {
                    ggml_fp16_t d;       // delta
                    int8_t  qs[QK8_0]; // quants
                } block_q8_0;

                char *pp = (char *)data.data();
                for (size_t i = 0; i < ggml::nbytes(tensor) / sizeof(block_q8_0); i++)
                {
                    block_q8_0 *p = (block_q8_0 *)(pp + i * sizeof(block_q8_0));
                    float scale = ggml_fp16_to_fp32(p->d);

                    printf("[%3d] =", (int)i * QK8_0);

                    for (int j = 0; j < QK8_0; j++)
                    {
                        printf(" %+3.15f", p->qs[j] * scale);
                    }
                    printf("\n");
                }
            }
            break;
        default:
            {
                char * p = (char *)data.data();
                p += offset;
                for (size_t i = 0; i < ggml::nbytes(tensor); i++)
                {
                    if ((i & 0xf) == 0) printf("\n%05d: ", (int)i);
                    printf("%5d", p[i]);
                }
            }
            break;
        }

        printf("\n");
    }

    static bool need_observe_tensor_evaluation_callback(ggml::tensor *tensor, void *user_data)
    {
        return inspected_set.find(tensor) != inspected_set.end();
    }

    static bool observe_tensor_evaluation_callback(ggml::tensor *tensor, void *user_data)
    {
        auto it = inspected_set.find(tensor);
        if (it == inspected_set.end()) return true;

        if (dbg_w)
        {
            printf("\n--------------- dbg_w ----------------------\n");
            print_tensor(dbg_w);

            dbg_w = nullptr;
        }

        printf("\n--------------- %s ----------------------\n", it->second.c_str());
        bool full = true;
        print_tensor(tensor, 0, full);

        return true;
    }

    void dump_weight_tensor(ggml::tensor *tensor)
    {
        dbg_w = tensor;
    }

    void inspect_tensor(ggml::tensor *tensor, const char *format, ...)
    { //return;
        if (nullptr == dbg_ctx) return;
        if (tensor == nullptr) return;

        std::string tag;

        va_list args;
        va_start(args, format);
        int size = vsnprintf(nullptr, 0, format, args) + 1; // +1 for the null terminator
        va_end(args);

        if (size > 0)
        {
            std::unique_ptr<char[]> buffer(new char[size]);

            va_start(args, format);
            vsnprintf(buffer.get(), size, format, args);
            va_end(args);

            tag = buffer.get();
        }

        inspected_set[tensor] = tag;

        dbg_ctx->get_backend_context()->set_eval_observe_callback(need_observe_tensor_evaluation_callback, observe_tensor_evaluation_callback, nullptr);
    }

    ModelPurpose get_model_purpose(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_BCE_Embedding:
        case MODEL_TYPE_BGE_M3:
        case MODEL_TYPE_MiniCPM_Embedding_Light:
        case MODEL_TYPE_QWEN3_Embedding:
            return ModelPurpose::TextEmbedding;
        case MODEL_TYPE_BCE_ReRanker:
        case MODEL_TYPE_BGE_ReRanker_M3:
        case MODEL_TYPE_MiniCPM_ReRanker_Light:
        case MODEL_TYPE_QWEN3_ReRanker:
            return ModelPurpose::Ranker;
        case MODEL_TYPE_ORPHEUS_TTS:
        case MODEL_TYPE_OUTE_TTS_LLAMA:
        case MODEL_TYPE_OUTE_TTS_QWEN3:
            return ModelPurpose::TTS;
        default:
            return ModelPurpose::Chat;
        }
    }

    ChatModelAccessPoints get_chat_model_access_points(ModelType model_type)
    {
        if (get_model_purpose(model_type) != ModelPurpose::Chat) return 0;

        switch (model_type)
        {
        case MODEL_TYPE_LLAMA_MULTI:
            return ChatModelAccessPoint::Text;
        default:
            break;
        }

        ChatModelAccessPoints tag = model_type >> 24;
        return (tag << 1) | 1;
    }

    static std::string format_access_points(ChatModelAccessPoints bitmap)
    {
        const static std::vector<std::string> names({
            "Text", "Image Input", "Image Output", "Audio Input", "Audio Output", "Video Input", "Video Output"
        });

        std::vector<std::string> aps;
        for (size_t i = 0; i < names.size(); i++)
            if (bitmap & (1 << i))
                aps.push_back(names[i]);
        return utils::join(aps, ", ");
    }

    std::string to_string(ModelPurpose purpose)
    {
        switch (purpose)
        {
        case ModelPurpose::TextEmbedding:
            return "Text Embedding";
        case ModelPurpose::Ranker:
            return "Ranker";
        case ModelPurpose::Chat:
            return "Chat";
        case ModelPurpose::TTS:
            return "TTS";
        case ModelPurpose::ASR:
            return "ASR";
        default:
            CHATLLM_THROW << "unknown model purpose: " << purpose;
            return "???";
        }
    }

    std::string format_model_capabilities(ModelType model_type)
    {
        if (get_model_purpose(model_type) != ModelPurpose::Chat)
        {
            return to_string(get_model_purpose(model_type));
        }
        ChatModelAccessPoints bitmap = get_chat_model_access_points(model_type);
        return format_access_points(bitmap);
    }

    std::string format_model_capabilities(uint32_t model_type)
    {
        return format_model_capabilities((ModelType)model_type);
    }

    std::string to_string(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_CHATGLM:
            return "ChatGLM";
        case MODEL_TYPE_CHATGLM2:
            return "ChatGLM2";
        case MODEL_TYPE_CHATGLM3:
            return "ChatGLM3";
        case MODEL_TYPE_GLM4:
            return "GLM-4";
        case MODEL_TYPE_CODEGEEX2:
            return "CodeGeeX2";
        case MODEL_TYPE_CODEGEEX4:
            return "CodeGeeX4";
        case MODEL_TYPE_CHARACTERGLM:
            return "CharacterGLM";
        case MODEL_TYPE_INTERNLM:
        case MODEL_TYPE_INTERNLM2:
        case MODEL_TYPE_INTERNLM2_1:
        case MODEL_TYPE_INTERNLM3:
            return "InternLM";
        case MODEL_TYPE_LLAMA2:
        case MODEL_TYPE_LLAMA2PLUS:
            return "LlaMA2";
        case MODEL_TYPE_MEGREZ:
            return "Megrez";
        case MODEL_TYPE_FALCON3:
            return "Falcon3";
        case MODEL_TYPE_REKA_FLASH3:
            return "Reka-Flash-3";
        case MODEL_TYPE_CODELLAMA:
            return "CodeLlaMa";
        case MODEL_TYPE_BAICHUAN:
        case MODEL_TYPE_BAICHUANLLAMA:
            return "Baichuan";
        case MODEL_TYPE_BAICHUAN_M1:
            return "Baichuan-M1";
        case MODEL_TYPE_DEEPSEEK:
            return "DeepSeek-LLM";
        case MODEL_TYPE_DEEPSEEK_CODER:
            return "DeepSeek-Coder";
        case MODEL_TYPE_CODEFUSE_DEEPSEEK:
            return "CodeFuse-DeepSeek";
        case MODEL_TYPE_NUMINAMATH:
            return "NumiaMath";
        case MODEL_TYPE_DEEPSEEK_V2:
        case MODEL_TYPE_DEEPSEEK_V2_LIGHT:
            return "DeepSeek-V2";
        case MODEL_TYPE_DEEPSEEK_V3:
        case MODEL_TYPE_DEEPSEEK_V3_LIGHT:
            return "DeepSeek-V3";
        case MODEL_TYPE_DEEPSEEK_V1_MoE:
            return "DeepSeek-V1-MoE";
        case MODEL_TYPE_GIGACHAT:
            return "GigaChat";
        case MODEL_TYPE_YI:
            return "Yi";
        case MODEL_TYPE_MAP_NEO:
            return "MAP-Neo";
        case MODEL_TYPE_PHI2:
        case MODEL_TYPE_PHI2_V2:
            return "Phi-2";
        case MODEL_TYPE_PHI3:
        case MODEL_TYPE_PHI3_SU:
        case MODEL_TYPE_PHI3_SU2:
        case MODEL_TYPE_PHI3_SU3:
            return "Phi-3";
        case MODEL_TYPE_PHI3_MOE:
            return "Phi-3.5 MoE";
        case MODEL_TYPE_PHI4:
        case MODEL_TYPE_PHI4_MINI:
            return "Phi-4";
        case MODEL_TYPE_DOLPHINPHI2:
        case MODEL_TYPE_DOLPHINPHI2_V2:
            return "Dolphin Phi-2";
        case MODEL_TYPE_WIZARDCODER:
            return "WizardCoder";
        case MODEL_TYPE_WIZARDLM:
            return "WizardLM";
        case MODEL_TYPE_WIZARDMATH:
            return "WizardMath";
        case MODEL_TYPE_MISTRAL:
        case MODEL_TYPE_MISTRAL2:
            return "Mistral";
        case MODEL_TYPE_DEEPHERMES3_MISTRAL:
            return "DeepHermes-3-Mistral";
        case MODEL_TYPE_MIXTRAL:
            return "Mixtral MoE";
        case MODEL_TYPE_OPENCHAT:
            return "OpenChat";
        case MODEL_TYPE_NEURALBEAGLE:
            return "NeuralBeagle";
        case MODEL_TYPE_STARLING:
            return "Starling";
        case MODEL_TYPE_WIZARDLM2_MOE:
            return "WizardLM-2-MoE";
        case MODEL_TYPE_QWEN:
            return "QWen";
        case MODEL_TYPE_QWEN2:
        case MODEL_TYPE_QWEN2TIE:
            return "QWen2";
        case MODEL_TYPE_QWEN2MoE:
            return "QWen2-MoE";
        case MODEL_TYPE_MARCO_O1:
            return "Marco-o1";
        case MODEL_TYPE_QWQ:
            return "QwQ";
        case MODEL_TYPE_TIGERBOT:
            return "TigerBot";
        case MODEL_TYPE_BLUELM:
            return "BlueLM";
        case MODEL_TYPE_STABLELM:
            return "StableLM";
        case MODEL_TYPE_ORION:
            return "Orion";
        case MODEL_TYPE_MINICPM:
        case MODEL_TYPE_MINICPM2:
            return "MiniCPM";
        case MODEL_TYPE_MINICPM3:
            return "MiniCPM3";
        case MODEL_TYPE_MINICPM_MoE:
            return "MiniCPM-MoE";
        case MODEL_TYPE_PERSIMMON:
            return "Persimmon";
        case MODEL_TYPE_FUYU:
            return "Fuyu";
        case MODEL_TYPE_GEMMA:
            return "Gemma";
        case MODEL_TYPE_GEMMA2:
            return "Gemma-2";
        case MODEL_TYPE_GEMMA3:
            return "Gemma-3";
        case MODEL_TYPE_COHERE_COMMAND_R:
            return "Command-R";
        case MODEL_TYPE_COHERE_COMMAND_R7B:
            return "Command-R7B";
        case MODEL_TYPE_COHERE_AYA_23:
            return "Aya-23";
        case MODEL_TYPE_GROK_1:
            return "Grok-1";
        case MODEL_TYPE_ZHINAO:
            return "Zhinao";
        case MODEL_TYPE_LLAMA3:
            return "LlaMA3";
        case MODEL_TYPE_LLAMA3_1:
            return "LlaMA3.1";
        case MODEL_TYPE_LLAMA3_2:
            return "LlaMA3.2";
        case MODEL_TYPE_EXAONE:
            return "EXAONE";
        case MODEL_TYPE_BCE_Embedding:
            return "BCE-Embedding";
        case MODEL_TYPE_BCE_ReRanker:
            return "BCE-ReRanker";
        case MODEL_TYPE_BGE_M3:
            return "BGE-M3";
        case MODEL_TYPE_BGE_ReRanker_M3:
            return "BGE-ReRanker-M3";
        case MODEL_TYPE_STARCODER2:
            return "StarCoder2";
        case MODEL_TYPE_XVERSE:
            return "XVERSE";
        case MODEL_TYPE_INDEX:
            return "Index";
        case MODEL_TYPE_OLMoE:
            return "OLMoE";
        case MODEL_TYPE_OLMo2:
            return "OLM-2";
        case MODEL_TYPE_LLAMA_MULTI:
            return "LlaMA-Multi";
        case MODEL_TYPE_SMOLLM:
            return "SmolLM";
        case MODEL_TYPE_LLAMA3_GROQ_TOOL:
            return "LlaMA-Groq-Tool-Use";
        case MODEL_TYPE_ALPHAGEO_LM:
            return "AlphaGeometry-LM";
        case MODEL_TYPE_GRANITE_MoE:
            return "Granite-MoE";
        case MODEL_TYPE_GRANITE:
            return "Granite";
        case MODEL_TYPE_TELECHAT2:
            return "TeleChat2";
        case MODEL_TYPE_HUNYUAN_DENSE:
            return "HuanYuan";
        case MODEL_TYPE_READERLM2:
            return "ReaderLM-v2";
        case MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN:
            return "DeepSeek-R1-Distill-QWen";
        case MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA:
            return "DeepSeek-R1-Distill-LlaMA";
        case MODEL_TYPE_AQUILA2:
            return "Aquila2";
        case MODEL_TYPE_MiniCPM_Embedding_Light:
            return "MiniCPM-Embedding-Light";
        case MODEL_TYPE_MiniCPM_ReRanker_Light:
            return "MiniCPM-ReRanker-Light";
        case MODEL_TYPE_MOONLIGHT:
            return "Moonlight";
        case MODEL_TYPE_INSTELLA:
            return "Instella";
        case MODEL_TYPE_DECILM:
            return "DeciLM";
        case MODEL_TYPE_SOLARPRO:
            return "Solar-Pro";
        case MODEL_TYPE_BAILINGMOE:
            return "Bailing";
        default:
            return "???";
        }
    }

    std::string to_native_string(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_INTERNLM:
        case MODEL_TYPE_INTERNLM2:
        case MODEL_TYPE_INTERNLM2_1:
        case MODEL_TYPE_INTERNLM3:
            return "书生·浦语";
        case MODEL_TYPE_BAICHUAN:
        case MODEL_TYPE_BAICHUANLLAMA:
        case MODEL_TYPE_BAICHUAN_M1:
            return "百川";
        case MODEL_TYPE_PHI2:
        case MODEL_TYPE_PHI2_V2:
        case MODEL_TYPE_PHI3:
        case MODEL_TYPE_PHI3_SU:
        case MODEL_TYPE_PHI3_SU2:
        case MODEL_TYPE_PHI3_SU3:
        case MODEL_TYPE_PHI3_MOE:
        case MODEL_TYPE_PHI4:
            return "Φ";
        case MODEL_TYPE_QWEN:
        case MODEL_TYPE_QWEN2:
        case MODEL_TYPE_QWEN2TIE:
        case MODEL_TYPE_QWEN2MoE:
        case MODEL_TYPE_QWQ:
            return "通义千问";
        case MODEL_TYPE_TIGERBOT:
            return "虎博";
        case MODEL_TYPE_BLUELM:
            return "蓝心";
        case MODEL_TYPE_NEURALBEAGLE:
            return "🐶";
        case MODEL_TYPE_COHERE_COMMAND_R:
            return "⌘-R";
        case MODEL_TYPE_ZHINAO:
            return "360智脑";
        case MODEL_TYPE_XVERSE:
            return "元象";
        case MODEL_TYPE_MEGREZ:
            return "无穹天权";
        case MODEL_TYPE_TELECHAT2:
            return "星辰";
        case MODEL_TYPE_ORION:
            return "猎户星空";
        case MODEL_TYPE_HUNYUAN_DENSE:
            return "混元";
        default:
            return "";
        }
    }

    class LogitsPenalty
    {
    public:
        LogitsPenalty()
            : repeat_penalty_en(false),
              freq_penalty_en(false),
              inv_repeat_penalty(0.0f), repeat_penalty(0.0f), freq_penalty(0.0f), presence_penalty(0.0f)
        {}

        LogitsPenalty(const GenerationConfig &gen_config)
            : repeat_penalty_en((gen_config.penalty_window > 0) && (gen_config.repeat_penalty != 1.0f) && (gen_config.repeat_penalty > 0.0f)),
              freq_penalty_en((gen_config.penalty_window > 0) && ((gen_config.frequency_penalty != 0.0f) || (gen_config.presence_penalty != 0.0f))),
              inv_repeat_penalty(repeat_penalty_en ? 1 / gen_config.repeat_penalty : 0.0f),
              repeat_penalty(gen_config.repeat_penalty),
              freq_penalty(freq_penalty_en ? gen_config.frequency_penalty / gen_config.penalty_window : 0.0f),
              presence_penalty(gen_config.presence_penalty)
        {
            if (gen_config.penalty_window > 0)
            {
                token_history.resize(gen_config.penalty_window);
            }
            reset();
        }

        virtual void skip_this(int token_id)
        {
            skip_tokens.emplace(token_id);
        }

        virtual void reset()
        {
            for (size_t i = 0; i < token_history.size(); i++)
                token_history[i] = -1;
            hist_write = 0;
            memset(token_count.data(), 0, token_count.size() * sizeof(token_count[0]));
        }

        virtual void accept_choice(int token_id)
        {
            if (token_history.size() < 1) return;
            int id = token_history[hist_write];
            if ((0 <= id) && (id < (int)token_count.size()))
                token_count[id]--;
            token_history[hist_write++] = token_id;
            if (hist_write >= token_history.size()) hist_write = 0;
            if ((0 <= token_id) && (token_id < (int)token_count.size()))
                token_count[token_id]++;
        }

        virtual void process(float *logits, const int vocab_size)
        {
            if (token_history.size() < 1) return;

            if (vocab_size != (int)token_count.size())
            {
                token_count.resize(vocab_size);
            }

            for (int i = 0; i < vocab_size; i++)
            {
                if (repeat_penalty_en)
                {
                    if (token_count[i] > 0)
                        logits[i] *= logits[i] > 0 ? inv_repeat_penalty : repeat_penalty;
                }

                if (freq_penalty_en)
                    logits[i] -= float(token_count[i]) * freq_penalty + float(token_count[i] > 0) * presence_penalty;
            }
        }

    protected:
        const bool repeat_penalty_en;
        const bool freq_penalty_en;
        const float inv_repeat_penalty;
        const float repeat_penalty;
        const float freq_penalty;
        const float presence_penalty;
        std::vector<int> token_history;
        std::vector<int> token_count;
        size_t hist_write;
        std::set<int> skip_tokens;
    };

    class Sampler
    {
    public:
        static const int ABORT = -1;
        Sampler() : penalty() {}
        virtual ~Sampler() = default;

        Sampler(const GenerationConfig &gen_config)
            : penalty(gen_config)
        {}
    public:
        virtual void seed(int x)
        {
            gen.seed((unsigned int)x);
        }

        virtual void reset()
        {
            penalty.reset();
        }

        virtual int sampling(float *logits, const int vocab_size) = 0;
    public:
        LogitsPenalty penalty;
    protected:
        std::mt19937 gen;
    };

    class GreedySampler : public Sampler
    {
    public:
        int sampling(float *logits, const int vocab_size) override
        {
            return (int)(std::max_element(logits, logits + vocab_size) - logits);
        }
    };

    class NonGreedySampler: public Sampler
    {
    public:
        NonGreedySampler(const GenerationConfig &gen_config, float temperature, int top_k)
            : Sampler(gen_config),
              inv_temp(0.0f), top_k(top_k)
        {
            temp_en = fabs(temperature - 1.0f) > 1e-5f;
            if (temp_en) inv_temp = 1.f / temperature;
        }


        int sampling(float *logits, const int vocab_size) override
        {
            if (temp_en)
            {
                for (int i = 0; i < vocab_size; i++)
                    logits[i] *= inv_temp;
            }

            penalty.process(logits, vocab_size);

            token_scores.resize(vocab_size);

            for (int i = 0; i < vocab_size; i++)
            {
                token_scores[i] = {.id = i, .score = logits[i]};
            }

            // top_k sampling
            if (0 < top_k && top_k < (int)token_scores.size())
            {
                std::nth_element(token_scores.begin(), token_scores.begin() + top_k, token_scores.end(),
                                std::greater<TokenIdScore>());
                token_scores.resize(top_k);
            }

            do_sampling(logits, vocab_size);

            if (token_scores.size() < 1)
                return ABORT;

            // sample next token
            for (size_t i = 0; i < token_scores.size(); i++)
            {
                logits[i] = token_scores[i].score;
            }

            std::discrete_distribution<> dist(logits, logits + token_scores.size());
            int next_token_id = token_scores[dist(gen)].id;

            penalty.accept_choice(next_token_id);

            return next_token_id;
        }

    protected:
        struct TokenIdScore
        {
            int id;
            float score;

            bool operator<(const TokenIdScore &other) const { return score < other.score; }
            bool operator>(const TokenIdScore &other) const { return score > other.score; }
        };

        void sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last)
        {
            float max_score = std::max_element(first, last)->score;
            float sum = 0.f;
            for (TokenIdScore *p = first; p != last; p++)
            {
                float s = std::exp(p->score - max_score);
                p->score = s;
                sum += s;
            }
            float inv_sum = 1.f / sum;
            for (TokenIdScore *p = first; p != last; p++)
            {
                p->score *= inv_sum;
            }
        }

        virtual void do_sampling(float *logits, const int vocab_size) = 0;
        bool temp_en;
        float inv_temp;
        int top_k;
        std::vector<TokenIdScore> token_scores;
    };

    class TopPSampler : public NonGreedySampler
    {
    public:
        TopPSampler(const GenerationConfig &gen_config, float temperature, int top_k, float top_p)
            : NonGreedySampler(gen_config, temperature, top_k), top_p(top_p)
        {}

    protected:
        void do_sampling (float *next_token_logits, const int vocab_size) override
        {
            // top_p sampling
            if (0.f < top_p && top_p < 1.f)
            {
                std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>()); // hot code!
                sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());

                float cumsum = 0.f;
                for (size_t i = 0; i < token_scores.size(); i++)
                {
                    cumsum += token_scores[i].score;
                    if (cumsum >= top_p)
                    {
                        token_scores.resize(i + 1);
                        break;
                    }
                }
            }

            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        }

    protected:
        const float top_p;
    };

    // Reference:
    // https://www.trentonbricken.com/Tail-Free-Sampling/#tail-free-sampling-algorithm
    class FreeTailSampler : public NonGreedySampler
    {
    public:
        FreeTailSampler(const GenerationConfig &gen_config, float temperature, int top_k, float z)
            : NonGreedySampler(gen_config, temperature, top_k), z(z)
        {}

    protected:

        void do_sampling(float *next_token_logits, const int vocab_size) override
        {
            if (token_scores.size() < 3) return;

            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
            std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>()); // hot code!

            snd_d.resize(token_scores.size() - 2);
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                snd_d[i] = token_scores[i].score + token_scores[i + 2].score - 2 * token_scores[i + 1].score;
            }

            // abs, then norm
            float sum = 1e-6f;
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                snd_d[i] = fabs(snd_d[i]);
                sum += snd_d[i];
            }
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                snd_d[i] /= sum;
            }

            float cdf = 0.0;
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                cdf += snd_d[i];
                if (cdf > z)
                {
                    token_scores.resize(i + 1);
                    break;
                }
            }
        }

    protected:
        const float z;
        std::vector<float> snd_d;
    };

    class SamplerFactory
    {
    public:
        static Sampler *Create(const GenerationConfig &gen_config, int seed)
        {
            Sampler *r = nullptr;
            if (gen_config.do_sample)
            {
                if (gen_config.sampling == "top_p")
                    r = new TopPSampler(gen_config, gen_config.temperature, gen_config.top_k, gen_config.top_p);
                else if (gen_config.sampling == "tfs")
                    r = new FreeTailSampler(gen_config, gen_config.temperature, gen_config.top_k, gen_config.tfs_z);
                else if (gen_config.sampling != "greedy")
                    CHATLLM_CHECK(false) << "unknown sampling algorithm: " << gen_config.sampling;
            }

            if (nullptr == r)
                r = new GreedySampler();

            r->seed(seed);
            return r;
        }
    };

    BaseModelForConditionalGeneration::BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, const RuntimeConfig &runtime_config, size_t GRAPH_SIZE)
        : BaseModel(model_type, get_model_purpose(model_type)),
            transformer(nullptr),
            GRAPH_SIZE(GRAPH_SIZE),
            batch_input(runtime_config.batch_input_size), logit_scale(-1.0f),
            w_ctx_(&backend_context),
            config_(config)
    {
        w_ctx_.cache_dtype = runtime_config.cache_type;
        prepare(runtime_config);
        for (int i = 0; i < config.num_hidden_layers; i++)
            layer_ids.push_back(i);
    }

    void BaseModelForConditionalGeneration::set_layer_ids(const std::vector<int> &ids)
    {
        CHATLLM_CHECK((int)ids.size() == config_.num_hidden_layers) << "length(layer_ids) must be " << config_.num_hidden_layers;
        layer_ids.clear();
        for (auto x : ids)
            layer_ids.push_back(x);
    }

    int BaseModelForConditionalGeneration::get_max_length(void)
    {
        return config_.max_length;
    }

    void BaseModelForConditionalGeneration::shift_memory(int keep)
    {
        if (keep >= n_past) return;

        transformer->shift_cache(n_past - keep, n_past);
        BaseModel::shift_memory(keep);
    }

    int64_t BaseModelForConditionalGeneration::get_param_num(bool effective_only) const
    {
        return transformer->get_param_num(effective_only);
    }

    std::vector<int> BaseModelForConditionalGeneration::generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                const bool continuous,
                                bool &completed,
                                ModelPerfInfo *performance,
                                int gen_max_tokens,
                                BaseStreamer *streamer)
    {
        CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
            << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
            << config_.max_length << ")";

        //for (int i = 0; i < (int)input_ids.size(); i++)
        //    printf("%d, ", input_ids[i]);
        //printf("\nn_past = %d, %d\n\n", n_past, continuous);

        std::unique_ptr<Sampler> sampler = std::unique_ptr<Sampler>(SamplerFactory::Create(gen_config, _seed));

        aborted = false;

        std::vector<int> curr_input_ids(input_ids);

        std::vector<int> output_ids;
        output_ids.reserve(gen_config.max_length);

        if (!continuous)
        {
            n_past = 0;
            n_past_offset = 0;
        }

        completed = false;

        transformer->set_ctx((int)input_ids.size());
        int next_output_idx = 0;

        if (gen_max_tokens > 0)
            gen_max_tokens = n_past + (int)curr_input_ids.size() + gen_max_tokens;

        bool first_call = true;

        if (performance)
            performance->Reset();

        before_generate(gen_config);

        #if (0)
        for (auto i : curr_input_ids)
            printf("%d, ", i);
        printf("\n");
        #endif

        while (!aborted && !completed && (n_past + (int)curr_input_ids.size() < gen_config.max_length))
        {
            std::vector<float> lm_logits;
            if (!generate_next_token(curr_input_ids, gen_config, lm_logits))
            {
                ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
                aborted = true;
                break;
            }

            if (first_call)
            {
                if (performance)
                    performance->Accumulate(ModelPerfInfo::Type::Prompt, curr_input_ids.size());
                first_call = false;
            }

//#define DISABLE_CACHE
#ifndef DISABLE_CACHE
            n_past += (int)curr_input_ids.size();
            curr_input_ids.clear();
#endif
            float *logits = lm_logits.data();
            const size_t tok_num = lm_logits.size() / config_.vocab_size;

            for (size_t tok_idx = 0; (tok_idx < tok_num) && !aborted; tok_idx++, logits +=  config_.vocab_size)
            {
                int next_token_id = sampler->sampling(logits,  config_.vocab_size);

//printf("\n>>next = %d<<\n", next_token_id);
//fflush(stdout);
//exit(-1);

                if (next_token_id == Sampler::ABORT)
                {
                    aborted = true;
                    break;
                }

                curr_input_ids.push_back(next_token_id);

                int pop_output = 0;
                int keep_idx = 0;
                output_ids.push_back(next_token_id);

                if (is_output_terminated(output_ids, keep_idx, pop_output))
                {
                    while (pop_output-- > 0)
                        output_ids.pop_back();
                    keep_idx = (int)output_ids.size();
                    completed = true;
                }

                if (streamer)
                {
                    if (keep_idx > (int)output_ids.size())
                        keep_idx = (int)output_ids.size();
                    for (; next_output_idx < keep_idx; next_output_idx++)
                        streamer->put({output_ids[next_output_idx]});
                }

                if ((gen_max_tokens > 0) && ((n_past + (int)curr_input_ids.size() >= gen_max_tokens)))
                {
                    aborted = true;
                    break;
                }
            }
        }

        if (aborted && !completed)
            completed = true;

        if (performance)
            performance->Accumulate(ModelPerfInfo::Type::Generation, output_ids.size() - curr_input_ids.size());

        after_generate();

//printf("\nn_past = %d\n", n_past);
        return output_ids;
    }

    void BaseModelForConditionalGeneration::text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                std::vector<float> &embedding)
    {
        auto r = run_model(input_ids, gen_config, 0, embedding);
        if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
    }

    float BaseModelForConditionalGeneration::qa_rank(const GenerationConfig &gen_config, const std::vector<int> &input_ids)
    {
        std::vector<float> output;
        auto r = run_model(input_ids, gen_config, 0, output);
        if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
        CHATLLM_CHECK(output.size() == 1) << "ouput must be scaler";

        return output[0];
    }

    bool BaseModelForConditionalGeneration::generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits)
    {
        int batch = batch_input > 1 ? batch_input : 1;

        const int *p = input_ids.data();
        int remain = (int)input_ids.size();
        int past = n_past + n_past_offset;

        for (; (remain > batch) && !aborted; p += batch, remain -= batch, past += batch)
        {
            if (!run_model(p, batch, gen_config, past, lm_logits))
                return false;
        }

        return run_model(p, remain, gen_config,past, lm_logits);
    }

    int BaseModelForConditionalGeneration::save_session(FILE *f) const
    {
        int r = BaseModel::save_session(f);
        if (r != 0)
            return r;
        return transformer->save_session(f);
    }

    int BaseModelForConditionalGeneration::load_session(FILE *f)
    {
        int r = BaseModel::load_session(f);
        if (r != 0) return r;
        return transformer->load_session(f);
    }

    int BaseModelForConditionalGeneration::save_session(ModelSessionMemory &session) const
    {
        int r = BaseModel::save_session(session);
        if (r != 0)
            return r;
        return transformer->save_session(session);
    }

    int BaseModelForConditionalGeneration::load_session(ModelSessionMemory &session)
    {
        int r = BaseModel::load_session(session);
        if (r != 0) return r;
        return transformer->load_session(session);
    }

    void BaseModelForConditionalGeneration::prepare(const RuntimeConfig &rt_config)
    {
        w_ctx_.user_options.moe_on_cpu = rt_config.moe_on_cpu;
        backend_context.init(rt_config.model_gpu_layers, "main", config_.num_hidden_layers, GRAPH_SIZE, rt_config.n_threads);
    }

    LayerAllocatorManager *BaseModelForConditionalGeneration::get_alloc_manager(void)
    {
        return &backend_context.layer_allocators;
    }

    void BaseModelForConditionalGeneration::load(ModelLoader &loader)
    {
        transformer->load("model.", &loader, layer_ids);
    }

    void BaseModelForConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {}

    void BaseModelForConditionalGeneration::after_generate(void)
    {
        tokenizer->media_emb.clear();
    }

    void BaseModelForConditionalGeneration::do_build_graph(ForwardContext &ctc, const std::vector<int> &input_ids,
                                    const GenerationConfig &gen_config,
                                    int past)
    {

    }

    bool BaseModelForConditionalGeneration::before_initial_run(const int ids_count,
                                    const GenerationConfig &gen_config,
                                    int past)
    {
        //printf("before_initial_run 1\n");
        //backend_context.show_buffer_sizes();

        ForwardContext ctx(&backend_context);
        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        ggml::tensor *input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);

        ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::build_forward_expand(&ctx, r);

        bool s = ctx.reserve_memory();

        //printf("before_initial_run 2\n");
        //backend_context.show_buffer_sizes();

        return s;
    }

    bool BaseModelForConditionalGeneration::run_model(const std::vector<int> &input_ids,
        const GenerationConfig &gen_config,
        int past,
        std::vector<float> &output)
    {
        return run_model(input_ids.data(), (int)input_ids.size(), gen_config, past, output);
    }

    bool BaseModelForConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output)
    {
        if (!initial_run)
        {
            initial_run = true;
            int past = gen_config.max_length - ids_count;
            if (past < 0) past = 0;
            if (!before_initial_run(ids_count, gen_config, past))
                return false;
        }

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        dbg_ctx = &ctx;

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);

        ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

        output.resize(ggml::nbytes(r) / sizeof(output[0]));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, input_ids);

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        ctx.compute();

        Backend::read_tensor_data(r, output.data());

        ctx.reset();

        return true;
    }

    bool BaseModelForConditionalGeneration::is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output)
    {
        if (output_ids.size() < 1)
            return false;

        int last_tok_id = output_ids[output_ids.size() - 1];

        if (tokenizer->is_terminate_token_id(last_tok_id))
        {
            pop_output = 1;
            return true;
        }
        else
        {
            keep_idx = (int)output_ids.size();
            return false;
        }
    }

    bool BaseModelForConditionalGeneration::match_output_sequence(const std::vector<int> &output_ids, const std::vector<int> &pattern)
    {
        if (output_ids.size() < pattern.size())
            return false;

        auto x0 = output_ids.begin() + output_ids.size() - pattern.size();
        auto x1 = pattern.begin();

        for (size_t i = 0; i < pattern.size(); i++, x0++, x1++)
        {
            if (*x0 != *x1)
                return false;
        }
        return true;
    }

    static std::string regex_replace(const std::string &input, const std::regex &regex,
                                     std::function<std::string(const std::smatch &)> format)
    {
        std::ostringstream oss;
        size_t last_index = 0;
        for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++)
        {
            oss << it->prefix() << format(*it);
            last_index = it->position() + it->length();
        }
        oss << input.substr(last_index);
        return oss.str();
    }

    HeterogeneousModel::HeterogeneousModel(InitContext *ctx, int num_hidden_layers, int hidden_size,
        Block *word_embeddings, Block *final_layernorm,
        Block *lm_head, std::function<Block *(InitContext *, int)> create_layer)
    : num_hidden_layers(num_hidden_layers), hidden_size(hidden_size),
        word_embeddings(word_embeddings),
        final_layernorm(final_layernorm),
        lm_head(lm_head),
        logits_pp(nullptr),
        cache_size(0),
        final_steps(std::make_unique<LMFinalSteps>())
    {
        layers.reserve(num_hidden_layers);
        for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = create_layer(ctx, layer_id);
            layers.emplace_back(layer);

            layer->set_id(layer_id);
            cache_size += layer->get_cache_size();

            auto allocator = ctx->get_allocator();
            auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
            layer->set_cache_buffer(buf);
        }
    }

    HeterogeneousModel::~HeterogeneousModel()
    {
        if (word_embeddings) delete word_embeddings;
        if (final_layernorm) delete final_layernorm;
        if (lm_head) delete lm_head;
        for (auto b : layers) delete b;
    }

    ggml::tensor *HeterogeneousModel::forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past)
    {
        before_forward(ctx, input_ids, n_past);

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        ggml::tensor *hidden_states = word_embeddings->forward(ctx, input_ids);
        for (auto &layer : layers)
        {
            ctx->move_to_layer(layer->get_id());
            hidden_states = layer->forward(ctx, hidden_states, n_past);
        }

        ctx->move_to_layer(LayerAllocatorManager::Epilog);
        return final_steps->forward(this, ctx, input_ids, hidden_states);
    }

    void HeterogeneousModel::set_ctx(int n_ctx)
    {
        for (auto &layer : layers)
            layer->set_ctx(n_ctx);
    }

    void HeterogeneousModel::shift_cache(int shift, int total)
    {
        for (auto &layer : layers)
            layer->shift_cache(shift, total);
    }

    int64_t HeterogeneousModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += word_embeddings->get_param_num(effective_only);
        r += get_param_num_of_layers(effective_only);
        r += final_layernorm->get_param_num(effective_only);
        if (lm_head)
            r += lm_head->get_param_num(effective_only);
        if (logits_pp)
            r += logits_pp->get_param_num(effective_only);
        return r;
    }

    Block *HeterogeneousModel::get_layer(int index)
    {
        return layers[index];
    }

    void HeterogeneousModel::set_final_steps(std::unique_ptr<ModelFinalSteps> final_steps)
    {
        this->final_steps = std::move(final_steps);
    }

    ModelFinalSteps *HeterogeneousModel::get_final_steps()
    {
        return final_steps.get();
    }

    int HeterogeneousModel::save_session(FILE *f)
    {
        struct state state = {.cache_size = cache_size };
        if (fwrite(&state, sizeof(state), 1, f) != 1)
            return -1;

        std::vector<uint8_t> buffer;

        for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
        {
            auto layer = layers[layer_id];
            buffer.resize(layer->get_cache_size());
            size_t size = layer->read_cache_data(buffer.data(), buffer.size());
            if (size != buffer.size())
                return -4;
            if (fwrite(buffer.data(), 1, size, f) != size)
                return -3;
        }

        return 0;
    }

    int HeterogeneousModel::load_session(FILE *f)
    {
        struct state state = {0};
        if (fread(&state, sizeof(state), 1, f) != 1)
            return -10;
        if (state.cache_size != cache_size)
            return -1;

        std::vector<uint8_t> buffer;

        for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
        {
            auto layer = layers[layer_id];
            buffer.resize(layer->get_cache_size());
            if (fread(buffer.data(), 1, buffer.size(), f) != buffer.size())
                return -4;
            size_t size = layer->write_cache_data(buffer.data(), buffer.size());
            if (size != buffer.size())
                return -3;
        }

        return 0;
    }

    int HeterogeneousModel::save_session(ModelSessionMemory &session) const
    {
        for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
        {
            auto layer = layers[layer_id];
            const size_t size = layer->get_cache_size();
            void *buf = session.prepare_buffer(layer_id, size);
            if (layer->read_cache_data(buf, size) != size)
                return -1;
        }

        return 0;
    }

    int HeterogeneousModel::load_session(ModelSessionMemory &session)
    {
        for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
        {
            auto layer = layers[layer_id];
            size_t size = 0;
            void *buf = session.get_buffer(layer_id, &size);
            if (size != layer->get_cache_size()) return -1;
            if (layer->write_cache_data(buf, size) != size)
                return -3;
        }

        return 0;
    }

    void HeterogeneousModel::load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids)
    {
        word_embeddings->load(path + "embed_tokens.", loader);
        final_layernorm->load(path + "norm.", loader);
        if (lm_head)
            lm_head->load("lm_head.", loader);

        for (int i = 0; i < num_hidden_layers; i++)
        {
            std::string layer_prefix = path + "layers." + std::to_string(layer_ids[i]) + '.';
            layers[i]->load(layer_prefix, loader);
        }
    }

    int64_t HeterogeneousModel::get_param_num_of_layers(bool effective_only) const
    {
        int64_t r = 0;
        for (auto &layer : layers)
            r += layer->get_param_num(effective_only);
        return r;
    }

    ggml::tensor *LMFinalSteps::forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
    {
        hidden_states = ggml::view_2d(ctx, hidden_states, model->hidden_size, 1,
            ggml::row_size(hidden_states),
            (ggml::get_dim(input_ids, 0) - 1) * ggml::row_size(hidden_states));

        ggml::tensor *transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);

        transformer_outputs =
                ggml::view_1d(ctx, transformer_outputs, model->hidden_size, 0);

        ggml::tensor *lm_logits = model->lm_head ? model->lm_head->forward(ctx, transformer_outputs)
                                                 : model->word_embeddings->forward(ctx, transformer_outputs);

        if (model->logits_pp)
            lm_logits = model->logits_pp->forward(ctx, lm_logits);
        return lm_logits;
    }

    ggml::tensor *EmbeddingPoolingFinalSteps::forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
    {
        ggml::tensor *transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);

        return transformer_outputs;
    }

    ggml::tensor *EmbeddingLastTokenFinalSteps::forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
    {
        hidden_states = ggml::view_2d(ctx, hidden_states, model->hidden_size, 1,
            ggml::row_size(hidden_states),
            (ggml::get_dim(input_ids, 0) - 1) * ggml::row_size(hidden_states));
        ggml::tensor *transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);
        transformer_outputs = ggml::simple_norm(ctx, transformer_outputs, 1e-5f);
        return transformer_outputs;
    }

    Block *create_lm_head(InitContext *ctx, const BaseConfig &config, bool bias)
    {
        return new Linear(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.vocab_size, bias);
    }

    namespace glm
    {
        #include "../models/chatglm.cpp"
    }

    namespace codegeex
    {
        #include "../models/codegeex.cpp"
    }

    namespace internlm
    {
        #include "../models/internlm.cpp"
    }

    namespace codellama
    {
        #include "../models/codellama.cpp"
    }

    namespace deepseek
    {
        #include "../models/deepseek.cpp"
    }

    namespace deepseek_coder
    {
        #include "../models/deepseek_coder.cpp"
    }

    namespace baichuan
    {
        #include "../models/baichuan.cpp"
    }

    namespace yi
    {
        #include "../models/yi.cpp"
    }
    namespace phi
    {
        #include "../models/phi.cpp"
    }

    namespace mistral
    {
        #include "../models/mistral.cpp"
    }

    namespace openchat
    {
        #include "../models/openchat.cpp"
    }

    namespace starling
    {
        #include "../models/starling.cpp"
    }

    namespace wizard
    {
        #include "../models/wizard.cpp"
    }

    namespace tigerbot
    {
        #include "../models/tigerbot.cpp"
    }

    namespace bluelm
    {
        #include "../models/bluelm.cpp"
    }

    namespace dolphinphi2
    {
        #include "../models/dolphinphi2.cpp"
    }

    namespace stablelm
    {
        #include "../models/stablelm.cpp"
    }

    namespace neuralbeagle
    {
        #include "../models/neuralbeagle.cpp"
    }

    namespace bce
    {
        #include "../models/bce.cpp"
    }

    namespace bge
    {
        #include "../models/bge.cpp"
    }

    namespace orion
    {
        #include "../models/orion.cpp"
    }

    namespace minicpm
    {
        #include "../models/minicpm.cpp"
    }

    namespace adept
    {
        #include "../models/adept.cpp"
    }

    namespace gemma
    {
        #include "../models/gemma.cpp"
    }

    namespace codefuse
    {
        #include "../models/codefuse.cpp"
    }

    namespace characterglm
    {
        #include "../models/characterglm.cpp"
    }

    namespace cohere
    {
        #include "../models/cohere.cpp"
    }

    namespace grok
    {
        #include "../models/grok.cpp"
    }

    namespace zhinao
    {
        #include "../models/zhinao.cpp"
    }

    namespace starcoder
    {
        #include "../models/starcoder.cpp"
    }

    namespace m_a_p
    {
        #include "../models/m_a_p.cpp"
    }

    namespace xverse
    {
        #include "../models/xverse.cpp"
    }

    namespace index
    {
        #include "../models/index.cpp"
    }

    namespace numinamath
    {
        #include "../models/numinamath.cpp"
    }

    namespace groq
    {
        #include "../models/groq.cpp"
    }

    namespace allenai
    {
        #include "../models/allenai.cpp"
    }

    namespace alphageo
    {
        #include "../models/alphageo.cpp"
    }

    namespace granite
    {
        #include "../models/granite.cpp"
    }

    namespace megrez
    {
        #include "../models/megrez.cpp"
    }

    namespace falcon
    {
        #include "../models/falcon.cpp"
    }

    namespace telechat
    {
        #include "../models/telechat.cpp"
    }

    namespace jina
    {
        #include "../models/jina.cpp"
    }

    namespace moonshot
    {
        #include "../models/moonshot.cpp"
    }

    namespace instella
    {
        #include "../models/instella.cpp"
    }

    namespace reka
    {
        #include "../models/reka.cpp"
    }

    namespace hermes
    {
        #include "../models/hermes.cpp"
    }

    namespace decilm
    {
        #include "../models/decilm.cpp"
    }

    namespace solar
    {
        #include "../models/solar.cpp"
    }

    namespace gigachat
    {
        #include "../models/gigachat.cpp"
    }

    namespace aquila
    {
        #include "../models/aquila.cpp"
    }

    namespace bailing
    {
        #include "../models/bailing.cpp"
    }

    namespace kimi
    {
        #include "../models/kimi.cpp"
    }

    namespace apriel
    {
        #include "../models/apriel.cpp"
    }

    namespace orpheus
    {
        #include "../models/orpheus.cpp"
    }

    namespace oute
    {
        #include "../models/oute.cpp"
    }

    static void load_file_header(ModelLoader &loader)
    {
        // load magic
        loader.seek(0, SEEK_SET);
        std::string magic = loader.read_string(4);

        if (magic == "ggml")
        {
            loader.ff = ModelLoader::FileFormat::GGML;
        }
        else if (magic == "ggmm")
        {
            loader.ff = ModelLoader::FileFormat::GGMM;
            const uint32_t GGMM_VER = 1;
            uint32_t ver = loader.read_basic<uint32_t>();
            CHATLLM_CHECK(GGMM_VER == ver) << "GGMM file version error: " << ver;

            loader.ggml_header = loader.read_basic<ModelLoader::GGMMHeader>();
            if ((int64_t)loader.ggml_header.offset_config > loader.tell())
            {
                loader.meta = loader.read_string(loader.ggml_header.offset_config - loader.tell());
                size_t last_non_null = loader.meta.find_last_not_of('\0');
                if (last_non_null != std::string::npos)
                    loader.meta.erase(last_non_null + 1);
                else;

                loader.meta_json = json::JSON::Load(loader.meta);
                auto name = loader.meta_json["model_name"];
                auto native = loader.meta_json["model_native_name"];
                if (name.IsString())
                    loader.model_name        = name.ToString();
                if (native.IsString())
                    loader.model_native_name = native.ToString();
            }
            loader.seek(loader.ggml_header.offset_config, 0);
        }
        else
        {
            CHATLLM_CHECK(false) << "model file is broken (bad magic): " << magic;
        }

        loader.model_type = loader.read_basic<int>();
        loader.version = loader.read_basic<int>();

        if (loader.model_name.size() < 1)
            loader.model_name = to_string((ModelType(loader.model_type)));
        if (loader.model_native_name.size() < 1)
            loader.model_native_name = to_native_string((ModelType(loader.model_type)));
    }

    std::string ModelFactory::load_info(ModelLoader &loader)
    {
        load_file_header(loader);
        BaseConfig config = loader.read_basic<BaseConfig>();
        std::ostringstream oss;
        auto model_type = (ModelType)(loader.model_type);
        auto purpose = get_model_purpose(model_type);
        oss << "Model name  : " << loader.model_name;
        if (loader.model_native_name.size() > 0)
            oss << " (" << loader.model_native_name << ")";
        oss << " (" << std::hex << std::setw(8) << std::setfill('0') << model_type << ")" << std::dec;
        oss << std::endl;

        oss << "Model type  : " << to_string(purpose);
        if (ModelPurpose::Chat == purpose)
            oss << " {" << format_access_points(get_chat_model_access_points(model_type)) << "}";
        oss << std::endl;

        oss << "File version: " << loader.version << " (" << ModelLoader::ff_to_str(loader.ff) << ")" << std::endl
            << "Quantization: " << ggml::type_to_str(config.dtype) << std::endl;

        oss << std::endl
            << "vocab_size          : " << config.vocab_size << std::endl
            << "hidden_size         : " << config.hidden_size << std::endl
            << "num_attention_heads : " << config.num_attention_heads << std::endl
            << "num_hidden_layers   : " << config.num_hidden_layers << std::endl
            << "intermediate_size   : " << config.intermediate_size << std::endl
            << "max_length          : " << config.max_length << std::endl << std::endl

            << "bos_token_id        : " << config.bos_token_id << std::endl
            << "eos_token_id        : " << config.eos_token_id << std::endl
            << "pad_token_id        : " << config.pad_token_id << std::endl
            << "sep_token_id        : " << config.sep_token_id << std::endl;

            if (loader.meta.size() > 0)
            {
                ggml::log(GGML_LOG_LEVEL_INFO, "meta: %s", loader.meta.c_str());
            }

        return oss.str();
    }

    bool ModelFactory::load(ModelLoader &loader, Result &result, const ModelObject::extra_args &args)
    {
        load_file_header(loader);
        return ModelFactory::load(loader.model_type, loader.version, loader, result, args);
    }

    #define ALL_MODELS  \
        CASE(CHATGLM,               glm::v1, 1)                 \
        CASE(CHATGLM2,              glm::v2, 1)                 \
        CASE(CHATGLM3,              glm::v3, 1)                 \
        CASE(CODEGEEX2,             codegeex::v2, 1)            \
        CASE(CHARACTERGLM,          characterglm, 1)            \
        CASE(GLM4,                  glm::v4, 1)                 \
        CASE(CODEGEEX4,             codegeex::v4, 1)            \
        CASE(GLM4_0414,             glm::glm4_0414, 1)          \
                                                                \
        CASE(INTERNLM,              internlm::v1, 1)            \
        CASE(INTERNLM2,             internlm::v2, 1)            \
        CASE(INTERNLM2_1,           internlm::v2_1, 1)          \
        CASE(INTERNLM3,             internlm::v3, 1)            \
                                                                \
        CASE(LLAMA2,                llama::v2, 1)               \
        CASE(LLAMA3,                llama::v3, 1)               \
        CASE(CODELLAMA,             codellama, 1)               \
        CASE(LLAMA2PLUS,            llama::v2_plus, 1)          \
        CASE(LLAMA_MULTI,           llama::multi, 1)            \
        CASE(LLAMA3_1,              llama::v3_1, 1)             \
        CASE(LLAMA3_2,              llama::v3_2, 1)             \
        CASE(MEGREZ,                megrez::chat, 1)            \
        CASE(FALCON3,               falcon::v3, 1)              \
        CASE(REKA_FLASH3,           reka::flash, 1)             \
        CASE(DEEPSEEK_R1_DISTILL_LLAMA, llama::ds_r1_distill, 1)\
        CASE(LLAMA4,                llama::v4, 1)               \
                                                                \
        CASE(DEEPSEEK,              deepseek::v1, 1)            \
        CASE(DEEPSEEK_CODER,        deepseek_coder, 1)          \
        CASE(CODEFUSE_DEEPSEEK,     codefuse::deepseek, 1)      \
        CASE(NUMINAMATH,            numinamath, 1)              \
        CASE(DEEPSEEK_V2_LIGHT,     deepseek::v2_light, 1)      \
        CASE(DEEPSEEK_V2,           deepseek::v2, 1)            \
        CASE(DEEPSEEK_V3_LIGHT,     deepseek::v3_light, 1)      \
        CASE(DEEPSEEK_V1_MoE,       deepseek::v1_moe, 1)        \
        CASE(BAILINGMOE,            bailing::moe, 1)            \
        CASE(XVERSEMOE,             xverse::moe, 1)             \
                                                                \
        CASE(BAICHUANLLAMA,         baichuan::_7b, 1)           \
        CASE(BAICHUAN,              baichuan::larger, 1)        \
        CASE(BAICHUAN_M1,           baichuan::m1, 1)            \
                                                                \
        CASE(YI,                    yi, 1)                      \
        CASE(MAP_NEO,               m_a_p::neo, 1)              \
                                                                \
        CASE(PHI2,                  phi::v2::v1, 1)             \
        CASE(PHI2_V2,               phi::v2::v2, 1)             \
        CASE(PHI3,                  phi::v3, 1)                 \
        CASE(PHI3_SU,               phi::v3_su, 1)              \
        CASE(PHI3_SU2,              phi::v3_su2, 1)             \
        CASE(PHI3_SU3,              phi::v3_su3, 1)             \
        CASE(PHI3_MOE,              phi::v3_moe, 1)             \
        CASE(PHI4,                  phi::v4, 1)                 \
        CASE(PHI4_MINI,             phi::v4_mini, 1)            \
                                                                \
        CASE(WIZARDCODER,           wizard::coder, 1)           \
        CASE(WIZARDLM,              wizard::lm, 1)              \
        CASE(WIZARDMATH,            wizard::math, 1)            \
                                                                \
        CASE(MISTRAL,               mistral::mistral, 1)        \
        CASE(OPENCHAT,              openchat, 1)                \
        CASE(MIXTRAL,               mistral::mixtral, 1)        \
        CASE(MISTRAL2,              mistral::mistral2, 1)       \
        CASE(DEEPHERMES3_MISTRAL,   hermes::_mistral, 1)        \
                                                                \
        CASE(QWEN,                  qwen::v1, 2)                \
        CASE(QWEN2,                 qwen::v2, 1)                \
        CASE(QWEN2MoE,              qwen::v2_moe, 1)            \
        CASE(QWEN2TIE,              qwen::v2_tie, 1)            \
        CASE(MARCO_O1,              qwen::marco_o1, 1)          \
        CASE(QWQ,                   qwen::qwq, 1)               \
        CASE(READERLM2,             jina::readerlm, 1)          \
        CASE(DEEPSEEK_R1_DISTILL_QWEN, qwen::ds_r1_distill, 1)  \
        CASE(DEEPSEEK_R1_DISTILL_QWEN3,qwen::ds_r1_distill_v3, 1)\
        CASE(AQUILA2,               aquila::v2, 1)              \
        CASE(QWEN2_AUDIO,           qwen::v2_audio, 1)          \
        CASE(QWEN2_5_VL,            qwen::v2_5_vl, 1)           \
        CASE(QWEN3,                 qwen::v3, 1)                \
                                                                \
        CASE(TIGERBOT,              tigerbot, 1)                \
                                                                \
        CASE(BLUELM,                bluelm, 1)                  \
                                                                \
        CASE(DOLPHINPHI2,           dolphinphi2::v1, 1)         \
                                                                \
        CASE(STABLELM,              stablelm, 1)                \
                                                                \
        CASE(NEURALBEAGLE,          neuralbeagle, 1)            \
        CASE(STARLING,              starling, 1)                \
        CASE(WIZARDLM2_MOE,         wizard::moe, 1)             \
                                                                \
        CASE(ORION,                 orion, 1)                   \
                                                                \
        CASE(MINICPM,               minicpm::v1, 1)             \
        CASE(MINICPM2,              minicpm::v2, 1)             \
        CASE(MINICPM_MoE,           minicpm::moe, 1)            \
        CASE(MINICPM3,              minicpm::v3, 1)             \
        CASE(MINICPM4,              minicpm::v4, 1)             \
                                                                \
        CASE(PERSIMMON,             adept::persimmon, 1)        \
        CASE(FUYU,                  adept::fuyu, 1)             \
                                                                \
        CASE(GEMMA,                 gemma::v1, 1)               \
        CASE(GEMMA2,                gemma::v2, 2)               \
        CASE(GEMMA3,                gemma::v3, 1)               \
        CASE(GEMMA3Vis,             gemma::v3, 1)               \
                                                                \
        CASE(COHERE_COMMAND_R,      cohere::command_r, 1)       \
        CASE(COHERE_AYA_23,         cohere::aya_23, 1)          \
        CASE(COHERE_COMMAND_R7B,    cohere::v2, 1)              \
                                                                \
        CASE(GROK_1,                grok::v1, 1)                \
                                                                \
        CASE(ZHINAO,                zhinao, 1)                  \
                                                                \
        CASE(STARCODER2,            starcoder::v2, 1)           \
                                                                \
        CASE(XVERSE,                xverse::dense, 1)           \
                                                                \
        CASE(INDEX,                 index, 1)                   \
                                                                \
        CASE(SMOLLM,                smol::lm, 1)                \
        CASE(SMOL_VLM,              smol::vlm, 1)               \
        CASE(LLAMA3_GROQ_TOOL,      groq, 1)                    \
                                                                \
        CASE(OLMoE,                 allenai::moe, 1)            \
        CASE(OLMo2,                 allenai::dense, 1)          \
                                                                \
        CASE(ALPHAGEO_LM,           alphageo, 1)                \
                                                                \
        CASE(GRANITE_MoE,           granite::moe, 1)            \
        CASE(GRANITE,               granite::dense, 1)          \
                                                                \
        CASE(TELECHAT2,             telechat::v2, 1)            \
                                                                \
        CASE(HUNYUAN_DENSE,         hunyuan::dense, 1)          \
        CASE(HUNYUAN_MOE_V1,        hunyuan::moe_v1, 1)         \
                                                                \
        CASE(MOONLIGHT,             moonshot::moonlight, 1)     \
                                                                \
        CASE(INSTELLA,              instella, 1)                \
                                                                \
        CASE(DECILM,                decilm, 1)                  \
                                                                \
        CASE(SOLARPRO,              solar::pro, 1)              \
                                                                \
        CASE(GIGACHAT,              gigachat, 1)                \
                                                                \
        CASE(KIMI_VL,               kimi::vl, 1)                \
                                                                \
        CASE(APRIEL,                apriel, 1)                  \
                                                                \
        CASE(ERNIE_DENSE,           ernie::dense, 1)            \
        CASE(ERNIE_MOE,             ernie::moe, 1)              \
                                                                \
        CASE(PANGU_MOE,             pangu::moe, 1)              \
                                                                \
        CASE(SMOLLM3,               smol::lm3, 1)               \
                                                                \
        CASE(BCE_Embedding,         bce::embedding, 1)          \
        CASE(BCE_ReRanker,          bce::ranker, 1)             \
        CASE(BGE_M3,                bge::embedding, 1)          \
        CASE(BGE_ReRanker_M3,       bge::ranker, 1)             \
        CASE(MiniCPM_Embedding_Light,   minicpm::emb_light, 1)  \
        CASE(MiniCPM_ReRanker_Light,    minicpm::ranker_light, 1)\
        CASE(ORPHEUS_TTS,               orpheus::tts, 1)        \
        CASE(OUTE_TTS_LLAMA,            oute::tts_llama, 1)     \
        CASE(OUTE_TTS_QWEN3,            oute::tts_qwen3, 1)     \
        CASE(QWEN3_Embedding,           qwen::v3_emb, 1)        \
        CASE(QWEN3_ReRanker,            qwen::v3_ranker, 1)     \
        \

    class ModelLoadRegistry
    {
    public:
        static void reg(int model_type, BaseImplModelLoader *loader);
        static BaseImplModelLoader *get_loader(int model_type);
    protected:
        ModelLoadRegistry() {}
        ModelLoadRegistry(const ModelLoadRegistry&) = delete;
        static ModelLoadRegistry *get();
        std::unordered_map<int, BaseImplModelLoader *> loaders;
    };

    ModelLoadRegistry *ModelLoadRegistry::get()
    {
        static ModelLoadRegistry * obj = new ModelLoadRegistry();
        return obj;
    }

    void ModelLoadRegistry::reg(int model_type, BaseImplModelLoader *loader)
    {
        get()->loaders.insert(std::pair(model_type, loader));
    }

    BaseImplModelLoader *ModelLoadRegistry::get_loader(int model_type)
    {
        auto obj = get();
        auto v = obj->loaders.find(model_type);
        return v != obj->loaders.end() ? v->second : nullptr;
    }

    BaseImplModelLoader::BaseImplModelLoader(int model_type)
    {
        ModelLoadRegistry::reg(model_type, this);
    }

    AbstractModel *ModelFactory::load_model_again(ModelLoader &loader, const ModelObject::extra_args &args)
    {
        int model_type = loader.model_type;
        int version = loader.version;

        #define CASE(TYPE, ns, ver)         \
            case MODEL_TYPE_ ##TYPE:        \
            {                               \
                CHATLLM_CHECK(version == ver) << "only support version " #ver " for now but got " << version;   \
                return load_model<ns::Config,                                                                   \
                                  ns::ConditionalGeneration>(loader, args);                                     \
            }

        switch ((ModelType)model_type)
        {
        ALL_MODELS
        default:
            {
                auto _loader = ModelLoadRegistry::get_loader(model_type);
                CHATLLM_CHECK(_loader != nullptr) << "invalid model type " << model_type;
                return _loader->load_model(loader, args);
            }
        }

        #undef CASE
    }

    bool ModelFactory::load(int model_type, int version, ModelLoader &loader, Result &result, const ModelObject::extra_args &args)
    {
        #define CASE(TYPE, ns, ver)         \
            case MODEL_TYPE_ ##TYPE:        \
            {                               \
                CHATLLM_CHECK(version == ver) << "only support version " #ver " for now but got " << version;   \
                return load_model<ns::Config,                                                                   \
                                  ns::Tokenizer,                                                                \
                                  ns::ConditionalGeneration>(loader, result, args);                             \
            }

        switch ((ModelType)model_type)
        {
        ALL_MODELS
        default:
            {
                auto _loader = ModelLoadRegistry::get_loader(model_type);
                CHATLLM_CHECK(_loader != nullptr) << "invalid model type " << model_type;
                return _loader->load_model(loader, result, args);
            }
        }

        #undef CASE
    }

} // namespace chatllm
