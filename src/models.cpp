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

namespace chatllm
{
    struct RuntimeConfig
    {
        std::string gpu_layers;
        bool moe_on_cpu;
        int n_threads;
        ggml::type cache_type;
        RuntimeConfig(const std::string &gpu_layers, bool moe_on_cpu, int n_threads, ggml::type cache_type):
            gpu_layers(gpu_layers), moe_on_cpu(moe_on_cpu), n_threads(n_threads), cache_type(cache_type)
        {}
    };

    class ForwardContext : public ComputeContext
    {
    public:
        ForwardContext(BackendContext *backend_context) : ComputeContext(backend_context)
        {
        }

        struct ggml_context *get_ctx() override { return gctx.get(); }
        ggml_cgraph *get_cgraph(void) override { return gf; }

    public:
        GGMLContext gctx;
        ggml_cgraph *gf;
    };

    static ForwardContext *dbg_ctx = nullptr;
    static std::unordered_map<ggml::tensor *, std::string> inspected_set;
    static ggml::tensor *dbg_w = nullptr;

    void print_tensor_shape(const char *info, ggml::tensor *tensor)
    {
        printf("%s: shape of %s (%p): [%zd, %zd, %zd, %zd] [%zd, %zd, %zd, %zd]\n",
            info, tensor->name, tensor->data,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    }

    void print_tensor(ggml::tensor *tensor, int offset, const bool full)
    {
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
                    if (!full && ((50 < i) && (i < n - 50))) continue;
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
                for (size_t i = 0; i < ggml::nbytes(tensor) / sizeof(ggml_fp16_t); i++)
                {
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
        bool full = false;
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

    enum ModelType
    {
        MODEL_TYPE_CHATGLM  = 1,
        MODEL_TYPE_CHATGLM2 = 2,
        MODEL_TYPE_CHATGLM3 = 3,
        MODEL_TYPE_CODEGEEX2 = 4,
        MODEL_TYPE_CHARACTERGLM = 5,
        MODEL_TYPE_GLM4 = 6,
        MODEL_TYPE_CODEGEEX4 = 7,

        MODEL_TYPE_INTERNLM     = 0x100,
        MODEL_TYPE_INTERNLM2    = 0x101, // extended model, supporting 7B & 20B
        MODEL_TYPE_INTERNLM2_1  = 0x102,
        MODEL_TYPE_INTERNLM3    = 0x103,

        MODEL_TYPE_LLAMA2   = 0x150,
        MODEL_TYPE_CODELLAMA= 0x151,
        MODEL_TYPE_WIZARDCODER      = 0x152,
        MODEL_TYPE_WIZARDLM         = 0x153,
        MODEL_TYPE_WIZARDMATH       = 0x154,
        MODEL_TYPE_TIGERBOT         = 0x155,
        MODEL_TYPE_LLAMA2PLUS       = 0x156,
        MODEL_TYPE_MEGREZ           = 0x157,
        MODEL_TYPE_FALCON3          = 0x158,
        MODEL_TYPE_REKA_FLASH3      = 0x159,

        MODEL_TYPE_BAICHUANLLAMA    = 0x200,
        MODEL_TYPE_BAICHUAN         = 0x201,
        MODEL_TYPE_BAICHUAN_M1      = 0x202,

        MODEL_TYPE_DEEPSEEK = 0x300,
        MODEL_TYPE_DEEPSEEK_CODER   = 0x301,
        MODEL_TYPE_CODEFUSE_DEEPSEEK = 0x302,
        MODEL_TYPE_NUMINAMATH        = 0x303,
        MODEL_TYPE_DEEPSEEK_V2_LIGHT = 0x320,
        MODEL_TYPE_DEEPSEEK_V2       = 0x321,
        MODEL_TYPE_DEEPSEEK_V3_LIGHT = 0x322,   // DOES NOT EXIST
        MODEL_TYPE_DEEPSEEK_V3       = 0x323,

        MODEL_TYPE_YI       = 0x400,
        MODEL_TYPE_MAP_NEO  = 0x401,

        MODEL_TYPE_PHI2     = 0x500,
        MODEL_TYPE_PHI2_V2  = 0x501,
        MODEL_TYPE_PHI3     = 0x520,
        MODEL_TYPE_PHI3_SU  = 0x521,
        MODEL_TYPE_PHI3_SU2 = 0x522,
        MODEL_TYPE_PHI3_SU3 = 0x523,
        MODEL_TYPE_PHI3_MOE = 0x530,
        MODEL_TYPE_PHI4     = 0x531,
        MODEL_TYPE_PHI4_MINI= 0x532,

        MODEL_TYPE_DOLPHINPHI2      = 0x510,
        MODEL_TYPE_DOLPHINPHI2_V2   = 0x511,

        MODEL_TYPE_MISTRAL  = 0x600,
        MODEL_TYPE_MIXTRAL  = 0x601,
        MODEL_TYPE_OPENCHAT = 0x602,
        MODEL_TYPE_NEURALBEAGLE = 0x603,
        MODEL_TYPE_STARLING     = 0x604,
        MODEL_TYPE_WIZARDLM2_MOE  = 0x605,
        MODEL_TYPE_MISTRAL2       = 0x606,

        MODEL_TYPE_QWEN     = 0x700,
        MODEL_TYPE_QWEN2    = 0x710,
        MODEL_TYPE_QWEN2TIE = 0x711,
        MODEL_TYPE_QWEN2MoE = 0x750,
        MODEL_TYPE_MARCO_O1 = 0x751,
        MODEL_TYPE_QWQ      = 0x752,
        MODEL_TYPE_READERLM2= 0x753,

        MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN = 0x754,

        MODEL_TYPE_BLUELM   = 0x800,

        MODEL_TYPE_STABLELM = 0x900,

        MODEL_TYPE_ORION    = 0x1000,

        MODEL_TYPE_MINICPM  = 0x1100,
        MODEL_TYPE_MINICPM2 = 0x1101,
        MODEL_TYPE_MINICPM_MoE = 0x1102,
        MODEL_TYPE_MINICPM3 = 0x1110,

        MODEL_TYPE_PERSIMMON= 0x1200,
        MODEL_TYPE_FUYU     = 0x1201,

        MODEL_TYPE_GEMMA    = 0x1300,
        MODEL_TYPE_GEMMA2   = 0x1301,
        MODEL_TYPE_GEMMA3   = 0x1302,

        MODEL_TYPE_COHERE_COMMAND_R     = 0x1400,
        MODEL_TYPE_COHERE_AYA_23        = 0x1401,
        MODEL_TYPE_COHERE_COMMAND_R7B   = 0x1402,

        MODEL_TYPE_GROK_1           = 0x1500,

        MODEL_TYPE_ZHINAO           = 0x1600,

        MODEL_TYPE_LLAMA3           = 0x1700,
        MODEL_TYPE_SMOLLM           = 0x1701,
        MODEL_TYPE_LLAMA3_GROQ_TOOL = 0x1702,
        MODEL_TYPE_LLAMA3_1         = 0x1703,
        MODEL_TYPE_LLAMA3_2         = 0x1704,
        MODEL_TYPE_EXAONE           = 0x1705,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA = 0x1706,

        MODEL_TYPE_STARCODER2       = 0x1800,

        MODEL_TYPE_XVERSE           = 0x1900,

        MODEL_TYPE_INDEX            = 0x1a00,

        MODEL_TYPE_OLMoE            = 0x1b00,
        MODEL_TYPE_OLMo2            = 0x1b01,

        MODEL_TYPE_ALPHAGEO_LM      = 0x1c00,

        MODEL_TYPE_GRANITE_MoE      = 0x1d00,
        MODEL_TYPE_GRANITE          = 0x1d01,

        MODEL_TYPE_TELECHAT2        = 0x1e00,

        MODEL_TYPE_HUNYUAN_DENSE    = 0x1f00,

        MODEL_TYPE_MOONLIGHT        = 0x2000,

        MODEL_TYPE_INSTELLA         = 0x2100,

        MODEL_TYPE_BCE_Embedding = 0x10000100,
        MODEL_TYPE_BCE_ReRanker  = 0x10000101,
        MODEL_TYPE_BGE_M3        = 0x10000102,
        MODEL_TYPE_BGE_ReRanker_M3  = 0x10000103,
        MODEL_TYPE_MiniCPM_Embedding_Light  = 0x10000104,
        MODEL_TYPE_MiniCPM_ReRanker_Light   = 0x10000105,


        MODEL_TYPE_LLAMA_MULTI      = 0x20000001,
    };

    ModelPurpose get_model_purpose(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_BCE_Embedding:
        case MODEL_TYPE_BGE_M3:
        case MODEL_TYPE_MiniCPM_Embedding_Light:
            return ModelPurpose::TextEmbedding;
        case MODEL_TYPE_BCE_ReRanker:
        case MODEL_TYPE_BGE_ReRanker_M3:
        case MODEL_TYPE_MiniCPM_ReRanker_Light:
            return ModelPurpose::Ranker;
        default:
            return ModelPurpose::Chat;
        }
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
        default:
            CHATLLM_THROW << "unknown model purpose: " << purpose;
            return "???";
        }
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
        case MODEL_TYPE_MiniCPM_Embedding_Light:
            return "MiniCPM-Embedding-Light";
        case MODEL_TYPE_MiniCPM_ReRanker_Light:
            return "MiniCPM-ReRanker-Light";
        case MODEL_TYPE_MOONLIGHT:
            return "Moonlight";
        case MODEL_TYPE_INSTELLA:
            return "Instella";
        default:
            CHATLLM_THROW << "unknown model type: " << model_type;
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

    class Sampler
    {
    public:
        static const int ABORT = -1;

    public:
        virtual void seed(int x)
        {
            gen.seed((unsigned int)x);
        }

        virtual void reset() {}

        virtual int sampling(float *logits, const int vocab_size) = 0;
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
        NonGreedySampler(float temperature, float presence_penalty, int top_k)
            : inv_temp(0.0f), inv_presence_penalty(0.0f), presence_penalty(presence_penalty), top_k(top_k)
        {
            temp_en = fabs(temperature - 1.0f) > 1e-5f;
            if (temp_en) inv_temp = 1.f / temperature;

            presence_penalty_en = fabs(presence_penalty - 1.0f) > 1e-5f;
            if (presence_penalty_en) inv_presence_penalty = 1.0f / presence_penalty;
        }

        void reset() override
        {
            g.clear();
        }

        int sampling(float *logits, const int vocab_size) override
        {
            g.resize(vocab_size, 0);
            token_scores.resize(vocab_size);

            if (temp_en)
            {
                for (int i = 0; i < vocab_size; i++)
                    logits[i] *= inv_temp;
            }

            if (presence_penalty_en)
            {
                for (int i = 0; i < vocab_size; i++)
                {
                    if (g[i] > 0)
                        logits[i] *= logits[i] > 0 ? inv_presence_penalty : presence_penalty;
                }
            }

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

            g[next_token_id] += 1;
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
        bool presence_penalty_en;
        float inv_temp;
        float inv_presence_penalty;
        float presence_penalty;
        int top_k;
        std::vector<TokenIdScore> token_scores;
        std::vector<int> g;
    };

    class TopPSampler : public NonGreedySampler
    {
    public:
        TopPSampler(float temperature, float presence_penalty, int top_k, float top_p)
            : NonGreedySampler(temperature, presence_penalty, top_k), top_p(top_p)
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
        FreeTailSampler(float temperature, float presence_penalty, int top_k, float z)
            : NonGreedySampler(temperature, presence_penalty, top_k), z(z)
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
                    r = new TopPSampler(gen_config.temperature, gen_config.presence_penalty, gen_config.top_k, gen_config.top_p);
                else if (gen_config.sampling == "tfs")
                    r = new FreeTailSampler(gen_config.temperature, gen_config.presence_penalty, gen_config.top_k, gen_config.tfs_z);
                else if (gen_config.sampling != "greedy")
                    CHATLLM_CHECK(false) << "unknown sampling algorithm: " << gen_config.sampling;
            }

            if (nullptr == r)
                r = new GreedySampler();

            r->seed(seed);
            return r;
        }
    };

    class ModelBlock: public Block
    {
    public:
        virtual int save_session(FILE *f) = 0;
        virtual int load_session(FILE *f) = 0;

        virtual int save_session(ModelSessionMemory &session) const = 0;
        virtual int load_session(ModelSessionMemory &session) = 0;
    };

    static bool parse_gpu_cfg(BackendContext::gpu_cfg &cfg, const std::string &s)
    {
        cfg.id = 0;
        cfg.n_layers = -1;
        cfg.epilog = false;
        cfg.prolog = false;

        std::string t(s);

        size_t pos = t.find_first_of(':');
        std::string part = t.substr(0, pos);
        if (pos != std::string::npos)
        {
            cfg.id = atoi(part.c_str());
            t = t.substr(pos + 1);
        }

        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(',');
            part = t.substr(0, pos);

            if (part.size() > 0)
            {
                if (part.compare("all") == 0)
                {
                    cfg.prolog = true;
                    cfg.epilog = true;
                    cfg.n_layers = 99999;
                }
                else if (part.compare("prolog") == 0)
                    cfg.prolog = true;
                else if (part.compare("epilog") == 0)
                    cfg.epilog = true;
                else
                    cfg.n_layers = std::max(atoi(part.c_str()), 0);
            }

            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }

        return (cfg.n_layers >= 1) || cfg.prolog || cfg.epilog;
    }

    static int index_of_gpu_cfg(const std::vector<BackendContext::gpu_cfg> &gpu_cfgs, int id)
    {
        for (int i = 0; i < (int)gpu_cfgs.size(); i++)
            if (gpu_cfgs[i].id == id)
                return i;
        return -1;
    }

    static bool parse_gpu_layers(std::vector<BackendContext::gpu_cfg> &gpu_cfgs, const std::string &s)
    {
        std::string t(s);
        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(';');

            BackendContext::gpu_cfg cfg;
            if (parse_gpu_cfg(cfg, t.substr(0, pos)))
            {
                int index = index_of_gpu_cfg(gpu_cfgs, cfg.id);
                if (index >= 0)
                    gpu_cfgs[index].merge(cfg);
                else
                    gpu_cfgs.push_back(cfg);
            }

            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }
        return true;
    }

    class BaseModelForConditionalGeneration : public BaseModel
    {
    public:
        BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096)
            : BaseModel(model_type, to_string(model_type), to_native_string(model_type), get_model_purpose(model_type)),
              transformer(nullptr),
              GRAPH_SIZE(GRAPH_SIZE),
              batch_input(true), logit_scale(-1.0f),
              w_ctx_(&backend_context),
              config_(config)
        {
            w_ctx_.cache_dtype = runtime_config.cache_type;
            prepare(runtime_config);
            for (int i = 0; i < config.num_hidden_layers; i++)
                layer_ids.push_back(i);
        }

        virtual ~BaseModelForConditionalGeneration() = default;

        void set_layer_ids(const std::vector<int> &ids) override
        {
            CHATLLM_CHECK((int)ids.size() == config_.num_hidden_layers) << "length(layer_ids) must be " << config_.num_hidden_layers;
            layer_ids.clear();
            for (auto x : ids)
                layer_ids.push_back(x);
        }

        int get_max_length(void) override
        {
            return config_.max_length;
        }

        void shift_memory(int keep) override
        {
            if (keep >= n_past) return;

            transformer->shift_cache(n_past - keep, n_past);
            BaseModel::shift_memory(keep);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return transformer->get_param_num(effective_only);
        }

        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  int gen_max_tokens,
                                  BaseStreamer *streamer = nullptr)
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

//printf("\nn_past = %d\n", n_past);
            return output_ids;
        }

        void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding) override
        {
            auto r = run_model(input_ids, gen_config, 0, embedding);
            if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
        }

        float qa_rank(const GenerationConfig &gen_config, const std::vector<int> &input_ids) override
        {
            std::vector<float> output;
            auto r = run_model(input_ids, gen_config, 0, output);
            if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
            CHATLLM_CHECK(output.size() == 1) << "ouput must be scaler";

            return output[0];
        }

        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override
        {
            if (batch_input)
            {
                return run_model(input_ids, gen_config, n_past + n_past_offset, lm_logits);
            }
            else
            {
                int past = n_past + n_past_offset;
                for (size_t i = 0 ; (i < input_ids.size()) & !aborted; i++, past++)
                {
                    if (!run_model({input_ids[i]}, gen_config, past, lm_logits))
                        return false;
                }
                return true;
            }
        }

        int save_session(FILE *f) const override
        {
            int r = BaseModel::save_session(f);
            if (r != 0)
                return r;
            return transformer->save_session(f);
        }

        int load_session(FILE *f) override
        {
            int r = BaseModel::load_session(f);
            if (r != 0) return r;
            return transformer->load_session(f);
        }

        int save_session(ModelSessionMemory &session) const override
        {
            int r = BaseModel::save_session(session);
            if (r != 0)
                return r;
            return transformer->save_session(session);
        }

        int load_session(ModelSessionMemory &session) override
        {
            int r = BaseModel::load_session(session);
            if (r != 0) return r;
            return transformer->load_session(session);
        }

        void prepare(const RuntimeConfig &rt_config)
        {
            std::vector<BackendContext::gpu_cfg> gpu_cfgs;
            parse_gpu_layers(gpu_cfgs, rt_config.gpu_layers);
            w_ctx_.user_options.moe_on_cpu = rt_config.moe_on_cpu;
            backend_context.init(gpu_cfgs, config_.num_hidden_layers, GRAPH_SIZE, rt_config.n_threads);
        }

        LayerAllocatorManager *get_alloc_manager(void) override
        {
            return &backend_context.layer_allocators;
        }

    protected:
        virtual void do_build_graph(ForwardContext &ctc, const std::vector<int> &input_ids,
                                       const GenerationConfig &gen_config,
                                       int past)
        {

        }

        virtual bool before_initial_run(const std::vector<int> &input_ids,
                                       const GenerationConfig &gen_config,
                                       int past)
        {
            //printf("before_initial_run 1\n");
            //backend_context.show_buffer_sizes();

            ForwardContext ctx(&backend_context);
            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            ggml::tensor *input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, input_ids.size());

            ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);

            ggml::build_forward_expand(&ctx, r);

            bool s = ctx.reserve_memory();

            //printf("before_initial_run 2\n");
            //backend_context.show_buffer_sizes();

            return s;
        }

        virtual bool run_model(const std::vector<int> &input_ids,
                                       const GenerationConfig &gen_config,
                                       int past,
                                       std::vector<float> &output)
        {
            if (!initial_run)
            {
                initial_run = true;
                int past = gen_config.max_length - (int)input_ids.size();
                if (past < 0) past = 0;
                if (!before_initial_run(input_ids, gen_config, past))
                    return false;
            }

            ForwardContext ctx(&backend_context);
            ctx.user_options = w_ctx_.user_options;

            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            dbg_ctx = &ctx;

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
            ggml::tensor *input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, input_ids.size());

            ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);

            ggml::build_forward_expand(&ctx, r);

            CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

            output.resize(ggml::nbytes(r) / sizeof(output[0]));

            if (!ctx.allocate()) return false;

            Backend::write_tensor_data(input_ids_tensor, input_ids.data());

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

        virtual bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output)
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

        bool match_output_sequence(const std::vector<int> &output_ids, const std::vector<int> &pattern)
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

        template <class T> T *get_typed_transformer(void) const
        {
            return dynamic_cast<T *>(transformer);
        }

    protected:
        ModelBlock *transformer;
        const size_t GRAPH_SIZE;
        bool batch_input;
        float logit_scale;
        std::vector<int> layer_ids;
        BackendContext backend_context;
        InitContext w_ctx_; // weight context
    private:
        BaseConfig config_;
        bool initial_run = false;
    };

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

    template <class Config, class Embedding, class FinalNorm> class HeterogeneousModel : public ModelBlock
    {
    public:
        HeterogeneousModel() = default;

        HeterogeneousModel(InitContext *ctx, const Config &config, bool lm_head_bias, std::function<Block *(InitContext *, int)> create_layer)
                : HeterogeneousModel(ctx, config, new Linear(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.vocab_size, lm_head_bias), create_layer)
        {}

        HeterogeneousModel(InitContext *ctx, const Config &config, Block *lm_head, std::function<Block *(InitContext *, int)> create_layer)
        : config(config),
          word_embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config.vocab_size, config.hidden_size, config.max_length),
          final_layernorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
          lm_head(lm_head), logits_pp(nullptr),
          cache_size(0)
        {
            layers.reserve(config.num_hidden_layers);
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
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

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override
        {
            before_forward(ctx, input_ids, n_past);

            ctx->move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
            for (auto &layer : layers)
            {
                ctx->move_to_layer(layer->get_id());
                hidden_states = layer->forward(ctx, hidden_states, n_past);
            }

            ctx->move_to_layer(LayerAllocatorManager::Epilog);
            return final_steps(ctx, input_ids, hidden_states);
        }

        void set_ctx(int n_ctx) override
        {
            for (auto &layer : layers)
                layer->set_ctx(n_ctx);
        }

        void shift_cache(int shift, int total) override
        {
            for (auto &layer : layers)
                layer->shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += word_embeddings.get_param_num(effective_only);
            r += get_param_num_of_layers(effective_only);
            r += final_layernorm.get_param_num(effective_only);
            if (lm_head)
                r += lm_head->get_param_num(effective_only);
            if (logits_pp)
                r += logits_pp->get_param_num(effective_only);
            return r;
        }

        Block *get_layer(int index)
        {
            return layers[index];
        }

        int save_session(FILE *f) override
        {
            struct state state = {.cache_size = cache_size };
            if (fwrite(&state, sizeof(state), 1, f) != 1)
                return -1;

            std::vector<uint8_t> buffer;

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
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

        int load_session(FILE *f) override
        {
            struct state state = {0};
            if (fread(&state, sizeof(state), 1, f) != 1)
                return -10;
            if (state.cache_size != cache_size)
                return -1;

            std::vector<uint8_t> buffer;

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
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

        int save_session(ModelSessionMemory &session) const override
        {
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                auto layer = layers[layer_id];
                const size_t size = layer->get_cache_size();
                void *buf = session.prepare_buffer(layer_id, size);
                if (layer->read_cache_data(buf, size) != size)
                    return -1;
            }

            return 0;
        }

        int load_session(ModelSessionMemory &session) override
        {
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
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

    private:
        struct state
        {
            size_t cache_size;
        };
    protected:
        virtual void before_forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) {}

        virtual int64_t get_param_num_of_layers(bool effective_only) const
        {
            int64_t r = 0;
            for (auto &layer : layers)
                r += layer->get_param_num(effective_only);
            return r;
        }

        ggml::tensor *final_steps(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
        {
            hidden_states = ggml::view_2d(ctx, hidden_states, config.hidden_size, 1,
                config.hidden_size * ggml::element_size(hidden_states),
                (input_ids->ne[0] - 1) * config.hidden_size * ggml::element_size(hidden_states));

            ggml::tensor *transformer_outputs = final_layernorm.forward(ctx, hidden_states);

            transformer_outputs =
                    ggml::view_1d(ctx, transformer_outputs, config.hidden_size, 0);

            ggml::tensor *lm_logits = lm_head ? lm_head->forward(ctx, transformer_outputs)
                                             : word_embeddings.forward(ctx, transformer_outputs);

            if (logits_pp)
                lm_logits = logits_pp->forward(ctx, lm_logits);
            return lm_logits;
        }
    public:
        Config config;
        Embedding word_embeddings;
        FinalNorm final_layernorm;
        Block *lm_head;
        Block *logits_pp;
    protected:
        // std::vector<std::unique_ptr<Block>> layers;
        std::vector<Block *> layers;
        size_t cache_size;
    };

    template <class Config, class Embedding, class FinalNorm, class LayerBlock, typename... _Types> class Model :
        public HeterogeneousModel<Config, Embedding, FinalNorm>
    {
    private:
        typedef HeterogeneousModel<Config, Embedding, FinalNorm> Base;
    protected:
        class Accessor
        {
        friend Model;
        protected:
            Accessor() : m(nullptr) {}
        public:
            LayerBlock & operator[](int index)
            {
                if (nullptr == m)
                {
                    uintptr_t offset = (uintptr_t)&(((Model *)(nullptr))->layers);
                    m = (Model *)(uintptr_t(this) - offset);
                }
                return *(dynamic_cast<LayerBlock *>((m->Base::layers)[index])); // .get()));
            }
        private:
            Model *m;
        };
    public:
        Model() = default;
        Model(InitContext *ctx, const Config &config, bool lm_head_bias, _Types... layer_args)
            : Model(ctx, config, new Linear(ctx, config.hidden_size, config.vocab_size, lm_head_bias), std::forward<_Types>(layer_args)...)
        {}

        Model(InitContext *ctx, const Config &config, Block *lm_head, _Types... layer_args)
        : HeterogeneousModel<Config, Embedding, FinalNorm>(ctx, config, lm_head,
                            [&](InitContext *ctx, int layer_index) {
                                return new LayerBlock(ctx, std::forward<_Types>(layer_args)...);
                            })
        {
        }

    protected:
        int64_t get_param_num_of_layers(bool effective_only) const override
        {
            int64_t r = 0;
            if (Base::layers.size() > 0)
                r += Base::layers[0]->get_param_num(effective_only) * Base::layers.size();
            return r;
        }
    public:
        Accessor layers;
    };

    template <class Config, class Embedding, class LayerBlock, class FinalBlock, typename... _Types> class EmbeddingModel : public Model
        <Config, Embedding, FinalBlock, LayerBlock, _Types...>
    {
    public:
        typedef Model<Config, Embedding, FinalBlock, LayerBlock, _Types...> Base;
        typedef HeterogeneousModel<Config, Embedding, FinalBlock> BaseBase;

        EmbeddingModel() = default;

        EmbeddingModel(InitContext *ctx, const Config &config, _Types... layer_args)
        : Base(ctx, config, nullptr, std::forward<_Types>(layer_args)...)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override
        {
            Base::before_forward(ctx, input_ids, n_past);

            ctx->move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor *hidden_states = Base::word_embeddings.forward(ctx, input_ids, n_past);
            for (auto &layer : BaseBase::layers)
            {
                ctx->move_to_layer(layer->get_id());
                hidden_states = layer->forward(ctx, hidden_states, n_past);
            }
            ctx->move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
            return final_steps(ctx, input_ids, hidden_states);
        }

    protected:
        ggml::tensor *final_steps(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
        {
            ggml::tensor *transformer_outputs = Base::final_layernorm.forward(ctx, hidden_states);

            return transformer_outputs;
        }
    };

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

    namespace llama
    {
        #include "../models/llama.cpp"
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

    namespace qwen
    {
        #include "../models/qwen.cpp"
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

    namespace smollm
    {
        #include "../models/smollm.cpp"
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

    namespace exaone
    {
        #include "../models/exaone.cpp"
    }

    namespace telechat
    {
        #include "../models/telechat.cpp"
    }

    namespace jina
    {
        #include "../models/jina.cpp"
    }

    namespace hunyuan
    {
        #include "../models/hunyuan.cpp"
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

    template <class Config>
    void load_config(ModelLoader &loader, Config &config, const ModelObject::extra_args &args)
    {
        if (0 == loader.offset_config)
            loader.offset_config = loader.tell();
        else
            loader.seek(loader.offset_config, SEEK_SET);

        // load config
        config = loader.read_basic<Config>();
        if (args.max_length > 0)
            config.max_length = args.max_length;

        loader.offset_tokenizer = loader.tell();
    }

    template <class Config, class Tokenizer>
    Tokenizer *load_tokenizer(ModelLoader &loader, Config &config)
    {
        loader.seek(loader.offset_tokenizer, SEEK_SET);

        // load tokenizer
        Tokenizer *tokenizer = new Tokenizer(config);
        tokenizer->load(loader.get_reader(), config.vocab_size);
        loader.load_all_tensors();

        return tokenizer;
    }

    static void parse_slice(std::vector<int> &values, const std::string &s, int num_hidden_layers)
    {
        int spec[3] = {0, num_hidden_layers, 1};
        int index = 0;
        std::string t(s);
        if (t.size() > 0) index = 1;

        while ((t.size() > 0) && (index <= 3))
        {
            size_t pos = t.find_first_of(':');
            std::string part = t.substr(0, pos);
            if (part.size() > 0)
                spec[index - 1] = atoi(part.c_str());
            if (pos == std::string::npos) break;
            index++;
            t = t.substr(pos + 1);
        }

        if (index < 1) return;

        if (index == 1)
        {
            values.push_back(spec[0]);
            return;
        }

        if (spec[2] == 0) return;
        if (spec[0] < 0) spec[0] += num_hidden_layers;
        if (spec[1] < 0) spec[1] += num_hidden_layers;

        if (spec[2] > 0)
        {
            for (int i = spec[0]; i < spec[1]; i += spec[2])
            {
                values.push_back(i);
            }
        }
        else
        {
            for (int i = spec[0]; i > spec[1]; i += spec[2])
                values.push_back(i);
        }
    }

    static int parse_int_lists(std::vector<int> &values, const std::string &s, int num_hidden_layers)
    {
        const static std::regex r(R""([\r\n]+)"");
        std::string t(s);
        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(',');
            parse_slice(values, t.substr(0, pos), num_hidden_layers);
            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }
        return 0;
    }

    template <class Config, class ConditionalGeneration>
    ConditionalGeneration *load_model(ModelLoader &loader, Config &config, const ModelObject::extra_args &args)
    {
        std::vector<int> layers;
        if (args.layer_spec.size() > 0)
        {
            parse_int_lists(layers, args.layer_spec, config.num_hidden_layers);
            config.num_hidden_layers = (int)layers.size();
        }

        RuntimeConfig rt_config(args.gpu_layers, args.moe_on_cpu, args.n_threads, ggml::parse(args.cache_type));

        // load model
        ConditionalGeneration *model = new ConditionalGeneration(config, rt_config);
        if (layers.size() > 0)
            model->set_layer_ids(layers);
        loader.set_allocator_manager(model->get_alloc_manager());
        model->load(loader);

        return model;
    }

    template <class Config, class ConditionalGeneration>
    ConditionalGeneration *load_model(ModelLoader &loader, const ModelObject::extra_args &args)
    {
        Config config;

        load_config<Config>(loader, config, args);

        return load_model<Config, ConditionalGeneration>(loader, config, args);
    }

    template <class Config, class Tokenizer, class ConditionalGeneration>
    bool load_model(ModelLoader &loader, ModelFactory::Result &result, const ModelObject::extra_args &args)
    {
        // load config
        Config config;

        load_config<Config>(loader, config, args);

        // load tokenizer
        result.tokenizer = std::unique_ptr<BaseTokenizer>(load_tokenizer<Config, Tokenizer>(loader, config));

#if (0)
        // test tokenizer
        std::vector<int> ids = result.tokenizer->encode("\nAlice:");
        for (auto x : ids) std::cout << x << ", ";
        std::cout << std::endl;

        //ids = {0,1,2,195,196};
        std::cout << result.tokenizer->decode(ids) << std::endl;
        exit(-1);
#endif
        // load model
        result.model = std::unique_ptr<AbstractModel>(load_model<Config, ConditionalGeneration>(loader, config, args));

        result.model->set_tokenizer(result.tokenizer.get());

        return true;
    }

    static void load_file_header(ModelLoader &loader)
    {
        // load magic
        loader.seek(0, SEEK_SET);
        std::string magic = loader.read_string(4);
        CHATLLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

        loader.model_type = loader.read_basic<int>();
        loader.version = loader.read_basic<int>();
    }

    std::string ModelFactory::load_info(ModelLoader &loader)
    {
        load_file_header(loader);
        BaseConfig config = loader.read_basic<BaseConfig>();
        std::ostringstream oss;
        oss << "Model name  : " << to_string((ModelType(loader.model_type))) << std::endl
            << "Model type  : " << to_string(get_model_purpose((ModelType(loader.model_type)))) << std::endl
            << "File version: " << loader.version << std::endl << std::endl

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
        CASE(EXAONE,                exaone, 1)                  \
        CASE(DEEPSEEK_R1_DISTILL_LLAMA, llama::ds_r1_distill, 1)\
                                                                \
        CASE(DEEPSEEK,              deepseek::v1, 1)            \
        CASE(DEEPSEEK_CODER,        deepseek_coder, 1)          \
        CASE(CODEFUSE_DEEPSEEK,     codefuse::deepseek, 1)      \
        CASE(NUMINAMATH,            numinamath, 1)              \
        CASE(DEEPSEEK_V2_LIGHT,     deepseek::v2_light, 1)      \
        CASE(DEEPSEEK_V2,           deepseek::v2, 1)            \
        CASE(DEEPSEEK_V3_LIGHT,     deepseek::v3_light, 1)      \
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
                                                                \
        CASE(QWEN,                  qwen::v1, 2)                \
        CASE(QWEN2,                 qwen::v2, 1)                \
        CASE(QWEN2MoE,              qwen::v2_moe, 1)            \
        CASE(QWEN2TIE,              qwen::v2_tie, 1)            \
        CASE(MARCO_O1,              qwen::marco_o1, 1)          \
        CASE(QWQ,                   qwen::qwq, 1)               \
        CASE(READERLM2,             jina::readerlm, 1)          \
        CASE(DEEPSEEK_R1_DISTILL_QWEN, qwen::ds_r1_distill, 1)  \
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
                                                                \
        CASE(PERSIMMON,             adept::persimmon, 1)        \
                                                                \
        CASE(GEMMA,                 gemma::v1, 1)               \
        CASE(GEMMA2,                gemma::v2, 2)               \
        CASE(GEMMA3,                gemma::v3, 1)               \
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
        CASE(SMOLLM,                smollm, 1)                  \
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
                                                                \
        CASE(MOONLIGHT,             moonshot::moonlight, 1)     \
                                                                \
        CASE(INSTELLA,              instella, 1)                \
                                                                \
        CASE(BCE_Embedding,         bce::embedding, 1)          \
        CASE(BCE_ReRanker,          bce::ranker, 1)             \
        CASE(BGE_M3,                bge::embedding, 1)          \
        CASE(BGE_ReRanker_M3,       bge::ranker, 1)             \
        CASE(MiniCPM_Embedding_Light,   minicpm::emb_light, 1)  \
        CASE(MiniCPM_ReRanker_Light,    minicpm::ranker_light, 1)

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
            CHATLLM_THROW << "invalid model type " << model_type;
            return nullptr;
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
            CHATLLM_THROW << "invalid model type " << model_type;
            return false;
        }

        #undef CASE
    }

} // namespace chatllm
