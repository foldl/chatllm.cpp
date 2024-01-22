#include "models.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
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

#include "layers.h"

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

namespace chatllm
{
    ForwardContext *dbg_ctx = nullptr;

    void print_tensor(ggml_tensor *tensor, int offset = 0)
    {
        printf("\n%s: [%ld, %ld, %ld] [%ld, %ld, %ld]\n", tensor->name, tensor->ne[0], tensor->ne[1], tensor->ne[2],
                                                                        tensor->nb[0], tensor->nb[1], tensor->nb[2]);
        switch (tensor->type)
        {
        case GGML_TYPE_F32:
            {
                float * p = (float *)tensor->data;
                for (size_t i = 0; i < ggml_nbytes(tensor) / sizeof(float); i++)
                {
                    printf("[%3d] = %.15e\n", (int)i, p[i]);
                }
            }
            break;

        default:
            {
                char * p = (char *)tensor->data;
                p += offset;
                for (size_t i = 0; i < ggml_nbytes(tensor); i++)
                {
                    if ((i & 0xf) == 0) printf("\n%05d: ", (int)i);
                    printf("%5d", p[i]);
                }
            }
            break;
        }

        printf("\n");
    }

    void inspect_tensor(ggml_tensor *tensor, const char *msg, ggml_tensor *temp1, ggml_tensor *temp2, ggml_tensor *temp3, ggml_tensor *temp4, ggml_tensor *temp5)
    {
        ggml_tensor *dup = ggml_dup(dbg_ctx->gctx.get(), tensor);
        ggml_build_forward_expand(dbg_ctx->gf, dup);
        ggml_graph_compute_with_ctx(dbg_ctx->gctx.get(), dbg_ctx->gf, 4);
        printf("%s:\n", msg);
        print_tensor(dup);

        #define CHECK_AND_PRINT(tt, msg) do { if (tt) { printf("\n--------------- %s ----------------------\n", msg); print_tensor(tt); } } while (0)

        CHECK_AND_PRINT(temp1, "1");
        CHECK_AND_PRINT(temp2, "2");
        CHECK_AND_PRINT(temp3, "3");
        CHECK_AND_PRINT(temp4, "4");
        CHECK_AND_PRINT(temp5, "5");

        exit(-3);
    }

    enum ModelType
    {
        MODEL_TYPE_CHATGLM  = 1,
        MODEL_TYPE_CHATGLM2 = 2,
        MODEL_TYPE_CHATGLM3 = 3,
        MODEL_TYPE_CODEGEEX2 = 4,

        MODEL_TYPE_INTERNLM = 0x100,
        MODEL_TYPE_INTERNLM2= 0x101, // extended model, supporting 7B & 20B

        MODEL_TYPE_LLAMA2   = 0x150,
        MODEL_TYPE_CODELLAMA= 0x151,
        MODEL_TYPE_WIZARDCODER      = 0x152,
        MODEL_TYPE_WIZARDLM         = 0x153,
        MODEL_TYPE_WIZARDMATH       = 0x154,
        MODEL_TYPE_TIGERBOT         = 0x155,

        MODEL_TYPE_BAICHUANLLAMA = 0x200,
        MODEL_TYPE_BAICHUAN      = 0x201,

        MODEL_TYPE_DEEPSEEK = 0x300,
        MODEL_TYPE_DEEPSEEK_CODER   = MODEL_TYPE_DEEPSEEK + 1,

        MODEL_TYPE_YI       = 0x400,

        MODEL_TYPE_PHI2     = 0x500,
        MODEL_TYPE_PHI2_V2  = 0x501,

        MODEL_TYPE_DOLPHINPHI2      = 0x510,
        MODEL_TYPE_DOLPHINPHI2_V2   = 0x511,

        MODEL_TYPE_MISTRAL  = 0x600,
        MODEL_TYPE_MIXTRAL  = 0x601,
        MODEL_TYPE_OPENCHAT = 0x602,
        MODEL_TYPE_NEURALBEAGLE = 0x603,

        MODEL_TYPE_QWEN     = 0x700,

        MODEL_TYPE_BLUELM   = 0x800,

        MODEL_TYPE_STABLELM = 0x900,
    };

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
        case MODEL_TYPE_CODEGEEX2:
            return "CodeGeeX2";
        case MODEL_TYPE_INTERNLM:
        case MODEL_TYPE_INTERNLM2:
            return "InternLM";
        case MODEL_TYPE_LLAMA2:
            return "LlaMa2";
        case MODEL_TYPE_CODELLAMA:
            return "CodeLlaMa";
        case MODEL_TYPE_BAICHUAN:
        case MODEL_TYPE_BAICHUANLLAMA:
            return "Baichuan";
        case MODEL_TYPE_DEEPSEEK:
            return "DeepSeek-LLM";
        case MODEL_TYPE_DEEPSEEK_CODER:
            return "DeepSeek-Coder";
        case MODEL_TYPE_YI:
            return "Yi";
        case MODEL_TYPE_PHI2:
        case MODEL_TYPE_PHI2_V2:
            return "Phi-2";
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
            return "Mistral";
        case MODEL_TYPE_MIXTRAL:
            return "Mixtral MoE";
        case MODEL_TYPE_OPENCHAT:
            return "OpenChat";
        case MODEL_TYPE_NEURALBEAGLE:
            return "NeuralBeagle";
        case MODEL_TYPE_QWEN:
            return "QWen";
        case MODEL_TYPE_TIGERBOT:
            return "TigerBot";
        case MODEL_TYPE_BLUELM:
            return "BlueLM";
        case MODEL_TYPE_STABLELM:
            return "StableLM";
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
            return "书生·浦语";
        case MODEL_TYPE_BAICHUAN:
        case MODEL_TYPE_BAICHUANLLAMA:
            return "百川";
        case MODEL_TYPE_PHI2:
        case MODEL_TYPE_PHI2_V2:
            return "Φ";
        case MODEL_TYPE_QWEN:
            return "通义千问";
        case MODEL_TYPE_TIGERBOT:
            return "虎博";
        case MODEL_TYPE_BLUELM:
            return "蓝心";
        case MODEL_TYPE_NEURALBEAGLE:
            return "🐶";
        default:
            return "";
        }
    }

    template<class LM> class BaseModelForConditionalGeneration : public BaseModel
    {
    public:
        BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, size_t mem_size, size_t scratch_size)
            : BaseModel(model_type, to_string(model_type), to_native_string(model_type)),
              GRAPH_SIZE(GGML_DEFAULT_GRAPH_SIZE),
              batch_input(true),
              config_(config), mem_size_(mem_size), mem_buffer_(new char[mem_size]),
              scratch_size_(scratch_size), scratch_buffer_(new char[scratch_size])
        {
        }

        virtual ~BaseModelForConditionalGeneration() = default;

        int get_max_length(void) override
        {
            return config_.max_length;
        }

        void shift_memory(int keep) override
        {
            if (keep >= n_past) return;

            transformer.shift_cache(n_past - keep, n_past);
            BaseModel::shift_memory(keep);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return transformer.get_param_num(effective_only);
        }

        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  BaseStreamer *streamer = nullptr)
        {
            CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
                << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
                << config_.max_length << ")";

            std::vector<int> curr_input_ids(input_ids);

            std::vector<int> output_ids;
            output_ids.reserve(gen_config.max_length);
            output_ids = input_ids;
            if (streamer)
                streamer->put(input_ids);

            if (!continuous) n_past = 0;
            completed = false;

            transformer.set_ctx(input_ids.size());
            int next_output_idx = input_ids.size();

            while (!completed && (n_past < gen_config.max_length))
            {
                int next_token_id = generate_next_token(curr_input_ids, gen_config);
//printf("\nnext = %d\n", next_token_id);

//#define DISABLE_CACHE
#ifndef DISABLE_CACHE
                n_past += curr_input_ids.size();
                curr_input_ids = {next_token_id};
#else
                curr_input_ids.push_back(next_token_id);
#endif

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
            }

            if (streamer)
                streamer->end();
//printf("\nn_past = %d\n", n_past);
            return output_ids;
        }

        void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding) override
        {
            ggml_tensor *lm = run_model(input_ids, gen_config, n_past + n_past_offset);
            embedding.resize(lm->ne[0]);
            memcpy(embedding.data(), lm->data, embedding.size() * sizeof(embedding[0]));
        }

        int generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config)
        {
            ggml_tensor *lm_logits = nullptr;

            if (batch_input)
            {
                lm_logits = run_model(input_ids, gen_config, n_past + n_past_offset);
            }
            else
            {
                int past = n_past + n_past_offset;
                for (size_t i = 0 ; i < input_ids.size(); i++, past++)
                    lm_logits = run_model({input_ids[i]}, gen_config, past);
            }

            int vocab_size = lm_logits->ne[0];
            float *next_token_logits = (float *)lm_logits->data;

            int next_token_id;

            if (!gen_config.do_sample)
            {
                // greedy search
                return std::max_element(next_token_logits, next_token_logits + vocab_size) - next_token_logits;
            }

            // temperature sampling
            float inv_temp = 1.f / gen_config.temperature;
            for (int i = 0; i < vocab_size; i++)
            {
                //if (i < 200)
                //    printf("%d: %.3f\n", i, next_token_logits[i]);
                next_token_logits[i] *= inv_temp;
            }

            std::vector<TokenIdScore> token_scores(vocab_size);
            for (int i = 0; i < vocab_size; i++)
            {
                token_scores[i] = {.id = i, .score = next_token_logits[i]};
            }

            // top_k sampling
            if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size())
            {
                std::nth_element(token_scores.begin(), token_scores.begin() + gen_config.top_k, token_scores.end(),
                                std::greater<TokenIdScore>());
                token_scores.resize(gen_config.top_k);
            }

            // top_p sampling
            if (0.f < gen_config.top_p && gen_config.top_p < 1.f)
            {
                std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>()); // hot code!
                sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());

                float cumsum = 0.f;
                for (size_t i = 0; i < token_scores.size(); i++)
                {
                    cumsum += token_scores[i].score;
                    if (cumsum >= gen_config.top_p)
                    {
                        token_scores.resize(i + 1);
                        break;
                    }
                }
            }

            // sample next token
            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
            for (size_t i = 0; i < token_scores.size(); i++)
            {
                next_token_logits[i] = token_scores[i].score;
            }

            std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
            next_token_id = token_scores[dist(gen)].id;

            return next_token_id;
        }

        struct TokenIdScore
        {
            int id;
            float score;

            bool operator<(const TokenIdScore &other) const { return score < other.score; }
            bool operator>(const TokenIdScore &other) const { return score > other.score; }
        };

    protected:
        virtual ggml_tensor *run_model(const std::vector<int> &input_ids,
                                       const GenerationConfig &gen_config,
                                       int past)
        {
            ForwardContext ctx;
            ctx.gctx = GGMLContext({.mem_size = mem_size_, .mem_buffer = mem_buffer_.get(), .no_alloc = false});
            ctx.scratch = {.offs = 0, .size = scratch_size_, .data = scratch_buffer_.get()};
            int n_threads = input_ids.size() >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : gen_config.num_threads;
            ctx.gf = ggml_new_graph_custom(ctx.gctx.get(), GRAPH_SIZE, false);

            dbg_ctx = &ctx;

            ggml_tensor *input_ids_tensor = ggml_new_tensor_1d(ctx.gctx.get(), GGML_TYPE_I32, input_ids.size());
            memcpy(input_ids_tensor->data, input_ids.data(), ggml_nbytes(input_ids_tensor));

            ggml_tensor *r = transformer.forward(&ctx, input_ids_tensor, past);

            ggml_build_forward_expand(ctx.gf, r);
            ggml_graph_compute_with_ctx(ctx.gctx.get(), ctx.gf, n_threads);

#ifdef GGML_PERF
            ggml_graph_print(&ctx.gf);
#endif

            return r;
        }

        virtual bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output)
        {
            if (output_ids.size() < 1)
                return false;

            int last_tok_id = output_ids[output_ids.size() - 1];

            if (last_tok_id == terminate_token_id)
            {
                pop_output = 1;
                return true;
            }
            else if (last_tok_id == tokenizer->eos_token_id)
            {
                pop_output = 1;
                return true;
            }
            else
            {
                keep_idx = output_ids.size();
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

    private:
        static void sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last)
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

    protected:
        LM transformer;
        size_t GRAPH_SIZE;
        bool batch_input;
    private:
        BaseConfig config_;
        size_t mem_size_;
        std::unique_ptr<char[]> mem_buffer_; // BLAS buffer
        size_t scratch_size_;
        std::unique_ptr<char[]> scratch_buffer_; // intermediate tensor buffer
    };

    static std::string regex_replace(const std::string &input, const std::regex &regex,
                                     std::function<std::string(const std::smatch &)> format)
    {
        std::ostringstream oss;
        int last_index = 0;
        for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++)
        {
            oss << it->prefix() << format(*it);
            last_index = it->position() + it->length();
        }
        oss << input.substr(last_index);
        return oss.str();
    }

    template <class Config, class Embedding, class FinalNorm, class LayerBlock, typename... _Types> class Model : public Block
    {
    public:
        Model() = default;
        Model(InitContext *ctx, const Config &config, bool lm_head_bias, _Types... layer_args)
            : Model(ctx, config, new Linear(ctx, config.hidden_size, config.vocab_size, lm_head_bias), std::forward<_Types>(layer_args)...)
        {}

        Model(InitContext *ctx, const Config &config, Block *lm_head, _Types... layer_args)
        : config(config),
          word_embeddings(ctx, config.vocab_size, config.hidden_size),
          final_layernorm(ctx, config.hidden_size),
          lm_head(lm_head)
        {
            layers.reserve(config.num_hidden_layers);
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                layers.emplace_back(ctx, std::forward<_Types>(layer_args)...);
                layers[layer_id].set_id(layer_id);
            }
        }

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past) override
        {
            ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
            for (auto &layer : layers)
            {
                ggml_set_scratch(ctx->gctx.get(), ctx->scratch);
                hidden_states = layer.forward(ctx, hidden_states, n_past);
            }

            return final_steps(ctx, input_ids, hidden_states);
        }

        void set_ctx(int n_ctx) override
        {
            for (auto &layer : layers)
                layer.set_ctx(n_ctx);
        };

        void shift_cache(int shift, int total) override
        {
            for (auto &layer : layers)
                layer.shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += word_embeddings.get_param_num(effective_only);
            if (layers.size() > 0)
                r += layers[0].get_param_num(effective_only) * layers.size();
            r += final_layernorm.get_param_num(effective_only);
            if (lm_head)
                r += lm_head->get_param_num(effective_only);
            return r;
        }

    protected:
        ggml_tensor *final_steps(ForwardContext *ctx, ggml_tensor *input_ids, ggml_tensor *hidden_states)
        {
            ggml_set_scratch(ctx->gctx.get(), {.offs = 0, .size = 0, .data = nullptr});

            // NOTE: only compute next_token_logits for the last token
            hidden_states = ggml_view_2d(ctx->gctx.get(), hidden_states, config.hidden_size, 1,
                                        config.hidden_size * ggml_element_size(hidden_states),
                                        (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(hidden_states));

            ggml_tensor *transformer_outputs = final_layernorm.forward(ctx, hidden_states);

            transformer_outputs =
                    ggml_view_1d(ctx->gctx.get(), transformer_outputs, config.hidden_size, 0);

            ggml_tensor *lm_logits = lm_head ? lm_head->forward(ctx, transformer_outputs)
                                             : word_embeddings.forward(ctx, transformer_outputs);
            return lm_logits;
        }
    public:
        Config config;
        Embedding word_embeddings;
        std::vector<LayerBlock> layers;
        FinalNorm final_layernorm;
        Block *lm_head;
    };

    namespace glm
    {
        namespace v1
        {
            #include "models/chatglm_v1.cpp"
        }

        namespace v2
        {
            #include "models/chatglm_v2.cpp"
        }

        namespace v3
        {
            #include "models/chatglm_v3.cpp"
        }
    }

    namespace codegeex
    {
        namespace v2
        {
            #include "models/codegeex_v2.cpp"
        }
    }

    namespace internlm
    {
        #include "models/internlm.cpp"
    }

    namespace llama
    {
        #include "models/llama.cpp"
    }

    namespace codellama
    {
        #include "models/codellama.cpp"
    }

    namespace deepseek
    {
        #include "models/deepseek.cpp"
    }

    namespace deepseek_coder
    {
        #include "models/deepseek_coder.cpp"
    }

    namespace baichuan
    {
        #include "models/baichuan.cpp"
    }

    namespace yi
    {
        #include "models/yi.cpp"
    }
    namespace phi2
    {
        #include "models/phi2.cpp"
    }

    namespace mistral
    {
        #include "models/mistral.cpp"
    }

    namespace wizard
    {
        #include "models/wizard.cpp"
    }

    namespace openchat
    {
        #include "models/openchat.cpp"
    }

    namespace mixtral
    {
        #include "models/mixtral.cpp"
    }

    namespace qwen
    {
        #include "models/qwen.cpp"
    }

    namespace tigerbot
    {
        #include "models/tigerbot.cpp"
    }

    namespace bluelm
    {
        #include "models/bluelm.cpp"
    }

    namespace dolphinphi2
    {
        #include "models/dolphinphi2.cpp"
    }

    namespace stablelm
    {
        #include "models/stablelm.cpp"
    }

    namespace neuralbeagle
    {
        #include "models/neuralbeagle.cpp"
    }

    template <class Config, class Tokenizer, class ConditionalGeneration>
    bool load_model(ModelLoader &loader, ModelFactory::Result &result)
    {
        // load config
        Config config = loader.read_basic<Config>();

        // load tokenizer
        result.tokenizer = std::make_unique<Tokenizer>(config);
        size_t proto_size = result.tokenizer->load(loader.data + loader.tell(), config.vocab_size);

        loader.seek(proto_size, SEEK_CUR);

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
        result.model = std::make_unique<ConditionalGeneration>(config);
        result.model->load(loader);

        return true;
    }

    bool ModelFactory::load(int model_type, int version, ModelLoader &loader, Result &result)
    {
        switch ((ModelType)model_type)
        {
        case MODEL_TYPE_CHATGLM:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<glm::v1::Config,
                              glm::v1::Tokenizer,
                              glm::v1::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_CHATGLM2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<glm::v2::Config,
                              glm::v2::Tokenizer,
                              glm::v2::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_CHATGLM3:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<glm::v3::Config,
                              glm::v3::Tokenizer,
                              glm::v3::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_CODEGEEX2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<codegeex::v2::Config,
                              codegeex::v2::Tokenizer,
                              codegeex::v2::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_INTERNLM:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<internlm::v1::Config,
                              internlm::v1::Tokenizer,
                              internlm::v1::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_INTERNLM2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<internlm::v2::Config,
                              internlm::v2::Tokenizer,
                              internlm::v2::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_LLAMA2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<llama::Config,
                              llama::Tokenizer,
                              llama::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_CODELLAMA:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<codellama::Config,
                              codellama::Tokenizer,
                              codellama::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_DEEPSEEK:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<deepseek::Config,
                              deepseek::Tokenizer,
                              deepseek::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_DEEPSEEK_CODER:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<deepseek_coder::Config,
                              deepseek_coder::Tokenizer,
                              deepseek_coder::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_BAICHUANLLAMA:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<baichuan::_7b::Config,
                              baichuan::_7b::Tokenizer,
                              baichuan::_7b::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_BAICHUAN:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<baichuan::larger::Config,
                              baichuan::larger::Tokenizer,
                              baichuan::larger::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_YI:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<yi::Config,
                              yi::Tokenizer,
                              yi::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_PHI2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<phi2::v1::Config,
                              phi2::v1::Tokenizer,
                              phi2::v1::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_PHI2_V2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<phi2::v2::Config,
                              phi2::v2::Tokenizer,
                              phi2::v2::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_WIZARDCODER:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<wizard::coder::Config,
                              wizard::coder::Tokenizer,
                              wizard::coder::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_WIZARDLM:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<wizard::lm::Config,
                              wizard::lm::Tokenizer,
                              wizard::lm::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_WIZARDMATH:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<wizard::math::Config,
                              wizard::math::Tokenizer,
                              wizard::math::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_MISTRAL:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<mistral::Config,
                              mistral::Tokenizer,
                              mistral::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_OPENCHAT:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<openchat::Config,
                              openchat::Tokenizer,
                              openchat::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_MIXTRAL:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<mixtral::Config,
                              mixtral::Tokenizer,
                              mixtral::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_QWEN:
        {
            CHATLLM_CHECK(version == 2) << "only support version 2 for now but got " << version;

            return load_model<qwen::Config,
                              qwen::Tokenizer,
                              qwen::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_TIGERBOT:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<tigerbot::Config,
                              tigerbot::Tokenizer,
                              tigerbot::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_BLUELM:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<bluelm::Config,
                              bluelm::Tokenizer,
                              bluelm::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_DOLPHINPHI2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<dolphinphi2::v1::Config,
                              dolphinphi2::v1::Tokenizer,
                              dolphinphi2::v1::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_STABLELM:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<stablelm::Config,
                              stablelm::Tokenizer,
                              stablelm::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_NEURALBEAGLE:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<neuralbeagle::Config,
                              neuralbeagle::Tokenizer,
                              neuralbeagle::ConditionalGeneration>(loader, result);
        }
        default:
            CHATLLM_THROW << "invalid model type " << model_type;
            return false;
        }
    }

} // namespace chatllm
