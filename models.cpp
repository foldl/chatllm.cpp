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

    enum ModelType
    {
        MODEL_TYPE_CHATGLM  = 1,
        MODEL_TYPE_CHATGLM2 = 2,
        MODEL_TYPE_INTERNLM = 0x100,
        MODEL_TYPE_LLAMA2   = 0x150,
        MODEL_TYPE_BAICHUAN = 0x200,
    };

    std::string to_string(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_CHATGLM:
            return "ChatGLM";
        case MODEL_TYPE_CHATGLM2:
            return "ChatGLM2";
        case MODEL_TYPE_INTERNLM:
            return "InternLM";
        case MODEL_TYPE_LLAMA2:
            return "LlaMa2";
        default:
            CHATLLM_THROW << "unknown model type: " << model_type;
            return "???";
        }
    }

    template<class LM> class BaseModelForConditionalGeneration : public BaseModel
    {
    public:
        BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, size_t mem_size, size_t scratch_size)
            : BaseModel(model_type, to_string(model_type)), config_(config), mem_size_(mem_size), mem_buffer_(new char[mem_size]),
              scratch_size_(scratch_size), scratch_buffer_(new char[scratch_size])
        {
            n_past = 0;
        }

        virtual ~BaseModelForConditionalGeneration() = default;

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

            while ((int)output_ids.size() < gen_config.max_length)
            {
                int next_token_id = generate_next_token(curr_input_ids, gen_config, n_past);

                if (next_token_id == terminate_token_id)
                {
                    completed = true;
                    break;
                }

                n_past += curr_input_ids.size();
                curr_input_ids = {next_token_id};
                output_ids.emplace_back(next_token_id);

                if (streamer)
                    streamer->put({next_token_id});

                if (next_token_id == config_.eos_token_id)
                {
                    completed = true;
                    break;
                }
            }

            if (streamer)
                streamer->end();

            return output_ids;
        }

        int generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, int n_past) const
        {
            ForwardContext ctx;
            ctx.gctx = GGMLContext({.mem_size = mem_size_, .mem_buffer = mem_buffer_.get(), .no_alloc = false});
            ctx.scratch = {.offs = 0, .size = scratch_size_, .data = scratch_buffer_.get()};
            int n_threads = input_ids.size() >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : gen_config.num_threads;
            ctx.gf = ggml_new_graph(ctx.gctx.get());

            ggml_tensor *input_ids_tensor = ggml_new_tensor_1d(ctx.gctx.get(), GGML_TYPE_I32, input_ids.size());
            memcpy(input_ids_tensor->data, input_ids.data(), ggml_nbytes(input_ids_tensor));

            ggml_tensor *lm_logits = transformer.forward(&ctx, input_ids_tensor, n_past);

            ggml_build_forward_expand(ctx.gf, lm_logits);
            ggml_graph_compute_with_ctx(ctx.gctx.get(), ctx.gf, n_threads);

    #ifdef GGML_PERF
            ggml_graph_print(&ctx.gf);
    #endif

            int vocab_size = lm_logits->ne[0];
            float *next_token_logits = (float *)lm_logits->data;

            int next_token_id;
            if (gen_config.do_sample)
            {
                // temperature sampling
                float inv_temp = 1.f / gen_config.temperature;
                for (int i = 0; i < vocab_size; i++)
                {
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

                thread_local std::random_device rd;
                thread_local std::mt19937 gen(rd());

                std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
                next_token_id = token_scores[dist(gen)].id;
            }
            else
            {
                // greedy search
                next_token_id = std::max_element(next_token_logits, next_token_logits + vocab_size) - next_token_logits;
            }

            return next_token_id;
        }

        struct TokenIdScore
        {
            int id;
            float score;

            bool operator<(const TokenIdScore &other) const { return score < other.score; }
            bool operator>(const TokenIdScore &other) const { return score > other.score; }
        };

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
    private:
        int n_past;
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

    template <class Config, class FinalNorm, class LayerBlock, typename... _Types> class Model : public Block
    {
    public:
        Model() = default;
        Model(InitContext *ctx, const Config &config, bool shared_head, _Types... layer_args)
        : config(config),
          word_embeddings(ctx, config.vocab_size, config.hidden_size),
          final_layernorm(ctx, config.hidden_size),
          shared_head(shared_head)
        {
            lm_head = shared_head ? Linear() : Linear(ctx, config.hidden_size, config.vocab_size, false),

            layers.reserve(config.num_hidden_layers);
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                layers.emplace_back(ctx, std::forward<_Types>(layer_args)...);
            }
        }

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past) const override
        {
            ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
            for (const auto &layer : layers)
            {
                ggml_set_scratch(ctx->gctx.get(), ctx->scratch);
                hidden_states = layer.forward(ctx, hidden_states, n_past);
            }

            return final_steps(ctx, input_ids, hidden_states);
        }

        void set_ctx(int n_ctx) const override
        {
            for (const auto &layer : layers)
                layer.set_ctx(n_ctx);
        };

    protected:
        ggml_tensor *final_steps(ForwardContext *ctx, ggml_tensor *input_ids, ggml_tensor *hidden_states) const
        {
            ggml_set_scratch(ctx->gctx.get(), {.offs = 0, .size = 0, .data = nullptr});
            ggml_tensor *transformer_outputs = final_layernorm.forward(ctx, hidden_states);
            // NOTE: only compute next_token_logits for the last token
            if (input_ids->ne[0] > 1)
            {
                transformer_outputs =
                    ggml_view_1d(ctx->gctx.get(), transformer_outputs, config.hidden_size,
                                    (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs));
            }

            ggml_tensor *lm_logits = shared_head ?
                ggml_mul_mat(ctx->gctx.get(), word_embeddings.weight, transformer_outputs)
                :
                lm_head.forward(ctx, transformer_outputs);
            return lm_logits;
        }
    public:
        Config config;
        Embedding word_embeddings;
        std::vector<LayerBlock> layers;
        FinalNorm final_layernorm;
        Linear lm_head;
    private:
        bool shared_head;
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

    }

    namespace internlm
    {
        #include "models/internlm.cpp"
    }

    namespace llama
    {
        #include "models/llama.cpp"
    }

    template <class Config, class Tokenizer, class ConditionalGeneration>
    bool load_model(ModelLoader &loader, ModelFactory::Result &result)
    {
        // load config
        Config config = loader.read_basic<Config>();

        // load tokenizer
        result.tokenizer = std::make_unique<Tokenizer>();
        size_t proto_size = result.tokenizer->load(loader.data + loader.tell(), config.vocab_size);
        loader.seek(proto_size, SEEK_CUR);

        if (0)
        {
            std::vector<int> ids = result.tokenizer->encode("你好呀？");
            for (auto x : ids)
                printf("%d ", x);
            printf("\n%s\n", result.tokenizer->decode(ids).c_str());

            exit(-1);
        }

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
        case MODEL_TYPE_INTERNLM:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<internlm::Config,
                              internlm::Tokenizer,
                              internlm::ConditionalGeneration>(loader, result);
        }
        case MODEL_TYPE_LLAMA2:
        {
            CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            return load_model<llama::Config,
                              llama::Tokenizer,
                              llama::ConditionalGeneration>(loader, result);
        }
        default:
            CHATLLM_THROW << "invalid model type " << model_type;
            return false;
        }
    }

} // namespace chatllm
