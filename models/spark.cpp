#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::spark
{
    const int MAX_LAYERS = 128;

    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        int sliding_window;
        int tie_word_embeddings;
        int layer_is_swa[MAX_LAYERS];

        float full_partial_rotary_factor;
        float full_rope_theta;
        float swa_partial_rotary_factor;
        float swa_rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config) : BaseTokenizer::BaseTokenizer(config, &_chat_encoder)
        {
            sys_prompt = "you are a helpful assistant.";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
            {
                "\\p{N}{1,3}",
                "[一-龥ࠀ-一가-퟿]+",
                "[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+",
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+",
                " ?[\\p{P}\\p{S}]+",
                "[\r\n]",
                "\\s+(?!\\S)",
                "\\s+"
            });
            size_t size = tp->Load(buffer, n_vocab);
            tp->EnableReturnSpecialToken(true);
            return size;
        }
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        append_ai_opening(round_idx, ids);
        tokenizer->encode(ai, ids);
        ids.push_back(tokenizer->eos_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        ids.push_back(tokenizer->bos_token_id);
        oss_prompt << "|System|>\n" << tokenizer->get_system_prompt();\
        tokenizer->encode(oss_prompt.str(), ids);
        ids.push_back(tokenizer->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        ids.push_back(tokenizer->bos_token_id);
        oss_prompt << "|User|>" << user;
        tokenizer->encode(oss_prompt.str(), ids);
        ids.push_back(tokenizer->eos_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        ids.push_back(tokenizer->bos_token_id);
        tokenizer->encode("<|Bot|>", ids);
    }

    template <class BaseAtt> class GatedAttention : public RoPESelfAttention<BaseAtt>
    {
    public:
        typedef RoPESelfAttention<BaseAtt> Base;
        GatedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
            Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false),
            g_proj(ctx, hidden_size, num_attention_heads, false)
        {}

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += g_proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            g_proj.load(path + "g_proj.", loader);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            rt_gate = g_proj.forward(ctx, hidden_states);
            rt_gate = ggml::sigmoid(ctx, rt_gate);
            return Base::forward(ctx, hidden_states, n_past);
        }

        ggml::tensor *cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                    ggml::tensor *q, ggml::tensor *k, ggml::tensor *v) override
        {
            const int num_heads = (int)ggml::get_dim(rt_gate, 0);
            ggml::tensor *scores = Base::cross_attention(ctx, hidden_size, n_past, qlen, q, k, v);
            scores = ggml::reshape(ctx, scores, ggml::get_dim(scores, 0) / num_heads, num_heads,
                ggml::get_dim(scores, 1), ggml::get_dim(scores, 2));
            rt_gate = ggml::reshape(ctx, rt_gate, 1, num_heads, ggml::get_dim(rt_gate, 1), ggml::get_dim(rt_gate, 2));
            scores = ggml::mul(ctx, scores, rt_gate);
            scores = ggml::reshape(ctx, scores, ggml::get_dim(scores, 0) * num_heads,
                ggml::get_dim(scores, 2), ggml::get_dim(scores, 3));
            return scores;
        }

    protected:
        ggml::tensor *rt_gate = nullptr;

    public:
        Linear g_proj;
    };


    template <int sliding_window_len> class DenseSWABlock : public LMBlock1<RMSNorm,
        GatedAttention<SlidingWindowAttentionImpl<sliding_window_len>>, RMSNorm, GELUMLP>
    {
    public:
        typedef LMBlock1<RMSNorm, GatedAttention<SlidingWindowAttentionImpl<sliding_window_len>>, RMSNorm, GELUMLP> Base;
        DenseSWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
            int head_dim, int max_length,
            float partial_rotary_factor, float rope_theta)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {
            Base::attention.rope_dim = int(Base::attention.rope_dim * partial_rotary_factor);
            Base::attention.freq_base = rope_theta;
        }
    };

    class DenseFullBlock : public LMBlock1<RMSNorm, GatedAttention<BaseAttention>, RMSNorm, GELUMLP>
    {
    public:
        typedef LMBlock1<RMSNorm, GatedAttention<BaseAttention>, RMSNorm, GELUMLP> Base;
        DenseFullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads,
            int head_dim, int max_length,
            float partial_rotary_factor, float rope_theta)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {
            attention.rope_dim = int(attention.rope_dim * partial_rotary_factor);
            attention.freq_base = rope_theta;
        }
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_SPARK_2_5);
        void prepare(const std::vector<int> &input_ids, const GenerationConfig &gen_config, const bool continuous);

    protected:
        Block *create_layer(InitContext *ctx, int layer_index);
        int get_num_of_swa_layer(void) const;
    private:
        const Config config;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        BaseModelForConditionalGeneration(type, config, runtime_config),
        config(config)
    {
        const bool   tie_lm_head = config.tie_word_embeddings != 0;
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * 13 + get_num_of_swa_layer() * 1;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new HeterogeneousModel(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                                            create_embedding<Embedding>(&w_ctx_, config),
                                            create_final_norm<RMSNorm>(&w_ctx_, config),
                                            !tie_lm_head ? create_lm_head(&w_ctx_, config, false) : nullptr,
                                            [&](InitContext *ctx, int layer_index) {
                                                return create_layer(ctx, layer_index);
                                            });
        w_ctx_.check_used_mem_size(true, 0);
    }

    int ConditionalGeneration::get_num_of_swa_layer(void) const
    {
        int r = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
            r += config.layer_is_swa[i] != 0 ? 1 : 0;
        return r;
    }

    void ConditionalGeneration::prepare(const std::vector<int> &input_ids, const GenerationConfig &gen_config, const bool continuous)
    {
        switch (gen_config.enable_thinking)
        {
        case trilean::Default:  // default is true
        case trilean::True:
            tokenizer->ai_prefix = "<think>";
            break;
        default:
            tokenizer->ai_prefix = "<think></think>";
            break;
        }
    }

    Block *ConditionalGeneration::create_layer(InitContext *ctx, int layer_index)
    {
        if (config.layer_is_swa[layer_index] != 0)
        {
            switch (config.sliding_window)
            {
            case 512:
                return new DenseSWABlock<512>(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length,
                    config.swa_partial_rotary_factor, config.swa_rope_theta);

            default:
                CHATLLM_CHECK(false) << "unsupported sliding_window";
                return nullptr;
            }
        }
        else
        {
            return new DenseFullBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length,
                    config.full_partial_rotary_factor, config.full_rope_theta);
        }
    }

    REGISTER_MODEL_LOADER(SPARK_2_5,            spark, 1);
}