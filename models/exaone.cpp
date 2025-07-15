#include "../src/models.h"
#include "../src/models_priv.h"
#include "llama.h"

namespace chatllm::exaone::v1
{
    typedef llama::v3_2::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
                : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "You are EXAONE model from LG AI Research, a helpful assistant.";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);
            tp->EnableReturnSpecialToken(true);
            return size;
        }
    };

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        oss << "[|system|]" << tok->get_system_prompt() << "[|endofturn|]\n";
        tok->encode(oss.str(), ids);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        oss << "[|assistant|]" << ai << "[|endofturn|]\n";
        tok->encode(oss.str(), ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        oss << "[|user|]" << user << "\n";
        tok->encode(oss.str(), ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode("[|assistant|]", ids);
    }

    class ConditionalGeneration : public llama::v3_2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3_2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_EXAONE)
        {}
    };

    static ImplModelLoader<MODEL_TYPE_EXAONE, Config, Tokenizer, ConditionalGeneration> _loader;
}

namespace chatllm::exaone::v4
{
    const int MAX_LAYERS = 128;

    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int sliding_window;
        int tie_word_embeddings;

        float rope_theta;
        int   rope_scaling_original_max_position_embeddings;
        float rope_scaling_factor;
        float rope_scaling_low_freq_factor;
        float rope_scaling_high_freq_factor;

        int   is_sliding_window[MAX_LAYERS];
    };

    class Params
    {
    public:
        static bool use_rope;
    };

    bool Params::use_rope = true;

    class ExaoneSelfAttention : public QKNormedAttention<RMSNorm, BaseAttention>
    {
    public:
        ExaoneSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : QKNormedAttention<RMSNorm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
            use_rope = Params::use_rope;
            if (use_rope)
            {
                freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size / num_attention_heads / 2);
                ctx->get_allocator()->alloc(freq_factors);
            }
        }
    };

    template <int sliding_window_len> class ExaoneSWASelfAttention : public QKNormedAttention<RMSNorm, SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        ExaoneSWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : QKNormedAttention<RMSNorm, SlidingWindowAttentionImpl<sliding_window_len>>
                (ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
            typedef QKNormedAttention<RMSNorm, SlidingWindowAttentionImpl<sliding_window_len>> Base;
            Base::freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size / num_attention_heads / 2);
            ctx->get_allocator()->alloc(Base::freq_factors);
        }
    };

    template <int sliding_window_len> class ExaoneSWABlock : public LMBlock4<Identity, ExaoneSWASelfAttention<sliding_window_len>, RMSNorm, Identity, SiLUMLP, RMSNorm>
    {
    public:
        ExaoneSWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock4<Identity, ExaoneSWASelfAttention<sliding_window_len>, RMSNorm, Identity, SiLUMLP, RMSNorm>
                    (ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    typedef LMBlock4<Identity, ExaoneSelfAttention, RMSNorm, Identity, SiLUMLP, RMSNorm> ExaoneBlock;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_EXAONE4, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            size_t num_tensors = 2 + (config.tie_word_embeddings ? 0 : 1);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_sliding(i))
                    num_tensors += 16;
                else
                    num_tensors += (config.sliding_window <= 0) ? 15 : 14;

            }
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            std::vector<float> freq_factors_value;
            llama::v3_1::init_llama3_freq_factors(freq_factors_value, config.hidden_size / config.num_attention_heads,
                                        config.rope_theta,
                                        config.rope_scaling_factor,
                                        config.rope_scaling_low_freq_factor,
                                        config.rope_scaling_high_freq_factor,
                                        config.rope_scaling_original_max_position_embeddings);

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
            {
                if (is_sliding(layer_index))
                {
                    Params::use_rope = true;
                    batch_input = false;

                    switch (config.sliding_window)
                    {
                    case 4096:
                        {
                            auto block = new ExaoneSWABlock<4096>(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                    config.num_key_value_heads, config.max_length);
                            block->attention.freq_base    = config.rope_theta;
                            Backend::write_tensor_data(block->attention.freq_factors, freq_factors_value.data());
                            return block;
                        }
                    default:
                        CHATLLM_CHECK(false);
                        return nullptr;
                    }
                }
                else
                {
                    Params::use_rope = config.sliding_window <= 0;

                    auto block = new ExaoneBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                    if (block->attention.use_rope)
                    {
                        block->attention.freq_base    = config.rope_theta;
                        Backend::write_tensor_data(block->attention.freq_factors, freq_factors_value.data());
                    }
                    return block;
                }
            };

            transformer = new HeterogeneousModel(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                        create_embedding<Embedding>(&w_ctx_, config),
                        create_final_norm<RMSNorm>(&w_ctx_, config),
                        config.tie_word_embeddings ? nullptr : create_lm_head(&w_ctx_, config), create_layer);

            w_ctx_.check_used_mem_size(true);
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".post_mlp_layernorm.",     ".post_feedforward_layernorm."}
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    private:
        bool is_sliding(int layer_index) const
        {
            return config.is_sliding_window[layer_index];
        }
    public:
        Config config;
    };

    static ImplModelLoader<MODEL_TYPE_EXAONE4, Config, v1::Tokenizer, ConditionalGeneration> _loader;
}