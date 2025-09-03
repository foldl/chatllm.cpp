#include "../src/models.h"
#include "../src/models_priv.h"
#include "llama.h"

namespace chatllm::apertus
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_theta;
        int   rope_scaling_original_max_position_embeddings;
        float rope_scaling_factor;
        float rope_scaling_low_freq_factor;
        float rope_scaling_high_freq_factor;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : Tokenizer(config, &_chat_encoder)

        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr)
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder),
              enable_thinking(false)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    public:
        void encode(std::vector<int> &ids, const std::string &content, int leading_tok, int trailing_token) const;

    public:
        int system_token_id;
        int end_system_token_id;
        int developer_token_id;
        int end_developer_token_id;
        int user_token_id;
        int end_user_token_id;
        int assistant_token_id;
        int end_assistant_token_id;
        int inner_token_id;
        int outer_token_id;
        int tool_calls_token_id;
        int end_tool_calls_token_id;
        int nl_token_id;
    public:
        bool enable_thinking;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
                {
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                }
            );
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        system_token_id         = tp->PieceToId("<|system_start|>");
        end_system_token_id     = tp->PieceToId("<|system_end|>");
        developer_token_id      = tp->PieceToId("<|developer_start|>");
        end_developer_token_id  = tp->PieceToId("<|developer_end|>");
        user_token_id           = tp->PieceToId("<|user_start|>");
        end_user_token_id       = tp->PieceToId("<|user_end|>");
        assistant_token_id      = tp->PieceToId("<|assistant_start|>");
        end_assistant_token_id  = tp->PieceToId("<|assistant_end|>");
        inner_token_id          = tp->PieceToId("<|inner_prefix|>");
        outer_token_id          = tp->PieceToId("<|inner_suffix|>");
        tool_calls_token_id     = tp->PieceToId("<|tools_prefix|>");
        end_tool_calls_token_id = tp->PieceToId("<|tools_suffix|>");

        terminate_ids.insert(end_assistant_token_id);

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        return size;
    }

    void Tokenizer::encode(std::vector<int> &ids, const std::string &content, int leading_tok, int trailing_token) const
    {
        if (leading_tok >= 0)
            ids.push_back(leading_tok);
        BaseTokenizer::encode(content, ids);
        if (trailing_token >= 0)
            ids.push_back(trailing_token);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);

        auto s = tok->get_system_prompt();
        if (s.size() == 0)
        {
            s = "You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: " + utils::now();
            tok->encode(ids, s, tok->system_token_id, tok->end_system_token_id);
            s = "Deliberation: " + (tok->enable_thinking ? std::string("enabled\n") : "disabled\n")
               + "Tool Capabilities: disabled";
            tok->encode(ids, s, tok->developer_token_id, tok->end_developer_token_id);
        }
        else
        {
            if (!s._Starts_with("<|system_start|>"))
                ids.push_back(tok->system_token_id);
            tokenizer->encode(s, ids);
            if (s.find("<|system_end|>") == std::string::npos)
                ids.push_back(tok->end_system_token_id);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode(ids, ai, tok->assistant_token_id, tok->end_assistant_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode(ids, user, tok->user_token_id, tok->end_user_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->user_token_id);
    }

    class SelfAttention : public QKNormedAttention<RMSNorm, BaseAttention>
    {
    public:
        SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<RMSNorm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
            ctx->get_allocator()->alloc(freq_factors);
        }
    };

    class XIEMLP : public Block
    {
    public:
        XIEMLP() = default;
        XIEMLP(InitContext *ctx, int hidden_size, int intermediate_size)
            : XIEMLP(ctx, hidden_size, intermediate_size, false, false)
        {}

        XIEMLP(InitContext *ctx, int hidden_size, int intermediate_size,
               bool down_use_bias, bool up_use_bias)
            : down_proj(ctx, intermediate_size, hidden_size, down_use_bias),
                up_proj(ctx, hidden_size, intermediate_size, up_use_bias),
              alpha_n(0.8f), alpha_p(0.8f), beta(0.5f), eps(-1e-6f)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 2;
            r += down_proj.get_param_num(effective_only);
            r += up_proj.get_param_num(effective_only);
            return r;
        }

        void set_prec(ggml::prec prec) override
        {
            Block::set_prec(prec);
            down_proj.set_prec(prec);
        }

        void load(const std::string &path, TensorLoader *loader) override;

    public:
        Linear down_proj;
        Linear up_proj;
        float  alpha_n;
        float  alpha_p;
        float  beta;
        float  eps;
    };

    static float softplus(float input, float beta=1.0f, float threshold=20.0f)
    {
        if (input * beta > threshold) return input;
        return (1/beta) * std::logf(1 + std::expf(beta * input));
    }

    void XIEMLP::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);
        up_proj.load(path + "up_proj.", loader);
        down_proj.load(path + "down_proj.", loader);
        loader->read_scaler(path + "act_fn.alpha_n", &alpha_n);
        loader->read_scaler(path + "act_fn.alpha_p", &alpha_p);
        loader->read_scaler(path + "act_fn.beta",    &beta);
        loader->read_scaler(path + "act_fn.eps",     &eps);
        alpha_p = softplus(alpha_p);
        alpha_n = beta + softplus(alpha_n);
    }

    ggml::tensor *XIEMLP::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *x = up_proj.forward(ctx, hidden_states);
        x = ggml::xielu(ctx, x, alpha_n, alpha_p, beta, eps);
        ggml::tensor *output = down_proj.forward(ctx, x);
        return output;
    }

    class LayerBlock : public LMBlock1<RMSNorm, SelfAttention, RMSNorm, XIEMLP>
    {
    public:
        LayerBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, LayerBlock, int, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_APERTUS);

        void set_additional_args(const std::map<std::string, std::string> &args) override;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * 14;
        const size_t ctx_size = num_tensors * tensor_ovhd;

        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, false,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads,
                                        config.max_length);

        std::vector<float> freq_factors_value;
        llama::v3_1::init_llama3_freq_factors(freq_factors_value, config.hidden_size / config.num_attention_heads,
                                    config.rope_theta,
                                    config.rope_scaling_factor,
                                    config.rope_scaling_low_freq_factor,
                                    config.rope_scaling_high_freq_factor,
                                    config.rope_scaling_original_max_position_embeddings);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base = config.rope_theta;
            Backend::write_tensor_data(layer.attention.freq_factors, freq_factors_value.data());
        }

        w_ctx_.check_used_mem_size(true);
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->enable_thinking            = utils::get_opt(args, "enable_thinking", tok->enable_thinking);
    }

    REGISTER_MODEL_LOADER(APERTUS,              apertus, 1);
}