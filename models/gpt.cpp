#include "../src/models.h"
#include "../src/models_priv.h"

#define MODEL_TYPE_GPT_OSS  (MODEL_TYPE_OPENAI + 0)

namespace chatllm::gpt_oss
{
    const int MAX_LAYERS = 128;

    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        int experts_per_token;
        int num_experts_per_tok;
        int num_local_experts;
        int sliding_window;
        int layer_type[MAX_LAYERS];

        float router_aux_loss_coef;
        float swiglu_limit;

        float rope_theta;
        float original_max_position_embeddings;
        float beta_fast;
        float beta_slow;
        float factor;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config) : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer(config, encoder)
        {
            sys_prompt = R"""(You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: )""" + utils::now() + R"""(

Reasoning: medium
)""";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+",
                    "\\s+(?!\\S)|\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            start_token_id      = tp->PieceToId("<|start|>");
            message_token_id    = tp->PieceToId("<|message|>");
            end_token_id        = tp->PieceToId("<|end|>");
            channel_token_id    = tp->PieceToId("<|channel|>");
            return_token_id     = tp->PieceToId("<|return|>");
            bos_token_id        = tp->PieceToId("<|startoftext|>");

            terminate_ids.insert(end_token_id);
            terminate_ids.insert(return_token_id);

            return size;
        }

        void encode_item(const char *tag, std::vector<int> &ids)
        {
            ids.push_back(start_token_id);
            encode(std::string(tag), ids);
            ids.push_back(message_token_id);
        }

        void encode_item(const char *tag, const std::string &content, std::vector<int> &ids)
        {
            encode_item(tag, ids);
            encode(content, ids);
            ids.push_back(end_token_id);
        }
    public:
        int start_token_id;
        int message_token_id;
        int end_token_id;
        int channel_token_id;
        int return_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("assistant", ai, ids);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("system", tok->get_system_prompt(), ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("user", user, ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("user", ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("assistant", ids);
    }

    template <class Base> class OssSelfAttention : public Base
    {
    public:
        OssSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, true, true)
        {
            Base::rope_mode = RoPEMode::Original;
        }
    };

    typedef OssSelfAttention<RoPESelfAttention<BaseAttention>> OssFullSelfAttention;

    template <int sliding_window_len> class OssSWASelfAttention : public OssSelfAttention<RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>>
    {
    public:
        typedef OssSelfAttention<RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>> Base;
        OssSWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class OssSparseMoE : public BaseSparseMLP
    {
    public:
        OssSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false, false, false, true, true),
              limit(7.0f), alpha(1.702f)
        {
            norm_topk_prob = false;
        }
    public:
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states)
        {
            const int64_t qlen        = hidden_states->ne[1];
            const int n_expert = num_local_experts;

            ggml::tensor * logits = gate.forward(ctx, hidden_states); // [qlen, num_experts]

            CPUMover mover(ctx, ctx->user_options.moe_on_cpu);

            // select experts
            ggml::tensor * selected_experts = nullptr;

            selected_experts = ggml::top_k(ctx, logits, num_experts_per_tok); // [qlen, num_experts_per_tok]

            ggml::tensor * weights = ggml::get_rows(ctx,
                ggml::reshape_3d(ctx, logits, 1, n_expert, qlen), selected_experts); // [1, num_experts_per_tok, qlen]

            weights = ggml::reshape_2d(ctx, weights, num_experts_per_tok, qlen);

            weights = ggml::soft_max(ctx, weights); // [qlen, num_experts]

            weights = ggml::reshape_3d(ctx, weights, 1, num_experts_per_tok, qlen);

            return forward_with_experts(ctx, hidden_states, selected_experts, weights);
        }

        ggml::tensor *calc_experts_outputs(ComputeContext *ctx, ggml::tensor *hidden_states,
            ggml::tensor *selected_experts) override
        {
            ggml::tensor *gated = experts.gate.forward(ctx, hidden_states, selected_experts); // [n_ff, num_experts_per_tok, qlen]
            ggml::tensor *up = experts.up.forward(ctx, hidden_states, selected_experts); // [n_ff, num_experts_per_tok, qlen]

            ggml::tensor *par = ggml::swiglu_oai(ctx, gated, up, alpha, limit); // [n_ff, num_experts_per_tok, qlen]

            ggml::tensor * experts_out = experts.down.forward(ctx, par, selected_experts); // [hidden_size, num_experts_per_tok, qlen]
            return experts_out;
        }
    private:
        const float limit;
        const float alpha;
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int sliding_window_len> class OssSWABlock :
        public LMBlock1<RMSNorm, OssSWASelfAttention<sliding_window_len>, RMSNorm, OssSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
    {
    public:
        OssSWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1<RMSNorm, OssSWASelfAttention<sliding_window_len>, RMSNorm, OssSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
                    (ctx, hidden_size, num_attention_heads, intermediate_size,
                     num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class OssFullBlock :
        public LMBlock1<RMSNorm, OssFullSelfAttention, RMSNorm, OssSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
    {
    public:
        OssFullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1<RMSNorm, OssFullSelfAttention, RMSNorm, OssSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
                    (ctx, hidden_size, num_attention_heads, intermediate_size,
                     num_kv_heads, head_dim, max_length)
        {}
    };

    template <class Layer> static void setup_layer(Block *block, const Config &config)
    {
        auto layer = dynamic_cast<Layer *>(block);
        auto &attention = layer->attention;
        attention.setup_yarn(config.rope_theta,
            (int)config.original_max_position_embeddings, config.beta_fast, config.beta_slow, config.factor);
    }

    class AdditionalParams
    {
    public:
        AdditionalParams(int sinks_size)
            : use_sinks(sinks_size)
        {}
    private:
        BlockParams::CoreAttentionUseSinks use_sinks;
    };

    template <class SWABock, class FullBlock> class ClassConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel ModelClass;
    public:
        ClassConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_GPT_OSS)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
              config(config)
        {
            AdditionalParams params(config.head_dim);

            const size_t tensor_ovhd = ggml_tensor_overhead();
            size_t num_tensors = 3;
            for (int i = 0; i < config.num_hidden_layers; i++) num_tensors += is_sliding(i) ? 23 : 22;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK(128 == config.sliding_window) << "unsupported SWA param";
            CHATLLM_CHECK((32 == config.num_local_experts) || (128 == config.num_local_experts)) << "unsupported SWA param";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
            {
                return is_sliding(layer_index) ?
                        (Block *)(new SWABock(ctx, config.hidden_size, config.num_attention_heads,
                            config.intermediate_size,
                            config.num_key_value_heads, config.head_dim, config.max_length))
                        :
                        new FullBlock(ctx, config.hidden_size, config.num_attention_heads,
                            config.intermediate_size,
                            config.num_key_value_heads, config.head_dim, config.max_length);
            };

            transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                        create_embedding<Embedding>(&w_ctx_, config),
                        create_final_norm<RMSNorm>(&w_ctx_, config),
                        create_lm_head(&w_ctx_, config, false), create_layer);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_sliding(i))
                {
                    setup_layer<SWABock>(get_typed_transformer<ModelClass>()->get_layer(i), config);
                }
                else
                {
                    setup_layer<FullBlock>(get_typed_transformer<ModelClass>()->get_layer(i), config);
                }
            }

            w_ctx_.check_used_mem_size(true);

            batch_input = false;
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".gate.",                          ".router."},
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:

        bool is_sliding(int layer_id)
        {
            return config.layer_type[layer_id] != 0;
        }

    public:
        Config config;
    };

    const int SLIDING_WINDOW_SIZE = 128;

    namespace experts_32
    {
        const int NUM_EXPERTS                   =  32;

        typedef OssSWABlock<NUM_EXPERTS, 4, SLIDING_WINDOW_SIZE>  OssSWABlock32;
        typedef OssFullBlock<NUM_EXPERTS, 4>  OssFullBlock32;

        typedef ClassConditionalGeneration<OssSWABlock32, OssFullBlock32> ConditionalGeneration;
    }

    namespace experts_128
    {
        const int NUM_EXPERTS                   =  128;

        typedef OssSWABlock<NUM_EXPERTS, 4, SLIDING_WINDOW_SIZE> OssSWABlock128;
        typedef OssFullBlock<NUM_EXPERTS, 4> OssFullBlock128;


        typedef ClassConditionalGeneration<OssSWABlock128, OssFullBlock128> ConditionalGeneration;
    }

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ModelProxy()
        {
            CHATLLM_CHECK(config.sliding_window == 128) << "unsupported MoE param: sliding_window = " << config.sliding_window;

            switch (config.num_local_experts)
            {
            case experts_32::NUM_EXPERTS:
                set_proxy_model(new experts_32::ConditionalGeneration(config, runtime_config));
                break;
            case experts_128::NUM_EXPERTS:
                set_proxy_model(new experts_128::ConditionalGeneration(config, runtime_config));
                break;
            default:
                CHATLLM_CHECK(false) << "unsupported MoE param: num_local_experts = " << config.num_local_experts;
                break;
            }
        }

    };

    REGISTER_MODEL_LOADER(GPT_OSS ,             gpt_oss, 1);
}
