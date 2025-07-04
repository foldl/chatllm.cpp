#include "ernie.h"

namespace chatllm::ernie::dense
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : chatllm::llama::v2::Tokenizer(config, &_chat_encoder)
    {}

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            oss_prompt << tok->get_system_prompt() << "\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "User:  " + user << "\n";
        auto text = oss_prompt.str();
        tok->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("Assistant:  ", ids);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : chatllm::llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type,
            config.num_key_value_heads, config.head_dim, config.max_length, 12, config.tie_word_embeddings != 0)
    {
        auto transformer = Base::get_typed_transformer<ModelClass2>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
}

namespace chatllm::ernie::moe
{
    template <class ErnieMoEMLP> class ErnieMoEBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, ErnieMoEMLP>
    {
    public:
        ErnieMoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size1, int mlp_intermediate_size2,
                int num_kv_heads, int head_dim, int max_length)
            : LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, ErnieMoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
            num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class ErnieSparseMoE : public BaseSparseMLP
    {
    public:
        ErnieSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<ErnieSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>, SiLUMLP> ErnieMoEMLP;
        typedef ErnieMoEBlock<ErnieMoEMLP> MoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        GenericConditionalGeneration() = default;

        GenericConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_ERNIE_MOE, config, runtime_config, 4096 * 4),
            config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t moe_layers  = get_moe_layer_num();
            const size_t dense_layers = config.num_hidden_layers - moe_layers;
            const size_t num_tensors = 2 + dense_layers * (12) + moe_layers * (16 + 0) + (config.tie_word_embeddings ? 0 : 1);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (config.use_correction_bias)
                ggml::log(GGML_LOG_LEVEL_WARN, "use_correction_bias is ignored, see https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/modeling_ernie4_5_moe.py#L369");

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new MoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.moe_num_shared_experts,
                        config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                        config.max_length);
                    layer->attention.freq_base = config.rope_theta;
                    layer->mlp.mlp1.norm_topk_prob = true;
                    return layer;
                }
                else
                {
                    auto layer = new LlamaBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                        config.num_key_value_heads, config.max_length);
                    layer->attention.freq_base = config.rope_theta;
                    return layer;
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                config.tie_word_embeddings ? nullptr : create_lm_head(&w_ctx_, config, false), create_layer);
            Base::transformer = transformer;

            w_ctx_.check_used_mem_size(true);
        }

    protected:
        int  get_moe_layer_num(void)
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
                if (is_layer_moe(i)) r++;
            return r;
        }

        bool is_layer_moe(int i)
        {
            if (i < config.moe_layer_start_index) return false;
            return (i % config.moe_layer_interval) == 0;
        }
    public:
        Config config;
    };

    namespace experts_64
    {
        const int NUM_EXPERTS                   =  64;
        const int EXPERTS_PER_TOK               =  6;

        // make it easy to test with different number of experts.
        const int EFFECTIVE_EXPERTS_PER_TOK     =  EXPERTS_PER_TOK;

        typedef GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
    {
        switch (config.moe_num_experts)
        {
        case experts_64::NUM_EXPERTS:
            set_proxy_model(new experts_64::ConditionalGeneration(config, runtime_config));
            break;
        default:
            CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.moe_num_experts;
            break;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".mlp2.",              ".shared_experts."},
            {".mlp1.gate.",         ".gate."},
            {".mlp1.experts.",      ".experts."},
            {".mlp1.gate_score_correction_bias",     ".moe_statics.e_score_correction_bias"}
        });

        ModelProxy::load(loader);
    }
}