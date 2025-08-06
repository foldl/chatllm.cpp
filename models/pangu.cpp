#include "pangu.h"

namespace chatllm::pangu::moe
{
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

    Tokenizer::Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = R"""(你必须严格遵守法律法规和社会道德规范。生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。例如，如果输入内容包含暴力威胁或色情描述，应返回错误信息：“您的输入包含不当内容，无法处理。)""";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        pad_token_id = tp->PieceToId("<pad>");
        unused9_token_id  = tp->PieceToId("[unused9]");
        unused10_token_id = tp->PieceToId("[unused10]");
        tp->OverrideTokenDecoding(tp->PieceToId("[unused16]"), "<think>");
        tp->OverrideTokenDecoding(tp->PieceToId("[unused17]"), "</think>");
        return size;
    }

    void Tokenizer::encode_item(const char *tag, std::vector<int> &ids)
    {
        ids.push_back(unused9_token_id);
        encode(std::string(tag) + "：", ids);
    }

    void Tokenizer::encode_item(const char *tag, const std::string &content, std::vector<int> &ids)
    {
        ids.push_back(unused9_token_id);
        encode(std::string(tag) + "：" + content, ids);
        ids.push_back(unused10_token_id);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("助手", ai, ids);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        tok->encode_item("系统", tok->get_system_prompt(), ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("用户", user, ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("用户", ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_item("助手", ids);
    }

    template <class PanguMoEMLP> class PanguMoEBlock : public LMBlock1<RMSNorm, FullBiasedSelfAttention, RMSNorm, PanguMoEMLP>
    {
    public:
        PanguMoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size1, int mlp_intermediate_size2,
                int num_kv_heads, int head_dim, int max_length)
            : LMBlock1<RMSNorm, FullBiasedSelfAttention, RMSNorm, PanguMoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
            num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class PanguSparseMoE : public BaseSparseMLP
    {
    public:
        PanguSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false, true, true)
        {
            norm_topk_prob = false;
        }
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<PanguSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>, SiLUMLP> PanguMoEMLP;
        typedef PanguMoEBlock<PanguMoEMLP> MoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, MoEBlock, int, int, int, int, int, int, int, int> ModelClass;
    public:
        GenericConditionalGeneration() = default;

        GenericConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_PANGU_MOE, config, runtime_config, 4096 * 4),
            config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * 22;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            Base::transformer = new ModelClass(
                &w_ctx_, config, false,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.moe_intermediate_size, config.intermediate_size,
                config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = Base::get_typed_transformer<ModelClass>()->layers[i];
                layer.attention.freq_base = config.rope_theta;
            }

            w_ctx_.check_used_mem_size(true);
        }

    public:
        Config config;
    };

    namespace experts_64
    {
        const int NUM_EXPERTS                   =  64;
        const int EXPERTS_PER_TOK               =  8;

        typedef GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK> ConditionalGeneration;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
    {
        switch (config.num_experts)
        {
        case experts_64::NUM_EXPERTS:
            set_proxy_model(new experts_64::ConditionalGeneration(config, runtime_config));
            break;
        default:
            CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.num_experts;
            break;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".mlp2.",              ".shared_expert."},
            {".mlp1.gate.",         ".gate."},
            {".mlp1.router_scale",  ".router_scale"},
            {".mlp1.experts.",      ".experts."},
        });

        ModelProxy::load(loader);
    }

    REGISTER_MODEL_LOADER(PANGU_MOE,             pangu::moe, 1);
}

namespace chatllm::pangu::embedded
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int tie_word_embeddings;

        float rope_theta;
    };

    typedef moe::Tokenizer Tokenizer;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef LMBlock1<RMSNorm, FullBiasedSelfAttention, RMSNorm, SiLUMLP> PanguDenseBlock;
        typedef Model<Config, Embedding, RMSNorm, PanguDenseBlock, int, int, int, int, int> ModelClass;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_PANGU_EMBEDDED, config, runtime_config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * 16 + (config.tie_word_embeddings ? -1 : 0);
            const size_t ctx_size = num_tensors * tensor_ovhd;

            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config,
                (0 == config.tie_word_embeddings) ? create_embedding<Embedding>(&w_ctx_, config) : nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads,
                config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = get_typed_transformer<ModelClass>()->layers[i];
                layer.attention.freq_base = config.rope_theta;
            }

            w_ctx_.check_used_mem_size(true);
        }
    };

    REGISTER_MODEL_LOADER(PANGU_EMBEDDED,             pangu::embedded, 1);
}