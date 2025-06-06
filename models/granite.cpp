namespace moe
{

    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int tie_word_embeddings;
        int num_experts_per_tok;
        int num_local_experts;

        float attention_multiplier;
        float logits_scaling;
        float residual_multiplier;
        float rope_theta;
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

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);

            start_of_role_id = tp->PieceToId("<|start_of_role|>");
            end_of_role_id   = tp->PieceToId("<|end_of_role|>");
            tool_call_id     = tp->PieceToId("<|tool_call|>");

            std::vector<int> ids;
            tp->Encode("\n", &ids);
            nl_token_id = ids[0];

            return size;
        }

        bool is_special_id(int id) const override
        {
            return (id == start_of_role_id) || (id == end_of_role_id);
        }

        void encode_header(const std::string &text, std::vector<int> &ids) const
        {
            ids.push_back(start_of_role_id);
            encode(text, ids);
            ids.push_back(end_of_role_id);
        }

        void encode_content(const std::string &text, std::vector<int> &ids) const
        {
            encode(text, ids);
            ids.push_back(eos_token_id);
            ids.push_back(nl_token_id);
        }

    public:
        int start_of_role_id;
        int end_of_role_id;
        int tool_call_id;
        int nl_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode_content(ai, ids);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode_header("system", ids);
            tok->encode_content(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        tok->encode_header("user", ids);
        tok->encode_content(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_header("assistant", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_header("user", ids);
    }

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class GraniteSparseMoE : public BaseSparseMLP
    {
    public:
        GraniteSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template<int num_local_experts, int num_experts_per_tok> class GraniteBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm,
                        GraniteSparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        GraniteBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm,
                       GraniteSparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, GraniteBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK>, int, int, int, int, int> ModelClass;
    public:
        _ConditionalGeneration() = default;

        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : Base(type, config, runtime_config, 4096 * 2), config(config), tie_embeddings(config.tie_word_embeddings != 0)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * (11 + 2) + (tie_embeddings ? -1 : 0);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_local_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            if (tie_embeddings)
            {
                transformer = new ModelClass(&w_ctx_, config, nullptr,
                                            config.hidden_size, config.num_attention_heads,
                                            config.intermediate_size, config.num_key_value_heads,
                                            config.max_length);
            }
            else
            {
                transformer = new ModelClass(&w_ctx_, config, false,
                                            config.hidden_size, config.num_attention_heads,
                                            config.intermediate_size, config.num_key_value_heads,
                                            config.max_length);
            }


            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = Base::get_typed_transformer<ModelClass>()->layers[i];
                layer.scale_depth                   = config.residual_multiplier;
                layer.attention.attn_scaling_factor = config.attention_multiplier;
                layer.attention.freq_base           = config.rope_theta;
            }

            Base::logit_scale = config.logits_scaling;

            CHATLLM_CHECK(w_ctx_.get_used_mem() / tensor_ovhd == num_tensors)
                << "actual tensors: " << w_ctx_.get_used_mem() / tensor_ovhd << ", expected " << num_tensors << ".";
            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size()) << "corrupted model weights";
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".down_proj.weight", transformer->layers[i].mlp.experts_down.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".gate_proj.weight", transformer->layers[i].mlp.experts_gate.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".up_proj.weight", transformer->layers[i].mlp.experts_up.weight);

                loader.read_tensor(layer_prefix + "block_sparse_moe.router.layer.weight",
                                transformer->layers[i].mlp.gate.weight);

                loader.read_tensor(layer_prefix + "input_layernorm.weight",
                                transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            if (!tie_embeddings)
                loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);
        }

        static AbstractModel *create(const Config &config, const RuntimeConfig &runtime_config)
        {
            return new _ConditionalGeneration(config, runtime_config);
        }

    public:
        Config config;
        bool tie_embeddings;
    };

    namespace experts_32
    {
        const int NUM_EXPERTS                   =  32;
        const int EXPERTS_PER_TOK               =  8;

        typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_GRANITE_MoE> ConditionalGeneration;
    }

    namespace experts_40
    {
        const int NUM_EXPERTS                   =  40;
        const int EXPERTS_PER_TOK               =  8;

        typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_GRANITE_MoE> ConditionalGeneration;
    }

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config) : ModelProxy()
        {
            switch (config.num_local_experts)
            {
            case experts_32::NUM_EXPERTS:
                set_proxy_model(experts_32::ConditionalGeneration::create(config, runtime_config));
                break;
            case experts_40::NUM_EXPERTS:
                set_proxy_model(experts_40::ConditionalGeneration::create(config, runtime_config));
                break;
            default:
                CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.num_local_experts;
                break;
            }
        }
    };
}

namespace dense
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int tie_word_embeddings;

        float attention_multiplier;
        float logits_scaling;
        float residual_multiplier;
        float rope_theta;
    };

    typedef moe::Tokenizer Tokenizer;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_GRANITE)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length, 12, config.tie_word_embeddings)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = get_typed_transformer<ModelClass>()->layers[i];
                layer.scale_depth                   = config.residual_multiplier;
                layer.attention.attn_scaling_factor = config.attention_multiplier;
                layer.attention.freq_base           = config.rope_theta;
            }

            logit_scale = config.logits_scaling;
        }
    };
}