namespace v1
{
    struct Config : public BaseConfig
    {
        int seq_length;
        int rope_dim;
        int flags;
        float rotary_emb_base;
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
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr)
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
        {
            sys_prompt = "You are a helpful assistant.";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_eos) const;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const;

    protected:
        virtual size_t do_load(tokenizer::DataReader *buffer, int n_vocab);

    public:
        int im_start_token_id;
        int im_end_token_id;
        int nl_token_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_QWEN);

        void load(ModelLoader &loader) override;

    public:
        Config config;
    };

    size_t Tokenizer::do_load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
        return tp->Load(buffer, n_vocab);
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t size = do_load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        // for QAnything
        pad_token_id = eos_token_id = bos_token_id = tp->PieceToId("<|endoftext|>");
        im_start_token_id           = tp->PieceToId("<|im_start|>");
        im_end_token_id             = tp->PieceToId("<|im_end|>");

        if (im_end_token_id < 0)
        {
            // QWen v1
            pad_token_id = eos_token_id = bos_token_id = tp->GetPieceSize() + 0;
            im_start_token_id           = eos_token_id + 1;
            im_end_token_id             = eos_token_id + 2;
        }

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        if (im_end_token_id >= 0)
            terminate_ids.insert(im_end_token_id);

        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const
    {
        if (add_im_start)
            ids.push_back(im_start_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_im_end)
            ids.push_back(im_end_token_id);
        if (add_nl)
            ids.push_back(nl_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode("system", ids, true, false, true);
        tok->encode(tok->get_system_prompt(), ids, false, true, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        tok->encode("user", ids, true, false, true);
        tok->encode(user, ids, false, true, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("assistant", ids, true, false, true);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("user", ids, true, false, true);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == im_start_token_id) || (id == im_end_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * 16;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        // TODO: support of `use_dynamic_ntk`
        transformer = new ModelClass(&w_ctx_, config, false,
                                    config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, config.max_length);

        bool use_dynamic_ntk = (config.flags & 1) != 0;
        bool use_logn_attn   = (config.flags & 2) != 0;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer =  get_typed_transformer<ModelClass>()->layers[i];
            auto att = dynamic_cast<QWenSelfAttention *>(&layer.attention);
            att->config(config.rope_dim, config.rotary_emb_base, config.seq_length,
                        use_dynamic_ntk, use_logn_attn);
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();
        transformer->word_embeddings->load("transformer.wte.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "transformer.h." + std::to_string(layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "attn.c_proj.weight", transformer->layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "ln_1.weight",        transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "ln_2.weight",        transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "mlp.c_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.w1.weight",   transformer->layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.w2.weight", transformer->layers[i].mlp.gate_proj.weight);
        }
        transformer->final_layernorm->load("transformer.ln_f.", &loader);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace v2
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int sliding_window;
        float rope_theta;
    };

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v1::Tokenizer(config, &v1::_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = v1::Tokenizer::load(buffer, n_vocab);

            im_start_token_id = tp->PieceToId("<|im_start|>");
            im_end_token_id   = tp->PieceToId("<|im_end|>");
            bos_token_id = pad_token_id = eos_token_id = im_start_token_id - 1;

            std::vector<int> ids;
            tp->Encode("\n", &ids);
            nl_token_id = ids[0];

            if (im_end_token_id >= 0)
                terminate_ids.insert(im_end_token_id);

            return r;
        }
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_QWEN2, bool tie_embeddings = false);

    public:
        Config config;

    private:
        const bool tie_embeddings;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, bool tie_embeddings)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
        config(config), tie_embeddings(tie_embeddings)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * 15 + (tie_embeddings ? -1 : 0);
        const size_t ctx_size = num_tensors * tensor_ovhd;

        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

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
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base = config.rope_theta;
        }
    }
}

namespace v2_tie
{
    typedef v2::Config Config;
    typedef v2::Tokenizer Tokenizer;

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_QWEN2TIE, true)
        {}
    };
}

namespace v2_moe
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int moe_intermediate_size;
        int shared_expert_intermediate_size;
        int sliding_window;
        int num_experts_per_tok;
        int num_experts;
        int norm_topk_prob;
        float rope_theta;
    };

    typedef v2::Tokenizer Tokenizer;

    template <class QWenMoEMLP> class QWen2MoEBlock : public LMBlock1<RMSNorm, QWen2SelfAttention, RMSNorm, QWenMoEMLP>
    {
    public:
        QWen2MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : LMBlock1<RMSNorm, QWen2SelfAttention, RMSNorm, QWenMoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
              num_kv_heads, head_dim, max_length)
        {}
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK, class MoEBlock> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, MoEBlock, int, int, int, int, int, int, int, int> ModelClass;
    public:
        GenericConditionalGeneration() = default;

        GenericConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_QWEN2MoE, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * (17 + 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.num_experts) && (EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            Base::transformer = new ModelClass(
                &w_ctx_, config, false,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.moe_intermediate_size, config.shared_expert_intermediate_size,
                config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = Base::get_typed_transformer<ModelClass>()->layers[i];
                layer.attention.freq_base = config.rope_theta;
                layer.mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = Base::get_typed_transformer<ModelClass>();

            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "mlp.mlp1.experts_down.weight", layer_prefix + "mlp.experts.", config.num_experts, ".down_proj.weight", transformer->layers[i].mlp.mlp1.experts_down.weight);
                loader.read_tensor(layer_prefix + "mlp.mlp1.experts_gate.weight", layer_prefix + "mlp.experts.", config.num_experts, ".gate_proj.weight", transformer->layers[i].mlp.mlp1.experts_gate.weight);
                loader.read_tensor(layer_prefix + "mlp.mlp1.experts_up.weight",   layer_prefix + "mlp.experts.", config.num_experts, ".up_proj.weight",   transformer->layers[i].mlp.mlp1.experts_up.weight);

                loader.read_tensor(layer_prefix + "mlp.gate.weight", transformer->layers[i].mlp.mlp1.gate.weight);

                loader.read_tensor(layer_prefix + "mlp.shared_expert.down_proj.weight", transformer->layers[i].mlp.mlp2.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.shared_expert.gate_proj.weight", transformer->layers[i].mlp.mlp2.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.shared_expert.up_proj.weight",   transformer->layers[i].mlp.mlp2.up_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.shared_expert_gate.weight",   transformer->layers[i].mlp.mlp2.gate.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class QWenSparseMoE : public BaseSparseMLP
    {
    public:
        QWenSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK> class ClassConditionalGeneration
    {
    public:
        typedef GatedMLP<SiLUMLP> QWenGatedMLP;

        typedef CombinedMLP<QWenSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>, QWenGatedMLP> QWenMoEMLP;

        typedef QWen2MoEBlock<QWenMoEMLP> MoEBlock;

        class ConditionalGeneration : public GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK, MoEBlock>
        {
        public:
            ConditionalGeneration() = default;

            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
                : GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK, MoEBlock>(config, runtime_config) {}
        };

        static AbstractModel *create(const Config &config, const RuntimeConfig &runtime_config)
        {
            return new ConditionalGeneration(config, runtime_config);
        }
    };

    namespace experts_60
    {
        const int NUM_EXPERTS                   =  60;
        const int EXPERTS_PER_TOK               =  4;

        // make it easy to test with different number of experts.
        #define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

        typedef ClassConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    namespace experts_64
    {
        const int NUM_EXPERTS                   =  64;
        const int EXPERTS_PER_TOK               =  8;

        // make it easy to test with different number of experts.
        #define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

        typedef ClassConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config) : ModelProxy()
        {
            switch (config.num_experts)
            {
            case experts_60::NUM_EXPERTS:
                set_proxy_model(experts_60::ConditionalGeneration::create(config, runtime_config));
                break;
            case experts_64::NUM_EXPERTS:
                set_proxy_model(experts_64::ConditionalGeneration::create(config, runtime_config));
                break;
            default:
                CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.num_experts;
                break;
            }
        }
    };
}

namespace marco_o1
{
    typedef v2::Config Config;

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v2::Tokenizer(config)
        {
            sys_prompt = "\n你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.\n        \n## 重要！！！！！\n当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。\n<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。\n        ";
        }
    };

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_MARCO_O1)
        {}
    };
}

namespace qwq
{
    typedef v2::Config Config;

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v2::Tokenizer(config)
        {
            sys_prompt = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.";
        }
    };

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_QWQ)
        {}
    };
}

namespace ds_r1_distill
{
    struct Config : v2::Config
    {
        int tie;
    };

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

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr)
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);
            tp->EnableReturnSpecialToken(true);

            user_token_id           = tp->PieceToId("<｜User｜>");
            assistant_token_id      = tp->PieceToId("<｜Assistant｜>");

            std::vector<int> ids;
            tp->Encode("\n", &ids);

            nl_token_id = -1;

            if (ids.size()  == 1)
                nl_token_id = ids[0];

            bos_token_id            = tp->PieceToId("<｜begin▁of▁sentence｜>");
            eos_token_id            = tp->PieceToId("<｜end▁of▁sentence｜>");

            return size;
        }

    public:
        int user_token_id;
        int assistant_token_id;
        int nl_token_id;
    };

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->user_token_id);
        tok->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
    }

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN)
            : v2::ConditionalGeneration(config, runtime_config, type, config.tie != 0)
        {}
    };
}

namespace v2_5_vl
{

    const int MROPE_SECTION_MAX = 4;

    struct Config : v2::Config
    {
        int tie_word_embeddings;
        int mrope_section[MROPE_SECTION_MAX];
    };

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config) : v2::Tokenizer(config)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = v2::Tokenizer::load(buffer, n_vocab);



            return r;
        }
    public:
        int object_ref_start_token_id;
        int object_ref_end_token_id;
        int box_start_token_id;
        int box_end_token_id;
        int quad_start_token_id;
        int quad_end_token_id;
        int vision_start_token_id;
        int vision_end_token_id;
        int vision_pad_token_id;
        int image_pad_token_id;
        int video_pad_token_id;
    };

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    struct rgb_image
    {
        std::vector<uint8_t> rgb_pixels;
        int height;
        int width;
    };

    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_QWEN2, config.tie_word_embeddings != 0)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            }
        }

        int append_image(const uint8_t *rgb_pixels, int width, int height) override
        {
            //return model->append_image(rgb_pixels, width, height);
            return 0;
        }

        void clear_images(void) override
        {
            media_emb.clear();
        }

    public:
        std::vector<rgb_image> media_emb;
    };
}

namespace v3
{
    const int MAX_LAYERS = 128;

    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        float rope_theta;
        float yarn_scaling_factor;
        int yarn_scaling_original_max_position_embeddings;
        int decoder_sparse_step;
        int moe_intermediate_size;
        int num_experts_per_tok;
        int num_experts;
        int norm_topk_prob;
        int tie_word_embeddings;
        int layer_is_sparse[MAX_LAYERS];
    };

    typedef v2::Tokenizer Tokenizer;

    class QWen3SelfAttention : public QKNormedAttention<RMSNorm, BaseAttention>
    {
    public:
        QWen3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<RMSNorm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {

        }
    };

    class QWen3Block : public LMBlock1<RMSNorm, QWen3SelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        QWen3Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class QWen3MoEBlock : public LMBlock1<RMSNorm, QWen3SelfAttention, RMSNorm, v2_moe::QWenSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
    {
    public:
        typedef v2_moe::QWenSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK> QWenMoEMLP;
    public:
        QWen3MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : LMBlock1<RMSNorm, QWen3SelfAttention, RMSNorm, QWenMoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size,
              num_kv_heads, head_dim, max_length)
        {}
    };

    typedef QWen3MoEBlock<128, 8> QWen3MoEBlock128_8;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_QWEN3, const bool skip_lm_head = false, int extra_tensors = 0)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int sparse_layers = get_sparse_layer_num();
            const size_t num_tensors = 3 + (config.tie_word_embeddings ? -1 : 0)
                                         + (config.num_hidden_layers - sparse_layers) * 14
                                         + sparse_layers * (14 + 1)
                                         + extra_tensors;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (skip_lm_head || config.tie_word_embeddings)
            {
                transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                    create_embedding<Embedding>(&w_ctx_, config),
                    create_final_norm<RMSNorm>(&w_ctx_, config),
                    nullptr,
                    [&](InitContext *ctx, int layer_index) {
                        return create_layer(ctx, layer_index);
                    });
            }
            else
            {
                transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                    create_embedding<Embedding>(&w_ctx_, config),
                    create_final_norm<RMSNorm>(&w_ctx_, config),
                    create_lm_head(&w_ctx_, config, false),
                    [&](InitContext *ctx, int layer_index) {
                        return create_layer(ctx, layer_index);
                    });
            }

            if (config.yarn_scaling_factor > 0.0)
                ggml::log(GGML_LOG_LEVEL_WARN, "TODO: YaRN (yarn_scaling_factor = %f) not implemented", config.yarn_scaling_factor);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (config.layer_is_sparse[i])
                {
                    auto layer = (QWen3MoEBlock128_8 *)get_typed_transformer<ModelClass>()->get_layer(i);
                    layer->attention.freq_base = config.rope_theta;
                    layer->mlp.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    auto layer = (QWen3Block *)get_typed_transformer<ModelClass>()->get_layer(i);
                    layer->attention.freq_base = config.rope_theta;
                }
            }

            CHATLLM_CHECK(w_ctx_.get_used_mem() + extra_tensors * ggml_tensor_overhead() == w_ctx_.get_mem_size())
                << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " != " << w_ctx_.get_mem_size() / ggml_tensor_overhead();
        }

    private:
        int get_sparse_layer_num()
        {
            int num = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (config.layer_is_sparse[i])
                    num++;
            }
            return num;
        }

        Block *create_layer(InitContext *ctx, int layer_index)
        {
            if (config.layer_is_sparse[layer_index])
            {
                if ((config.num_experts_per_tok == 8) && (config.num_experts == 128))
                {
                    return new QWen3MoEBlock128_8(ctx, config.hidden_size, config.num_attention_heads,
                        config.intermediate_size, config.moe_intermediate_size,
                        config.num_key_value_heads, config.head_dim, config.max_length);
                }
                else
                {
                    CHATLLM_CHECK(false) << "unsupported MoE param";
                    return nullptr;
                }
            }
            else
            {
                return new QWen3Block(ctx, config.hidden_size, config.num_attention_heads,
                    config.intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length);
            }
        }

    public:
        Config config;
    };
}

namespace ds_r1_distill_v3
{
    typedef v3::Config Config;

    class Tokenizer : public ds_r1_distill::Tokenizer
    {
    public:
        Tokenizer(BaseConfig config)
            : ds_r1_distill::Tokenizer(config)
        {
            std::time_t now = std::time(nullptr);
            std::tm* timeinfo = std::localtime(&now);
            char buffer[1000];
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d, %A", timeinfo);
            std::string date_str(buffer);
            sys_prompt = "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是" + date_str + "。";
        }
    };

    typedef v3::ConditionalGeneration ConditionalGeneration;
}


namespace v3_emb
{
    typedef v3::Config Config;

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v3::Tokenizer(config)
        {
            task = "Given a web search query, retrieve relevant passages that answer the query";
        }

        void encode_embedding(const std::string &text, std::vector<int> &ids, EmbeddingPurpose purpose) const override;

    public:
        std::string task;
    };

    void Tokenizer::encode_embedding(const std::string &text, std::vector<int> &ids, EmbeddingPurpose purpose) const
    {
        std::ostringstream oss;
        switch (purpose)
        {
        case EmbeddingPurpose::Query:
            oss << "Instruct: " << task << "\nQuery:" << text;
            BaseTokenizer::encode(oss.str(), ids);
            break;

        default:
            BaseTokenizer::encode(text, ids);
            break;
        }
        ids.push_back(eos_token_id);
    }


    class ConditionalGeneration : public v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_QWEN3_Embedding, const bool skip_lm_head = true, int extra_tensors = 0)
            : v3::ConditionalGeneration(config, runtime_config, type, skip_lm_head, extra_tensors)
        {
            dynamic_cast<HeterogeneousModel *>(transformer)->set_final_steps(std::make_unique<EmbeddingLastTokenFinalSteps>());
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            auto it = args.find("task");
            if (it != args.end())
            {
                tok->task = it->second;
            }
        }
    };
}

namespace v3_ranker
{
    typedef v3::Config Config;

    class Tokenizer : public v3_emb::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v3_emb::Tokenizer(config)
        {
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const override;
    public:
        int yes_token_id;
        int no_token_id;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t size = v3_emb::Tokenizer::load(buffer, n_vocab);

        yes_token_id = tp->PieceToId("yes");
         no_token_id = tp->PieceToId("no");

        return size;
    }

    void Tokenizer::encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const
    {
        std::ostringstream oss;
        oss << "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        oss << "<Instruct>: " << task << "\n<Query>: " << q << "\n<Document>: " << a;
        oss << "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        BaseTokenizer::encode(oss.str(), ids);
    }

    class FinalSteps : public LMFinalSteps
    {
    public:
        ggml::tensor *forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states) override;
    public:
        ggml::tensor *yes_no_ids;
    };

    ggml::tensor *FinalSteps::forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
    {
        ggml::tensor *logits = LMFinalSteps::forward(model, ctx, input_ids, hidden_states);
        logits = ggml::reshape_2d(ctx, logits, 1, ggml::get_dim(logits, 0));
        logits = ggml::get_rows(ctx, logits, yes_no_ids);
        logits = ggml::reshape_1d(ctx, logits, 2);
        logits = ggml::soft_max(ctx, logits);
        logits = ggml::view_1d(ctx, logits, 1, 0);
        return logits;
    }

    class ConditionalGeneration : public v3_emb::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v3_emb::ConditionalGeneration(config, runtime_config, MODEL_TYPE_QWEN3_ReRanker, false, 1)
        {
            dynamic_cast<HeterogeneousModel *>(transformer)->set_final_steps(std::make_unique<FinalSteps>());

            FinalSteps *steps = dynamic_cast<FinalSteps *>(dynamic_cast<HeterogeneousModel *>(transformer)->get_final_steps());
            steps->yes_no_ids = ggml::new_tensor_1d(&w_ctx_, ggml::type::GGML_TYPE_I32, 2);
            w_ctx_.get_allocator()->alloc(steps->yes_no_ids);
            yes_no_ids = steps->yes_no_ids;
        }

        void set_tokenizer(BaseTokenizer *tokenizer) override
        {
            v3::ConditionalGeneration::set_tokenizer(tokenizer);

            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            int ids[2];
            ids[0] = tok->yes_token_id;
            ids[1] = tok->no_token_id;
            Backend::write_tensor_data(yes_no_ids, ids, 0, sizeof(ids));
        }
    protected:
        ggml::tensor *yes_no_ids = nullptr;
    };
}