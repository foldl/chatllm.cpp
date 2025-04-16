namespace _7b
{
    struct Config : public llama::v2::Config
    {
        int user_token_id;
        int assistant_token_id;
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

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : llama::v2::Tokenizer::Tokenizer(config, &_chat_encoder),
            user_token_id(config.user_token_id),
            assistant_token_id(config.assistant_token_id)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

    public:
        int user_token_id;
        int assistant_token_id;
    };

    class ConditionalGeneration : public llama::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_BAICHUANLLAMA)
        {
        }
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->user_token_id);
        tok->encode(user, ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->assistant_token_id);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == user_token_id) || (id == assistant_token_id);
    }
}

namespace larger
{
    typedef _7b::Config Config;
    typedef _7b::Tokenizer Tokenizer;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<BaichuanBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_BAICHUAN)
            : ConditionalGeneration(config, runtime_config, type, config.num_attention_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<BaichuanBlock>(config, runtime_config, type, num_key_value_heads, max_length, 13)
        {}
    };
}

namespace m1
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int conv_window;
        int num_swa_attention_heads;
        int num_swa_key_value_heads;
        int sliding_window;
        int sliding_window_pattern;
        float rope_theta;
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

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : llama::v2::Tokenizer(config, &_chat_encoder),
              im_end_token_id(-1)
        {
            sys_prompt = "You are a helpful assistant.";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = llama::v2::Tokenizer::load(buffer, n_vocab);

            int id = tp->PieceToId("<reserved_147>");
            if (id >= 0) tp->OverrideTokenDecoding(id, "<think>");
            id = tp->PieceToId("<reserved_148>");
            if (id >= 0) tp->OverrideTokenDecoding(id, "</think>");

            b_sys_token_id      = 71;
            b_usys_token_id     = 72;
            c_q_token_id        = 73;
            c_a_token_id        = 74;
            b_func_token_id     = 75;
            b_code_token_id     = 76;
            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids, int prefix_token_id = -1)
        {
            if (prefix_token_id >= 0)
                ids.push_back(prefix_token_id);
            llama::v2::Tokenizer::encode(text, ids);
        }

        bool load_config(const json::JSON &config) override
        {
            load_added_tokens(config, {
                {"<B_SYS>",          &b_sys_token_id},
                {"<B_USYS>",         &b_usys_token_id},
                {"<C_Q>",            &c_q_token_id},
                {"<C_A>",            &c_a_token_id},
                {"<B_FUNC>",         &b_func_token_id},
                {"<B_CODE>",         &b_code_token_id},
                {"<|im_start|>",     &im_start_token_id},
                {"<|im_end|>",       &im_end_token_id},
            });

            if (im_end_token_id >= 0)
                terminate_ids.insert(im_end_token_id);

            return true;
        }

    public:
        int b_sys_token_id;
        int b_usys_token_id;
        int c_q_token_id;
        int c_a_token_id;
        int b_func_token_id;
        int b_code_token_id;
        int im_start_token_id;
        int im_end_token_id;
    };

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids, tok->b_sys_token_id);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(user, ids, tok->c_q_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->c_a_token_id);
    }

    static class ImChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

            if (tok->get_system_prompt().size() > 0)
            {
                ids.push_back(tok->im_start_token_id);
                tok->encode("system\n", ids);
                tok->encode(tok->get_system_prompt(), ids);
                ids.push_back(tok->im_end_token_id);
                tok->encode("\n", ids);
            }
        }
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            append_ai_opening(round_idx, ids);
            tok->encode(ai, ids);
            ids.push_back(tok->im_end_token_id);
            tok->encode("\n", ids);
        }
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            append_user_opening(round_idx, ids);
            tok->encode(user, ids);
            ids.push_back(tok->im_end_token_id);
            tok->encode("\n", ids);
        }

        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            ids.push_back(tok->im_start_token_id);
            tok->encode("assistant\n", ids);
        }

        void append_user_opening(int round_idx, std::vector<int> &ids) const override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            ids.push_back(tok->im_start_token_id);
            tok->encode("user\n", ids);
        }
    } _im_chat_encoder;

    template <int sliding_window_len> class BaiChuanSWASelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        BaiChuanSWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias) {}
    };

    template <class BaseAttn> class BaiChuanFIR2SelfAttention : public BaseAttn
    {
    public:
        BaiChuanFIR2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : BaseAttn(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
              conv_k(ctx, num_kv_heads, hidden_size / num_attention_heads * num_kv_heads),
              conv_v(ctx, num_kv_heads, hidden_size / num_attention_heads * num_kv_heads)
        {
            BaseAttn::rope_mode = RoPEMode::Original;
        }

        ggml::tensor *cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
            ggml::tensor *q, ggml::tensor *k, ggml::tensor *v) override
        {
            const int head_size = hidden_size / BaseAttn::num_attention_heads;

            // [qlen, heads, head_size]
            k = ggml::reshape_3d(ctx, k, head_size, BaseAttn::num_kv_heads, qlen);
            k = conv_k.forward(ctx, k, n_past);
            k = ggml::reshape_2d(ctx, k, head_size * BaseAttn::num_kv_heads, qlen);

            v = ggml::reshape_3d(ctx, v, head_size, BaseAttn::num_kv_heads, qlen);
            v = conv_v.forward(ctx, v, n_past);
            v = ggml::reshape_2d(ctx, v, head_size * BaseAttn::num_kv_heads, qlen);

            auto attn = BaseAttn::cross_attention(ctx, hidden_size, n_past, qlen, q, k, v);

            return attn;
        }

        void set_id(int id) override
        {
            BaseAttn::set_id(id);
            conv_k.set_id(id);
            conv_v.set_id(id);
        }

    public:
        FIR2 conv_k;
        FIR2 conv_v;
    };

    const int SLIDING_WINDOW_LEN  =  8192;

    typedef BaiChuanFIR2SelfAttention<RoPESelfAttention<BaseAttention>> BaiChuanFIR2FullSelfAttention;
    typedef BaiChuanFIR2SelfAttention<BaiChuanSWASelfAttention<SLIDING_WINDOW_LEN>> BaiChuanFIR2SWASelfAttention8k;

    typedef LMBlock1<RMSNorm, BaiChuanFIR2FullSelfAttention, RMSNorm, SiLUMLP> BaiChuanFullBlock;
    typedef LMBlock1<RMSNorm, BaiChuanFIR2SWASelfAttention8k, RMSNorm, SiLUMLP> BaiChuanSWABlock8k;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel<BaseConfig, Embedding, RMSNorm> ModelClass;

    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_BAICHUAN_M1)
            : BaseModelForConditionalGeneration(type, config, runtime_config),
              sliding_window_pattern(config.sliding_window_pattern), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            size_t num_tensors = 3;
            for (int i = 0; i < config.num_hidden_layers; i++)
                num_tensors += is_swa_layer(i) ? 17 : 16;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((SLIDING_WINDOW_LEN == config.sliding_window) && (sliding_window_pattern == 2)) << "unsupported SWA param";
            CHATLLM_CHECK((2 == config.conv_window)) << "unsupported conv_window param";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
            {
                if (is_swa_layer(layer_index))
                {
                    return new BaiChuanSWABlock8k(ctx, config.hidden_size, config.num_swa_attention_heads, config.intermediate_size,
                                                config.num_swa_key_value_heads, config.max_length);
                }
                else
                {
                    return new BaiChuanFullBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                }
            };

            transformer = new ModelClass(&w_ctx_, config, false, create_layer);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_swa_layer(i))
                {
                    auto &attention = dynamic_cast<BaiChuanSWABlock8k *>(get_typed_transformer<ModelClass>()->get_layer(i))->attention;
                    attention.freq_base = config.rope_theta;
                }
                else
                {
                    auto &attention = dynamic_cast<BaiChuanFullBlock *>(get_typed_transformer<ModelClass>()->get_layer(i))->attention;
                    attention.freq_base = config.rope_theta;
                }
            }

            batch_input = false;

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " != " << w_ctx_.get_mem_size() / ggml_tensor_overhead();
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            auto it = args.find("chat_template");
            if (it != args.end())
            {
                if (it->second == "im")
                {
                    tokenizer->set_chat_encoder(&_im_chat_encoder);
                }
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            #define LOAD_TENSORS()  \
                loader.read_tensor(layer_prefix + "input_layernorm.weight", layer->input_layernorm.weight); \
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight",   layer->mlp.down_proj.weight);   \
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",   layer->mlp.gate_proj.weight);   \
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",     layer->mlp.up_proj.weight);     \
                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight); \
                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", layer->attention.k_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", layer->attention.o_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", layer->attention.q_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", layer->attention.v_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.conv_k", layer->attention.conv_k.weight);                \
                loader.read_tensor(layer_prefix + "self_attn.conv_v", layer->attention.conv_v.weight);

            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                if (is_swa_layer(i))
                {
                    auto layer = dynamic_cast<BaiChuanSWABlock8k *>(transformer->get_layer(i));
                    LOAD_TENSORS();
                }
                else
                {
                    auto layer = dynamic_cast<BaiChuanFullBlock *>(transformer->get_layer(i));
                    LOAD_TENSORS();
                }
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            #undef LOAD_TENSORS
        }

    private:
        bool is_swa_layer(int layer_index) const
        {
            return layer_index % sliding_window_pattern == 1;
        }
        const int sliding_window_pattern;
    public:
        BaseConfig config;
    };
}