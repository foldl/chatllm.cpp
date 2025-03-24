namespace pro
{
    const int MAX_LEN = 20;

    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int sliding_window;
        int pairs_num;
        int layer_forwarding_ids[2 * MAX_LEN];
        float rope_theta;
        float layer_alpha;
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
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_eos) const;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const;

    public:
        int im_start_token_id;
        int im_end_token_id;
        int nl_token_id;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        // for QAnything
        im_start_token_id           = tp->PieceToId("<|im_start|>");
        im_end_token_id             = tp->PieceToId("<|im_end|>");

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        CHATLLM_CHECK(im_end_token_id >= 0) << "<|im_end|> is missing. Forget to download `added_tokens.json`?";

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
        ids.push_back(tok->bos_token_id);
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

    const int SLIDING_WINDOW_LEN            =  2048;

    template <int sliding_window_len> class SolarSelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        SolarSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        SolarSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    typedef LMBlock1<RMSNorm, SolarSelfAttention<SLIDING_WINDOW_LEN>, RMSNorm, SiLUMLP> SolarBlock2k;

    class SolarModel : public Model<BaseConfig, Embedding, RMSNorm, SolarBlock2k, int, int, int, int, int>
    {
    public:
        SolarModel(InitContext *ctx, const Config &config, bool lm_head_bias, int hidden_size, int num_attention_heads,
            int intermediate_size, int num_key_value_heads, int max_length)
        : Model(ctx, config, lm_head_bias, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length),
          forwarded_hidden_states(config.num_hidden_layers, nullptr),
          forwarded_to_layer(config.num_hidden_layers, -1),
          alpha(0.0f)
        {

        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override
        {
            before_forward(ctx, input_ids, n_past);

            ctx->move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
            for (auto &layer : HeterogeneousModel::layers)
            {
                int id = layer->get_id();
                ctx->move_to_layer(id);

                if (forwarded_hidden_states[id])
                {
                    auto forwarded     = ggml::scale(ctx, forwarded_hidden_states[id], alpha);
                    hidden_states      = ggml::scale(ctx, forwarded_hidden_states[id], 1.0f - alpha);
                    hidden_states      = ggml::add(ctx, forwarded, hidden_states);
                }

                hidden_states = layer->forward(ctx, hidden_states, n_past);

                if (forwarded_to_layer[id] >= 0)
                {
                    forwarded_hidden_states[forwarded_to_layer[id]] = hidden_states;
                }
            }

            ctx->move_to_layer(LayerAllocatorManager::Epilog);
            return final_steps(ctx, input_ids, hidden_states);
        }

        void init_layer_fowarding(float alpha, int pair_num, const int *pairs)
        {
            this->alpha = alpha;
            for (int i = 0; i < pair_num; i++, pairs += 2)
            {
                forwarded_to_layer[pairs[0]] = pairs[1];
            }
        }

    protected:
        std::vector<ggml::tensor *> forwarded_hidden_states;
        std::vector<int> forwarded_to_layer;
        float alpha;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_SOLARPRO)
            : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * 13;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK(config.sliding_window == SLIDING_WINDOW_LEN - 1)
                << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN - 1;

            transformer = new SolarModel(&w_ctx_, config, false,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention =  get_typed_transformer<SolarModel>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }

            get_typed_transformer<SolarModel>()->init_layer_fowarding(config.layer_alpha, config.pairs_num, config.layer_forwarding_ids);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " vs "
                << w_ctx_.get_mem_size() / ggml_tensor_overhead();

            batch_input = false;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<SolarModel>();
            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer->layers[i].mlp.up_proj.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

            if (transformer->lm_head)
                loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);
        }
    protected:
        Config config;
    };
}