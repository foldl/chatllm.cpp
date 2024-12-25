namespace v2
{

    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int base_seqlen;
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
        Tokenizer(const Config &config)
            : BaseTokenizer(config, &_chat_encoder)
        {
            sys_prompt = "你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab)
        {
            tp = new tokenizer::BPEProcessor1();
            size_t size = tp->Load(buffer, n_vocab);

            _end_token_id = eos_token_id;

            int id = pad_token_id;
            _user_token_id = ++id;
            _bot_token_id = ++id;
            _system_token_id = ++id;
            _observation_token_id = ++id;
            _sep_token_id = ++id;
            tool_call_token_id = ++id;
            tool_call_end_token_id = ++id;
            tool_response_token_id = ++id;
            tool_response_end_token_id = ++id;
            nl_token_id = tp->PieceToId("\n");

            return size;
        }

    public:
        int _user_token_id;
        int _bot_token_id;
        int _end_token_id;
        int _system_token_id;
        int _observation_token_id;
        int _sep_token_id;
        int tool_call_token_id;
        int tool_call_end_token_id;
        int tool_response_token_id;
        int tool_response_end_token_id;
        int nl_token_id;
    };

    class TeleChat2SelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        TeleChat2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, true) {}
    };

    class TeleChat2MLP : public BaseMLP
    {
    public:
        TeleChat2MLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::SILU, false, true, false)
        {}
    };

    class TeleChat2Block : public LMBlock1<RMSNorm, TeleChat2SelfAttention, RMSNorm, TeleChat2MLP>
    {
    public:
        TeleChat2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    double log_base(double a, double b) {
        return log(a) / log(b);
    }

    static float get_ntk_alpha(const int true_seq_len, const int base_seqlen)
    {
        double context_value = log_base((double)true_seq_len / base_seqlen, 2.0) + 1;
        double ntk_alpha = pow(2.0, ceil(context_value)) - 1;
        return (float)(ntk_alpha > 1.0 ? ntk_alpha : 1.0);
    }

    static float get_mscale(float scale)
    {
        if (scale <= 1.0f) return 1.0f;
        return 0.1f * logf(scale) + 1.0f;
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, TeleChat2Block, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_TELECHAT2)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * 14;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            // TODO: support NTK scalling
            transformer = new ModelClass(&w_ctx_, config, false,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_key_value_heads, config.max_length);


            const int head_dim = config.hidden_size / config.num_attention_heads;
            const float base = 10000.0f;
            float ntk_alpha = get_ntk_alpha(config.max_length, config.base_seqlen);
            const float freq_base = base * powf(ntk_alpha, (float)head_dim / (head_dim - 2));
            const float mscale = get_mscale((float)config.max_length / config.base_seqlen);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = freq_base;
                attention.attn_factor = mscale;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                loader.read_tensor(layer_prefix + "input_layernorm.weight",
                                transformer->layers[i].input_layernorm.weight);
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.down_proj.bias",   transformer->layers[i].mlp.down_proj.bias);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids);
        ids.push_back(tok->_end_token_id);
        ids.push_back(tok->nl_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (tok->get_system_prompt().size() > 0)
        {
            ids.push_back(tok->_system_token_id);
            tok->encode(tok->get_system_prompt(), ids);
            ids.push_back(tok->nl_token_id);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->_user_token_id);
        tok->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->_bot_token_id);
    }
}