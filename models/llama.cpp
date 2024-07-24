namespace v2
{
    struct Config : public BaseConfig
    {
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
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = R"""(You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.)""";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

    public:
        virtual void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
    };

    template <class LayerBlock> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<BaseConfig, Embedding, RMSNorm, LayerBlock, int, int, int, int, int> ModelClass;
        typedef Model<BaseConfig, Embedding, RMSNorm, LayerBlock, int, int, int, int, int, int> ModelClass2;
    public:
        GenericConditionalGeneration(const BaseConfig &config, ModelType type, int num_key_value_heads, int max_length, int tensors_per_layer = 12, bool tie_lm_head = false)
            : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config), type_class(1)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * tensors_per_layer;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (tie_lm_head)
                Base::transformer = new ModelClass(&w_ctx_, config, nullptr,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, max_length);
            else
                Base::transformer = new ModelClass(&w_ctx_, config, false,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, max_length);
            Base::GRAPH_SIZE = 4096 * 2;
        }

        GenericConditionalGeneration(const BaseConfig &config, ModelType type, int num_key_value_heads, int head_dim, int max_length, int tensors_per_layer, bool tie_lm_head)
            : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config), type_class(2)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * tensors_per_layer;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (tie_lm_head)
                Base::transformer = new ModelClass2(&w_ctx_, config, nullptr,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, head_dim, max_length);
            else
                Base::transformer = new ModelClass2(&w_ctx_, config, false,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, head_dim, max_length);
            Base::GRAPH_SIZE = 4096 * 2;
        }

        void load(ModelLoader &loader) override
        {
            switch (type_class)
            {
            case 1:
                load0<ModelClass>(loader);
                break;
            default:
                load0<ModelClass2>(loader);
                break;
            }
        }

    protected:
        template <class T> void load0(ModelLoader &loader)
        {
            auto transformer = Base::get_typed_transformer<T>();
            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';
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

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

    public:
        static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 2844ull * 1024 * 1024;

        BaseConfig config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
        const int type_class;
    };

    class ConditionalGeneration : public GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_LLAMA2)
            : ConditionalGeneration(config, type, config.num_attention_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, ModelType type,
                            int num_key_value_heads, int max_length)
            : GenericConditionalGeneration<LlamaBlock>(config, type, num_key_value_heads, max_length)
        {}
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        pad_token_id = tp->PieceToId("<pad>");
        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
    {
        if (add_bos)
            ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            ids.push_back(eos_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false);
    }

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

        oss_prompt << "<<SYS>>\n" << tok->get_system_prompt() << "\n<</SYS>>\n\n";
        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "[INST] " + user << "[/INST] ";

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return id == pad_token_id;
    }
}

namespace v3
{
    struct Config : public v2::Config
    {
        int num_key_value_heads;
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

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            start_header_id = tp->PieceToId("<|start_header_id|>");
            end_header_id   = tp->PieceToId("<|end_header_id|>");
            eot_id          = tp->PieceToId("<|eot_id|>");

            std::vector<int> ids;
            tp->Encode("\n", &ids);
            nl_token_id = ids[0];

            terminate_ids.insert(eot_id);

            return size;
        }

        bool is_special_id(int id) const override
        {
            return (id == start_header_id) || (id == end_header_id) || (id == eot_id);
        }

        void encode_header(const std::string &text, std::vector<int> &ids) const
        {
            ids.push_back(start_header_id);
            encode(text, ids);
            ids.push_back(end_header_id);
            ids.push_back(nl_token_id);
            ids.push_back(nl_token_id);
        }

        void encode_content(const std::string &text, std::vector<int> &ids) const
        {
            encode(text, ids);
            ids.push_back(eot_id);
        }

    public:
        int start_header_id;
        int end_header_id;
        int eot_id;
        int nl_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eot_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);
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
        std::ostringstream oss;

        tok->encode_header("assistant", ids);
    }

    class ConditionalGeneration : public v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_LLAMA3)
            : ConditionalGeneration(config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, ModelType type,
                            int num_key_value_heads, int max_length)
            : v2::GenericConditionalGeneration<LlamaBlock>(config, type, num_key_value_heads, max_length)
        {
            auto transformer = Base::get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }
        }
    };
}

namespace v3_1
{
    struct Config : public v3::Config
    {
        int rope_scaling_original_max_position_embeddings;
        float rope_scaling_factor;
        float rope_scaling_low_freq_factor;
        float rope_scaling_high_freq_factor;
    };

    class ChatHistoryEncoder : public v3::ChatHistoryEncoder
    {
    public:
        void append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : v3::Tokenizer(config, &_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t size = v3::Tokenizer::load(buffer, n_vocab);

            eom_id          = tp->PieceToId("<|eom_id|>");
            python_tag_id   = tp->PieceToId("<|python_tag|>");

            terminate_ids.insert(eom_id);

            tp->EnableReturnSpecialToken(true);

            return size;
        }
    public:
        int eom_id;
        int python_tag_id;
    };

    void ChatHistoryEncoder::append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        tok->encode_header("ipython", ids);
        tok->encode_content(content, ids);
    }

    template <class BaseAttn> class LlamaRoPESelfAttention : public BaseAttn
    {
    public:
        LlamaRoPESelfAttention() : BaseAttention() {}
        LlamaRoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias, bool o_bias)
            : LlamaRoPESelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, qkv_bias, o_bias)
        {
        }

        LlamaRoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : LlamaRoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads,  max_length, qkv_bias, o_bias)
        {
        }

        LlamaRoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttn(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias),
              rope_dim(head_dim)
        {
        }

        void config_rope(float theta, float scaling_factor, float scaling_low_freq_factor, float scaling_high_freq_factor, int original_max_position_embeddings)
        {
            rope.inv_freq.clear();
            rope.scaling_factor = 1.0f;
            const float base = theta;

            for (int i = 0; i < rope_dim; i += 2)
            {
                rope.inv_freq.push_back(1.0f / (powf(base,  (float)i / rope_dim)));
            }

            const float factor = scaling_factor;
            const float low_freq_factor  = scaling_low_freq_factor;
            const float high_freq_factor = scaling_high_freq_factor;
            const int   old_context_len  = original_max_position_embeddings;

            const float low_freq_wavelen = old_context_len / low_freq_factor;
            const float high_freq_wavelen = old_context_len / high_freq_factor;

            for (size_t i = 0; i < rope.inv_freq.size(); i++)
            {
                const float freq = rope.inv_freq[i];
                const float wavelen = 2 * std::numbers::pi_v<float> / freq;
                if (wavelen < high_freq_wavelen)
                {
                    ;
                }
                else if (wavelen > low_freq_wavelen)
                {
                    rope.inv_freq[i] = freq / factor;
                }
                else
                {
                    const float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                    rope.inv_freq[i] = (1 - smooth) * freq / factor + smooth * freq;
                }
            }
        }

    public:
        int rope_dim;
        CustomInvFreqScalingRope rope;

    protected:
        // input & output: [qlen, heads, head_size]
        ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return ggml_map_custom2(ctx->gctx.get(), k, past, ggml_compute_forward_inv_freq_scaling_rope, GGML_N_TASKS_MAX, (void *)&rope);
        }
        ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
        {
            return ggml_map_custom2(ctx->gctx.get(), q, past, ggml_compute_forward_inv_freq_scaling_rope, GGML_N_TASKS_MAX, (void *)&rope);
        }
    };

    class Llama3_1SelfAttention : public LlamaRoPESelfAttention<BaseAttention>
    {
    public:
        Llama3_1SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : LlamaRoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    class Llama3_1Block : public LMBlock1<RMSNorm, Llama3_1SelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        Llama3_1Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class ToolCallingInterceptor : public ChunkInterceptor
    {
    public:
        ToolCallingInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(streamer->tokenizer);

            if (!found_tool_call)
            {
                if (chunk == "<|python_tag|>")
                {
                    found_tool_call = true;
                }
                else
                    streamer->put_chunk(first, chunk);
            }
            else
            {
                oss << chunk;
            }
        }

        void end() override
        {
            found_tool_call = false;
            auto s = oss.str();
            oss.str("");
            if (s.size() > 0)
                streamer->putln(s.c_str(), BaseStreamer::TextType::TOOL_CALLING);

            ChunkInterceptor::end();
        }
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };

    static class ToolCallingInterceptor interceptor;

    class ConditionalGeneration : public v2::GenericConditionalGeneration<Llama3_1Block>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_LLAMA3_1)
            : ConditionalGeneration(config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, ModelType type,
                            int num_key_value_heads, int max_length)
            : v2::GenericConditionalGeneration<Llama3_1Block>(config, type, num_key_value_heads, max_length)
        {
            auto transformer = Base::get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.config_rope(config.rope_theta,
                                      config.rope_scaling_factor,
                                      config.rope_scaling_low_freq_factor,
                                      config.rope_scaling_high_freq_factor,
                                      config.rope_scaling_original_max_position_embeddings);
            }
        }

        ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
    };
}

namespace v2_plus
{
    typedef v3::Config Config;
    typedef v2::Tokenizer Tokenizer;

    class ConditionalGeneration : public v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_LLAMA2)
            : v3::ConditionalGeneration(config, type)
        {}
    };
}

namespace multi
{
    struct Config : public v3::Config
    {
        int n_future_tokens;
    };

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : v2::Tokenizer(config)
        {
           // sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = v2::Tokenizer::load(buffer, n_vocab);
            return r;
        }
    };

    class LlamaBlockNonInplace : public LMBlock1<RMSNormNonInplace, LlamaSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        LlamaBlockNonInplace(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    // MultiPredModel seems better to inherit from HeterogeneousModel (or move into it)
    // If there are more models use this technique, then DO IT.
    class MultiPredModel : public Model<BaseConfig, Embedding, RMSNorm, LlamaBlockNonInplace, int, int, int, int, int>
    {
    public:
        typedef Model<BaseConfig, Embedding, RMSNorm, LlamaBlockNonInplace, int, int, int, int, int> Base;
        MultiPredModel(InitContext *ctx, const Config &config, int hidden_size, int num_attention_heads, int intermediate_size,
                       int num_key_value_heads, int max_length, int n_future_tokens)
            : Model<BaseConfig, Embedding, RMSNorm, LlamaBlockNonInplace, int, int, int, int, int>(
                    ctx, config, false, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length),
              n_future_tokens(n_future_tokens)
        {
            size_t extra_cache_size = 0;
            effective_n = n_future_tokens;

            for (int i = 0; i < n_future_tokens - 1; i++)
            {
                auto layer = new LlamaBlockNonInplace(ctx, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length);
                extra_heads.push_back(layer);
                layer->set_id(config.num_hidden_layers + i);
                extra_cache_size += layer->get_cache_size();;
            }

            extra_cache.resize(extra_cache_size);
            void *buffer = (void *)extra_cache.data();

            prediction_heads.push_back(get_layer(config.num_hidden_layers - 1));
            for (int i = 0; i < n_future_tokens - 1; i++)
            {
                auto layer = extra_heads[i];
                buffer = layer->set_cache_buffer(buffer);
                prediction_heads.push_back(layer);
            }
        }

        ~MultiPredModel()
        {
        }

        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past) override
        {
            ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
            for (int i = 0; i <= config.num_hidden_layers - 2; i++)
            {
                // reserve scratch memory for h_truck
                if (i <= config.num_hidden_layers - 3)
                    ggml_set_scratch(ctx->gctx.get(), ctx->scratch);
                hidden_states = get_layer(i)->forward(ctx, hidden_states, n_past);
            }

            auto h_trunk = hidden_states;
            ggml_tensor *lm_logits = ggml_new_tensor_2d(ctx->gctx.get(), GGML_TYPE_F32, config.vocab_size, effective_n);

            for (int i = 0; i < effective_n; i++)
            {
                ggml_set_scratch(ctx->gctx.get(), ctx->scratch);
                ggml_tensor *tok_states = prediction_heads[i]->forward(ctx, h_trunk, n_past);

                ggml_set_scratch(ctx->gctx.get(), ctx->scratch);

                ggml_tensor *logits = calc_logits(ctx, input_ids, tok_states);
                ggml_tensor *view = ggml_view_1d(ctx->gctx.get(), lm_logits, config.vocab_size,
                                                 i * ggml_nbytes(logits));
                ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), logits, view));
            }

            return lm_logits;
        }

        int set_n_future_tokens(int n)
        {
            if ((1 <= n) && (n <= n_future_tokens))
                effective_n = n;
            return effective_n;
        }

    protected:
        int64_t get_param_num_of_layers(bool effective_only) const override
        {
            int64_t r = Base::get_param_num_of_layers(effective_only);
            if (extra_heads.size() > 0)
                r += extra_heads[0]->get_param_num(effective_only) * extra_heads.size();
            return r;
        }

        ggml_tensor *calc_logits(ForwardContext *ctx, ggml_tensor *input_ids, ggml_tensor *hidden_states)
        {
            // NOTE: only compute next_token_logits for the last token
            hidden_states = ggml_view_2d(ctx->gctx.get(), hidden_states, config.hidden_size, 1,
                                        config.hidden_size * ggml_element_size(hidden_states),
                                        (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(hidden_states));

            ggml_tensor *transformer_outputs = final_layernorm.forward(ctx, hidden_states);

            transformer_outputs =
                    ggml_view_1d(ctx->gctx.get(), transformer_outputs, config.hidden_size, 0);

            ggml_tensor *logits = lm_head ? lm_head->forward(ctx, transformer_outputs)
                                             : word_embeddings.forward(ctx, transformer_outputs);

            if (logits_pp)
                logits = logits_pp->forward(ctx, logits);
            return logits;
        }
    public:
        const int n_future_tokens;
        std::vector<LlamaBlockNonInplace *> extra_heads;
        std::vector<Block *> prediction_heads;
        std::vector<uint8_t> extra_cache;
    private:
        int effective_n;
    };


    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    private:
        typedef BaseModelForConditionalGeneration Base;
    public:
        ConditionalGeneration(const Config &config, ModelType type = MODEL_TYPE_LLAMA_MULTI,
                                     int tensors_per_layer = 12)
            : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 3 + (config.num_hidden_layers + config.n_future_tokens - 1) * tensors_per_layer;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            Base::transformer = new MultiPredModel(&w_ctx_, config,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, config.num_key_value_heads, config.max_length,
                                                    config.n_future_tokens);
            Base::GRAPH_SIZE = 4096 * 2;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = Base::get_typed_transformer<MultiPredModel>();
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

            for (int i = 0; i < (int)transformer->extra_heads.size(); i++)
            {
                std::string layer_prefix = "model.extra_heads." + std::to_string(i) + '.';
                loader.read_tensor(layer_prefix + "input_layernorm.weight",             transformer->extra_heads[i]->input_layernorm.weight);
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight",               transformer->extra_heads[i]->mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",               transformer->extra_heads[i]->mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",                 transformer->extra_heads[i]->mlp.up_proj.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",    transformer->extra_heads[i]->post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->extra_heads[i]->attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->extra_heads[i]->attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->extra_heads[i]->attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->extra_heads[i]->attention.v_proj.weight);
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            #define get_value(n) do {                       \
                auto it = args.find(#n);                    \
                if (it != args.end()) n = it->second;       \
            } while (0)

            std::string n_future_tokens("");
            get_value(n_future_tokens);
            if (n_future_tokens.size() > 0)
                dynamic_cast<MultiPredModel *>(Base::transformer)->set_n_future_tokens(std::atoi(n_future_tokens.c_str()));
        }

    public:
        static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 2844ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };
}