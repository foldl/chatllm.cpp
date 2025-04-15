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

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
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
        GenericConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type, int num_key_value_heads, int max_length, int tensors_per_layer = 12, bool tie_lm_head = false, int additional_tensor = 0)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config), type_class(1)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * tensors_per_layer + additional_tensor;
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
        }

        GenericConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type, int num_key_value_heads, int head_dim, int max_length, int tensors_per_layer, bool tie_lm_head)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config), type_class(2)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
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

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " vs "
                << w_ctx_.get_mem_size() / ggml_tensor_overhead();
        }

    public:
        BaseConfig config;

    protected:
        const int type_class;
    };

    class ConditionalGeneration : public GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA2)
            : ConditionalGeneration(config, runtime_config, type, config.num_attention_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length)
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

            if (eot_id >= 0)
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
        tok->encode_header("assistant", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_header("user", ids);
    }

    class ConditionalGeneration : public v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA3)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length)
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

    class ToolCallingInterceptor : public ChunkInterceptor
    {
    public:
        ToolCallingInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            if (!found_tool_call)
            {
                if (chunk == "<|python_tag|>")
                {
                    found_tool_call = true;
                }
                else
                    next->put_chunk(first, chunk);
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

    void init_llama3_freq_factors(std::vector<float> &freq_factors, int rope_dim,
                float theta, float scaling_factor, float scaling_low_freq_factor, float scaling_high_freq_factor, int original_max_position_embeddings)
    {
        std::vector<float > inv_freq;
        const float base = theta;

        for (int i = 0; i < rope_dim; i += 2)
        {
            inv_freq.push_back(1.0f / (powf(base,  (float)i / rope_dim)));
        }

        freq_factors.clear();

        const float factor = scaling_factor;
        const float low_freq_factor  = scaling_low_freq_factor;
        const float high_freq_factor = scaling_high_freq_factor;
        const int   old_context_len  = original_max_position_embeddings;

        const float low_freq_wavelen = old_context_len / low_freq_factor;
        const float high_freq_wavelen = old_context_len / high_freq_factor;

        for (int i = 0; i < (int)inv_freq.size(); i++)
        {
            const float freq = inv_freq[i];
            const float wavelen = 2 * std::numbers::pi_v<float> / freq;
            if (wavelen < high_freq_wavelen)
            {
                freq_factors.push_back(1.0f);
            }
            else if (wavelen > low_freq_wavelen)
            {
                freq_factors.push_back(1.0f / factor);
            }
            else
            {
                const float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                freq_factors.push_back((1 - smooth) / factor + smooth);
            }
        }

        for (int i = 0; i < (int)freq_factors.size(); i++)
        {
            // factor is used as a divider in ggml
            freq_factors[i] = 1 / freq_factors[i];
        }
    }

    class ConditionalGeneration : public v2::GenericConditionalGeneration<Llama31Block>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA3_1)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length, int tensors_per_layer = 13, bool tie_lm_head = false, int additional_tensor = 0)
            : v2::GenericConditionalGeneration<Llama31Block>(config, runtime_config, type, num_key_value_heads, max_length, tensors_per_layer, tie_lm_head, additional_tensor)
        {
            init_llama3_freq_factors(freq_factors_value, config.hidden_size / config.num_attention_heads,
                                      config.rope_theta,
                                      config.rope_scaling_factor,
                                      config.rope_scaling_low_freq_factor,
                                      config.rope_scaling_high_freq_factor,
                                      config.rope_scaling_original_max_position_embeddings);

            auto transformer = Base::get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.freq_base    = config.rope_theta;
                Backend::write_tensor_data(attention.freq_factors, freq_factors_value.data());
            }
        }

        ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
    protected:
        std::vector<float> freq_factors_value;
    };
}

namespace v3_2
{
    struct Config : public v3_1::Config
    {
        int tie_word_embeddings;
    };

    typedef v3_1::Tokenizer Tokenizer;

    class ConditionalGeneration : public v3_1::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA3_2)
            : v3_1::ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length,
                                          13, config.tie_word_embeddings != 0, 0)
        {}
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
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA2)
            : v3::ConditionalGeneration(config, runtime_config, type)
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

    class LlamaBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    // MultiPredModel seems better to inherit from HeterogeneousModel (or move into it)
    // If there are more models use this technique, then DO IT.
    class MultiPredModel : public Model<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int>
    {
    public:
        typedef Model<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int> Base;
        MultiPredModel(InitContext *ctx, const Config &config, int hidden_size, int num_attention_heads, int intermediate_size,
                       int num_key_value_heads, int max_length, int n_future_tokens)
            : Model<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int>(
                    ctx, config, false, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length),
              n_future_tokens(n_future_tokens)
        {
            effective_n = n_future_tokens;

            for (int i = 0; i < n_future_tokens - 1; i++)
            {
                auto layer = new LlamaBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length);
                extra_heads.push_back(layer);
                layer->set_id(config.num_hidden_layers + i);
            }

            prediction_heads.push_back(get_layer(config.num_hidden_layers - 1));
            for (int i = 0; i < n_future_tokens - 1; i++)
            {
                auto layer = extra_heads[i];
                prediction_heads.push_back(layer);

                ctx->move_to_layer(i);
                auto allocator = ctx->get_allocator();
                auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
                layer->set_cache_buffer(buf);
            }
        }

        ~MultiPredModel()
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override
        {
            ggml::tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
            for (int i = 0; i <= config.num_hidden_layers - 2; i++)
            {
                hidden_states = get_layer(i)->forward(ctx, hidden_states, n_past);
            }

            auto h_trunk = hidden_states;
            ggml::tensor *lm_logits = ggml::new_tensor_2d(ctx, GGML_TYPE_F32, config.vocab_size, effective_n);

            for (int i = 0; i < effective_n; i++)
            {
                ggml::tensor *tok_states = prediction_heads[i]->forward(ctx, h_trunk, n_past);

                ggml::tensor *logits = calc_logits(ctx, input_ids, tok_states);
                ggml::tensor *view = ggml::view_1d(ctx, lm_logits, config.vocab_size,
                                                 i * ggml::nbytes(logits));
                ggml::build_forward_expand(ctx, ggml::cpy(ctx, logits, view));
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

        ggml::tensor *calc_logits(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
        {
            // NOTE: only compute next_token_logits for the last token
            hidden_states = ggml::view_2d(ctx, hidden_states, config.hidden_size, 1,
                                        config.hidden_size * ggml::element_size(hidden_states),
                                        (input_ids->ne[0] - 1) * config.hidden_size * ggml::element_size(hidden_states));

            ggml::tensor *transformer_outputs = final_layernorm.forward(ctx, hidden_states);

            transformer_outputs =
                    ggml::view_1d(ctx, transformer_outputs, config.hidden_size, 0);

            ggml::tensor *logits = lm_head ? lm_head->forward(ctx, transformer_outputs)
                                             : word_embeddings.forward(ctx, transformer_outputs);

            if (logits_pp)
                logits = logits_pp->forward(ctx, logits);
            return logits;
        }
    public:
        const int n_future_tokens;
        std::vector<LlamaBlock *> extra_heads;
        std::vector<Block *> prediction_heads;
    private:
        int effective_n;
    };


    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    private:
        typedef BaseModelForConditionalGeneration Base;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_LLAMA_MULTI,
                                     int tensors_per_layer = 12)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + (config.num_hidden_layers + config.n_future_tokens - 1) * tensors_per_layer;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            Base::transformer = new MultiPredModel(&w_ctx_, config,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, config.num_key_value_heads, config.max_length,
                                                    config.n_future_tokens);
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

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
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
        Config config;
    };
}

namespace ds_r1_distill
{
    typedef v3_2::Config Config;

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

    class ConditionalGeneration : public v3_2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA)
            : v3_2::ConditionalGeneration(config, runtime_config, type)
        {}
    };
}

namespace v4
{
    struct Config : public v2::Config
    {
        int num_key_value_heads;
        int attention_chunk_size;
        int head_dim;
        int interleave_moe_layer_step;
        int intermediate_size_mlp;
        int num_experts_per_tok;
        int n_routed_experts;
        int use_qk_norm;

        float router_aux_loss_coef;
        float rope_theta;
        int   rope_scaling_original_max_position_embeddings;
        float rope_scaling_factor;
        float rope_scaling_low_freq_factor;
        float rope_scaling_high_freq_factor;
    };

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : v3::Tokenizer(config)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            // TODO: FIX regexp
            size_t size = v3::Tokenizer::load(buffer, n_vocab);

            start_header_id = tp->PieceToId("<|header_start|>");
            end_header_id   = tp->PieceToId("<|header_end|>");
            eot_id          = tp->PieceToId("<|eot|>");

            terminate_ids.insert(eot_id);

            return size;
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class LlamaSparseMoE : public BaseSparseMLP
    {
    public:
        LlamaSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
            score_func = ScoreFunc::Sigmoid;
            norm_topk_prob = false;
            pre_weighting  = true;
        }
    };

    class LlamaNormedSelfAttention : public QKNormedAttention<L2Norm, BaseAttention>
    {
    public:
        LlamaNormedSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<L2Norm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            post_norm = true;
        }
    };

    template <class SelfAttention, class MoEMLP> class LlamaMoEBlock_ : public LMBlock1<RMSNorm, SelfAttention, RMSNorm, MoEMLP>
    {
    public:
        LlamaMoEBlock_(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : LMBlock1<RMSNorm, SelfAttention, RMSNorm, MoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
              num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK, class SelfAttention> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<LlamaSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> LlamaMoEMLP;
        typedef LlamaMoEBlock_<SelfAttention, LlamaMoEMLP> LlamaMoEBlock;
        typedef LMBlock1<RMSNorm, SelfAttention, RMSNorm, SiLUMLP> LlamaDenseBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel<Config, Embedding, RMSNorm> ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_LLAMA4)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            init_moe_vector();

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3 + moe_layer_num * (13 + 3) + dense_layer_num * 13 +
                (config.rope_scaling_original_max_position_embeddings > 0 ?  config.num_hidden_layers : 0);

            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK))
                << "unsupported MoE param";

            CHATLLM_CHECK(config.max_length <= 8192)
                << "TODO: support longer context (attn_temperature_tuning)." << std::endl
                << "use `--max_length 8192` to specify a shorter context.";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new LlamaMoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.intermediate_size, config.intermediate_size,
                        config.num_key_value_heads, config.head_dim, config.max_length);
                    return layer;
                }
                else
                {
                    return new LlamaDenseBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size_mlp,
                                                config.num_key_value_heads, config.head_dim, config.max_length);
                }
            };

            auto transformer = new ModelClass(
                &w_ctx_, config, false, create_layer);
            Base::transformer = transformer;

            std::vector<float> freq_factors_value;
            if (config.rope_scaling_original_max_position_embeddings > 0)
            {
                v3_1::init_llama3_freq_factors(freq_factors_value, config.head_dim,
                    config.rope_theta,
                    config.rope_scaling_factor,
                    config.rope_scaling_low_freq_factor,
                    config.rope_scaling_high_freq_factor,
                    config.rope_scaling_original_max_position_embeddings);
            }

            #define config_rope(attention)     do { \
                    attention.use_rope     = (i + 1) % 4 != 0;      \
                    attention.freq_base    = config.rope_theta;     \
                    attention.rope_mode = RoPEMode::Original;       \
                    if (config.rope_scaling_original_max_position_embeddings > 0) { \
                        attention.freq_factors = ggml::new_tensor_1d(&w_ctx_, GGML_TYPE_F32, config.head_dim / 2);  \
                        w_ctx_.get_allocator()->alloc(attention.freq_factors);                                      \
                        Backend::write_tensor_data(attention.freq_factors, freq_factors_value.data());  \
                    }                                               \
                } while (false)

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<LlamaMoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
                else
                {
                    auto *layer = dynamic_cast<LlamaDenseBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                    << "corrupted model weights: " << w_ctx_.get_used_mem() / tensor_ovhd << " vs " << w_ctx_.get_mem_size() / tensor_ovhd;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            SelfAttention *attention = nullptr;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<LlamaMoEBlock *>(transformer->get_layer(i));
                    attention = &layer->attention;

                    loader.read_tensor(layer_prefix + "mlp.mlp1.experts_down.weight", layer_prefix + "mlp.experts.", config.n_routed_experts, ".down_proj.weight", layer->mlp.mlp1.experts_down.weight);
                    loader.read_tensor(layer_prefix + "mlp.mlp1.experts_gate.weight", layer_prefix + "mlp.experts.", config.n_routed_experts, ".gate_proj.weight", layer->mlp.mlp1.experts_gate.weight);
                    loader.read_tensor(layer_prefix + "mlp.mlp1.experts_up.weight",   layer_prefix + "mlp.experts.", config.n_routed_experts, ".up_proj.weight",   layer->mlp.mlp1.experts_up.weight);

                    loader.read_tensor(layer_prefix + "mlp.gate.weight", layer->mlp.mlp1.gate.weight);

                    loader.read_tensor(layer_prefix + "mlp.shared_expert.down_proj.weight", layer->mlp.mlp2.down_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.shared_expert.gate_proj.weight", layer->mlp.mlp2.gate_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.shared_expert.up_proj.weight",   layer->mlp.mlp2.up_proj.weight);

                    loader.read_tensor(layer_prefix + "input_layernorm.weight",          layer->input_layernorm.weight);
                    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);
                }
                else
                {
                    auto *layer = dynamic_cast<LlamaDenseBlock *>(transformer->get_layer(i));
                    attention = &layer->attention;

                    loader.read_tensor(layer_prefix + "mlp.down_proj.weight", layer->mlp.down_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", layer->mlp.gate_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   layer->mlp.up_proj.weight);

                    loader.read_tensor(layer_prefix + "input_layernorm.weight",          layer->input_layernorm.weight);
                    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);
                }

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", attention->k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", attention->o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", attention->q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", attention->v_proj.weight);
            }
        }

    private:
        void init_moe_vector()
        {
            layer_is_moe.resize(config.num_hidden_layers, false);
            for (int i = config.interleave_moe_layer_step - 1; i < config.num_hidden_layers; i += config.interleave_moe_layer_step)
                layer_is_moe[i] = true;
        }

    public:
        Config config;
        std::vector<bool> layer_is_moe;

        bool is_layer_moe(int layer_index)
        {
            return layer_is_moe[layer_index];
        }

        int get_moe_layer_num()
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                    r++;
            }
            return r;
        }
    };

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config) : ModelProxy()
        {
            if (config.use_qk_norm)
            {
                switch (config.n_routed_experts)
                {
                case 16:
                    set_proxy_model(new ConditionalGeneration0<16, 1, 1, LlamaNormedSelfAttention>(config, runtime_config));
                    break;
                default:
                    CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.n_routed_experts;
                    break;
                }
            }
            else
            {
                switch (config.n_routed_experts)
                {
                case 128:
                    set_proxy_model(new ConditionalGeneration0<128, 1, 1, LlamaSelfAttention>(config, runtime_config));
                    break;
                default:
                    CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.n_routed_experts;
                    break;
                }
            }
        }
    };
}