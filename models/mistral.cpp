namespace mistral
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        int sliding_window;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    const int SLIDING_WINDOW_LEN            =  4096;

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const llama::v2::Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const llama::v2::Config &config, BaseHistoryEncoder *encoder)
            : llama::v2::Tokenizer::Tokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = llama::v2::Tokenizer::load(buffer, n_vocab);
            if (tp->GetPieceSize() == 32768)
            {
                // Mistral v0.3
                start_inst_token_id             = 3;
                end_inst_token_id               = 4;
                tool_calls_token_id             = 5;
                start_avail_tools_token_id      = 6;
                end_avail_tools_token_id        = 7;
                start_tool_results_token_id     = 8;
                end_tool_results_token_id       = 9;
                tp->AddAddedToken("[INST]",             start_inst_token_id);
                tp->AddAddedToken("[/INST]",            end_inst_token_id);
                tp->AddAddedToken("[TOOL_CALLS]",       tool_calls_token_id);
                tp->AddAddedToken("[AVAILABLE_TOOLS]",  start_avail_tools_token_id);
                tp->AddAddedToken("[/AVAILABLE_TOOLS]", end_avail_tools_token_id);
                tp->AddAddedToken("[TOOL_RESULTS]",     start_tool_results_token_id);
                tp->AddAddedToken("[/TOOL_RESULTS]",    end_tool_results_token_id);
            }
            else
            {
                start_inst_token_id             = tp->PieceToId("[INST]");
                end_inst_token_id               = tp->PieceToId("[/INST]");
                tool_calls_token_id             = tp->PieceToId("[TOOL_CALLS]");
                start_avail_tools_token_id      = tp->PieceToId("[AVAILABLE_TOOLS]");
                end_avail_tools_token_id        = tp->PieceToId("[/AVAILABLE_TOOLS]");
                start_tool_results_token_id     = tp->PieceToId("[TOOL_RESULTS]");
                end_tool_results_token_id       = tp->PieceToId("[/TOOL_RESULTS]");
            }
            return r;
        }
    public:
        int start_inst_token_id;
        int end_inst_token_id;
        int tool_calls_token_id;
        int start_avail_tools_token_id;
        int end_avail_tools_token_id;
        int start_tool_results_token_id;
        int end_tool_results_token_id;
    };

    class MistralInterceptor : public ChunkInterceptor
    {
    public:
        MistralInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(streamer->tokenizer);
            if (tok->start_avail_tools_token_id < 0)
            {
                streamer->put_chunk(first, chunk);
                return;
            }

            if (first)
            {
                found_tool_call = chunk.starts_with("[TOOL_CALLS]");
                if (found_tool_call) return;
            }

            if (found_tool_call)
                oss << chunk;
            else
                streamer->put_chunk(first, chunk);
        }

        void end() override
        {
            if (found_tool_call)
                streamer->putln(oss.str(), BaseStreamer::TextType::TOOL_CALLING);

            oss.str("");
            found_tool_call = false;

            ChunkInterceptor::end();
        }
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };

    static MistralInterceptor interceptor;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
            : llama::v2::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>(config, runtime_config, type,
                config.num_key_value_heads, config.max_length, 13)
        {
            CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == SLIDING_WINDOW_LEN))
                << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }

            batch_input = false;
        }

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ConditionalGeneration(config, runtime_config, MODEL_TYPE_MISTRAL)
        {
        }

        ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[TOOL_RESULTS]" << content << "[/TOOL_RESULTS]";

        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        if (tok->start_avail_tools_token_id >= 0)
        {
            if (user.starts_with("[TOOL_RESULTS]"))
            {
                oss_prompt << user;
            }
            else
            {
                std::string user_input = user;
                const std::string tag_tools = "[/AVAILABLE_TOOLS]";
                size_t pos = user.find(tag_tools);
                if (pos != std::string::npos)
                {
                    oss_prompt <<  user.substr(0, pos + tag_tools.size());
                    user_input = user.substr(pos + tag_tools.size());
                }

                oss_prompt << "[INST] " << user_input << " [/INST]";
                tok->encode(oss_prompt.str(), ids, false, false);
            }
        }
        else
        {
            oss_prompt << "[INST] " << user << " [/INST]";
        }

        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }
}

namespace mixtral
{
    struct Config : public mistral::Config
    {
        int num_experts_per_tok;
        int num_local_experts;
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

    class Tokenizer : public mistral::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : mistral::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = "";
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class MixtralSparseMoE : public BaseSparseMLP
    {
    public:
        MixtralSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template<int num_local_experts, int num_experts_per_tok, int sliding_window_len> class MixtralBlock : public LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm,
                        MixtralSparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        MixtralBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, MistralSelfAttention<sliding_window_len>, RMSNorm,
                       MixtralSparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, MixtralBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int> ModelClass;
    public:
        _ConditionalGeneration() = default;

        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : Base(type, config, runtime_config, 4096 * 2), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 3 + config.num_hidden_layers * (11 + 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_local_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            CHATLLM_CHECK((mistral::SLIDING_WINDOW_LEN == config.sliding_window) || (config.sliding_window <= 0))
                << "sliding_window (" << config.sliding_window << ") must equal to " << mistral::SLIDING_WINDOW_LEN;

            Base::transformer = new ModelClass(
                                &w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            Base::batch_input = false;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w2.weight", transformer->layers[i].mlp.experts_down.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w1.weight", transformer->layers[i].mlp.experts_gate.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "block_sparse_moe.experts.", _NUM_EXPERTS, ".w3.weight", transformer->layers[i].mlp.experts_up.weight);

                loader.read_tensor(layer_prefix + "block_sparse_moe.gate.weight",
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

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[INST] " << user << " [/INST]";
        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }

    const int NUM_EXPERTS                   =  8;
    const int EXPERTS_PER_TOK               =  2;

    typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_MIXTRAL> ConditionalGeneration;
}

namespace mistral2
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        int head_dim;
        int sliding_window;
        float rope_theta;
    };

    class Tokenizer : public mistral::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : mistral::Tokenizer(config, &mistral::_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);

            start_inst_token_id             = tp->PieceToId("[INST]");
            end_inst_token_id               = tp->PieceToId("[/INST]");
            tool_calls_token_id             = tp->PieceToId("[TOOL_CALLS]");
            start_avail_tools_token_id      = tp->PieceToId("[AVAILABLE_TOOLS]");
            end_avail_tools_token_id        = tp->PieceToId("[/AVAILABLE_TOOLS]");
            start_tool_results_token_id     = tp->PieceToId("[TOOL_RESULTS]");
            end_tool_results_token_id       = tp->PieceToId("[/TOOL_RESULTS]");
            return size;
        }
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<MistralBlock<mistral::SLIDING_WINDOW_LEN>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_MISTRAL2)
            : llama::v2::GenericConditionalGeneration<MistralBlock<mistral::SLIDING_WINDOW_LEN>>(config, runtime_config, type,
                config.num_key_value_heads, config.head_dim, config.max_length, 13, false)
        {
            CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == mistral::SLIDING_WINDOW_LEN))
                << "sliding_window (" << config.sliding_window << ") must be " << mistral::SLIDING_WINDOW_LEN;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass2>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }

            batch_input = false;
        }

        ChunkInterceptor *get_interceptor(void) override { return &mistral::interceptor; }
    };
}