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
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    const int SLIDING_WINDOW_LEN            =  4096;

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : llama::v2::Tokenizer::Tokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(const char *buffer, int n_vocab) override
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

        ConditionalGeneration(const Config &config, ModelType type)
            : llama::v2::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>(config, type,
                config.num_key_value_heads, config.max_length, 13)
        {
            CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == SLIDING_WINDOW_LEN))
                << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }

            batch_input = false;
        }

        ConditionalGeneration(const Config &config)
            : ConditionalGeneration(config, MODEL_TYPE_MISTRAL)
        {
        }

        ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
    };

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user(round_idx, user, ids);

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
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
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
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

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration<Model<Config, Embedding, RMSNorm,
        MixtralBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int>>
    {
    typedef BaseModelForConditionalGeneration<Model<Config, Embedding, RMSNorm,
        MixtralBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int>> Base;
    public:
        _ConditionalGeneration() = default;

        _ConditionalGeneration(const Config &config)
        : Base(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 3 + config.num_hidden_layers * (11 + _NUM_EXPERTS * 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_local_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            CHATLLM_CHECK((mistral::SLIDING_WINDOW_LEN == config.sliding_window) || (config.sliding_window <= 0))
                << "sliding_window (" << config.sliding_window << ") must equal to " << mistral::SLIDING_WINDOW_LEN;

            Base::GRAPH_SIZE = 4096 * 2;

            Base::transformer = new Model<Config, Embedding, RMSNorm, MixtralBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int>(
                                &w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            Base::batch_input = false;
        }

        void load(ModelLoader &loader) override
        {
            loader.read_tensor("model.embed_tokens.weight", Base::transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                for (int j = 0; j < _NUM_EXPERTS; j++)
                {
                    std::string prefix = layer_prefix + "block_sparse_moe.experts." + std::to_string(j) + '.';
                    loader.read_tensor(prefix + "w1.weight", Base::transformer->layers[i].mlp.experts[j].gate_proj.weight);
                    loader.read_tensor(prefix + "w2.weight", Base::transformer->layers[i].mlp.experts[j].down_proj.weight);
                    loader.read_tensor(prefix + "w3.weight", Base::transformer->layers[i].mlp.experts[j].up_proj.weight);
                }

                loader.read_tensor(layer_prefix + "block_sparse_moe.gate.weight",
                                Base::transformer->layers[i].mlp.gate.weight);

                loader.read_tensor(layer_prefix + "input_layernorm.weight",
                                Base::transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                Base::transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", Base::transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", Base::transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", Base::transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", Base::transformer->layers[i].attention.v_proj.weight);
            }
            loader.read_tensor("model.norm.weight", Base::transformer->final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(Base::transformer->lm_head)->weight);

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 244ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user(round_idx, user, ids);

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[INST] " << user << " [/INST]";
        tok->encode(oss_prompt.str(), ids, false, false);
    }

    const int NUM_EXPERTS                   =  8;
    const int EXPERTS_PER_TOK               =  2;

    typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_MIXTRAL> ConditionalGeneration;
}