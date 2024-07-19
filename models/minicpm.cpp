namespace v1
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_scaling;
        float rope_theta;
        float scale_depth;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
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

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, MiniCPMBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_MINICPM, bool tie_word_embeddings = true)
            : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = (tie_word_embeddings ? 2 : 3) + config.num_hidden_layers * 12;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (tie_word_embeddings)
            {
                transformer = new ModelClass(&w_ctx_, config, nullptr,
                                            config.hidden_size, config.num_attention_heads,
                                            config.intermediate_size, config.num_key_value_heads, config.max_length);
            }
            else
            {
                transformer = new ModelClass(&w_ctx_, config, false,
                                            config.hidden_size, config.num_attention_heads,
                                            config.intermediate_size, config.num_key_value_heads, config.max_length);
            }

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
                attention.freq_scale = 1 / config.rope_scaling;
                attention.set_prec(ggml_prec::GGML_PREC_F32);

                get_typed_transformer<ModelClass>()->layers[i].hidden_scaling = config.scale_depth;
            }

            GRAPH_SIZE = 4096;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight",   transformer->layers[i].mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",   transformer->layers[i].mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",     transformer->layers[i].mlp.up_proj.weight);
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
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 1344ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        append_user(round_idx, user, ids);
        tokenizer->encode(ai, ids);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        ids.push_back(tokenizer->bos_token_id);
        oss_prompt << " ";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "<用户>" << user
                   << "<AI>";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }
}

namespace v2
{
    typedef v1::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : v1::Tokenizer(config, &_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t size = v1::Tokenizer::load(buffer, n_vocab);
            im_end_token_id     = tp->PieceToId("<|im_end|>");
            _pre_token_id       = tp->PieceToId("▁<PRE>");
            _mid_token_id       = tp->PieceToId("▁<MID>");
            tool_call_token_id  = tp->PieceToId("<|tool_call|>");
            im_start_token_id   = tp->PieceToId("<|im_start|>");
            _eot_token_id       = tp->PieceToId("▁<EOT>");
            _suf_token_id       = tp->PieceToId("▁<SUF>");

            terminate_ids.insert(im_end_token_id);

            return size;
        }

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_start, bool add_end) const
        {
            if (add_start)
                ids.push_back(im_start_token_id);
            v1::Tokenizer::encode(text, ids);
            if (add_end)
                ids.push_back(im_end_token_id);
        }

        int im_end_token_id;
        int _pre_token_id;
        int _mid_token_id;
        int tool_call_token_id;
        int im_start_token_id;
        int _eot_token_id;
        int _suf_token_id;
    };

    class ConditionalGeneration : public v1::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config)
            : v1::ConditionalGeneration(config, ModelType::MODEL_TYPE_MINICPM2, false)
        {}
    };

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user(round_idx, user, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        ids.push_back(tokenizer->bos_token_id);
    }

    void ChatHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "user\n" << user;
        tok->encode(oss_prompt.str(), ids, true, true);

        oss_prompt.str("");
        oss_prompt << "assistant\n";
        tok->encode(oss_prompt.str(), ids, true, false);
    }
}

namespace moe
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int num_experts;
        int num_experts_per_tok;
        float rope_scaling;
        float rope_theta;
        float scale_depth;
    };

    typedef v1::Tokenizer Tokenizer;

    const int NUM_EXPERTS                   =  8;
    const int EXPERTS_PER_TOK               =  2;

    // make it easy to test with different number of experts.
    #define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

    class MiniCPMSparseMoE : public BaseSparseMLP
    {
    public:
        MiniCPMSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {}
    };

    class MiniCPMBlock : public LMBlock3<RMSNorm, LlamaSelfAttention, RMSNorm, MiniCPMSparseMoE>
    {
    public:
        MiniCPMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock3(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, MiniCPMBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, ModelType type = ModelType::MODEL_TYPE_MINICPM_MoE)
            : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 2 + config.num_hidden_layers * (10 + 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.num_experts) && (EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            transformer = new ModelClass(&w_ctx_, config, nullptr,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_key_value_heads, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
                attention.freq_scale = 1 / config.rope_scaling;
                attention.set_prec(ggml_prec::GGML_PREC_F32);

                get_typed_transformer<ModelClass>()->layers[i].hidden_scaling = config.scale_depth;
            }
            GRAPH_SIZE = 4096;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "mlp.experts_gate.weight", layer_prefix + "mlp.experts.", config.num_experts, ".w1.weight", transformer->layers[i].mlp.experts_gate.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_down.weight", layer_prefix + "mlp.experts.", config.num_experts, ".w2.weight", transformer->layers[i].mlp.experts_down.weight);
                loader.read_tensor(layer_prefix + "mlp.experts_up.weight",   layer_prefix + "mlp.experts.", config.num_experts, ".w3.weight", transformer->layers[i].mlp.experts_up.weight);

                loader.read_tensor(layer_prefix + "mlp.gate.weight",                transformer->layers[i].mlp.gate.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 1344ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };
}