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
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
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

        size_t load(const char *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

        int get_terminate_token_id(void) const override
        {
            return im_end_token_id;
        }

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_eos) const;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end, bool add_nl) const;

    public:
        int im_start_token_id;
        int im_end_token_id;
        int nl_token_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int>>
    {
    public:
        ConditionalGeneration(const Config &config, ModelType type = MODEL_TYPE_QWEN);

        void load(ModelLoader &loader) override;

    public:
        static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 844ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };

    size_t Tokenizer::load(const char *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
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

            ids[0] = 32;
std::string s; tp->Decode(ids, &s);
        std::cout << "32: " << int(s[0]) << ":" <<  std::endl << s<< ":"<<  std::endl;

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

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user(round_idx, user, ids);

        tok->encode(ai, ids, false, true, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        if (round_idx == 0)
        {
            tok->encode("system", ids, true, false, true);
            tok->encode(tok->get_system_prompt(), ids, false, true, true);
        }

        tok->encode("user", ids, true, false, true);
        tok->encode(user, ids, false, true, true);

        tok->encode("assistant", ids, true, false, true);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == im_start_token_id) || (id == im_end_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, ModelType type)
        : BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int>>(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 3 + config.num_hidden_layers * 16;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        // TODO: support of `use_dynamic_ntk`
        transformer = Model<Config, Embedding, RMSNorm, QWenBlock, int, int, int, int>(&w_ctx_, config, false,
                                                                                config.hidden_size, config.num_attention_heads,
                                                                                config.intermediate_size, config.max_length);

        bool use_dynamic_ntk = (config.flags & 1) != 0;
        bool use_logn_attn   = (config.flags & 2) != 0;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = transformer.layers[i];
            auto att = dynamic_cast<QWenSelfAttention *>(&layer.attention);
            att->config(config.rope_dim, config.rotary_emb_base, config.seq_length,
                        use_dynamic_ntk, use_logn_attn);
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.read_tensor("transformer.wte.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "transformer.h." + std::to_string(i) + '.';

            loader.read_tensor(layer_prefix + "attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "attn.c_proj.weight", transformer.layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "ln_1.weight",        transformer.layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "ln_2.weight",        transformer.layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "mlp.c_proj.weight", transformer.layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.w1.weight",   transformer.layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.w2.weight", transformer.layers[i].mlp.gate_proj.weight);
        }
        loader.read_tensor("transformer.ln_f.weight", transformer.final_layernorm.weight);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
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

        size_t load(const char *buffer, int n_vocab) override
        {
            size_t r = v1::Tokenizer::load(buffer, n_vocab);

            im_start_token_id = tp->PieceToId("<|im_start|>");
            im_end_token_id   = tp->PieceToId("<|im_end|>");
            bos_token_id = pad_token_id = eos_token_id = im_start_token_id - 1;

            std::vector<int> ids;
            tp->Encode("\n", &ids);
            nl_token_id = ids[0];

            return r;
        }
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int>>
    {
    public:
        ConditionalGeneration(const Config &config);

        void load(ModelLoader &loader) override;

    public:
        static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 444ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config)
        : BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int>>(MODEL_TYPE_QWEN2, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 3 + config.num_hidden_layers * 15;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int>(&w_ctx_, config, false,
                                                                                config.hidden_size, config.num_attention_heads,
                                                                                config.intermediate_size, config.num_key_value_heads,
                                                                                config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = transformer.layers[i];
            layer.attention.freq_base = config.rope_theta;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer.layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer.layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer.layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer.layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
        }
        loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
            << "corrupted model weights";
    }
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

    const int NUM_EXPERTS                   =  60;
    const int EXPERTS_PER_TOK               =  4;

    // make it easy to test with different number of experts.
    #define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

    typedef SparseMoE<SiLUMLP, NUM_EXPERTS, EXPERTS_PER_TOK> QWenSparseMoE;

    typedef GatedMLP<SiLUMLP> QWenGatedMLP;

    typedef CombinedMLP<QWenSparseMoE, QWenGatedMLP> QWenMoEMLP;

    class QWen2MoEBlock : public LMBlock1<RMSNorm, QWen2SelfAttention, RMSNorm, QWenMoEMLP>
    {
    public:
        QWen2MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
              num_kv_heads, head_dim, max_length)
        {}
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, RMSNorm, QWen2MoEBlock, int, int, int, int, int, int, int, int>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : BaseModelForConditionalGeneration<
                                        Model<Config, Embedding, RMSNorm, QWen2MoEBlock, int, int, int, int, int, int, int, int>>(MODEL_TYPE_QWEN2MoE, config, MEM_SIZE, SCRATCH_SIZE),
              config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 3 + config.num_hidden_layers * (17 + 3 * config.num_experts);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.num_experts) && (EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            transformer = Model<Config, Embedding, RMSNorm, QWen2MoEBlock, int, int, int, int, int, int, int, int>(
                &w_ctx_, config, false,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.moe_intermediate_size, config.shared_expert_intermediate_size,
                config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = transformer.layers[i];
                layer.attention.freq_base = config.rope_theta;
                layer.mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
            }

            GRAPH_SIZE = 4096 * 4;
        }

        void load(ModelLoader &loader) override
        {
            loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(i) + '.';

                loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer.layers[i].input_layernorm.weight);

                for (int j = 0; j < config.num_experts; j++)
                {
                    std::string prefix = layer_prefix + "mlp.experts." + std::to_string(j) + '.';
                    loader.read_tensor(prefix + "down_proj.weight", transformer.layers[i].mlp.mlp1.experts[j].down_proj.weight);
                    loader.read_tensor(prefix + "gate_proj.weight", transformer.layers[i].mlp.mlp1.experts[j].gate_proj.weight);
                    loader.read_tensor(prefix + "up_proj.weight", transformer.layers[i].mlp.mlp1.experts[j].up_proj.weight);
                }

                loader.read_tensor(layer_prefix + "mlp.gate.weight", transformer.layers[i].mlp.mlp1.gate.weight);

                loader.read_tensor(layer_prefix + "mlp.shared_expert.down_proj.weight", transformer.layers[i].mlp.mlp2.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.shared_expert.gate_proj.weight", transformer.layers[i].mlp.mlp2.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.shared_expert.up_proj.weight",   transformer.layers[i].mlp.mlp2.up_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.shared_expert_gate.weight",   transformer.layers[i].mlp.mlp2.gate.weight);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer.layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);
            }
            loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 1844ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };

}
