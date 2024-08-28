namespace v2
{
    class QAHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    void QAHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);

        std::ostringstream oss_prompt;
        oss_prompt << ai << "\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void QAHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Instruct: " << user << "\n";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void QAHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Output: ";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);

        std::ostringstream oss_prompt;
        oss_prompt << ai << "\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Alice: " << user << "\n";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Bob: ";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    static QAHistoryEncoder _qa_encoder;
    static ChatHistoryEncoder _chat_encoder;

    class Phi2Tokenizer : public BaseTokenizer
    {
    public:
        Phi2Tokenizer(const BaseConfig &config)
            : BaseTokenizer::BaseTokenizer(config, &_chat_encoder, &_qa_encoder),
            qa_seq_max_len(0)
        {
        }

        Phi2Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder),
            qa_seq_max_len(0)
        {
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

    public:
        std::vector<int> qa_terminate_seq1;
        std::vector<int> qa_terminate_seq2;
        std::vector<int> qa_terminate_seq3;
        std::vector<int> qa_terminate_seq4;
        int qa_seq_max_len;
        std::vector<int> chat_terminate_seq;
    };

    class Phi2ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<BaseConfig, Embedding, LayerNorm, Phi2Block, int, int, int, int, int> ModelClass;
    public:
        Phi2ConditionalGeneration() = default;
        Phi2ConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type)
            : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 5 + config.num_hidden_layers * 17;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config, true,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_attention_heads, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                get_typed_transformer<ModelClass>()->layers[i].set_id(i);
                get_typed_transformer<ModelClass>()->layers[i].attention.set_id(i);
                get_typed_transformer<ModelClass>()->layers[i].attention.set_prec(ggml::prec::GGML_PREC_F32);
            }
        }

    public:



        BaseConfig config;

    protected:
        bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override;
    };

    bool Phi2ConditionalGeneration::is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output)
    {
        if (output_ids.size() < 1) return false;

        Phi2Tokenizer *tokenizer = dynamic_cast<Phi2Tokenizer *>(this->tokenizer);
        int len = 0;

        switch (tokenizer->get_chat_format())
        {
        case ChatFormat::QA:
            if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq1))
            {
                pop_output = (int)tokenizer->qa_terminate_seq1.size();
                return true;
            }
            else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq2))
            {
                pop_output = (int)tokenizer->qa_terminate_seq2.size();
                return true;
            }
            else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq3))
            {
                pop_output = (int)tokenizer->qa_terminate_seq3.size();
                return true;
            }
            else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq4))
            {
                pop_output = (int)tokenizer->qa_terminate_seq4.size();
                return true;
            }
            len = tokenizer->qa_seq_max_len;
            break;
        case ChatFormat::CHAT:
            if (match_output_sequence(output_ids, tokenizer->chat_terminate_seq))
            {
                pop_output = (int)tokenizer->chat_terminate_seq.size();
                return true;
            }
            len = (int)tokenizer->chat_terminate_seq.size();
            break;
        default:
            ;
        }

        if (BaseModelForConditionalGeneration::is_output_terminated(output_ids, keep_idx, pop_output))
            return true;

        keep_idx = (int)(output_ids.size()) - len + 1;
        return false;
    }

    #define MAX_IT(x)  if (qa_seq_max_len < (int)x.size()) qa_seq_max_len = (int)x.size()

    size_t Phi2Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        tp->Encode("\nInstruct:", &qa_terminate_seq1);
        tp->Encode("\nInstruction:", &qa_terminate_seq2);
        tp->Encode("\nUser:", &qa_terminate_seq3);
        tp->Encode("\nINPUT:", &qa_terminate_seq4);
        tp->Encode("\nAlice", &chat_terminate_seq);
        bos_token_id = eos_token_id = tp->PieceToId("<|endoftext|>");

        MAX_IT(qa_terminate_seq1);
        MAX_IT(qa_terminate_seq2);
        MAX_IT(qa_terminate_seq3);
        MAX_IT(qa_terminate_seq4);

        return size;
    }

    bool Phi2Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id);
    }

    namespace v1
    {
        struct Config : public BaseConfig
        {
        };

        class Tokenizer : public Phi2Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config)
                : Phi2Tokenizer(config)
            {
            }
        };

        class ConditionalGeneration : public Phi2ConditionalGeneration
        {
        public:
            ConditionalGeneration() = default;
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type);

            void load(ModelLoader &loader) override;
        };

        ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ConditionalGeneration::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_PHI2)
        {

        }

        ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
            : Phi2ConditionalGeneration(config, runtime_config, type)
        {

        }

        void ConditionalGeneration::load(ModelLoader &loader)
        {
            auto transformer = get_typed_transformer<ModelClass>();
            loader.read_tensor("transformer.embd.wte.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "transformer.h." + std::to_string(layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "ln.bias",   transformer->layers[i].input_layernorm.bias);
                loader.read_tensor(layer_prefix + "ln.weight", transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "mixer.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "mixer.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "mixer.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "mixer.out_proj.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.out_proj.weight", transformer->layers[i].attention.o_proj.weight);

                loader.read_tensor(layer_prefix + "mlp.fc1.bias",   transformer->layers[i].mlp.fc0.bias);
                loader.read_tensor(layer_prefix + "mlp.fc1.weight", transformer->layers[i].mlp.fc0.weight);
                loader.read_tensor(layer_prefix + "mlp.fc2.bias",   transformer->layers[i].mlp.fc1.bias);
                loader.read_tensor(layer_prefix + "mlp.fc2.weight", transformer->layers[i].mlp.fc1.weight);
            }

            loader.read_tensor("lm_head.ln.bias", transformer->final_layernorm.bias);
            loader.read_tensor("lm_head.ln.weight", transformer->final_layernorm.weight);

            loader.read_tensor("lm_head.linear.bias", dynamic_cast<Linear *>(transformer->lm_head)->bias);
            loader.read_tensor("lm_head.linear.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }
    }

    namespace v2
    {
        struct Config : public BaseConfig
        {
            int rope_dim;
            float rope_theta;
        };

        class Tokenizer : public Phi2Tokenizer
        {
        public:
            Tokenizer(const BaseConfig &config)
                : Phi2Tokenizer(config)
            {
            }
        };

        class ConditionalGeneration : public Phi2ConditionalGeneration
        {
        public:
            ConditionalGeneration() = default;
            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

            void load(ModelLoader &loader) override;
        };

        ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : Phi2ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_PHI2)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                get_typed_transformer<ModelClass>()->layers[i].attention.rope_dim = config.rope_dim;
                get_typed_transformer<ModelClass>()->layers[i].attention.freq_base = config.rope_theta;
            }
        }

        void ConditionalGeneration::load(ModelLoader &loader)
        {
            auto transformer = get_typed_transformer<ModelClass>();
            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "input_layernorm.bias",   transformer->layers[i].input_layernorm.bias);
                loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.dense.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.dense.weight", transformer->layers[i].attention.o_proj.weight);

                loader.read_tensor(layer_prefix + "mlp.fc1.bias",   transformer->layers[i].mlp.fc0.bias);
                loader.read_tensor(layer_prefix + "mlp.fc1.weight", transformer->layers[i].mlp.fc0.weight);
                loader.read_tensor(layer_prefix + "mlp.fc2.bias",   transformer->layers[i].mlp.fc1.bias);
                loader.read_tensor(layer_prefix + "mlp.fc2.weight", transformer->layers[i].mlp.fc1.weight);
            }

            loader.read_tensor("model.final_layernorm.bias", transformer->final_layernorm.bias);
            loader.read_tensor("model.final_layernorm.weight", transformer->final_layernorm.weight);

            loader.read_tensor("lm_head.bias", dynamic_cast<Linear *>(transformer->lm_head)->bias);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }
    }
}

namespace v3
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int original_max_position_embeddings;
        int sliding_window;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    public:
        bool add_bos = true;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Phi3Tokenizer : public BaseTokenizer
    {
    public:
        Phi3Tokenizer(const BaseConfig &config)
            : Phi3Tokenizer(config, &_chat_encoder)
        {
        }

        Phi3Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder),
              append_nl_after_end_tok(false)
        {
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor1();
            size_t size = tp->Load(buffer, n_vocab);

            system_token_id     = tp->PieceToId("<|system|>");
            user_token_id       = tp->PieceToId("<|user|>");
            assistant_token_id  = tp->PieceToId("<|assistant|>");
            end_token_id        = tp->PieceToId("<|end|>");
            nl_token_id         = tp->PieceToId("\n");

            if (-1 == system_token_id)
            {
                CHATLLM_CHECK(tp->GetPieceSize() == 32000) << " unsupported tokenizer";
                system_token_id     = 32006;
                user_token_id       = 32010;
                assistant_token_id  = 32001;
                end_token_id        = 32007;
            }

            pad_token_id = eos_token_id;

            terminate_ids.insert(end_token_id);

            return size;
        }

        void encode(const std::string &msg, std::vector<int> &ids, int type_token_id, int end_token_id = -1)
        {
            if (type_token_id >= 0)
            {
                ids.push_back(type_token_id);
                ids.push_back(nl_token_id);
            }
            BaseTokenizer::encode(msg, ids);
            if (end_token_id >= 0)
            {
                ids.push_back(end_token_id);
                if (append_nl_after_end_tok)
                    ids.push_back(nl_token_id);
            }
        }

    public:
        int system_token_id;
        int user_token_id;
        int assistant_token_id;
        int end_token_id;
        int nl_token_id;
        bool append_nl_after_end_tok;
    };

    typedef Phi3Tokenizer Tokenizer;

    // https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/25
    const int SLIDING_WINDOW_LEN            =  2048;

    template <int sliding_window_len> class Phi3SelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        Phi3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, max_length, false, false) {}

        Phi3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
    };

    template <int sliding_window_len> class Phi3Block : public LMBlock1<RMSNorm, Phi3SelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>
    {
    public:
        Phi3Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
            : LMBlock1<RMSNorm, Phi3SelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, max_length)
        {}

        Phi3Block(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<RMSNorm, Phi3SelfAttention<sliding_window_len>, RMSNorm, SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    typedef Phi3Block<SLIDING_WINDOW_LEN> Phi3Block4k;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<Phi3Block4k>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI3)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<Phi3Block4k>(config, runtime_config, type, num_key_value_heads, max_length, 13)
        {
            CHATLLM_CHECK(config.sliding_window == SLIDING_WINDOW_LEN - 1)
                << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN - 1;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }

            batch_input = false;
        }
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, -1, tok->end_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (add_bos)
            ids.push_back(tok->bos_token_id);

        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids, tok->system_token_id, tok->end_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(user, ids, tok->user_token_id, tok->end_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("", ids, tok->assistant_token_id, -1);
    }
}

namespace v3_su
{
    const int MAX_FACTOR_LEN = 128;

    struct Config : public BaseConfig
    {
        int max_position_embeddings;
        int num_key_value_heads;
        int original_max_position_embeddings;
        int sliding_window;
        int rope_scaling;
        float rope_theta;
        float short_factor[MAX_FACTOR_LEN];
        float long_factor[MAX_FACTOR_LEN];
    };

    typedef v3::Phi3Tokenizer Tokenizer;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<Phi3SUBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI3_SU)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<Phi3SUBlock>(config, runtime_config, type, num_key_value_heads, max_length, 12)
        {
            CHATLLM_CHECK(config.sliding_window >= config.max_length)
                << "sliding_window (" << config.sliding_window << ") must >= " << config.max_length;

            CHATLLM_CHECK(config.rope_scaling == 1)
                << "rope_scaling (" << config.rope_scaling << ") must == " << 1;

            float scaling_factor = (float)config.max_position_embeddings / config.original_max_position_embeddings;
            if (scaling_factor <= 1.0f)
                scaling_factor = 1.0f;
            else
                scaling_factor = sqrtf(1.0f + logf(scaling_factor) / logf((float)config.original_max_position_embeddings));

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.config(config.original_max_position_embeddings, config.rope_theta,
                                 scaling_factor,
                                 scaling_factor,
                                 config.hidden_size / config.num_attention_heads / 2,
                                 config.short_factor,
                                 config.long_factor);
            }
        }
    };
}

namespace v3_su2
{
    typedef v3_su::Config Config;

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config) : v3::Tokenizer(config, &v3::_chat_encoder)
        {
            append_nl_after_end_tok = true;
            v3::_chat_encoder.add_bos = false;
        }
    };

    typedef v3_su::ConditionalGeneration ConditionalGeneration;
}

namespace v3_su3
{
    struct Config : public v3_su2::Config
    {
        float short_mscale;
        float long_mscale;
    };

    typedef v3_su2::Tokenizer Tokenizer;

    class ConditionalGeneration : public v3_su2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_PHI3_SU3)
            : v3_su2::ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.config(config.original_max_position_embeddings, config.rope_theta,
                                 config.short_mscale,
                                 config.long_mscale,
                                 config.hidden_size / config.num_attention_heads / 2,
                                 config.short_factor,
                                 config.long_factor);
            }
        }
    };
}

namespace v3_moe
{
    struct Config : public v3_su3::Config
    {
        int num_experts_per_tok;
        int num_local_experts;
    };

    typedef v3_su3::Tokenizer Tokenizer;

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class Phi3SparseMoE : public BaseSparseMLP
    {
    public:
        Phi3SparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    class Phi3SUSelfAttentionBiased : public Phi3SUSelfAttention
    {
    public:
        Phi3SUSelfAttentionBiased(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : Phi3SUSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true)
        {}
    };

    template<int num_local_experts, int num_experts_per_tok> class Phi3MoEBlock : public LMBlock1<LayerNorm, Phi3SUSelfAttentionBiased, LayerNorm,
                        Phi3SparseMoE<num_local_experts, num_experts_per_tok>>
    {
    public:
        Phi3MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1<LayerNorm, Phi3SUSelfAttentionBiased, LayerNorm,
                       Phi3SparseMoE<num_local_experts, num_experts_per_tok>>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template<int _NUM_EXPERTS, int _EXPERTS_PER_TOK, ModelType type> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, LayerNorm, Phi3MoEBlock<_NUM_EXPERTS, _EXPERTS_PER_TOK>, int, int, int, int, int> ModelClass;
    public:
        _ConditionalGeneration() = default;

        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : Base(type, config, runtime_config), config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 3 + 2 + config.num_hidden_layers * (11 + 3 + 5);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((_NUM_EXPERTS == config.num_local_experts) && (_EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            Base::GRAPH_SIZE = 4096 * 2;

            Base::transformer = new ModelClass(
                                &w_ctx_, config, true,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = Base::get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.config(config.original_max_position_embeddings, config.rope_theta,
                                 config.short_mscale,
                                 config.long_mscale,
                                 config.hidden_size / config.num_attention_heads / 2,
                                 config.short_factor,
                                 config.long_factor);
            }

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size()) << "corrupted model weights";
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
                loader.read_tensor(layer_prefix + "input_layernorm.bias",
                                transformer->layers[i].input_layernorm.bias);

                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                                transformer->layers[i].post_attention_layernorm.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",
                                transformer->layers[i].post_attention_layernorm.bias);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
            loader.read_tensor("model.norm.bias",   transformer->final_layernorm.bias);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);
            loader.read_tensor("lm_head.bias",   dynamic_cast<Linear *>(transformer->lm_head)->bias);
        }

    public:
        Config config;
    };

    const int NUM_EXPERTS                   =  16;
    const int EXPERTS_PER_TOK               =  2;

    typedef _ConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, MODEL_TYPE_PHI3_MOE> ConditionalGeneration;
}