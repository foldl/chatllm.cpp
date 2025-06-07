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
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
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
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_MINICPM, bool tie_word_embeddings = true)
            : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
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
                attention.set_prec(ggml::prec::GGML_PREC_F32);

                get_typed_transformer<ModelClass>()->layers[i].scale_depth = config.scale_depth;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            transformer->word_embeddings->load("model.embed_tokens.", &loader);
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
            transformer->final_layernorm->load("model.norm.", &loader);

            if (transformer->lm_head)
                loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);
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

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "<用户>" << user;

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "<AI>";

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
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v1::Tokenizer(config, &_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t size = v1::Tokenizer::load(buffer, n_vocab);
            im_end_token_id     = tp->PieceToId("<|im_end|>");
            tool_call_token_id  = tp->PieceToId("<|tool_call|>");
            im_start_token_id   = tp->PieceToId("<|im_start|>");
            nl_token_id         = tp->PieceToId("\n");

            if (im_end_token_id < 0)
            {
                CHATLLM_CHECK(n_vocab == 73448) << "unsupported vocab";

                im_end_token_id    = 73440;
                im_start_token_id  = 73441;
                tool_call_token_id = 73442;

            }

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
            {
                ids.push_back(im_end_token_id);
                if (append_nl_token)
                    ids.push_back(append_nl_token);
            }
        }

        int im_end_token_id;
        int tool_call_token_id;
        int im_start_token_id;
        int nl_token_id;
        bool append_nl_token = false;
    };

    class ConditionalGeneration : public v1::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v1::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_MINICPM2, false)
        {}
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        ids.push_back(tokenizer->bos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "user\n" << user;
        tok->encode(oss_prompt.str(), ids, true, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("assistant\n", ids, true, false);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("user\n", ids, true, false);
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

    class MiniCPMBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, MiniCPMSparseMoE>
    {
    public:
        MiniCPMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, MiniCPMBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_MINICPM_MoE)
            : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
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
                attention.set_prec(ggml::prec::GGML_PREC_F32);

                get_typed_transformer<ModelClass>()->layers[i].scale_depth = config.scale_depth;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            transformer->word_embeddings->load("model.embed_tokens.", &loader);
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
            transformer->final_layernorm->load("model.norm.", &loader);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };
}

namespace v3
{
    const int MAX_FACTOR_LEN = 16;

    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int kv_lora_rank;
        int q_lora_rank;
        int qk_nope_head_dim;
        int qk_rope_head_dim;
        int original_max_position_embeddings;
        int v_head_dim;
        int dim_model_base;

        float scale_depth;
        float short_factor[MAX_FACTOR_LEN];
        float long_factor[MAX_FACTOR_LEN];
    };

    #define MIN(a, b) ((a) < (b) ? (a) : (b))

    class MiniCPM3SelfAttention : public RoPESelfAttention<deepseek::v2_light::BaseMLAttention<true>>
    {
    private:
        typedef RoPESelfAttention<deepseek::v2_light::BaseMLAttention<true>> Base;
    public:
        MiniCPM3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim, bool use_bias)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim, use_bias)
        {
            freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, rope_dim / 2);
            ctx->get_allocator()->alloc(freq_factors);
            rope_mode = RoPEMode::Original;
            this->rope_dim = rope_dim;
        }

        void config(int original_max_position_embeddings,
            float rope_theta, float short_scaling_factor, float long_scaling_factor, int factor_len, const float *short_factor, const float *long_factor)
        {
            const float *factors = nullptr;
            this->freq_base = rope_theta;
            if (max_length > original_max_position_embeddings)
            {
                attn_factor = long_scaling_factor;
                factors = long_factor;
            }
            else
            {
                attn_factor = short_scaling_factor;
                factors = short_factor;
            }

            CHATLLM_CHECK(freq_factors->ne[0] == factor_len) << "factor_len mismatch!";

            Backend::write_tensor_data(freq_factors, factors);
        }
    };


    typedef LMBlock1<RMSNorm, MiniCPM3SelfAttention, RMSNorm, SiLUMLP> MiniCPM3Block;

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v2::Tokenizer(config)
        {
            append_nl_token = true;
        }
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    private:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, MiniCPM3Block, int, int, int, int, int, int, int, int, int, int, bool> ModelClass;
    public:

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ConditionalGeneration(config, runtime_config, MODEL_TYPE_MINICPM3, config.q_lora_rank)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int q_lora_rank)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 2 + config.num_hidden_layers * 18;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config, nullptr,
                                        config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                        config.num_key_value_heads, config.max_length,
                                        q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                        false);

            const double scale = (double)config.max_length / config.original_max_position_embeddings;
            const float freq_scale =
                config.max_length > config.original_max_position_embeddings ?
                     (float)std::sqrt( 1.0 + std::log(scale) / std::log((double)config.original_max_position_embeddings)) : 1.0f;

            float attn_scaling_factor = 1 / sqrtf((float)(config.qk_rope_head_dim + config.qk_nope_head_dim));

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto layer = &get_typed_transformer<ModelClass>()->layers[i];
                layer->scale_depth          = config.scale_depth;
                layer->attention.config(config.original_max_position_embeddings, 10000.0f,
                                 freq_scale,
                                 freq_scale,
                                 config.qk_rope_head_dim / 2,
                                 config.short_factor,
                                 config.long_factor);
                layer->attention.attn_scaling_factor = attn_scaling_factor;
                layer->attention.set_prec(ggml::prec::GGML_PREC_F32);
            }

            CHATLLM_CHECK(w_ctx_.get_used_mem() / tensor_ovhd == num_tensors)
                << "actual tensors: " << w_ctx_.get_used_mem() / tensor_ovhd << ", expected " << num_tensors << ".";
            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size()) << "corrupted model weights";
        }

    public:
        Config config;
    };
}

namespace emb_light
{
    const int MAX_FACTOR_LEN = 32;
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int original_max_position_embeddings;

        float rope_theta;
        float scale_depth;
        float short_factor[MAX_FACTOR_LEN];
        float long_factor[MAX_FACTOR_LEN];
    };

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v1::Tokenizer(config, nullptr)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = v1::Tokenizer::load(buffer, n_vocab);
            tp->RegisterPreprocessor(new tokenizer::TextPrepNewlineToSpaces());
            tp->RegisterPreprocessor(new tokenizer::TextPrepDeleteMultiSpaces());
            tp->RegisterPreprocessor(new tokenizer::TextPrepAddLeadingSpace());
            im_end_token_id = tp->PieceToId("<|im_end|>");
            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override
        {
            encode(text, ids, true, true);
        }

    protected:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
        {
            if (add_bos)
                ids.push_back(bos_token_id);

            BaseTokenizer::encode(text, ids);

            if (add_eos)
                ids.push_back(im_end_token_id);
        }

        int im_end_token_id;
    };

    class MiniCPMLongRoPESelfAttention : public RoPESelfAttention<BaseAttention>
    {
    private:
        typedef RoPESelfAttention<BaseAttention> Base;
    public:
        MiniCPMLongRoPESelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false)
        {
            causal = false;

            freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size / num_attention_heads / 2);
            ctx->get_allocator()->alloc(freq_factors);
            rope_mode = RoPEMode::Original;
        }

        void config(int original_max_position_embeddings,
            float rope_theta, float short_scaling_factor, float long_scaling_factor, int factor_len, const float *short_factor, const float *long_factor)
        {
            const float *factors = nullptr;
            this->freq_base = rope_theta;
            if (max_length > original_max_position_embeddings)
            {
                attn_factor = long_scaling_factor;
                factors = long_factor;
            }
            else
            {
                attn_factor = short_scaling_factor;
                factors = short_factor;
            }

            CHATLLM_CHECK(freq_factors->ne[0] == factor_len) << "factor_len mismatch!";

            Backend::write_tensor_data(freq_factors, factors);
        }
    };

    class MiniCPMBlock : public LMBlock1<RMSNorm, MiniCPMLongRoPESelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        MiniCPMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    template <class FinalBlock, int final_tensor_num> class _ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef EmbeddingModel<Config, Embedding, MiniCPMBlock, FinalBlock, int, int, int, int, int> ModelClass;
        typedef BaseModelForConditionalGeneration Base;
    public:
        _ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
            : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 1 + final_tensor_num + config.num_hidden_layers * 13;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            Base::transformer = new ModelClass(&w_ctx_, config,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_key_value_heads, config.max_length);

            const double scale = (double)config.max_length / config.original_max_position_embeddings;
            const float freq_scale =
                config.max_length > config.original_max_position_embeddings ?
                     (float)std::sqrt( 1.0 + std::log(scale) / std::log((double)config.original_max_position_embeddings)) : 1.0f;

            int head_dim = config.hidden_size / config.num_attention_heads;

            CHATLLM_CHECK(head_dim / 2 == MAX_FACTOR_LEN) << "dim = " << head_dim;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = Base::get_typed_transformer<ModelClass>()->layers[i];
                auto &attention = layer.attention;
                layer.scale_depth          = config.scale_depth;

                attention.config(config.original_max_position_embeddings, config.rope_theta,
                                 freq_scale,
                                 freq_scale,
                                 head_dim / 2,
                                 config.short_factor,
                                 config.long_factor);
                attention.set_prec(ggml::prec::GGML_PREC_F32);
            }
        }

    public:
        Config config;
    };

    class ConditionalGeneration : public _ConditionalGeneration<MiniCPMMeanPooling, 2>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_MiniCPM_Embedding_Light)
            : _ConditionalGeneration<MiniCPMMeanPooling, 2>(config, runtime_config, type)
        {
            {
                auto transformer = get_typed_transformer<ModelClass>();
                (dynamic_cast<MiniCPMMeanPooling *>(transformer->final_layernorm))->set_max_length(&w_ctx_, config.max_length);
            }
        }
    };
}

namespace ranker_light
{
    typedef emb_light::Config Config;

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : v1::Tokenizer(config, nullptr)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = v1::Tokenizer::load(buffer, n_vocab);
            tp->RegisterPreprocessor(new tokenizer::TextPrepNewlineToSpaces());
            tp->RegisterPreprocessor(new tokenizer::TextPrepDeleteMultiSpaces());
            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override
        {
            BaseTokenizer::encode(text, ids);
        }

        void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const override
        {
            const int max_length = this->max_length;

            std::vector<int> ids_q;
            std::vector<int> ids_a;
            BaseTokenizer::encode(q, ids_q);
            BaseTokenizer::encode(a, ids_a);

            int total = (int)ids_q.size() + (int)ids_a.size();

            // this is bad
            if (total > max_length - 2)
            {
                int remain = max_length - 2 - (int)ids_q.size();
                CHATLLM_CHECK(remain > 0) << "query is TOOOO long.";

                ids_a.resize(remain);
            }

            ids.push_back(bos_token_id);
            ids.insert(std::end(ids), std::begin(ids_q), std::end(ids_q));
            ids.push_back(eos_token_id);
            ids.insert(std::end(ids), std::begin(ids_a), std::end(ids_a));
        }
    };

    class MiniCPMClassificationHead : public Block
    {
    public:
        MiniCPMClassificationHead(InitContext *ctx, int hidden_size)
            : layernorm(ctx, hidden_size), score(ctx, hidden_size, 1, false)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            int hidden_size = (int)hidden_states->ne[0];

            ggml::tensor *first_token_tensor = ggml::view_2d(ctx, hidden_states, hidden_size, 1,
                                                        hidden_size * ggml::element_size(hidden_states), 0);
            first_token_tensor = layernorm.forward(ctx, first_token_tensor);
            ggml::tensor *output = score.forward(ctx, first_token_tensor);
            output = ggml::sigmoid(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += layernorm.get_param_num(effective_only);
            r += score.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader)
        {
            Block::load(path, loader);
            layernorm.load("model.norm.", loader);
            score.load("score.", loader);
        }
    public:
        RMSNorm layernorm;
        Linear score;
    };

    class ConditionalGeneration : public emb_light::_ConditionalGeneration<MiniCPMClassificationHead, 2>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_MiniCPM_ReRanker_Light)
            : emb_light::_ConditionalGeneration<MiniCPMClassificationHead, 2>(config, runtime_config, type)
        {
        }
    };
}

namespace v4
{
    const int MAX_FACTOR_LEN = 128;
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int max_position_embeddings;
        int original_max_position_embeddings;
        int tie_word_embeddings;
        int factor_len;

        float mup_denominator;
        float lm_head_pre_scale;
        float rope_theta;
        float scale_depth;
        float short_factor[MAX_FACTOR_LEN];
        float long_factor[MAX_FACTOR_LEN];
    };

    typedef v3::Tokenizer Tokenizer;

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<Phi3SUBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_MINICPM4)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<Phi3SUBlock>(config, runtime_config, type, num_key_value_heads, max_length, 13, config.tie_word_embeddings != 0)
        {
            float scaling_factor = (float)config.max_length / config.original_max_position_embeddings;
            if (scaling_factor <= 1.0f)
                scaling_factor = 1.0f;
            else
                scaling_factor = sqrtf(1.0f + logf(scaling_factor) / logf((float)config.original_max_position_embeddings));

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                if (config.factor_len > 0)
                {
                    attention.config(&w_ctx_, config.original_max_position_embeddings, config.rope_theta,
                                    scaling_factor,
                                    scaling_factor,
                                    config.factor_len,
                                    config.short_factor,
                                    config.long_factor);
                }
                get_typed_transformer<ModelClass>()->layers[i].scale_depth = config.scale_depth;
            }
        }
    };
}