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

    static void ggml_compute_forward_su_rope_f16(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
    {
        LongRoPEParam *data = reinterpret_cast<LongRoPEParam *>(userdata);

        const ggml::tensor *src0 = a;
        const ggml::tensor *src1 = b;

        GGML_TENSOR_UNARY_OP_LOCALS

        int n_dims = data->rope_dim;

        const int nr = (int)ggml::nrows(dst);

        GGML_ASSERT(n_dims == ne0);
        GGML_ASSERT(n_dims % 2 == 0);

        CHATLLM_CHECK(nb0 == sizeof(float)) << "nb0 = " << nb0;
        CHATLLM_CHECK(nb00 == sizeof(float)) << "nb0 = " << nb0;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        // row index used to determine which thread to use
        int ir = 0;

        const int32_t * pos = (const int32_t *) src1->data;

        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                const float p = (float)pos[i2];
                const float *inv_freq = data->get_inv_freq(pos[i2]);
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    if (ir++ < ir0) continue;
                    if (ir   > ir1) break;

                    for (int64_t i0 = 0; i0 < n_dims / 2; i0 += 1) {

                        float theta = p * inv_freq[i0];

                        float cos_theta = cosf(theta) * data->scaling_factor;
                        float sin_theta = sinf(theta) * data->scaling_factor;

                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = ggml_fp16_to_fp32(src[0]);
                        const float x1 = ggml_fp16_to_fp32(src[n_dims / 2]);

                        dst_data[0]             = ggml_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                        dst_data[n_dims / 2]    = ggml_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                    }
                }
            }
        }
    }

    static void ggml_compute_forward_su_rope_f32(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
    {
        LongRoPEParam *data = reinterpret_cast<LongRoPEParam *>(userdata);

        const ggml::tensor *src0 = a;
        const ggml::tensor *src1 = b;

        GGML_TENSOR_UNARY_OP_LOCALS

        int n_dims = data->rope_dim;

        const int nr = (int)ggml::nrows(dst);

        GGML_ASSERT(n_dims == ne0);
        GGML_ASSERT(n_dims % 2 == 0);

        CHATLLM_CHECK(nb0 == sizeof(float)) << "nb0 = " << nb0;
        CHATLLM_CHECK(nb00 == sizeof(float)) << "nb0 = " << nb0;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        // row index used to determine which thread to use
        int ir = 0;

        const int32_t * pos = (const int32_t *) src1->data;

        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                const float p = (float)pos[i2];
                const float *inv_freq = data->get_inv_freq(pos[i2]);
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    if (ir++ < ir0) continue;
                    if (ir   > ir1) break;

                    for (int64_t i0 = 0; i0 < n_dims / 2; i0 += 1) {

                        float theta = p * inv_freq[i0];

                        float cos_theta = cosf(theta) * data->scaling_factor;
                        float sin_theta = sinf(theta) * data->scaling_factor;

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims / 2];

                        dst_data[0]             = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims / 2]    = x0*sin_theta + x1*cos_theta;
                    }
                }
            }
        }
    }

    static void ggml_compute_forward_su_rope(ggml::tensor * dst , const ggml::tensor * a, const ggml::tensor * b, int ith, int nth, void * userdata)
    {
        switch (a->type)
        {
        case GGML_TYPE_F16:
            ggml_compute_forward_su_rope_f16(dst, a, b, ith, nth, userdata);
            break;
        case GGML_TYPE_F32:
            ggml_compute_forward_su_rope_f32(dst, a, b, ith, nth, userdata);
            break;
        default:
            GGML_ASSERT(false);
            break;
        }
    }

    class MiniCPM3SelfAttention : public RoPESelfAttention<deepseek::v2_light::BaseMLAttention<true>>
    {
    private:
        typedef RoPESelfAttention<deepseek::v2_light::BaseMLAttention<true>> Base;
    public:
        MiniCPM3SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim, bool use_bias)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim, use_bias)
        {
        }

        void config(int original_max_position_embeddings,
            float rope_theta, float short_scaling_factor, float long_scaling_factor, int factor_len, const float *short_factor, const float *long_factor)
        {
            this->original_max_position_embeddings = original_max_position_embeddings;
            this->freq_base = rope_theta;
            this->short_scaling_factor = short_scaling_factor;
            this->long_scaling_factor  = long_scaling_factor;
            build_inv_freq_from_factors(this->inv_freq_short, factor_len * 2, short_factor, freq_base);
            build_inv_freq_from_factors(this->inv_freq_long,  factor_len * 2, long_factor,  freq_base);

            if (max_length > original_max_position_embeddings)
            {
                long_rope.reset(new SimpleLongRoPEParam(rope_dim, long_scaling_factor, inv_freq_long.data()));
            }
            else
            {
                long_rope.reset(new SimpleLongRoPEParam(rope_dim, short_scaling_factor, inv_freq_short.data()));
            }
        }

    protected:
        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            return ggml::map_custom2_inplace(ctx, k, past, ggml_compute_forward_su_rope, GGML_N_TASKS_MAX, long_rope.get());
        }
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            return ggml::map_custom2_inplace(ctx, q, past, ggml_compute_forward_su_rope, GGML_N_TASKS_MAX, long_rope.get());
        }

    public:
        int original_max_position_embeddings;
        float short_scaling_factor;
        float long_scaling_factor;
        std::vector<float> inv_freq_short;
        std::vector<float> inv_freq_long;
    protected:
        std::unique_ptr<SimpleLongRoPEParam> long_rope;
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
            const size_t num_tensors = 2 + config.num_hidden_layers * 17;
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

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                auto *attention = &transformer->layers[i].attention;

                loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer->layers[i].input_layernorm.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);

                loader.read_tensor(layer_prefix + "self_attn.d_q_proj.weight",          attention->q_proj.d_q_proj->weight);
                loader.read_tensor(layer_prefix + "self_attn.q_norm.weight",            attention->q_proj.norm->weight);
                loader.read_tensor(layer_prefix + "self_attn.u_q_proj.weight",          attention->q_proj.u_q_proj->weight);

                loader.read_tensor(layer_prefix + "self_attn.d_kv_proj.weight",         attention->d_kv_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_pe_proj.weight",         attention->k_pe_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.kv_norm.weight",           attention->kv_norm.weight);
                loader.read_tensor(layer_prefix + "self_attn.u_k_nope_proj.weight",     attention->u_k_nope_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.u_v_proj.weight",          attention->u_v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight",            attention->o_proj.weight);
            }
        }

    public:
        Config config;
    };
}