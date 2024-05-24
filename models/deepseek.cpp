namespace v1
{
    struct Config : public llama::v2::Config
    {
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : llama::v2::Tokenizer::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = "";
        }

        size_t load(const char *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;
    };

    class ConditionalGeneration : public llama::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config)
            : llama::v2::ConditionalGeneration(config, MODEL_TYPE_DEEPSEEK)
        {
        }
    };

    size_t Tokenizer::load(const char *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
            oss_prompt << tok->get_system_prompt() << "\n\n";

        oss_prompt << "User: " << user << "\n\n"
                << "Assistant: " << ai;

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        if ((round_idx == 0) && (tok->get_system_prompt().size() > 0))
            oss_prompt << tok->get_system_prompt() << "\n\n";

        oss_prompt << "User: " << user << "\n\n"
                << "Assistant: ";

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id);
    }
}

#if (1)
namespace v2_light
{

    extern "C" void ggml_compute_forward_zero(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata)
    {
        memset(dst->data, 0, ggml_nbytes(dst));
    }

    template <bool opt_speed> class BaseMLAttention : public KVCacheAttention
    {
    public:
        BaseMLAttention() :
            KVCacheAttention(),
            kv_lora_rank(0),
            rope_dim(0),
            qk_nope_head_dim(0),
            v_head_dim(0) {}

        BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias,
                      ggml_type cache_type, int cache_length)
            : KVCacheAttention(ctx, num_attention_heads, num_kv_heads,
                               (qk_nope_head_dim + rope_dim) * num_kv_heads, v_head_dim * num_kv_heads,
                               max_length,
                               cache_type, cache_length),
              d_kv_proj(ctx, hidden_size, kv_lora_rank, nullptr, use_bias),
              k_pe_proj(ctx, hidden_size, rope_dim, nullptr, use_bias),
              u_k_nope_proj(ctx, kv_lora_rank, qk_nope_head_dim * num_kv_heads, nullptr, false),
              u_v_proj(ctx, kv_lora_rank, v_head_dim * num_kv_heads, nullptr, false),
              q_proj(ctx, hidden_size, (qk_nope_head_dim + rope_dim) * num_attention_heads, nullptr, false),
              o_proj(ctx, v_head_dim * num_attention_heads, hidden_size, use_bias),
              kv_norm(ctx, kv_lora_rank),
              kv_lora_rank(kv_lora_rank),
              rope_dim(rope_dim),
              qk_nope_head_dim(qk_nope_head_dim),
              v_head_dim(v_head_dim)
        {
        }

        BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias)
            : BaseMLAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                              kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                              use_bias,
                              GGML_TYPE_F16, max_length)
        {}

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += d_kv_proj.get_param_num(effective_only);
            r += k_pe_proj.get_param_num(effective_only);
            r += u_k_nope_proj.get_param_num(effective_only);
            r += u_v_proj.get_param_num(effective_only);
            r += q_proj.get_param_num(effective_only);
            r += kv_norm.get_param_num(effective_only);
            r += o_proj.get_param_num(effective_only);
            return r;
        }

        using Block::forward;
        ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
        {
            const int hidden_size = o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            KVCacheAttention::before_forward(ctx, n_past, qlen);

            ggml_tensor *kv_lora = d_kv_proj.forward(ctx, hidden_states);
            kv_lora = kv_norm.forward(ctx, kv_lora);

            ggml_tensor *tmpv = u_v_proj.forward(ctx, kv_lora);

            ggml_tensor *k_nope = u_k_nope_proj.forward(ctx, kv_lora);
            ggml_tensor *k_pe = k_pe_proj.forward(ctx, hidden_states);

            ggml_tensor *tmpq = q_proj.forward(ctx, hidden_states);

            ggml_tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, k_nope, k_pe, tmpv);

            ggml_tensor *attn_output = o_proj.forward(ctx, scores);

            return attn_output;
        }

        ggml_tensor *cross_attention(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml_tensor *q, ggml_tensor *k_nope, ggml_tensor *k_pe, ggml_tensor *v)
        {
            // [qlen, heads, head_size]
            k_pe = ggml_reshape_3d(ctx->gctx.get(), k_pe, rope_dim, 1, qlen);
            k_pe = apply_pos_embedding_k(ctx, k_pe, rope_dim * 1, qlen, pos);

            ggml_tensor * key_layer = ggml_new_tensor_3d(ctx->gctx.get(), k_pe->type,
                qk_nope_head_dim + rope_dim, num_kv_heads, qlen);

            key_layer = ggml_map_custom1_inplace(ctx->gctx.get(), key_layer, ggml_compute_forward_zero, 1, nullptr);

            ggml_tensor * k_nope_dst = ggml_view_3d(ctx->gctx.get(), key_layer,
                         qk_nope_head_dim, num_kv_heads, qlen,
                         key_layer->nb[1], key_layer->nb[2],
                         0);
            ggml_tensor * k_pe_dst = ggml_view_3d(ctx->gctx.get(), key_layer,
                         rope_dim, num_kv_heads, qlen,
                         key_layer->nb[1], key_layer->nb[2],
                         qk_nope_head_dim * ggml_element_size(key_layer));
            k_nope = ggml_reshape_3d(ctx->gctx.get(), k_nope, qk_nope_head_dim, num_kv_heads, qlen);

            ggml_build_forward_expand(ctx->gf, ggml_cpy(ctx->gctx.get(), k_nope, k_nope_dst));
            ggml_build_forward_expand(ctx->gf, ggml_add_inplace(ctx->gctx.get(), k_pe_dst, k_pe));  // auto-broadcasting

            // [qlen, heads, head_size]
            ggml_tensor * query_layer = ggml_reshape_3d(ctx->gctx.get(), q, qk_nope_head_dim + rope_dim, num_attention_heads, qlen);
            ggml_tensor * q_pe = ggml_view_3d(ctx->gctx.get(), query_layer,
                         rope_dim, num_attention_heads, qlen,
                         query_layer->nb[1], query_layer->nb[2],
                         qk_nope_head_dim * ggml_element_size(query_layer));
            q_pe = apply_pos_embedding_q(ctx, q_pe, rope_dim * num_attention_heads, qlen, pos);
            ggml_build_forward_expand(ctx->gf, q_pe);

            ggml_tensor *attn_scores = cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

            return attn_scores;
        }

    public:
        Linear d_kv_proj, k_pe_proj, u_k_nope_proj, u_v_proj;
        Linear q_proj;
        Linear o_proj;
        RMSNorm kv_norm;
        const int kv_lora_rank;
        const int rope_dim;
        const int qk_nope_head_dim;
        const int v_head_dim;
    };

    typedef RoPESelfAttention<BaseMLAttention<true>> SpeedMLAttention;

    struct Config : public v1::Config
    {
        int num_key_value_heads;
        int first_k_dense_replace;
        int kv_lora_rank;
        int moe_intermediate_size;
        int moe_layer_freq;
        int n_group;
        int n_routed_experts;
        int n_shared_experts;
        int norm_topk_prob;
        int num_experts_per_tok;
        int qk_nope_head_dim;
        int qk_rope_head_dim;
        int original_max_position_embeddings;
        int v_head_dim;

        float beta_fast;
        float beta_slow;
        float factor;
        float mscale;
        float mscale_all_dim;
        float rope_theta;
        float routed_scaling_factor;
    };

    typedef v1::Tokenizer Tokenizer;

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;

    // make it easy to test with different number of experts.
    #define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

    typedef SparseMoE<SiLUMLP, NUM_EXPERTS, EXPERTS_PER_TOK> DeepSeekSparseMoE;

    typedef CombinedMLP<DeepSeekSparseMoE, SiLUMLP> DeepSeekMoEMLP;

    typedef LMBlock1<RMSNorm, SpeedMLAttention, RMSNorm, DeepSeekMoEMLP> DeepSeek2MoEBlock;
    typedef LMBlock1<RMSNorm, SpeedMLAttention, RMSNorm, SiLUMLP> DeepSeek2Block;

    static float yarn_get_mscale(float scale = 1.0f, float mscale = 1.0f)
    {
        if (scale <= 1.0f)
            return 1.0f;
        return 0.1f * mscale * logf(scale) + 1.0f;
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                    HeterogeneousModel<Config, Embedding, RMSNorm>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : BaseModelForConditionalGeneration<
                                        HeterogeneousModel<Config, Embedding, RMSNorm>>(
                                            MODEL_TYPE_DEEPSEEK_V2, config, MEM_SIZE, SCRATCH_SIZE),
              config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 3 + (config.num_hidden_layers - 1) * (16 + 3 * config.n_routed_experts) + 15;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts) && (EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    return new DeepSeek2MoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                                                config.num_key_value_heads, config.max_length,
                                                config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                                false);
                }
                else
                {
                    return new DeepSeek2Block(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length,
                                                config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                                false);
                }
            };

            transformer = new HeterogeneousModel<Config, Embedding, RMSNorm>(
                &w_ctx_, config, false, create_layer);

            float m = yarn_get_mscale(config.factor, config.mscale) / yarn_get_mscale(config.factor, config.mscale_all_dim);
            float attn_scaling_factor = 1 / sqrtf((float)(config.qk_rope_head_dim + config.qk_nope_head_dim));
            float mscale = yarn_get_mscale(config.factor, config.mscale_all_dim);
            attn_scaling_factor *= mscale * mscale;

            m /= 1.0f + 0.1f * logf(config.factor);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    DeepSeek2MoEBlock *layer = dynamic_cast<DeepSeek2MoEBlock *>(transformer->get_layer(i));
                    layer->attention.rope_mode      = RoPEMode::Original;
                    layer->attention.n_ctx          = config.max_length;
                    layer->attention.n_original_ctx = config.original_max_position_embeddings;
                    layer->attention.freq_base      = config.rope_theta;
                    layer->attention.freq_scale     = 1 / config.factor;
                    layer->attention.beta_fast      = config.beta_fast;
                    layer->attention.beta_slow      = config.beta_slow;
                    layer->attention.ext_factor               = 1.0f;
                    layer->attention.attn_factor              = m;
                    layer->attention.attn_scaling_factor      = attn_scaling_factor;
                    layer->mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    DeepSeek2Block *layer = dynamic_cast<DeepSeek2Block *>(transformer->get_layer(i));
                    layer->attention.rope_mode      = RoPEMode::Original;
                    layer->attention.n_ctx          = config.max_length;
                    layer->attention.n_original_ctx = config.original_max_position_embeddings;
                    layer->attention.freq_base      = config.rope_theta;
                    layer->attention.freq_scale     = 1 / config.factor;
                    layer->attention.beta_fast      = config.beta_fast;
                    layer->attention.beta_slow      = config.beta_slow;
                    layer->attention.ext_factor               = 1.0f;
                    layer->attention.attn_factor              = m;
                    layer->attention.attn_scaling_factor      = attn_scaling_factor;
                }
            }

            GRAPH_SIZE = 4096 * 4;
        }

        void load(ModelLoader &loader) override
        {
            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            SpeedMLAttention *attention = nullptr;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

                if (is_layer_moe(i))
                {
                    DeepSeek2MoEBlock *layer = dynamic_cast<DeepSeek2MoEBlock *>(transformer->get_layer(i));
                    attention = &layer->attention;

                    loader.read_tensor(layer_prefix + "input_layernorm.weight",          layer->input_layernorm.weight);
                    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);

                    for (int j = 0; j < config.n_routed_experts; j++)
                    {
                        std::string prefix = layer_prefix + "mlp.experts." + std::to_string(j) + '.';
                        loader.read_tensor(prefix + "down_proj.weight", layer->mlp.mlp1.experts[j].down_proj.weight);
                        loader.read_tensor(prefix + "gate_proj.weight", layer->mlp.mlp1.experts[j].gate_proj.weight);
                        loader.read_tensor(prefix + "up_proj.weight", layer->mlp.mlp1.experts[j].up_proj.weight);
                    }

                    loader.read_tensor(layer_prefix + "mlp.gate.weight", layer->mlp.mlp1.gate.weight);

                    loader.read_tensor(layer_prefix + "mlp.shared_experts.down_proj.weight", layer->mlp.mlp2.down_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.shared_experts.gate_proj.weight", layer->mlp.mlp2.gate_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.shared_experts.up_proj.weight",   layer->mlp.mlp2.up_proj.weight);
                }
                else
                {
                    DeepSeek2Block *layer = dynamic_cast<DeepSeek2Block *>(transformer->get_layer(i));
                    attention = &layer->attention;

                    loader.read_tensor(layer_prefix + "input_layernorm.weight",          layer->input_layernorm.weight);
                    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);

                    loader.read_tensor(layer_prefix + "mlp.down_proj.weight", layer->mlp.down_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", layer->mlp.gate_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   layer->mlp.up_proj.weight);
                }

                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight",            attention->q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.d_kv_proj.weight",         attention->d_kv_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_pe_proj.weight",         attention->k_pe_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.kv_norm.weight",           attention->kv_norm.weight);
                loader.read_tensor(layer_prefix + "self_attn.u_k_nope_proj.weight",     attention->u_k_nope_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.u_v_proj.weight",          attention->u_v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight",            attention->o_proj.weight);
            }

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 1844ull * 1024 * 1024;

        Config config;

        bool is_layer_moe(int layer_index)
        {
            return (layer_index >= config.first_k_dense_replace) && (layer_index % config.moe_layer_freq == 0);
        }
    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };
}
#endif
