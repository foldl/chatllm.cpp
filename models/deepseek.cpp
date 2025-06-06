namespace v1
{
    struct Config : public llama::v2::Config
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

    class Tokenizer : public llama::v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder)
            : llama::v2::Tokenizer::Tokenizer(config, chat_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;
    };

    class ConditionalGeneration : public llama::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_DEEPSEEK)
        {
        }
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        return size;
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
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            std::ostringstream oss_prompt;
            oss_prompt << tok->get_system_prompt() << "\n\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids, false, false);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "User: " << user << "\n\n";

        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "Assistant: ";
        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id);
    }
}

namespace v1_moe
{
    struct Config : public v1::Config
    {
        int num_key_value_heads;
        int first_k_dense_replace;
        int moe_intermediate_size;
        int moe_layer_freq;
        int n_routed_experts;
        int n_shared_experts;
        int norm_topk_prob;
        int num_experts_per_tok;

        float rope_theta;
    };

    typedef v1::Tokenizer Tokenizer;

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class DeepSeekSparseMoE : public BaseSparseMLP
    {
    public:
        DeepSeekSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<DeepSeekSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> DeepSeekMoEMLP;
        typedef LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, DeepSeekMoEMLP> DeepSeekMoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_DEEPSEEK_V1_MoE, int head_dim = -1)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3
                                + moe_layer_num * (12 + 4)
                                + dense_layer_num * 12;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK))
                << "unsupported MoE param";

            if (head_dim < 1)
                head_dim = config.hidden_size / config.num_attention_heads;
            if (head_dim != config.hidden_size / config.num_attention_heads)
                CHATLLM_CHECK(dense_layer_num == 0) << "customized head_dim";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new DeepSeekMoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                        config.num_key_value_heads,
                        head_dim,
                        config.max_length);
                    return layer;
                }
                else
                {
                    return new LlamaBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);

            Base::transformer = transformer;

            #define config_rope(attention)     do { \
                    attention.freq_base      = config.rope_theta;                           \
                } while (false)

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<DeepSeekMoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                    layer->mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    auto *layer = dynamic_cast<LlamaBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".mlp2.",              ".shared_experts."},
                {".mlp1.gate.",         ".gate."},
                {".mlp1.experts.",      ".experts."},
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:
        Config config;

        bool is_layer_moe(int layer_index)
        {
            return (layer_index >= config.first_k_dense_replace) && (layer_index % config.moe_layer_freq == 0);
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

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    typedef ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK> ConditionalGeneration;
}

namespace v2_light
{
    class QProj : public Block
    {
    public:
        QProj(InitContext *ctx, int hidden_size, int num_attention_heads,
            int q_lora_rank, int rope_dim, int qk_nope_head_dim, bool use_bias)
            : d_q_proj(q_lora_rank > 0 ? new Linear(ctx, hidden_size, q_lora_rank, use_bias) : nullptr),
              u_q_proj(q_lora_rank > 0 ? new Linear(ctx, q_lora_rank, (qk_nope_head_dim + rope_dim) * num_attention_heads, false) : nullptr),
              norm(q_lora_rank > 0 ? new RMSNorm(ctx, q_lora_rank) : nullptr),
              q_proj(q_lora_rank <= 0 ? new Linear(ctx, hidden_size, (qk_nope_head_dim + rope_dim) * num_attention_heads, use_bias) : nullptr)
        {}

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            if (q_proj)
            {
                r += q_proj->get_param_num(effective_only);
            }
            else
            {
                r += d_q_proj->get_param_num(effective_only);
                r += u_q_proj->get_param_num(effective_only);
                r += norm->get_param_num(effective_only);
            }
            return r;
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            ggml::tensor *tmpq = nullptr;

            if (q_proj)
            {
                tmpq = q_proj->forward(ctx, hidden_states);
            }
            else
            {
                ggml::tensor *q_lora = d_q_proj->forward(ctx, hidden_states);
                q_lora = norm->forward(ctx, q_lora);
                tmpq = u_q_proj->forward(ctx, q_lora);
            }

            return tmpq;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);

            if (q_proj)
            {
                q_proj->load(path + "q_proj.", loader);
            }
            else
            {
                d_q_proj->load(path + "d_q_proj.", loader);
                u_q_proj->load(path + "u_q_proj.", loader);
                norm->load(path + "q_norm.", loader);
            }
        }
    public:
        Linear *d_q_proj, *u_q_proj;
        RMSNorm *norm;
        Linear *q_proj;
    };

    // for opt_speed == false:
    //      k_pe    -> K cache;
    //      kv_lora -> V cache.
    // FIXME: for opt_speed == false, GGML_TYPE_F32 is used because GGML complains
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
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias,
                      int cache_length)
            : KVCacheAttention(CacheTypeChanger(ctx, opt_speed ? ctx->cache_dtype : ggml::type::GGML_TYPE_F32),
                               num_attention_heads, num_kv_heads,
                               opt_speed ? (qk_nope_head_dim + rope_dim) * num_kv_heads : rope_dim * 1,
                               opt_speed ? v_head_dim * num_kv_heads : kv_lora_rank,
                               max_length,
                               cache_length),
              d_kv_proj(ctx, hidden_size, kv_lora_rank, nullptr, use_bias),
              k_pe_proj(ctx, hidden_size, rope_dim, nullptr, use_bias),
              u_k_nope_proj(ctx, kv_lora_rank, qk_nope_head_dim * num_kv_heads, nullptr, false),
              u_v_proj(ctx, kv_lora_rank, v_head_dim * num_kv_heads, nullptr, false),
              q_proj(ctx, hidden_size, num_attention_heads, q_lora_rank, rope_dim, qk_nope_head_dim, use_bias),
              o_proj(ctx, v_head_dim * num_attention_heads, hidden_size, use_bias),
              kv_norm(ctx, kv_lora_rank),
              kv_lora_rank(kv_lora_rank),
              rope_dim(rope_dim),
              qk_nope_head_dim(qk_nope_head_dim),
              v_head_dim(v_head_dim)
        {
        }

        BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias)
            : BaseMLAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                              q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                              use_bias,
                              max_length)
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
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            if (opt_speed)
                return forward_speed(ctx, hidden_states, n_past);
            else
                return forward_memory(ctx, hidden_states, n_past);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            KVCacheAttention::load(path, loader);
            q_proj.load(path + "", loader);

            d_kv_proj.load(path + "d_kv_proj.", loader);
            k_pe_proj.load(path + "k_pe_proj.", loader);
            u_k_nope_proj.load(path + "u_k_nope_proj.", loader);
            u_v_proj.load(path + "u_v_proj.", loader);
            o_proj.load(path + "o_proj.", loader);
            kv_norm.load(path + "kv_norm.", loader);
        }

    protected:
        ggml::tensor *forward_speed(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
        {
            const int hidden_size = o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            KVCacheAttention::before_forward(ctx, n_past, qlen);

            ggml::tensor *kv_lora = d_kv_proj.forward(ctx, hidden_states);
            kv_lora = kv_norm.forward(ctx, kv_lora);

            ggml::tensor *tmpv = u_v_proj.forward(ctx, kv_lora);

            ggml::tensor *k_nope = u_k_nope_proj.forward(ctx, kv_lora);
            ggml::tensor *k_pe = k_pe_proj.forward(ctx, hidden_states);

            ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);

            ggml::tensor *scores = cross_attention_speed(ctx, hidden_size, n_past, qlen, tmpq, k_nope, k_pe, tmpv);

            ggml::tensor *attn_output = o_proj.forward(ctx, scores);

            return attn_output;
        }

        ggml::tensor *cross_attention_speed(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k_nope, ggml::tensor *k_pe, ggml::tensor *v)
        {
            // [qlen, heads, head_size]
            k_pe = ggml::reshape_3d(ctx, k_pe, rope_dim, 1, qlen);
            k_pe = apply_pos_embedding_k(ctx, k_pe, rope_dim * 1, qlen, pos);

            ggml::tensor * key_layer = ggml::new_zeros(ctx, k_pe->type,
                qk_nope_head_dim + rope_dim, num_kv_heads, qlen);

            ggml::tensor * k_nope_dst = ggml::view_3d(ctx, key_layer,
                         qk_nope_head_dim, num_kv_heads, qlen,
                         key_layer->nb[1], key_layer->nb[2],
                         0);
            ggml::tensor * k_pe_dst = ggml::view_3d(ctx, key_layer,
                         rope_dim, num_kv_heads, qlen,
                         key_layer->nb[1], key_layer->nb[2],
                         qk_nope_head_dim * ggml::element_size(key_layer));
            k_nope = ggml::reshape_3d(ctx, k_nope, qk_nope_head_dim, num_kv_heads, qlen);

            ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_nope, k_nope_dst));
            ggml::build_forward_expand(ctx, ggml::add_inplace(ctx, k_pe_dst, k_pe));  // auto-broadcasting

            // [qlen, heads, head_size]
            ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, qk_nope_head_dim + rope_dim, num_attention_heads, qlen);
            ggml::tensor * q_pe = ggml::view_3d(ctx, query_layer,
                         rope_dim, num_attention_heads, qlen,
                         query_layer->nb[1], query_layer->nb[2],
                         qk_nope_head_dim * ggml::element_size(query_layer));

            if (ctx->is_using_gpu())
            {
                // TODO: optimize (GPU rope requires continuous)
                ggml::tensor * q_pe_cont = ggml::cont(ctx, q_pe);
                q_pe_cont = apply_pos_embedding_q(ctx, q_pe_cont, rope_dim * num_attention_heads, qlen, pos);
                q_pe = ggml::cpy(ctx, q_pe_cont, q_pe);
            }
            else
            {
                q_pe = apply_pos_embedding_q(ctx, q_pe, rope_dim * num_attention_heads, qlen, pos);
            }

            ggml::build_forward_expand(ctx, q_pe);

            ggml::tensor *attn_scores = cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

            return attn_scores;
        }

        ggml::tensor *forward_memory(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
        {
            const int hidden_size = o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            KVCacheAttention::before_forward(ctx, n_past, qlen);

            ggml::tensor *kv_lora = d_kv_proj.forward(ctx, hidden_states);
            kv_lora = kv_norm.forward(ctx, kv_lora);

            ggml::tensor *k_pe = k_pe_proj.forward(ctx, hidden_states);

            ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);

            ggml::tensor *scores = cross_attention_memory(ctx, hidden_size, n_past, qlen, tmpq, k_pe, kv_lora);

            ggml::tensor *attn_output = o_proj.forward(ctx, scores);

            return attn_output;
        }

        ggml::tensor *cross_attention_memory(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k_pe, ggml::tensor *kv_lora)
        {
            // [qlen, heads, head_size]
            k_pe = ggml::reshape_3d(ctx, k_pe, rope_dim, 1, qlen);
            k_pe = apply_pos_embedding_k(ctx, k_pe, rope_dim * 1, qlen, pos);
            k_pe = ggml::reshape_1d(ctx, k_pe, rope_dim * 1 * qlen);

            // [qlen, heads, head_size]
            ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, qk_nope_head_dim + rope_dim, num_attention_heads, qlen);
            ggml::tensor * q_pe = ggml::view_3d(ctx, query_layer,
                         rope_dim, num_attention_heads, qlen,
                         query_layer->nb[1], query_layer->nb[2],
                         qk_nope_head_dim * ggml::element_size(query_layer));

            if (ctx->is_using_gpu())
            {
                // TODO: optimize (GPU rope requires continuous)
                ggml::tensor * q_pe_cont = ggml::cont(ctx, q_pe);
                q_pe_cont = apply_pos_embedding_q(ctx, q_pe_cont, rope_dim * num_attention_heads, qlen, pos);
                q_pe = ggml::cpy(ctx, q_pe_cont, q_pe);
            }
            else
            {
                q_pe = apply_pos_embedding_q(ctx, q_pe, rope_dim * num_attention_heads, qlen, pos);
            }

            ggml::build_forward_expand(ctx, q_pe);

            ggml::tensor *attn_scores = cross_attention_after_pe_memory(ctx, hidden_size, n_past, qlen, query_layer, k_pe, kv_lora);

            return attn_scores;
        }

        ggml::tensor *get_k_pe_from_cache(ComputeContext *ctx, const int n_past, const int qlen)
        {
            ggml::tensor *k_pe = nullptr;

            k_pe = ggml::view_2d(ctx, k_cache, k_hidden_size, n_past + qlen,
                                ggml::row_size(k_cache),
                                0);

            return k_pe;
        }

        ggml::tensor *get_kv_lora_from_cache(ComputeContext *ctx, const int n_past, const int qlen)
        {
            ggml::tensor *kv_lora = nullptr;

            kv_lora = ggml::view_2d(ctx, v_cache, v_hidden_size, n_past + qlen,
                                   v_hidden_size * ggml::element_size(v_cache),
                                   0);

            return kv_lora;
        }

        void save_lora_to_cache(ComputeContext *ctx, const int n_past, const int qlen,
            ggml::tensor *k_pe, ggml::tensor *kv_lora)
        {
            ggml::tensor * pe_cache_view = ggml::view_1d(ctx, k_cache, qlen * k_hidden_size,
                                        ggml::row_size(k_cache) * n_past);

            ggml::tensor * kv_cache_view = ggml::view_1d(ctx, v_cache, qlen * v_hidden_size,
                                        ggml::element_size(v_cache) * v_hidden_size * n_past);

            ggml::tensor * pe_view = ggml::view_1d(ctx, k_pe,    qlen * k_hidden_size, 0);
            ggml::tensor * kv_view = ggml::view_1d(ctx, kv_lora, qlen * v_hidden_size, 0);

            // important: storing RoPE-ed version of K in the KV cache!
            ggml::build_forward_expand(ctx, ggml::cpy(ctx, pe_view, pe_cache_view));
            ggml::build_forward_expand(ctx, ggml::cpy(ctx, kv_view, kv_cache_view));
        }

        ggml::tensor *cross_attention_after_pe_memory(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen0,
                                             ggml::tensor *query_layer, ggml::tensor *k_pe, ggml::tensor *kv_lora)
        {
            const int head_size = qk_nope_head_dim + rope_dim;

            if (!attn_scaling)
                query_layer = ggml::scale(ctx, query_layer, 1.f / sqrtf((float)head_size));

            query_layer = ggml::permute(ctx, query_layer, 0, 2, 1, 3);                     // [heads, qlen, head_size]

            // store key and value to memory
            save_lora_to_cache(ctx, n_past, qlen0, k_pe, kv_lora);

            ggml::tensor *k_pe_all       = get_k_pe_from_cache(ctx, n_past, qlen0);
            ggml::tensor *kv_lora_all    = get_kv_lora_from_cache(ctx, n_past, qlen0);

            const int qlen = n_past + qlen0;

            ggml::tensor *k_nope = u_k_nope_proj.forward(ctx, kv_lora_all);

            ggml::tensor *key_layer = ggml::new_zeros(ctx, k_nope->type,
                qk_nope_head_dim + rope_dim, num_kv_heads, qlen); // [qlen, heads, head_size]

            ggml::tensor * k_nope_dst = ggml::view_3d(ctx, key_layer,
                         qk_nope_head_dim, num_kv_heads, qlen,
                         key_layer->nb[1], key_layer->nb[2],
                         0);
            ggml::tensor * k_pe_dst = ggml::view_3d(ctx, key_layer,
                         rope_dim, num_kv_heads, qlen,
                         key_layer->nb[1], key_layer->nb[2],
                         qk_nope_head_dim * ggml::element_size(key_layer));
            k_nope   = ggml::reshape_3d(ctx, k_nope,     qk_nope_head_dim, num_kv_heads, qlen);
            k_pe_all = ggml::reshape_3d(ctx, k_pe_all,   rope_dim,         1,            qlen);
            ggml::build_forward_expand(ctx, ggml::cpy(ctx, k_nope, k_nope_dst));
            ggml::build_forward_expand(ctx, ggml::add_inplace(ctx, k_pe_dst, k_pe_all));  // auto-broadcasting

            ggml::tensor *value_layer = u_v_proj.forward(ctx, kv_lora_all);
            value_layer = ggml::view_3d(ctx, value_layer,
                        v_head_dim, num_kv_heads, qlen,
                        v_head_dim * ggml::element_size(v_cache),
                        v_head_dim * num_kv_heads * ggml::element_size(v_cache),
                        0); // [qlen, heads, head_size]

            key_layer   = ggml::permute(ctx, key_layer,   0, 2, 1, 3); // [qlen, heads, head_size] -> [heads, qlen, head_size]
            value_layer = ggml::permute(ctx, value_layer, 1, 2, 0, 3); // [qlen, heads, head_size] -> [heads, head_size, qlen]
            value_layer = ggml::cont(ctx, value_layer);

            ggml::tensor *attn_scores = calc_attn_scores(ctx, hidden_size, n_past, qlen0, key_layer, query_layer, value_layer);
            return attn_scores;
        }

    public:
        Linear d_kv_proj, k_pe_proj, u_k_nope_proj, u_v_proj;
        QProj q_proj;
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

    typedef LMBlock1<RMSNorm, SpeedMLAttention, RMSNorm, SiLUMLP> DeepSeek2Block;

    static float yarn_get_mscale(float scale = 1.0f, float mscale = 1.0f)
    {
        if (scale <= 1.0f)
            return 1.0f;
        return 0.1f * mscale * logf(scale) + 1.0f;
    }

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class DeepSeekSparseMoE : public BaseSparseMLP
    {
    public:
        DeepSeekSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size, bool gate_use_bias)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, gate_use_bias)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<DeepSeekSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> DeepSeekMoEMLP;
        typedef LMBlock1<RMSNorm, SpeedMLAttention, RMSNorm, DeepSeekMoEMLP> DeepSeek2MoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config) : ConditionalGeneration0(config, runtime_config, MODEL_TYPE_DEEPSEEK_V2_LIGHT, -1)
        {}

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int q_lora_rank, BaseSparseMLP::ScoreFunc score_func = BaseSparseMLP::ScoreFunc::Softmax,
            bool gate_use_bias = false, bool always_scaling = false)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3
                                + moe_layer_num * (16 + 3 + (gate_use_bias ? 1 : 0))
                                + dense_layer_num * 15
                                + (q_lora_rank > 0 ? config.num_hidden_layers * 2 : 0) ;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK)
                            && (config.n_group == 1))
                << "unsupported MoE param";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new DeepSeek2MoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                        config.num_key_value_heads, config.max_length,
                        q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                        false, gate_use_bias);
                    auto sparse = dynamic_cast<BaseSparseMLP *>(&layer->mlp.mlp1);
                    sparse->score_func = score_func;
                    sparse->routed_scaling_factor = config.routed_scaling_factor;
                    sparse->always_scaling = always_scaling;
                    return layer;
                }
                else
                {
                    return new DeepSeek2Block(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length,
                                                q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                                false);
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);
            Base::transformer = transformer;

            float m = 1.0f;
            float attn_scaling_factor = -1.0f;
            if (config.original_max_position_embeddings > 0)
            {
                m = yarn_get_mscale(config.factor, config.mscale) / yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor = 1 / sqrtf((float)(config.qk_rope_head_dim + config.qk_nope_head_dim));
                float mscale = yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor *= mscale * mscale;
                m /= 1.0f + 0.1f * logf(config.factor);
            }

            #define config_rope(attention)     do { \
                    attention.rope_mode      = RoPEMode::Original;                          \
                    attention.freq_base      = config.rope_theta;                           \
                    if (config.original_max_position_embeddings > 0)                        \
                    {                                                                       \
                    attention.n_ctx          = config.max_length;                           \
                    attention.n_original_ctx = config.original_max_position_embeddings;     \
                    attention.freq_scale     = 1 / config.factor;                           \
                    attention.beta_fast      = config.beta_fast;                            \
                    attention.beta_slow      = config.beta_slow;                            \
                    attention.ext_factor               = 1.0f;                              \
                    attention.attn_factor              = m;                                 \
                    attention.attn_scaling_factor      = attn_scaling_factor;               \
                    }                                                                       \
                } while (false)

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    DeepSeek2MoEBlock *layer = dynamic_cast<DeepSeek2MoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                    layer->mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    DeepSeek2Block *layer = dynamic_cast<DeepSeek2Block *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".mlp2.",              ".shared_experts."},
                {".mlp1.gate.",         ".gate."},
                {".mlp1.experts.",      ".experts."},
                {".mlp1.gate_score_correction_bias",     ".gate.e_score_correction_bias"}
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:
        Config config;

        bool is_layer_moe(int layer_index)
        {
            return (layer_index >= config.first_k_dense_replace) && (layer_index % config.moe_layer_freq == 0);
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

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    typedef ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK> ConditionalGeneration;
}

namespace v2
{
    struct Config : public v2_light::Config
    {
        int q_lora_rank;
        int topk_group;     // TODO: group_limited_greedy
    };

    typedef v1::Tokenizer Tokenizer;

    const int NUM_EXPERTS                   =  160;
    const int EXPERTS_PER_TOK               =  6;

    class ConditionalGeneration : public v2_light::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : v2_light::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>(config, runtime_config, MODEL_TYPE_DEEPSEEK_V2, config.q_lora_rank)
        {
            CHATLLM_CHECK(config.topk_group == 1) << "unsupported MoE param";
        }
    };
}

namespace v3_light
{
    struct Config : public v2_light::Config
    {
    };

    typedef v1::Tokenizer Tokenizer;

    class ConditionalGeneration : public v2_light::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_DEEPSEEK_V3_LIGHT)
            : v2_light::ConditionalGeneration(config, runtime_config, type, -1, BaseSparseMLP::ScoreFunc::Sigmoid, true, true)
        {
        }
    };
}