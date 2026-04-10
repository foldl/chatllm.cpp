#include "gemma.h"

namespace chatllm::gemma::v4
{
    const int MAX_LAYERS        = 128;

    struct Config : public BaseConfig
    {
        int attention_k_eq_v;
        int global_head_dim;
        int head_dim;
        int hidden_size_per_layer_input;
        int moe_intermediate_size;
        int num_experts;
        int num_global_key_value_heads;
        int num_key_value_heads;
        int num_kv_shared_layers;
        int sliding_window;
        int top_k_experts;
        int use_double_wide_mlp;
        int layer_is_swa[MAX_LAYERS];

        float final_logit_soft_capping;
        float full_attn_partial_rotary_factor;
        float full_attn_rope_theta;
        float swa_attn_rope_theta;
    };

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    Tokenizer::Tokenizer(const BaseConfig &config)
        : v1::Tokenizer(config)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        auto t = new tokenizer::BPEProcessor2();
        tp = t;
        size_t size = tp->Load(buffer, n_vocab);
        t->SetDecoderType(tokenizer::BPEProcessor2::DecoderType::Sequence);

        int id = tp->PieceToId("<pad>");
        if (id >= 0) pad_token_id = id;
        start_of_turn_token_id = tp->PieceToId("<|turn>");
        end_of_turn_token_id   = tp->PieceToId("<turn|>");
        nl_token_id = tp->PieceToId("\n");

        terminate_ids.insert(end_of_turn_token_id);

        return size;
    }

    class PerLayerEmbedding : public Block
    {
    public:
        PerLayerEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim, int embedding_dim_per_layer, int num_hidden_layers);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *input_embeds) override;
        ggml::tensor *get_input_for_layer(ComputeContext *ctx, int layer_id);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

    public:
        const int embedding_dim_per_layer;
        const int num_hidden_layers;
        const float per_layer_model_projection_scale;
        const float per_layer_input_scale;
        Embedding   embed_tokens_per_layer;
        Linear      per_layer_model_projection;
        RMSNorm     per_layer_projection_norm;
    protected:
        ggml::tensor *layer_id;
        ggml::tensor *per_layer_embs = nullptr;
    };

    class PerLayerEmbeddingBlock : public Block
    {
    public:
        PerLayerEmbeddingBlock(InitContext *ctx, int hidden_size, int embedding_dim_per_layer, ActFunc act = ActFunc::GELU);

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds) override;

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

    public:
        ActFunc     act;
        Linear      per_layer_input_gate;
        Linear      per_layer_projection;
        RMSNorm     post_per_layer_input_norm;
    };

    PerLayerEmbedding::PerLayerEmbedding(InitContext *ctx, int num_embeddings, int embedding_dim, int embedding_dim_per_layer, int num_hidden_layers) :
        embedding_dim_per_layer(embedding_dim_per_layer),
        num_hidden_layers(num_hidden_layers),
        per_layer_model_projection_scale(sqrtf(1.0f / embedding_dim)),
        per_layer_input_scale(sqrtf(0.5f)),
        embed_tokens_per_layer(ctx, ggml::type_fallback(ctx->dtype, embedding_dim_per_layer), num_embeddings, embedding_dim_per_layer * num_hidden_layers),
        per_layer_model_projection(ctx, embedding_dim, embedding_dim_per_layer * num_hidden_layers, false),
        per_layer_projection_norm(ctx, embedding_dim_per_layer),
        layer_id(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, num_hidden_layers))
    {
        ctx->get_allocator()->alloc(layer_id);
        std::vector<int32_t> ids;
        for (int i = 0; i < num_hidden_layers; i++) ids.push_back(i);
        Backend::write_tensor_data(layer_id, ids.data());
    }

    ggml::tensor *PerLayerEmbedding::forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *input_embeds)
    {
        per_layer_embs = embed_tokens_per_layer.forward(ctx, input_ids);
        per_layer_embs = ggml::reshape(ctx, per_layer_embs, embedding_dim_per_layer, num_hidden_layers, ggml::get_dim(per_layer_embs, 1), ggml::get_dim(per_layer_embs, 2));

        auto x = input_embeds;
        x = per_layer_model_projection.forward(ctx, x);
        x = ggml::scale  (ctx, x, per_layer_model_projection_scale);
        x = ggml::reshape(ctx, x, embedding_dim_per_layer, num_hidden_layers, ggml::get_dim(x, 1), ggml::get_dim(x, 2));
        x = per_layer_projection_norm.forward(ctx, x);

        per_layer_embs = ggml::add(ctx, per_layer_embs, x);
        per_layer_embs = ggml::scale(ctx, per_layer_embs, per_layer_input_scale);
        ggml::build_forward_expand(ctx, per_layer_embs);
        return per_layer_embs;
    }

    ggml::tensor *PerLayerEmbedding::get_input_for_layer(ComputeContext *ctx, int layer_id)
    {
        auto id = ggml::view_1d (ctx, this->layer_id, 1, layer_id * ggml::element_size(this->layer_id));
             id = ggml::repeat  (ctx, id, ggml::get_dim(id, 0), ggml::get_dim(per_layer_embs, 2));
        auto r  = ggml::get_rows(ctx, per_layer_embs, id);
             r  = ggml::reshape (ctx, r, ggml::get_dim(r, 0), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
        return r;
    }

    int64_t PerLayerEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += per_layer_model_projection.get_param_num(effective_only);
        r +=  per_layer_projection_norm.get_param_num(effective_only);
        r +=     embed_tokens_per_layer.get_param_num(effective_only);
        return r;
    }

    void PerLayerEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        per_layer_model_projection.load(path + "per_layer_model_projection.", loader);
         per_layer_projection_norm.load(path + "per_layer_projection_norm.", loader);
            embed_tokens_per_layer.load(path + "embed_tokens_per_layer.", loader);
    }

    PerLayerEmbeddingBlock::PerLayerEmbeddingBlock(InitContext *ctx, int hidden_size, int embedding_dim_per_layer, ActFunc act):
        act(act),
        per_layer_input_gate(ctx, hidden_size, embedding_dim_per_layer, false),
        per_layer_projection(ctx, embedding_dim_per_layer, hidden_size, false),
        post_per_layer_input_norm(ctx, hidden_size)
    {
    }

    ggml::tensor *PerLayerEmbeddingBlock::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds)
    {
        auto residual = hidden_states;
        hidden_states = per_layer_input_gate.forward(ctx, hidden_states);
        hidden_states = ggml::act(ctx, act, hidden_states);
        hidden_states = ggml::mul(ctx, hidden_states, layer_embeds);
        hidden_states = per_layer_projection.forward(ctx, hidden_states);
        hidden_states = post_per_layer_input_norm.forward(ctx, hidden_states);
        hidden_states = ggml::add(ctx, residual, hidden_states);
        return hidden_states;
    }

    int64_t PerLayerEmbeddingBlock::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += post_per_layer_input_norm.get_param_num(effective_only);
        r += per_layer_input_gate.get_param_num(effective_only);
        r += per_layer_projection.get_param_num(effective_only);
        return r;
    }

    void PerLayerEmbeddingBlock::load(const std::string &path, TensorLoader *loader)
    {
        post_per_layer_input_norm.load(path + "post_per_layer_input_norm.", loader);
             per_layer_input_gate.load(path + "per_layer_input_gate.", loader);
             per_layer_projection.load(path + "per_layer_projection.", loader);
    }

    template <class BaseAttn> class Gemma4Attention : public BaseAttn
    {
    public:
        using BaseAttn::BaseAttn;

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = KVCacheAttention::get_param_num(effective_only);
            r += BaseAttn::q_proj.get_param_num(effective_only);
            r += BaseAttn::k_proj.get_param_num(effective_only);
            r += use_k_as_v ? 0 : BaseAttn::v_proj.get_param_num(effective_only);
            r += BaseAttn::o_proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            KVCacheAttention::load(path, loader);
            BaseAttn::q_proj.load(path + "q_proj.", loader);
            BaseAttn::k_proj.load(path + "k_proj.", loader);
            BaseAttn::o_proj.load(path + "o_proj.", loader);
            if (!use_k_as_v)
                BaseAttn::v_proj.load(path + "v_proj.", loader);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            const int hidden_size = BaseAttn::o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            BaseAttn::before_forward(ctx, n_past, qlen);

            ggml::tensor *tmpq = BaseAttn::q_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpk = shared_attn ? nullptr : BaseAttn::k_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpv = nullptr;

            if (!shared_attn)
            {
                const int head_size = BaseAttn::v_proj.out_features() / BaseAttn::num_kv_heads;
                tmpv = use_k_as_v ? tmpk : BaseAttn::v_proj.forward(ctx, hidden_states);
                tmpv = ggml::reshape (ctx, tmpv, head_size, BaseAttn::num_kv_heads, ggml::get_dim(tmpv, 1), ggml::get_dim(tmpv, 2));
                tmpv = ggml::rms_norm(ctx, tmpv, 1e-6f);
                tmpv = ggml::reshape (ctx, tmpv, head_size * BaseAttn::num_kv_heads, ggml::get_dim(tmpv, 2), ggml::get_dim(tmpv, 3));
            }

            ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

            ggml::tensor *attn_output = BaseAttn::o_proj.forward(ctx, scores);
            return attn_output;
        }

        using BaseAttn::apply_pos_embedding_k;
        using BaseAttn::apply_pos_embedding_q;

        ggml::tensor *cross_attention(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                             ggml::tensor *q, ggml::tensor *k, ggml::tensor *v) override
        {
            const int head_size = hidden_size / BaseAttn::num_attention_heads;
            const int batch_size = ggml::get_dim(q, 2);

            // [qlen, heads, head_size]
            ggml::tensor * key_layer = nullptr;
            if (k)
            {
                key_layer = ggml::reshape_4d(ctx, k, head_size, BaseAttn::num_kv_heads, qlen, batch_size);
                key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, BaseAttn::pos);
            }

            // [qlen, heads, head_size]
            ggml::tensor * query_layer = ggml::reshape_4d(ctx, q, head_size, BaseAttn::num_attention_heads, qlen, batch_size);
            query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, BaseAttn::pos);

            ggml::tensor *attn_scores = BaseAttn::cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

            return attn_scores;
        }

        void save_to_cache(ComputeContext *ctx, const int n_past, const int qlen, ggml::tensor *k, ggml::tensor *v) override
        {
            if (shared_attn) return;
            BaseAttn::save_to_cache(ctx, n_past, qlen, k, v);
        }

        ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            return shared_attn ? shared_attn->get_k_from_cache(ctx, hidden_size, n_past, qlen) : BaseAttn::get_k_from_cache(ctx, hidden_size, n_past, qlen);
        }

        ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
        {
            return shared_attn ? shared_attn->get_v_from_cache(ctx, hidden_size, n_past, qlen) : BaseAttn::get_v_from_cache(ctx, hidden_size, n_past, qlen);
        }

    public:
        bool use_k_as_v = false;
        Gemma4Attention *shared_attn = nullptr;
    };

    template <int sliding_window_len> class Gemma4SWASelfAttention : public QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>>
    {
    public:
        Gemma4SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>>
                (ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    template <int sliding_window_len> class Gemma4DenseSWABlock : public LMBlock4<RMSNorm, Gemma4SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
    {
    public:
        typedef LMBlock4<RMSNorm, Gemma4SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm> Base;
        Gemma4DenseSWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds, int n_past) override
        {
            hidden_states = Base::forward(ctx, hidden_states, n_past);
            if (layer_embeds)
            {
                hidden_states = per_layer->forward(ctx, hidden_states, layer_embeds);
            }
            hidden_states = ggml::scale(ctx, hidden_states, layer_scalar);

            return hidden_states;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            loader->read_scaler(path + "layer_scalar", &layer_scalar);
            if (per_layer)
                per_layer->load(path, loader);
        }

    public:
        PerLayerEmbeddingBlock *per_layer = nullptr;
        float layer_scalar = -1.0f;
    };

    class Gemma4FullSelfAttention : public QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<BaseAttention>>
    {
    public:
        Gemma4FullSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<BaseAttention>>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
        }
    };

    class Gemma4DenseFullBlock : public LMBlock4<RMSNorm, Gemma4FullSelfAttention, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
    {
    public:
        typedef LMBlock4<RMSNorm, Gemma4FullSelfAttention, RMSNorm, RMSNorm, GELUMLP, RMSNorm> Base;
        Gemma4DenseFullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *layer_embeds, int n_past) override
        {
            hidden_states = Base::forward(ctx, hidden_states, n_past);
            if (layer_embeds)
            {
                hidden_states = per_layer->forward(ctx, hidden_states, layer_embeds);
            }
            hidden_states = ggml::scale(ctx, hidden_states, layer_scalar);

            return hidden_states;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            loader->read_scaler(path + "layer_scalar", &layer_scalar);
            if (per_layer)
                per_layer->load(path, loader);
        }
    public:
        PerLayerEmbeddingBlock *per_layer = nullptr;
        float layer_scalar = -1.0f;
    };

    class PerLayerBlocks
    {
    public:
        PerLayerBlocks(InitContext *ctx, const Config &config);
        int64_t get_param_num(bool effective_only) const;
    public:
        std::unique_ptr<PerLayerEmbedding>  per_layer_emb;
        std::vector<PerLayerEmbeddingBlock> per_layer_blocks;
    };

    class ModelClass : public PerLayerBlocks, public HeterogeneousModel
    {
    public:
        ModelClass(InitContext *ctx, const Config &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override;
        void load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids) override;
        int64_t get_param_num(bool effective_only) const override;
    protected:
        static Block *create_layer(InitContext *ctx, PerLayerBlocks *layer_blocks, const Config &config, int layer_index);
    public:
        const Config config;
    };

    template <class T> static void fix_proportional_rope(InitContext *ctx, const int head_dim, const int rope_dim, RoPESelfAttention<T> &attn)
    {
        const int n_dims = attn.rope_dim;
        attn.freq_factors = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
        ctx->get_allocator()->alloc(attn.freq_factors);
        std::vector<float> freq_factors(n_dims / 2, 1.0f);

        for (int i = rope_dim / 2; i < n_dims / 2; i++)
        {
            freq_factors[i] = INFINITY;
        }
        Backend::write_tensor_data(attn.freq_factors, freq_factors.data());
    }

    Block *ModelClass::create_layer(InitContext *ctx, PerLayerBlocks *layer_blocks, const Config &config, int layer_index)
    {
        const bool is_kv_shared_layer = layer_index >= (config.num_hidden_layers - config.num_kv_shared_layers);
        BlockParams::DisableCache disable(is_kv_shared_layer);

        const int intermediate_size = is_kv_shared_layer && (config.use_double_wide_mlp != 0) ? config.intermediate_size * 2 : config.intermediate_size;

        if (config.layer_is_swa[layer_index])
        {
            switch (config.sliding_window)
            {
            case 512:
                {
                    auto r = new Gemma4DenseSWABlock<512>(ctx, config.hidden_size, config.num_attention_heads, intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
                    r->attention.freq_base = config.swa_attn_rope_theta;
                    r->attention.attn_scaling_factor = 1.0f;

                    static decltype(r->attention) *last_non_shared = nullptr;
                    if (is_kv_shared_layer)
                        r->attention.shared_attn = last_non_shared;
                    else
                        last_non_shared = &r->attention;
                    if (config.hidden_size_per_layer_input > 0)
                        r->per_layer = &layer_blocks->per_layer_blocks[layer_index];
                    return r;
                }
            default:
                CHATLLM_CHECK(false) << "unsupported sliding_window: " << config.sliding_window;
                break;
            }
        }
        else
        {
            const int head_dim = config.global_head_dim > 0 ? config.global_head_dim : config.head_dim;
            const int num_key_value_heads = config.attention_k_eq_v != 0 ? config.num_global_key_value_heads : config.num_key_value_heads;

            auto r = new Gemma4DenseFullBlock(ctx, config.hidden_size, config.num_attention_heads, intermediate_size,
                                        num_key_value_heads, head_dim, config.max_length);
            r->attention.freq_base = config.full_attn_rope_theta;
            r->attention.use_k_as_v = config.attention_k_eq_v != 0;
            r->attention.attn_scaling_factor = 1.0f;

            fix_proportional_rope(ctx, head_dim, (int)(r->attention.rope_dim * config.full_attn_partial_rotary_factor), r->attention);

            static decltype(r->attention) *last_non_shared = nullptr;
            if (is_kv_shared_layer)
                r->attention.shared_attn = last_non_shared;
            else
                last_non_shared = &r->attention;
            if (config.hidden_size_per_layer_input > 0)
                r->per_layer = &layer_blocks->per_layer_blocks[layer_index];
            return r;
        }
        CHATLLM_CHECK(false) << "bad";
        return nullptr;
    }

    int64_t ModelClass::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += PerLayerBlocks::get_param_num(effective_only);
        r += HeterogeneousModel::get_param_num(effective_only);
        return r;
    }

    PerLayerBlocks::PerLayerBlocks(InitContext *ctx, const Config &config)
    {
        if (config.hidden_size_per_layer_input > 0)
        {
            per_layer_emb.reset(new PerLayerEmbedding(ctx, config.vocab_size, config.hidden_size, config.hidden_size_per_layer_input, config.num_hidden_layers));
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                per_layer_blocks.emplace_back(ctx, config.hidden_size, config.hidden_size_per_layer_input);
            }
        }
    }

    int64_t PerLayerBlocks::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (per_layer_emb.get() != nullptr) r += per_layer_emb->get_param_num(effective_only);
        for (int i = 0; i < (int)per_layer_blocks.size(); i++)
            r += per_layer_blocks[i].get_param_num(effective_only);
        return r;
    }

    ModelClass::ModelClass(InitContext *ctx, const Config &config):
        PerLayerBlocks(ctx, config),
        HeterogeneousModel(ctx, config.num_hidden_layers, config.hidden_size,
            create_embedding<Embedding>(ctx, config),
            create_final_norm<RMSNorm>(ctx, config),
            nullptr,
            [this, &config](InitContext *ctx, int layer_index) -> Block * { return ModelClass::create_layer(ctx, this, config, layer_index); }),
        config(config)
    {
    }

    void ModelClass::load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids)
    {
        HeterogeneousModel::load(path, loader, layer_ids);
        if (per_layer_emb.get())
            per_layer_emb->load(path, loader);
    }

    ggml::tensor *ModelClass::forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past)
    {
        before_forward(ctx, input_ids, n_past);

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        ggml::tensor *hidden_states = custom_embedding ? custom_embedding(ctx, input_ids) :  word_embeddings->forward(ctx, input_ids);
        if (per_layer_emb.get())
        {
            per_layer_emb->forward(ctx, input_ids, hidden_states);
        }

        for (auto &layer : layers)
        {
            ctx->move_to_layer(layer->get_id());
            if (layer_preprocess.get())
            {
                auto t = layer_preprocess->forward(this, ctx, hidden_states, layer->get_id());
                if (t) hidden_states = t;
            }

            auto layer_embeds = per_layer_emb.get() ? per_layer_emb->get_input_for_layer(ctx, layer->get_id()) : nullptr;

            hidden_states = layer->forward(ctx, hidden_states, layer_embeds, n_past);
        }

        last_hidden_state = hidden_states;

        ctx->move_to_layer(LayerAllocatorManager::Epilog);
        return final_steps->forward(this, ctx, input_ids, hidden_states);
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA4);
        void load(ModelLoader &loader) override;

        void set_tokenizer(BaseTokenizer *tokenizer) override;
        bool load_more(const json::JSON &config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;

        void before_generate(const GenerationConfig &gen_config) override;
    public:
        int tensor_num_of_layer(const Config &config, int layer_id) const;
        v2::TanhScaling logits_pp;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
        logits_pp(1.0f / config.final_logit_soft_capping / sqrtf((float)config.hidden_size), config.final_logit_soft_capping)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        size_t num_tensors = 2;
        for (int i = 0; i < config.num_hidden_layers; i++) num_tensors += tensor_num_of_layer(config, i);
        if (config.hidden_size_per_layer_input)
        {
            num_tensors += 4 + 3 * config.num_hidden_layers;
        }
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config);
        get_typed_transformer<ModelClass>()->logits_pp = &logits_pp;

        batch_input = config.sliding_window - 1;
        w_ctx_.check_used_mem_size(true);
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".pre_attention_layernorm.",   ".input_layernorm."},
            {".pre_mlp_layernorm.",         ".pre_feedforward_layernorm."},
            {".post_mlp_layernorm.",        ".post_feedforward_layernorm."},
        });

        BaseModelForConditionalGeneration::load(loader);
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        BaseModelForConditionalGeneration::set_tokenizer(tokenizer);

    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        bool r = BaseModelForConditionalGeneration::load_more(config);
        return r;
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {

    }

    int ConditionalGeneration::tensor_num_of_layer(const Config &config, int layer_id) const
    {
        if (config.layer_is_swa[layer_id])
        {
            return 17;
        }
        else
        {
            return 17;
        }
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(GEMMA4,                gemma::v4, 1);
}