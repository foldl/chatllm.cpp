#include "gemma.h"

namespace chatllm::gemma::vit
{
    struct Config
    {
        ggml::type dtype;
        int max_patches;
        int global_head_dim;
        int head_dim;
        int hidden_size;
        int intermediate_size;
        int num_attention_heads;
        int num_hidden_layers;
        int num_key_value_heads;
        int patch_size;
        int pooling_kernel_size;
        int position_embedding_size;

        float rope_theta;
        float image_mean[3];
        float image_std[3];

        bool standardize;
        bool use_clipped_linears;
    };

    class VisionPatchEmbedder : public Block
    {
    public:
        VisionPatchEmbedder(InitContext *ctx, const Config &config):
            position_embedding_size(config.position_embedding_size),
            input_proj(ctx, 3 * config.patch_size * config.patch_size, config.hidden_size, false),
            position_embedding_table(ggml::new_tensor_3d(ctx, ggml::type_fallback(ctx->dtype, config.hidden_size), config.hidden_size, config.position_embedding_size, 2))
        {}

        void before_eval(ComputeContext *ctx) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *pixel_values, int width, int height);

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += input_proj.get_param_num(effective_only);
            r += ggml::nelements(position_embedding_table);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            input_proj.load(path + "input_proj.", loader);
            loader->read_tensor(path + "position_embedding_table", position_embedding_table);
        }
    public:
        const int       position_embedding_size;
        Linear          input_proj;
        ggml::tensor   *position_embedding_table;
    protected:
        int width = 0;
        int height = 0;
        ggml::tensor   *rt_pos = nullptr;
    };

    void VisionPatchEmbedder::before_eval(ComputeContext *ctx)
    {
        std::vector<int> pos(width * height * 2);
        int *p_x = pos.data();
        int *p_y = pos.data() + width * height;
        int cnt = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                p_x[cnt] = j;
                p_y[cnt] = i;
                cnt += 1;
            }
        }
        Backend::write_tensor_data(rt_pos, pos.data());
    }

    ggml::tensor *VisionPatchEmbedder::forward(ComputeContext *ctx, ggml::tensor *pixel_values, int width, int height)
    {
        this->width  = width;
        this->height = height;
        rt_pos = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, height * width, 2);

        // Gemma4 applies no normalization and instead scales in model code
        pixel_values = ggml::scale(ctx, pixel_values, 2, -1);
        auto hidden_states = input_proj.forward(ctx, pixel_values);


        auto pos_emb_xy = ggml::get_rows(ctx, position_embedding_table, rt_pos);
        auto pos_emb_x  = ggml::view_2d(ctx, pos_emb_xy, ggml::get_dim(pos_emb_xy, 0), ggml::get_dim(pos_emb_xy, 1), ggml::row_size(pos_emb_xy), 0);
        auto pos_emb_y  = ggml::view_2d(ctx, pos_emb_xy, ggml::get_dim(pos_emb_xy, 0), ggml::get_dim(pos_emb_xy, 1), ggml::row_size(pos_emb_xy), ggml::row_size(pos_emb_xy) * ggml::get_dim(pos_emb_xy, 1));
        auto pos_emb    = ggml::add(ctx, pos_emb_x, pos_emb_y);

        hidden_states   = ggml::add(ctx, hidden_states, pos_emb);

        return hidden_states;
    }

    class VisionMLP : public BaseMLP
    {
    public:
        VisionMLP(InitContext *ctx, int hidden_size, int intermediate_size):
            BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU)
        {
            enable_clipping();
        }
    };

    class TensorPosHelper
    {
    public:
        TensorPosHelper(int max_length)
            : max_length(max_length * 4)
        {
            v_pos.resize(this->max_length);
        }

        ggml::tensor *allocate_pos_tensor(InitContext *ctx)
        {
            ggml::tensor *r = ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length);
            ctx->get_allocator()->alloc(r);
            return r;
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen)
        {
            ggml::set_dim(pos, 0, length * 4);
            Backend::write_tensor_data(pos, v_pos.data(), 0, length * 2 * sizeof(v_pos[0]));
        }

        void prepare(int grid_h, int grid_w)
        {
            length = grid_h * grid_w;

            v_pos.clear();
            v_pos.resize(length * 2, 0);
            int *p_w = &v_pos[length * 0];
            int *p_h = &v_pos[length * 1];
            int index = 0;

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    p_w[index] = j;
                    p_h[index] = i;
                    index++;
                }
            }
        }
    public:
        const int max_length;
        int length;
        std::vector<int> v_pos;
    };

    class TensorPosHelper2D : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper2D(int max_length)
            : BaseTensorPosHelper(max_length * 4)
        {
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override
        {
            helper->prepare_pos_tensor(ctx, pos, n_past, qlen);
        }
    public:
        TensorPosHelper *helper;
    };

    class VisionAttention : public TensorPosHelperPrelude,  public QKNormedRoPEAttention<RMSNormInplace, BaseCachelessAttention>
    {
    public:
        typedef QKNormedRoPEAttention<RMSNormInplace, BaseCachelessAttention> Base;
        VisionAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
            TensorPosHelperPrelude(new TensorPosHelper2D(max_length)),
            Base(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
            head_size(head_dim)
        {
            TensorPosHelperPrelude::done();
            causal = false;
            CHATLLM_CHECK(rope_dim % 4 == 0);

            q_proj.enable_clipping();
            k_proj.enable_clipping();
            v_proj.enable_clipping();
            o_proj.enable_clipping();

            attn_scaling_factor = 1.0f;
        }

        void set_pos_helper(TensorPosHelper *helper)
        {
            TensorPosHelper2D *h = (TensorPosHelper2D *)pos_helper.get();
            h->helper = helper;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            const int hidden_size = o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            before_forward(ctx, n_past, qlen);

            ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpk = k_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpv = v_proj.forward(ctx, hidden_states);

            tmpv = ggml::reshape (ctx, tmpv, head_size,  num_kv_heads, ggml::get_dim(tmpv, 1), ggml::get_dim(tmpv, 2));
            tmpv = ggml::rms_norm(ctx, tmpv, 1e-6f);
            tmpv = ggml::reshape (ctx, tmpv, head_size * num_kv_heads, ggml::get_dim(tmpv, 2), ggml::get_dim(tmpv, 3));

            ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

            ggml::tensor *attn_output = o_proj.forward(ctx, scores);
            return attn_output;
        }
    protected:
        ggml::tensor *apply_2d_rope(ComputeContext *ctx, ggml::tensor *t, int hidden_size, int qlen) const
        {
            t = ggml::rope_2d_inplace(ctx, t, pos, freq_factors, rope_dim, n_original_ctx,
                                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            return t;
        }

        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            k = const_cast<VisionAttention *>(this)->k_layernorm.forward(ctx, k);
            k = apply_2d_rope(ctx, k, hidden_size, qlen);    // [qlen, heads, head_size]
            return k;
        }

        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            q = const_cast<VisionAttention *>(this)->q_layernorm.forward(ctx, q);
            q = apply_2d_rope(ctx, q, hidden_size, qlen);
            return q;
        }
    public:
        const int head_size;
        int grid_h = 0;
        int grid_w = 0;
    };

    typedef LMBlock4<RMSNorm, VisionAttention, RMSNorm, RMSNorm, VisionMLP, RMSNorm> VisionEncoderLayer;

    class VisionPooler : public Block
    {
    public:
        VisionPooler(InitContext *ctx, int hidden_size, int pooling_kernel_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int width, int height);
    public:
        const int   hidden_size;
        const float root_hidden_size;
        const int   pooling_kernel_size;
    };

    VisionPooler::VisionPooler(InitContext *ctx, int hidden_size, int pooling_kernel_size):
        hidden_size(hidden_size),
        root_hidden_size(sqrtf((float)hidden_size)),
        pooling_kernel_size(pooling_kernel_size)
    {
    }

    ggml::tensor *VisionPooler::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int width, int height)
    {
        CHATLLM_CHECK((width  % pooling_kernel_size) == 0);
        CHATLLM_CHECK((height % pooling_kernel_size) == 0);

        // re-arange patches
        hidden_states = ggml::reshape(ctx, hidden_states, hidden_size, width, height);
        hidden_states = ggml::permute(ctx, hidden_states, 2, 0, 1);
        auto r = ggml::avg_pool_2d(ctx, hidden_states, pooling_kernel_size, pooling_kernel_size);
        r = ggml::permute(ctx, r, 1, 2, 0);
        r = ggml::cont   (ctx, r);
        r = ggml::reshape(ctx, r, hidden_size, width / pooling_kernel_size * height / pooling_kernel_size);
        return r;
    }

    class MultimodalEmbedder : public Block
    {
    public:
        MultimodalEmbedder(InitContext *ctx, int multimodal_hidden_size, int text_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *mm_outputs) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int       multimodal_hidden_size;
        const int       text_hidden_size;
        Linear          embedding_projection;
        RMSNormNoScale  embedding_pre_projection_norm;
    };

    MultimodalEmbedder::MultimodalEmbedder(InitContext *ctx, int multimodal_hidden_size, int text_hidden_size):
        multimodal_hidden_size(multimodal_hidden_size),
        text_hidden_size(text_hidden_size),
        embedding_projection(ctx, multimodal_hidden_size, text_hidden_size, false),
        embedding_pre_projection_norm(ctx, multimodal_hidden_size)
    {}

    ggml::tensor *MultimodalEmbedder::forward(ComputeContext *ctx, ggml::tensor *mm_outputs)
    {
        auto output = embedding_pre_projection_norm.forward(ctx, mm_outputs);
             output = embedding_projection.forward(ctx, output);
        return output;
    }

    int64_t MultimodalEmbedder::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embedding_projection.get_param_num(effective_only);
        r += embedding_pre_projection_norm.get_param_num(effective_only);
        return r;
    }

    void MultimodalEmbedder::load(const std::string &path, TensorLoader *loader)
    {
        embedding_pre_projection_norm.load(path + "embedding_pre_projection_norm.", loader);
                 embedding_projection.load(path + "embedding_projection.", loader);
    }

    class VisionModel : public DynamicBlock
    {
    public:
        VisionModel(InitContext *ctx, const Config &config, const int text_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *pixel_values, int grid_w, int grid_h);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        VisionPatchEmbedder             patch_embedder;
        std::vector<VisionEncoderLayer> layers;
        VisionPooler                    pooler;
        MultimodalEmbedder              embed_vision;
        std::unique_ptr<TensorPosHelper> pos_helper;
    };

    VisionModel::VisionModel(InitContext *ctx, const Config &config, const int text_hidden_size):
        patch_embedder(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
        pooler(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.pooling_kernel_size),
        embed_vision(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, text_hidden_size)
    {
        pos_helper.reset(new TensorPosHelper(config.max_patches));

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            ctx->move_to_layer(i);
            layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, config.num_key_value_heads, config.head_dim, config.position_embedding_size);
            layers[i].attention.set_pos_helper(pos_helper.get());
            layers[i].attention.freq_base = config.rope_theta;
        }
    }

    ggml::tensor *VisionModel::forward(ComputeContext *ctx, ggml::tensor *pixel_values, int grid_w, int grid_h)
    {
        auto hidden_states = pixel_values;
        pos_helper->prepare(grid_h, grid_w);

        hidden_states = patch_embedder.forward(ctx, hidden_states, grid_w, grid_h);
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layers[i].attention.grid_h = grid_h;
            layers[i].attention.grid_w = grid_w;
            hidden_states = layers[i].forward(ctx, hidden_states, 0);
        }
        hidden_states = pooler.forward(ctx, hidden_states, grid_w, grid_h);
        hidden_states = embed_vision.forward(ctx, hidden_states);
        return hidden_states;
    }

    int64_t VisionModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (!is_loaded()) return r;
        r += patch_embedder.get_param_num(effective_only);
        r += embed_vision.get_param_num(effective_only);
        r += pooler.get_param_num(effective_only);
        r += layers[0].get_param_num(effective_only) * layers.size();
        return r;
    }

    void VisionModel::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "patch_embedder.input_proj.weight")) return;

        patch_embedder.load(path + "patch_embedder.", loader);
        embed_vision.load(path, loader);
        pooler.load(path + "pooler.", loader);
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layers[i].load(path + "blocks." + std::to_string(i) + '.', loader);
        }

        _loaded = true;
    }

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096);
        bool load(ModelLoader &loader);
        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config, const int max_projected_tokens);
        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);
        VisionModel *get_model(void) const { return vis_model.get(); }
        bool is_loaded() const;
    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        const int max_llm_tokens;
        std::unique_ptr<VisionModel> vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };

    VisualEmbeddingGeneration::VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "vis", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool VisualEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (vis_model.get())
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            vis_model->load("visual.", &loader);
            loader.pop_allocator_manager();
            return vis_model->is_loaded();
        }
        else
            return false;
    }

    bool VisualEmbeddingGeneration::is_loaded() const
    {
        if (vis_model.get() == nullptr) return false;
        return vis_model->is_loaded();
    }

    bool VisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config, const int max_projected_tokens)
    {
        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vis_config.dtype = dtype;

        vis_config.global_head_dim      = (int)vis_cfg["global_head_dim"].ToInt();
        vis_config.head_dim             = (int)vis_cfg["head_dim"].ToInt();
        vis_config.num_key_value_heads  = (int)vis_cfg["num_key_value_heads"].ToInt();
        vis_config.pooling_kernel_size  = (int)vis_cfg["pooling_kernel_size"].ToInt();
        vis_config.position_embedding_size  = (int)vis_cfg["position_embedding_size"].ToInt();
        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();

        vis_config.rope_theta           = (float)vis_cfg["rope_parameters"]["rope_theta"].ToFloat();
        vis_config.standardize          = vis_cfg["standardize"].ToBool();
        vis_config.use_clipped_linears  = vis_cfg["use_clipped_linears"].ToBool();

        vis_config.max_patches          = std::min(max_projected_tokens * vis_config.pooling_kernel_size * vis_config.pooling_kernel_size,
                                                   vis_config.position_embedding_size * vis_config.position_embedding_size);

        auto pp_cfg = config["processor_config.json"]["video_processor"];
        if (pp_cfg.IsObject())
        {
            auto image_mean = pp_cfg["image_mean"];
            auto image_std = pp_cfg["image_std"];
            CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
            CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

            vis_config.image_mean[0] = (float)image_mean[0].ToFloat();
            vis_config.image_mean[1] = (float)image_mean[1].ToFloat();
            vis_config.image_mean[2] = (float)image_mean[2].ToFloat();
            vis_config.image_std[0] = (float)image_std[0].ToFloat();
            vis_config.image_std[1] = (float)image_std[1].ToFloat();
            vis_config.image_std[2] = (float)image_std[2].ToFloat();
        }
        else
            return false;

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + 14 * vis_config.num_hidden_layers;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        vis_model.reset(new VisionModel(&_ctx, vis_config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

        return true;
    }

    void VisualEmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if (!is_loaded() || (tok->media_emb.size() < 1)) return;

        for (auto &image : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, image, buf);
        }
    }

    bool VisualEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
    {
        ggml::tensor *media_emb = nullptr;
        const auto make_graph = [this, &media_emb, &image](ComputeContext *ctx) -> ggml::tensor * {
            media_emb = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size * vis_config.patch_size * 3, image.grid_width * image.grid_height);
            auto r = vis_model->forward(ctx, media_emb, image.grid_width, image.grid_height);
            return r;
        };
        const auto write_input_data = [this, &media_emb, &image](ComputeContext *ctx) {
            Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            vis_model->patch_embedder.before_eval(ctx);
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }
}

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

    class ChatHistoryEncoder : public v1::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    protected:
        void append_image_piece(const ContentPiece &piece, std::vector<int> &ids) const;
        void append_audio_piece(const ContentPiece &piece, std::vector<int> &ids) const;
        void append_video_piece(const ContentPiece &piece, std::vector<int> &ids) const;
    public:
        const vit::Config *vis_config = nullptr;
        bool  vit_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public v1::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int boa_token_id = -1;
        int eoa_token_id = -1;
        int boi_token_id = -1;
        int eoi_token_id = -1;
    };

    Tokenizer::Tokenizer(const BaseConfig &config)
        : v1::Tokenizer(config, &_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        auto t = new tokenizer::BPEProcessor2();
        tp = t;
        size_t size = tp->Load(buffer, n_vocab);
        t->SetDecoderType(tokenizer::BPEProcessor2::DecoderType::Sequence);

        int id = tp->PieceToId("<pad>");
        if (id >= 0) pad_token_id = id;
        start_of_turn_token_id  = tp->PieceToId("<|turn>");
        end_of_turn_token_id    = tp->PieceToId("<turn|>");
        nl_token_id             = tp->PieceToId("\n");

        boa_token_id            = tp->PieceToId("<|audio>");
        eoa_token_id            = tp->PieceToId("<audio|>");
        boi_token_id            = tp->PieceToId("<|image>");
        eoi_token_id            = tp->PieceToId("<image|>");

        terminate_ids.insert(end_of_turn_token_id);

        return size;
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        const int b = media_type == "image" ? boi_token_id : boa_token_id;
        const int e = media_type == "image" ? eoi_token_id : eoa_token_id;
        ids.push_back(b);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(e);
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
            if (nullptr == shared_attn)
            {
                r += BaseAttn::k_proj.get_param_num(effective_only);
                r += use_k_as_v ? 0 : BaseAttn::v_proj.get_param_num(effective_only);
            }
            r += BaseAttn::o_proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            KVCacheAttention::load(path, loader);
            BaseAttn::q_proj.load(path + "q_proj.", loader);
            BaseAttn::o_proj.load(path + "o_proj.", loader);
            if (nullptr == shared_attn)
            {
                BaseAttn::k_proj.load(path + "k_proj.", loader);
                if (!use_k_as_v)
                    BaseAttn::v_proj.load(path + "v_proj.", loader);
            }
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            const int hidden_size = BaseAttn::o_proj.in_features();
            const int qlen = (int)hidden_states->ne[1];

            BaseAttn::before_forward(ctx, n_past, qlen);

            ggml::tensor *tmpq = BaseAttn::q_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpk = shared_attn ? nullptr : BaseAttn::k_proj.forward(ctx, hidden_states);
            ggml::tensor *tmpv = nullptr;

            if (nullptr == shared_attn)
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
        typedef RoPESelfAttention<Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>> BaseBase;
        typedef QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<SlidingWindowAttentionImpl<sliding_window_len>>> Base;
        Gemma4SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            Base::attn_scaling_factor = 1.0f;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += Base::get_param_num(effective_only);
            r -= Base::shared_attn != nullptr ? Base::k_layernorm.get_param_num(effective_only) : 0;
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            BaseBase::load(path, loader);
            Base::q_layernorm.load(path + "q_norm.", loader);
            if (nullptr == BaseBase::shared_attn)
                Base::k_layernorm.load(path + "k_norm.", loader);
        }
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
        typedef RoPESelfAttention<Gemma4Attention<BaseAttention>> BaseBase;
        typedef QKNormedRoPEAttention<RMSNormInplace, Gemma4Attention<BaseAttention>> Base;
        Gemma4FullSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            Base::attn_scaling_factor = 1.0f;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += Base::get_param_num(effective_only);
            r -= Base::shared_attn != nullptr ? Base::k_layernorm.get_param_num(effective_only) : 0;
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            BaseBase::load(path, loader);
            Base::q_layernorm.load(path + "q_norm.", loader);
            if (nullptr == BaseBase::shared_attn)
                Base::k_layernorm.load(path + "k_norm.", loader);
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
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *per_layer_input_ids, int n_past) override;
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

    ggml::tensor *ModelClass::forward(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *per_layer_input_ids, int n_past)
    {
        before_forward(ctx, input_ids, n_past);

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        ggml::tensor *hidden_states = word_embeddings->forward(ctx, input_ids);
        if (per_layer_emb.get())
        {
            per_layer_emb->forward(ctx, per_layer_input_ids, hidden_states);
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

    class Prelude
    {
    public:
        Prelude(int n = 2048):
            pad_arg(new BlockParams::PadEmbedding(n, n))
        {
        }

        void done(void)
        {
            pad_arg.reset(nullptr);
        }
    protected:
        std::unique_ptr<BlockParams::PadEmbedding> pad_arg;
    };

    class ConditionalGeneration : public Prelude, public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA4);
        void load(ModelLoader &loader) override;

        void set_tokenizer(BaseTokenizer *tokenizer) override;
        bool load_more(const json::JSON &config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void before_generate(const GenerationConfig &gen_config) override;

        int64_t get_param_num(bool effective_only) const override;

    public:
        int tensor_num_of_layer(const Config &config, int layer_id) const;
        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output, const int batch_size,
                            std::function<ggml::tensor *(ComputeContext *, ggml::tensor *)> func_epilog) override;
    public:
        const Config config;
        const int max_projected_tokens;
        v2::TanhScaling logits_pp;
        vit::VisualEmbeddingGeneration visual;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        Prelude(),
        BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
        config(config),
        max_projected_tokens(BlockParams::get_padded_embedding_num()),
        logits_pp(1.0f / config.final_logit_soft_capping / sqrtf((float)config.hidden_size), config.final_logit_soft_capping),
        visual(runtime_config, pad_arg->get())
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

        Prelude::done();
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".pre_attention_layernorm.",   ".input_layernorm."},
            {".pre_mlp_layernorm.",         ".pre_feedforward_layernorm."},
            {".post_mlp_layernorm.",        ".post_feedforward_layernorm."},
        });

        BaseModelForConditionalGeneration::load(loader);

        _chat_encoder.vit_loaded  = visual.load(loader);
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        BaseModelForConditionalGeneration::set_tokenizer(tokenizer);
        if (visual.is_loaded())
        {
            _chat_encoder.vis_config = &visual.vis_config;
        }
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        BaseModelForConditionalGeneration::load_more(config);
        bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config, max_projected_tokens);
        return r;
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return;
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = BaseModelForConditionalGeneration::get_param_num(effective_only);
        if (visual.is_loaded())
            r += visual.get_model()->get_param_num(effective_only);
        return r;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;

        CHATLLM_CHECK(tokenizer->get_image_total_emb_vectors() < max_projected_tokens) << "too many projected tokens.";

        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
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

    bool ConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output, const int batch_size,
                            std::function<ggml::tensor *(ComputeContext *, ggml::tensor *)> func_epilog)
    {
        if (!initial_run)
        {
            initial_run = true;
            n_past = gen_config.max_length / transformer->get_reserved_batch_size() - ids_count;
            if (n_past < 0) n_past = 0;
            if (!before_initial_run(ids_count, gen_config, n_past))
                return false;
            n_past = 0;
        }

        before_run_model(input_ids, ids_count, gen_config, past);

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        std::vector<int> ids_for_layer;
        ids_for_layer.resize(ids_count * batch_size);
        for (int i = 0; i < ids_count * batch_size; i++)
        {
            ids_for_layer[i] = input_ids[i] < config.vocab_size ? input_ids[i] : config.pad_token_id;
        }

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);
        ggml::tensor *layer_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);

        ggml::tensor *r = get_typed_transformer<ModelClass>()->forward(&ctx, input_ids_tensor, layer_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (func_epilog)
        {
            r = func_epilog(&ctx, r);
        }
        else
        {
            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);
        }

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK((r->type == GGML_TYPE_F32) || (r->type == GGML_TYPE_I32)) << "output type must be float/int32: " << r->type;

        output.resize(ggml::nbytes(r) / sizeof(output[0]));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, input_ids);
        Backend::write_tensor_data(layer_ids_tensor, ids_for_layer.data());

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        transformer->before_eval(&ctx);
        ctx.compute();

        Backend::read_tensor_data(r, output.data());

        ctx.reset();

        return true;
    }

    void ChatHistoryEncoder::append_image_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

        int w, h;
        std::vector<uint8_t> pixels;
        const int patch_size = vis_config->patch_size;

        vision::MaxPatchNum     param1(vis_config->max_patches);
        vision::MergeKernel     param2(vis_config->pooling_kernel_size, vis_config->pooling_kernel_size);

        vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
        if ((w <= 0) || (h <= 0)) return;

        std::vector<float> scaled;
        vision::image_rescale(pixels, scaled);

        tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

        auto &image = tok->media_emb.back();

        vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_PixelsLeftRightDown_ChannelsRGB);

        const int merge_length = vis_config->pooling_kernel_size * vis_config->pooling_kernel_size;
        image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

        const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
        tok->inject_media("image", ids, id_start, image.emb_vec_number);
    }

    void ChatHistoryEncoder::append_audio_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {

    }

    void ChatHistoryEncoder::append_video_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {

    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        oss << "user\n";

        ids.push_back(tok->start_of_turn_token_id);

        auto add_and_flush = [&oss, &ids, tok](const std::string &s) {
            oss << s;
            std::string x = oss.str();
            tok->encode(x, ids, false, false);
            oss.str("");
        };

        for (auto &piece : user.pieces)
        {
            switch (piece.type)
            {
            case ContentPiece::Type::Text:
                oss << piece.content;
                break;
            case ContentPiece::Type::Image:
                add_and_flush("\n\n");
                append_image_piece(piece, ids);
                oss << "\n\n";
                break;
            case ContentPiece::Type::Audio:
                add_and_flush("\n\n");
                append_audio_piece(piece, ids);
                oss << "\n\n";
                break;
            case ContentPiece::Type::Video:
                add_and_flush("\n\n");
                append_video_piece(piece, ids);
                oss << "\n\n";
                break;
            default:
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
                break;
            }
        }

        tok->encode(oss.str(), ids, false, true);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(GEMMA4,                gemma::v4, 1);
}