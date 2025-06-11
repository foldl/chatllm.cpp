namespace vit
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int init_pos_emb_height;
        int init_pos_emb_width;
        int merge_kernel_size[2];

        int   in_token_limit;
        float image_mean[3];
        float image_std[3];
        bool  pad_input;
    };

    class Learnable2DInterpPosEmb : public Block
    {
    public:
        Learnable2DInterpPosEmb() {}
        Learnable2DInterpPosEmb(InitContext *ctx, int height, int width, int dim)
            : pos_emb(ggml::new_tensor_3d(ctx, ggml::type::GGML_TYPE_F32, dim, width, height))
        {
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml::nelements(pos_emb);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
        {
            CHATLLM_CHECK(ggml::get_dim(input, 3) == 1);

            auto output = ggml::add(ctx, input, pos_emb);

            return output;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            loader->read_tensor(path + "weight", pos_emb);
        }

    public:
        ggml::tensor *pos_emb;
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int pos_emb_height, int pos_emb_width, int in_dim = 3)
            : proj(ctx, in_dim, out_dim, patch_size, patch_size),
              pos_emb(ctx, pos_emb_height, pos_emb_width, out_dim)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
        {
            ggml::tensor *x = proj.forward(ctx, input);
            x = ggml::reshape_3d(ctx, x, ggml::get_dim(x, 2), grid_h, grid_w);
            x = pos_emb.forward(ctx, x, grid_h, grid_w);
            x = ggml::reshape_2d(ctx, x, ggml::get_dim(x, 0), grid_h * grid_w);
            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += proj.get_param_num(effective_only);
            r += pos_emb.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            proj.load   (path + "proj.", loader);
            pos_emb.load(path + "pos_emb.", loader);
        }

    public:
        Conv2D                  proj;
        Learnable2DInterpPosEmb pos_emb;
    };

    typedef TheBiasedGELUMLP MLP2;

    class ViTSelfAttention : public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RoPESelfAttention<BaseCachelessAttention>(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true),
              pos_h(ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length))
        {
            ctx->get_allocator()->alloc(pos_h);
            causal = false;

            CHATLLM_CHECK(rope_dim % 4 == 0);
        }

    protected:
        void before_forward(ComputeContext *ctx, const int n_past, const int qlen) override
        {
            const int len = grid_h * grid_w;

            CHATLLM_CHECK(len <= max_length);

            ggml::set_dim(pos,   0, len);
            ggml::set_dim(pos_h, 0, len);


            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                    v_pos[i * grid_w + j] = j;
            }

            Backend::write_tensor_data(pos, v_pos.data(), 0, len * sizeof(v_pos[0]));

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                    v_pos[i * grid_w + j] = i;
            }

            Backend::write_tensor_data(pos_h, v_pos.data(), 0, len * sizeof(v_pos[0]));
        }

        ggml::tensor *apply_2d_rope(ComputeContext *ctx, ggml::tensor *hidden, int hidden_size, ggml::tensor *pos_w, ggml::tensor *pos_h) const
        {
            // ggml shape of hidden: [head_size, heads, qlen]
            CHATLLM_CHECK(ggml::get_dim(hidden, 3) == 1);

            ggml::tensor *part = ggml::view_4d(ctx, hidden, 2, ggml::get_dim(hidden, 0) / 4, ggml::get_dim(hidden, 1), ggml::get_dim(hidden, 2),
                                               4 * ggml::element_size(hidden), ggml::row_size(hidden), ggml::row_size(hidden) * ggml::get_dim(hidden, 1), 0);
            ggml::tensor *copy = ggml::cont(ctx, part);
            copy = ggml::reshape_3d(ctx, copy, ggml::get_dim(copy, 0) * ggml::get_dim(copy, 1), ggml::get_dim(copy, 2), ggml::get_dim(copy, 3));
            copy = ggml::rope_ext_inplace(ctx, copy, pos_w, freq_factors, rope_dim / 2, RoPEMode::Interleaved, n_original_ctx,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            copy = ggml::cpy(ctx, copy, part);
            ggml::build_forward_expand(ctx, copy);

            part = ggml::view_4d(ctx, hidden, 2, ggml::get_dim(hidden, 0) / 4, ggml::get_dim(hidden, 1), ggml::get_dim(hidden, 2),
                                4 * ggml::element_size(hidden), ggml::row_size(hidden), ggml::row_size(hidden) * ggml::get_dim(hidden, 1), 2 * ggml::element_size(hidden));
            copy = ggml::cont(ctx, part);
            copy = ggml::reshape_3d(ctx, copy, ggml::get_dim(copy, 0) * ggml::get_dim(copy, 1), ggml::get_dim(copy, 2), ggml::get_dim(copy, 3));
            copy = ggml::rope_ext_inplace(ctx, copy, pos_h, freq_factors, rope_dim / 2, RoPEMode::Interleaved, n_original_ctx,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            copy = ggml::cpy(ctx, copy, part);
            ggml::build_forward_expand(ctx, copy);

            hidden = ggml::reshape_4d(ctx, hidden, ggml::get_dim(hidden, 0), ggml::get_dim(hidden, 1), ggml::get_dim(hidden, 2), ggml::get_dim(hidden, 3));

            return hidden;
        }

        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            k = apply_2d_rope(ctx, k, hidden_size, pos, pos_h);
            return k;
        }

        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            q = apply_2d_rope(ctx, q, hidden_size, pos, pos_h);
            return q;
        }
    public:
        ggml::tensor *pos_h;
        int grid_w = 0;
        int grid_h = 0;
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size)
            :
            hidden_size(config.hidden_size * config.merge_kernel_size[0] * config.merge_kernel_size[1]),
            pre_norm(ctx, config.hidden_size),
            linear_1(ctx, hidden_size, hidden_size),
            linear_2(ctx, hidden_size, lm_hidden_size)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features) override
        {
            auto output = pre_norm.forward(ctx, image_features);
            output = ggml::reshape_2d(ctx, output,
                hidden_size, ggml::get_dim(output, 1) / (hidden_size / ggml::get_dim(output, 0)) * ggml::get_dim(output, 2) * ggml::get_dim(output, 3));
            output = linear_1.forward(ctx, output);
            output = ggml::act(ctx, ActFunc::GELU, output);
            output = linear_2.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += pre_norm.get_param_num(effective_only);
            r += linear_1.get_param_num(effective_only);
            r += linear_2.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            pre_norm.load(path + "pre_norm.", loader);
            linear_1.load(path + "linear_1.", loader);
            linear_2.load(path + "linear_2.", loader);
        }

    public:
        const int hidden_size;
        LayerNorm pre_norm;
        Linear    linear_1;
        Linear    linear_2;
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP2> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size)
            : embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
                        config.hidden_size, config.patch_size, config.init_pos_emb_height, config.init_pos_emb_width),
            post_layernorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, lm_hidden_size),
            loaded(false)
        {
            const int max_length = config.in_token_limit * config.merge_kernel_size[0] * config.merge_kernel_size[1];
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
                layer->set_id(layer_id);
                layers.emplace_back(layer);
            }
            memcpy(merge_param.merge_kernel_size, config.merge_kernel_size, sizeof(merge_param.merge_kernel_size));
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += embeddings.get_param_num(effective_only);
            r += post_layernorm.get_param_num(effective_only);
            r += multi_modal_projector.get_param_num(effective_only);
            for (size_t i = 0; i < layers.size(); i++)
                r += layers[i]->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            if (!loader->has_tensor(path + "patch_embed.pos_emb.weight")) return;

            embeddings.load(path + "patch_embed.", loader);
            post_layernorm.load(path + "final_layernorm.", loader);
            multi_modal_projector.load("multi_modal_projector.", loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "encoder.blocks." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            loaded = true;
        }

        ggml::tensor *merge_patch(ComputeContext *ctx, ggml::tensor *x, int grid_h, int grid_w)
        {
            merge_param.grid_h = grid_h;
            merge_param.grid_w = grid_w;

            auto reshaped_seq = ggml::merge_patch(ctx, x, &merge_param);
            return reshaped_seq;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
        {
            auto output = embeddings.forward(ctx, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->attention.grid_h = grid_h;
                layers[i]->attention.grid_w = grid_w;
                output = layers[i]->forward(ctx, output, 0);
            }
            output = post_layernorm.forward(ctx, output);
            output = merge_patch(ctx, output, grid_h, grid_w);
            output = multi_modal_projector.forward(ctx, output);
            return output;
        }

        bool is_loaded(void) const
        {
            return loaded;
        }
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        LayerNorm  post_layernorm;
        MultiModalProjector multi_modal_projector;
    protected:
        bool loaded;
        ggml::merge_patch_param merge_param;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096)
            :
            GRAPH_SIZE(GRAPH_SIZE), _ctx(&backend_context),
            n_threads(runtime_config.n_threads)
        {
            _ctx.cache_dtype = runtime_config.cache_type;
            model_gpu_layers = BackendContext::get_ngl_of_model(runtime_config.model_gpu_layers, "vis");
        }

        bool load(ModelLoader &loader)
        {
            if (vis_model.get())
            {
                loader.push_allocator_manager(&backend_context.layer_allocators);
                vis_model->load("vision_model.", &loader);
                loader.pop_allocator_manager();
                return vis_model->is_loaded();
            }
            else
                return false;
        }

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
        {
            const auto vis_cfg = config["config.json"]["vision_config"];
            if (!vis_cfg.IsObject()) return false;

            vis_config.dtype = dtype;

            vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
            vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
            vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
            vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
            vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
            vis_config.init_pos_emb_height  = (int)vis_cfg["init_pos_emb_height"].ToInt();
            vis_config.init_pos_emb_width   = (int)vis_cfg["init_pos_emb_width"].ToInt();

            auto size = vis_cfg["merge_kernel_size"];
            CHATLLM_CHECK(size.length() == 2) << "invalid merge_kernel_size";
            vis_config.merge_kernel_size[0]   = (int)size[0].ToInt();
            vis_config.merge_kernel_size[1]   = (int)size[1].ToInt();

            vis_config.in_token_limit       = 4096;
            vis_config.pad_input            = true;
            for (int i = 0; i < 3; i++)
            {
                vis_config.image_mean[i]    = 0.5f;
                vis_config.image_std[i]     = 0.5f;
            }

            auto pp_cfg = config["preprocessor_config.json"];
            if (pp_cfg.IsObject())
            {
                vis_config.in_token_limit   = (int  )pp_cfg["in_token_limit"].ToInt();
                vis_config.pad_input        =        pp_cfg["pad_input"].ToBool();

                auto image_mean = pp_cfg["image_mean"];
                auto image_std  = pp_cfg["image_std"];
                CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
                CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

                vis_config.in_token_limit   = (int  )pp_cfg["in_token_limit"].ToInt();
                vis_config.pad_input        =        pp_cfg["pad_input"].ToBool();
                vis_config.image_mean[0]    = (float)image_mean[0].ToFloat();
                vis_config.image_mean[1]    = (float)image_mean[1].ToFloat();
                vis_config.image_mean[2]    = (float)image_mean[2].ToFloat();
                vis_config.image_std[0]     = (float)image_std[0].ToFloat();
                vis_config.image_std[1]     = (float)image_std[1].ToFloat();
                vis_config.image_std[2]     = (float)image_std[2].ToFloat();
            }

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 11 + vis_config.num_hidden_layers * 18;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            _ctx.dtype = dtype;
            backend_context.init(model_gpu_layers, vis_config.num_hidden_layers, GRAPH_SIZE, n_threads);

            vis_model.reset(new VisionTransformer(&_ctx, vis_config, lm_hidden_size));

            CHATLLM_CHECK(_ctx.get_used_mem() == _ctx.get_mem_size()) << "tensor number mismatch: " << _ctx.get_used_mem() / tensor_ovhd << " vs "  << _ctx.get_mem_size() / tensor_ovhd;

            return true;
        }

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
        {
            if ((vis_model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
            if (!vis_model->is_loaded()) return;

            for (auto &image : tok->media_emb)
            {
                run_model(gen_config, tok, dtype, image, buf);
            }
        }

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
        {
            ForwardContext ctx(&backend_context);
            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            const int total_patches = tok->get_image_total_emb_vectors();

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
            ggml::tensor *media_emb = ggml::new_tensor_4d(&ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size, vis_config.patch_size, 3, image.grid_width * image.grid_height);

            dbg_ctx = &ctx;

            auto r = vis_model->forward(&ctx, media_emb, image.grid_height, image.grid_width);

            if (ggml::type_of(r) != dtype)
            {
                ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
                ggml::tensor *t = ggml::new_tensor_4d(&ctx, dtype, ggml::get_dim(r, 0), ggml::get_dim(r, 1), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
                r = ggml::cpy(&ctx, r, t);
            }

            ggml::build_forward_expand(&ctx, r);

            CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";

            if (gen_config.dump_dot.size() > 0)
            {
                backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
                exit(-1);
            }

            Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));

            ctx.compute();

            size_t offset = buf.size();
            buf.resize(offset + ggml::nbytes(r));
            Backend::read_tensor_data(r, buf.data() + offset);
            ctx.reset();

            return true;
        }

    protected:
        std::unique_ptr<VisionTransformer> vis_model;
        BackendContext backend_context;
        const size_t GRAPH_SIZE;
        InitContext _ctx; // weight context
        std::string model_gpu_layers;
        const int n_threads;
    public:
        Config vis_config;
    };
}

namespace vl
{
    struct Config : public deepseek::v3_light::Config
    {
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;

    public:
        const vit::Config *vis_config = nullptr;
        bool vit_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder)
            : BaseTokenizer(config, chat_encoder, nullptr, nullptr)
        {
            sys_prompt = "You are a helpful assistant";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    "[\\p{Han}]+",
                    // FIXME: support &&
                    //"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    //"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            return size;
        }

        bool load_config(const json::JSON &config) override
        {
            #define check_token(tok) {std::string("<|" #tok "|>"), &(tok ## _token_id)}

            load_added_tokens(config, {
                check_token(im_end),
                check_token(im_user),
                check_token(im_assistant),
                check_token(im_system),
                check_token(im_middle),
                check_token(media_start),
                check_token(media_content),
                check_token(media_end),
                check_token(media_pad),
            });
            #undef check_token

            if (im_end_token_id >= 0)
                terminate_ids.insert(im_end_token_id);

            return true;
        }


        void inject_media(const std::string &media_type, std::vector<int> &ids, const std::vector<int> &ids_to_inject)
        {
            ids.push_back(media_start_token_id);
            encode(media_type, ids);
            ids.push_back(media_content_token_id);
            ids.insert(ids.end(), ids_to_inject.begin(), ids_to_inject.end());
            ids.push_back(media_end_token_id);
        }

        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
        {
            ids.push_back(media_start_token_id);
            encode(media_type, ids);
            ids.push_back(media_content_token_id);
            for (int i = 0; i < ids_to_inject_count; i++)
                ids.push_back(ids_to_inject_start + i);
            ids.push_back(media_end_token_id);
        }

    public:
        int im_end_token_id;
        int im_user_token_id;
        int im_assistant_token_id;
        int im_system_token_id;
        int im_middle_token_id;
        int media_start_token_id;
        int media_content_token_id;
        int media_end_token_id;
        int media_pad_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->im_end_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->im_system_token_id);
        tok->encode("system", ids);
        ids.push_back(tok->im_middle_token_id);
        tok->encode(tok->get_system_prompt(), ids);
        ids.push_back(tok->im_end_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_user_opening(round_idx, ids);
        tok->encode(user, ids);
        ids.push_back(tok->im_end_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->im_assistant_token_id);
        tok->encode("assistant", ids);
        ids.push_back(tok->im_middle_token_id);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->im_user_token_id);
        tok->encode("user", ids);
        ids.push_back(tok->im_middle_token_id);
    }

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding() : pad_arg(new BlockParams::PadEmbedding(4096, 4096)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public deepseek::v3_light::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_KIMI_VL)
            : ExtendEmbedding(),
              deepseek::v3_light::ConditionalGeneration(config, runtime_config, type),
              visual(runtime_config)
        {
            delete pad_arg;
            pad_arg = nullptr;
        }

        bool load_more(const json::JSON &config) override
        {
            BaseModelForConditionalGeneration::load_more(config);
            bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config);
            if (r)
            {
                _chat_encoder.vis_config = &visual.vis_config;
            }
            return r;
        }

        void load(ModelLoader &loader) override
        {
            deepseek::v3_light::ConditionalGeneration::load(loader);

            loader.add_tensor_name_translations({
                {".self_attn.",                         ".attn."},
                {".input_layernorm.",                   ".norm0."},
                {".post_attention_layernorm.",          ".norm1."},
            });

            _chat_encoder.vit_loaded = visual.load(loader);
        }

        void before_generate(const GenerationConfig &gen_config) override
        {
            std::vector<uint8_t> buf;
            auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
            visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
            if (buf.size() < 1) return;

            size_t offset = emb->get_base_nbytes();
            Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
        }
    public:
        vit::VisualEmbeddingGeneration visual;
    };

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_user_opening(round_idx, ids);

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                int w, h;
                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->patch_size;

                vision::MergeKernel     param1(vis_config->merge_kernel_size);
                vision::MaxPatchNum     param2(vis_config->in_token_limit);
                vision::MaxGridHeight   param3(512);
                vision::MaxGridWidth    param4(512);

                // TODO: cubic interpolation not ready yet. image size fixed.
                vision::Resize          resize(896, 896);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_ChannelsRGB_PixelsLeftRightDown);

                const int merge_length = vis_config->merge_kernel_size[0] * vis_config->merge_kernel_size[1];
                image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_media("image", ids, id_start, image.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
        ids.push_back(tok->im_end_token_id);
    }
}