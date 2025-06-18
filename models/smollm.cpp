namespace lm
{
    typedef llama::v3::Config Config;

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

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);
            return size;
        }
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
        {
            ids.push_back(tok->bos_token_id);
            tok->encode("system\n", ids);
            tok->encode(tok->get_system_prompt(), ids);
            ids.push_back(tok->eos_token_id);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        ids.push_back(tok->bos_token_id);
        tok->encode("user\n", ids);
        tok->encode(user, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        tok->encode("assistant\n", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        tok->encode("user\n", ids);
    }

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_SMOLLM)
            : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length)
            : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length, 12, true)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
            }
        }
    };
}

namespace vit
{
    struct VideoConfig
    {
        float fps;
        int max_frames;
        int longest_edge;
    };

    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;

        int scale_factor;
        int image_seq_len;
        int image_max_size;
        int image_size_per_split;

        float image_mean[3];
        float image_std[3];

        VideoConfig video;
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int image_max_size, int in_dim = 3)
            : num_patches_per_side(image_max_size / patch_size),
              num_patches(num_patches_per_side * num_patches_per_side),
              patch_embedding(ctx, in_dim, out_dim, patch_size, patch_size),
              position_embedding(TypeChanger(ctx, ggml::type::GGML_TYPE_F32), num_patches, out_dim)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int image_width, int image_height)
        {
            ggml::tensor *x = patch_embedding.forward(ctx, input);
            x = ggml::reshape_3d(ctx, x, ggml::get_dim(x, 0) * ggml::get_dim(x, 1), ggml::get_dim(x, 2), ggml::get_dim(x, 3));
            x = ggml::transpose(ctx, x);
            x = ggml::cont(ctx, x);
            x = ggml::add(ctx, x, position_embedding.weight);
            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r +=    patch_embedding.get_param_num(effective_only);
            r += position_embedding.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
               patch_embedding.load(path + "patch_embedding.", loader);
            position_embedding.load(path + "position_embedding.", loader);
        }

    public:
        const int  num_patches_per_side;
        const int  num_patches;
        Conv2D     patch_embedding;
        Embedding  position_embedding;
    };

    typedef TheBiasedGELUMLP MLP2;

    class ViTSelfAttention : public BaseCachelessAttention
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true)
        {
            causal = false;
        }
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size)
            :
            hidden_size(config.hidden_size), scale_factor(config.scale_factor),
            patch_size(config.patch_size),
            proj(ctx, hidden_size * scale_factor * scale_factor, lm_hidden_size, false)
        {
            merge_param.merge_kernel_size[0] = scale_factor;
            merge_param.merge_kernel_size[1] = scale_factor;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int image_width, int image_height)
        {
            auto output = pixel_shuffle(ctx, image_features, image_width, image_height);
            output = proj.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            proj.load(path + "proj.", loader);
        }
    protected:
        ggml::tensor *pixel_shuffle(ComputeContext *ctx, ggml::tensor *image_features, int image_width, int image_height)
        {
            merge_param.grid_w = image_width  / patch_size;
            merge_param.grid_h = image_height / patch_size;

            ggml::tensor *r = ggml::merge_patch(ctx, image_features, &merge_param);
            r = ggml::reshape_4d(ctx, r, ggml::get_dim(r, 0) * scale_factor * scale_factor, ggml::get_dim(r, 1) / (scale_factor * scale_factor), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
            return r;
        }
    public:
        const int hidden_size;
        const int scale_factor;
        const int patch_size;
        ggml::merge_patch_param merge_param;
        Linear    proj;
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP2> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size)
            : embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
                        config.hidden_size, config.patch_size, config.image_size_per_split),
            post_layernorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, lm_hidden_size),
            loaded(false)
        {
            const int max_length = config.image_seq_len * config.video.max_frames;
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
                layer->set_id(layer_id);
                layers.emplace_back(layer);
            }
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
            if (!loader->has_tensor(path + "embeddings.patch_embedding.weight")) return;

            embeddings.load(path + "embeddings.", loader);
            post_layernorm.load(path + "final_layernorm.", loader);
            multi_modal_projector.load("multi_modal_projector.", loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "encoder.layers." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            loaded = true;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int image_width, int image_height)
        {
            auto output = embeddings.forward(ctx, input, image_width, image_height);

            for (size_t i = 0; i < layers.size(); i++)
            {
                output = layers[i]->forward(ctx, output, 0);
            }
            output = post_layernorm.forward(ctx, output);
            output = multi_modal_projector.forward(ctx, output, image_width, image_height);
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
            auto pp_cfg = config["preprocessor_config.json"];
            if (!vis_cfg.IsObject() || !pp_cfg.IsObject()) return false;

            vis_config.dtype = dtype;

            vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
            vis_config.num_attention_heads  = vis_cfg["num_attention_heads"].IsNull() ? 16 : (int)vis_cfg["num_attention_heads"].ToInt();
            vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
            vis_config.hidden_size          = vis_cfg["hidden_size"].IsNull() ? 1152 : (int)vis_cfg["hidden_size"].ToInt();
            vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();

            vis_config.scale_factor         = (int)config["config.json"]["scale_factor"].ToInt();
            vis_config.image_max_size       = (int)vis_cfg["size"]["longest_edge"].ToInt();
            vis_config.image_size_per_split = (int)vis_cfg["max_image_size"]["longest_edge"].ToInt();

            vis_config.image_seq_len        = vis_config.image_size_per_split / vis_config.patch_size;
            vis_config.image_seq_len       *= vis_config.image_seq_len;
            vis_config.image_seq_len       /= vis_config.scale_factor * vis_config.scale_factor;

            vis_config.video.fps            = (float)pp_cfg["video_sampling"]["fps"].ToFloat();
            vis_config.video.max_frames     = (int  )pp_cfg["video_sampling"]["max_frames"].ToInt();
            vis_config.video.longest_edge   = (int  )pp_cfg["video_sampling"]["video_size"]["longest_edge"].ToInt();

            auto image_mean = pp_cfg["image_mean"];
            auto image_std  = pp_cfg["image_std"];
            CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
            CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

            vis_config.image_mean[0]    = (float)image_mean[0].ToFloat();
            vis_config.image_mean[1]    = (float)image_mean[1].ToFloat();
            vis_config.image_mean[2]    = (float)image_mean[2].ToFloat();
            vis_config.image_std[0]     = (float)image_std[0].ToFloat();
            vis_config.image_std[1]     = (float)image_std[1].ToFloat();
            vis_config.image_std[2]     = (float)image_std[2].ToFloat();

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 6 + vis_config.num_hidden_layers * 17;
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

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
            ggml::tensor *media_emb = ggml::new_tensor_3d(&ctx, ggml::type::GGML_TYPE_F32, image.width, image.height, 3);

            dbg_ctx = &ctx;

            auto r = vis_model->forward(&ctx, media_emb, image.width, image.height);

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

namespace vlm
{
    typedef lm::Config Config;

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

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "";
            do_split = false;
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void inject_image(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count, int col_id, int row_id);
        void inject_global_image(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count);

    public:
        int end_of_utterance_token_id;
        int fake_token_around_image_token_id;
        int global_image_token_token_id;
        int nl_token_id;
        bool do_split;

        std::vector<std::vector<int>> row_col_token_ids;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        bos_token_id = tp->PieceToId("<|im_start|>");
        end_of_utterance_token_id           = tp->PieceToId("<end_of_utterance>");
        fake_token_around_image_token_id    = tp->PieceToId("<fake_token_around_image>");
        global_image_token_token_id         = tp->PieceToId("<global-img>");

        std::vector<int> toks;
        tp->Encode("\n", &toks);
        CHATLLM_CHECK(toks.size() == 1);
        nl_token_id = toks[0];

        eos_token_id = end_of_utterance_token_id;

        terminate_ids.insert(tp->PieceToId("<|im_end|>"));
        terminate_ids.insert(tp->PieceToId("<|endoftext|>"));
        terminate_ids.insert(tp->PieceToId("<tool_call>"));
        terminate_ids.insert(bos_token_id);
        terminate_ids.insert(end_of_utterance_token_id);

        for (int r = 0; true; r++)
        {
            std::vector<int> col;
            for (int c = 0; true; c++)
            {
                char s[100];
                sprintf(s, "<row_%d_col_%d>", r + 1, c + 1);
                int id = tp->PieceToId(s);
                if (id < 0) break;
                col.push_back(id);
            }
            if (col.size() < 1) break;
            row_col_token_ids.push_back(col);
        }

        return size;
    }

    void Tokenizer::inject_image(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count, int col_id, int row_id)
    {
        ids.push_back(fake_token_around_image_token_id);
        ids.push_back(row_col_token_ids[row_id][col_id]);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
    }

    void Tokenizer::inject_global_image(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(fake_token_around_image_token_id);
        ids.push_back(global_image_token_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(fake_token_around_image_token_id);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->end_of_utterance_token_id);
        ids.push_back(tok->nl_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode("System:", ids);
            tok->encode(" " + tok->get_system_prompt(), ids);
            ids.push_back(tok->end_of_utterance_token_id);
            ids.push_back(tok->nl_token_id);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        append_user_opening(round_idx, ids);
        tok->encode(" " + user, ids);
        ids.push_back(tok->end_of_utterance_token_id);
        ids.push_back(tok->nl_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("Assistant:", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("User:", ids);
    }

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int extra_ids) : pad_arg(new BlockParams::PadEmbedding(extra_ids, extra_ids)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        typedef llama::v2::GenericConditionalGeneration<LlamaBlock> Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_SMOL_VLM)
            : ExtendEmbedding(4096),
              Base(config, runtime_config, type, config.num_key_value_heads, config.max_length, 12, false),
              visual(runtime_config)
        {
            delete pad_arg;
            pad_arg = nullptr;
        }

    public:
        bool load_more(const json::JSON &config) override
        {
            Base::load_more(config);
            bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config);
            if (r)
            {
                _chat_encoder.vis_config = &visual.vis_config;
            }
            return r;
        }

        void load(ModelLoader &loader) override
        {
            Base::load(loader);

            loader.add_tensor_name_translations({
                {".input_layernorm.",                   ".layer_norm1."},
                {".post_attention_layernorm.",          ".layer_norm2."},
            });

            _chat_encoder.vit_loaded = visual.load(loader);
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            tok->do_split = utils::get_opt(args, "do-split", false);
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

        std::vector<std::unique_ptr<vision::VideoLoader>> videos;

        // expand video into images
        std::vector<ContentPiece> pieces;
        for (auto &piece : user.pieces)
        {
            if (piece.type != ContentPiece::Type::Video)
            {
                pieces.push_back(piece);
                continue;
            }

            auto video = new vision::VideoLoader(piece.content.c_str(), vis_config->video.fps, vis_config->video.max_frames);
            videos.emplace_back(video);
            if (video->frames.size() < 1)
                continue;

            std::ostringstream oss;
            oss << "You are provided the following series of " << utils::num2words((int)video->frames.size())
                << " frames from a " << utils::sec2hms((float)video->frames.size() / vis_config->video.fps) << " [H:MM:SS] video.\n";
            pieces.emplace_back(oss.str());

            for (size_t i = 0; i < video->frames.size(); i++)
            {
                oss.str("");
                oss << "\nFrame from " << utils::sec2ms(i / vis_config->video.fps);
                pieces.emplace_back(oss.str());

                pieces.emplace_back(video->frames[i], ContentPiece::Type::Image);
            }

            // DEFAULT_MEDIA_OUTTRO = "\n\n"
            pieces.emplace_back("\n\n");
        }

        for (auto &piece : pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->patch_size;
                const int image_size_per_split = vis_config->image_size_per_split;

                vision::PreMaxImageSize   max_size(vis_config->image_max_size, vis_config->image_max_size);

                std::vector<vision::image_pixels_t> splits;

                int splits_cols_num = 0;
                int splits_rows_num = 0;

                vision::image_load_split(piece.content.c_str(), splits, tok->do_split, image_size_per_split,  image_size_per_split, splits_cols_num, splits_rows_num);

                std::vector<float> scaled;

                for (int r = 0; r < splits_rows_num; r++)
                {
                    for (int c = 0; c < splits_cols_num; c++)
                    {
                        const int split_id = r * splits_cols_num + c;
                        vision::image_rescale(splits[split_id], scaled);
                        vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                        tok->media_emb.push_back({.width = image_size_per_split, .height = image_size_per_split, .patch_size = patch_size, .data = {}});
                        auto &image = tok->media_emb.back();
                        vision::image_arrange(scaled, image_size_per_split, -1, image.data, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);
                        image.emb_vec_number = vis_config->image_seq_len;

                        const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                        tok->inject_image(ids, id_start, image.emb_vec_number, c, r);
                    }
                }

                vision::image_rescale(splits[splits_cols_num * splits_rows_num], scaled);
                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.width = image_size_per_split, .height = image_size_per_split, .patch_size = patch_size, .data = {}});
                auto &image = tok->media_emb.back();
                vision::image_arrange(scaled, image_size_per_split, -1, image.data, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);
                image.emb_vec_number = vis_config->image_seq_len;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_global_image(ids, id_start, image.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        ids.push_back(tok->end_of_utterance_token_id);
        ids.push_back(tok->nl_token_id);
    }
}