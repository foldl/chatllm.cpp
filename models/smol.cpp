#include "smol.h"
#include "../src/vision_process.h"

namespace chatllm::smol::lm
{
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

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

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

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length)
        : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length, 12, true)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
}

namespace chatllm::smol::vit
{
    PatchEmbedding::PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int image_max_size, int in_dim)
        : num_patches_per_side(image_max_size / patch_size),
            num_patches(num_patches_per_side * num_patches_per_side),
            patch_embedding(ctx, in_dim, out_dim, patch_size, patch_size),
            position_embedding(TypeChanger(ctx, ggml::type::GGML_TYPE_F32), num_patches, out_dim)
    {
    }

    ggml::tensor *PatchEmbedding::forward(ComputeContext *ctx, ggml::tensor *input, int image_width, int image_height)
    {
        ggml::tensor *x = patch_embedding.forward(ctx, input);
        x = ggml::reshape_3d(ctx, x, ggml::get_dim(x, 0) * ggml::get_dim(x, 1), ggml::get_dim(x, 2), ggml::get_dim(x, 3));
        x = ggml::transpose(ctx, x);
        x = ggml::cont(ctx, x);
        x = ggml::add(ctx, x, position_embedding.weight);
        return x;
    }

    int64_t PatchEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r +=    patch_embedding.get_param_num(effective_only);
        r += position_embedding.get_param_num(effective_only);
        return r;
    }

    void PatchEmbedding::load(const std::string &path, TensorLoader *loader)
    {
            patch_embedding.load(path + "patch_embedding.", loader);
        position_embedding.load(path + "position_embedding.", loader);
    }

    ViTSelfAttention::ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
        : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true)
    {
        causal = false;
    }

    MultiModalProjector::MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size)
        :
        hidden_size(config.hidden_size), scale_factor(config.scale_factor),
        patch_size(config.patch_size),
        proj(ctx, hidden_size * scale_factor * scale_factor, lm_hidden_size, false)
    {
        merge_param.merge_kernel_size[0] = scale_factor;
        merge_param.merge_kernel_size[1] = scale_factor;
    }

    ggml::tensor *MultiModalProjector::forward(ComputeContext *ctx, ggml::tensor *image_features, int image_width, int image_height)
    {
        auto output = pixel_shuffle(ctx, image_features, image_width, image_height);
        output = proj.forward(ctx, output);
        return output;
    }

    int64_t MultiModalProjector::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += proj.get_param_num(effective_only);
        return r;
    }

    void MultiModalProjector::load(const std::string &path, TensorLoader *loader)
    {
        proj.load(path + "proj.", loader);
    }

    ggml::tensor *MultiModalProjector::pixel_shuffle(ComputeContext *ctx, ggml::tensor *image_features, int image_width, int image_height)
    {
        merge_param.grid_w = image_width  / patch_size;
        merge_param.grid_h = image_height / patch_size;

        ggml::tensor *r = ggml::merge_patch(ctx, image_features, &merge_param);
        r = ggml::reshape_4d(ctx, r, ggml::get_dim(r, 0) * scale_factor * scale_factor, ggml::get_dim(r, 1) / (scale_factor * scale_factor), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
        return r;
    }

    VisionTransformer::VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size)
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

    int64_t VisionTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embeddings.get_param_num(effective_only);
        r += post_layernorm.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void VisionTransformer::load(const std::string &path, TensorLoader *loader)
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

    ggml::tensor *VisionTransformer::forward(ComputeContext *ctx, ggml::tensor *input, int image_width, int image_height)
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

    bool VisionTransformer::is_loaded(void) const
    {
        return loaded;
    }

    VisualEmbeddingGeneration::VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE)
        :
        GRAPH_SIZE(GRAPH_SIZE), _ctx(&backend_context),
        n_threads(runtime_config.n_threads)
    {
        _ctx.cache_dtype = runtime_config.cache_type;
        model_gpu_layers = BackendContext::get_ngl_of_model(runtime_config.model_gpu_layers, "vis");
    }

    bool VisualEmbeddingGeneration::load(ModelLoader &loader)
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

    bool VisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
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

    void VisualEmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if ((vis_model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!vis_model->is_loaded()) return;

        for (auto &image : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, image, buf);
        }
    }

    bool VisualEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
    {
        ForwardContext ctx(&backend_context);
        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *media_emb = ggml::new_tensor_3d(&ctx, ggml::type::GGML_TYPE_F32, image.width, image.height, 3);

        set_dbg_ctx(&ctx);

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
}

namespace chatllm::smol::vlm
{
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

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
        do_split = false;
    }

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

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ExtendEmbedding(4096),
            Base(config, runtime_config, type, config.num_key_value_heads, config.max_length, 12, false),
            visual(runtime_config)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);
        bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config);
        if (r)
        {
            _chat_encoder.vis_config = &visual.vis_config;
        }
        return r;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {".input_layernorm.",                   ".layer_norm1."},
            {".post_attention_layernorm.",          ".layer_norm2."},
        });

        _chat_encoder.vit_loaded = visual.load(loader);
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->do_split = utils::get_opt(args, "do-split", false);
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

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

namespace chatllm::smol::lm3
{
    class ChatHistoryEncoder : public lm::ChatHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : lm::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "/think";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = lm::Tokenizer::load(buffer, n_vocab);
        bos_token_id = tp->PieceToId("<|im_start|>");
        eos_token_id = tp->PieceToId("<|im_end|>");

        tp->OverrideTokenDecoding(tp->PieceToId("<think>"),  "<think>");
        tp->OverrideTokenDecoding(tp->PieceToId("</think>"), "</think>");

        return r;
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        if (tok->get_system_prompt().find("/system_override") != std::string::npos)
        {
            oss << utils::replace_all(tok->get_system_prompt(), "/system_override", "");
        }
        else
        {
            std::string s = tok->get_system_prompt();
            bool reason = s.find("/think") != std::string::npos;
            s = utils::replace_all(s, "/think", "");
            s = utils::replace_all(s, "/no_think", "");
            s = utils::trim(s);
            std::string date = utils::now("%d %B %Y");

            oss << "## Metadata\n\n"
                << "Knowledge Cutoff Date: June 2025\n"
                << "Today Date: " << date << "\n"
                << "Reasoning Mode: " << (reason ? "/think" : "/no_think") << "\n"
                << "## Custom Instructions\n\n";

            if (s.size() > 0)
                oss << s;
            else if (reason)
                oss << "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.";
            else
                oss << "You are a helpful AI assistant named SmolLM, trained by Hugging Face.";
            oss << "\n\n";
        }

        ids.push_back(tok->bos_token_id);
        tok->encode("system\n", ids);
        tok->encode(oss.str(), ids);
        ids.push_back(tok->eos_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length)
        : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length, 12, config.tie_word_embeddings != 0)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.rope_mode = RoPEMode::Original;
            attention.freq_base = config.rope_theta;
            attention.use_rope  = (i % config.no_rope_layer_interval) != (config.no_rope_layer_interval - 1);
        }
    }
}