
namespace persimmon
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int rope_dim;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
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

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
    public:
        int image_newline_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, LayerNorm, PersimmonBlock, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        void load(ModelLoader &loader) override;

    public:
        Config config;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::UnigramProcessor(0);
        size_t size = tp->Load(buffer, n_vocab);

        image_newline_id     = tp->PieceToId("|NEWLINE|");

        if (bos_token_id == eos_token_id)
            bos_token_id = 1;
        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
    {
        if (add_bos)
            ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            ids.push_back(eos_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        append_ai_opening(round_idx, ids);

        oss_prompt << ai << "\n\n";
        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << " human: " << user << "\n\n";
        tok->encode(oss_prompt.str(), ids, true, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("adept: ", ids, true, false);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : BaseModelForConditionalGeneration(MODEL_TYPE_PERSIMMON, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 4 + config.num_hidden_layers * 23;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, false,
                                                            config.hidden_size, config.num_attention_heads,
                                                            config.intermediate_size, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.rope_dim = config.rope_dim;
            layer.attention.freq_base = config.rope_theta;
        }

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size()) << "corrupted model weights";
    }

    static void translate_tensor_names(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
                {".fc1.",               ".dense_4h_to_h."},
                {".fc0.",               ".dense_h_to_4h."},
                {".o_proj.",            ".dense."},
                {".k_norm.",            ".k_layernorm."},
                {".q_norm.",            ".q_layernorm."},
                {"model.norm.",         "model.final_layernorm."},
                {"model.norm.",         "model.final_layernorm."},
            });
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        translate_tensor_names(loader);
        BaseModelForConditionalGeneration::load(loader);
    }
}

namespace fuyu
{
    struct Config : public persimmon::Config
    {
        int patch_size;
    };

    struct VisualConfig
    {
        ggml::type dtype;
        int channel_num;
        int hidden_size;
        int patch_size;
        int target_width;
        int target_height;
        float image_mean[3];
        float image_std[3];
    };

    const int MAX_PATCH_NUM     = 3000;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {}
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {}
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    public:
        const VisualConfig *vis_config = nullptr;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public persimmon::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : persimmon::Tokenizer(config, &_chat_encoder)
        {}
    };

    class VisualEmbedding
    {
    public:
        VisualEmbedding(InitContext *ctx, const VisualConfig &config)
            : vision_embed(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
                           config.patch_size * config.patch_size * config.channel_num, config.hidden_size)
        {}

        void load(const std::string &path, ModelLoader &loader)
        {
            vision_embed.load(path + "vision_embed_tokens.", &loader);
        }

    public:
        Linear vision_embed;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(BackendContext *backend_context, const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 128)
            :
            backend_context(backend_context), GRAPH_SIZE(GRAPH_SIZE), _ctx(backend_context)
        {
        }

        void load(ModelLoader &loader)
        {
            if (vis_emb.get())
            {
                vis_emb->load("", loader);
            }
        }

        bool load_more(ggml::type dtype, int patch_size, int hidden_size, const json::JSON &config)
        {
            auto pp_cfg = config["preprocessor_config.json"];
            if (!pp_cfg.IsObject()) return false;

            vis_config.dtype                = dtype;
            vis_config.channel_num          = 3;
            vis_config.patch_size           = patch_size;
            vis_config.hidden_size          = hidden_size;

            vis_config.target_height        = (int)pp_cfg["target_height"].ToInt();
            vis_config.target_width         = (int)pp_cfg["target_width"].ToInt();

            for (int i = 0; i < vis_config.channel_num; i++)
            {
                vis_config.image_mean[i]    = 0.5f;
                vis_config.image_std[i]     = 0.5f;
            }

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 2;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            _ctx.dtype = dtype;

            vis_emb.reset(new VisualEmbedding(&_ctx, vis_config));

            _chat_encoder.vis_config = &vis_config;

            return true;
        }

        void generate(const GenerationConfig &gen_config, Tokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
        {
            if (vis_emb.get() == nullptr) return;

            if (tok->media_emb.size() < 1) return;

            run_model(gen_config, tok, dtype, buf);
        }

    protected:
        bool run_model(const GenerationConfig &gen_config, Tokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
        {
            ForwardContext ctx(backend_context);
            ctx.gctx = GGMLContext({.mem_size = backend_context->buf_compute_meta.size(), .mem_buffer = backend_context->buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            const int total_patches = tok->get_image_total_emb_vectors();
            ggml::tensor *patches = ggml::new_tensor_2d(&ctx, ggml::type::GGML_TYPE_F32, 3 * vis_config.patch_size * vis_config.patch_size, total_patches);

            dbg_ctx = &ctx;

            auto r = vis_emb->vision_embed.forward(&ctx, patches);

            if (ggml::type_of(r) != dtype)
            {
                ggml::tensor *t = ggml::new_tensor_4d(&ctx, dtype, ggml::get_dim(r, 0), ggml::get_dim(r, 1), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
                r = ggml::cpy(&ctx, r, t);
            }

            ggml::build_forward_expand(&ctx, r);

            CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";

            if (gen_config.dump_dot.size() > 0)
            {
                backend_context->dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
                exit(-1);
            }

            size_t offset = 0;
            for (auto &image : tok->media_emb)
            {
                size_t size = image.data.size() * sizeof(image.data[0]);
                Backend::write_tensor_data(patches, image.data.data(), offset, size);
                offset += size;
            }

            ctx.compute();

            buf.resize(ggml::nbytes(r));
            Backend::read_tensor_data(r, buf.data());

            ctx.reset();

            return true;
        }

    protected:
        std::unique_ptr<VisualEmbedding> vis_emb;
        BackendContext *backend_context;
        const size_t GRAPH_SIZE;
        InitContext _ctx; // weight context
        VisualConfig vis_config;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, LayerNorm, PersimmonBlock, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_FUYU, config, runtime_config), config(config),
              visual(w_ctx_.get_backend_context(), runtime_config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 4 + config.num_hidden_layers * 23;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            BlockParams::PadEmbedding padding(MAX_PATCH_NUM, MAX_PATCH_NUM);

            transformer = new ModelClass(&w_ctx_, config, false,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = get_typed_transformer<ModelClass>()->layers[i];
                layer.attention.rope_dim = config.rope_dim;
                layer.attention.freq_base = config.rope_theta;
            }

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size()) << "corrupted model weights";
        }

        void load(ModelLoader &loader) override
        {
            persimmon::translate_tensor_names(loader);
            BaseModelForConditionalGeneration::load(loader);

            visual.load(loader);
        }

        bool load_more(const json::JSON &config) override
        {
            BaseModelForConditionalGeneration::load_more(config);
            return visual.load_more(this->config.dtype, this->config.patch_size, this->config.hidden_size, config);
        }

        void before_generate(const GenerationConfig &gen_config) override
        {
            std::vector<uint8_t> buf;
            auto &emb = dynamic_cast<ModelClass *>(transformer)->word_embeddings;
            visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb.weight), buf);
            if (buf.size() < 1) return;

            size_t offset = emb.get_base_nbytes();
            Backend::write_tensor_data(emb.weight, buf.data(), offset, buf.size());
        }

        void after_generate(void) override
        {
            tokenizer->media_emb.clear();
        }

    public:
        Config config;
        VisualEmbeddingGeneration visual;
    };

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(user, ids, true, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vis_config) << "Vision model not loaded";

                int w, h;
                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->patch_size;
                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, -1, MAX_PATCH_NUM, vision::PaddingMode::White);

                if (w <= 0) continue;

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_PixelsLeftRightDown_ChannelsRGB);

                image.emb_vec_number = image.grid_width * image.grid_height;

                CHATLLM_CHECK(image.emb_vec_number) << "too many image patches!";

                const int total_patches = tok->get_image_total_emb_vectors();
                CHATLLM_CHECK(total_patches <= MAX_PATCH_NUM) << "too many image patches!";

                int id = total_patches - image.emb_vec_number + tok->vocab_size;
                for (int j = 0; j < image.grid_height; j++)
                {
                    for (int i = 0; i < image.grid_width; i++)
                    {
                        ids.push_back(id++);
                    }
                    ids.push_back(tok->image_newline_id);
                }
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        tok->encode("\n\n", ids, false, false);
    }
}