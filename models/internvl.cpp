
#include <cstring>
#include "qwen.h"
#include "../src/vision_process.h"

namespace chatllm::intern::vit
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int max_patches;
        int image_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int out_hidden_size;

        bool use_thumbnail;
        float downsample_ratio;

        float image_mean[3];
        float image_std[3];

        Config()
        {
            memset(this, 0, sizeof(Config));
        }
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int pos_emb_height, int pos_emb_width, int in_dim = 3):
            patch_size(patch_size), pos_emb_height(pos_emb_height), pos_emb_width(pos_emb_width),
            proj(ctx, in_dim, out_dim, patch_size, patch_size),
            // NOTE: F32 to make ggml happy
            pos_emb(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, out_dim, pos_emb_height * pos_emb_width + 1)),
            cls_emb(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, out_dim))
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            ggml::tensor *x = proj.forward(ctx, input);

            auto emb = _get_pos_embed(ctx, ggml::get_dim(x, 1), ggml::get_dim(x, 0));

            x = ggml::reshape_3d(ctx, x, ggml::get_dim(x, 0) * ggml::get_dim(x, 1), ggml::get_dim(x, 2), ggml::get_dim(x, 3));
            x = ggml::permute(ctx, x, 1, 0);

            x = ggml::add(ctx, x, emb);
            auto pad = ggml::view_1d(ctx, pos_emb, ggml::get_dim(pos_emb, 0), 0);
            pad = ggml::add(ctx, cls_emb, pad);
            pad = ggml::repeat(ctx, pad, ggml::get_dim(pad, 0), 1, ggml::get_dim(x, 2));
            x = ggml::concat(ctx, pad, x, 1);
            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += proj.get_param_num(effective_only);
            r += ggml::nelements(pos_emb);
            r += ggml::nelements(cls_emb);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            proj.load   (path + "patch_embedding.", loader);
            loader->read_tensor(path + "class_embedding", cls_emb);
            loader->read_tensor(path + "position_embedding", pos_emb);
        }

    public:
        ggml::tensor *_get_pos_embed(ComputeContext *ctx, int grid_h, int grid_w)
        {
            auto pos_emb = ggml::view_3d(ctx, this->pos_emb, ggml::get_dim(this->pos_emb, 0),
                pos_emb_width, pos_emb_height,
                ggml::row_size(this->pos_emb),
                ggml::row_size(this->pos_emb) * pos_emb_width,
                ggml::row_size(this->pos_emb));

            ggml::tensor * interp = nullptr;
            if ((grid_w == ggml::get_dim(pos_emb, 1)) && (grid_h == ggml::get_dim(pos_emb, 2)))
            {
                interp = pos_emb;
            }
            else
            {
                auto permuted = ggml::permute(ctx, pos_emb, 2, 0, 1, 3);
                interp = ggml::interpolate(ctx, permuted, ggml::InterpolateMode::Bicubic,
                    grid_w, grid_h, ggml::get_dim(permuted, 2), ggml::get_dim(permuted, 3));

                interp = ggml::permute(ctx, interp, 1, 2, 0, 3);
            }

            interp = ggml::reshape_2d(ctx, interp, ggml::get_dim(interp, 0), ggml::get_dim(interp, 1) * ggml::get_dim(interp, 2));

            return interp;
        }

    public:
        const int patch_size;
        const int pos_emb_height;
        const int pos_emb_width;
        Conv2D                  proj;
        ggml::tensor           *pos_emb;
        ggml::tensor           *cls_emb;
    };

    typedef TheBiasedGELUMLP MLP;

    class ViTSelfAttention : public BaseCachelessAttention
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true)
        {
            causal = false;
        }
    };

    class MultiModalProjector : public Sequential
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size):
            downsample_ratio(config.downsample_ratio),
            hidden_size(config.hidden_size),
            crop_size(config.image_size),
            patch_size(config.patch_size)
        {
            const int ratio = ((int)(1 / config.downsample_ratio)) * ((int)(1 / config.downsample_ratio));
            add_block(new LayerNorm(ctx, hidden_size * ratio));
            add_block(new Linear(ctx, hidden_size * ratio, lm_hidden_size));
            add_block(new ActivationBlock(ActFunc::GELU));
            add_block(new Linear(ctx, lm_hidden_size, lm_hidden_size));
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            const int w = (crop_size / patch_size);

            // vit_embeds = vit_embeds[:, 1:, :]
            auto vit_embeds = ggml::view_4d(ctx, input,
                ggml::get_dim(input, 0), ggml::get_dim(input, 1) - 1, ggml::get_dim(input, 2), ggml::get_dim(input, 3),
                ggml::row_size(input),
                ggml::row_size(input) * ggml::get_dim(input, 1),
                ggml::row_size(input) * ggml::get_dim(input, 1) * ggml::get_dim(input, 2),
                0);

            CHATLLM_CHECK(ggml::get_dim(vit_embeds, 1) == w * w);

            vit_embeds = ggml::reshape_4d(ctx, vit_embeds, ggml::get_dim(vit_embeds, 0), w, w, ggml::get_dim(vit_embeds, 2));
            vit_embeds = pixel_shuffle(ctx, vit_embeds, downsample_ratio);
            vit_embeds = ggml::reshape_3d(ctx, vit_embeds, ggml::get_dim(vit_embeds, 0), ggml::get_dim(vit_embeds, 1) * ggml::get_dim(vit_embeds, 2), ggml::get_dim(vit_embeds, 3));
            vit_embeds = Sequential::forward(ctx, vit_embeds);
            vit_embeds = ggml::reshape_2d(ctx, vit_embeds, ggml::get_dim(vit_embeds, 0), ggml::get_dim(vit_embeds, 1) * ggml::get_dim(vit_embeds, 2));
            return vit_embeds;
        }

    protected:
        ggml::tensor *pixel_shuffle(ComputeContext *ctx, ggml::tensor *input, const float scale)
        {
            auto c = ggml::get_dim(input, 0);
            auto h = ggml::get_dim(input, 1);
            auto w = ggml::get_dim(input, 2);
            auto b = ggml::get_dim(input, 3);

            input = ggml::reshape(ctx, input, (int64_t)(c / scale), (int64_t)(h * scale), w, b);
            input = ggml::permute(ctx, input, 0, 2, 1, 3);
            input = ggml::reshape(ctx, input, (int64_t)(c / (scale * scale)), (int64_t)(w * scale), (int64_t)(h * scale), b);
            input = ggml::cont   (ctx, input);
            input = ggml::permute(ctx, input, 0, 2, 1, 3);
            return input;
        }
    public:
        const float downsample_ratio;
        const int hidden_size;
        const int crop_size;
        const int patch_size;
    };

    class LayerBlock : public LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP>
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP> Base;
        LayerBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length):
            Base(ctx, hidden_size, num_attention_heads, intermediate_size, max_length),
            ls1(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, hidden_size)),
            ls2(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, hidden_size))
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features) override
        {
            LMBlock1Forward eval(&input_layernorm, &(Base::attention), &post_attention_layernorm, &mlp, Base::get_id(), scale_depth);
            eval.set_attn_scaling(ls1);
            eval.set_mlp_scaling(ls2);
            auto hidden_states = eval.forward(ctx, image_features, 0);
            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += Base::get_param_num(effective_only);
            r += ggml::nelements(ls1);
            r += ggml::nelements(ls2);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            loader->read_tensor(path + "ls1", ls1);
            loader->read_tensor(path + "ls2", ls2);
        }

    public:
        ggml::tensor *ls1;
        ggml::tensor *ls2;
    };

    class VisionTransformer : public DynamicBlock
    {
    public:
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size):
            config(config),
            embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
                        config.hidden_size, config.patch_size, config.image_size / config.patch_size, config.image_size / config.patch_size),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, lm_hidden_size)
        {
            const int max_length = config.max_patches;
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
            r += multi_modal_projector.get_param_num(effective_only);
            for (size_t i = 0; i < layers.size(); i++)
                r += layers[i]->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            if (!loader->has_tensor(path + "embeddings.class_embedding")) return;

            embeddings.load(path + "embeddings.", loader);
            multi_modal_projector.load("vision_projector.", loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "layers." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            _loaded = true;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            auto output = embeddings.forward(ctx, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                output = layers[i]->forward(ctx, output);
            }
            output = multi_modal_projector.forward(ctx, output);
            return output;
        }

    public:
        const Config config;
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        MultiModalProjector multi_modal_projector;
    };
}

namespace chatllm::intern::vl
{
    typedef qwen::v3::Config Config;

    class ChatHistoryEncoder : public qwen::v2_5_vl::ChatHistoryEncoder
    {
    public:
        void append_content(const Content &user, std::vector<int> &ids) const override;
    protected:
        void append_content_internvl(const Content &user, std::vector<int> &ids) const;
        void append_content_qianfanvl(const Content &user, std::vector<int> &ids) const;
        void append_image_piece(const ContentPiece &piece, std::vector<int> &ids) const;
    public:
        const vit::Config *vis_config = nullptr;
        std::string template_name;
    };

    static class ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public qwen::v2_5_vl::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder = nullptr):
            qwen::v2_5_vl::Tokenizer(config, encoder ? encoder : &_chat_encoder)
        {}
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = qwen::v3::Tokenizer::load(buffer, n_vocab);

            img_start_token_id = tp->PieceToId("<img>");
            img_end_token_id   = tp->PieceToId("</img>");

            return r;
        }
        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count, bool add_nl)
        {
            ids.push_back(img_start_token_id);
            for (int i = 0; i < ids_to_inject_count; i++)
                ids.push_back(i + ids_to_inject_start);
            ids.push_back(img_end_token_id);
            if (add_nl)
                ids.push_back(nl_token_id);
        }
    public:
        int img_start_token_id;
        int img_end_token_id;
    };

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int n = 4096) : pad_arg(new BlockParams::PadEmbedding(n, n)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public qwen::v3::ConditionalGeneration
    {
    public:
        typedef qwen::v3::ConditionalGeneration Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = (ModelType)MODEL_TYPE_INTERN_VL);
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void before_generate(const GenerationConfig &gen_config) override;
        int64_t get_param_num(bool effective_only) const;
    protected:
        const int max_patches;
        std::unique_ptr<vit::VisionTransformer> vision;
        TensorGraphEvaluator vis_eval;
        InitContext vis_ctx;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type):
        ExtendEmbedding(),
        qwen::v3::ConditionalGeneration(config, runtime_config, type),
        max_patches(pad_arg->get()),
        vis_eval(runtime_config, "vis"),
        vis_ctx(&backend_context)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);

        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        _chat_encoder.template_name = config["config.json"]["template"].ToString();

        vit::Config vis_config;

        vis_config.dtype = this->config.dtype;

        vis_config.max_patches          = max_patches;
        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.image_size           = (int)vis_cfg["image_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
        vis_config.out_hidden_size      = this->config.hidden_size;

        vis_config.use_thumbnail        =        config["config.json"]["use_thumbnail"].ToBool();
        vis_config.downsample_ratio     = (float)config["config.json"]["downsample_ratio"].ToFloat();

        CHATLLM_CHECK((int)vis_cfg["num_channels"].ToInt()    == 3);

        auto pp_cfg = config["preprocessor_config.json"];
        if (pp_cfg.IsObject())
        {
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
        }

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 10 + vis_config.num_hidden_layers * 19;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        vis_ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        vis_ctx.dtype = vis_config.dtype;

        vision.reset(new vit::VisionTransformer(&vis_ctx, vis_config, this->config.hidden_size));

        vis_ctx.check_used_mem_size(true);

        return true;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {".input_layernorm.",               ".norm1."},
            {".post_attention_layernorm.",      ".norm2."},
            {".mlp.fc1.",                       ".mlp.fc2."},
            {".mlp.fc0.",                       ".mlp.fc1."},
        });

        if (vision.get() == nullptr) return;

        vision->load("visual.", &loader);
        if (vision->is_loaded())
        {
            _chat_encoder.vis_config = &vision->config;
            _chat_encoder.vit_loaded = true;
        }
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = Base::get_param_num(effective_only);
        if (vision.get() != nullptr)       r += vision->get_param_num(effective_only);
        return r;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (vision.get() == nullptr) return;
        if (!vision->is_loaded()) return;

        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        const vit::Config vis_config = vision->config;

        for (auto &image : tok->media_emb)
        {
            ggml::tensor *media_emb = nullptr;
            const auto make_graph = [this, &media_emb, &image, &vis_config](ComputeContext *ctx) -> ggml::tensor * {
                media_emb = ggml::new_tensor_4d(ctx,
                    ggml::type::GGML_TYPE_F32, image.patch_size, image.patch_size, 3, image.grid_height * image.grid_width + 1);
                auto r = vision->forward(ctx, media_emb, image.grid_height, image.grid_width);
                return r;
            };
            const auto write_input_data = [&media_emb, &image](ComputeContext *ctx) {
                Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            };

            std::vector<int64_t> shape;
            vis_eval.evaluate(gen_config, make_graph, write_input_data, ggml::type_of(emb->weight),
                shape, buf);
        }

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    void ChatHistoryEncoder::append_content_internvl(const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                append_image_piece(piece, ids);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
    }

    void ChatHistoryEncoder::append_content_qianfanvl(const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::string question;

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                question = question + piece.content;
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                append_image_piece(piece, ids);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        if (question.size() == 0)
            question = "Parse this document to Markdown.";
        tok->encode(question, ids);
    }

    void ChatHistoryEncoder::append_image_piece(const ContentPiece &piece, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        const int patch_num_per_crop =
            int((vis_config->image_size / vis_config->patch_size) * (vis_config->image_size / vis_config->patch_size) * (double)vis_config->downsample_ratio * vis_config->downsample_ratio);

        CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";
        CHATLLM_CHECK(piece.type == ContentPiece::Type::Image);

        int crops_per_row = 0;
        int num_crop_rows = 0;
        std::vector<vision::image_pixels_t> crops;

        vision::image_load_pan_and_scan(piece.content.c_str(), crops,
            crops_per_row, num_crop_rows, 1, 12,
            vis_config->image_size, true);

        tok->media_emb.push_back({.grid_width = crops_per_row, .grid_height = num_crop_rows, .patch_size = vis_config->image_size, .data = {}});
        auto &image = tok->media_emb.back();

        image.emb_vec_number = (crops_per_row * num_crop_rows + 1) * patch_num_per_crop;
        const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
        tok->inject_media("image", ids, id_start, image.emb_vec_number, true);

        for (auto &pixels : crops)
        {
            std::vector<float> scaled;
            std::vector<float> arranged;

            vision::image_rescale(pixels, scaled);
            vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

            vision::image_arrange(scaled, vis_config->image_size, -1, arranged, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);

            image.data.insert(image.data.end(), arranged.begin(), arranged.begin() + arranged.size());
        }
    }

    void ChatHistoryEncoder::append_content(const Content &user, std::vector<int> &ids) const
    {
        if (template_name == "qianfanvl")
            append_content_qianfanvl(user, ids);
        else
            append_content_internvl(user, ids);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(INTERN_VL,           intern::vl, 1);
}
