#include "qwen.h"
#include "../src/vision_process.h"

namespace chatllm::dots::vit
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int spatial_merge_size;
        int temporal_patch_size;
        int min_pixels;
        int max_pixels;
        int max_patches;

        float image_mean[3];
        float image_std[3];
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, const Config &config)
            : proj(ctx, 3, config.hidden_size, config.patch_size, config.patch_size),
              norm(ctx, config.hidden_size)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
        {
            ggml::tensor *x = nullptr;
            x = proj.forward(ctx, input);
            x = ggml::reshape_2d(ctx, x, ggml::get_dim(x, 2), ggml::get_dim(x, 3));
            x = norm.forward(ctx, x);
            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += proj.get_param_num(effective_only);
            r += norm.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            proj.load(path + "proj.", loader);
            norm.load(path + "norm.", loader);
        }
    public:
        Conv2D  proj;
        RMSNorm norm;
    };

    class TensorPosHelper
    {
    public:
        TensorPosHelper(int max_length, int patch_size, int spatial_merge_size)
            : max_length(max_length * 4), original_length(max_length), patch_size(patch_size),
              spatial_merge_size(spatial_merge_size)
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
            CHATLLM_CHECK(grid_h % spatial_merge_size == 0);
            CHATLLM_CHECK(grid_w % spatial_merge_size == 0);

            length = grid_h * grid_w;

            v_pos.clear();
            v_pos.resize(length * 2, 0);
            int *p_h = &v_pos[length * 0];
            int *p_w = &v_pos[length * 1];
            int index = 0;

            for (int i = 0; i < grid_h / spatial_merge_size; i++)
            {
                for (int j = 0; j < grid_w / spatial_merge_size; j++)
                {
                    for (int i0 = 0; i0 < spatial_merge_size; i0++)
                    {
                        for (int j0 = 0; j0 < spatial_merge_size; j0++)
                        {
                            p_w[index] = j * spatial_merge_size + j0;
                            p_h[index] = i * spatial_merge_size + i0;
                            index++;
                        }
                    }

                }
            }
        }
    public:
        const int max_length;
        const int original_length;
        const int patch_size;
        const int spatial_merge_size;
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

    class ViTSelfAttention : public TensorPosHelperPrelude, public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : TensorPosHelperPrelude(new TensorPosHelper2D(max_length)),
            RoPESelfAttention<BaseCachelessAttention>(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, false, false)
        {
            TensorPosHelperPrelude::done();
            causal = false;
            CHATLLM_CHECK(rope_dim % 4 == 0);
        }
        void set_pos_helper(TensorPosHelper *helper)
        {
            TensorPosHelper2D *h = (TensorPosHelper2D *)pos_helper.get();
            h->helper = helper;
        }
    protected:
        ggml::tensor *apply_2d_rope(ComputeContext *ctx, ggml::tensor *hidden, int hidden_size) const
        {
            // ggml shape of hidden: [head_size, heads, qlen]
            int sections[4] = {rope_dim / 4, rope_dim / 4, 0, 0};

            hidden = ggml::rope_ext_inplace(ctx, hidden, pos, freq_factors, rope_dim / 2, GGML_ROPE_TYPE_VISION, n_original_ctx,
                                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
                                sections);

            return hidden;
        }
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            k = apply_2d_rope(ctx, k, hidden_size);
            return k;
        }
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            q = apply_2d_rope(ctx, q, hidden_size);
            return q;
        }
    public:
        int grid_w = 0;
        int grid_h = 0;
    };

    class MultiModalProjector : public qwen::vit::GenMultiModalProjector<LayerNorm, qwen::vit::MLP>
    {
    public:
        MultiModalProjector(InitContext *ctx, int hidden_size, int spatial_merge_size, int lm_hidden_size):
            qwen::vit::GenMultiModalProjector<LayerNorm, qwen::vit::MLP>(ctx, hidden_size, spatial_merge_size, lm_hidden_size)
        {}
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<RMSNorm, ViTSelfAttention, RMSNorm, SiLUMLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size):
            embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.spatial_merge_size, lm_hidden_size),
            post_trunk_norm(ctx, config.hidden_size),
            config(config), loaded(false)
        {
            const int max_length = config.max_pixels / config.patch_size / config.patch_size;
            pos_helper.reset(new TensorPosHelper(max_length, config.patch_size, config.spatial_merge_size));

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
                layer->set_id(layer_id);
                layer->attention.set_pos_helper(pos_helper.get());
                layers.emplace_back(layer);
            }
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += embeddings.get_param_num(effective_only);
            r += multi_modal_projector.get_param_num(effective_only);
            r += post_trunk_norm.get_param_num(effective_only);
            for (size_t i = 0; i < layers.size(); i++)
                r += layers[i]->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            if (!loader->has_tensor(path + "patch_embed.proj.weight")) return;

            embeddings.load(path + "patch_embed.", loader);
            multi_modal_projector.load(path + "merger.", loader);
            post_trunk_norm.load(path + "post_trunk_norm.", loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "layers." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            loaded = true;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
        {
            pos_helper->prepare(grid_h, grid_w);

            auto output = embeddings.forward(ctx, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->attention.grid_h = grid_h;
                layers[i]->attention.grid_w = grid_w;
                output = layers[i]->forward(ctx, output, 0);
            }
            output = post_trunk_norm.forward(ctx, output);
            output = multi_modal_projector.forward(ctx, output, grid_h, grid_w);
            return output;
        }

        bool is_loaded(void) const
        {
            return loaded;
        }
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        MultiModalProjector multi_modal_projector;
        RMSNorm post_trunk_norm;
        std::unique_ptr<TensorPosHelper> pos_helper;
        const Config config;
    protected:
        bool loaded;
    };
}

namespace chatllm::dots::ocr
{
    typedef qwen::v2::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    public:
        const vit::Config *vis_config;
    };

    static class ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public qwen::v1::Tokenizer
    {
    public:
        Tokenizer(const Config &config) : qwen::v1::Tokenizer(config, &_chat_encoder)
        {
            auto_add_bos = false;
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);

    public:
        void encode(const std::string &text, std::vector<int> &ids, int prefix_id, int suffix_id) const;

        using qwen::v1::Tokenizer::encode;

    public:
        int user_token_id;
        int assistant_token_id;
        int end_user_token_id;
        int end_assistant_token_id;
        int vision_start_token_id;
        int vision_end_token_id;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t size = do_load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        pad_token_id        = tp->PieceToId("[PAD]");
        eos_token_id        = tp->PieceToId("<|endoftext|>");

        user_token_id       = tp->PieceToId("<|user|>");
        assistant_token_id  = tp->PieceToId("<|assistant|>");
        end_user_token_id       = tp->PieceToId("<|endofuser|>");
        end_assistant_token_id  = tp->PieceToId("<|endofassistant|>");
        vision_start_token_id   = tp->PieceToId("<|vision_start|>");
        vision_end_token_id     = tp->PieceToId("<|vision_end|>");

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        terminate_ids.insert(end_assistant_token_id);
        terminate_ids.insert(end_user_token_id);

        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, int prefix_id, int suffix_id) const
    {
        ids.push_back(prefix_id);
        qwen::v1::Tokenizer::encode(text, ids);
        ids.push_back(suffix_id);
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(vision_start_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(vision_end_token_id);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode(ai, ids, tok->assistant_token_id, tok->end_assistant_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (tok->get_system_prompt().size() == 0) return;
        std::ostringstream oss;
        oss << "<|system|>" << tok->get_system_prompt() << "<|endofsystem|>\n";
        tok->encode(oss.str(), ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode(user, ids, tok->user_token_id, tok->end_user_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->user_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_user_opening(round_idx, ids);

        tok->media_emb.clear();

        std::unique_ptr<vision::Resize> resize;

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

                vision::MinMaxPixels    param1(vis_config->min_pixels, vis_config->max_pixels);
                vision::MergeKernel     param2(vis_config->spatial_merge_size, vis_config->spatial_merge_size);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
                if ((w <= 0) || (h <= 0)) continue;

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_MergeN_ChannelsRGB_PixelsLeftRightDown);

                const int merge_length = vis_config->spatial_merge_size * vis_config->spatial_merge_size;
                image.emb_vec_number = image.grid_width * image.grid_height;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_media("image", ids, id_start, image.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
        ids.push_back(tok->end_user_token_id);
    }

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int n) : pad_arg(new BlockParams::PadEmbedding(n, n)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public qwen::v2::ConditionalGeneration
    {
    public:
        typedef qwen::v2::ConditionalGeneration Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
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
        const Config config;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config):
        ExtendEmbedding(14400),
        Base(config, runtime_config, ModelType::MODEL_TYPE_DOTS_OCR, false),
        max_patches(pad_arg->get()),
        vis_eval(runtime_config, "vis"),
        vis_ctx(&backend_context),
        config(config)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        qwen::v2::ConditionalGeneration::load_more(config);

        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vit::Config vis_config;

        vis_config.dtype = this->config.dtype;

        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
        vis_config.spatial_merge_size   = (int)vis_cfg["spatial_merge_size"].ToInt();
        vis_config.temporal_patch_size  = (int)vis_cfg["temporal_patch_size"].ToInt();

        CHATLLM_CHECK(vis_config.temporal_patch_size == 1);

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

            vis_config.min_pixels       = (int)pp_cfg["min_pixels"].ToInt();
            vis_config.max_pixels       = (int)pp_cfg["max_pixels"].ToInt();
            vis_config.max_patches      = vis_config.max_pixels / (vis_config.patch_size * vis_config.patch_size);
            if (vis_config.max_patches > max_patches)
            {
                vis_config.max_patches      = max_patches;
                vis_config.max_pixels       = max_patches * vis_config.patch_size * vis_config.patch_size;
            }
        }

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 10 + vis_config.num_hidden_layers * 10;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        vis_ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        vis_ctx.dtype = vis_config.dtype;

        vision.reset(new vit::VisionTransformer(&vis_ctx, vis_config, this->config.hidden_size));

        vis_ctx.check_used_mem_size(true);

        return true;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        qwen::v2::ConditionalGeneration::load(loader);

        loader.add_tensor_name_translations({
            {".self_attn.",                 ".attn."},
            {".input_layernorm.",           ".norm1."},
            {".post_attention_layernorm.",  ".norm2."},
        });

        if (vision.get() == nullptr) return;

        vision->load("vision_model.", &loader);
        if (vision->is_loaded())
        {
            _chat_encoder.vis_config = &vision->config;
        }
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = Base::get_param_num(effective_only);
        if (vision)       r += vision->get_param_num(effective_only);
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
                    ggml::type::GGML_TYPE_F32, vis_config.patch_size, vis_config.patch_size, 3, image.grid_width * image.grid_height);
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

    REGISTER_MODEL_LOADER(DOTS_OCR,               dots::ocr, 1);
}