#include <ostream>
#include <iomanip>

#include "qwen.h"
#include "moonshot.h"
#include "../src/vision_process.h"

namespace chatllm::locate_anything::vit
{
    using kimi::vit::Config;

    class MultiModalProjector : public Sequential
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size);

        using Sequential::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w) override;
    public:
        ggml::tensor *merge_patch(ComputeContext *ctx, ggml::tensor *x);
        ggml::merge_patch_param merge_param;
    };

    MultiModalProjector::MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size):
        Sequential(ctx)
    {
        merge_param.merge_kernel_size[0] = config.merge_kernel_size[0];
        merge_param.merge_kernel_size[1] = config.merge_kernel_size[1];

        const int vit_merge_size = config.hidden_size * config.merge_kernel_size[0] * config.merge_kernel_size[1];
        add_block(new LayerNorm(ctx, vit_merge_size));
        add_block(new Linear(ctx, vit_merge_size, lm_hidden_size));
        add_block(new ActivationBlock(ActFunc::GELU));
        add_block(new Linear(ctx, lm_hidden_size, lm_hidden_size));
    }

    ggml::tensor *MultiModalProjector::merge_patch(ComputeContext *ctx, ggml::tensor *x)
    {
        const int merge = merge_param.merge_kernel_size[0] * merge_param.merge_kernel_size[1];
        auto reshaped = ggml::merge_patch(ctx, x, &merge_param);
        reshaped = ggml::reshape_4d(ctx, reshaped,
                ggml::get_dim(reshaped, 0) * merge, ggml::get_dim(reshaped, 1) / merge,
                ggml::get_dim(reshaped, 2), ggml::get_dim(reshaped, 3));

        return reshaped;
    }

    ggml::tensor *MultiModalProjector::forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w)
    {
        merge_param.grid_h = grid_h;
        merge_param.grid_w = grid_w;

        auto output = merge_patch(ctx, image_features);

        output = forward(ctx, output);
        return output;
    }

    class VisualEmbeddingGeneration : public kimi::vit::VisualEmbeddingGeneration
    {
    public:
        using kimi::vit::VisualEmbeddingGeneration::VisualEmbeddingGeneration;
    protected:
        DynamicBlock *do_create_transformer(InitContext *ctx, const Config &config, const int lm_hidden_size) override
        {
            return new kimi::vit::VisionTransformer(ctx, config, lm_hidden_size,
                new MultiModalProjector(ctx, config, lm_hidden_size));
        }
    };
}

namespace chatllm::locate_anything
{
    struct Config : qwen::v2::Config
    {
        int tie_word_embeddings;
    };

    class ChatHistoryEncoder : public qwen::v1::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    public:
        const vit::Config *vis_config = nullptr;
        bool vit_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public qwen::v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder = nullptr);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void inject_media(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int image_start_token_id;
        int image_end_token_id;
        bool native_resolution = false;
        double fps = 1.0;
        int video_max_frames = 20;
    };

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder):
        qwen::v2::Tokenizer(config, encoder ? encoder : &_chat_encoder)
    {
        sys_prompt = "You are a helpful assistant.";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = qwen::v2::Tokenizer::load(buffer, n_vocab);

        image_start_token_id = tp->PieceToId("<img>");
        image_end_token_id   = tp->PieceToId("</img>");

        return r;
    }

    void Tokenizer::inject_media(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(image_start_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(ids_to_inject_start + i);
        ids.push_back(image_end_token_id);
    }

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding() : pad_arg(new BlockParams::PadEmbedding(4096, 4096)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public qwen::v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LocateAnything);
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void before_generate(const GenerationConfig &gen_config) override;
    public:
        vit::VisualEmbeddingGeneration visual;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        ExtendEmbedding(),
        qwen::v2::ConditionalGeneration(config, runtime_config, type, config.tie_word_embeddings != 0),
        visual(runtime_config)
    {
        delete pad_arg;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        BaseModelForConditionalGeneration::load_more(config);
        bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config);
        if (r)
        {
            _chat_encoder.vis_config = &visual.vis_config;
        }
        return r;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        qwen::v2::ConditionalGeneration::load(loader);

        loader.add_tensor_name_translations({
            {".self_attn.",                         ".attn."},
            {".input_layernorm.",                   ".norm0."},
            {".post_attention_layernorm.",          ".norm1."},
            {"multi_modal_projector.",              "mlp1."},
        });

        _chat_encoder.vit_loaded = visual.load(loader);
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->video_max_frames       = utils::get_opt(args, "video_max_frames", tok->video_max_frames);
        tok->native_resolution      = utils::get_opt(args, "native_resolution", tok->native_resolution);
        tok->fps                    = utils::get_opt(args, "fps", tok->fps);
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

        std::unique_ptr<vision::Resize> resize;
        std::unique_ptr<vision::PreMaxImageSize> max_size;

        if (!tok->native_resolution)
            resize.reset(new vision::Resize(896, 896));

        int image_counter = 0;
        int video_counter = 0;

        // expand video into images
        std::vector<ContentPiece> pieces;
        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                pieces.push_back(piece);
                continue;
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                std::ostringstream oss;
                oss << "<image " << ++image_counter << ">";
                pieces.push_back(oss.str());
                pieces.push_back(piece);
                continue;
            }
            else if (piece.type != ContentPiece::Type::Video)
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
                break;
            }

            auto video = new vision::VideoLoader(piece.content.c_str(), (float)tok->fps, tok->video_max_frames);
            videos.emplace_back(video);
            if (video->frames.size() < 1)
                continue;

            std::ostringstream oss;
            oss << "The " << utils::index2words(++video_counter) << "video sampled with " << std::fixed << std::setprecision(2) << tok->fps << " fps: ";
            pieces.push_back(oss.str());

            if (max_size.get() == nullptr)
                max_size.reset(new vision::PreMaxImageSize(448, 448));

            for (size_t i = 0; i < video->frames.size(); i++)
            {
                std::ostringstream oss;
                oss << "Frame-" << i + 1 << ": ";
                pieces.push_back(oss.str());
                pieces.emplace_back(video->frames[i], ContentPiece::Type::Image);
            }
        }

        for (auto &piece : pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
                continue;
            }

            // now, it's image
            CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

            int w, h;
            std::vector<uint8_t> pixels;
            const int patch_size = vis_config->patch_size;

            vision::MergeKernel     param1(vis_config->merge_kernel_size);
            vision::MaxPatchNum     param2(vis_config->in_token_limit);
            vision::MaxGridHeight   param3(512);
            vision::MaxGridWidth    param4(512);

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
            tok->inject_media(ids, id_start, image.emb_vec_number);
        }
        ids.push_back(tok->im_end_token_id);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(LocateAnything,                       locate_anything, 1);
}