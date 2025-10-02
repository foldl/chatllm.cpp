#include "deepseek.h"
#include "siglip.h"
#include "../src/vision_process.h"

namespace chatllm::janus::pro
{
    typedef deepseek::v1::Config Config;

    struct MultiModalConfig
    {
        int image_patch_size;
        int image_size;
        float image_mean[3];
        float image_std[3];
    };

    class ChatHistoryEncoder : public deepseek::v1::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    public:
        const MultiModalConfig *vis_config = nullptr;
        bool vit_loaded = false;
    };

    class Tokenizer : public deepseek::v1::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int image_start_tag_token_id;
        int image_end_tag_token_id;
        int user_token_id;
        int assistant_token_id;
    };

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int n) : pad_arg(new BlockParams::PadEmbedding(n, n)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public deepseek::v1::ConditionalGeneration
    {
    public:
        typedef deepseek::v1::ConditionalGeneration Base;
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
    public:
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void before_generate(const GenerationConfig &gen_config) override;
    public:
        const Config config;
        MultiModalConfig mm_config;
        siglip::VisualEmbeddingGeneration vision;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config) : deepseek::v1::Tokenizer(config, &_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = deepseek::v1::Tokenizer::load(buffer, n_vocab);
        bos_token_id = tp->PieceToId("<｜begin▁of▁sentence｜>");
        eos_token_id = tp->PieceToId("<｜end▁of▁sentence｜>");
        image_start_tag_token_id    = tp->PieceToId("<begin_of_image>");
        image_end_tag_token_id      = tp->PieceToId("<end_of_image>");
        user_token_id               = tp->PieceToId("<|User|>");
        assistant_token_id          = tp->PieceToId("<|Assistant|>");
        return r;
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(image_start_tag_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(ids_to_inject_start + i);
        ids.push_back(image_end_tag_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user_opening(round_idx, ids);

        bool last_is_image = false;

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                if (last_is_image && !piece.content.starts_with("\n"))
                    tok->encode("\n", ids);
                tok->encode(piece.content, ids);
                last_is_image = false;
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";
                last_is_image = true;

                int w, h;
                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->image_patch_size;

                vision::Resize      param3(vis_config->image_size, vis_config->image_size);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Gray);

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.width = w, .height = h, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);

                image.emb_vec_number = (vis_config->image_size / patch_size) * (vis_config->image_size / patch_size);

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_media("image", ids, id_start, image.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        tok->encode("\n\n", ids, false, false);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : ExtendEmbedding(24 * 24),
          deepseek::v1::ConditionalGeneration(config, runtime_config), config(config),
          vision(runtime_config)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);

        const auto vis_cfg = config["config.json"]["vision_config"];

        if (vis_cfg["params"]["model_name"].ToString() != "siglip_large_patch16_384")
            return false;

        siglip::Config vis_config;

        vis_config.dtype = this->config.dtype;
        vis_config.image_size = 384;
        vis_config.patch_size = 16;;
        vis_config.width      = 1024;
        vis_config.layers     = 24;
        vis_config.heads      = 16;;
        vis_config.mlp_ratio  = 4.0f;
        vis_config.global_pool= siglip::CONFIG_GLOBAL_POOL_NONE;
        vis_config.lm_hidden_size = this->config.hidden_size;

        mm_config.image_size = vis_config.image_size;
        mm_config.image_patch_size = vis_config.patch_size;
        mm_config.image_mean[0] = 0.5f;
        mm_config.image_mean[1] = 0.5f;
        mm_config.image_mean[2] = 0.5f;
        mm_config.image_std[0] = 0.5f;
        mm_config.image_std[1] = 0.5f;
        mm_config.image_std[2] = 0.5f;

        bool r = vision.load_model(vis_config);

        if (r)
            _chat_encoder.vis_config = &mm_config;

        return true;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);
        loader.add_tensor_name_translations({
            {".pos_embed.weight",       ".pos_embed"},
            {".patch_embed.",           ".patch_embed.proj."},
            {".self_attn.",             ".attn."},
            {".input_layernorm.",       ".norm1."},
            {".post_attention_layernorm.",          ".norm2."},
        });

        _chat_encoder.vit_loaded = vision.load("vision_model.", loader);
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        vision.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }
}

namespace chatllm::janus
{
    REGISTER_MODEL_LOADER(JANUS_PRO,           pro, 1);
}