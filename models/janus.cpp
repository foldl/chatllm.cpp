#include <cstring>
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

        int gen_head_n_embed;
        int gen_head_image_token_embed;
        int gen_head_image_token_size;
        int gen_vision_n_embed;
    };

    struct ModelArgs
    {
        int codebook_size;
        int codebook_embed_dim;
        bool codebook_l2_norm;
        bool codebook_show_usage;
        float commit_loss_beta;
        float entropy_loss_ratio;

        std::vector<int> encoder_ch_mult;
        std::vector<int> decoder_ch_mult;
        int z_channels;
    };

    const ModelArgs ARGS_VQ_16 =
    {
        .codebook_size = 16384,
        .codebook_embed_dim = 8,
        .codebook_l2_norm = true,
        .codebook_show_usage = true,
        .commit_loss_beta = 0.25f,
        .entropy_loss_ratio = 0.0f,

        .encoder_ch_mult = {1, 1, 2, 2, 4},
        .decoder_ch_mult = {1, 1, 2, 2, 4},
        .z_channels = 256,
    };

    class ChatHistoryEncoder : public deepseek::v1::ChatHistoryEncoder
    {
    public:
        typedef deepseek::v1::ChatHistoryEncoder Base;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    public:
        const MultiModalConfig *vis_config = nullptr;
        bool vit_loaded = false;
        bool gen_token_padding = false;
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

    const float NORM_EPS    = 1e-6f;

    class GenHead : public TheBiasedGELUMLP
    {
    private:
        typedef TheBiasedGELUMLP Base;
    public:
        GenHead(InitContext *ctx, int n_embed, int image_token_embed, int image_token_size)
            : Base(ctx, n_embed, image_token_embed, image_token_size)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            ggml::tensor *logits = Base::forward(ctx, hidden_states);
            ggml::tensor *logit_cond   = ggml::view_2d(ctx, logits, ggml::get_dim(logits, 0), ggml::get_dim(logits, 1) / 2,
                        ggml::row_size(logits) * 2, 0);
            ggml::tensor *logit_uncond = ggml::view_2d(ctx, logits, ggml::get_dim(logits, 0), ggml::get_dim(logits, 1) / 2,
                        ggml::row_size(logits) * 2, ggml::row_size(logits));

            auto sub = ggml::sub(ctx,logit_cond, logit_uncond);
            sub = ggml::scale(ctx, sub, cfg_weight);
            logits = ggml::add(ctx, logit_uncond, sub);
            logits = ggml::scale(ctx, logits, 1/temperature);
            logits = ggml::soft_max(ctx, logits);
            return logits;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            fc0.load(path + "output_mlp_projector.", loader);
            fc1.load(path + "vision_head.", loader);
        }
    public:
        float cfg_weight = 5.0f;
        float temperature = 1.0f;
    };

    class GenImageEmbed : public Block
    {
    public:
        GenImageEmbed(InitContext *ctx, int image_token_size, int n_embed, int lm_hidden_size)
            : Block(),
              embedding(ctx, image_token_size, n_embed),
              aligner(ctx, n_embed, lm_hidden_size, lm_hidden_size)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            auto output = embedding.forward(ctx, input);
            output = aligner.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += embedding.get_param_num(effective_only);
            r += aligner.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            embedding.load("gen_embed.", loader);
            aligner.load("gen_aligner.", loader);
        }
    public:
        Embedding embedding;
        TheBiasedGELUMLP aligner;
    };

    class VectorQuantizer : public Embedding
    {
    public:
        VectorQuantizer(InitContext *ctx, int n_e, int e_dim, bool l2_norm)
            : Embedding(ctx, n_e, e_dim), l2_norm(l2_norm)
        {}

        ggml::tensor *get_codebook_entry(ComputeContext *ctx, ggml::tensor *input,
            const int64_t *shape = nullptr, bool channel_first = true)
        {
            ggml::tensor *r = Embedding::forward(ctx, input);
            if (l2_norm)
                r = ggml::norm_p2(ctx, r, NORM_EPS);
            if (shape)
            {
                // shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
                if (channel_first)
                {
                    r = ggml::reshape_4d(ctx, r, shape[1], shape[3], shape[2], shape[0]);

                    // reshape back to match original input shape
                    r = ggml::permute(ctx, r, 2, 0, 1, 3);
                    r = ggml::cont(ctx, r);
                }
                else
                {
                    r = ggml::reshape_4d(ctx, r, shape[3], shape[2], shape[1], shape[0]);
                }
            }

            return r;
        }
    public:
        const bool l2_norm;
    };

    class AttnBlock : public Block
    {
    public:
        AttnBlock(InitContext *ctx, int in_channels) :
            Block(),
              norm(ctx, 32, in_channels, true, NORM_EPS),
            q_proj(ctx, in_channels, in_channels, 1, 1, 0),
            k_proj(ctx, in_channels, in_channels, 1, 1, 0),
            v_proj(ctx, in_channels, in_channels, 1, 1, 0),
            o_proj(ctx, in_channels, in_channels, 1, 1, 0)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x) override
        {
            ggml::tensor *h_ =   norm.forward(ctx, x);
            ggml::tensor *q  = q_proj.forward(ctx, h_);
            ggml::tensor *k  = k_proj.forward(ctx, h_);
            ggml::tensor *v  = v_proj.forward(ctx, h_);

            const int64_t b = ggml::get_dim(q, 3);
            const int64_t c = ggml::get_dim(q, 2);
            const int64_t h = ggml::get_dim(q, 1);
            const int64_t w = ggml::get_dim(q, 0);
            q = ggml::reshape(ctx, q, h * w, c, b);
            q = ggml::permute(ctx, q, 1, 0, 2);
            q = ggml::cont(ctx, q);
            k = ggml::reshape(ctx, k, h * w, c, b);
            k = ggml::permute(ctx, k, 1, 0, 2);
            k = ggml::cont(ctx, k);

            ggml::tensor *w_ = ggml::mul_mat(ctx, k, q);
            w_ = ggml::scale(ctx, w_, 1.0f / sqrtf((float)c));
            w_ = ggml::soft_max(ctx, w_);
            v  = ggml::reshape(ctx, v, h * w, c, b);
            h_ = ggml::mul_mat(ctx, w_, v);
            h_ = ggml::reshape(ctx, h_, h, w, c, b);

            h_ = o_proj.forward(ctx, h_);

            return ggml::add(ctx, x, h_);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r +=   norm.get_param_num(effective_only);
            r += q_proj.get_param_num(effective_only);
            r += k_proj.get_param_num(effective_only);
            r += v_proj.get_param_num(effective_only);
            r += o_proj.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
              norm.load(path + "norm.", loader);
            q_proj.load(path + "q_proj.", loader);
            k_proj.load(path + "k_proj.", loader);
            v_proj.load(path + "v_proj.", loader);
            o_proj.load(path + "o_proj.", loader);
        }

    public:
        GroupNorm norm;
        Conv2D q_proj;
        Conv2D k_proj;
        Conv2D v_proj;
        Conv2D o_proj;
    };

    class ResnetBlock : public Block
    {
    public:
        ResnetBlock(InitContext *ctx, int in_channels, int out_channels = -1):
            Block(),
            out_channels(out_channels > 0 ? out_channels : in_channels),
            norm1(ctx, 32, in_channels),
            norm2(ctx, 32, this->out_channels),
            conv1(ctx, in_channels, this->out_channels, 3, 1, 1),
            conv2(ctx, this->out_channels, this->out_channels, 3, 1, 1),
            nin_shortcut(in_channels != this->out_channels ? new Conv2D(ctx, in_channels, this->out_channels, 1, 1) : nullptr)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x) override
        {
            ggml::tensor *h = norm1.forward(ctx, x);
            h = ggml::act(ctx, ActFunc::SWISH, h);
            h = conv1.forward(ctx, h);
            h = norm2.forward(ctx, h);
            h = ggml::act(ctx, ActFunc::SWISH, h);
            h = conv2.forward(ctx, h);

            if (nin_shortcut.get())
            {
                x = nin_shortcut->forward(ctx, x);
            }
            ggml::tensor *r = ggml::add(ctx, x, h);

            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += norm1.get_param_num(effective_only);
            r += norm2.get_param_num(effective_only);
            r += conv1.get_param_num(effective_only);
            r += conv2.get_param_num(effective_only);
            if (nin_shortcut.get()) r += nin_shortcut->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            norm1.load(path + "norm1.", loader);
            norm2.load(path + "norm2.", loader);
            conv1.load(path + "conv1.", loader);
            conv2.load(path + "conv2.", loader);
            if (nin_shortcut.get()) nin_shortcut->load(path + "nin_shortcut.", loader);
        }
    public:
        const int out_channels;
        GroupNorm norm1;
        GroupNorm norm2;
        Conv2D    conv1;
        Conv2D    conv2;
        std::unique_ptr<Conv2D> nin_shortcut;
    };

    class Upsample : public Block
    {
    public:
        Upsample(InitContext *ctx, int in_channels, bool with_conv, float scale_factor = 2.0f):
            scale_factor(scale_factor),
            conv(with_conv ? new Conv2D(ctx, in_channels, in_channels, 3, 1, 1) : nullptr)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x) override
        {
            x = ggml::interpolate(ctx, x, ggml::InterpolateMode::Nearest, scale_factor);
            if (conv.get())
                x = conv->forward(ctx, x);
            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            if (conv.get()) r += conv->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            if (conv.get()) conv->load(path + "conv.", loader);
        }

    public:
        const float scale_factor;
        std::unique_ptr<Conv2D> conv;
    };

    class ConvBlock : public Block
    {
    public:
        ConvBlock() {}
        ConvBlock(InitContext *ctx) {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            ggml::tensor *h = input;
            for (int i = 0; i < (int)res.blocks.size(); i++)
            {
                h = res.blocks[i]->forward(ctx, h);
                if (attn.blocks.size() > 0)
                {
                    h = attn.blocks[i]->forward(ctx, h);
                }
            }
            if (upsample.get())
            {
                h = upsample->forward(ctx, h);
            }
            return h;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += res.get_param_num(effective_only);
            r += attn.get_param_num(effective_only);
            if (upsample.get()) r += upsample->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            res.load(path + "res.", loader);
            attn.load(path + "attn.", loader);
            if (upsample.get()) upsample->load(path + "upsample.", loader);
        }

    public:
        Sequential res;
        Sequential attn;
        std::unique_ptr<Upsample> upsample;
    };

    class Decoder : public Block
    {
    private:

    public:
        Decoder(InitContext *ctx, const std::vector<int> &ch_mult,
                int z_channels = 256,
                int ch = 128,
                int num_res_blocks = 2,
                bool resamp_with_conv = true,
                int out_channels = 3) :
            Block(),
            out_channels(out_channels),
            num_resolutions((int)ch_mult.size()),
            num_res_blocks(num_res_blocks),
            conv_in(ctx, z_channels, ch * ch_mult.back(), 3, 1, 1),
            norm_out(ctx, 32, ch * ch_mult.front()),
            conv_out(ctx, ch * ch_mult.front(), out_channels, 3, 1, 1)
        {
            int block_in = ch * ch_mult.back();
            mid.add_block(new ResnetBlock(ctx, block_in, block_in));
            mid.add_block(new AttnBlock(ctx, block_in));
            mid.add_block(new ResnetBlock(ctx, block_in, block_in));

            for (int i_level = num_resolutions - 1; i_level >= 0; i_level--)
            {
                auto conv_block = new ConvBlock();
                const int block_out = ch * ch_mult[i_level];
                for (int j = 0; j <= num_res_blocks; j++)
                {
                    conv_block->res.add_block(new
                        ResnetBlock(ctx, block_in, block_out));
                    block_in = block_out;
                    if (i_level == num_resolutions - 1)
                    {
                        conv_block->attn.add_block(new
                            AttnBlock(ctx, block_in));
                    }
                }

                if (i_level != 0)
                    conv_block->upsample.reset(new Upsample(ctx, block_in, resamp_with_conv));

                conv_block->set_id(i_level);
                conv_blocks.add_block(conv_block);
            }
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += conv_in.get_param_num(effective_only);
            r += mid.get_param_num(effective_only);
            r += conv_blocks.get_param_num(effective_only);
            r += norm_out.get_param_num(effective_only);
            r += conv_out.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            conv_in.load(path + "conv_in.", loader);
            mid.load(path + "mid.", loader);
            conv_blocks.load(path + "conv_blocks.", loader);
            norm_out.load(path + "norm_out.", loader);
            conv_out.load(path + "conv_out.", loader);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            ggml::tensor *h = conv_in.forward(ctx, input);
            h = mid.forward(ctx, h);
            h = conv_blocks.forward(ctx, h);
            h = norm_out.forward(ctx, h);
            h = ggml::act(ctx, ActFunc::SWISH, h);
            h = conv_out.forward(ctx, h);
            return h;
        }

    public:
        const int out_channels;
        const int num_resolutions;
        const int num_res_blocks;
        Conv2D conv_in;
        Sequential mid;
        Sequential conv_blocks;
        GroupNorm norm_out;
        Conv2D conv_out;
    };

    class VQModel : public Block
    {
    public:
        VQModel(InitContext *ctx, const ModelArgs &args)
            : Block(),
              quantize(ctx, args.codebook_size, args.codebook_embed_dim, args.codebook_l2_norm),
              decoder(ctx, args.decoder_ch_mult, args.z_channels),
              post_quant_conv(ctx, args.codebook_embed_dim, args.z_channels, 1, 1)
        {}

        ggml::tensor *decode_code(ComputeContext *ctx, ggml::tensor *input, const int64_t *shape = nullptr, bool channel_first = true)
        {
            auto quant_b = quantize.get_codebook_entry(ctx, input, shape, channel_first);
            auto quant   = post_quant_conv.forward(ctx, quant_b);
            auto dec     = decoder.forward(ctx, quant);
            return dec;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += quantize.get_param_num(effective_only);
            r += decoder.get_param_num(effective_only);
            r += post_quant_conv.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            quantize.load(path + "quantize.embedding.", loader);
            decoder.load(path + "decoder.", loader);
            post_quant_conv.load(path + "post_quant_conv.", loader);
        }
    public:
        VectorQuantizer quantize;
        Decoder decoder;
        Conv2D post_quant_conv;
    };

    class ConditionalGeneration : public ExtendEmbedding, public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        typedef llama::v2::GenericConditionalGeneration<LlamaBlock> Base;
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
    public:
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override;
        void before_generate(const GenerationConfig &gen_config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
    protected:
        void decode_images(const GenerationConfig &gen_config, const std::vector<std::vector<int>> &generated_tokens);
    public:
        const Config config;
        MultiModalConfig mm_config;
        siglip::VisualEmbeddingGeneration vision;
        std::unique_ptr<GenHead> gen_head;
        std::unique_ptr<GenImageEmbed> gen_image_emb;
        std::unique_ptr<VQModel> gen_vision;
        int parallel_size = 2;
        const int image_token_num_per_image = 576;
    protected:
        TensorGraphEvaluator eval;
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
        pad_token_id                = tp->PieceToId("<｜▁pad▁｜>");
        return r;
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(image_start_tag_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(ids_to_inject_start + i);
        ids.push_back(image_end_tag_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        if (!user.starts_with("/gen "))
        {
            Base::append_user(round_idx, user, ids);
            return;
        }

        std::string s = user.substr(5);
        Base::append_user(round_idx, s, ids);
        ((ChatHistoryEncoder *)(this))->gen_token_padding = true;
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        Base::append_ai_opening(round_idx, ids);
        if (gen_token_padding)
        {
            ids.push_back(tok->image_start_tag_token_id);
            ((ChatHistoryEncoder *)(this))->gen_token_padding = false;
        }
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
          Base(config, runtime_config, MODEL_TYPE_JANUS_PRO, config.num_attention_heads,
               config.max_length, 12, false, 4 + 5 + 197),
          config(config),
          vision(runtime_config),
          eval(runtime_config)
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

        auto &gen_head_cfg = config["config.json"]["gen_head_config"]["params"];
        mm_config.gen_head_n_embed               = (int)gen_head_cfg["n_embed"].ToInt();
        mm_config.gen_head_image_token_embed     = (int)gen_head_cfg["image_token_embed"].ToInt();
        mm_config.gen_head_image_token_size      = (int)gen_head_cfg["image_token_size"].ToInt();

        auto &gen_vision_cfg = config["config.json"]["gen_vision_config"]["params"];
        mm_config.gen_vision_n_embed            = (int)gen_vision_cfg["n_embed"].ToInt();

        CHATLLM_CHECK(config["config.json"]["gen_vision_config"]["cls"].ToString() == "VQ-16");

        bool r = vision.load_model(vis_config);

        if (r)
        {
            _chat_encoder.vis_config = &mm_config;

            gen_head.reset(new GenHead(&w_ctx_,
                mm_config.gen_head_n_embed,
                mm_config.gen_head_image_token_embed,
                mm_config.gen_head_image_token_size));
            gen_image_emb.reset(new GenImageEmbed(&w_ctx_,
                mm_config.gen_head_image_token_size,
                mm_config.gen_vision_n_embed,
                this->config.hidden_size));
            gen_vision.reset(new VQModel(&w_ctx_, ARGS_VQ_16));
        }

        w_ctx_.check_used_mem_size(true);

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
        if (gen_head) gen_head->load("gen_head.", &loader);
        if (gen_image_emb) gen_image_emb->load("", &loader);
        if (gen_vision) gen_vision->load("gen_vision_model.", &loader);
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        parallel_size       = utils::get_opt(args, "parallel_size", parallel_size);
        if (parallel_size < 1) parallel_size = 1;
        transformer->reserve_batch_size(parallel_size * 2);
        if (gen_head.get())
            gen_head->temperature = (float)utils::get_opt(args, "gen-head-temperature", gen_head->temperature);
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

    bool ConditionalGeneration::generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (input_ids.back() != tok->image_start_tag_token_id)
            return Base::generate_next_token(input_ids, gen_config, lm_logits);

        if (parallel_size < 1) parallel_size = 1;

        std::vector<int> para_input_ids;
        std::vector<std::vector<int>> generated_tokens;
        generated_tokens.resize(parallel_size);

        para_input_ids.resize(input_ids.size() * 2);
        {
            int *p_cond  = para_input_ids.data();
            memcpy(p_cond, input_ids.data(), input_ids.size() * sizeof(input_ids[0]));

            p_cond += input_ids.size();
            p_cond[0] = input_ids[0];
            p_cond[input_ids.size() - 1] = input_ids.back();
            for (int j = 1; j <= (int)input_ids.size() - 2; j++)
                p_cond[j] = tok->pad_token_id;
        }

        const int batch = batch_input;
        const int *p = para_input_ids.data();
        int remain = (int)input_ids.size();
        int past = n_past + n_past_offset;

        std::vector<float> hidden_state;

        bool skipping = transformer->skip_lm_head;
        transformer->skip_lm_head = true;
        bool r = true;

        const auto custom_embedding = [this](ComputeContext *ctx, ggml::tensor *input)
        {
            auto r = gen_image_emb->forward(ctx, input);
            return r;
        };

        const auto call_head = [this](ComputeContext *ctx, ggml::tensor *hidden_states)
        {
            auto r = gen_head->forward(ctx, hidden_states);
            return r;
        };

        for (; (remain > batch) && !aborted; p += batch, remain -= batch, past += batch)
        {
            if (!run_model(p, batch, gen_config, past, hidden_state, 2))
            {
                r = false;
                goto exit;
            }
        }

        r = run_model(p, remain, gen_config, past, hidden_state, 2, call_head);
        past += remain;
        if (!r) goto exit;

        // fill data for all parallels
        if (parallel_size > 1)
        {
            const auto size = hidden_state.size();
            hidden_state.resize(size * parallel_size);
            float *src = hidden_state.data();
            for (int i = 1; i < parallel_size; i++)
            {
                memcpy(src + i * size, src, size * sizeof(float));
            }
        }

        transformer->custom_embedding = custom_embedding;
        para_input_ids.resize(1 * parallel_size * 2);
        p = para_input_ids.data();

        for (int i = 0; i < image_token_num_per_image; i++)
        {
            std::vector<int> next_token;
            next_token.resize(parallel_size, 0);
            utils::multinomial(hidden_state.data(), next_token.data(), mm_config.gen_head_image_token_size, 1, parallel_size);

            for (int j = 0; j < parallel_size; j++)
            {
                para_input_ids[2 * j + 0] = next_token[j];
                para_input_ids[2 * j + 1] = next_token[j];
                generated_tokens[j].push_back(next_token[j]);
            }

            if (i == image_token_num_per_image - 1) break;

            run_model(p, 1, gen_config, past, hidden_state, parallel_size * 2, call_head);
            past++;
        }

        decode_images(gen_config, generated_tokens);

    exit:
        transformer->skip_lm_head = skipping;
        transformer->custom_embedding = nullptr;
        n_past = past - n_past_offset;
        return r;
    }

    void ConditionalGeneration::decode_images(const GenerationConfig &gen_config, const std::vector<std::vector<int>> &generated_tokens)
    {
        ggml::tensor *input_ids;
        const auto make_graph = [this, &input_ids, &generated_tokens](ComputeContext *ctx) -> ggml::tensor * {
            const int parallel_size = (int)generated_tokens.size();
            const int64_t shape[4] = {parallel_size, 8, mm_config.image_size / mm_config.image_patch_size, mm_config.image_size / mm_config.image_patch_size};
            input_ids = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, generated_tokens[0].size(), parallel_size);
            auto r = gen_vision->decode_code(ctx, input_ids, shape);

            // ggml shape: [w, h, c, b] -> [c, w, h, b]
            r = ggml::permute(ctx, r, 1, 2, 0, 3);
            r = ggml::cont(ctx, r);
            r = ggml::scale(ctx, r, 1.0f / 2, 1.0f / 2);
            return r;
        };
        const auto write_input_data = [this, &input_ids, &generated_tokens](ComputeContext *ctx) {
            const int parallel_size = (int)generated_tokens.size();
            int64_t offset = 0;
            for (int i = 0; i < parallel_size; i++)
            {
                size_t size = generated_tokens[i].size() * sizeof(int32_t);
                Backend::write_tensor_data(input_ids, generated_tokens[i].data(), offset, size);
                offset += size;
            }
        };
        std::vector<int64_t> shape;
        std::vector<uint8_t> buffer;
        eval.evaluate(gen_config, make_graph, write_input_data, ggml::type::GGML_TYPE_F32, shape, buffer);
        const float *p = (const float *)buffer.data();
        const size_t data_len = mm_config.image_size * mm_config.image_size * 3;
        for (int i = 0; i < parallel_size; i++, p += data_len)
        {
            vision::image_view(p, data_len,
                mm_config.image_size, vision::ImageDataFormat::PixelsLeftRightDown_ChannelRGB);
        }
    }
}

namespace chatllm::janus
{
    REGISTER_MODEL_LOADER(JANUS_PRO,           pro, 1);
}