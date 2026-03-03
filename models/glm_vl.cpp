#include "chatglm.h"
#include <regex>
#include <codecvt>
#include "qwen.h"
#include "llama.h"
#include "deepseek.h"
#include "../src/vision_process.h"
#include "../src/chat_encoders.h"
#include "../src/audio_process.h"

namespace chatllm::glm::vit
{
    struct Config
    {
        ggml::type dtype;
        bool is_ocr_vision;
        int max_length;
        int image_ref_size;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int out_hidden_size;
        int spatial_merge_size;
        int temporal_patch_size;
        int mlp_intermediate_size;

        float image_mean[3];
        float image_std[3];
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, const Config &config, const bool use_pos_emb);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input0, ggml::tensor *input1, int grid_h, int grid_w);

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const bool use_pos_emb;
        const int merge_size;
        const int max_patches;
        const int ref_w;
        const int ref_h;
        Conv2D                      proj0;
        Conv2D                      proj1;
        ggml::tensor               *proj_bias;
        std::unique_ptr<RMSNorm>    post_conv_layernorm;
        std::unique_ptr<Embedding>  position_embedding;
    };

    using qwen::vit::TensorPosHelper;

    class ViTSelfAttention : public qwen::vit::ViTSelfAttention
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length):
            qwen::vit::ViTSelfAttention(ctx, hidden_size, num_attention_heads, max_length, false, true)
        {}
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, int hidden_size, int intermediate_size, int spatial_merge_size, int lm_hidden_size, int mlp_intermediate_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w) override;
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int spatial_merge_size;
        const int hidden_size;
        RMSNorm     post_layernorm;
        Conv2D      downsample;
        Linear      proj;
        LayerNorm   post_projection_norm;
        SiLUMLP     mlp;
    };

    template <class ViTSelfAttention, class MLP, bool use_pos_emb> class VisionTransformer : public DynamicBlock
    {
    public:
        typedef LMBlock1<RMSNorm, ViTSelfAttention, RMSNorm, MLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size):
            DynamicBlock(),
            embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config, use_pos_emb),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog),
                config.hidden_size, config.intermediate_size, config.spatial_merge_size, lm_hidden_size, config.mlp_intermediate_size)
        {
            const int max_length = config.max_length;
            pos_helper.reset(new TensorPosHelper(max_length, 0, config.patch_size, config.spatial_merge_size));

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads,
                    config.is_ocr_vision ? config.intermediate_size : config.out_hidden_size,
                    max_length);
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
            for (size_t i = 0; i < layers.size(); i++)
                r += layers[i]->get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            if (!loader->has_tensor(path + "patch_embed.proj.bias")) return;

            embeddings.load(path, loader);
            multi_modal_projector.load(path, loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "layers." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            _loaded = true;
        }
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            pos_helper->prepare(grid_h, grid_w);

            auto output = embeddings.forward(ctx, input, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->attention.grid_h = grid_h;
                layers[i]->attention.grid_w = grid_w;
                output = layers[i]->forward(ctx, output, 0);
            }
            output = multi_modal_projector.forward(ctx, output, grid_h, grid_w);
            return output;
        }
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        MultiModalProjector multi_modal_projector;
        std::unique_ptr<TensorPosHelper> pos_helper;
    };

    class BaseVisualEmbeddingGeneration
    {
    public:
        BaseVisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096);

        bool load(ModelLoader &loader);

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config);

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);

    protected:
        virtual DynamicBlock *create_vis_model(InitContext *ctx, const Config &config, const int max_llm_tokens) = 0;
        virtual int get_tensor_number(const Config &config) = 0;

        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        const int max_llm_tokens;
        DynamicBlock *_vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };


    template <class VisionTransformer, int misc_tenso_num, int tensors_per_layer> class VisualEmbeddingGeneration : public BaseVisualEmbeddingGeneration
    {
    public:
        using BaseVisualEmbeddingGeneration::BaseVisualEmbeddingGeneration;

    protected:
        DynamicBlock *create_vis_model(InitContext *ctx, const Config &config, const int max_llm_tokens) override
        {
            vis_model.reset(new VisionTransformer(ctx, config, max_llm_tokens));
            return vis_model.get();
        }

        int get_tensor_number(const Config &config) override
        {
            return misc_tenso_num + vis_config.num_hidden_layers * tensors_per_layer;
        }

    private:
        std::unique_ptr<VisionTransformer> vis_model;
    };
}

namespace chatllm::glm::vit
{
    PatchEmbedding::PatchEmbedding(InitContext *ctx, const Config &config, const bool use_pos_emb):
        use_pos_emb(use_pos_emb),
        merge_size(config.spatial_merge_size),
        max_patches(config.max_length), ref_w(config.image_ref_size / config.patch_size), ref_h(config.image_ref_size / config.patch_size),
        proj0(ctx, 3, config.hidden_size, config.patch_size, config.patch_size, 0, 1, 1, false),
        proj1(ctx, 3, config.hidden_size, config.patch_size, config.patch_size, 0, 1, 1, false),
        proj_bias(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size))
    {
        CHATLLM_CHECK(config.temporal_patch_size == 2);
        CHATLLM_CHECK(merge_size == 2);

        if (use_pos_emb)
        {
            post_conv_layernorm.reset(new RMSNorm(ctx, config.hidden_size));
            position_embedding.reset(new Embedding(ctx, ggml::type::GGML_TYPE_F32, ref_w * ref_h, config.hidden_size));
        }
    }

    ggml::tensor *PatchEmbedding::forward(ComputeContext *ctx, ggml::tensor *input0, ggml::tensor *input1, int grid_h, int grid_w)
    {
        ggml::tensor *x0 = proj0.forward(ctx, input0);
        ggml::tensor *x1 = proj1.forward(ctx, input1);
        ggml::tensor *x  = ggml::add(ctx, x0, x1);
        auto bias_view = ggml::reshape_3d(ctx, proj_bias, 1, 1, ggml::get_dim(proj_bias, 0));
        x = ggml::add(ctx, x, bias_view);
        x = ggml::reshape_2d(ctx, x, ggml::get_dim(x, 2), grid_h * grid_w);

        if (use_pos_emb)
        {
            x = post_conv_layernorm->forward(ctx, x);

            //input: ggml[dim, w, h]
            CHATLLM_CHECK(ggml::get_dim(x, 3) == 1);

            ggml::tensor * interp = nullptr;
            auto pos_emb = position_embedding->weight;

            const int hidden_size = ggml::get_dim(pos_emb, 0);

            if ((ref_w == grid_w) && (ref_h == grid_h))
            {
                interp = ggml::reshape_3d(ctx, pos_emb, hidden_size, ref_w, ref_h);
            }
            else
            {
                auto permuted = ggml::reshape_3d(ctx, pos_emb, hidden_size, ref_w, ref_h);
                permuted = ggml::permute(ctx, permuted, 2, 0, 1);
                interp = ggml::interpolate(ctx, permuted, ggml::InterpolateMode::Bicubic,
                    grid_w, grid_h, hidden_size, 1);

                interp = ggml::permute(ctx, interp, 1, 2, 0); // -> [dim, w, h]
                interp = ggml::cont(ctx, interp);
            }

            // rearrange for `merge_size` == 2
            interp = ggml::reshape(ctx, interp, hidden_size * merge_size, grid_w / merge_size, merge_size, grid_h / merge_size);
            interp = ggml::permute(ctx, interp, 0, 2, 1, 3); // -> [dim * 2, 2, w / 2, h / 2]
            interp = ggml::cont(ctx, interp);
            interp = ggml::reshape_2d(ctx, interp, hidden_size, grid_h * grid_w);

            x = ggml::add(ctx, x, interp);
        }

        return x;
    }

    int64_t PatchEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += proj0.get_param_num(effective_only);
        r += proj1.get_param_num(effective_only);
        r += ggml::nelements(proj_bias);
        if (use_pos_emb)
        {
            r += post_conv_layernorm->get_param_num(effective_only);
            r += position_embedding->get_param_num(effective_only);
        }
        return r;
    }

    void PatchEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        proj0.load(path + "patch_embed.proj.0.", loader);
        proj1.load(path + "patch_embed.proj.1.", loader);
        loader->read_tensor(path + "patch_embed.proj.bias", proj_bias);

        if (use_pos_emb)
        {
            post_conv_layernorm->load(path + "post_conv_layernorm.", loader);
            position_embedding->load(path + "embeddings.position_embedding.", loader);
        }
    }

    MultiModalProjector::MultiModalProjector(InitContext *ctx, int hidden_size, int intermediate_size, int spatial_merge_size, int lm_hidden_size, int mlp_intermediate_size):
        spatial_merge_size(spatial_merge_size),
        hidden_size(hidden_size * spatial_merge_size * spatial_merge_size),
        post_layernorm(ctx, hidden_size),
        downsample(ctx, hidden_size, lm_hidden_size, spatial_merge_size, spatial_merge_size),
        proj(ctx, lm_hidden_size, lm_hidden_size, false),
        post_projection_norm(ctx, lm_hidden_size),
        mlp(ctx, lm_hidden_size, mlp_intermediate_size > 0 ? mlp_intermediate_size : intermediate_size)
    {
    }

    ggml::tensor *MultiModalProjector::forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w)
    {
        auto output = post_layernorm.forward(ctx, image_features);
        output = ggml::reshape(ctx, output, ggml::get_dim(output, 0), spatial_merge_size, spatial_merge_size,
            ggml::get_dim(output, 1) / spatial_merge_size / spatial_merge_size);
        output = ggml::permute(ctx, output, 2, 0, 1, 3);
        output = ggml::cont(ctx, output);
        output = downsample.forward(ctx, output);
        output = ggml::reshape(ctx, output, ggml::get_dim(output, 2), ggml::get_dim(output, 3));
        output = proj.forward(ctx, output);
        output = post_projection_norm.forward(ctx, output);
        output = ggml::act(ctx, ActFunc::GELU, output);
        output = mlp.forward(ctx, output);
        return output;
    }

    int64_t MultiModalProjector::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r +=       post_layernorm.get_param_num(effective_only);
        r +=           downsample.get_param_num(effective_only);
        r += post_projection_norm.get_param_num(effective_only);
        r +=                 proj.get_param_num(effective_only);
        r +=                  mlp.get_param_num(effective_only);
        return r;
    }

    void MultiModalProjector::load(const std::string &path, TensorLoader *loader)
    {
              post_layernorm.load(path + "post_layernorm.", loader);
                  downsample.load(path + "downsample.", loader);
        post_projection_norm.load(path + "merger.post_projection_norm.", loader);
                        proj.load(path + "merger.proj.",  loader);
                         mlp.load(path + "merger.",  loader);
    }

    BaseVisualEmbeddingGeneration::BaseVisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "vis", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool BaseVisualEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (_vis_model)
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            _vis_model->load("visual.", &loader);
            loader.pop_allocator_manager();
            return _vis_model->is_loaded();
        }
        else
            return false;
    }

    bool BaseVisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
    {
        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vis_config.dtype = dtype;

        vis_config.is_ocr_vision        = vis_cfg["model_type"].ToString() == "glm_ocr_vision";
        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.image_ref_size       = (int)vis_cfg["image_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["depth"].ToInt();
        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
        vis_config.out_hidden_size      = (int)vis_cfg["out_hidden_size"].ToInt();
        vis_config.spatial_merge_size   = (int)vis_cfg["spatial_merge_size"].ToInt();
        vis_config.temporal_patch_size  = (int)vis_cfg["temporal_patch_size"].ToInt();
        vis_config.mlp_intermediate_size= vis_config.is_ocr_vision ? lm_hidden_size * 3 : -1;

        auto pp_cfg = config["preprocessor_config.json"];
        if (pp_cfg.IsObject())
        {
            pp_cfg = config["preprocessor_config.json"];
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

        vis_config.max_length = max_llm_tokens * vis_config.spatial_merge_size * vis_config.spatial_merge_size;

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = get_tensor_number(vis_config);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        _vis_model = create_vis_model(&_ctx, vis_config, lm_hidden_size);

        _ctx.check_used_mem_size(true);

        return true;
    }

    void BaseVisualEmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if ((_vis_model == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!_vis_model->is_loaded()) return;

        for (auto &image : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, image, buf);
        }
    }

    bool BaseVisualEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
    {
        ggml::tensor *media_emb = nullptr;
        const auto make_graph = [this, &media_emb, &image](ComputeContext *ctx) -> ggml::tensor * {
            media_emb = ggml::new_tensor_4d(ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size, vis_config.patch_size, 3, image.grid_width * image.grid_height);
            auto r = _vis_model->forward(ctx, media_emb, image.grid_height, image.grid_width);
            return r;
        };
        const auto write_input_data = [&media_emb, &image](ComputeContext *ctx) {
            Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }
}

namespace chatllm::glm::v4v
{
    struct Config : glm::glm4_0414::Config
    {
        int mrope_section[qwen::v2_5_vl::MROPE_SECTION_MAX];
    };

    class ChatHistoryEncoder : public v4::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    protected:
        void append_image_piece(int round_idx, const ContentPiece &piece, std::vector<int> &ids) const;

    public:
        const vit::Config *vis_config = nullptr;
        bool vit_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public glm::glm4_0414::Tokenizer
    {
    public:
        Tokenizer(const Config &config) : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder) : glm::glm4_0414::Tokenizer(config, chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int begin_of_image_token_id;
        int end_of_image_token_id;

        int     video_max_frames = 20;
        bool    native_resolution = true;
        double  fps = 1.0;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = glm::glm4_0414::Tokenizer::load(buffer, n_vocab);

        begin_of_image_token_id = tp->PieceToId("<|begin_of_image|>");
        end_of_image_token_id   = tp->PieceToId("<|end_of_image|>");

        return r;
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(begin_of_image_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(end_of_image_token_id);
    }

    class BaseVisualConditionalGeneration;

    class TensorPosHelper3D : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper3D(int max_length, int image_id_start, BaseVisualConditionalGeneration *gen)
            : BaseTensorPosHelper(max_length * 4),
              original_length(max_length), image_id_start(image_id_start),
              gen(gen)
        {
        }

        ggml::tensor *allocate_pos_tensor(InitContext *ctx) override
        {
            ggml::tensor *r = ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length);
            ctx->get_allocator()->alloc(r);
            return r;
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override;
    protected:
        const int original_length;
        const int image_id_start;
        BaseVisualConditionalGeneration *gen;
    };

    #define MAX_PROJECTED_TOKENS    2048

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int size) : pad_arg(new BlockParams::PadEmbedding(size, size)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class BaseVisualConditionalGeneration : public TensorPosHelperPrelude, public ExtendEmbedding, public glm::glm4_0414::ConditionalGeneration
    {
    public:
        typedef glm::glm4_0414::ConditionalGeneration Base;
    public:
        BaseVisualConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int head_dim);

        void load(ModelLoader &loader) override;
        bool load_more(const json::JSON &config) override;
        void before_generate(const GenerationConfig &gen_config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
    protected:
        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override;
        void set_visual(vit::BaseVisualEmbeddingGeneration *_visual);
    public:
        const Config config;
        std::vector<int> v_pos;
    protected:
        std::vector<qwen::v2_5_vl::ImageGridSize> images_grid;
        int token_time;
    private:
        vit::BaseVisualEmbeddingGeneration *_visual;
    };

    class ConditionalGeneration : public BaseVisualConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = MODEL_TYPE_GLM4V):
            BaseVisualConditionalGeneration(config, runtime_config, type, -1),
            visual(runtime_config, MAX_PROJECTED_TOKENS)
        {
            set_visual(&visual);
        }
    protected:
        vit::VisualEmbeddingGeneration<vit::VisionTransformer<vit::ViTSelfAttention, SiLUMLP, true>, 14, 10> visual;
    };

    void TensorPosHelper3D::prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen)
    {
        pos->ne[0] = (int)(gen->v_pos.size());
        Backend::write_tensor_data(pos, gen->v_pos.data(), 0, gen->v_pos.size() * sizeof(gen->v_pos[0]));
    }

    BaseVisualConditionalGeneration::BaseVisualConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
        ModelType type, int head_dim) :
        TensorPosHelperPrelude(new TensorPosHelper3D(config.max_length, config.vocab_size, this)),
        ExtendEmbedding(MAX_PROJECTED_TOKENS),
        Base(config, runtime_config, type, head_dim),
        config(config),
        token_time(0)
    {
        delete pad_arg;
        pad_arg = nullptr;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<glm::glm4_0414::ModelClass>()->layers[i];
            layer.attention.mrope_sections = this->config.mrope_section;
            layer.attention.rope_mode = RoPEMode::MROPE;
        }
    }

    void BaseVisualConditionalGeneration::set_visual(vit::BaseVisualEmbeddingGeneration *_visual)
    {
        this->_visual = _visual;
    }

    void BaseVisualConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {".self_attn.",                         ".attn."},
            {".input_layernorm.",                   ".norm1."},
            {".post_self_attn_layernorm.",          ".norm2."},
        });

        _chat_encoder.vit_loaded = _visual->load(loader);
    }

    bool BaseVisualConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);
        bool r = _visual->load_more(this->config.dtype, this->config.hidden_size, config);
        if (r)
        {
            _chat_encoder.vis_config = &_visual->vis_config;
        }
        return r;
    }

    void BaseVisualConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->video_max_frames       = utils::get_opt(args, "video_max_frames", tok->video_max_frames);
        tok->native_resolution      = utils::get_opt(args, "native_resolution", tok->native_resolution);
        tok->fps                    = utils::get_opt(args, "fps", tok->fps);
    }

    void BaseVisualConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        images_grid.clear();
        for (auto &mm : tokenizer->media_emb)
        {
            images_grid.emplace_back(mm.grid_width  / _visual->vis_config.spatial_merge_size,
                                     mm.grid_height / _visual->vis_config.spatial_merge_size);
        }

        auto emb = dynamic_cast<Embedding *>(transformer->word_embeddings);
        _visual->generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    bool BaseVisualConditionalGeneration::generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits)
    {
        const int image_id_start = config.vocab_size;
        const int length = (int)input_ids.size();

        // TODO:
        int token_n_inc = 1;

        v_pos.clear();
        v_pos.resize(length * 4, 0);
        int *p_t = &v_pos[length * 0];
        int *p_h = &v_pos[length * 1];
        int *p_w = &v_pos[length * 2];

        if ((n_past == 0) && (n_past_offset == 0))
            token_time = 0;

        int t = token_time;
        int mm_index = 0;

        int i = 0;
        while (i < length)
        {
            if (input_ids[i] < image_id_start)
            {
                p_t[i] = t;
                p_h[i] = t;
                p_w[i] = t;
                i++;
                t++;
                continue;
            }

            CHATLLM_CHECK(mm_index < (int)images_grid.size());

            auto &dim = images_grid[mm_index++];
            for (int f = 0; f < dim.frame_num; f++, t += token_n_inc)
            {
                for (int h = 0; h < dim.h; h++)
                {
                    for (int w = 0; w < dim.w; w++)
                    {
                        CHATLLM_CHECK(input_ids[i] >= image_id_start);
                        p_t[i] = t;
                        p_h[i] = t + h;
                        p_w[i] = t + w;
                        i++;
                    }
                }
            }
            t = std::max(p_h[i - 1], p_w[i - 1]) + 1;
        }

        token_time = t;
        auto r = Base::generate_next_token(input_ids, gen_config, lm_logits);

        return r;
    }

    void ChatHistoryEncoder::append_image_piece(int round_idx, const ContentPiece &piece, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        CHATLLM_CHECK(piece.type == ContentPiece::Type::Image);
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                int w, h;
                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->patch_size;

                vision::MaxPatchNum     param1(vis_config->max_length);
                vision::MergeKernel     param2(vis_config->spatial_merge_size, vis_config->spatial_merge_size);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
        if ((w <= 0) || (h <= 0)) return;

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_MergeN_ChannelsRGB_PixelsLeftRightDown);

                const int merge_length = vis_config->spatial_merge_size * vis_config->spatial_merge_size;
                image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_media("image", ids, id_start, image.emb_vec_number);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.insert(ids.end(), {tok->user_token_id, tok->nl_token_id});

        std::unique_ptr<vision::Resize> resize;

        if (!tok->native_resolution)
            resize.reset(new vision::Resize(vis_config->image_ref_size, vis_config->image_ref_size));

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                if (piece.content.size() > 0)
                {
                    tok->encode(piece.content, ids);
                }
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                append_image_piece(round_idx, piece, ids);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
    }
}

namespace chatllm::glm::ocr::vit
{
    class VitSiLUMLP : public BaseMLP
    {
    public:
        VitSiLUMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::SILU, true, true, true)
        {}
    };

    class ViTSelfAttention : public QKNormedAttention<RMSNorm, qwen::vit::GenericViTSelfAttention2D<BaseCachelessAttention>>
    {
    public:
        typedef QKNormedAttention<RMSNorm, qwen::vit::GenericViTSelfAttention2D<BaseCachelessAttention>> Base;
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length):
            Base(ctx, hidden_size, num_attention_heads, max_length, true, true)
        {}
    };
}

namespace chatllm::glm::ocr
{
    struct Config : v4v::Config
    {
        int head_dim;
    };

    class ChatHistoryEncoder : public v4v::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public v4v::Tokenizer
    {
    public:
        Tokenizer(const Config &config) : v4v::Tokenizer(config, &_chat_encoder)
        {}
    };

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        bool found_image = false;

        const_cast<ChatHistoryEncoder *>(this)->vit_loaded = v4v::_chat_encoder.vit_loaded;
        const_cast<ChatHistoryEncoder *>(this)->vis_config = v4v::_chat_encoder.vis_config;

        std::string text_prompt = "";
        ids.insert(ids.end(), {tok->user_token_id, tok->nl_token_id});

        std::unique_ptr<vision::Resize> resize;

        if (!tok->native_resolution)
            resize.reset(new vision::Resize(vis_config->image_ref_size, vis_config->image_ref_size));

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                text_prompt += piece.content;
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                append_image_piece(round_idx, piece, ids);
                found_image = true;
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        if (text_prompt.size() > 0)
        {
            if (text_prompt[text_prompt.size() - 1] != ':')
                text_prompt.push_back(':');
        }
        else if (found_image)
            text_prompt = "Text Recognition:";
        else;

        tok->encode(text_prompt, ids);
    }

    class ConditionalGeneration : public v4v::BaseVisualConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = MODEL_TYPE_GLM_OCR):
            BaseVisualConditionalGeneration(config, runtime_config, type, config.head_dim),
            visual(runtime_config, MAX_PROJECTED_TOKENS)
        {
            set_visual(&visual);
        }
    protected:
        glm::vit::VisualEmbeddingGeneration<glm::vit::VisionTransformer<ocr::vit::ViTSelfAttention, ocr::vit::VitSiLUMLP, false>, 12, 19> visual;
    };
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(GLM4V,                 glm::v4v, 1);
    REGISTER_MODEL_LOADER(GLM_OCR,               glm::ocr, 1);
}