#include "qwen.h"
#include <cstring>
#include "../src/vision_process.h"

namespace chatllm::step::vit
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int max_patches;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int image_size;
        int big_patch_size;

        float image_mean[3];
        float image_std[3];
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, const Config &config):
            Block(),
            posemb_grid_size(config.image_size / config.patch_size),
            conv1(ctx, 3, config.hidden_size, config.patch_size, config.patch_size, 0, 1, 1, false),
            positional_embedding(ctx, ggml::type::GGML_TYPE_F32, posemb_grid_size * posemb_grid_size, config.hidden_size),
            ln_pre(ctx, config.hidden_size)
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input0, int grid_h, int grid_w)
        {
            auto hidden_state = conv1.forward(ctx, input0);

            // change shape to [hidden, w * h]
            hidden_state = ggml::reshape_3d(ctx, hidden_state,
                ggml::get_dim(hidden_state, 0) * ggml::get_dim(hidden_state, 1), ggml::get_dim(hidden_state, 2), ggml::get_dim(hidden_state, 3));
            hidden_state = ggml::transpose(ctx, hidden_state);
            hidden_state = ggml::cont(ctx, hidden_state);

            {
                ggml::tensor * interp = nullptr;
                auto pos_emb = positional_embedding.weight;

                const int hidden_size = ggml::get_dim(pos_emb, 0);

                if ((posemb_grid_size == grid_w) && (posemb_grid_size == grid_h))
                {
                    interp = pos_emb;
                }
                else
                {
                    auto permuted = ggml::reshape_3d(ctx, pos_emb, hidden_size, posemb_grid_size, posemb_grid_size);
                    permuted = ggml::permute(ctx, permuted, 2, 0, 1);
                    interp = ggml::interpolate(ctx, permuted, ggml::InterpolateMode::Bilinear,
                        grid_w, grid_h, hidden_size, 1);

                    interp = ggml::permute(ctx, interp, 1, 2, 0); // -> [hidden, w, h]
                    interp = ggml::cont(ctx, interp);
                    interp = ggml::reshape_2d(ctx, interp, hidden_size, grid_w * grid_h);
                }

                hidden_state = ggml::add(ctx, hidden_state, interp);
            }

            hidden_state = ln_pre.forward(ctx, hidden_state);
            return hidden_state;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += positional_embedding.get_param_num(effective_only);
            r +=               ln_pre.get_param_num(effective_only);
            r +=                conv1.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            loader->read_tensor(path + "positional_embedding", positional_embedding.weight);
            ln_pre.load(path + "ln_pre.",               loader);
             conv1.load(path + "conv1.",                loader);
        }
    public:
        const int               posemb_grid_size;
        Conv2D                  conv1;
        Embedding               positional_embedding;
        LayerNorm               ln_pre;
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, int hidden_size, int lm_hidden_size):
            hidden_size(hidden_size),
            vit_downsampler1(ctx, hidden_size,     hidden_size * 2, 3, 2, 1),
            vit_downsampler2(ctx, hidden_size * 2, hidden_size * 4, 3, 2, 1),
            vit_large_projector(ctx, hidden_size * 4, lm_hidden_size, false)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w) override
        {
            auto output = image_features;
            output = ggml::transpose(ctx, output);
            output = ggml::reshape_4d(ctx, output, grid_w, grid_h, ggml::get_dim(output, 1), ggml::get_dim(output, 2));
            output = vit_downsampler1.forward(ctx, output);
            output = vit_downsampler2.forward(ctx, output);

            output = ggml::reshape_3d(ctx, output, ggml::get_dim(output, 0) * ggml::get_dim(output, 1), ggml::get_dim(output, 2), ggml::get_dim(output, 3));
            output = ggml::transpose(ctx, output);
            output = ggml::cont(ctx, output);
            output = vit_large_projector.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += vit_large_projector.get_param_num(effective_only);
            r +=    vit_downsampler1.get_param_num(effective_only);
            r +=    vit_downsampler2.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            vit_large_projector.load(path + "vit_large_projector.", loader);
               vit_downsampler1.load(path + "vit_downsampler1.",    loader);
               vit_downsampler2.load(path + "vit_downsampler2.",    loader);
        }

    public:
        const int hidden_size;
        Conv2D vit_downsampler1;
        Conv2D vit_downsampler2;
        Linear vit_large_projector;
    };

    class VitSiLUMLP : public TheMLP
    {
    public:
        VitSiLUMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  TheMLP(ctx, hidden_size, intermediate_size, ActFunc::GELU_QUICK, true)
        {}

        void load(const std::string &path, TensorLoader *loader) override
        {
            fc0.load(path + "c_fc.",    loader);
            fc1.load(path + "c_proj.",  loader);
        }
    };

    class EncoderLayerScale : public Block
    {
    public:
        EncoderLayerScale(InitContext *ctx, int hidden_size):
            Block(),
            gamma(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, hidden_size))
        {}

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
        {
            auto output = ggml::mul(ctx, hidden_states, gamma);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += ggml::nelements(gamma);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            loader->read_tensor(path + "gamma", gamma);
        }

    public:
        ggml::tensor *gamma;
    };

    class TensorPosHelper
    {
    public:
        TensorPosHelper(int max_length)
            : max_length(max_length * 4)
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
            length = grid_h * grid_w;

            v_pos.clear();
            v_pos.resize(length * 2, 0);
            int *p_w = &v_pos[length * 0];
            int *p_h = &v_pos[length * 1];
            int index = 0;

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    p_w[index] = j;
                    p_h[index] = i;
                    index++;
                }
            }
        }
    public:
        const int max_length;
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
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool use_bias = true):
            TensorPosHelperPrelude(new TensorPosHelper2D(max_length)),
            RoPESelfAttention<BaseCachelessAttention>(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, use_bias, use_bias)
        {
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

    class LayerBlock : public LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, VitSiLUMLP>
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, VitSiLUMLP> Base;

        LayerBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int max_length)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, max_length),
              ls_1(ctx, hidden_size),
              ls_2(ctx, hidden_size)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            ggml::tensor *residual = hidden_states;

            hidden_states = input_layernorm.forward(ctx, hidden_states);
            hidden_states = attention.forward(ctx, hidden_states, n_past);
            hidden_states = ls_1.forward(ctx, hidden_states);
            hidden_states = ggml::add(ctx, hidden_states, residual);

            residual = hidden_states;

            hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
            hidden_states = mlp.forward(ctx, hidden_states);
            hidden_states = ls_2.forward(ctx, hidden_states);
            hidden_states = ggml::add(ctx, hidden_states, residual);

            return hidden_states;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            r += ls_1.get_param_num(effective_only);
            r += ls_2.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Base::load(path, loader);
            ls_1.load(path + "ls_1.", loader);
            ls_2.load(path + "ls_2.", loader);
        }
    public:
        EncoderLayerScale ls_1;
        EncoderLayerScale ls_2;
    };

    class VisionTransformer : public Block
    {
    public:
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size):
            Block(),
            embeddings(ctx, config),
            multi_modal_projector(ctx, config.hidden_size, lm_hidden_size)
        {
            const int max_length = config.max_patches;
            pos_helper.reset(new TensorPosHelper(max_length));

            const auto flag = qwen::vit::ViTParams::is_full_attention();
            qwen::vit::ViTParams::set_full_attention(true);

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
                layer->set_id(layer_id);
                layer->attention.set_pos_helper(pos_helper.get());
                layers.emplace_back(layer);
            }

            qwen::vit::ViTParams::set_full_attention(flag);
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
            if (!loader->has_tensor(path + "positional_embedding")) return;

            embeddings.load(path , loader);
            multi_modal_projector.load(path, loader);

            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "blocks." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            loaded = true;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            pos_helper->prepare(grid_h, grid_w);

            auto output = embeddings.forward(ctx, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                auto *attn = &layers[i]->attention;

                attn->grid_h = grid_h;
                attn->grid_w = grid_w;
                output = layers[i]->forward(ctx, output, 0);
            }

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
        std::unique_ptr<TensorPosHelper> pos_helper;
    protected:
        bool loaded = false;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096):
            max_llm_tokens(max_llm_tokens),
            eval(runtime_config, "vis", GRAPH_SIZE),
            _ctx(eval.get_backend_context())
        {
            _ctx.cache_dtype = runtime_config.cache_type;
        }

        bool load(ModelLoader &loader)
        {
            if (vis_model.get())
            {
                loader.push_allocator_manager(eval.get_layer_allocators());
                vis_model->load("visual.", &loader);
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

            vis_config.image_size           = (int)vis_cfg["image_size"].ToInt();
            vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
            vis_config.hidden_size          = (int)vis_cfg["width"].ToInt();
            vis_config.num_hidden_layers    = (int)vis_cfg["layers"].ToInt();
            vis_config.num_attention_heads  = (int)vis_cfg["heads"].ToInt();
            vis_config.intermediate_size    = (int)(vis_config.hidden_size * (8960.0 / 1536));

            vis_config.big_patch_size       = 504;

            const float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
            const float std[]  = {0.26862954f, 0.26130258f, 0.27577711f};
            memcpy(vis_config.image_mean, mean, sizeof(mean));
            memcpy(vis_config.image_std,  std,  sizeof(mean));

            vis_config.max_patches = (vis_config.image_size / vis_config.patch_size) * (vis_config.image_size / vis_config.patch_size);

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 9 + vis_config.num_hidden_layers * 19;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            _ctx.dtype = dtype;

            vis_model.reset(new VisionTransformer(&_ctx, vis_config, lm_hidden_size));

            _ctx.check_used_mem_size(true);

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

        VisionTransformer *get_model(void) const
        {
            return vis_model.get();
        }

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
        {
            ggml::tensor *media_emb = nullptr;
            const auto make_graph = [this, &media_emb, &image](ComputeContext *ctx) -> ggml::tensor * {
                media_emb = ggml::new_tensor_3d(ctx, ggml::type::GGML_TYPE_F32, image.width, image.height, 3);
                auto r = vis_model->forward(ctx, media_emb, image.grid_height, image.grid_width);
                return r;
            };
            const auto write_input_data = [&media_emb, &image](ComputeContext *ctx) {
                Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            };

            std::vector<int64_t> shape;
            eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
            return true;
        }

    public:
        const int max_llm_tokens;
        std::unique_ptr<VisionTransformer> vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };
}

namespace chatllm::step::vl
{
    typedef qwen::v3::Config Config;

    class ChatHistoryEncoder : public qwen::v1::ChatHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    protected:
        void append_content(const Content &user, std::vector<int> &ids) const;
    public:
        vit::Config *vis_config = nullptr;
        bool         vit_loaded = false;
    };

    ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public qwen::v3::Tokenizer
    {
    public:
        typedef qwen::v3::Tokenizer Base;
        Tokenizer(const BaseConfig &config):
            Base(config, &_chat_encoder)
        {
            auto_add_bos = false;
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = Base::load(buffer, n_vocab);
            patch_newline_token_id = tp->PieceToId("<patch_newline>");
              image_start_token_id = tp->PieceToId("<im_start>");
              patch_start_token_id = tp->PieceToId("<patch_start>");
                image_end_token_id = tp->PieceToId("<im_end>");
                patch_end_token_id = tp->PieceToId("<patch_end>");

            return r;
        }

        void inject_image(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
        {
            ids.push_back(image_start_token_id);
            for (int i = 0; i < ids_to_inject_count; i++)
            {
                ids.push_back(i + ids_to_inject_start);
            }
            ids.push_back(image_end_token_id);
        }

        void inject_big_patch(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count, bool append_newline = false)
        {
            ids.push_back(patch_start_token_id);
            for (int i = 0; i < ids_to_inject_count; i++)
            {
                ids.push_back(i + ids_to_inject_start);
            }
            ids.push_back(patch_end_token_id);
            if (append_newline) ids.push_back(patch_newline_token_id);
        }
    public:
        int patch_start_token_id;
        int patch_end_token_id;
        int patch_newline_token_id;
        int image_start_token_id;
        int image_end_token_id;
        bool do_pan_and_scan = true;
        bool native_global_view = false;
    };

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("assistant", ids, true, false, true);
        tok->encode("<think>", ids, false, false, true);
    }

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding() : pad_arg(new BlockParams::PadEmbedding(4096, 4096)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public qwen::v3::ConditionalGeneration
    {
    public:
        typedef qwen::v3::ConditionalGeneration Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = ModelType::MODEL_TYPE_STEP_VL);

        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        int64_t get_param_num(bool effective_only) const;
        void before_generate(const GenerationConfig &gen_config) override;
        void set_tokenizer(BaseTokenizer *tokenizer) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
    protected:
        vit::VisualEmbeddingGeneration visual;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type):
        ExtendEmbedding(),
        Base(config, runtime_config, type),
        visual(runtime_config, pad_arg->get())
    {
        delete pad_arg;
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        Base::set_tokenizer(tokenizer);
        auto_output_prefix = tokenizer->encode("<think>\n");
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

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return;
        tok->do_pan_and_scan      = utils::get_opt(args, "do-pan-and-scan", tok->do_pan_and_scan);
        tok->native_global_view   = utils::get_opt(args, "native-resolution", tok->native_global_view);
        if (tok->native_global_view)
            tok->do_pan_and_scan = false;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {".self_attn.",                 ".attn."},
            {".input_layernorm.",           ".ln_1."},
            {".post_attention_layernorm.",  ".ln_2."},
        });

        _chat_encoder.vit_loaded = visual.load(loader);
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = Base::get_param_num(effective_only);
        if (_chat_encoder.vit_loaded)
            r += visual.vis_model->get_param_num(effective_only);
        return r;
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

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        if (tokenizer->get_system_prompt().size() < 1) return;
        qwen::v1::ChatHistoryEncoder::append_sys_prompt(ids);
    }

    void ChatHistoryEncoder::append_content(const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

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
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                std::vector<vision::image_pixels_t> crops;
                int crops_per_row = 0;
                int whole_image_width  = vis_config->image_size;
                int whole_image_height = vis_config->image_size;

                if (!tok->do_pan_and_scan)
                {
                    std::unique_ptr<vision::Resize> resize;
                    if (!tok->native_global_view)
                        resize.reset(new vision::Resize(vis_config->image_size, vis_config->image_size));

                    int w = 0;
                    int h = 0;
                    crops.emplace_back(vision::image_pixels_t());
                    auto &image = crops.back();
                    vision::image_load(piece.content.c_str(), image, w, h, vis_config->patch_size, vision::PaddingMode::Black);
                    if (w < 1) continue;
                    whole_image_width  = w;
                    whole_image_height = h;
                }
                else
                {
                    vision::image_load_pan_and_scan(piece.content.c_str(),
                        crops,
                        vis_config->image_size, vis_config->big_patch_size,
                        crops_per_row);
                }

                if (crops.size() < 1) continue;

                for (int i = 0; i < (int)crops.size(); i++)
                {
                    // <patch>...<patch><image>
                    const int id = (i + 1) % crops.size();
                    const auto &pixels = crops[id];

                    const int w = id == 0 ? whole_image_width  : vis_config->big_patch_size;
                    const int h = id == 0 ? whole_image_height : vis_config->big_patch_size;
                    std::vector<float> scaled;
                    const int patch_size = vis_config->patch_size;

                    vision::image_rescale(pixels, scaled);
                    vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                    tok->media_emb.push_back({.width = w, .height = h, .grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                    auto &image = tok->media_emb.back();
                    image.emb_vec_number = (image.grid_height / 4) * (image.grid_width / 4);

                    vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);

                    const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;

                    if (0 == id)
                    {
                        tok->inject_image(ids, id_start, image.emb_vec_number);
                    }
                    else
                    {
                        const bool append_nl = ((id % crops_per_row) == 0) && (id < (int)crops.size() - 1);
                        tok->inject_big_patch(ids, id_start, image.emb_vec_number, append_nl);
                    }
                }
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_user_opening(round_idx, ids);

        append_content(user, ids);

        ids.push_back(tok->im_end_token_id);
        ids.push_back(tok->nl_token_id);
    }
}

namespace chatllm::step
{
    REGISTER_MODEL_LOADER(STEP_VL,              step::vl, 1);
}