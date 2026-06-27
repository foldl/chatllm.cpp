#include "ernie.h"
#include <cstring>
#include "qwen.h"
#include "../src/vision_process.h"

namespace chatllm::ernie::dense
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class ChatHistoryThinkingEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder           _chat_encoder;
    static ChatHistoryThinkingEncoder   _chat_thinking_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : chatllm::llama::v2::Tokenizer(config, &_chat_encoder),
          im_start_token_id(-1), im_end_token_id(-1),
          nl_token_id(-1), think_start_token_id(-1),
          think_end_token_id(-1)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t size = chatllm::llama::v2::Tokenizer::load(buffer, n_vocab);
        im_start_token_id = tp->PieceToId("<|im_start|>");
        im_end_token_id   = tp->PieceToId("<|im_end|>");
        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        think_start_token_id = tp->PieceToId("<think>");
        think_end_token_id   = tp->PieceToId("</think>");
        if (im_end_token_id >= 0)
            terminate_ids.emplace(im_end_token_id);
        return size;
    }

    void Tokenizer::encode_role(const std::string &role, const std::string &text, std::vector<int> &ids) const
    {
        ids.push_back(im_start_token_id);
        BaseTokenizer::encode(role, ids);
        ids.push_back(nl_token_id);
        BaseTokenizer::encode(text, ids);
        ids.push_back(im_end_token_id);
        ids.push_back(nl_token_id);
        ids.push_back(nl_token_id);
    }

    void Tokenizer::encode_role(const std::string &role, std::vector<int> &ids) const
    {
        ids.push_back(im_start_token_id);
        BaseTokenizer::encode(role, ids);
    }

    bool Tokenizer::load_config(const json::JSON &config)
    {
        auto cfg = config["tokenizer_config.json"];
        std::string s = cfg["chat_template"].ToString();
        if (s.find("think_mode=True") != std::string::npos)
        {
            set_chat_encoder(&_chat_thinking_encoder);
        }

        return true;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            oss_prompt << tok->get_system_prompt() << "\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "User:  " + user << "\n";
        auto text = oss_prompt.str();
        tok->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("Assistant:  ", ids);
    }

    void ChatHistoryThinkingEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role("assistant", ai, ids);
    }

    void ChatHistoryThinkingEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            oss_prompt << "<system_setting>\n" << tok->get_system_prompt() << "\n</system_setting>\n\n";
        }
        oss_prompt << "<global_setting>\n"
                    << "think_mode=True\n"
                    << "</global_setting>";
        tok->encode_role("system", oss_prompt.str(), ids);
    }

    void ChatHistoryThinkingEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role("user", user, ids);
    }

    void ChatHistoryThinkingEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role("assistant", ids);
    }

    void ChatHistoryThinkingEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role("user", ids);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : chatllm::llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type,
            config.num_key_value_heads, config.head_dim, config.max_length, 12, config.tie_word_embeddings != 0)
    {
        auto transformer = Base::get_typed_transformer<ModelClass2>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }

}

namespace chatllm::ernie::moe
{
    template <class ErnieMoEMLP> class ErnieMoEBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, ErnieMoEMLP>
    {
    public:
        ErnieMoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size1, int mlp_intermediate_size2,
                int num_kv_heads, int head_dim, int max_length)
            : LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, ErnieMoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
            num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class ErnieSparseMoE : public BaseSparseMLP
    {
    public:
        ErnieSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<ErnieSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>, SiLUMLP> ErnieMoEMLP;
        typedef ErnieMoEBlock<ErnieMoEMLP> MoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        GenericConditionalGeneration() = default;

        GenericConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_ERNIE_MOE, config, runtime_config, 4096 * 4),
            config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t moe_layers  = get_moe_layer_num();
            const size_t dense_layers = config.num_hidden_layers - moe_layers;
            const size_t num_tensors = 2 + dense_layers * (12) + moe_layers * (16 + 0) + (config.tie_word_embeddings ? 0 : 1);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (config.use_correction_bias)
                ggml::log(GGML_LOG_LEVEL_WARN, "use_correction_bias is ignored, see https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/modeling_ernie4_5_moe.py#L369");

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new MoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.moe_num_shared_experts,
                        config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                        config.max_length);
                    layer->attention.freq_base = config.rope_theta;
                    layer->mlp.mlp1.norm_topk_prob = true;
                    return layer;
                }
                else
                {
                    auto layer = new LlamaBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                        config.num_key_value_heads, config.max_length);
                    layer->attention.freq_base = config.rope_theta;
                    return layer;
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                config.tie_word_embeddings ? nullptr : create_lm_head(&w_ctx_, config, false), create_layer);
            Base::transformer = transformer;

            w_ctx_.check_used_mem_size(true);
        }

    protected:
        int  get_moe_layer_num(void)
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
                if (is_layer_moe(i)) r++;
            return r;
        }

        bool is_layer_moe(int i)
        {
            if (i < config.moe_layer_start_index) return false;
            return (i % config.moe_layer_interval) == 0;
        }
    public:
        Config config;
    };

    namespace experts_64
    {
        const int NUM_EXPERTS                   =  64;
        const int EXPERTS_PER_TOK               =  6;

        // make it easy to test with different number of experts.
        const int EFFECTIVE_EXPERTS_PER_TOK     =  EXPERTS_PER_TOK;

        typedef GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
    {
        switch (config.moe_num_experts)
        {
        case experts_64::NUM_EXPERTS:
            set_proxy_model(new experts_64::ConditionalGeneration(config, runtime_config));
            break;
        default:
            CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.moe_num_experts;
            break;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".mlp2.",              ".shared_experts."},
            {".mlp1.gate.",         ".gate."},
            {".mlp1.experts.",      ".experts."},
            {".mlp1.gate_score_correction_bias",     ".moe_statics.e_score_correction_bias"}
        });

        ModelProxy::load(loader);
    }
}

namespace chatllm::paddle::ocr::vit
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int max_patches;
        int posemb_grid_size;
        int num_channels;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int spatial_merge_size;

        float image_mean[3];
        float image_std[3];

        Config()
        {
            memset(this, 0, sizeof(Config));
        }
    };

    class Projector : public Block
    {
    public:
        Projector(InitContext *ctx, const Config &config, int llm_hidden_size);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int merge_kernel_size;
        const int hidden_size;
        LayerNorm pre_norm;
        TheBiasedGELUMLP mlp;
    };

    Projector::Projector(InitContext *ctx, const Config &config, int llm_hidden_size):
        merge_kernel_size(2),
        hidden_size(config.hidden_size * merge_kernel_size * merge_kernel_size),
        pre_norm(ctx, config.hidden_size),
        mlp(ctx, hidden_size, hidden_size, llm_hidden_size)
    {
    }

    ggml::tensor *Projector::forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
    {
        const auto vit_hidden_size = ggml::get_dim(input, 0);
        auto hidden_states = pre_norm.forward(ctx, input);
        hidden_states = ggml::reshape(ctx, hidden_states, vit_hidden_size * merge_kernel_size, grid_w / merge_kernel_size, grid_h);
        hidden_states = ggml::cont   (ctx, hidden_states);
        hidden_states = ggml::reshape(ctx, hidden_states, vit_hidden_size * merge_kernel_size, grid_w / merge_kernel_size, merge_kernel_size, grid_h / merge_kernel_size);
        hidden_states = ggml::permute(ctx, hidden_states, 0, 2, 1, 3);
        hidden_states = ggml::cont   (ctx, hidden_states);
        hidden_states = ggml::reshape(ctx, hidden_states, hidden_size, grid_w * grid_h / merge_kernel_size / merge_kernel_size);
        hidden_states = mlp.forward(ctx, hidden_states);
        return hidden_states;
    }

    int64_t Projector::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += pre_norm.get_param_num(effective_only);
        r += mlp.get_param_num(effective_only);
        return r;
    }

    void Projector::load(const std::string &path, TensorLoader *loader)
    {
        pre_norm.load(path + "pre_norm.", loader);
        mlp.load(path, loader);
    }

    class VisionEmbeddings : public Block
    {
    public:
        VisionEmbeddings(InitContext *ctx, const Config &config);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int   posemb_grid_size;
        Conv2D      patch_embedding;
        Embedding   position_embedding;
    };

    VisionEmbeddings::VisionEmbeddings(InitContext *ctx, const Config &config):
        posemb_grid_size(config.posemb_grid_size),
        patch_embedding(ctx, config.num_channels, config.hidden_size, config.patch_size, config.patch_size),
        position_embedding(ctx, ggml::type::GGML_TYPE_F32, posemb_grid_size * posemb_grid_size, config.hidden_size, false)
    {
    }

    ggml::tensor *VisionEmbeddings::forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
    {
        auto hidden_state = patch_embedding.forward(ctx, input);

        // change shape to [hidden, w * h]
        hidden_state = ggml::reshape_2d(ctx, hidden_state,
            ggml::get_dim(hidden_state, 0) * ggml::get_dim(hidden_state, 1) * ggml::get_dim(hidden_state, 2), ggml::get_dim(hidden_state, 3));

        {
            auto pos_emb = position_embedding.weight;
            ggml::tensor * interp = pos_emb;

            const int hidden_size = ggml::get_dim(pos_emb, 0);

            if ((posemb_grid_size != grid_w) || (posemb_grid_size != grid_h))
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

        return hidden_state;
    }

    int64_t VisionEmbeddings::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += patch_embedding.get_param_num(effective_only);
        r += position_embedding.get_param_num(effective_only);
        return r;
    }

    void VisionEmbeddings::load(const std::string &path, TensorLoader *loader)
    {
        patch_embedding.load(path + "patch_embedding.", loader);
        position_embedding.load(path + "position_embedding.", loader);
    }

    using qwen::vit::ViTSelfAttention;
    using qwen::vit::MLP;
    using qwen::vit::TensorPosHelper;

    class VisionTransformer : public DynamicBlock
    {
    public:
        typedef LMBlock1<LayerNorm, ViTSelfAttention, LayerNorm, MLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image,  int grid_h, int grid_w);
    public:
        VisionEmbeddings embeddings;
        std::vector<std::unique_ptr<Block>> layers;
        LayerNorm                           post_layernorm;
        Projector                           mlp_AR;
        std::unique_ptr<TensorPosHelper>    pos_helper;
    };

    VisionTransformer::VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size) :
        embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
        post_layernorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
        mlp_AR(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, lm_hidden_size)
    {
        const int max_length = config.max_patches;
        pos_helper.reset(new TensorPosHelper(max_length, 0, config.patch_size, 1)); // merging is done by Projector

        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
            layer->set_id(layer_id);
            layer->attention.set_pos_helper(pos_helper.get());
            layers.emplace_back(layer);
        }
    }

    int64_t VisionTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embeddings.get_param_num(effective_only);
        r += post_layernorm.get_param_num(effective_only);
        r += mlp_AR.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void VisionTransformer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "embeddings.patch_embedding.weight")) return;

        embeddings.load(path + "embeddings.", loader);
        post_layernorm.load(path + "post_layernorm.", loader);
        mlp_AR.load("mlp_AR.", loader);
        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        _loaded = true;
    }

    ggml::tensor *VisionTransformer::forward(ComputeContext *ctx, ggml::tensor *image,  int grid_h, int grid_w)
    {
        pos_helper->prepare(grid_h, grid_w);

        auto output = embeddings.forward(ctx, image, grid_h, grid_w);

        for (size_t i = 0; i < layers.size(); i++)
        {
            auto l = (LayerBlock *)layers[i].get();
            auto attn = &l->attention;

            attn->grid_h = grid_h;
            attn->grid_w = grid_w;
            output = l->forward(ctx, output, 0);
        }

        output = post_layernorm.forward(ctx, output);
        output = mlp_AR.forward(ctx, output, grid_h, grid_w);

        return output;
    }

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096);
        bool load(ModelLoader &loader);
        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config);
        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);
        VisionTransformer *get_model(void) const;
    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);

    protected:
        const int max_llm_tokens;
        std::unique_ptr<VisionTransformer> vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };

    VisualEmbeddingGeneration::VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "vis", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool VisualEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (vis_model.get())
        {
            loader.add_tensor_name_translations({
                {"mlp_AR.fc0.",                     "mlp_AR.linear_1."},
                {"mlp_AR.fc1.",                     "mlp_AR.linear_2."},
                {".self_attn.o_proj.",              ".self_attn.out_proj."},
                {".input_layernorm.",               ".layer_norm1."},
                {".post_attention_layernorm.",      ".layer_norm2."},
                {".mlp.fc1.",                       ".mlp.fc2."},
                {".mlp.fc0.",                       ".mlp.fc1."},
            });
            loader.push_allocator_manager(eval.get_layer_allocators());
            vis_model->load("visual.", &loader);
            loader.pop_allocator_manager();
            return vis_model->is_loaded();
        }
        else
            return false;
    }

    bool VisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
    {
        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vis_config.dtype = dtype;

        vis_config.num_channels         = 3;
        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
        vis_config.spatial_merge_size   = (int)vis_cfg["spatial_merge_size"].ToInt();

        vis_config.max_patches          = max_llm_tokens * vis_config.spatial_merge_size * vis_config.spatial_merge_size;
        vis_config.posemb_grid_size     = (int)vis_cfg["image_size"].ToInt() / vis_config.patch_size;

        auto pp_cfg = config["preprocessor_config.json"];
        if (pp_cfg.IsObject())
        {
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

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 11 + vis_config.num_hidden_layers * 18;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        vis_model.reset(new VisionTransformer(&_ctx, vis_config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

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

    VisionTransformer *VisualEmbeddingGeneration::get_model(void) const
    {
        return vis_model.get();
    }

    bool VisualEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &media, std::vector<uint8_t> &buf)
    {
        ggml::tensor *image = nullptr;

        const auto make_graph = [this, &image, &media](ComputeContext *ctx) -> ggml::tensor * {
            image = ggml::new_tensor_4d(ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size, vis_config.patch_size, 3, media.grid_width * media.grid_height);
            auto r = vis_model->forward(ctx, image, media.grid_height, media.grid_width);
            return r;
        };
        const auto write_input_data = [&image, &media](ComputeContext *ctx) {
            Backend::write_tensor_data(image, media.data.data());
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }
}

namespace chatllm::paddle::ocr
{
    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        float rope_theta;
        int mrope_section[3];
        int tie_word_embeddings;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    public:
        const vit::Config *vis_config = nullptr;
        bool  vit_loaded              = false;
    };

    static ChatHistoryEncoder           _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void inject_media(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int image_start_tok_id;
        int image_end_tok_id;
    };

    Tokenizer::Tokenizer(const BaseConfig &config):
        BaseTokenizer(config, &_chat_encoder)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);
        bos_token_id        = tp->PieceToId("<|begin_of_sentence|>");
        image_start_tok_id  = tp->PieceToId("<|IMAGE_START|>");
        image_end_tok_id    = tp->PieceToId("<|IMAGE_END|>");
        eos_token_id        = tp->PieceToId("</s>");
        terminate_ids.emplace(tp->PieceToId("<|end_of_sentence|>"));
        if (eos_token_id < 0) eos_token_id = 2;
        return size;
    }

    void Tokenizer::inject_media(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(image_start_tok_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(image_end_tok_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt() + "\n", ids);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("User: ", ids);
        tok->encode(user + "\n", ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("Assistant:\n", ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss;
        tok->encode("User: ", ids);

        for (auto &piece : user.pieces)
        {
            if (ContentPiece::Type::Text == piece.type)
            {
                oss << piece.content;
                continue;
            }
            if (piece.type != ContentPiece::Type::Image) continue;

            int w, h;
            std::vector<uint8_t> pixels;
            const int patch_size = vis_config->patch_size;

            vision::MaxPatchNum     param1(vis_config->max_patches);
            //vision::MergeKernel     param2(vis_config->spatial_merge_size, vis_config->spatial_merge_size);

            vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
            if ((w <= 0) || (h <= 0)) return;

            std::vector<float> scaled;
            vision::image_rescale(pixels, scaled);
            vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

            tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

            auto &image = tok->media_emb.back();

            //vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_MergeN_ChannelsRGB_PixelsLeftRightDown);
            vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_ChannelsRGB_PixelsLeftRightDown);

            const int merge_length = vis_config->spatial_merge_size * vis_config->spatial_merge_size;
            image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

            const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
            tok->inject_media(ids, id_start, image.emb_vec_number);
        }

        if (oss.str().size() == 0)
            oss << "OCR:";
        oss << "\n";
        tok->encode(oss.str(), ids);
    }

    class SelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            rope_mode = RoPEMode::MROPE;
        }
    };

    class LayerBlock : public LMBlock1<RMSNorm, SelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        LayerBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class ConditionalGeneration : public TensorPosHelperPrelude, public qwen::v2_5_vl::ExtendEmbedding, public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<BaseConfig, Embedding, RMSNorm, LayerBlock, int, int, int, int, int, int> ModelClass;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_PADDLE_OCR_VL);

        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        int64_t get_param_num(bool effective_only) const;
        void before_generate(const GenerationConfig &gen_config) override;
        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override;
    protected:
        void before_generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config);
    public:
        std::vector<int> v_pos;
        const Config config;
    protected:
        std::vector<qwen::v2_5_vl::ImageGridSize> images_grid;
        int token_time = 0;
        vit::VisualEmbeddingGeneration visual;
        int mrope_sections[qwen::v2_5_vl::MROPE_SECTION_MAX];
    };

    class TensorPosHelper3D : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper3D(int max_length, int image_id_start, ConditionalGeneration *gen)
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

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override
        {
            pos->ne[0] = (int)(gen->v_pos.size());
            Backend::write_tensor_data(pos, gen->v_pos.data(), 0, gen->v_pos.size() * sizeof(gen->v_pos[0]));
        }
    protected:
        const int original_length;
        const int image_id_start;
        ConditionalGeneration *gen;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        TensorPosHelperPrelude(new TensorPosHelper3D(config.max_length, config.vocab_size, this)),
        qwen::v2_5_vl::ExtendEmbedding(2048, 1),
        BaseModelForConditionalGeneration(type, config, runtime_config),
        config(config),
        visual(runtime_config, pad_arg->get())
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = (config.tie_word_embeddings ? 2 : 3) + config.num_hidden_layers * 12;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        memset(mrope_sections, 0, sizeof(mrope_sections));
        memcpy(mrope_sections, config.mrope_section, sizeof(config.mrope_section));

        if (config.tie_word_embeddings)
            transformer = new ModelClass(&w_ctx_, config, nullptr,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);
        else
            transformer = new ModelClass(&w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);

        {
            auto transformer = get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.freq_base = config.rope_theta;
                attention.mrope_sections = mrope_sections;
            }
        }
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
        BaseModelForConditionalGeneration::load(loader);

        _chat_encoder.vit_loaded = visual.load(loader);
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = BaseModelForConditionalGeneration::get_param_num(effective_only);
        if (_chat_encoder.vit_loaded)
            r += visual.get_model()->get_param_num(effective_only);
        return r;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        images_grid.clear();
        for (auto &mm : tokenizer->media_emb)
        {
            images_grid.emplace_back(mm.grid_width  / visual.vis_config.spatial_merge_size,
                                     mm.grid_height / visual.vis_config.spatial_merge_size);
        }

        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    void ConditionalGeneration::before_generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config)
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
    }

    bool ConditionalGeneration::generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits)
    {
        before_generate_next_token(input_ids, gen_config);
        auto r = BaseModelForConditionalGeneration::generate_next_token(input_ids, gen_config, lm_logits);
        return r;
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(ERNIE_DENSE,           ernie::dense, 1);
    REGISTER_MODEL_LOADER(ERNIE_MOE,             ernie::moe, 1);
    REGISTER_MODEL_LOADER(PADDLE_OCR_VL,         paddle::ocr, 1);
}