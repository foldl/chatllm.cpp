#include "qwen.h"
#include <cstring>
#include "../src/vision_process.h"

namespace chatllm::neochat::vit
{
    const float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
    const float IMAGENET_STD[]  = {0.229f, 0.224f, 0.225f};

    struct Config
    {
        ggml::type dtype;

        int hidden_size;
        int patch_size;
        int min_pixels;
        int max_pixels;
        int merge_size;
        int max_patches;

        float rope_theta;
        float image_mean[3];
        float image_std[3];

        bool add_noise_scale_embedding;
        bool use_deep_fm_head;
        bool use_pixel_head;

        Config()
        {
            memset(this, 0, sizeof(*this));
            memcpy(image_mean, IMAGENET_MEAN, sizeof(image_mean));
            memcpy(image_std,  IMAGENET_STD,  sizeof(image_std));
        }
    };

    class FMHead : public Sequential
    {
    public:
        FMHead(InitContext *ctx, const Config &config, int llm_embed_dim);
    };

    FMHead::FMHead(InitContext *ctx, const Config &config, int llm_embed_dim)
    {
        const int output_dim = 3 * (config.patch_size * config.merge_size) * (config.patch_size * config.merge_size);
        add_block(new Linear(ctx, llm_embed_dim, 4096, true));
        add_block(new ActivationBlock(ActFunc::GELU));
        add_block(new Linear(ctx, 4096, output_dim, true));
    }

    class TimestepEmbedder : public Block
    {
    public:
        TimestepEmbedder(InitContext *ctx, int hidden_size, int frequency_embedding_size = 256);
        int64_t get_param_num(bool effective_only) const override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int frequency_embedding_size;
        Sequential mlp;
    protected:
        ggml::tensor *freqs;
    };

    TimestepEmbedder::TimestepEmbedder(InitContext *ctx, int hidden_size, int frequency_embedding_size):
        frequency_embedding_size(frequency_embedding_size),
        freqs(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, frequency_embedding_size / 2))
    {
        mlp.add_block(new Linear(ctx, frequency_embedding_size, hidden_size, true));
        mlp.add_block(new ActivationBlock(ActFunc::SILU));
        mlp.add_block(new Linear(ctx, hidden_size, hidden_size, true));
        ctx->get_allocator()->alloc(freqs);

        CHATLLM_CHECK((frequency_embedding_size % 2) == 0);

        const int half = frequency_embedding_size / 2;
        const float max_period = 10000.0f;
        std::vector<float> v_freqs(half);
        for (int i = 0; i < half; i++)
        {
            v_freqs[i] = (float)expf(-logf(max_period) * i / half);
        }
        Backend::write_tensor_data(freqs, v_freqs.data());
    }

    int64_t TimestepEmbedder::get_param_num(bool effective_only) const
    {
        return mlp.get_param_num(effective_only);
    }

    ggml::tensor *TimestepEmbedder::forward(ComputeContext *ctx, ggml::tensor *t)
    {
        auto args = ggml::mul(ctx, t, freqs);
        auto c    = ggml::cos(ctx, args);
        auto s    = ggml::sin(ctx, args);
        auto t_freq = ggml::concat(ctx, c, s, 0);

        auto t_emb  = mlp.forward(ctx, t_freq);
        return t_emb;
    }

    void TimestepEmbedder::load(const std::string &path, TensorLoader *loader)
    {
        mlp.load(path + "mlp.", loader);
    }

    class NEOVisionEmbeddings : public DynamicBlock
    {
    public:
        NEOVisionEmbeddings(InitContext *ctx, const Config &config, int llm_embed_dim);
        int64_t get_param_num(bool effective_only) const override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override;
        void load(const std::string &path, TensorLoader *loader) override;
        void before_eval(ComputeContext *ctx) override;
    public:
        const int llm_embed_dim;
        const int patch_size;
        const int hidden_size;
        const float rope_theta;
        Conv2D patch_embedding;
        Conv2D dense_embedding;
    protected:
        ggml::tensor *rt_pos = nullptr;
        int grid_h = 0;
        int grid_w = 0;
    };

    NEOVisionEmbeddings::NEOVisionEmbeddings(InitContext *ctx, const Config &config, int llm_embed_dim):
        llm_embed_dim(llm_embed_dim),
        patch_size(config.patch_size), hidden_size(config.hidden_size),
        rope_theta(config.rope_theta),
        patch_embedding(ctx, 3, config.hidden_size, config.patch_size, config.patch_size),
        dense_embedding(ctx, config.hidden_size, llm_embed_dim, config.merge_size, config.merge_size)
    {
    }

    int64_t NEOVisionEmbeddings::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += patch_embedding.get_param_num(effective_only);
        r += dense_embedding.get_param_num(effective_only);
        return r;
    }

    void NEOVisionEmbeddings::before_eval(ComputeContext *ctx)
    {
        std::vector<int> pos(grid_h * grid_w * 2);
        int *pos_w = pos.data();
        int *pos_h = pos.data() + grid_h * grid_w;
        int c = 0;
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                pos_w[c] = j;
                pos_h[c] = i;
                c += 1;
            }
        }
        Backend::write_tensor_data(rt_pos, pos.data());
    }

    ggml::tensor *NEOVisionEmbeddings::forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
    {
        this->grid_h = grid_h;
        this->grid_w = grid_w;
        rt_pos = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, grid_h * grid_w * 4);
        ggml::set_input(rt_pos);

        auto pixel_values = ggml::reshape(ctx, input, patch_size, patch_size, 3, -1);
        auto patch_embeds = patch_embedding.forward(ctx, pixel_values);
        patch_embeds = ggml::act    (ctx, ActFunc::GELU, patch_embeds);
        patch_embeds = ggml::reshape(ctx, patch_embeds, -1, 1, grid_w * grid_h);

        {
            const float freq_scale(1.0f);
            const float ext_factor(0.0f);
            const float attn_factor(1.0f);
            const float beta_fast(0.0f);
            const float beta_slow(0.0f);
            const int   rope_dim(hidden_size);
            const int   n_original_ctx(0);
            patch_embeds = ggml::rope_2d(ctx, patch_embeds, rt_pos, nullptr, rope_dim, n_original_ctx,
                rope_theta, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, false);
        }
        patch_embeds = ggml::reshape(ctx, patch_embeds, -1, grid_w, grid_h);
        patch_embeds = ggml::permute(ctx, patch_embeds, 2, 0, 1);
        patch_embeds = ggml::cont   (ctx, patch_embeds);
        patch_embeds = dense_embedding.forward(ctx, patch_embeds);
        patch_embeds = ggml::permute(ctx, patch_embeds, 1, 2, 0);
        patch_embeds = ggml::cont   (ctx, patch_embeds);
        patch_embeds = ggml::reshape(ctx, patch_embeds, llm_embed_dim, -1);

        return patch_embeds;
    }

    void NEOVisionEmbeddings::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "patch_embedding.weight")) return;

        patch_embedding.load(path + "patch_embedding.", loader);
        dense_embedding.load(path + "dense_embedding.", loader);
        _loaded = true;
    }

    class FlowMatchingModules
    {
    public:
        FlowMatchingModules(InitContext *ctx, const Config &config, int llm_embed_dim);
        int64_t get_param_num(bool effective_only) const;
        void load(const std::string &path, TensorLoader *loader);
    public:
        NEOVisionEmbeddings                 vision_model_mot_gen;
        TimestepEmbedder                    timestep_embedder;
        std::unique_ptr<Block>              fm_head;
        std::unique_ptr<TimestepEmbedder>   noise_scale_embedder;
    };

    FlowMatchingModules::FlowMatchingModules(InitContext *ctx, const Config &config, int llm_embed_dim):
        vision_model_mot_gen(ctx, config, llm_embed_dim),
        timestep_embedder(ctx, llm_embed_dim)
    {
        CHATLLM_CHECK(!config.use_deep_fm_head);
        fm_head.reset(new FMHead(ctx, config, llm_embed_dim));
        if (config.add_noise_scale_embedding)
            noise_scale_embedder.reset(new TimestepEmbedder(ctx, llm_embed_dim));
    }

    int64_t FlowMatchingModules::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += vision_model_mot_gen.get_param_num(effective_only);
        r += timestep_embedder.get_param_num(effective_only);
        if (fm_head.get())              r += fm_head->get_param_num(effective_only);
        if (noise_scale_embedder.get()) r += noise_scale_embedder->get_param_num(effective_only);
        return r;
    }

    void FlowMatchingModules::load(const std::string &path, TensorLoader *loader)
    {
        vision_model_mot_gen.load(path + "vision_model_mot_gen.embeddings.", loader);
           timestep_embedder.load(path + "timestep_embedder.", loader);

        if (fm_head.get())              fm_head->load(path + "fm_head.", loader);
        if (noise_scale_embedder.get()) noise_scale_embedder->load(path + "noise_scale_embedder.", loader);
    }

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_patches, size_t GRAPH_SIZE = 4096);
        bool load(ModelLoader &loader);
        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config);
        int64_t get_param_num(bool effective_only) const;
        void gen_embedding(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);

    protected:
        bool run_embedding_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf);
    protected:
        const int max_llm_tokens;
        std::unique_ptr<NEOVisionEmbeddings> vis_embedding;
        std::unique_ptr<FlowMatchingModules> fm_modules;
        TensorGraphEvaluator eval;
        InitContext _ctx;
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

    int64_t VisualEmbeddingGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += vis_embedding.get() ? vis_embedding->get_param_num(effective_only) : 0;
        r +=    fm_modules.get() ?    fm_modules->get_param_num(effective_only) : 0;
        return r;
    }

    bool VisualEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (vis_embedding.get())
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            vis_embedding->load("vision.embeddings.", &loader);
               fm_modules->load("fm_modules.", &loader);
            loader.pop_allocator_manager();
        }
        else
            return false;

        return true;
    }

    bool VisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
    {
        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vis_config.dtype = dtype;

        vis_config.hidden_size = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.patch_size  = (int)vis_cfg["patch_size"].ToInt();
        vis_config.min_pixels  = (int)vis_cfg["min_pixels"].ToInt();
        vis_config.max_pixels  = (int)vis_cfg["max_pixels"].ToInt();
        vis_config.merge_size  = (int)(1 / vis_cfg["downsample_ratio"].ToFloat());
        vis_config.rope_theta  = (float)(vis_cfg["rope_theta_vision"].ToFloat());

        vis_config.max_patches = max_llm_tokens * vis_config.merge_size * vis_config.merge_size;

        vis_config.add_noise_scale_embedding = config["config.json"]["add_noise_scale_embedding"].ToBool();
        vis_config.use_deep_fm_head          = config["config.json"]["fm_head_layers"].ToInt() > 2;
        vis_config.use_pixel_head            = config["config.json"]["use_pixel_head"].ToBool();


        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 4 + 16 + 1 + (vis_config.add_noise_scale_embedding ? 1 : 0);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        vis_embedding.reset(new vit::NEOVisionEmbeddings(&_ctx, vis_config, lm_hidden_size));
           fm_modules.reset(new vit::FlowMatchingModules(&_ctx, vis_config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

        return true;
    }

    void VisualEmbeddingGeneration::gen_embedding(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if ((vis_embedding.get() == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!vis_embedding->is_loaded()) return;

        for (auto &image : tok->media_emb)
        {
            run_embedding_model(gen_config, tok, dtype, image, buf);
        }
    }

    bool VisualEmbeddingGeneration::run_embedding_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype,
        const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
    {
        ggml::tensor *img = nullptr;

        const auto make_graph = [this, &img, &image](ComputeContext *ctx) -> ggml::tensor * {
            img = ggml::new_tensor_4d(ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size, vis_config.patch_size, 3, image.grid_width * image.grid_height);
            auto r = vis_embedding->forward(ctx, img, image.grid_height, image.grid_width);
            return r;
        };
        const auto write_input_data = [this, &img, &image](ComputeContext *ctx) {
            vis_embedding->before_eval(ctx);
            Backend::write_tensor_data(img, image.data.data());
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }
};

namespace chatllm::neochat
{
    typedef qwen::v3::Config Config;

    class ChatHistoryEncoder : public qwen::v1::ChatHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override
        {}
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    protected:
        void do_append_sys_prompt(std::vector<int> &ids) const;
    public:
        const vit::Config *vis_config = nullptr;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public qwen::v3::Tokenizer
    {
    public:
        enum GenMode
        {
            VQA,
            ImageGeneration,    // T2I,
            ImageEdit,          // IT2I,
            Interleaved,
        };

        Tokenizer(const BaseConfig &config):
            qwen::v3::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = qwen::v3::Tokenizer::load(buffer, n_vocab);
            img_context_token_id = tp->PieceToId("<IMG_CONTEXT>");
            img_start_token_id   = tp->PieceToId("<img>");
            img_end_token_id     = tp->PieceToId("</img>");
            return r;
        }

        void inject_media(std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
        {
            ids.push_back(img_start_token_id);
            for (int i = 0; i < ids_to_inject_count; i++)
                ids.push_back(ids_to_inject_start + i);
            ids.push_back(img_end_token_id);
        }
    public:
        int img_context_token_id;
        int img_start_token_id;
        int img_end_token_id;

        int res_width = 256;
        int res_height = 256;
        int num_steps = 2;
    public:
        bool think_mode = false;
        GenMode gen_mode = GenMode::VQA;
        const std::string sys_prompt_def_interleave = "You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer.";
        const std::string sys_prompt_def_gen        = "You are an image generation and editing assistant that accurately understands and executes user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\nB. Image Editing:\n- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly changed.\n\nGeneral Rules:\n- For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language.";
    };

    class BaseNeoAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        BaseNeoAttention() : RoPESelfAttention<BaseAttention>() {}

        BaseNeoAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads,
                  int head_dim, int max_length):
            BaseNeoAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length,
                    false, false, max_length)
        {}

        BaseNeoAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
                      bool qkv_bias, bool o_bias,
                      int cache_length) :
            RoPESelfAttention<BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false),
            num_attention_heads(num_attention_heads),
            num_kv_heads(num_kv_heads),
            head_dim(head_dim),
            q_proj_mot_gen(ctx, hidden_size, head_dim * num_attention_heads, nullptr, qkv_bias),
            k_proj_mot_gen(ctx, hidden_size, head_dim * num_kv_heads, nullptr, BlockParams::OverrideKProjBiased::get(qkv_bias)),
            v_proj_mot_gen(ctx, hidden_size, head_dim * num_kv_heads, nullptr, qkv_bias),
            o_proj_mot_gen(ctx, head_dim * num_attention_heads, hidden_size, o_bias),
            q_norm(ctx, head_dim / 2),
            q_norm_mot_gen(ctx, head_dim / 2),
            q_norm_hw(ctx, head_dim / 2),
            q_norm_hw_mot_gen(ctx, head_dim / 2),
            k_norm(ctx, head_dim / 2),
            k_norm_mot_gen(ctx, head_dim / 2),
            k_norm_hw(ctx, head_dim / 2),
            k_norm_hw_mot_gen(ctx, head_dim / 2)
        {
            rope_mode = RoPEMode::Original;
        }

        BaseNeoAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      bool qkv_bias, bool o_bias,
                      int cache_length)
            : BaseNeoAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            cache_length)
        {}

        BaseNeoAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseNeoAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            max_length)
        {}

        BaseNeoAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
             bool qkv_bias, bool o_bias)
            : BaseNeoAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias,
                            max_length)
        {}

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = BaseAttention::get_param_num(effective_only);
            r += q_proj_mot_gen.get_param_num(effective_only);
            r += k_proj_mot_gen.get_param_num(effective_only);
            r += v_proj_mot_gen.get_param_num(effective_only);
            r += o_proj_mot_gen.get_param_num(effective_only);
            r += q_norm.get_param_num(effective_only);
            r += q_norm_mot_gen.get_param_num(effective_only);
            r += q_norm_hw.get_param_num(effective_only);
            r += q_norm_hw_mot_gen.get_param_num(effective_only);
            r += k_norm.get_param_num(effective_only);
            r += k_norm_mot_gen.get_param_num(effective_only);
            r += k_norm_hw.get_param_num(effective_only);
            r += k_norm_hw_mot_gen.get_param_num(effective_only);
            return r;
        }

        void set_id(int id) override
        {
            BaseAttention::set_id(id);

            q_proj_mot_gen.set_id(id);
            k_proj_mot_gen.set_id(id);
            v_proj_mot_gen.set_id(id);
            o_proj_mot_gen.set_id(id);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
            int n_past);

        void before_eval(ComputeContext *ctx) override;

        void load(const std::string &path, TensorLoader *loader) override
        {
            BaseAttention::load(path, loader);
            q_proj_mot_gen.load(path + "q_proj_mot_gen.", loader);
            k_proj_mot_gen.load(path + "k_proj_mot_gen.", loader);
            v_proj_mot_gen.load(path + "v_proj_mot_gen.", loader);
            o_proj_mot_gen.load(path + "o_proj_mot_gen.", loader);
            q_norm        .load(path + "q_norm.", loader);
            q_norm_mot_gen.load(path + "q_norm_mot_gen.", loader);
            q_norm_hw     .load(path + "q_norm_hw.", loader);
            k_norm        .load(path + "k_norm.", loader);
            k_norm_mot_gen.load(path + "k_norm_mot_gen.", loader);
            k_norm_hw     .load(path + "k_norm_hw.", loader);
            q_norm_hw_mot_gen.load(path + "q_norm_hw_mot_gen.", loader);
            k_norm_hw_mot_gen.load(path + "k_norm_hw_mot_gen.", loader);
        }
    protected:
        ggml::tensor * proj_split_norm_rope(ComputeContext *ctx, ggml::tensor *hidden_states, Block *proj, Block *norm_t,
            Block *norm_hw, ggml::tensor *indexes_t, ggml::tensor *indexes_h, ggml::tensor *indexes_w);
        ggml::tensor * split_norm_rope(ComputeContext *ctx, ggml::tensor *states,
            Block *norm_t, Block *norm_hw, ggml::tensor *indexes_t, ggml::tensor *indexes_h, ggml::tensor *indexes_w, const int heads);
        ggml::tensor * split_norm_rope(ComputeContext *ctx, ggml::tensor *states,
            Block *norm_t, Block *norm_t_mot_gen, Block *norm_hw, Block *norm_hw_mot_gen,
            ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
            ggml::tensor *indexes_t, ggml::tensor *indexes_h, ggml::tensor *indexes_w, const int heads);
        ggml::tensor *mixed_proj(ComputeContext *ctx, ggml::tensor *hidden_states, Block *proj, Block *proj_mot_gen,
            ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos);

        ggml::tensor *forward_und(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past);
        ggml::tensor *forward_gen(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past);
        ggml::tensor *forward_mix(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
            int n_past);

        // input & output: [qlen, heads, head_size]
        ggml::tensor *apply_pos_embedding(ComputeContext *ctx, ggml::tensor *k, ggml::tensor * past, float rope_theta)
        {
            const int rope_dim = ggml::get_dim(k, 0);
            return ggml::rope_ext(ctx, k, past, freq_factors, rope_dim, rope_mode, n_original_ctx,
                            rope_theta, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, mrope_sections);
        }
        ggml::tensor *calc_output(ComputeContext *ctx, ggml::tensor *query_states, ggml::tensor *key_states, ggml::tensor *value_states, const int qlen, const int n_past);
    public:
        const int num_attention_heads   = 0;
        const int num_kv_heads          = 0;
        const int head_dim              = 0;
        Linear q_proj_mot_gen, k_proj_mot_gen, v_proj_mot_gen;
        Linear o_proj_mot_gen;
        RMSNorm q_norm;
        RMSNorm q_norm_mot_gen;
        RMSNorm q_norm_hw;
        RMSNorm q_norm_hw_mot_gen;
        RMSNorm k_norm;
        RMSNorm k_norm_mot_gen;
        RMSNorm k_norm_hw;
        RMSNorm k_norm_hw_mot_gen;
        float freq_base_hw = 0.0f;
    public:
        ggml::tensor *rt_indexes_t = nullptr; // fill this before evaluation
        ggml::tensor *rt_indexes_h = nullptr;
        ggml::tensor *rt_indexes_w = nullptr;

        const uint16_t *p_mask_f16 = nullptr;
    protected:
        bool update_cache = true;
        bool use_cache = true;
    };

    void BaseNeoAttention::before_eval(ComputeContext *ctx)
    {
        //return;
        CHATLLM_CHECK(ggml::type_of(rt_mask) == ggml::type::GGML_TYPE_F16);

        Backend::write_tensor_data(rt_mask, p_mask_f16);
    }

    ggml::tensor * BaseNeoAttention::split_norm_rope(ComputeContext *ctx, ggml::tensor *states,
            Block *norm_t, Block *norm_t_mot_gen, Block *norm_hw, Block *norm_hw_mot_gen,
            ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
            ggml::tensor *indexes_t, ggml::tensor *indexes_h, ggml::tensor *indexes_w, const int heads)
    {
        const int batch = (int)ggml::get_dim(states, 2);
        const int qlen  = (int)ggml::get_dim(states, 1);
        auto states_t  = ggml::view_4d(ctx, states, head_dim / 2, heads, qlen, batch,
                            ggml::row_size(states),
                            ggml::row_size(states) * qlen,
                            ggml::row_size(states) * qlen * batch,
                            0);
        auto states_hw  = ggml::view_4d(ctx, states, head_dim / 2, heads, qlen, batch,
                            ggml::row_size(states),
                            ggml::row_size(states) * qlen,
                            ggml::row_size(states) * qlen * batch,
                            ggml::element_size(states) * head_dim / 2);

        states_t  = mixed_proj(ctx, states_t,  norm_t,  norm_t_mot_gen,  image_gen_id_pos, non_image_id_pos);
        states_hw = mixed_proj(ctx, states_hw, norm_hw, norm_hw_mot_gen, image_gen_id_pos, non_image_id_pos);

        auto states_h = ggml::view_4d(ctx, states_hw, head_dim / 4, heads, qlen, batch,
                            ggml::row_size(states_hw),
                            ggml::row_size(states_hw) * qlen,
                            ggml::row_size(states_hw) * qlen * batch,
                            0);
        auto states_w = ggml::view_4d(ctx, states_hw, head_dim / 4, heads, qlen, batch,
                            ggml::row_size(states_hw),
                            ggml::row_size(states_hw) * qlen,
                            ggml::row_size(states_hw) * qlen * batch,
                            ggml::element_size(states_hw) * head_dim / 4);

        states_t = apply_pos_embedding(ctx, states_t, indexes_t, freq_base);
        states_h = apply_pos_embedding(ctx, states_h, indexes_h, freq_base_hw);
        states_w = apply_pos_embedding(ctx, states_w, indexes_w, freq_base_hw);

        states = ggml::concat(ctx, {states_t, states_h, states_w}, 0);
        return states;
    }

    ggml::tensor * BaseNeoAttention::split_norm_rope(ComputeContext *ctx, ggml::tensor *states,
        Block *norm_t, Block *norm_hw, ggml::tensor *indexes_t, ggml::tensor *indexes_h, ggml::tensor *indexes_w, const int heads)
    {
        CHATLLM_CHECK(!ggml::is_quantized(states) && ggml::is_contiguous(states));

        const int batch = (int)ggml::get_dim(states, 2);
        const int qlen  = (int)ggml::get_dim(states, 1);
        const int es    = (int)ggml::element_size(states);

        auto states_t  = ggml::view_4d(ctx, states, head_dim / 2, heads, qlen, batch,
                            es * head_dim,
                            es * head_dim * heads,
                            es * head_dim * heads * qlen,
                            0);
        auto states_hw  = ggml::view_4d(ctx, states, head_dim / 2, heads, qlen, batch,
                            es * head_dim,
                            es * head_dim * heads,
                            es * head_dim * heads * qlen,
                            es * head_dim / 2);

        states_t  = norm_t->forward(ctx, states_t);
        states_hw = norm_hw->forward(ctx, states_hw);

        CHATLLM_CHECK(!ggml::is_quantized(states_hw) && ggml::is_contiguous(states_hw));
        const int es_hw    = (int)ggml::element_size(states_hw);

        auto states_h = ggml::view_4d(ctx, states_hw, head_dim / 4, heads, qlen, batch,
                            es_hw * head_dim / 2,
                            es_hw * head_dim / 2 * heads,
                            es_hw * head_dim / 2 * heads * qlen,
                            0);
        auto states_w = ggml::view_4d(ctx, states_hw, head_dim / 4, heads, qlen, batch,
                            es_hw * head_dim / 2,
                            es_hw * head_dim / 2 * heads,
                            es_hw * head_dim / 2 * heads * qlen,
                            es_hw * head_dim / 4);

        states_t = apply_pos_embedding(ctx, states_t, indexes_t, freq_base);
        states_h = apply_pos_embedding(ctx, states_h, indexes_h, freq_base_hw);
        states_w = apply_pos_embedding(ctx, states_w, indexes_w, freq_base_hw);

        states = ggml::concat(ctx, {states_t, states_h, states_w}, 0);
        return states;
    }

    ggml::tensor * BaseNeoAttention::proj_split_norm_rope(ComputeContext *ctx, ggml::tensor *hidden_states, Block *proj,
        Block *norm_t, Block *norm_hw, ggml::tensor *indexes_t, ggml::tensor *indexes_h, ggml::tensor *indexes_w)
    {
        auto states = proj->forward(ctx, hidden_states);
        const int heads = (int)ggml::get_dim(states, 0) / head_dim;

        states = split_norm_rope(ctx, states, norm_t, norm_hw, indexes_t, indexes_h, indexes_w, heads);
        return states;
    }

    ggml::tensor *BaseNeoAttention::forward_und(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int qlen  = (int)ggml::get_dim(hidden_states, 1);

        CHATLLM_CHECK(ggml::get_dim(rt_indexes_t, 0) == qlen);
        auto indexes_t = rt_indexes_t;
        auto indexes_h = rt_indexes_h;
        auto indexes_w = rt_indexes_w;

        auto query_states = proj_split_norm_rope(ctx, hidden_states, &q_proj, &q_norm, &q_norm_hw, indexes_t, indexes_h, indexes_w);
        auto   key_states = proj_split_norm_rope(ctx, hidden_states, &k_proj, &k_norm, &k_norm_hw, indexes_t, indexes_h, indexes_w);

        auto value_states = v_proj.forward(ctx, hidden_states);

        auto output = calc_output(ctx, query_states, key_states, value_states, qlen, n_past);

        return output;
    }

    ggml::tensor *BaseNeoAttention::forward_gen(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int qlen  = (int)ggml::get_dim(hidden_states, 1);

        CHATLLM_CHECK(ggml::get_dim(rt_indexes_t, 0) == qlen);
        auto indexes_t = rt_indexes_t;
        auto indexes_h = rt_indexes_h;
        auto indexes_w = rt_indexes_w;

        auto query_states = proj_split_norm_rope(ctx, hidden_states, &q_proj_mot_gen, &q_norm_hw, &q_norm_hw_mot_gen, indexes_t, indexes_h, indexes_w);
        auto   key_states = proj_split_norm_rope(ctx, hidden_states, &k_proj_mot_gen, &k_norm_hw, &k_norm_hw_mot_gen, indexes_t, indexes_h, indexes_w);

        auto value_states = v_proj_mot_gen.forward(ctx, hidden_states);

        auto output = calc_output(ctx, query_states, key_states, value_states, qlen, n_past);
        return output;
    }

    ggml::tensor *BaseNeoAttention::mixed_proj(ComputeContext *ctx, ggml::tensor *hidden_states, Block *proj, Block *proj_mot_gen, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos)
    {
        const bool exist_non_image_gen_tokens = non_image_id_pos != nullptr;
        const bool exist_image_gen_tokens     = image_gen_id_pos != nullptr;
        if (exist_non_image_gen_tokens && !exist_image_gen_tokens)
            return proj->forward(ctx, hidden_states);
        if (!exist_non_image_gen_tokens && exist_image_gen_tokens)
            return proj_mot_gen->forward(ctx, hidden_states);

        CHATLLM_CHECK(exist_image_gen_tokens || exist_non_image_gen_tokens);
        CHATLLM_CHECK(false) << "TODO: mix";

        // mixed generation
        auto non_image_part = ggml::get_rows(ctx, hidden_states, non_image_id_pos);
             non_image_part = proj->forward(ctx, non_image_part);
        auto gen_image_part = ggml::get_rows(ctx, hidden_states, image_gen_id_pos);
             gen_image_part = proj_mot_gen->forward(ctx, gen_image_part);
        auto states = ggml::new_tensor_3d(ctx, ggml::type_of(non_image_part),
                                ggml::get_dim(non_image_part, 0),
                                ggml::get_dim(non_image_part, 1) + ggml::get_dim(gen_image_part, 1),
                                ggml::get_dim(non_image_part, 2));
        states = ggml::set_rows(ctx, states, non_image_part, non_image_id_pos);
        states = ggml::set_rows(ctx, states, gen_image_part, image_gen_id_pos);
        return states;
    }

    ggml::tensor *BaseNeoAttention::forward_mix(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
        int n_past)
    {
        const int qlen  = (int)ggml::get_dim(hidden_states, 1);

        CHATLLM_CHECK(ggml::get_dim(rt_indexes_h, 0) == qlen);
        auto indexes_t = rt_indexes_t;
        auto indexes_h = rt_indexes_h;
        auto indexes_w = rt_indexes_w;

        auto query_states = mixed_proj(ctx, hidden_states, &q_proj, &q_proj_mot_gen, image_gen_id_pos, non_image_id_pos);
        auto   key_states = proj_split_norm_rope(ctx, hidden_states, &k_proj_mot_gen, &k_norm_hw, &k_norm_hw_mot_gen, indexes_t, indexes_h, indexes_w);
             query_states = split_norm_rope(ctx, query_states, &q_norm, &q_norm_mot_gen, &q_norm_hw, &q_norm_hw_mot_gen, image_gen_id_pos, non_image_id_pos, indexes_t, indexes_h, indexes_w, num_attention_heads);
               key_states = split_norm_rope(ctx, key_states,   &k_norm, &k_norm_mot_gen, &k_norm_hw, &k_norm_hw_mot_gen, image_gen_id_pos, non_image_id_pos, indexes_t, indexes_h, indexes_w, num_kv_heads);

        auto value_states = v_proj_mot_gen.forward(ctx, hidden_states);

        auto output = calc_output(ctx, query_states, key_states, value_states, qlen, n_past);
        return output;
    }

    ggml::tensor *BaseNeoAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
        int n_past)
    {
        const int batch = (int)ggml::get_dim(hidden_states, 2);
        const int qlen  = (int)ggml::get_dim(hidden_states, 1);

        const bool has_image_gen = ggml::nelements(image_gen_id_pos) > 0;
        const bool has_non_image = ggml::nelements(non_image_id_pos) > 0;

        CHATLLM_CHECK(batch == 1) << "is this possible to be supported?";

        RoPESelfAttention<BaseAttention>::before_forward(ctx, n_past, qlen);

        rt_indexes_t = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, qlen, batch);
        rt_indexes_h = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, qlen, batch);
        rt_indexes_w = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, qlen, batch);
        rt_mask      = ggml::new_tensor_3d(ctx, ggml::type::GGML_TYPE_F16, n_past + qlen, qlen, batch);

        if (has_image_gen && !has_non_image)
        {
            auto r = forward_gen(ctx, hidden_states, n_past);
            return r;
        }

        if (has_image_gen && has_non_image)
        {
            auto r = forward_mix(ctx, hidden_states, image_gen_id_pos, non_image_id_pos, n_past);
            return r;
        }

        auto r = forward_und(ctx, hidden_states, n_past);
        return r;
    }

    ggml::tensor *BaseNeoAttention::calc_output(ComputeContext *ctx, ggml::tensor *query_states, ggml::tensor *key_states, ggml::tensor *value_states, const int qlen, const int n_past)
    {
        const int batch = (int)ggml::get_dim(value_states, 2);

        if (use_cache)
        {
            if (update_cache)
            {
                save_to_cache(ctx, n_past, qlen, key_states, value_states);
                key_states = get_k_from_cache(ctx, num_kv_heads * head_dim, n_past, qlen);
                value_states = get_shaped_v_from_cache(ctx, num_kv_heads * head_dim, n_past, qlen, VShapeFromCache::Len_HeadSize_Heads_Batch);
            }
            else
            {
                auto   key_cache = get_k_from_cache(ctx, num_kv_heads * head_dim, n_past, 0);
                auto value_cache = get_shaped_v_from_cache(ctx, num_kv_heads * head_dim, n_past, 0, VShapeFromCache::Len_HeadSize_Heads_Batch);
                key_states   = ggml::concat(ctx,   key_cache, key_states, 0);

                value_states = ggml::transpose(ctx, value_states);
                value_states = ggml::reshape(ctx, value_states, qlen, head_dim, num_kv_heads, batch);
                value_states = ggml::concat(ctx, value_cache, value_states, 0);
            }
        }

        query_states = ggml::permute(ctx, query_states, 0, 2, 1, 3);                     // [heads, qlen, head_size]

        auto attn_scores = calc_attn_scores(ctx, head_dim * num_attention_heads, n_past, qlen, key_states, query_states, value_states);
        auto attn_output = o_proj.forward(ctx, attn_scores);

        return attn_output;
    }

    class LayerBlockForward : public LMBlock1Forward
    {
    public:
        using LMBlock1Forward::LMBlock1Forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states,  ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos, int n_past);
    };

    ggml::tensor *LayerBlockForward::forward(ComputeContext *ctx, ggml::tensor *hidden_states,  ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos, int n_past)
    {
        auto attention = dynamic_cast<BaseNeoAttention *>(this->attention);

        ggml::tensor *residual = hidden_states;

        hidden_states = input_layernorm->forward(ctx, hidden_states);

        hidden_states = attention->forward(ctx, hidden_states, image_gen_id_pos, non_image_id_pos, n_past);

        if (attn_scale)
        {
            hidden_states = ggml::mul(ctx, hidden_states, attn_scale);
        }
        else if (scale_depth > 0.0f)
        {
            hidden_states = ggml::scale(ctx, hidden_states, scale_depth);
        }
        else;

        hidden_states = ggml::add(ctx, hidden_states, residual);
        residual = hidden_states;

        hidden_states = post_attention_layernorm->forward(ctx, hidden_states);
        last_result_post_attn_norm = hidden_states;

        hidden_states = mlp->forward(ctx, hidden_states);

        if (attn_scale)
        {
            hidden_states = ggml::mul(ctx, hidden_states, mlp_scale);
        }
        else if (scale_depth > 0.0f)
        {
            hidden_states = ggml::scale(ctx, hidden_states, scale_depth);
        }
        else;

        hidden_states = ggml::add(ctx, hidden_states, residual);
        //inspect_tensor(hidden_states, "layer[%].output", id);

        return hidden_states;
    }

    template <class MLPBlock> class LayerBlock : public LMBlock1<RMSNorm, BaseNeoAttention, RMSNorm, MLPBlock>
    {
    private:
        typedef LMBlock1<RMSNorm, BaseNeoAttention, RMSNorm, MLPBlock> Base;
    public:
        using LMBlock1<RMSNorm, BaseNeoAttention, RMSNorm, MLPBlock>::LMBlock1;

        void before_eval(ComputeContext *ctx) override
        {
            Base::attention.before_eval(ctx);
        }

        using Base::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
            int n_past)
        {
            LayerBlockForward eval(&(Base::input_layernorm), &(Base::attention), &(Base::post_attention_layernorm), &(Base::mlp), Base::get_id(), Base::scale_depth);
            hidden_states = eval.forward(ctx, hidden_states, image_gen_id_pos, non_image_id_pos, n_past);
            Base::last_result_post_attn_norm = eval.last_result_post_attn_norm;
            return hidden_states;
        }
    };

    class DenseBlock : public LayerBlock<SiLUMLP>
    {
    public:
        DenseBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length);
    };

    DenseBlock::DenseBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
        : LayerBlock<SiLUMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
    {}

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class SparseMoE : public BaseSparseMLP
    {
    public:
        SparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class MoEBlock : public LayerBlock<SparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>
    {
    public:
        typedef SparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK> MoEMLP;
    public:
        MoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size,
                int num_kv_heads,
                int head_dim, int max_length)
            : LayerBlock<SparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size,
            num_kv_heads, head_dim, max_length)
        {}
    };

    typedef std::function<ggml::tensor *(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
            int n_past)> layer_forward_func;

    class Prelude : public qwen::v2_5_vl::ExtendEmbedding
    {
    public:
        Prelude(int per_image, int image_num) :
            qwen::v2_5_vl::ExtendEmbedding(per_image, image_num)
        {
        }
    };

    class ConditionalGeneration : public Prelude, public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type = ModelType::MODEL_TYPE_NEOCHAT);

        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        int64_t get_param_num(bool effective_only) const override;
        int get_sparse_layer_num() const;

        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  BaseStreamer *streamer) override;
        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override;
    protected:
        void before_eval_model(ComputeContext *ctx) override;
        void before_generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config);
        void before_generate(const GenerationConfig &gen_config) override;
        void before_run_model(const int *input_ids, const int ids_count,
                               const GenerationConfig &gen_config,
                               int past) override;
        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output, const int batch_size,
                            std::function<ggml::tensor *(ComputeContext *, ggml::tensor *)> func_epilog) override;

        void vqa_gen(const GenerationConfig &gen_config, const int gen_max_tokens,
                                const bool continuous,
                                bool &completed,
                                ModelPerfInfo *performance,
                                BaseStreamer *streamer,
                                std::vector<int> &output_ids,
                                std::vector<int> &cur_input_ids,
                                Sampler *sampler);
        void t2i_gen(const GenerationConfig &gen_config, const int gen_max_tokens,
                                const bool continuous,
                                bool &completed,
                                ModelPerfInfo *performance,
                                BaseStreamer *streamer,
                                std::vector<int> &output_ids,
                                std::vector<int> &cur_input_ids,
                                Sampler *sampler);
    public:
        const Config config;
        std::vector<BaseNeoAttention *> layer_attentions;
        std::vector<layer_forward_func> layer_forward_funcs;
    protected:
        vit::VisualEmbeddingGeneration visual;
        std::vector<qwen::v2_5_vl::ImageGridSize> images_grid;
        std::vector<int> v_pos;
        std::vector<uint16_t> v_mask_f16;
        int token_time = 0;
        int parallel_size = 1;
    };

    static Block *create_layer(const Config &config, InitContext *ctx, ConditionalGeneration *prelude, int layer_index)
    {
        if (config.layer_is_sparse[layer_index])
        {
            if ((config.num_experts_per_tok == 8) && (config.num_experts == 128))
            {
                auto r = new MoEBlock<128, 8>(ctx, config.hidden_size, config.num_attention_heads,
                    config.intermediate_size, config.moe_intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length);
                prelude->layer_attentions[layer_index] = &r->attention;
                prelude->layer_forward_funcs[layer_index] = [r](ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
                    int n_past)
                {
                    return r->forward(ctx, hidden_states, image_gen_id_pos, non_image_id_pos, n_past);
                };
                r->attention.freq_base    = config.rope_theta;
                r->attention.freq_base_hw = 10000.0f;
                r->mlp.norm_topk_prob = config.norm_topk_prob != 0;
                return r;
            }
            else
            {
                CHATLLM_CHECK(false) << "unsupported MoE param";
                return nullptr;
            }
        }
        else
        {
            auto r = new DenseBlock(ctx, config.hidden_size, config.num_attention_heads,
                config.intermediate_size,
                config.num_key_value_heads, config.head_dim, config.max_length);
            prelude->layer_attentions[layer_index] = &r->attention;
            prelude->layer_forward_funcs[layer_index] = [r](ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *image_gen_id_pos, ggml::tensor *non_image_id_pos,
                int n_past)
            {
                return r->forward(ctx, hidden_states, image_gen_id_pos, non_image_id_pos, n_past);
            };
            r->attention.freq_base    = config.rope_theta;
            r->attention.freq_base_hw = 10000.0f;
            return r;
        }
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config,
            ModelType type):
        Prelude(2028, 5),
        BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
        config(config),
        visual(runtime_config, pad_arg->get() / image_num)
    {
        layer_attentions.resize(config.num_hidden_layers);
        layer_forward_funcs.resize(config.num_hidden_layers);

        const int added_tensors = config.num_hidden_layers * 10;

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const int sparse_layers = get_sparse_layer_num();
        const size_t num_tensors =    (config.tie_word_embeddings ? 2 : 3)
                                    + (config.num_hidden_layers - sparse_layers) * 14
                                    + sparse_layers * (14 + 1)
                                    + added_tensors;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        auto cb_create_tensor = [this, &config](InitContext *ctx, int layer_index) {
                return create_layer(config, ctx, this, layer_index);
        };

        if (config.tie_word_embeddings)
        {
            transformer = new HeterogeneousModel(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                nullptr,
                cb_create_tensor);
        }
        else
        {
            transformer = new HeterogeneousModel(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false),
                cb_create_tensor);
        }

        if (config.yarn_scaling_factor > 0.0)
            ggml::log(GGML_LOG_LEVEL_WARN, "TODO: YaRN (yarn_scaling_factor = %f) not implemented", config.yarn_scaling_factor);

        w_ctx_.check_used_mem_size(true, 0);

        delete pad_arg;
        pad_arg = nullptr;
        multi_turn = false;
    }

    int ConditionalGeneration::get_sparse_layer_num() const
    {
        int num = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (config.layer_is_sparse[i])
                num++;
        }
        return num;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        bool r = BaseModelForConditionalGeneration::load_more(config);

        visual.load_more(this->config.dtype, this->config.hidden_size, config);
        return r;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        BaseModelForConditionalGeneration::load(loader);
        if (visual.load(loader))
        {
            _chat_encoder.vis_config = &visual.vis_config;
        }
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {

    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += BaseModelForConditionalGeneration::get_param_num(effective_only);
        r += visual.get_param_num(effective_only);
        return r;
    }

    void ConditionalGeneration::before_generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config)
    {
        const int image_id_start = config.vocab_size;
        const int length = (int)input_ids.size();

        // TODO:
        int token_n_inc = 1;

        v_pos.clear();
        v_pos.resize(length * 3, 0);
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
                p_h[i] = 0;
                p_w[i] = 0;
                i += 1;
                t += token_n_inc;
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
                        i += 1;
                    }
                }
            }
            t += token_n_inc;
        }

        token_time = t;
    }

    void ConditionalGeneration::before_eval_model(ComputeContext *ctx)
    {
        BaseModelForConditionalGeneration::before_eval_model(ctx);
        const size_t len = v_pos.size() / 3;
        for (int i = 0; i < (int)layer_attentions.size(); i++)
        {
            Backend::write_tensor_data(layer_attentions[i]->rt_indexes_t, v_pos.data() + 0);
            Backend::write_tensor_data(layer_attentions[i]->rt_indexes_h, v_pos.data() + len * 1);
            Backend::write_tensor_data(layer_attentions[i]->rt_indexes_w, v_pos.data() + len * 2);
        }
    }

    void ConditionalGeneration::vqa_gen(const GenerationConfig &gen_config,
                                const int gen_max_tokens,
                                const bool continuous,
                                bool &completed,
                                ModelPerfInfo *performance,
                                BaseStreamer *streamer,
                                std::vector<int> &output_ids,
                                std::vector<int> &curr_input_ids,
                                Sampler *sampler)
    {
        bool first_call = true;
        int next_output_idx = 0;

        while (!aborted && !completed && (n_past + (int)curr_input_ids.size() < gen_config.max_length))
        {
            std::vector<float> lm_logits;
            const int last_n_past = n_past;
            if (!generate_next_token(curr_input_ids, gen_config, lm_logits))
            {
                ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
                aborted = true;
                break;
            }

            if (lm_logits.size() == 0)
            {
                int num = n_past > last_n_past ? n_past - last_n_past : 0;
                performance->Accumulate(ModelPerfInfo::Type::Generation, num);
                completed = true;
                break;
            }

            if (first_call)
            {
                if (performance)
                    performance->Accumulate(ModelPerfInfo::Type::Prompt, curr_input_ids.size());
                first_call = false;
            }


            n_past += (int)curr_input_ids.size();
            curr_input_ids.clear();

            float *logits = lm_logits.data();
            const size_t tok_num = lm_logits.size() / config_.vocab_size;

            for (size_t tok_idx = 0; (tok_idx < tok_num) && !aborted; tok_idx++, logits +=  config_.vocab_size)
            {
                int next_token_id = sampler->sampling(logits,  config_.vocab_size);

//printf("\n>>next = %d<<\n", next_token_id);
//fflush(stdout);
//exit(-1);

                if (next_token_id == Sampler::ABORT)
                {
                    aborted = true;
                    break;
                }

                curr_input_ids.push_back(next_token_id);

                int pop_output = 0;
                int keep_idx = 0;
                output_ids.push_back(next_token_id);

                if (is_output_terminated(output_ids, keep_idx, pop_output))
                {
                    while (pop_output-- > 0)
                        output_ids.pop_back();
                    keep_idx = (int)output_ids.size();
                    completed = true;
                }

                if (streamer)
                {
                    if (keep_idx > (int)output_ids.size())
                        keep_idx = (int)output_ids.size();
                    for (; next_output_idx < keep_idx; next_output_idx++)
                        streamer->put({output_ids[next_output_idx]});
                }

                if ((gen_max_tokens > 0) && ((n_past + (int)curr_input_ids.size() >= gen_max_tokens)))
                {
                    aborted = true;
                    break;
                }
            }
        }
    }

    void ConditionalGeneration::t2i_gen(const GenerationConfig &gen_config,
                                const int gen_max_tokens,
                                const bool continuous,
                                bool &completed,
                                ModelPerfInfo *performance,
                                BaseStreamer *streamer,
                                std::vector<int> &output_ids,
                                std::vector<int> &curr_input_ids,
                                Sampler *sampler)
    {
        n_past = 0;
        n_past_offset = 0;

        bool first_call = true;
        int next_output_idx = 0;

        while (!aborted && !completed && (n_past + (int)curr_input_ids.size() < gen_config.max_length))
        {
            std::vector<float> lm_logits;
            const int last_n_past = n_past;
            if (!generate_next_token(curr_input_ids, gen_config, lm_logits))
            {
                ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
                aborted = true;
                break;
            }

            if (lm_logits.size() == 0)
            {
                int num = n_past > last_n_past ? n_past - last_n_past : 0;
                performance->Accumulate(ModelPerfInfo::Type::Generation, num);
                completed = true;
                break;
            }

            if (first_call)
            {
                if (performance)
                    performance->Accumulate(ModelPerfInfo::Type::Prompt, curr_input_ids.size());
                first_call = false;
            }


            n_past += (int)curr_input_ids.size();
            curr_input_ids.clear();

            float *logits = lm_logits.data();
            const size_t tok_num = lm_logits.size() / config_.vocab_size;

            for (size_t tok_idx = 0; (tok_idx < tok_num) && !aborted; tok_idx++, logits +=  config_.vocab_size)
            {
                int next_token_id = sampler->sampling(logits,  config_.vocab_size);

//printf("\n>>next = %d<<\n", next_token_id);
//fflush(stdout);
//exit(-1);

                if (next_token_id == Sampler::ABORT)
                {
                    aborted = true;
                    break;
                }

                curr_input_ids.push_back(next_token_id);

                int pop_output = 0;
                int keep_idx = 0;
                output_ids.push_back(next_token_id);

                if (is_output_terminated(output_ids, keep_idx, pop_output))
                {
                    while (pop_output-- > 0)
                        output_ids.pop_back();
                    keep_idx = (int)output_ids.size();
                    completed = true;
                }

                if (streamer)
                {
                    if (keep_idx > (int)output_ids.size())
                        keep_idx = (int)output_ids.size();
                    for (; next_output_idx < keep_idx; next_output_idx++)
                        streamer->put({output_ids[next_output_idx]});
                }

                if ((gen_max_tokens > 0) && ((n_past + (int)curr_input_ids.size() >= gen_max_tokens)))
                {
                    aborted = true;
                    break;
                }
            }
        }
    }

    std::vector<int> ConditionalGeneration::generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                const bool continuous,
                                bool &completed,
                                ModelPerfInfo *performance,
                                BaseStreamer *streamer)
    {
        CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
            << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
            << config_.max_length << ")";

        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::unique_ptr<Sampler> sampler = std::unique_ptr<Sampler>(SamplerFactory::Create(gen_config));

        aborted = false;

        std::vector<int> curr_input_ids(input_ids);

        std::vector<int> output_ids;
        output_ids.reserve(gen_config.max_length);

        n_past = 0;
        n_past_offset = 0;

        completed = false;

        transformer->set_ctx((int)input_ids.size());

        int gen_max_tokens = gen_config.max_new_tokens;
        if (gen_max_tokens > 0)
            gen_max_tokens = n_past + (int)curr_input_ids.size() + gen_max_tokens;

        if ((auto_output_prefix.size() > 0) && streamer)
            streamer->put(auto_output_prefix);

        if (performance)
            performance->Reset();

        before_generate(gen_config);

        switch (tok->gen_mode)
        {
        case Tokenizer::GenMode::VQA:
            vqa_gen(gen_config, gen_max_tokens, continuous, completed, performance, streamer, output_ids, curr_input_ids, sampler.get());
            break;
        case Tokenizer::GenMode::ImageGeneration:
            t2i_gen(gen_config, gen_max_tokens, continuous, completed, performance, streamer, output_ids, curr_input_ids, sampler.get());
            break;
        default:
            aborted = true;
            break;
        }

        if (aborted && !completed)
            completed = true;

        if (performance)
        {
            size_t num = output_ids.size() > curr_input_ids.size() ? output_ids.size() - curr_input_ids.size() : 0;
            performance->Accumulate(ModelPerfInfo::Type::Generation, num);
        }

        after_generate();

        return output_ids;
    }

    bool ConditionalGeneration::generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits)
    {
        before_generate_next_token(input_ids, gen_config);
        auto r = BaseModelForConditionalGeneration::generate_next_token(input_ids, gen_config, lm_logits);
        return r;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        images_grid.clear();
        for (auto &mm : tokenizer->media_emb)
        {
            images_grid.emplace_back(mm.grid_width  / visual.vis_config.merge_size,
                                     mm.grid_height / visual.vis_config.merge_size);
        }

        auto emb = dynamic_cast<Embedding *>((transformer)->word_embeddings);
        visual.gen_embedding(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    void ConditionalGeneration::before_run_model(const int *input_ids, const int ids_count,
                               const GenerationConfig &gen_config,
                               int past)
    {
        BaseModelForConditionalGeneration::before_run_model(input_ids, ids_count, gen_config, past);

        if (nullptr == input_ids) return;

        const int image_id_start = config.vocab_size;

        const int64_t qlen = ids_count;
        const int64_t n_kv = n_past + qlen;

        v_mask_f16.clear();
        v_mask_f16.resize(n_kv * qlen);

        const uint16_t _inf = ggml::fp32_to_fp16(- INFINITY);

        for (int64_t j = 0; j < qlen; j++)
        {
            bool in_image = input_ids[j] >= image_id_start;
            for (int64_t i = 1 + j + n_past; i < n_kv; i++)
            {
                if (input_ids[i - n_past] < image_id_start)
                    in_image = false;

                if (in_image)
                {
                    ; // non-causual
                }
                else
                {
                    v_mask_f16[n_kv * j + i] = _inf;
                }
            }
        }

        for (auto p : layer_attentions)
            p->p_mask_f16 = v_mask_f16.data();
    }

    bool ConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output, const int batch_size,
                            std::function<ggml::tensor *(ComputeContext *, ggml::tensor *)> func_epilog)
    {
        if (!initial_run)
        {
            initial_run = true;
            n_past = gen_config.max_length / transformer->get_reserved_batch_size() - ids_count;
            if (n_past < 0) n_past = 0;
            if (!before_initial_run(ids_count, gen_config, n_past))
                return false;
            n_past = 0;
        }

        before_run_model(input_ids, ids_count, gen_config, past);

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);

        //ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);
        ggml::tensor *r = nullptr;
        {
            transformer->before_forward(&ctx, input_ids_tensor, n_past);

            ctx.move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor *hidden_states = transformer->word_embeddings->forward(&ctx, input_ids_tensor);
            for (int i = 0; i < transformer->get_layer_num(); i++)
            {
                ctx.move_to_layer(i);

                hidden_states = layer_forward_funcs[i](&ctx, hidden_states, nullptr, nullptr, n_past);
            }

            transformer->last_hidden_state = hidden_states;

            ctx.move_to_layer(LayerAllocatorManager::Epilog);
            r = transformer->get_final_steps()->forward(transformer, &ctx, input_ids_tensor, hidden_states);
        }

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (func_epilog)
        {
            r = func_epilog(&ctx, r);
        }
        else
        {
            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);
        }

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK((r->type == GGML_TYPE_F32) || (r->type == GGML_TYPE_I32)) << "output type must be float/int32: " << r->type;

        output.resize(ggml::nbytes(r) / sizeof(output[0]));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, input_ids);

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        before_eval_model(&ctx);
        ctx.compute();

        Backend::read_tensor_data(r, output.data());

        ctx.reset();

        return true;
    }

    void ChatHistoryEncoder::do_append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::string s = tok->get_system_prompt();
        if (s == "")
        {
            switch (tok->gen_mode)
            {
            case Tokenizer::GenMode::ImageGeneration:
            case Tokenizer::GenMode::ImageEdit:
                s = tok->sys_prompt_def_gen;
                break;
            case Tokenizer::GenMode::Interleaved:
                s = tok->sys_prompt_def_interleave;
            default:
                break;
            }
        }
        if (s.size() < 1) return;
        tok->encode("system", ids, true, false, true);
        tok->encode(s, ids, false, true, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        tok->gen_mode = Tokenizer::GenMode::VQA;
        if (user.starts_with("/gen "))
        {
            tok->gen_mode = Tokenizer::GenMode::ImageGeneration;
            oss << user.substr(5);
        }
        else
        {
            oss << user;
        }

        if (0 == round_idx) do_append_sys_prompt(ids);

        tok->encode("user", ids, true, false, true);
        tok->encode(oss.str(), ids, false, true, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("assistant", ids, true, false, true);
        if (tok->think_mode)
            tok->encode("<think>\n", ids, false, false, false);
        else
        {
            tok->encode("<think>\n\n</think>\n\n", ids, false, false, false);

            switch (tok->gen_mode)
            {
            case Tokenizer::GenMode::ImageGeneration:
                ids.push_back(tok->im_start_token_id);
                break;

            default:
                break;
            }
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::vector<ContentPiece> pieces;
        tok->gen_mode = Tokenizer::GenMode::VQA;
        bool first_text = true;
        bool found_image = false;
        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Image) found_image = true;

            if (!first_text || (piece.type != ContentPiece::Type::Text))
            {
                pieces.push_back(piece);
                continue;
            }

            first_text = false;

            if (piece.content.starts_with("/gen "))
            {
                tok->gen_mode = Tokenizer::GenMode::ImageGeneration;
                pieces.push_back(piece.content.substr(5));
            }
            else if (piece.content.starts_with("/edit "))
            {
                tok->gen_mode = Tokenizer::GenMode::ImageEdit;
                pieces.push_back(piece.content.substr(6));
            }
            else if (piece.content.starts_with("/interleave "))
            {
                tok->gen_mode = Tokenizer::GenMode::Interleaved;
                pieces.push_back(piece.content.substr(12));
            }
            else
                pieces.push_back(piece);
        }

        if ((Tokenizer::GenMode::ImageGeneration == tok->gen_mode) && found_image)
            tok->gen_mode = Tokenizer::GenMode::ImageEdit;

        if (0 == round_idx) do_append_sys_prompt(ids);

        append_user_opening(round_idx, ids);

        for (auto &piece : pieces)
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

                vision::MergeKernel     param1(vis_config->merge_size, vis_config->merge_size);
                vision::MaxPatchNum     param2(vis_config->max_pixels / patch_size);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_ChannelsRGB_PixelsLeftRightDown);

                const int merge_length = vis_config->merge_size * vis_config->merge_size;
                image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_media(ids, id_start, image.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
        tok->encode("", ids, false, true, true);
    }
}

namespace chatllm::neochat
{
    REGISTER_MODEL_LOADER(NEOCHAT,          neochat, 1);
}