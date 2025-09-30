#include "siglip.h"

namespace chatllm::siglip
{
    PatchEmbedding::PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int image_width, int in_dim)
        : num_patches_per_side(image_width / patch_size),
            num_patches(num_patches_per_side * num_patches_per_side),
            patch_embedding(ctx, in_dim, out_dim, patch_size, patch_size),
            position_embedding(TypeChanger(ctx, ggml::type::GGML_TYPE_F32), num_patches, out_dim)
    {
    }

    ggml::tensor *PatchEmbedding::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        ggml::tensor *x = patch_embedding.forward(ctx, input);
        x = ggml::reshape_3d(ctx, x, ggml::get_dim(x, 0) * ggml::get_dim(x, 1), ggml::get_dim(x, 2), ggml::get_dim(x, 3));
        x = ggml::transpose(ctx, x);
        x = ggml::cont(ctx, x);
        x = ggml::add(ctx, x, position_embedding.weight);
        return x;
    }

    int64_t PatchEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r +=    patch_embedding.get_param_num(effective_only);
        r += position_embedding.get_param_num(effective_only);
        return r;
    }

    void PatchEmbedding::load(const std::string &path, TensorLoader *loader)
    {
           patch_embedding.load(path + "patch_embed.", loader);
        position_embedding.load(path + "pos_embed.", loader);
    }


    MLPProjector::MLPProjector(InitContext *ctx, int in_dim, int out_dim, ActFunc act, bool bias)
        : TheMLP(ctx, in_dim, out_dim, out_dim, act, bias)
    {
    }

    ViTSelfAttention::ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
        : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true)
    {
        causal = false;
    }

    AttentionPoolLatent::AttentionPoolLatent(InitContext *ctx,
        int embed_dim,
        int num_heads,
        int max_length,
        float mlp_ratio) : BaseCachelessAttention(ctx, embed_dim, num_heads, num_heads, max_length, true, true),
        latent(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, embed_dim)),
        norm(ctx, embed_dim),
        mlp(ctx, embed_dim, (int)(embed_dim * mlp_ratio))
    {
    }

    ggml::tensor *AttentionPoolLatent::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        const int hidden_size = o_proj.in_features();
        const int qlen = ggml::get_dim(input, 1);
        const int latent_len = ggml::get_dim(latent, 1);

        ggml::tensor *q = q_proj.forward(ctx, latent);
        ggml::tensor *k = k_proj.forward(ctx, input);
        ggml::tensor *v = v_proj.forward(ctx, input);

        const int head_size = hidden_size / num_attention_heads;

        // [qlen, heads, head_size]
        ggml::tensor * key_layer = ggml::reshape_3d(ctx, k, head_size, num_kv_heads, qlen);
        // [qlen, heads, head_size] -> [heads, qlen, head_size]
        key_layer = ggml::permute(ctx, key_layer, 0, 2, 1, 3);

        // [qlen, heads, head_size]
        ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, head_size, num_attention_heads, latent_len);
        query_layer = ggml::permute(ctx, query_layer, 0, 2, 1, 3);                     // [heads, latent_len, head_size]

        ggml::tensor *value_layer = ggml::reshape_3d(ctx, v, head_size, num_kv_heads, qlen);  // -> [qlen, heads, head_size]
        value_layer = ggml::permute(ctx, value_layer, 1, 2, 0, 3);   // [heads, head_size, qlen]
        value_layer = ggml::cont(ctx, value_layer);

        // note auto-broadcasting in ggml_mul_mat for `repeat > 1`
        ggml::tensor *attn_scores = ggml::mul_mat(ctx, key_layer, query_layer); // [heads, qlen, klen]
        ggml::mul_mat_set_prec(attn_scores, GGML_PREC_F32);

        attn_scores = ggml::scale(ctx, attn_scores, 1.f / sqrtf((float)head_size));

        // attn_probs = soft_max(attn_masked)
        ggml::tensor * attn_probs = ggml::soft_max_inplace(ctx, attn_scores);

        ggml::tensor *context_layer = ggml::mul_mat(ctx, value_layer, attn_probs); // [heads, qlen, head_size]
        context_layer = ggml::reshape_2d(ctx,
            ggml::cont(ctx, ggml::permute(ctx, context_layer, 0, 2, 1, 3)),
            hidden_size, latent_len);

        ggml::tensor *attn_output = o_proj.forward(ctx, context_layer);

        auto output = norm.forward(ctx, attn_output);
        output = mlp.forward(ctx, output);
        output = ggml::add(ctx, attn_output, output);

        return output;
    }

    int64_t AttentionPoolLatent::get_param_num(bool effective_only) const
    {
        int64_t r = BaseCachelessAttention::get_param_num(effective_only);
        r += ggml::nelements(latent);
        r += norm.get_param_num(effective_only);
        r += mlp.get_param_num(effective_only);
        return r;
    }

    void AttentionPoolLatent::load(const std::string &path, TensorLoader *loader)
    {
        BaseCachelessAttention::load(path, loader);
        loader->read_tensor(path + "latent", latent);
        norm.load(path + "norm.", loader);
        mlp.load(path + "mlp.", loader);
    }

    MultiModalProjector::MultiModalProjector(InitContext *ctx, const Config &config, int max_length)
        :
        image_size(config.image_size),
        hidden_size(config.width),
        patch_size(config.patch_size),
        norm(ctx, hidden_size),
        attn_pool(config.global_pool == CONFIG_GLOBAL_POOL_MAP ? new AttentionPoolLatent(ctx, config.width, config.heads, max_length, config.mlp_ratio) : nullptr),
        aligner(ctx, config.width, config.lm_hidden_size)
    {
    }

    ggml::tensor *MultiModalProjector::forward(ComputeContext *ctx, ggml::tensor *image_features)
    {
        auto output = norm.forward(ctx, image_features);
        if (attn_pool.get())
            output = attn_pool->forward(ctx, output);
        output = aligner.forward(ctx, output);
        return output;
    }

    int64_t MultiModalProjector::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += norm.get_param_num(effective_only);
        if (attn_pool.get())
            r += attn_pool->get_param_num(effective_only);
        r += aligner.get_param_num(effective_only);
        return r;
    }

    void MultiModalProjector::load(const std::string &path, TensorLoader *loader)
    {
        norm.load(path + "norm.", loader);
        if (attn_pool.get())
            attn_pool->load(path + "attn_pool.", loader);
        aligner.load("aligner.", loader);
    }

    VisionTransformer::VisionTransformer(InitContext *ctx, const Config &config)
        : DynamicBlock(),
          max_length(config.image_size / config.patch_size * config.image_size / config.patch_size),
          embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config.width, config.patch_size, config.image_size),
          multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, max_length)
    {
        for (int layer_id = 0; layer_id < config.layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.width, config.heads, (int)(config.mlp_ratio * config.width), max_length);
            layer->set_id(layer_id);
            layers.emplace_back(layer);
        }
    }

    int64_t VisionTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embeddings.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void VisionTransformer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "pos_embed.weight")) return;

        embeddings.load(path, loader);
        multi_modal_projector.load(path, loader);
        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        _loaded = true;
    }

    ggml::tensor *VisionTransformer::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        input = ggml::scale(ctx, input, 1.0f);

        auto output = embeddings.forward(ctx, input);

        for (size_t i = 0; i < layers.size(); i++)
        {
            output = layers[i]->forward(ctx, output, 0);
        }
        output = multi_modal_projector.forward(ctx, output);
        return output;
    }

    VisualEmbeddingGeneration::VisualEmbeddingGeneration(const RuntimeConfig &runtime_config)
        : BaseMediaProjectedEmbeddingGeneration(runtime_config)
    {
    }

    bool VisualEmbeddingGeneration::load_model(Config vis_config)
    {
        CHATLLM_CHECK((vis_config.global_pool == CONFIG_GLOBAL_POOL_NONE) || (vis_config.global_pool == CONFIG_GLOBAL_POOL_MAP));

        config = vis_config;

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 9 + vis_config.layers * 17 +
            (vis_config.global_pool == CONFIG_GLOBAL_POOL_MAP ? 16 : 0);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = config.dtype;
        backend_context.init(model_gpu_layers, vis_config.layers, GRAPH_SIZE, n_threads);

        model.reset(new VisionTransformer(&_ctx, vis_config));

        _ctx.check_used_mem_size(true);
        return true;
    }

    bool VisualEmbeddingGeneration::load(const std::string &path, ModelLoader &loader)
    {
        if (model.get())
        {
            loader.push_allocator_manager(&backend_context.layer_allocators);
            model->load(path, &loader);
            loader.pop_allocator_manager();
            return model->is_loaded();
        }
        else
            return false;
    }

    ggml::tensor *VisualEmbeddingGeneration::make_media_tensor(ComputeContext *ctx, const BaseTokenizer::MediaAsEmbeddingVector &media)
    {
        CHATLLM_CHECK((media.width == config.image_size) && (media.width == config.image_size));
        return ggml::new_tensor_3d(ctx, ggml::type::GGML_TYPE_F32, media.width, media.height, 3);
    }
}