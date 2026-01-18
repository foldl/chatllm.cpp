#include "gemma.h"

namespace chatllm::gemma::v1
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}
    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *chat_encoder)
        : BaseTokenizer(config, chat_encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        int id = tp->PieceToId("<pad>");
        if (id >= 0) pad_token_id = id;
        start_of_turn_token_id = tp->PieceToId("<start_of_turn>");
        end_of_turn_token_id   = tp->PieceToId("<end_of_turn>");
        nl_token_id = tp->PieceToId("\n");

        const int first_added_id = tp->PieceToId("<unused0>");
        for (int i = tp->PieceToId("</code>"); i >= first_added_id; i--)
            tp->AddAddedToken(tp->IdToPiece(i), i);

        terminate_ids.insert(end_of_turn_token_id);
        terminate_ids.insert(tp->PieceToId("<|fim_prefix|>"));
        terminate_ids.insert(tp->PieceToId("<|fim_suffix|>"));
        terminate_ids.insert(tp->PieceToId("<|fim_middle|>"));
        terminate_ids.insert(tp->PieceToId("<|file_separator|>"));

        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_start, bool add_end)
    {
        if (add_start)
            ids.push_back(start_of_turn_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_end)
        {
            ids.push_back(end_of_turn_token_id);
            ids.push_back(nl_token_id);
        }
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 2 + config.num_hidden_layers * 12;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config,
                nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();

        transformer->word_embeddings->load("model.embed_tokens.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer->layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
        }
        transformer->final_layernorm->load("model.norm.", &loader);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
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
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "user" << "\n" << user;
        tok->encode(oss_prompt.str(), ids, true, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("model\n", ids, true, false);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("user\n", ids, true, false);
    }
}

namespace chatllm::gemma::v2
{
    ggml::tensor *TanhScaling::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        input = ggml::scale_inplace(ctx, input, scale_pre);
        input = ggml::inplace_act(ctx, ActFunc::Tanh, input);
        input = ggml::scale_inplace(ctx, input, scale_post);
        return input;
    }

    template <class Layer> static void setup_layer(Block *block, const Config &config, Block *attn_scores_pp)
    {
        auto layer = dynamic_cast<Layer *>(block);
        auto &attention = layer->attention;
        attention.freq_base = config.rope_theta;
        attention.attn_scaling_factor = powf((float)config.query_pre_attn_scalar, -0.5);
        attention.attn_scores_pp = attn_scores_pp;
    }

    template <class Layer> static void load_layer(ModelLoader &loader, const std::string &layer_prefix, Block *block)
    {
        auto layer = dynamic_cast<Layer *>(block);
        loader.read_tensor(layer_prefix + "input_layernorm.weight",             layer->pre_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "mlp.down_proj.weight",               layer->mlp.down_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",               layer->mlp.gate_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.up_proj.weight",                 layer->mlp.up_proj.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",    layer->post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "pre_feedforward_layernorm.weight",   layer->pre_mlp_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_feedforward_layernorm.weight",  layer->post_mlp_layernorm.weight);

        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", layer->attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", layer->attention.o_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", layer->attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", layer->attention.v_proj.weight);
    }


    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config),
        logits_pp(1.0f / config.final_logit_soft_capping / sqrtf((float)config.hidden_size), config.final_logit_soft_capping),
        attn_scores_pp(1.0f / config.attn_logit_soft_capping, config.attn_logit_soft_capping)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 2 + (config.num_hidden_layers - config.num_hidden_layers / 2) * 14 + config.num_hidden_layers / 2 * 15;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        CHATLLM_CHECK(SLIDING_WINDOW_LEN == config.sliding_window) << "unsupported SWA param";

        auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
        {
            if (is_sliding(layer_index))
            {
                return new Gemma2SWABlock4k(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
            }
            else
            {
                return new Gemma2FullBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
            }
        };

        transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                    create_embedding<Embedding>(&w_ctx_, config),
                    create_final_norm<RMSNorm>(&w_ctx_, config),
                    nullptr,
                    create_layer);

        get_typed_transformer<ModelClass>()->logits_pp = &logits_pp;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (is_sliding(i))
            {
                setup_layer<Gemma2SWABlock4k>(get_typed_transformer<ModelClass>()->get_layer(i), config, &attn_scores_pp);
            }
            else
            {
                setup_layer<Gemma2FullBlock>(get_typed_transformer<ModelClass>()->get_layer(i), config, &attn_scores_pp);
            }
        }

        batch_input = false;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();

        transformer->word_embeddings->load("model.embed_tokens.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
            if (is_sliding(i))
            {
                load_layer<Gemma2SWABlock4k>(loader, layer_prefix, transformer->get_layer(i));
            }
            else
            {
                load_layer<Gemma2FullBlock>(loader, layer_prefix, transformer->get_layer(i));
            }
        }
        transformer->final_layernorm->load("model.norm.", &loader);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace chatllm::gemma::siglip
{
    PatchEmbedding::PatchEmbedding(InitContext *ctx, int num_channels, int embed_dim, int image_size, int patch_size)
        : num_patches((image_size / patch_size) * (image_size / patch_size)),
        num_positions(num_patches),
        patch_embedding(ctx, num_channels, embed_dim, patch_size, patch_size),
        position_embedding(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, embed_dim, num_positions))
    {}

    ggml::tensor *PatchEmbedding::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto embedding = patch_embedding.forward(ctx, input);
        embedding = ggml::reshape_3d(ctx, embedding, ggml::get_dim(embedding, 0) * ggml::get_dim(embedding, 1), ggml::get_dim(embedding, 2), ggml::get_dim(embedding, 3));
        embedding = ggml::transpose(ctx, embedding);
        embedding = ggml::cont(ctx, embedding);
        embedding = ggml::add(ctx, embedding, position_embedding);
        return embedding;
    }

    int64_t PatchEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += patch_embedding.get_param_num(effective_only);
        r += ggml::nelements(position_embedding);
        return r;
    }

    void PatchEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        patch_embedding.load(path + "patch_embedding.", loader);
        loader->read_tensor(path + "position_embedding.weight", position_embedding);
    }

    MultiModalProjector::MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size)
        : mm_tokens_per_image(config.mm_tokens_per_image),
        patches_per_image(config.image_size / config.patch_size),
        tokens_per_side((int)std::sqrt(config.mm_tokens_per_image)),
        kernel_size(patches_per_image / tokens_per_side),
        input_projection(ctx, config.hidden_size, lm_hidden_size, false),
        soft_emb_norm(ctx, config.hidden_size)
    {
    }

    ggml::tensor *MultiModalProjector::forward(ComputeContext *ctx, ggml::tensor *vision_outputs)
    {
        CHATLLM_CHECK(ggml::get_dim(vision_outputs, 3) == 1) << "too many dimensions!";

        auto output = ggml::transpose(ctx, vision_outputs);
        output = ggml::cont(ctx, output);
        output = ggml::reshape_4d(ctx, output, patches_per_image, patches_per_image, ggml::get_dim(output, 1), ggml::get_dim(output, 2));
        output = ggml::avg_pool_2d(ctx, output, kernel_size, kernel_size);
        output = ggml::reshape_3d(ctx, output, mm_tokens_per_image, ggml::get_dim(output, 2), ggml::get_dim(output, 3));
        output = ggml::transpose(ctx, output);
        output = ggml::cont(ctx, output);
        output = soft_emb_norm.forward(ctx, output);
        output = input_projection.forward(ctx, output);
        return output;
    }

    int64_t MultiModalProjector::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += input_projection.get_param_num(effective_only);
        r += soft_emb_norm.get_param_num(effective_only);
        return r;
    }

    void MultiModalProjector::load(const std::string &path, TensorLoader *loader)
    {
        input_projection.load(path + "mm_input_projection.", loader);
        soft_emb_norm.load(path + "mm_soft_emb_norm.", loader);
    }


    VisionTransformer::VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size)
        : embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
                    config.num_channels, config.hidden_size, config.image_size, config.patch_size),
        post_layernorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
        multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, lm_hidden_size),
        loaded(false)
    {
        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, 1024 * 10);
            layer->set_id(layer_id);
            layers.emplace_back(layer);
        }
    }

    int64_t VisionTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embeddings.get_param_num(effective_only);
        r += post_layernorm.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void VisionTransformer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "embeddings.patch_embedding.weight")) return;

        embeddings.load(path + "embeddings.", loader);
        post_layernorm.load(path + "post_layernorm.", loader);
        multi_modal_projector.load("multi_modal_projector.", loader);
        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "encoder.layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        loaded = true;
    }

    ggml::tensor *VisionTransformer::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto output = embeddings.forward(ctx, input);
        for (size_t i = 0; i < layers.size(); i++)
        {
            output = layers[i]->forward(ctx, output, 0);
        }
        output = post_layernorm.forward(ctx, output);
        output = multi_modal_projector.forward(ctx, output);
        return output;
    }


    VisualEmbeddingGeneration::VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE)
        : GRAPH_SIZE(GRAPH_SIZE), _ctx(&backend_context), n_threads(runtime_config.n_threads)
    {
        _ctx.cache_dtype = runtime_config.cache_type;
        model_gpu_layers = BackendContext::get_ngl_of_model(runtime_config.model_gpu_layers, "vis");
    }

    bool VisualEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (vis_model.get())
        {
            loader.push_allocator_manager(&backend_context.layer_allocators);
            vis_model->load("vision_model.", &loader);
            loader.pop_allocator_manager();
            return vis_model->is_loaded();
        }
        else
            return false;
    }

    bool VisualEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
    {
        auto cfg = config["config.json"];
        if (!cfg.IsObject()) return false;
        auto pp_cfg = cfg["vision_config"];
        if (!pp_cfg.IsObject()) return false;

        vis_config.dtype                = dtype;
        vis_config.mm_tokens_per_image  = (int)cfg["mm_tokens_per_image"].ToInt();
        vis_config.hidden_size          = (int)pp_cfg["hidden_size"].ToInt();
        vis_config.image_size           = (int)pp_cfg["image_size"].ToInt();
        vis_config.intermediate_size    = (int)pp_cfg["intermediate_size"].ToInt();
        vis_config.num_attention_heads  = (int)pp_cfg["num_attention_heads"].ToInt();
        vis_config.num_channels         = (int)pp_cfg["num_channels"].ToInt(3);
        vis_config.num_hidden_layers    = (int)pp_cfg["num_hidden_layers"].ToInt();
        vis_config.patch_size           = (int)pp_cfg["patch_size"].ToInt();

        vis_config.vision_use_head      =      pp_cfg["vision_use_head"].ToBool();

        for (int i = 0; i < vis_config.num_channels; i++)
        {
            vis_config.image_mean[i]    = 0.5f;
            vis_config.image_std[i]     = 0.5f;
        }

        // ref: https://github.com/huggingface/transformers/blob/9487765f07ef4e5500d6ec21cad99aed4a037a3d/src/transformers/models/gemma3/processing_gemma3.py#L36
        vis_config.min_crop_size = 256;
        vis_config.max_num_crops =  4;
        vis_config.min_ratio_to_activate = 1.2f;

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 7 + vis_config.num_hidden_layers * 17;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;
        backend_context.init(model_gpu_layers, vis_config.num_hidden_layers, GRAPH_SIZE, n_threads);

        vis_model.reset(new VisionTransformer(&_ctx, vis_config, lm_hidden_size));

        CHATLLM_CHECK(_ctx.get_used_mem() == _ctx.get_mem_size()) << "tensor number mismatch: " << _ctx.get_used_mem() / tensor_ovhd << " vs "  << _ctx.get_mem_size() / tensor_ovhd;

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

    bool VisualEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
    {
        ForwardContext ctx(&backend_context);
        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *media_emb = ggml::new_tensor_3d(&ctx, ggml::type::GGML_TYPE_F32, vis_config.image_size, vis_config.image_size, 3);

        set_dbg_ctx(&ctx);

        auto r = vis_model->forward(&ctx, media_emb);

        if (ggml::type_of(r) != dtype)
        {
            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
            ggml::tensor *t = ggml::new_tensor_4d(&ctx, dtype, ggml::get_dim(r, 0), ggml::get_dim(r, 1), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
            r = ggml::cpy(&ctx, r, t);
        }

        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));

        ctx.compute();

        size_t offset = buf.size();
        buf.resize(offset + ggml::nbytes(r));
        Backend::read_tensor_data(r, buf.data() + offset);

        ctx.reset();

        return true;
    }
}

namespace chatllm::gemma::translation
{
    const static json::JSON language_code_dict({
    "aa", "Afar", "aa-DJ", "Afar", "aa-ER", "Afar", "ab", "Abkhazian", "af", "Afrikaans", "af-NA", "Afrikaans", "ak", "Akan", "am", "Amharic", "an", "Aragonese", "ar", "Arabic",
    "ar-AE", "Arabic", "ar-BH", "Arabic", "ar-DJ", "Arabic", "ar-DZ", "Arabic", "ar-EG", "Arabic", "ar-EH", "Arabic", "ar-ER", "Arabic", "ar-IL", "Arabic", "ar-IQ", "Arabic", "ar-JO", "Arabic",
    "ar-KM", "Arabic", "ar-KW", "Arabic", "ar-LB", "Arabic", "ar-LY", "Arabic", "ar-MA", "Arabic", "ar-MR", "Arabic", "ar-OM", "Arabic", "ar-PS", "Arabic", "ar-QA", "Arabic", "ar-SA", "Arabic",
    "ar-SD", "Arabic", "ar-SO", "Arabic", "ar-SS", "Arabic", "ar-SY", "Arabic", "ar-TD", "Arabic", "ar-TN", "Arabic", "ar-YE", "Arabic", "as", "Assamese", "az", "Azerbaijani", "az-Arab", "Azerbaijani",
    "az-Arab-IQ", "Azerbaijani", "az-Arab-TR", "Azerbaijani", "az-Cyrl", "Azerbaijani", "az-Latn", "Azerbaijani", "ba", "Bashkir", "be", "Belarusian", "be-tarask", "Belarusian", "bg", "Bulgarian", "bg-BG", "Bulgarian", "bm", "Bambara",
    "bm-Nkoo", "Bambara", "bn", "Bengali", "bn-IN", "Bengali", "bo", "Tibetan", "bo-IN", "Tibetan", "br", "Breton", "bs", "Bosnian", "bs-Cyrl", "Bosnian", "bs-Latn", "Bosnian", "ca", "Catalan",
    "ca-AD", "Catalan", "ca-ES", "Catalan", "ca-FR", "Catalan", "ca-IT", "Catalan", "ce", "Chechen", "co", "Corsican", "cs", "Czech", "cs-CZ", "Czech", "cv", "Chuvash", "cy", "Welsh",
    "da", "Danish", "da-DK", "Danish", "da-GL", "Danish", "de", "German", "de-AT", "German", "de-BE", "German", "de-CH", "German", "de-DE", "German", "de-IT", "German", "de-LI", "German",
    "de-LU", "German", "dv", "Divehi", "dz", "Dzongkha", "ee", "Ewe", "ee-TG", "Ewe", "el", "Greek", "el-CY", "Greek", "el-GR", "Greek", "el-polyton", "Greek", "en", "English",
    "en-AE", "English", "en-AG", "English", "en-AI", "English", "en-AS", "English", "en-AT", "English", "en-AU", "English", "en-BB", "English", "en-BE", "English", "en-BI", "English", "en-BM", "English",
    "en-BS", "English", "en-BW", "English", "en-BZ", "English", "en-CA", "English", "en-CC", "English", "en-CH", "English", "en-CK", "English", "en-CM", "English", "en-CX", "English", "en-CY", "English",
    "en-CZ", "English", "en-DE", "English", "en-DG", "English", "en-DK", "English", "en-DM", "English", "en-ER", "English", "en-ES", "English", "en-FI", "English", "en-FJ", "English", "en-FK", "English",
    "en-FM", "English", "en-FR", "English", "en-GB", "English", "en-GD", "English", "en-GG", "English", "en-GH", "English", "en-GI", "English", "en-GM", "English", "en-GS", "English", "en-GU", "English",
    "en-GY", "English", "en-HK", "English", "en-HU", "English", "en-ID", "English", "en-IE", "English", "en-IL", "English", "en-IM", "English", "en-IN", "English", "en-IO", "English", "en-IT", "English",
    "en-JE", "English", "en-JM", "English", "en-KE", "English", "en-KI", "English", "en-KN", "English", "en-KY", "English", "en-LC", "English", "en-LR", "English", "en-LS", "English", "en-MG", "English",
    "en-MH", "English", "en-MO", "English", "en-MP", "English", "en-MS", "English", "en-MT", "English", "en-MU", "English", "en-MV", "English", "en-MW", "English", "en-MY", "English", "en-NA", "English",
    "en-NF", "English", "en-NG", "English", "en-NL", "English", "en-NO", "English", "en-NR", "English", "en-NU", "English", "en-NZ", "English", "en-PG", "English", "en-PH", "English", "en-PK", "English",
    "en-PL", "English", "en-PN", "English", "en-PR", "English", "en-PT", "English", "en-PW", "English", "en-RO", "English", "en-RW", "English", "en-SB", "English", "en-SC", "English", "en-SD", "English",
    "en-SE", "English", "en-SG", "English", "en-SH", "English", "en-SI", "English", "en-SK", "English", "en-SL", "English", "en-SS", "English", "en-SX", "English", "en-SZ", "English", "en-TC", "English",
    "en-TK", "English", "en-TO", "English", "en-TT", "English", "en-TV", "English", "en-TZ", "English", "en-UG", "English", "en-UM", "English", "en-VC", "English", "en-VG", "English", "en-VI", "English",
    "en-VU", "English", "en-WS", "English", "en-ZA", "English", "en-ZM", "English", "en-ZW", "English", "eo", "Esperanto", "es", "Spanish", "es-AR", "Spanish", "es-BO", "Spanish", "es-BR", "Spanish",
    "es-BZ", "Spanish", "es-CL", "Spanish", "es-CO", "Spanish", "es-CR", "Spanish", "es-CU", "Spanish", "es-DO", "Spanish", "es-EA", "Spanish", "es-EC", "Spanish", "es-ES", "Spanish", "es-GQ", "Spanish",
    "es-GT", "Spanish", "es-HN", "Spanish", "es-IC", "Spanish", "es-MX", "Spanish", "es-NI", "Spanish", "es-PA", "Spanish", "es-PE", "Spanish", "es-PH", "Spanish", "es-PR", "Spanish", "es-PY", "Spanish",
    "es-SV", "Spanish", "es-US", "Spanish", "es-UY", "Spanish", "es-VE", "Spanish", "et", "Estonian", "et-EE", "Estonian", "eu", "Basque", "fa", "Persian", "fa-AF", "Persian", "fa-IR", "Persian",
    "ff", "Fulah", "ff-Adlm", "Fulah", "ff-Adlm-BF", "Fulah", "ff-Adlm-CM", "Fulah", "ff-Adlm-GH", "Fulah", "ff-Adlm-GM", "Fulah", "ff-Adlm-GW", "Fulah", "ff-Adlm-LR", "Fulah", "ff-Adlm-MR", "Fulah", "ff-Adlm-NE", "Fulah",
    "ff-Adlm-NG", "Fulah", "ff-Adlm-SL", "Fulah", "ff-Adlm-SN", "Fulah", "ff-Latn", "Fulah", "ff-Latn-BF", "Fulah", "ff-Latn-CM", "Fulah", "ff-Latn-GH", "Fulah", "ff-Latn-GM", "Fulah", "ff-Latn-GN", "Fulah", "ff-Latn-GW", "Fulah",
    "ff-Latn-LR", "Fulah", "ff-Latn-MR", "Fulah", "ff-Latn-NE", "Fulah", "ff-Latn-NG", "Fulah", "ff-Latn-SL", "Fulah", "fi", "Finnish", "fi-FI", "Finnish", "fil-PH", "Filipino", "fo", "Faroese", "fo-DK", "Faroese",
    "fr", "French", "fr-BE", "French", "fr-BF", "French", "fr-BI", "French", "fr-BJ", "French", "fr-BL", "French", "fr-CA", "French", "fr-CD", "French", "fr-CF", "French", "fr-CG", "French",
    "fr-CH", "French", "fr-CI", "French", "fr-CM", "French", "fr-DJ", "French", "fr-DZ", "French", "fr-FR", "French", "fr-GA", "French", "fr-GF", "French", "fr-GN", "French", "fr-GP", "French",
    "fr-GQ", "French", "fr-HT", "French", "fr-KM", "French", "fr-LU", "French", "fr-MA", "French", "fr-MC", "French", "fr-MF", "French", "fr-MG", "French", "fr-ML", "French", "fr-MQ", "French",
    "fr-MR", "French", "fr-MU", "French", "fr-NC", "French", "fr-NE", "French", "fr-PF", "French", "fr-PM", "French", "fr-RE", "French", "fr-RW", "French", "fr-SC", "French", "fr-SN", "French",
    "fr-SY", "French", "fr-TD", "French", "fr-TG", "French", "fr-TN", "French", "fr-VU", "French", "fr-WF", "French", "fr-YT", "French", "fy", "Western Frisian", "ga", "Irish", "ga-GB", "Irish",
    "gd", "Scottish Gaelic", "gl", "Galician", "gn", "Guarani", "gu", "Gujarati", "gu-IN", "Gujarati", "gv", "Manx", "ha", "Hausa", "ha-Arab", "Hausa", "ha-Arab-SD", "Hausa", "ha-GH", "Hausa",
    "ha-NE", "Hausa", "he", "Hebrew", "he-IL", "Hebrew", "hi", "Hindi", "hi-IN", "Hindi", "hi-Latn", "Hindi", "hr", "Croatian", "hr-BA", "Croatian", "hr-HR", "Croatian", "ht", "Haitian",
    "hu", "Hungarian", "hu-HU", "Hungarian", "hy", "Armenian", "ia", "Interlingua", "id", "Indonesian", "id-ID", "Indonesian", "ie", "Interlingue", "ig", "Igbo", "ii", "Sichuan Yi", "ik", "Inupiaq",
    "io", "Ido", "is", "Icelandic", "it", "Italian", "it-CH", "Italian", "it-IT", "Italian", "it-SM", "Italian", "it-VA", "Italian", "iu", "Inuktitut", "iu-Latn", "Inuktitut", "ja", "Japanese",
    "ja-JP", "Japanese", "jv", "Javanese", "ka", "Georgian", "ki", "Kikuyu", "kk", "Kazakh", "kk-Arab", "Kazakh", "kk-Cyrl", "Kazakh", "kk-KZ", "Kazakh", "kl", "Kalaallisut", "km", "Central Khmer",
    "kn", "Kannada", "kn-IN", "Kannada", "ko", "Korean", "ko-CN", "Korean", "ko-KP", "Korean", "ko-KR", "Korean", "ks", "Kashmiri", "ks-Arab", "Kashmiri", "ks-Deva", "Kashmiri", "ku", "Kurdish",
    "kw", "Cornish", "ky", "Kyrgyz", "la", "Latin", "lb", "Luxembourgish", "lg", "Ganda", "ln", "Lingala", "ln-AO", "Lingala", "ln-CF", "Lingala", "ln-CG", "Lingala", "lo", "Lao",
    "lt", "Lithuanian", "lt-LT", "Lithuanian", "lu", "Luba-Katanga", "lv", "Latvian", "lv-LV", "Latvian", "mg", "Malagasy", "mi", "Maori", "mk", "Macedonian", "ml", "Malayalam", "ml-IN", "Malayalam",
    "mn", "Mongolian", "mn-Mong", "Mongolian", "mn-Mong-MN", "Mongolian", "mr", "Marathi", "mr-IN", "Marathi", "ms", "Malay", "ms-Arab", "Malay", "ms-Arab-BN", "Malay", "ms-BN", "Malay", "ms-ID", "Malay",
    "ms-SG", "Malay", "mt", "Maltese", "my", "Burmese", "nb", "Norwegian Bokmål", "nb-SJ", "Norwegian Bokmål", "nd", "North Ndebele", "ne", "Nepali", "ne-IN", "Nepali", "nl", "Dutch", "nl-AW", "Dutch",
    "nl-BE", "Dutch", "nl-BQ", "Dutch", "nl-CW", "Dutch", "nl-NL", "Dutch", "nl-SR", "Dutch", "nl-SX", "Dutch", "nn", "Norwegian Nynorsk", "no", "Norwegian", "no-NO", "Norwegian", "nr", "South Ndebele",
    "nv", "Navajo", "ny", "Chichewa", "oc", "Occitan", "oc-ES", "Occitan", "om", "Oromo", "om-KE", "Oromo", "or", "Oriya", "os", "Ossetian", "os-RU", "Ossetian", "pa", "Punjabi",
    "pa-IN", "Punjabi", "pa-Arab", "Punjabi", "pa-Guru", "Punjabi", "pl", "Polish", "pl-PL", "Polish", "ps", "Pashto", "ps-PK", "Pashto", "pt", "Portuguese", "pt-AO", "Portuguese", "pt-BR", "Portuguese",
    "pt-CH", "Portuguese", "pt-CV", "Portuguese", "pt-GQ", "Portuguese", "pt-GW", "Portuguese", "pt-LU", "Portuguese", "pt-MO", "Portuguese", "pt-MZ", "Portuguese", "pt-PT", "Portuguese", "pt-ST", "Portuguese", "pt-TL", "Portuguese",
    "qu", "Quechua", "qu-BO", "Quechua", "qu-EC", "Quechua", "rm", "Romansh", "rn", "Rundi", "ro", "Romanian", "ro-MD", "Romanian", "ro-RO", "Romanian", "ru", "Russian", "ru-BY", "Russian",
    "ru-KG", "Russian", "ru-KZ", "Russian", "ru-MD", "Russian", "ru-RU", "Russian", "ru-UA", "Russian", "rw", "Kinyarwanda", "sa", "Sanskrit", "sc", "Sardinian", "sd", "Sindhi", "sd-Arab", "Sindhi",
    "sd-Deva", "Sindhi", "se", "Northern Sami", "se-FI", "Northern Sami", "se-SE", "Northern Sami", "sg", "Sango", "si", "Sinhala", "sk", "Slovak", "sk-SK", "Slovak", "sl", "Slovenian", "sl-SI", "Slovenian",
    "sn", "Shona", "so", "Somali", "so-DJ", "Somali", "so-ET", "Somali", "so-KE", "Somali", "sq", "Albanian", "sq-MK", "Albanian", "sq-XK", "Albanian", "sr", "Serbian", "sr-RS", "Serbian",
    "sr-Cyrl", "Serbian", "sr-Cyrl-BA", "Serbian", "sr-Cyrl-ME", "Serbian", "sr-Cyrl-XK", "Serbian", "sr-Latn", "Serbian", "sr-Latn-BA", "Serbian", "sr-Latn-ME", "Serbian", "sr-Latn-XK", "Serbian", "ss", "Swati", "ss-SZ", "Swati",
    "st", "Southern Sotho", "st-LS", "Southern Sotho", "su", "Sundanese", "su-Latn", "Sundanese", "sv", "Swedish", "sv-AX", "Swedish", "sv-FI", "Swedish", "sv-SE", "Swedish", "sw", "Swahili", "sw-CD", "Swahili",
    "sw-KE", "Swahili", "sw-TZ", "Swahili", "sw-UG", "Swahili", "ta", "Tamil", "ta-IN", "Tamil", "ta-LK", "Tamil", "ta-MY", "Tamil", "ta-SG", "Tamil", "te", "Telugu", "te-IN", "Telugu",
    "tg", "Tajik", "th", "Thai", "th-TH", "Thai", "ti", "Tigrinya", "ti-ER", "Tigrinya", "tk", "Turkmen", "tl", "Tagalog", "tn", "Tswana", "tn-BW", "Tswana", "to", "Tonga",
    "tr", "Turkish", "tr-CY", "Turkish", "tr-TR", "Turkish", "ts", "Tsonga", "tt", "Tatar", "ug", "Uyghur", "uk", "Ukrainian", "uk-UA", "Ukrainian", "ur", "Urdu", "ur-IN", "Urdu",
    "ur-PK", "Urdu", "uz", "Uzbek", "uz-Arab", "Uzbek", "uz-Cyrl", "Uzbek", "uz-Latn", "Uzbek", "ve", "Venda", "vi", "Vietnamese", "vi-VN", "Vietnamese", "vo", "Volapük", "wa", "Walloon",
    "wo", "Wolof", "xh", "Xhosa", "yi", "Yiddish", "yo", "Yoruba", "yo-BJ", "Yoruba", "za", "Zhuang", "zh", "Chinese", "zh-CH", "Chinese", "zh-TW", "Chinese", "zh-Hans", "Chinese",
    "zh-Hans-HK", "Chinese", "zh-Hans-MO", "Chinese", "zh-Hans-MY", "Chinese", "zh-Hans-SG", "Chinese", "zh-Hant", "Chinese", "zh-Hant-HK", "Chinese", "zh-Hant-MO", "Chinese", "zh-Hant-MY", "Chinese", "zh-Latn", "Chinese", "zu", "Zulu",
    "zu-ZA", "Zulu"});

    class TranslateGemmaHistoryEncoder : public v3::ChatHistoryEncoder
    {
    public:
        typedef v3::ChatHistoryEncoder Base;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
    public:
        void parse_command(std::string &user) const;
    public:
        std::string def_source_lang_code = "zh";
        std::string def_target_lang_code = "en";
        std::string source_lang_code;
        std::string target_lang_code;
    };

    static TranslateGemmaHistoryEncoder _translate_encoder;

    void TranslateGemmaHistoryEncoder::parse_command(std::string &user) const
    {
        user = utils::trim(user);
        if (user.size() < 1) return;
        if (user[0] != '/') return;
        auto pos = user.find(' ');
        std::string command = user.substr(1, pos - 1);
        if (pos != std::string::npos)
            user = utils::trim(user.substr(pos + 1));
        else
            user = "";
        std::vector<std::string> items;
        utils::split(command, "->", items);
        if (items.size() !=2) return;

        auto obj = const_cast<TranslateGemmaHistoryEncoder *>(this);

        if (language_code_dict.hasKey(items[0]))
            obj->source_lang_code = items[0];
        if (language_code_dict.hasKey(items[1]))
            obj->target_lang_code = items[1];
    }

    void TranslateGemmaHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Content c(nullptr, user);
        append_user(round_idx, c, ids);
    }

    void TranslateGemmaHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        std::string text = "";
        std::string image = "";

        {
            auto obj = const_cast<TranslateGemmaHistoryEncoder *>(this);
            obj->source_lang_code = def_source_lang_code;
            obj->target_lang_code = def_target_lang_code;
        }

        for (auto &piece : user.pieces)
        {
            switch (piece.type)
            {
            case ContentPiece::Type::Text:
                {
                    std::string s = piece.content;
                    parse_command(s);
                    if (s.size() < 1) break;
                    CHATLLM_CHECK(text.size() < 1) << "only one text input is allowed";
                    text = s;
                }
                break;
            case ContentPiece::Type::Image:
                CHATLLM_CHECK(image.size() < 1) << "only one image input is allowed";
                image = piece.content;
                break;
            default:
                CHATLLM_CHECK(false) << "only text/image input are allowed";
                break;
            }
        }

        const std::string source_lang = language_code_dict[source_lang_code].ToString();
        const std::string target_lang = language_code_dict[target_lang_code].ToString();

        const std::string s1 = "You are a professional " + source_lang + " (" + source_lang_code + ") to " +
           target_lang + " (" + target_lang_code + ") translator. Your goal is to accurately convey the meaning and "
           "nuances of the original " + source_lang + " text while adhering to " + target_lang + " grammar, "
           "vocabulary, and cultural sensitivities.\n";

        const std::string s2 = text.size() > 0 ?
            "Produce only the " + target_lang + " translation, without any additional explanations or " +
            "commentary. Please translate the following " + source_lang + " text into " + target_lang + ":\n\n\n" + text
            :
            "Please translate the " + source_lang + " text in the provided image into " + target_lang + ". " +
            "Produce only the " + target_lang + " translation, without any additional explanations, " +
            "alternatives or commentary. Focus only on the text, do not output where the text is located, " +
            "surrounding objects or any other explanation about the picture. Ignore symbols, pictogram, and " +
            "arrows!\n\n\n";

        Content c(nullptr, s1 + s2);
        if (image.size() > 0)
            c.push_back(image, ContentPiece::Type::Image);
        Base::append_user(round_idx, c, ids);
    }
}

namespace chatllm::gemma::v3
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : v1::Tokenizer(config, &_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t size = v1::Tokenizer::load(buffer, n_vocab);

        boi_token_id = tp->PieceToId("<start_of_image>");
        eoi_token_id = tp->PieceToId("<end_of_image>");
        return size;
    }

    template <class Layer> static void setup_layer(Block *block, const Config &config, float rope_theta, float rope_scaling_factor = -1.0f)
    {
        auto layer = dynamic_cast<Layer *>(block);
        auto &attention = layer->attention;
        attention.freq_base = rope_theta;
        attention.attn_scaling_factor = powf((float)config.query_pre_attn_scalar, -0.5);
        if (rope_scaling_factor > 0)
            attention.freq_scale = 1 / rope_scaling_factor;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config),
        sliding_window(config.sliding_window),
        sliding_window_pattern(config.sliding_window_pattern),
        visual(runtime_config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        size_t num_tensors = 2;
        for (int i = 0; i < config.num_hidden_layers; i++) num_tensors += is_sliding(i) ? 17 : 16;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;
        const int num_sliding_layer = number_of_sliding_window_layer();

        if (num_sliding_layer > 0)
            CHATLLM_CHECK((512 == sliding_window) || (1024 == sliding_window))<< "unsupported SWA param";

        auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
        {
            if (is_sliding(layer_index))
            {
                switch (sliding_window)
                {
                case 512:
                    return new Gemma3SWABlock512(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.num_key_value_heads, config.head_dim, config.max_length);
                default:
                    return new Gemma3SWABlock1024(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.num_key_value_heads, config.head_dim, config.max_length);
                }
            }
            else
            {
                return new Gemma3FullBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
            }
        };

        BlockParams::PadEmbedding padding(1024, 1024); // 4 media_emb
        _chat_encoder.MAX_PATCH_NUM = padding.get();

        transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                    create_embedding<Embedding>(&w_ctx_, config),
                    create_final_norm<RMSNorm>(&w_ctx_, config),
                    nullptr, create_layer);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (is_sliding(i))
            {
                switch (sliding_window)
                {
                case 512:
                    setup_layer<Gemma3SWABlock512>(get_typed_transformer<ModelClass>()->get_layer(i), config, config.rope_local_base_freq);
                    break;
                default:
                    setup_layer<Gemma3SWABlock1024>(get_typed_transformer<ModelClass>()->get_layer(i), config, config.rope_local_base_freq);
                    break;
                }

            }
            else
            {
                setup_layer<Gemma3FullBlock>(get_typed_transformer<ModelClass>()->get_layer(i), config, config.rope_theta, config.rope_scaling_factor);
            }
        }

        if (num_sliding_layer > 0)
            batch_input = false;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".pre_attention_layernorm.",   ".input_layernorm."},
            {".pre_mlp_layernorm.",         ".pre_feedforward_layernorm."},
            {".post_mlp_layernorm.",        ".post_feedforward_layernorm."},
        });

        BaseModelForConditionalGeneration::load(loader);
        _chat_encoder.vit_loaded = visual.load(loader);
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer *tokenizer)
    {
        BaseModelForConditionalGeneration::set_tokenizer(tokenizer);
        if ("TranslateGemma" == name_)
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            tok->set_chat_encoder(&translation::_translate_encoder);
            multi_turn = false;
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

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return;
        tok->do_pan_and_scan = utils::get_opt(args, "do-pan-and-scan", false);
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return;
        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        visual.generate(gen_config, tok, ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    bool ConditionalGeneration::is_sliding(int layer_id) const
    {
        return ((layer_id + 1) % sliding_window_pattern) != 0;
    }

    int ConditionalGeneration::number_of_sliding_window_layer(void) const
    {
        int r = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
            r += is_sliding(i) ? 1 : 0;
        return r;
    }

    bool ChatHistoryEncoder::append_image(const vision::image_pixels_t pixels, const int w, const int h, std::vector<int> &ids) const
    {
        const int patch_size = vis_config->patch_size;
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (nullptr == tok) return false;

        std::vector<float> scaled;

        vision::image_rescale(pixels, scaled);

        vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

        tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

        auto &image = tok->media_emb.back();

        vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);

        image.emb_vec_number = vis_config->mm_tokens_per_image;

        const int total_patches = tok->get_image_total_emb_vectors();
        CHATLLM_CHECK(total_patches <= MAX_PATCH_NUM) << "too many image patches!";

        ids.push_back(tok->nl_token_id);
        ids.push_back(tok->nl_token_id);
        ids.push_back(tok->boi_token_id);
        int id = total_patches - image.emb_vec_number + tok->vocab_size;
        for (int j = 0; j < image.emb_vec_number; j++)
        {
            ids.push_back(id++);
        }
        ids.push_back(tok->eoi_token_id);
        ids.push_back(tok->nl_token_id);
        ids.push_back(tok->nl_token_id);

        return true;
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode("user\n", ids, true, false);

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids, false, false);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                if (tok->do_pan_and_scan)
                {
                    std::vector<vision::image_pixels_t> crops;

                    vision::PanScanDir dir = vision::PanScanDir::Horizontal;

                    vision::image_load_pan_and_scan(piece.content.c_str(),
                        crops, tok->do_pan_and_scan,
                        vis_config->min_crop_size, vis_config->max_num_crops, vis_config->min_ratio_to_activate,
                        vis_config->image_size, vis_config->image_size,
                        dir);

                    printf("crops: %d\n", (int)crops.size());
                    if (crops.size() < 1) continue;

                    if (crops.size() == 1)
                    {
                        append_image(crops[0], vis_config->image_size, vis_config->image_size, ids);
                        continue;
                    }

                    tok->encode("Here is the original image ", ids, false, false);
                    append_image(crops[0], vis_config->image_size, vis_config->image_size, ids);
                    tok->encode(" and here are some crops to help you see better", ids, false, false);

                    for (size_t i = 1; i < crops.size(); i++)
                    {
                        tok->encode(" ", ids, false, false);
                        append_image(crops[i], vis_config->image_size, vis_config->image_size, ids);
                    }
                }
                else
                {
                    vision::Resize resize(vis_config->image_size, vis_config->image_size);

                    int w, h;
                    std::vector<uint8_t> pixels;
                    const int patch_size = vis_config->patch_size;
                    vision::image_load(piece.content.c_str(), pixels, w, h, patch_size);

                    if (w <= 0) continue;

                    append_image(pixels, w, h, ids);
                }
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        tok->encode("", ids, false, true);
    }
}


namespace chatllm
{
    REGISTER_MODEL_LOADER(GEMMA,                 gemma::v1, 1);
    REGISTER_MODEL_LOADER(GEMMA2,                gemma::v2, 2);
    REGISTER_MODEL_LOADER(GEMMA3,                gemma::v3, 1);
    REGISTER_MODEL_LOADER(GEMMA3Vis,             gemma::v3, 1);
}