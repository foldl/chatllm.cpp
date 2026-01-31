#include "qwen_asr.h"

namespace chatllm::qwen::v3::audio_tower
{
    AudioSelfAttention::AudioSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
        BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, true, true)
    {
        causal = false;
    }

    MultiModalProjector::MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size):
        TheMLP(ctx, config.hidden_size, config.hidden_size, lm_hidden_size, chatllm::ActFunc::GELU, true)
    {
    }

    AudioEmbedding::AudioEmbedding(InitContext *ctx, const Config &config):
        n_window(config.n_window),
        conv_chunksize(config.conv_chunksize),
        conv2d1(ctx, 1, config.downsample_hidden_size, 3, 2, 1),
        conv2d2(ctx, config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, 1),
        conv2d3(ctx, config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, 1),
        conv_out(ctx,
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2),
            config.hidden_size, false),
        positional_embedding(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size, config.max_source_positions))
    {
        ctx->get_allocator()->alloc(positional_embedding);
        std::vector<float> emb;
        emb.resize(ggml::nelements(positional_embedding));

        const int channels = config.hidden_size;
        const double max_timescale = 10000.0f;
        const double log_timescale_increment = log(max_timescale) / (channels / 2 - 1);

        for (int i = 0; i < config.max_source_positions; i++)
        {
            for (int j = 0; j < config.hidden_size / 2; j++)
            {
                const int pos = config.hidden_size * i + j;
                const double inv_timescales = exp(-log_timescale_increment * j);
                const double scaled_time = i * inv_timescales;

                emb[pos + 0                     ] = (float)sin(scaled_time);
                emb[pos + config.hidden_size / 2] = (float)cos(scaled_time);

            }
        }
        Backend::write_tensor_data(positional_embedding, emb.data());
    }

    int64_t AudioEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += conv2d1.get_param_num(effective_only);
        r += conv2d2.get_param_num(effective_only);
        r += conv2d3.get_param_num(effective_only);
        r += conv_out.get_param_num(effective_only);
        return r;
    }

    void AudioEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        conv2d1.load(path + "conv2d1.", loader);
        conv2d2.load(path + "conv2d2.", loader);
        conv2d3.load(path + "conv2d3.", loader);
        conv_out.load(path + "conv_out.", loader);
    }

    ggml::tensor *AudioEmbedding::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        const int pad_window = n_window * 2;
        int64_t len = ggml::get_dim(input, 0);
        const int emb_num = get_feat_extract_output_lengths((int)len);
        len = (len + pad_window - 1) / pad_window * pad_window;
        input = ggml::pad(ctx, input, 0, (int)(len - ggml::get_dim(input, 0)));

        const int64_t hidden_size = ggml::get_dim(input, 1);
        input = ggml::reshape_3d(ctx, input, pad_window, ggml::get_dim(input, 0) / pad_window, hidden_size);
        input = ggml::permute(ctx, input, 0, 2, 1);
        input = ggml::cont(ctx, input);

        len = ggml::get_dim(input, 2);
        int offset = 0;
        ggml::tensor *hidden_states = nullptr;
        while (len > 0)
        {
            int block_len = len > conv_chunksize ? conv_chunksize : (int)len;
            if (len < conv_chunksize)
                len = len;
            auto block = ggml::view_3d(ctx, input, pad_window, hidden_size, block_len,
                ggml::row_size(input), ggml::row_size(input) * ggml::get_dim(input, 1),
                ggml::row_size(input) * ggml::get_dim(input, 1) * offset);
            block = ggml::reshape_4d(ctx, block, ggml::get_dim(block, 0), ggml::get_dim(block, 1), 1, ggml::get_dim(block, 2));
            block = conv2d1.forward(ctx, block);
            block = ggml::act(ctx, ActFunc::GELU, block);
            block = conv2d2.forward(ctx, block);
            block = ggml::act(ctx, ActFunc::GELU, block);
            block = conv2d3.forward(ctx, block);
            block = ggml::act(ctx, ActFunc::GELU, block);

            hidden_states = hidden_states ? ggml::concat(ctx, hidden_states, block, 0) : block;

            offset += block_len;
            len    -= block_len;
        }

        hidden_states = ggml::permute(ctx, hidden_states, 2, 0, 1);
        hidden_states = ggml::cont(ctx, hidden_states);
        hidden_states = ggml::reshape_3d(ctx, hidden_states, ggml::get_dim(hidden_states, 0) * ggml::get_dim(hidden_states, 1), ggml::get_dim(hidden_states, 2), ggml::get_dim(hidden_states, 3));
        hidden_states = conv_out.forward(ctx, hidden_states);

        // add pos emb
        len = ggml::get_dim(hidden_states, 1);
        auto pos_emb = ggml::view_2d(ctx, positional_embedding, ggml::get_dim(hidden_states, 0), len, ggml::row_size(positional_embedding), 0);
        hidden_states = ggml::add(ctx, hidden_states, pos_emb);

        hidden_states = ggml::reshape_2d(ctx, hidden_states, ggml::get_dim(hidden_states, 0), ggml::get_dim(hidden_states, 1) * ggml::get_dim(hidden_states, 2));
        hidden_states = ggml::view_2d(ctx, hidden_states, ggml::get_dim(hidden_states, 0), emb_num, ggml::row_size(hidden_states), 0);

        return hidden_states;
    }

    AudioTransformer::AudioTransformer(InitContext *ctx, const Config &config, int lm_hidden_size) :
        intermedia_size(config.intermediate_size),
        embed(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
        norm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
        multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog),
            config, lm_hidden_size),
        loaded(false)
    {
        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, config.num_key_value_heads,
                config.hidden_size / config.num_attention_heads, config.max_source_positions);
            layer->set_id(layer_id);
            layers.emplace_back(layer);
        }
    }

    int64_t AudioTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += norm.get_param_num(effective_only);
        r += embed.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void AudioTransformer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "conv2d1.bias")) return;

        norm.load(path + "ln_post.", loader);
        multi_modal_projector.load("multi_modal_projector.", loader);
        embed.load(path, loader);

        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        loaded = true;
    }

    ggml::tensor *AudioTransformer::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto output = embed.forward(ctx, input);

        for (size_t i = 0; i < layers.size(); i++)
        {
            output = layers[i]->forward(ctx, output, 0);
        }
        output = norm.forward(ctx, output);
        output = multi_modal_projector.forward(ctx, output);

        return output;
    }

    bool AudioTransformer::is_loaded(void) const
    {
        return loaded;
    }

    AudioEmbeddingGeneration::AudioEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "aud", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool AudioEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (model.get())
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            model->load("audio.", &loader);
            loader.pop_allocator_manager();
            return model->is_loaded();
        }
        else
            return false;
    }

    bool AudioEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &json_config)
    {
        const auto _cfg = json_config["config.json"]["thinker_config"]["audio_config"];
        if (!_cfg.IsObject()) return false;

        config.dtype = dtype;

        config.num_hidden_layers        = (int)_cfg["encoder_layers"].ToInt();
        config.num_attention_heads      = (int)_cfg["encoder_attention_heads"].ToInt();
        config.num_key_value_heads      = (int)_cfg["encoder_attention_heads"].ToInt();
        config.intermediate_size        = (int)_cfg["encoder_ffn_dim"].ToInt();
        config.hidden_size              = (int)_cfg["d_model"].ToInt();
        config.max_source_positions     = (int)_cfg["max_source_positions"].ToInt();
        config.num_mel_bins             = (int)_cfg["num_mel_bins"].ToInt();

        config.n_window                 = (int)_cfg["n_window"].ToInt();
        config.n_window_infer           = (int)_cfg["n_window_infer"].ToInt();
        config.downsample_hidden_size   = (int)_cfg["downsample_hidden_size"].ToInt();
        config.conv_chunksize           = (int)_cfg["conv_chunksize"].ToInt();

        auto pp_cfg = json_config["preprocessor_config.json"];
        if (!pp_cfg.IsObject()) return false;

        config.chunk_length     = (int  )pp_cfg["chunk_length"].ToInt();
        config.feature_size     = (int  )pp_cfg["feature_size"].ToInt();
        config.hop_length       = (int  )pp_cfg["hop_length"].ToInt();
        config.n_fft            = (int  )pp_cfg["n_fft"].ToInt();
        config.n_samples        = (int  )pp_cfg["n_samples"].ToInt();
        config.nb_max_frames    = (int  )pp_cfg["nb_max_frames"].ToInt();
        config.sampling_rate    = 16000;

        const size_t tensor_ovhd = ggml::tensor_overhead();
        const size_t num_tensors = 14 + config.num_hidden_layers * 17;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        model.reset(new AudioTransformer(&_ctx, config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

        return true;
    }

    void AudioEmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if ((model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!model->is_loaded()) return;

        for (auto &media : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, media, buf);
        }
    }

    bool AudioEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype,
        const BaseTokenizer::MediaAsEmbeddingVector &audio, std::vector<uint8_t> &buf)
    {
        ggml::tensor *media_emb = nullptr;
        model->emb_vector_len = audio.emb_vec_number;
        const int64_t n_len = audio.data.size() / config.feature_size;
        const auto make_graph = [this, &media_emb, &audio, n_len](ComputeContext *ctx) -> ggml::tensor * {
            media_emb = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, n_len, config.feature_size);;
            auto r = model->forward(ctx, media_emb);
            return r;
        };
        const auto write_input_data = [&media_emb, &audio](ComputeContext *ctx) {
            Backend::write_tensor_data(media_emb, audio.data.data(), 0, audio.data.size() * sizeof(audio.data[0]));
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }

    int get_feat_extract_output_lengths(int input_length)
    {
        auto input_lengths_leave = input_length % 100;
        auto output_lengths = (input_lengths_leave + 7) / 8 + (input_length / 100) * 13;
        return output_lengths;
    }
    std::vector<int> get_feat_extract_output_lengths(const std::vector<int> &input_lengths)
    {
        std::vector<int> r;
        for (auto x : input_lengths)
            r.push_back(get_feat_extract_output_lengths(x));
        return std::move(r);
    }

    int pad_mel_len(int len)
    {
        auto input_lengths_leave = len % 100;
        return (len / 100) * 100 + (input_lengths_leave + 7) / 8 * 8;
    }

}
