namespace snac
{
    const int MAX_VQ_STRIDES = 4;
    const int MAX_RATES = 4;

    struct Config
    {
        int sampling_rate;
        int encoder_dim;
        int encoder_rates[MAX_RATES];
        int decoder_dim;
        int decoder_rates[MAX_RATES];
        int attn_window_size;
        int codebook_size;
        int codebook_dim;
        int vq_strides[MAX_VQ_STRIDES];
        int vq_stride_count;
        bool noise;
        bool depthwise;
    };

    class Noise : public Block
    {
    public:
        Noise(InitContext *ctx, int dim)
            : linear(ctx, dim, dim, 1, 1, 0, 1, 1, false)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            ggml::tensor *output = linear.forward(ctx, input);
            ggml::tensor *noise = ggml::new_tensor_3d(ctx, ggml::type_of(output), input->ne[0], 1, input->ne[2]);
            noise = ggml::randn_inplace(ctx, noise);
            output = ggml::mul(ctx, output, noise);
            output = ggml::add(ctx, input, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return linear.get_param_num(effective_only);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            linear.load(path + "linear.", loader);
        }
    protected:
        Conv1D linear;
    };

    class Snake1D: public Block
    {
    public:
        Snake1D(InitContext *ctx, int dim)
            : alpha(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, dim)),
              alpha_reciprocal(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, 1, dim))
        {
            ctx->get_allocator()->alloc(alpha_reciprocal);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            auto alpha_view = ggml::reshape_2d(ctx, alpha, 1, ggml::get_dim(alpha, 0));
            ggml::tensor *output = ggml::mul(ctx, input, alpha_view);
            output = ggml::sin(ctx, output);
            output = ggml::square(ctx, output);
            output = ggml::mul(ctx, output, alpha_reciprocal);
            output = ggml::add(ctx, input, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml::nelements(alpha);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            loader->read_tensor(path + "alpha", alpha);

            std::vector<float> alpha_reciprocal_data(alpha->ne[0]);
            Backend::read_tensor_data(alpha, alpha_reciprocal_data.data(), 0, alpha->ne[0] * sizeof(float));
            for (int i = 0; i < alpha->ne[0]; i++)
            {
                alpha_reciprocal_data[i] = 1.0f / (alpha_reciprocal_data[i] + 1e-9f);
            }
            Backend::write_tensor_data(alpha_reciprocal, alpha_reciprocal_data.data(), 0, alpha->ne[0] * sizeof(float));
        }

    public:
        ggml::tensor *alpha;
        ggml::tensor *alpha_reciprocal;
    };

    class ResidualUnit : public Block
    {
    public:
        ResidualUnit(InitContext *ctx, int dim, int dilation = 1, int groups = 1, int kernel_size = 7)
            : block(ctx)
        {
            const int padding = ((kernel_size - 1) * dilation) / 2;
            block.add_block(new Snake1D(ctx, dim));
            block.add_block(new Conv1D(ctx, dim, dim, kernel_size, 1, padding, dilation, groups));
            block.add_block(new Snake1D(ctx, dim));
            block.add_block(new Conv1D(ctx, dim, dim, 1));
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x) override
        {
            ggml::tensor *y = block.forward(ctx, x);

            const int64_t pad = (x->ne[0] - y->ne[0]) / 2;
            if (pad > 0)
            {
                x = ggml::view_3d(ctx, x, y->ne[0], x->ne[1], x->ne[2],
                    x->ne[0] * ggml::element_size(x), x->ne[0] * x->ne[1] * ggml::element_size(x),
                    pad * ggml::element_size(x));
            }
            y = ggml::add(ctx, x, y);
            return y;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return block.get_param_num(effective_only);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            block.load(path + "block.layers.", loader);
        }
    public:
        Sequential block;
    };

    class DecoderBlock : public Sequential
    {
    public:
        DecoderBlock(InitContext *ctx, int input_dim = 16, int output_dim = 8, int stride = 1, bool noise = false, int groups = 1)
            : Sequential(ctx)
        {
            add_block(new Snake1D(ctx, input_dim));
            // ConvTransposed1D(InitContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, int padding,
            //                  int output_padding, int dilation, int groups, bool bias)
            add_block(new ConvTransposed1D(ctx, input_dim, output_dim, 2 * stride,
                                           stride,
                                           (stride + 1) / 2,
                                           stride % 2));
            if (noise)
            {
                add_block(new Noise(ctx, output_dim));
            }

            add_block(new ResidualUnit(ctx, output_dim, 1, groups));
            add_block(new ResidualUnit(ctx, output_dim, 3, groups));
            add_block(new ResidualUnit(ctx, output_dim, 9, groups));
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Sequential::load(path + "block.layers.", loader);
        }
    };

    class Decoder : public Sequential
    {
    public:
        Decoder(InitContext *ctx, int input_channel, int channels,
                int rates_count, const int *rates,
                bool noise = false, bool depthwise = false,
                int attn_window_size = 32, int d_out =1)
            : Sequential(ctx)
        {
            if (depthwise)
            {
                // int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
                // int dilation = 1, int groups = 1, bool bias = true
                add_block(new Conv1D(ctx, input_channel, input_channel, 7, 1, 3, 1, input_channel));
                add_block(new Conv1D(ctx, input_channel, channels, 1));
            }
            else
            {
                add_block(new Conv1D(ctx, input_channel, channels, 7, 1, 3));
            }

            CHATLLM_CHECK(attn_window_size <= 0) << "attn_window_size not supported";

            int output_dim  = 0;
            for (int i = 0; i < rates_count; i++)
            {
                const int input_dim = channels / (1 << i);
                output_dim = channels / (1 << (i + 1));
                const int groups = depthwise ? output_dim : 1;
                add_block(new DecoderBlock(ctx, input_dim, output_dim, rates[i], noise, groups));
            }

            add_block(new Snake1D(ctx, output_dim));
            add_block(new Conv1D(ctx, output_dim, d_out, 7, 1, 3));
            add_block(new Unary(ctx, Unary::Op::Tanh));
        }
    };

    class VectorQuantize : public Block
    {
    public:
        VectorQuantize(InitContext *ctx,
                       int input_dim, int codebook_size, int codebook_dim, int stride = 1)
            : codebook_size(codebook_size), codebook_dim(codebook_dim), stride(stride),
              in_proj(ctx, input_dim, codebook_dim, 1),
              out_proj(ctx, codebook_dim, input_dim, 1),
              codebook(ctx, codebook_size, codebook_dim)
        {
        }

        ggml::tensor *dequantize(ComputeContext *ctx, ggml::tensor *embed_id)
        {
            ggml::tensor *output = codebook.forward(ctx, embed_id);
            output = ggml::permute(ctx, output, 1, 0, 2, 3); // [emb_dim, len] -> [len, emb_dim, 1, 1]
            output = ggml::cont(ctx, output);
            output = out_proj.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r +=  in_proj.get_param_num(effective_only);
            r += out_proj.get_param_num(effective_only);
            r += codebook.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
             in_proj.load(path + "in_proj.", loader);
            out_proj.load(path + "out_proj.", loader);
            codebook.load(path + "codebook.", loader);
        }
    public:
        const int codebook_size;
        const int codebook_dim;
        const int stride;
        Conv1D in_proj;
        Conv1D out_proj;
        Embedding codebook;
    };

    class ResidualVectorQuantize : public Sequential
    {
    public:
        ResidualVectorQuantize(InitContext *ctx,
            int input_dim,
            int codebook_size,
            int codebook_dim,
            int vq_strides_count,
            const int *vq_strides)
            : Sequential(ctx),
              n_codebooks(vq_strides_count)
        {
            for (int i = 0; i < n_codebooks; i++)
            {
                add_block(new VectorQuantize(ctx, input_dim, codebook_size, codebook_dim, vq_strides[i]));
            }
        }

        ggml::tensor *dequantize(ComputeContext *ctx, const std::vector<ggml::tensor *> &embed_id)
        {
            CHATLLM_CHECK(embed_id.size() == n_codebooks) << "embed_id.size() != n_codebooks";

            ggml::tensor *output = nullptr;
            for (int i = 0; i < n_codebooks; i++)
            {
                VectorQuantize *q = (VectorQuantize *)blocks[i].get();
                auto z_q_i = q->dequantize(ctx, embed_id[i]);

                if (output)
                {
                    output = ggml::repeat_interleave(ctx, output, ggml::get_dim(z_q_i, 0) / ggml::get_dim(output, 0));

                    output = ggml::add(ctx, z_q_i, output); // CHECK: auto repeated?
                }
                else
                    output = z_q_i;
            }
            return output;
        }

    public:
        const int n_codebooks;
    };

    struct node
    {
        int level;
        node *left;
        node *right;
        node(int level, node *left, node *right): level(level), left(left), right(right) {}
        ~node(void) { delete left; delete right; }
    };

    static node *make_pyramid(int total_level, int level = 0)
    {
        if (level >= total_level) return nullptr;

        return new node(level, make_pyramid(total_level, level + 1), make_pyramid(total_level, level + 1));
    }

    static void pyramid_transverse(node *root, std::vector<int> &level_ids)
    {
        if (nullptr == root) return;

        level_ids.push_back(root->level);
        pyramid_transverse(root->left, level_ids);
        pyramid_transverse(root->right, level_ids);
    }

    class Codec
    {
    public:
        Codec(InitContext *ctx, const Config &config)
            : config(config), frame_size(0),
              decoder(ctx,
                      config.encoder_dim * (1 << MAX_RATES),
                      config.decoder_dim,
                      MAX_RATES, config.decoder_rates,
                      config.noise, config.depthwise, config.attn_window_size),
              quantizer(ctx,
                        config.encoder_dim * (1 << MAX_RATES),
                        config.codebook_size,
                        config.codebook_dim,
                        config.vq_stride_count, config.vq_strides),
              codes(config.vq_stride_count),
              GRAPH_SIZE(2048)
        {
            for (int i = 0; i < config.vq_stride_count; i++)
                frame_size += config.vq_strides[i];


            auto pyramid = make_pyramid(config.vq_stride_count);
            pyramid_transverse(pyramid, this->pyramid);
            delete pyramid;

            CHATLLM_CHECK(this->pyramid.size() == frame_size) << "what?";
        }

        static int64_t number_of_static_tensors(const Config &config)
        {
            int64_t r = 0;
            r += config.depthwise ? 4 : 2;

            const int ResidualUnit_tensors = 2 + 2 + 2 + 2;
            const int64_t decoder_layer_tensor = 2 + (config.noise ? 1 : 0) + 3 * ResidualUnit_tensors;

            r += MAX_RATES * decoder_layer_tensor;
            r += 2 + 2;

            // quantizer
            r += config.vq_stride_count * (2 + 2 + 1);

            return r;
        }

        void load(const std::string &path, ModelLoader &loader)
        {
            decoder.load(path + "decoder.model.layers.", &loader);
            quantizer.load(path + "quantizer.strides.", &loader);
        }

        void decode_frame(const GenerationConfig &gen_config, BackendContext &_context, const std::vector<int> &multiframe, const int64_t count, std::vector<float> &pcm_samples)
        {
            const int num_frames = (int)(multiframe.size() / frame_size);
            const int max_stride = config.vq_strides[0];

            for (int i = 0; i < config.vq_stride_count; i++)
                codes[i].clear();

            for (int j = 0; j < num_frames; j++)
            {
                int i = j * frame_size;

                for (int k : pyramid)
                {
                    codes[k].push_back(multiframe[i++]);
                }
            }

            run_model(gen_config, _context, pcm_samples);
        }
    protected:
        bool run_model(const GenerationConfig &gen_config, BackendContext &backend_context, std::vector<float> &pcm_samples)
        {
            ForwardContext ctx(&backend_context);
            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            std::vector<ggml::tensor *> embed_id;
            for (size_t i = 0; i < codes.size(); i++)
            {
                auto ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, codes[i].size());
                embed_id.push_back(ids_tensor);
            }

            dbg_ctx = &ctx;

            auto dequant = quantizer.dequantize(&ctx, embed_id);
            auto r = decoder.forward(&ctx, dequant);
            r = ggml::scale(&ctx, r, 32767.0f);

            ggml::build_forward_expand(&ctx, r);

            CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";
            CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;
            pcm_samples.resize(ggml::nbytes(r) / sizeof(pcm_samples[0]));

            if (gen_config.dump_dot.size() > 0)
            {
                backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
                exit(-1);
            }

            for (size_t i = 0; i < codes.size(); i++)
            {
                Backend::write_tensor_data(embed_id[i], codes[i].data());
            }

            ctx.compute();
            Backend::read_tensor_data(r, pcm_samples.data());
            ctx.reset();

            return true;
        }
    protected:
        const Config config;
        int frame_size;
        Decoder decoder;
        ResidualVectorQuantize quantizer;
        std::vector<std::vector<int>> codes;
        const size_t GRAPH_SIZE;
        std::vector<int> pyramid;
    };
}

namespace tts
{
    typedef llama::v3_2::Config Config;

    class Tokenizer : public llama::v3_2::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : llama::v3_2::Tokenizer(config)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            auto r = llama::v3_2::Tokenizer::load(buffer, n_vocab);

            terminate_ids.insert(128258);

            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override
        {
            // https://github.com/canopyai/Orpheus-TTS/blob/835ac7cd1870fa1fa5e342e14260c2d68cabb2e7/orpheus_tts_pypi/orpheus_tts/engine_class.py#L89
            ids.push_back(128259);
            ids.push_back(bos_token_id);
            auto prompt = (!voice.empty() ? "{" + voice + "}: " : "") + text;
            BaseTokenizer::encode(prompt, ids);
            ids.push_back(128009);
            ids.push_back(128260);
            ids.push_back(128261);
            ids.push_back(128257);
        }
    public:
        std::string voice;
        const int custom_token_start = 128256;
        const int custom_token_end = 156938;
    };

    class ConditionalGeneration : public llama::v3_2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3_2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_ORPHEUS_TTS),
              snac_ctx(w_ctx_.get_backend_context())
        {
        }

        void load(ModelLoader &loader) override
        {
            llama::v3_2::ConditionalGeneration::load(loader);
            if (codec.get())
            {
                codec->load("snac.", loader);
            }
        }

        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels) override
        {
            channels = 1;
            sample_rate = codec_config.sampling_rate;

            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            #if (1)
            bool completed = false;
            auto tokens = generate(input_ids, gen_config, false, completed, nullptr, 0, nullptr);
            ggml::log(GGML_LOG_LEVEL_INFO, "%zd vocoder tokens generated.", tokens.size());
            #else
            const int tokens[] = {
                128619, 134897, 139983, 141706, 145580, 149427, 152892, 129221, 133623, 138595, 142007, 145911, 149254, 156409, 129221, 134260, 138128, 142007, 145911, 149427, 153768, 132239, 136061, 137972, 141360, 145618, 149427, 153143, 129221, 132577, 140048, 143797, 146425, 149233, 156445, 129221, 132577, 137341, 143894, 144752, 149427, 156409, 128656, 135097, 137341, 141328, 147957, 150072, 155902, 128656,
                135097, 140088, 141920, 147491, 152204, 154509, 129221, 133623, 136787, 142324, 144752, 152313, 154862, 132239, 134858, 137610, 143512, 145486, 150619, 155490, 128343, 133163, 137019, 143444, 147627, 148829, 153322, 128343, 136373, 137738, 141609, 146757, 152383, 155946, 130590, 135962, 137798, 141548, 148250, 152081, 156568, 131187, 133088, 136802, 141510, 147254, 152157, 153471, 129264, 134344,
                140434, 141779, 146818, 151174, 153071, 128462, 135292, 139988, 140932, 147580, 150555, 153377, 128656, 134763, 138225, 141740, 146157, 150111, 154120, 131284, 134350, 138036, 140604, 145911, 149629, 154964, 130966, 135097, 137509, 144121, 147385, 152376, 156807, 131284, 133623, 139263, 140715, 145911, 149629, 154208, 129221, 132464, 136554, 142574, 145911, 149075, 154612, 129221, 133623, 137139,
                143025, 145911, 152494, 156041, 132028, 134350, 137769, 142925, 147609, 152409, 154168, 131132, 135926, 140025, 142972, 147153, 150765, 152951
            };
            #endif

            std::vector<float> pcm_samples;
            reset_decoder();
            for (auto id : tokens)
            {
                if (id < tok->custom_token_start || id > tok->custom_token_end) continue;

                decoder_push_llm_tok_id(gen_config, id - tok->custom_token_start, pcm_samples);
                if (pcm_samples.size() == 8192)
                {
                    for (int i = 2048; i < 4096; i++)
                        audio.push_back((int16_t)pcm_samples[i]);
                }
            }
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
            auto it = args.find("voice");
            if (it != args.end())
            {
                tok->voice = it->second;
            }
        }

        bool load_more(const json::JSON &config) override
        {
            const auto cfg = config["snac_config.json"];
            if (!cfg.IsObject()) return false;

            codec_config.sampling_rate          = (int)cfg["sampling_rate"].ToInt();
            codec_config.encoder_dim            = (int)cfg["encoder_dim"].ToInt();
            codec_config.decoder_dim            = (int)cfg["decoder_dim"].ToInt();
            codec_config.attn_window_size       = cfg["attn_window_size"].IsIntegral() ? (int)cfg["attn_window_size"].ToInt() : -1;
            codec_config.codebook_size          = (int)cfg["codebook_size"].ToInt();
            codec_config.codebook_dim           = (int)cfg["codebook_dim"].ToInt();
            codec_config.noise                  = cfg["noise"].ToBool();
            codec_config.depthwise              = cfg["depthwise"].ToBool();

            CHATLLM_CHECK(cfg["encoder_rates"].IsArray()) << "encoder_rates is not an array";
            CHATLLM_CHECK(cfg["decoder_rates"].IsArray()) << "decoder_rates is not an array";
            CHATLLM_CHECK(cfg["vq_strides"].IsArray()) << "vq_strides is not an array";

            CHATLLM_CHECK(cfg["encoder_rates"].length() == snac::MAX_RATES) << "encoder_rates invalid length";
            CHATLLM_CHECK(cfg["decoder_rates"].length() == snac::MAX_RATES) << "decoder_rates invalid length";
            CHATLLM_CHECK(cfg["vq_strides"].length() <= snac::MAX_VQ_STRIDES) << "vq_strides invalid length";

            for (int i = 0; i < snac::MAX_RATES; i++)
            {
                codec_config.encoder_rates[i] = (int)cfg["encoder_rates"][i].ToInt();
                codec_config.decoder_rates[i] = (int)cfg["decoder_rates"][i].ToInt();
            }

            codec_config.vq_stride_count = (int)cfg["vq_strides"].length();
            for (int i = 0; i < codec_config.vq_stride_count; i++)
            {
                codec_config.vq_strides[i] = (int)cfg["vq_strides"][i].ToInt();
            }

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = snac::Codec::number_of_static_tensors(codec_config) + 20;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            snac_ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            snac_ctx.dtype = llama::v3_2::ConditionalGeneration::config.dtype;

            codec.reset(new snac::Codec(&snac_ctx, codec_config));

            return true;
        }
    protected:
        void reset_decoder(void)
        {
            vocoder_ids.clear();
        }

        void decoder_push_llm_tok_id(const GenerationConfig &gen_config, int id, std::vector<float> &pcm_samples)
        {
            id = id - 10 - ((vocoder_ids.size() % 7) * 4096);
            if (id < 0) return;

            pcm_samples.clear();

            vocoder_ids.push_back(id);
            auto count = vocoder_ids.size();
            if (((count % 7) == 0) && (count >= 28))
            {
                std::vector<int> multiframe(vocoder_ids.end() - 28, vocoder_ids.end());
                codec->decode_frame(gen_config, backend_context, multiframe, count, pcm_samples);
            }
        }

    protected:
        InitContext snac_ctx;
        snac::Config codec_config;
        std::unique_ptr<snac::Codec> codec;
        std::vector<int> vocoder_ids;
    };
}