namespace dac
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
        int codebook_size;
        int codebook_dim;
        int n_codebooks;
    };

    using orpheus::snac::Snake1D;
    using orpheus::snac::ResidualUnit;
    using orpheus::snac::DecoderBlock;
    using orpheus::snac::Decoder;
    using orpheus::snac::VectorQuantize;
    using orpheus::snac::ResidualVectorQuantize;

    class Codec
    {
    public:
        Codec(InitContext *ctx, const Config &config)
            : config(config), frame_size(config.n_codebooks),
              decoder(ctx,
                      config.encoder_dim * (1 << MAX_RATES),
                      config.decoder_dim,
                      MAX_RATES, config.decoder_rates,
                      false, false, -1, false),
              quantizer(ctx,
                        config.encoder_dim * (1 << MAX_RATES),
                        config.codebook_size,
                        config.codebook_dim,
                        config.n_codebooks, nullptr),
              codes(config.n_codebooks),
              GRAPH_SIZE(2048)
        {
        }

        static int64_t number_of_static_tensors(const Config &config)
        {
            int64_t r = 0;
            r += 2;

            const int ResidualUnit_tensors = 2 + 2 + 2 + 2;
            const int64_t decoder_layer_tensor = 2 + 3 * ResidualUnit_tensors;

            r += MAX_RATES * decoder_layer_tensor;
            r += 2 + 2;

            // quantizer
            r += config.n_codebooks * (2 + 2 + 1);

            return r;
        }

        void load(const std::string &path, ModelLoader &loader)
        {
            decoder.load(path + "decoder.model.layers.", &loader);
            quantizer.load(path + "quantizer.strides.", &loader);
        }

        void decode_frame(const GenerationConfig &gen_config, BackendContext &_context, const std::vector<int> &multiframe, std::vector<float> &pcm_samples)
        {
            const int num_frames = (int)(multiframe.size() / frame_size);

            for (int i = 0; i < frame_size; i++)
                codes[i].clear();

            for (int j = 0; j < num_frames; j++)
            {
                int i = j * frame_size;

                for (int k = 0; k < frame_size; k++)
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
        const int frame_size;
        Decoder decoder;
        ResidualVectorQuantize quantizer;
        std::vector<std::vector<int>> codes;
        const size_t GRAPH_SIZE;
    };

    class CodecGeneration
    {
    public:
        CodecGeneration(BackendContext *backend_context, const RuntimeConfig &runtime_config)
            : backend_context(backend_context),
              dac_ctx(backend_context)
        {
        }

        void load(ModelLoader &loader)
        {
            if (codec.get())
            {
                codec->load("dac.", loader);
            }
        }

        bool load_more(ggml::type dtype, const json::JSON &config)
        {
            const auto cfg = config["dac_config.json"];
            if (!cfg.IsObject()) return false;

            this->config.sampling_rate          = (int)cfg["sampling_rate"].ToInt();
            this->config.encoder_dim            = (int)cfg["encoder_hidden_size"].ToInt();
            this->config.decoder_dim            = (int)cfg["decoder_hidden_size"].ToInt();
            this->config.codebook_size          = (int)cfg["codebook_size"].ToInt();
            this->config.codebook_dim           = (int)cfg["codebook_dim"].ToInt();
			this->config.n_codebooks			= (int)cfg["n_codebooks"].ToInt();

            CHATLLM_CHECK(cfg["downsampling_ratios"].IsArray()) << "downsampling_ratios is not an array";
            CHATLLM_CHECK(cfg["upsampling_ratios"].IsArray()) << "upsampling_ratios is not an array";

            CHATLLM_CHECK(cfg["downsampling_ratios"].length() == dac::MAX_RATES) << "downsampling_ratios invalid length";
            CHATLLM_CHECK(cfg["upsampling_ratios"].length() == dac::MAX_RATES) << "upsampling_ratios invalid length";

            for (int i = 0; i < dac::MAX_RATES; i++)
            {
                this->config.encoder_rates[i] = (int)cfg["downsampling_ratios"][i].ToInt();
                this->config.decoder_rates[i] = (int)cfg["upsampling_ratios"][i].ToInt();
            }

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = dac::Codec::number_of_static_tensors(this->config) + 20;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            dac_ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            dac_ctx.dtype = dtype;

            codec.reset(new dac::Codec(&dac_ctx, this->config));

            return true;
        }

        void generate(const GenerationConfig &gen_config,
            const std::vector<int> &codebook1, const std::vector<int> &codebook2,
            std::vector<float> &pcm_samples)
        {
            if (codec.get())
            {
                std::vector<int> frames;
                const size_t count = codebook1.size() <= codebook2.size() ? codebook1.size() : codebook2.size();
                for (size_t i = 0; i < count; i++)
                {
                    frames.push_back(codebook1[i]);
                    frames.push_back(codebook2[i]);
                }

                codec->decode_frame(gen_config, *backend_context, frames, pcm_samples);
            }
        }
    public:
        dac::Config config;
    protected:
        BackendContext *backend_context;
        InitContext dac_ctx;
        std::unique_ptr<dac::Codec> codec;
    };
}

namespace tts_llama
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

            c1_0_token_id       = tp->PieceToId("<|c1_0|>");
            c2_0_token_id       = tp->PieceToId("<|c2_0|>");
            audio_end_token_id  = tp->PieceToId("<|audio_end|>");
            terminate_ids.insert(audio_end_token_id);

            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override;
    public:
        int c1_0_token_id;
        int c2_0_token_id;
        int audio_end_token_id;
    };

    // translation of:
    // https://github.com/edwko/OuteTTS/blob/main/outetts/version/v3/prompt_processor.py
    // https://github.com/edwko/OuteTTS/blob/main/outetts/version/v3/tokens.py

    json::JSON speaker;

    // Special Tokens structure
    struct SpecialTokens
    {
        std::string bos = "<|im_start|>";
        std::string eos = "<|im_end|>";
        std::string c1 = "<|c1_{}|>";
        std::string c2 = "<|c2_{}|>";
        std::string text_start = "<|text_start|>";
        std::string text_end = "<|text_end|>";
        std::string voice_characteristic_start = "<|voice_characteristic_start|>";
        std::string voice_characteristic_end = "<|voice_characteristic_end|>";
        std::string emotion_start = "<|emotion_start|>";
        std::string emotion_end = "<|emotion_end|>";
        std::string audio_start = "<|audio_start|>";
        std::string audio_end = "<|audio_end|>";
        std::string time = "<|t_{}|>";
        std::string code = "<|code|>";
        std::string energy = "<|energy_{}|>";
        std::string spectral_centroid = "<|spectral_centroid_{}|>";
        std::string pitch = "<|pitch_{}|>";
        std::string word_start = "<|word_start|>";
        std::string word_end = "<|word_end|>";
        std::string features = "<|features|>";
        std::string global_features_start = "<|global_features_start|>";
        std::string global_features_end = "<|global_features_end|>";
    };

    std::string text_normalization(std::string result)
    {
        result = std::regex_replace(result, std::regex("\\s+"), " ");
        result = utils::replace_all(result, "…", "...");

        result = utils::trim(result);

        // Normalize common Unicode characters to ASCII equivalents
        const static std::pair<std::string, std::string> patterns[] = {
            {"“", "\""}, {"”", "\""}, // Curly quotes to straight quotes
            {"‘", "'"},  {"’", "'"},  // Curly single quotes
            {"–", "-"},  {"—", "-"},  // Various dashes to hyphen
        };

        for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++)
        {
            result = utils::replace_all(result, patterns[i].first, patterns[i].second);
        }

        return result;
    }

    std::string format_string(const std::string& format_str, const std::string& value)
    {
        return utils::replace_all(format_str, "{}", value);
    }

    std::string format_string(const std::string& format_str, double value)
    {
        char buffer[100];
        snprintf(buffer, sizeof(buffer), "%.2f", value);
        return format_string(format_str, std::string(buffer));
    }

    std::string format_string(const std::string& format_str, int value)
    {
        return format_string(format_str, std::to_string(value));
    }

    // Function to get features
    std::vector<std::string> get_features(const json::JSON& f, const SpecialTokens& special_tokens)
    {
        std::vector<std::string> result;

        int energy = f.hasKey("energy") ? (int)f["energy"].ToInt() : 0;
        int spectral_centroid = f.hasKey("spectral_centroid") ? (int)f["spectral_centroid"].ToInt() : 0;
        int pitch = f.hasKey("pitch") ? (int)f["pitch"].ToInt() : 0;

        result.push_back(format_string(special_tokens.energy, energy));
        result.push_back(format_string(special_tokens.spectral_centroid, spectral_centroid));
        result.push_back(format_string(special_tokens.pitch, pitch));

        return result;
    }

    // Function to get global features
    std::string get_global_features(const json::JSON& f, const SpecialTokens& special_tokens, const std::string& global_features_template)
    {
        std::vector<std::string> features = get_features(f, special_tokens);
        std::string codes;
        for (const auto& feature : features)
        {
            codes += feature;
        }

        std::string result = global_features_template;

        result = utils::replace_all(result, "{fs}",         special_tokens.global_features_start);
        result = utils::replace_all(result, "{codes}",      codes);
        result = utils::replace_all(result, "{fe}",         special_tokens.global_features_end);

        return result;
    }

    // Function to create codes
    std::string create_codes(const json::JSON& words, const SpecialTokens& special_tokens)
    {
        std::vector<std::string> codes;

        for (const auto& word_item : words.ArrayRange())
        {
            std::string word = word_item["word"].ToString() + special_tokens.features;
            word += format_string(special_tokens.time, word_item["duration"].ToFloat());

            // Add features
            std::vector<std::string> features = get_features(word_item["features"], special_tokens);
            for (const auto& feature : features)
            {
                word += feature;
            }

            // Add pairs of c1 and c2
            std::vector<std::string> pairs;
            for (size_t idx = 0; idx < word_item["c1"].length(); idx++)
            {
                std::string c1 = format_string(special_tokens.c1, (int)word_item["c1"][(unsigned)idx].ToInt());
                std::string c2 = format_string(special_tokens.c2, (int)word_item["c2"][(unsigned)idx].ToInt());
                pairs.push_back(c1 + c2);
            }

            word += special_tokens.code;
            for (const auto& pair : pairs)
            {
                word += pair;
            }

            codes.push_back(special_tokens.word_start + word + special_tokens.word_end);
        }

        // Join codes with newline
        std::string result;
        for (size_t i = 0; i < codes.size(); i++)
        {
            result += codes[i];
            if (i < codes.size() - 1)
            {
                result += "\n";
            }
        }

        return result;
    }

    // Function to initialize prompt
    std::string init_prompt(const std::string& text, const SpecialTokens& special_tokens, const std::string& input_prompt_template)
    {
        std::string result = input_prompt_template;

        result = utils::replace_all(result, "{bos}",            special_tokens.bos);
        result = utils::replace_all(result, "{text_start}",     special_tokens.text_start);
        result = utils::replace_all(result, "{text}",           text);
        result = utils::replace_all(result, "{text_end}",       special_tokens.text_end);
        result = utils::replace_all(result, "{audio_start}",    special_tokens.audio_start);

        return result;
    }

    // Function to get separator based on text
    std::string get_separator(const std::string& text)
    {
        bool has_hiragana = false;
        bool has_katakana = false;
        bool has_han = false;
        bool has_hangul = false;

        for (char c : text)
        {
            unsigned char uc = static_cast<unsigned char>(c);
            if (uc >= 0xE3 && uc <= 0xE3)       // Simplified check for hiragana (U+3040-U+309F)
                has_hiragana = true;
            else if (uc >= 0xE3 && uc <= 0xE3)  // Simplified check for katakana (U+30A0-U+30FF)
                has_katakana = true;
            else if (uc >= 0xE4 && uc <= 0xE9)  // Simplified check for han (U+4E00-U+9FFF)
                has_han = true;
            else if (uc >= 0xEA && uc <= 0xED)  // Simplified check for hangul (U+AC00-U+D7AF)
                has_hangul = true;
        }

        if (has_hiragana || has_katakana || has_han)
            return "。";
        else if (has_hangul)
            return ". ";
        else
            return ". ";
    }

    std::pair<std::string, std::string> merge_speaker_text(const std::string& input_text, const std::string& speaker_text_orig)
    {
        std::string speaker_text = utils::trim(speaker_text_orig);
        std::string separator = get_separator(speaker_text);

        // Determine allowed endings based on the separator
        std::vector<std::string> allowed_ends;
        if (separator == "。")
            allowed_ends = {"。", "？", "！", "?", "!"};
        else
            allowed_ends = {".", "?", "!"};

        std::string rs = ""; // This will be the separator/space to insert

        if (!speaker_text.empty())
        {
            bool ends_with_allowed_char = false;
            for (const std::string& end_char : allowed_ends)
            {
                if (utils::ends_with(speaker_text, end_char))
                {
                    ends_with_allowed_char = true;
                    break;
                }
            }

            if (!ends_with_allowed_char)
            {
                rs = separator;
            }
            else
            {
                if (separator != "。")
                    rs = " ";
            }
        }

        std::string output = speaker_text + rs + utils::trim(input_text);
        std::string trimmed_rs = utils::trim(rs);
        return std::make_pair(output, trimmed_rs);
    }

    // Main function to get completion prompt
    std::string get_completion_prompt(const std::string& text, json::JSON& speaker)
    {
        // Initialize special tokens
        SpecialTokens special_tokens;

        // Templates (would normally be passed as parameters)
        std::string input_prompt_template = "{bos}{text_start}{text}{text_end}\n{audio_start}\n";
        std::string global_features_template = "{fs}{codes}{fe}";

        // Normalize text

        std::string normalized_text = text_normalization(text);

        std::string prompt;
        if (!speaker.IsNull())
        {
            // Merge speaker text
            auto [merged_text, separator] = merge_speaker_text(normalized_text, speaker["text"].ToString());
            normalized_text = merged_text;

            // Update last word with separator if necessary
            if (!separator.empty())
            {
                auto last = (unsigned)(speaker["words"].length() - 1);
                speaker["words"][last]["word"] = speaker["words"][last]["word"].ToString() + separator;
            }

            // Create codes
            std::string codes = create_codes(speaker["words"], special_tokens);

            // Initialize prompt
            prompt = init_prompt(normalized_text, special_tokens, input_prompt_template);

            // Add codes and word_start
            prompt += codes + "\n" + special_tokens.word_start;
        }
        else
        {
            // Initialize prompt without speaker
            prompt = init_prompt(normalized_text, special_tokens, input_prompt_template);
        }

        return prompt;
    }

    std::string get_completion_prompt(const std::string& text)
    {
        json::JSON clone(speaker);
        return get_completion_prompt(text, clone);
    }

    static void load_speaker_from_args(const std::map<std::string, std::string> &args)
    {
        auto x = args.find("speaker");
        if (x != args.end())
        {
            speaker = json::JSON::Load(utils::load_file(x->second.c_str()));
        }
    }

    std::vector<std::vector<int>> extract_codebooks(const std::string& codes) {
        std::vector<int> codebook1;
        std::vector<int> codebook2;

        // Use regex to find all matches
        std::regex pattern1("<\\|c1_(\\d+)\\|>");
        std::regex pattern2("<\\|c2_(\\d+)\\|>");

        // Iterator for regex matches
        std::sregex_iterator iter1(codes.begin(), codes.end(), pattern1);
        std::sregex_iterator iter2(codes.begin(), codes.end(), pattern2);
        std::sregex_iterator end;

        // Extract codebook1 values
        for (; iter1 != end; ++iter1) {
            std::smatch match = *iter1;
            codebook1.push_back(std::stoi(match[1]));
        }

        // Extract codebook2 values
        for (; iter2 != end; ++iter2) {
            std::smatch match = *iter2;
            codebook2.push_back(std::stoi(match[1]));
        }

        // Truncate to the minimum size of both codebooks
        size_t t = std::min(codebook1.size(), codebook2.size());
        codebook1.resize(t);
        codebook2.resize(t);

        return {codebook1, codebook2};
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        auto prompt = get_completion_prompt(text);
        BaseTokenizer::encode(prompt, ids);
    }

    class ConditionalGeneration : public llama::v3_2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3_2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_OUTE_TTS_LLAMA),
              codec(w_ctx_.get_backend_context(), runtime_config)
        {
        }

        void load(ModelLoader &loader) override
        {
            llama::v3_2::ConditionalGeneration::load(loader);
            codec.load(loader);
        }

        bool load_more(const json::JSON &config) override
        {
            llama::v3_2::ConditionalGeneration::load_more(config);
            return codec.load_more(llama::v3_2::ConditionalGeneration::config.dtype, config);
        }

        static void generate_audio(const GenerationConfig &gen_config, dac::CodecGeneration &codec,
            const int C1_FIRST, const int C1_LAST,
            const int C2_FIRST, const int C2_LAST,
            const std::vector<int> &ids, std::vector<int16_t> &audio)
        {
            std::vector<float> pcm_samples;

            std::vector<int> codebook1;
            std::vector<int> codebook2;
            for (auto x : ids)
            {
                if ((C1_FIRST <= x) && (x <= C1_LAST))
                    codebook1.push_back(x - C1_FIRST);
                else if ((C2_FIRST <= x) && (x <= C2_LAST))
                    codebook2.push_back(x - C2_FIRST);
            }

            codec.generate(gen_config, codebook1, codebook2, pcm_samples);

            for (size_t i = 0; i < pcm_samples.size(); i++)
                audio.push_back((int16_t)pcm_samples[i]);
        }

        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels) override
        {
            channels = 1;
            sample_rate = codec.config.sampling_rate;

            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

            bool completed = false;
            auto tokens = generate(input_ids, gen_config, false, completed, nullptr, 0, nullptr);
            ggml::log(GGML_LOG_LEVEL_INFO, "%zd vocoder tokens generated.", tokens.size());

            generate_audio(gen_config, codec,
                tok->c1_0_token_id, tok->c1_0_token_id + codec.config.codebook_size - 1,
                tok->c2_0_token_id, tok->c2_0_token_id + codec.config.codebook_size - 1,
                tokens, audio);
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            load_speaker_from_args(args);
        }

    public:
        dac::CodecGeneration codec;
    };
}

namespace tts_qwen3
{
    typedef qwen::v3::Config Config;

    class Tokenizer : public qwen::v3::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : qwen::v3::Tokenizer(config)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            auto r = qwen::v3::Tokenizer::load(buffer, n_vocab);

            c1_0_token_id       = tp->PieceToId("<|c1_0|>");
            c2_0_token_id       = tp->PieceToId("<|c2_0|>");
            audio_end_token_id  = tp->PieceToId("<|audio_end|>");
            terminate_ids.insert(audio_end_token_id);

            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override;
    public:
        std::string voice;
        int c1_0_token_id;
        int c2_0_token_id;
        int audio_end_token_id;
    };

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        auto prompt = tts_llama::get_completion_prompt(text);
        BaseTokenizer::encode(prompt, ids);
    }

    class ConditionalGeneration : public qwen::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : qwen::v3::ConditionalGeneration(config, runtime_config, MODEL_TYPE_OUTE_TTS_QWEN3),
              codec(w_ctx_.get_backend_context(), runtime_config)
        {
        }

        void load(ModelLoader &loader) override
        {
            qwen::v3::ConditionalGeneration::load(loader);
            codec.load(loader);
        }

        bool load_more(const json::JSON &config) override
        {
            qwen::v3::ConditionalGeneration::load_more(config);
            return codec.load_more(qwen::v3::ConditionalGeneration::config.dtype, config);
        }

        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels) override
        {
            channels = 1;
            sample_rate = codec.config.sampling_rate;

            Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

            bool completed = false;
            auto tokens = generate(input_ids, gen_config, false, completed, nullptr, 0, nullptr);
            ggml::log(GGML_LOG_LEVEL_INFO, "%zd vocoder tokens generated.", tokens.size());

            tts_llama::ConditionalGeneration::generate_audio(gen_config, codec,
                tok->c1_0_token_id, tok->c1_0_token_id + codec.config.codebook_size - 1,
                tok->c2_0_token_id, tok->c2_0_token_id + codec.config.codebook_size - 1,
                tokens, audio);
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            tts_llama::load_speaker_from_args(args);
        }

    public:
        dac::CodecGeneration codec;
    };
}