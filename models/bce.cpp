namespace embedding
{
    struct Config : public BaseConfig
    {
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : BaseTokenizer::BaseTokenizer(config, nullptr)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    protected:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos, int max_length) const;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::UnigramProcessor(eos_token_id + 1);
        tp->RegisterPreprocessor(new tokenizer::TextPrepNewlineToSpaces());
        tp->RegisterPreprocessor(new tokenizer::TextPrepDeleteMultiSpaces());
        tp->RegisterPreprocessor(new tokenizer::TextPrepAddLeadingSpace());
        size_t size = tp->Load(buffer, n_vocab);

        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos, int max_length) const
    {
        if (add_bos) max_length--;
        if (add_eos) max_length--;

        if (add_bos)
            ids.push_back(bos_token_id);
        size_t start = ids.size();
        BaseTokenizer::encode(text, ids);
        size_t length = ids.size() - start;
        if ((max_length > 0) && ((int)length > max_length))
            ids.resize(start + max_length);

        if (add_eos)
            ids.push_back(eos_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        // position embedding offset = 2
        encode(text, ids, true, true, max_length - 2);
    }

    void fix_tensor_names(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {"model.embed_tokens.",         "embeddings."},
            {"model.layers.",               "encoder.layer."},
            {".ln.",                        ".LayerNorm."},
            {".mlp.",                       "."},
            {".q_proj.",                    ".query."},
            {".k_proj.",                    ".key."},
            {".v_proj.",                    ".value."},
            {".attention.self.o_proj.",     ".attention.output.dense."},
            {".post_attention_layernorm.",  ".attention.output.LayerNorm."},
            {".output_layernorm.",          ".output.LayerNorm."},
        });
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, BCEFinalNorm, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
            : BaseModelForConditionalGeneration(type, config, runtime_config),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 5 + config.num_hidden_layers * 19;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_attention_heads, config.max_length);
        }

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ConditionalGeneration(config, runtime_config, MODEL_TYPE_BCE_Embedding)
        {}

        void load(ModelLoader &loader) override
        {
            fix_tensor_names(loader);

            BaseModelForConditionalGeneration::load(loader);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

        int get_text_embedding_dim(void) const override
        {
            return config.hidden_size;
        }

    public:
        Config config;
    };
}

namespace ranker
{
    struct Config : public embedding::Config
    {
    };

    class Tokenizer : public embedding::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : embedding::Tokenizer(config)
        {
        }

        void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const override
        {
            const int max_length = this->max_length - 2;

            std::vector<int> ids_q;
            std::vector<int> ids_a;
            BaseTokenizer::encode(q, ids_q);
            BaseTokenizer::encode(a, ids_a);

            int total = (int)ids_q.size() + (int)ids_a.size();

            // this is bad
            if (total > max_length - 4)
            {
                int remain = max_length - 4 - (int)ids_q.size();
                CHATLLM_CHECK(remain > 0) << "query is TOOOO long.";

                ids_a.resize(remain);
            }

            ids.push_back(bos_token_id);
            ids.insert(std::end(ids), std::begin(ids_q), std::end(ids_q));
            ids.push_back(eos_token_id);

            ids.push_back(eos_token_id);
            ids.insert(std::end(ids), std::begin(ids_a), std::end(ids_a));
            ids.push_back(eos_token_id);
        }
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaClassificationHead, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ConditionalGeneration(config, runtime_config, MODEL_TYPE_BCE_ReRanker)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
            : BaseModelForConditionalGeneration(type, config, runtime_config),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 9 + config.num_hidden_layers * 19;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_attention_heads, config.max_length);
        }

        void load(ModelLoader &loader) override
        {
            embedding::fix_tensor_names(loader);
            loader.add_tensor_name_translations({
                {"model.norm.",           "classifier."},
            });

            BaseModelForConditionalGeneration::load(loader);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };
}