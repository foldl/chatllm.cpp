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

        size_t load(const char *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    protected:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos, int max_length) const;
    };

    size_t Tokenizer::load(const char *buffer, int n_vocab)
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

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, BCEFinalNorm, int, int, int, int, int>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : BaseModelForConditionalGeneration<
                EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, BCEFinalNorm, int, int, int, int, int>>(MODEL_TYPE_BCE_Embedding, config, MEM_SIZE, SCRATCH_SIZE),
              config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 5 + config.num_hidden_layers * 19;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, BCEFinalNorm, int, int, int, int, int>(&w_ctx_, config,
                                                                                    config.hidden_size, config.num_attention_heads,
                                                                                    config.intermediate_size, config.num_attention_heads, config.max_length);
        }

        void load(ModelLoader &loader) override
        {
            loader.read_tensor("embeddings.word_embeddings.weight",         transformer.word_embeddings.word_weight);
            loader.read_tensor("embeddings.position_embeddings.weight",     transformer.word_embeddings.position_weight);
            loader.read_tensor("embeddings.LayerNorm.weight",               transformer.word_embeddings.ln.weight);
            loader.read_tensor("embeddings.LayerNorm.bias",                 transformer.word_embeddings.ln.bias);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "encoder.layer." + std::to_string(i) + '.';
                loader.read_tensor(layer_prefix + "attention.self.query.weight",    transformer.layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "attention.self.query.bias",      transformer.layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "attention.self.key.weight",      transformer.layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "attention.self.key.bias",        transformer.layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "attention.self.value.weight",    transformer.layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "attention.self.value.bias",      transformer.layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "attention.output.dense.weight",  transformer.layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "attention.output.dense.bias",    transformer.layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "attention.output.LayerNorm.weight",    transformer.layers[i].post_attention_layernorm.weight);
                loader.read_tensor(layer_prefix + "attention.output.LayerNorm.bias",      transformer.layers[i].post_attention_layernorm.bias);

                loader.read_tensor(layer_prefix + "intermediate.dense.weight",  transformer.layers[i].mlp.intermediate.weight);
                loader.read_tensor(layer_prefix + "intermediate.dense.bias",    transformer.layers[i].mlp.intermediate.bias);
                loader.read_tensor(layer_prefix + "output.dense.weight",        transformer.layers[i].mlp.output.dense.weight);
                loader.read_tensor(layer_prefix + "output.dense.bias",          transformer.layers[i].mlp.output.dense.bias);

                loader.read_tensor(layer_prefix + "output.LayerNorm.weight",    transformer.layers[i].mlp.output.norm.weight);
                loader.read_tensor(layer_prefix + "output.LayerNorm.bias",      transformer.layers[i].mlp.output.norm.bias);
            }

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

        int get_text_embedding_dim(void) const override
        {
            return config.hidden_size;
        }

    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 44ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
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

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaClassificationHead, int, int, int, int, int>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : BaseModelForConditionalGeneration<
                EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaClassificationHead, int, int, int, int, int>>(MODEL_TYPE_BCE_ReRanker, config, MEM_SIZE, SCRATCH_SIZE),
              config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 9 + config.num_hidden_layers * 19;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaClassificationHead, int, int, int, int, int>(&w_ctx_, config,
                                                                                    config.hidden_size, config.num_attention_heads,
                                                                                    config.intermediate_size, config.num_attention_heads, config.max_length);
        }

        void load(ModelLoader &loader) override
        {
            loader.read_tensor("embeddings.word_embeddings.weight",         transformer.word_embeddings.word_weight);
            loader.read_tensor("embeddings.position_embeddings.weight",     transformer.word_embeddings.position_weight);
            loader.read_tensor("embeddings.LayerNorm.weight",               transformer.word_embeddings.ln.weight);
            loader.read_tensor("embeddings.LayerNorm.bias",                 transformer.word_embeddings.ln.bias);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "encoder.layer." + std::to_string(i) + '.';
                loader.read_tensor(layer_prefix + "attention.self.query.weight",    transformer.layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "attention.self.query.bias",      transformer.layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "attention.self.key.weight",      transformer.layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "attention.self.key.bias",        transformer.layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "attention.self.value.weight",    transformer.layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "attention.self.value.bias",      transformer.layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "attention.output.dense.weight",  transformer.layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "attention.output.dense.bias",    transformer.layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "attention.output.LayerNorm.weight",    transformer.layers[i].post_attention_layernorm.weight);
                loader.read_tensor(layer_prefix + "attention.output.LayerNorm.bias",      transformer.layers[i].post_attention_layernorm.bias);

                loader.read_tensor(layer_prefix + "intermediate.dense.weight",  transformer.layers[i].mlp.intermediate.weight);
                loader.read_tensor(layer_prefix + "intermediate.dense.bias",    transformer.layers[i].mlp.intermediate.bias);
                loader.read_tensor(layer_prefix + "output.dense.weight",        transformer.layers[i].mlp.output.dense.weight);
                loader.read_tensor(layer_prefix + "output.dense.bias",          transformer.layers[i].mlp.output.dense.bias);

                loader.read_tensor(layer_prefix + "output.LayerNorm.weight",    transformer.layers[i].mlp.output.norm.weight);
                loader.read_tensor(layer_prefix + "output.LayerNorm.bias",      transformer.layers[i].mlp.output.norm.bias);
            }

            loader.read_tensor("classifier.dense.weight",       transformer.final.dense.weight);
            loader.read_tensor("classifier.dense.bias",         transformer.final.dense.bias);
            loader.read_tensor("classifier.out_proj.weight",    transformer.final.out_proj.weight);
            loader.read_tensor("classifier.out_proj.bias",      transformer.final.out_proj.bias);

            CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
                << "corrupted model weights";
        }

    public:
        static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 44ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };
}