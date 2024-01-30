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
    };

    size_t Tokenizer::load(const char *buffer, int n_vocab)
    {
        tp = new tokenizer::UnigramProcessor();
        tp->RegisterPreprocessor(new tokenizer::TextPrepDeleteMultiSpaces());
        tp->RegisterPreprocessor(new tokenizer::TextPrepAddLeadingSpace());
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        ids.push_back(eos_token_id);
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaPooler, int, int, int, int, int>>
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config)
            : BaseModelForConditionalGeneration<
                EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaPooler, int, int, int, int, int>>(MODEL_TYPE_BCE_Embedding, config, MEM_SIZE, SCRATCH_SIZE),
              config(config)
        {
            constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
            const size_t num_tensors = 7 + config.num_hidden_layers * 19;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = EmbeddingModel<Config, RobertaEmbedding, RobertaBlock, RobertaPooler, int, int, int, int, int>(&w_ctx_, config,
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

            loader.read_tensor("pooler.dense.weight",   transformer.final.dense.weight);
            loader.read_tensor("pooler.dense.bias",     transformer.final.dense.bias);

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