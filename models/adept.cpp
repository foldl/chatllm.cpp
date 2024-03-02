
namespace persimmon
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int rope_dim;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(const char *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, LayerNorm, PersimmonBlock, int, int, int, int>>
    {
    public:
        ConditionalGeneration(const Config &config);

        void load(ModelLoader &loader) override;

    public:
        static constexpr size_t MEM_SIZE = 1812ull * 1024 * 1024;
        static constexpr size_t SCRATCH_SIZE = 444ull * 1024 * 1024;

        Config config;

    private:
        // hold ggml_context & kv_cache
        InitContext w_ctx_; // weight context
    };

    size_t Tokenizer::load(const char *buffer, int n_vocab)
    {
        tp = new tokenizer::UnigramProcessor(0);
        size_t size = tp->Load(buffer, n_vocab);
        if (bos_token_id == eos_token_id)
            bos_token_id = 1;
        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
    {
        if (add_bos)
            ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            ids.push_back(eos_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        append_user(round_idx, user, ids);

        oss_prompt << ai << "\n\n";
        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << " human: " << user << "\n\n";
        tok->encode(oss_prompt.str(), ids, true, false);

        oss_prompt.str("");
        oss_prompt << "adept: ";
        tok->encode(oss_prompt.str(), ids, true, false);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config)
        : BaseModelForConditionalGeneration<
                                    Model<Config, Embedding, LayerNorm, PersimmonBlock, int, int, int, int>>(MODEL_TYPE_PERSIMMON, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 4 + config.num_hidden_layers * 23;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = Model<Config, Embedding, LayerNorm, PersimmonBlock, int, int, int, int>(&w_ctx_, config, false,
                                                            config.hidden_size, config.num_attention_heads,
                                                            config.intermediate_size, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = transformer.layers[i];
            layer.attention.rope_dim = config.rope_dim;
            layer.attention.freq_base = config.rope_theta;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';

            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "input_layernorm.bias",   transformer.layers[i].input_layernorm.bias);

            loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.fc1.weight);
            loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.bias",   transformer.layers[i].mlp.fc1.bias);
            loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer.layers[i].mlp.fc0.weight);
            loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.bias",   transformer.layers[i].mlp.fc0.bias);

            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer.layers[i].post_attention_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",   transformer.layers[i].post_attention_layernorm.bias);

            loader.read_tensor(layer_prefix + "self_attn.dense.weight",  transformer.layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.dense.bias",    transformer.layers[i].attention.o_proj.bias);

            loader.read_tensor(layer_prefix + "self_attn.k_layernorm.weight",  transformer.layers[i].attention.k_layernorm.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_layernorm.bias",    transformer.layers[i].attention.k_layernorm.bias);
            loader.read_tensor(layer_prefix + "self_attn.q_layernorm.weight",  transformer.layers[i].attention.q_layernorm.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_layernorm.bias",    transformer.layers[i].attention.q_layernorm.bias);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
        }
        loader.read_tensor("model.final_layernorm.weight", transformer.final_layernorm.weight);
        loader.read_tensor("model.final_layernorm.bias",   transformer.final_layernorm.bias);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
            << "corrupted model weights";
    }
}

namespace fuyu
{

}