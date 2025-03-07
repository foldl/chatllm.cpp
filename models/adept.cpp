
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
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
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

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, LayerNorm, PersimmonBlock, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        void load(ModelLoader &loader) override;

    public:
        Config config;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
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

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        append_ai_opening(round_idx, ids);

        oss_prompt << ai << "\n\n";
        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << " human: " << user << "\n\n";
        tok->encode(oss_prompt.str(), ids, true, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("adept: ", ids, true, false);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : BaseModelForConditionalGeneration(MODEL_TYPE_PERSIMMON, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 4 + config.num_hidden_layers * 23;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, false,
                                                            config.hidden_size, config.num_attention_heads,
                                                            config.intermediate_size, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.rope_dim = config.rope_dim;
            layer.attention.freq_base = config.rope_theta;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();

        loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "input_layernorm.bias",   transformer->layers[i].input_layernorm.bias);

            loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer->layers[i].mlp.fc1.weight);
            loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.bias",   transformer->layers[i].mlp.fc1.bias);
            loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer->layers[i].mlp.fc0.weight);
            loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.bias",   transformer->layers[i].mlp.fc0.bias);

            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",   transformer->layers[i].post_attention_layernorm.bias);

            loader.read_tensor(layer_prefix + "self_attn.dense.weight",  transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.dense.bias",    transformer->layers[i].attention.o_proj.bias);

            loader.read_tensor(layer_prefix + "self_attn.k_layernorm.weight",  transformer->layers[i].attention.k_layernorm.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_layernorm.bias",    transformer->layers[i].attention.k_layernorm.bias);
            loader.read_tensor(layer_prefix + "self_attn.q_layernorm.weight",  transformer->layers[i].attention.q_layernorm.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_layernorm.bias",    transformer->layers[i].attention.q_layernorm.bias);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
        }
        loader.read_tensor("model.final_layernorm.weight", transformer->final_layernorm.weight);
        loader.read_tensor("model.final_layernorm.bias",   transformer->final_layernorm.bias);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace fuyu
{

}