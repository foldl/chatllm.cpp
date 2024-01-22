template<bool bias> class GenericConditionalGeneration : public BaseModelForConditionalGeneration<
                                    Model<BaseConfig, Embedding, RMSNorm, InternLMBlock<bias>, int, int, int, int, int>>
{
public:
    typedef BaseModelForConditionalGeneration<
                Model<BaseConfig, Embedding, RMSNorm, InternLMBlock<bias>, int, int, int, int, int>> Base;

    GenericConditionalGeneration() = default;
    GenericConditionalGeneration(const BaseConfig &config, ModelType type)
        : GenericConditionalGeneration(config, type, config.num_attention_heads, 10000, 1.0)
    {}

    GenericConditionalGeneration(const BaseConfig &config, ModelType type, int num_key_value_heads, float rope_theta, float rope_scaling)
        : Base(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 3 + config.num_hidden_layers * (bias ? 16 : 12);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        Base::transformer = Model<BaseConfig, Embedding, RMSNorm, InternLMBlock<bias>, int, int, int, int, int>(&w_ctx_, config, false,
                                    config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, num_key_value_heads, config.max_length);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = Base::transformer.layers[i].attention;
            attention.freq_base = rope_theta;
            attention.freq_scale = 1 / rope_scaling;
        }
    }

    void load(ModelLoader &loader) override
    {
        loader.read_tensor("model.embed_tokens.weight", Base::transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", Base::transformer.layers[i].attention.q_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   Base::transformer.layers[i].attention.q_proj.bias);
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", Base::transformer.layers[i].attention.k_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   Base::transformer.layers[i].attention.k_proj.bias);
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", Base::transformer.layers[i].attention.v_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   Base::transformer.layers[i].attention.v_proj.bias);
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", Base::transformer.layers[i].attention.o_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.o_proj.bias",   Base::transformer.layers[i].attention.o_proj.bias);

            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", Base::transformer.layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", Base::transformer.layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   Base::transformer.layers[i].mlp.up_proj.weight);

            loader.read_tensor(layer_prefix + "input_layernorm.weight",          Base::transformer.layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", Base::transformer.layers[i].post_attention_layernorm.weight);
        }
        loader.read_tensor("model.norm.weight", Base::transformer.final_layernorm.weight);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(Base::transformer.lm_head)->weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
            << "corrupted model weights";
    }

public:
    static constexpr size_t MEM_SIZE = 1512ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 144ull * 1024 * 1024;

    BaseConfig config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

class InternLMTokenizer : public BaseTokenizer
{
public:
    InternLMTokenizer(const BaseConfig &config, BaseHistoryEncoder *chat_encoder) : BaseTokenizer::BaseTokenizer(config, chat_encoder)
    {
    };

    size_t load(const char *buffer, int n_vocab) override
    {
        tp = new tokenizer::SentencePieceProcessor();
        size_t size = tp->Load(buffer, n_vocab);

        eoa_token_id = tp->PieceToId("<eoa>");

        return size;
    }

    int get_terminate_token_id(void) const override { return eoa_token_id; }

    bool is_special_id(int id) const override
    {
        return (id == eoa_token_id) || (id == bos_token_id) || (id == eos_token_id);
    }

public:
    int eoa_token_id;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    ChatHistoryEncoder(bool insert_eoh) : insert_eoh(insert_eoh) {}
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override
    {
        std::ostringstream oss_prompt;

        append_user(round_idx, user, ids);

        oss_prompt << ai << "<eoa>\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
    {
        std::ostringstream oss_prompt;

        oss_prompt << "<s>";

        if ((round_idx == 0) && (tokenizer->get_system_prompt().size() > 0))
        {
            oss_prompt << "<|System|>:"
                       << tokenizer->get_system_prompt()
                       << "\n";
        }

        oss_prompt << "<s><|User|>:" << user;
        if (insert_eoh) oss_prompt << "<eoh>";
        oss_prompt << "\n<|Bot|>:";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }
public:
    bool insert_eoh;
};

 namespace v2
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_theta;
        float rope_scaling;
    };

    static ChatHistoryEncoder _chat_encoder(false);

    class Tokenizer : public InternLMTokenizer
    {
    public:
        Tokenizer(const Config &config) : InternLMTokenizer::InternLMTokenizer(config, &_chat_encoder)
        {
            sys_prompt = R""(You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.)"";
        };
    };

    class ConditionalGeneration: public GenericConditionalGeneration<false>
    {
    public:
        ConditionalGeneration(const Config &config)
            : GenericConditionalGeneration(config, MODEL_TYPE_INTERNLM2, config.num_key_value_heads, config.rope_theta, config.rope_scaling)
        {
            // ready for 20B
            GRAPH_SIZE = 4096;
        }
    };
}

namespace v1
{
    struct Config : public BaseConfig
    {
    };

    static ChatHistoryEncoder _chat_encoder(true);

    class Tokenizer : public InternLMTokenizer
    {
    public:
        Tokenizer(const Config &config) : InternLMTokenizer::InternLMTokenizer(config, &_chat_encoder)
        {
        };
    };

    class ConditionalGeneration: public GenericConditionalGeneration<true>
    {
    public:
        ConditionalGeneration(const Config &config)
            : GenericConditionalGeneration(config, MODEL_TYPE_INTERNLM)
        {
        }
    };
}