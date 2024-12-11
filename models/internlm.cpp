template<bool bias> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef BaseModelForConditionalGeneration Base;
    typedef Model<BaseConfig, Embedding, RMSNorm, InternLMBlock<bias>, int, int, int, int, int> TransformerClass;

    GenericConditionalGeneration() = default;
    GenericConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type)
        : GenericConditionalGeneration(config, runtime_config, type, config.num_attention_heads, 10000, 1.0)
    {}

    GenericConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type, int num_key_value_heads, float rope_theta, float rope_scaling)
        : Base(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * (bias ? 16 : 12);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        Base::transformer = new TransformerClass(&w_ctx_, config, false,
                                    config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, num_key_value_heads, config.max_length);

        auto transformer = Base::get_typed_transformer<TransformerClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = rope_theta;
            attention.freq_scale = 1 / rope_scaling;
        }
    }

    void load(ModelLoader &loader) override
    {
        TransformerClass *transformer = dynamic_cast<TransformerClass *>(Base::transformer);

        loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
            if (true)
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            if (bias)
                loader.read_tensor(layer_prefix + "self_attn.o_proj.bias",   transformer->layers[i].attention.o_proj.bias);

            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);

            loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);
        }
        loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

public:
    BaseConfig config;
};

class InternLMTokenizer : public BaseTokenizer
{
public:
    InternLMTokenizer(const BaseConfig &config, BaseHistoryEncoder *chat_encoder) : BaseTokenizer::BaseTokenizer(config, chat_encoder)
    {
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        eoa_token_id = tp->PieceToId("<eoa>");

        terminate_ids.insert(eoa_token_id);

        return size;
    }

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
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
    {
        std::ostringstream oss_prompt;

        append_ai_opening(round_idx, ids);

        oss_prompt << ai << "<eoa>\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void append_sys_prompt(std::vector<int> &ids) const override
    {
        std::ostringstream oss_prompt;
        oss_prompt << "<s>";

        if (tokenizer->get_system_prompt().size() > 0)
        {
            oss_prompt << "<|System|>:"
                       << tokenizer->get_system_prompt()
                       << "\n";
        }
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
    {
        std::ostringstream oss_prompt;

        oss_prompt << "<s><|User|>:" << user;
        if (insert_eoh) oss_prompt << "<eoh>";
        oss_prompt << "\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void append_ai_opening(int round_idx, std::vector<int> &ids) const override
    {
        tokenizer->encode("<|Bot|>:", ids);
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
        }
    };

    class ConditionalGeneration: public GenericConditionalGeneration<false>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : GenericConditionalGeneration(config, runtime_config, MODEL_TYPE_INTERNLM2, config.num_key_value_heads, config.rope_theta, config.rope_scaling)
        {
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
        }
    };

    class ConditionalGeneration: public GenericConditionalGeneration<true>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : GenericConditionalGeneration(config, runtime_config, MODEL_TYPE_INTERNLM)
        {
        }
    };
}

namespace v3
{
    struct Config : public v2::Config
    {
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        ChatHistoryEncoder() {}
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : BaseTokenizer::BaseTokenizer(config, &_chat_encoder)
        {
            sys_prompt = R""(You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.)"";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor1();
            size_t size = tp->Load(buffer, n_vocab);

            int id = n_vocab;
            im_start_token_id       = --id;
            im_end_token_id         = --id;
            action_start_token_id   = --id;
            action_end_token_id     = --id;
            interpreter_token_id    = --id;
            plugin_token_id         = --id;

            newline_token_id = tp->PieceToId("\n");

            tp->AddAddedToken("<|action_start|>",       action_start_token_id);
            tp->AddAddedToken("<|action_end|>",         action_end_token_id);
            tp->AddAddedToken("<|interpreter|>",        interpreter_token_id);
            tp->AddAddedToken("<|plugin|>",             plugin_token_id);

            terminate_ids.insert(im_end_token_id);

            return size;
        }

        bool is_special_id(int id) const override
        {
            return (id == bos_token_id) || (id == eos_token_id) || (im_end_token_id == id) || (id == im_start_token_id);
        }

        void encode(const std::string &text, std::vector<int> &ids, bool add_im_start, bool add_im_end)
        {
            if (add_im_start)
                ids.push_back(im_start_token_id);
            BaseTokenizer::encode(text, ids);
            if (add_im_end)
            {
                ids.push_back(im_end_token_id);
                ids.push_back(newline_token_id);
            }
        }
    public:
        int im_start_token_id;
        int im_end_token_id;
        int action_start_token_id;
        int action_end_token_id;
        int interpreter_token_id;
        int plugin_token_id;
        int newline_token_id;
    };

    class InternLMInterceptor : public ChunkInterceptor
    {
    public:
        InternLMInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(streamer->tokenizer);
            if (tok->action_start_token_id < 0)
            {
                streamer->put_chunk(first, chunk);
                return;
            }

            if (!found_tool_call)
            {
                if (chunk.starts_with("<|action_start|>"))
                {
                    found_tool_call = true;
                    return;
                }
            }
            else
            {
                if (chunk.starts_with("<|action_end|>"))
                {
                    found_tool_call = false;
                    return;
                }
            }

            if (found_tool_call)
                oss << chunk;
            else
                streamer->put_chunk(first, chunk);
        }

        void end() override
        {
            std::string tool_call = oss.str();
            if (tool_call.size() > 0)
                streamer->putln(tool_call, BaseStreamer::TextType::TOOL_CALLING);

            oss.str("");
            found_tool_call = false;

            ChunkInterceptor::end();
        }
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };

    static InternLMInterceptor interceptor;

    class ConditionalGeneration: public GenericConditionalGeneration<false>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : GenericConditionalGeneration(config, runtime_config, MODEL_TYPE_INTERNLM3, config.num_key_value_heads, config.rope_theta, config.rope_scaling)
        {
        }

        ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            oss_prompt << "system\n"
                        << tok->get_system_prompt();
            tok->encode(oss_prompt.str(), ids, true, true);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt.str("");
        oss_prompt << "user\n"
                   << user;
        tok->encode(oss_prompt.str(), ids, true, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;
        oss_prompt << "assistant\n";
        tok->encode(oss_prompt.str(), ids, true, false);
    }
}
