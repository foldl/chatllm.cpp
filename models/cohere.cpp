namespace command_r
{
struct Config : public BaseConfig
{
    int num_key_value_heads;
    float rope_theta;
    float logit_scale;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    void append_user_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer(config, encoder)
    {
        sys_prompt = R"""(You are a powerful conversational AI trained by Cohere to help people.)""";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "\\p{N}",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);

        start_of_turn_token_id  = tp->PieceToId("<|START_OF_TURN_TOKEN|>");
        end_of_turn_token_id    = tp->PieceToId("<|END_OF_TURN_TOKEN|>");
        user_token_id           = tp->PieceToId("<|USER_TOKEN|>");
        chatbot_token_id        = tp->PieceToId("<|CHATBOT_TOKEN|>");
        system_token_id         = tp->PieceToId("<|SYSTEM_TOKEN|>");
        terminate_ids.insert(end_of_turn_token_id);
        return size;
    }

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_start, int start_token, bool add_end)
    {
        if (add_start)
        {
            ids.push_back(start_of_turn_token_id);
            if (start_token >= 0)
                ids.push_back(start_token);
        }
        BaseTokenizer::encode(text, ids);
        if (add_end)
        {
            ids.push_back(end_of_turn_token_id);
        }
    }

public:
    int start_of_turn_token_id;
    int end_of_turn_token_id;
    int user_token_id;
    int chatbot_token_id;
    int system_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<BaseConfig, Embedding, LayerNormNoBias, CohereBlock, int, int, int, int, int> ModelClass;

public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_COHERE_COMMAND_R)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 2 + config.num_hidden_layers * 11;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config,
                nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        logit_scale = config.logit_scale;
    }

    void load(ModelLoader &loader) override
    {
        auto transformer = get_typed_transformer<ModelClass>();

        loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer->layers[i].mlp.up_proj.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
        }
        loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

public:
    BaseConfig config;
};

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_ai_opening(round_idx, ids);

    tok->encode(ai, ids, false, -1, true);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->bos_token_id);
    if (tok->get_system_prompt().size() > 0)
    {
        tok->encode(tok->get_system_prompt(), ids, true, tok->system_token_id, true);
    }
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    tok->encode(user, ids, true, tok->user_token_id, true);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->start_of_turn_token_id);
    ids.push_back(tok->chatbot_token_id);
}

void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->start_of_turn_token_id);
    ids.push_back(tok->user_token_id);
}

}

namespace aya_23
{
typedef command_r::Config Config;

typedef command_r::Tokenizer Tokenizer;

class ConditionalGeneration : public command_r::ConditionalGeneration
{
public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : command_r::ConditionalGeneration(config, runtime_config, MODEL_TYPE_COHERE_AYA_23) {}
};
}

namespace v2
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int sliding_window;
        int sliding_window_pattern;
        float rope_theta;
        float logit_scale;
    };

    typedef command_r::Tokenizer Tokenizer;

    template <int sliding_window_len> class Cohere2SWASelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
    {
    public:
        Cohere2SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,  bool qkv_bias = false, bool o_bias = false)
            : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias) {}
    };

    template <int sliding_window_len> class Cohere2SWABlock : public LMBlock2<LayerNormNoBias, Cohere2SWASelfAttention<sliding_window_len>, SiLUMLP>
    {
    public:
        Cohere2SWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock2<LayerNormNoBias, Cohere2SWASelfAttention<sliding_window_len>, SiLUMLP>
                    (ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, false, false)
        {}
    };

    class Cohere2NoPosBlock : public LMBlock2<LayerNormNoBias, BaseAttention, SiLUMLP>
    {
    public:
        Cohere2NoPosBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock2<LayerNormNoBias, BaseAttention, SiLUMLP>
                    (ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, false, false)
        {}
    };

    const int SLIDING_WINDOW_LEN  =  4096;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef HeterogeneousModel<BaseConfig, Embedding, LayerNormNoBias> ModelClass;
        typedef Cohere2SWABlock<SLIDING_WINDOW_LEN> Cohere2SWABlock4k;

    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_COHERE_COMMAND_R7B)
            : BaseModelForConditionalGeneration(type, config, runtime_config),
              sliding_window_pattern(config.sliding_window_pattern), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            size_t num_tensors = 2;
            for (int i = 0; i < config.num_hidden_layers; i++)
                num_tensors += is_swa_layer(i) ? 12 : 11;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK(SLIDING_WINDOW_LEN == config.sliding_window) << "unsupported SWA param";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
            {
                if (is_swa_layer(layer_index))
                {
                    return new Cohere2SWABlock4k(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                }
                else
                {
                    return new Cohere2NoPosBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                }
            };

            transformer = new ModelClass(&w_ctx_, config, nullptr, create_layer);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_swa_layer(i))
                {
                    auto &attention = dynamic_cast<Cohere2SWABlock4k *>(get_typed_transformer<ModelClass>()->get_layer(i))->attention;
                    attention.freq_base = config.rope_theta;
                }
            }

            logit_scale = config.logit_scale;
            batch_input = false;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            #define LOAD_TENSORS()  \
                loader.read_tensor(layer_prefix + "input_layernorm.weight", layer->input_layernorm.weight); \
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight",   layer->mlp.down_proj.weight);   \
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",   layer->mlp.gate_proj.weight);   \
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",     layer->mlp.up_proj.weight);     \
                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", layer->attention.k_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", layer->attention.o_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", layer->attention.q_proj.weight);         \
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", layer->attention.v_proj.weight);

            loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                if (is_swa_layer(i))
                {
                    auto layer = dynamic_cast<Cohere2SWABlock4k *>(transformer->get_layer(i));
                    LOAD_TENSORS();
                }
                else
                {
                    auto layer = dynamic_cast<Cohere2NoPosBlock *>(transformer->get_layer(i));
                    LOAD_TENSORS();
                }
            }
            loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " != " << w_ctx_.get_mem_size() / ggml_tensor_overhead();

            #undef LOAD_TENSORS
        }

    private:
        bool is_swa_layer(int layer_index) const
        {
            return layer_index % sliding_window_pattern < (sliding_window_pattern - 1);
        }
        const int sliding_window_pattern;
    public:
        BaseConfig config;
    };
}