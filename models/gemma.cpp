namespace v1
{
struct Config : public BaseConfig
{
    int num_key_value_heads;
    int head_dim;
    float rope_theta;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const BaseConfig &config)
        : BaseTokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        int id = tp->PieceToId("<pad>");
        if (id >= 0) pad_token_id = id;
        start_of_turn_token_id = tp->PieceToId("<start_of_turn>");
        end_of_turn_token_id   = tp->PieceToId("<end_of_turn>");
        nl_token_id = tp->PieceToId("\n");

        const int first_added_id = tp->PieceToId("<unused0>");
        for (int i = tp->PieceToId("</code>"); i >= first_added_id; i--)
            tp->AddAddedToken(tp->IdToPiece(i), i);

        terminate_ids.insert(end_of_turn_token_id);
        terminate_ids.insert(tp->PieceToId("<|fim_prefix|>"));
        terminate_ids.insert(tp->PieceToId("<|fim_suffix|>"));
        terminate_ids.insert(tp->PieceToId("<|fim_middle|>"));
        terminate_ids.insert(tp->PieceToId("<|file_separator|>"));

        return size;
    }

public:
    void encode(const std::string &text, std::vector<int> &ids, bool add_start, bool add_end)
    {
        if (add_start)
            ids.push_back(start_of_turn_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_end)
        {
            ids.push_back(end_of_turn_token_id);
            ids.push_back(nl_token_id);
        }
    }

public:
    int start_of_turn_token_id;
    int end_of_turn_token_id;
    int nl_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<BaseConfig, Embedding, RMSNorm, GemmaBlock, int, int, int, int, int, int> ModelClass;

public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 2 + config.num_hidden_layers * 12;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config,
                nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
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
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

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

    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->bos_token_id);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    oss_prompt << "user" << "\n" << user;
    tok->encode(oss_prompt.str(), ids, true, true);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    oss_prompt << "model" << "\n";
    tok->encode(oss_prompt.str(), ids, true, false);
}
}

namespace v2
{
struct Config : public BaseConfig
{
    int num_key_value_heads;
    int head_dim;
    int query_pre_attn_scalar;
    int sliding_window;

    float rope_theta;
    float final_logit_soft_capping;
    float attn_logit_soft_capping;
};

typedef v1::Tokenizer Tokenizer;

template <int sliding_window_len> class Gemma2SWASelfAttention : public RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>
{
public:
    Gemma2SWASelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
        : RoPESelfAttention<SlidingWindowAttentionImpl<sliding_window_len>>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
};

template <int sliding_window_len> class Gemma2SWABlock : public LMBlock4<RMSNorm, Gemma2SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
{
public:
    Gemma2SWABlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
        : LMBlock4<RMSNorm, Gemma2SWASelfAttention<sliding_window_len>, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
                  (ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
    {}
};

const int SLIDING_WINDOW_LEN = 4096;

typedef Gemma2SWABlock<SLIDING_WINDOW_LEN> Gemma2SWABlock4k;

class Gemma2FullBlock : public LMBlock4<RMSNorm, GemmaSelfAttention, RMSNorm, RMSNorm, GELUMLP, RMSNorm>
{
public:
    Gemma2FullBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
        : LMBlock4(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
    {}
};

class TanhScaling: public Block
{
public:
    TanhScaling(float scale_pre, float scale_post) : scale_pre(scale_pre), scale_post(scale_post) {}

    ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
    {
        input = ggml::scale_inplace(ctx, input, scale_pre);
        input = ggml::inplace_act(ctx, ActFunc::Tanh, input);
        input = ggml::scale_inplace(ctx, input, scale_post);
        return input;
    }
public:
    float scale_pre;
    float scale_post;
};

template <class Layer> static void setup_layer(Block *block, const Config &config, Block *attn_scores_pp)
{
    auto layer = dynamic_cast<Layer *>(block);
    auto &attention = layer->attention;
    attention.freq_base = config.rope_theta;
    attention.attn_scaling_factor = powf((float)config.query_pre_attn_scalar, -0.5);
    attention.attn_scores_pp = attn_scores_pp;
}

template <class Layer> static void load_layer(ModelLoader &loader, const std::string &layer_prefix, Block *block)
{
    auto layer = dynamic_cast<Layer *>(block);
    loader.read_tensor(layer_prefix + "input_layernorm.weight",             layer->pre_attention_layernorm.weight);
    loader.read_tensor(layer_prefix + "mlp.down_proj.weight",               layer->mlp.down_proj.weight);
    loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",               layer->mlp.gate_proj.weight);
    loader.read_tensor(layer_prefix + "mlp.up_proj.weight",                 layer->mlp.up_proj.weight);
    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",    layer->post_attention_layernorm.weight);
    loader.read_tensor(layer_prefix + "pre_feedforward_layernorm.weight",   layer->pre_mlp_layernorm.weight);
    loader.read_tensor(layer_prefix + "post_feedforward_layernorm.weight",  layer->post_mlp_layernorm.weight);

    loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", layer->attention.k_proj.weight);
    loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", layer->attention.o_proj.weight);
    loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", layer->attention.q_proj.weight);
    loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", layer->attention.v_proj.weight);
}

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef HeterogeneousModel<BaseConfig, Embedding, RMSNorm> ModelClass;
public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GEMMA2)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config),
          logits_pp(1.0f / config.final_logit_soft_capping / sqrtf((float)config.hidden_size), config.final_logit_soft_capping),
          attn_scores_pp(1.0f / config.attn_logit_soft_capping, config.attn_logit_soft_capping)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 2 + (config.num_hidden_layers - config.num_hidden_layers / 2) * 14 + config.num_hidden_layers / 2 * 15;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        CHATLLM_CHECK(SLIDING_WINDOW_LEN == config.sliding_window) << "unsupported SWA param";

        auto create_layer = [&](InitContext *ctx, int layer_index) -> Block *
        {
            if (is_sliding(layer_index))
            {
                return new Gemma2SWABlock4k(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
            }
            else
            {
                return new Gemma2FullBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                            config.num_key_value_heads, config.head_dim, config.max_length);
            }
        };

        transformer = new ModelClass(&w_ctx_, config, nullptr, create_layer);

        get_typed_transformer<ModelClass>()->logits_pp = &logits_pp;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (is_sliding(i))
            {
                setup_layer<Gemma2SWABlock4k>(get_typed_transformer<ModelClass>()->get_layer(i), config, &attn_scores_pp);
            }
            else
            {
                setup_layer<Gemma2FullBlock>(get_typed_transformer<ModelClass>()->get_layer(i), config, &attn_scores_pp);
            }
        }

        batch_input = false;
    }

    void load(ModelLoader &loader) override
    {
        auto transformer = get_typed_transformer<ModelClass>();

        loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
            if (is_sliding(i))
            {
                load_layer<Gemma2SWABlock4k>(loader, layer_prefix, transformer->get_layer(i));
            }
            else
            {
                load_layer<Gemma2FullBlock>(loader, layer_prefix, transformer->get_layer(i));
            }
        }
        loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

public:

    bool is_sliding(int layer_id)
    {
        return layer_id % 2;
    }

public:
    BaseConfig config;
    TanhScaling logits_pp;
    TanhScaling attn_scores_pp;
};
}