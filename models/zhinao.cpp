struct Config : public BaseConfig
{
    int num_key_value_heads;
    float rope_theta;
};

class Tokenizer : public qwen::v1::Tokenizer
{
public:
    Tokenizer(const BaseConfig &config)
        : qwen::v1::Tokenizer(config, &qwen::v1::_chat_encoder)
    {}

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        size_t r = qwen::v1::Tokenizer::load(buffer, n_vocab);

        int i = tp->GetPieceSize();

        pad_token_id = i++;
        i++; // unk_token_id = i++;
        i++; // mask
        i++; // self.eod_token,
        i++; // self.eop_token,
        int space2 = i++;
        int space3 = i++;
        int space4 = i++;
        int space8 = i++;
        im_start_token_id = i++;
        im_end_token_id = i++;

        tp->OverrideTokenDecoding(space2, "  ");
        tp->OverrideTokenDecoding(space3, "   ");
        tp->OverrideTokenDecoding(space4, "    ");
        tp->OverrideTokenDecoding(space8, "        ");

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        terminate_ids.insert(im_end_token_id);

        return r;
    }
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int> ModelClass;
public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

    void load(ModelLoader &loader) override;

public:
    Config config;
};

ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_ZHINAO, config, runtime_config), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 15;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = new ModelClass(&w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads,
                                config.max_length);

    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        auto &layer = get_typed_transformer<ModelClass>()->layers[i];
        layer.attention.freq_base = config.rope_theta;
        layer.attention.set_prec(ggml_prec::GGML_PREC_F32);
    }
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    auto transformer = get_typed_transformer<ModelClass>();

    loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);

        loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer->layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

        loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);
        loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
    }
    loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
    loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}

