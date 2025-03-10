struct Config : public BaseConfig
{
    int num_key_value_heads;
    float rope_theta;
};

typedef allenai::moe::Tokenizer Tokenizer;

typedef LMBlock4<RMSNorm,
                allenai::moe::OLSelfAttention,
                Identity,
                RMSNorm,
                SiLUMLP,
                Identity> InstellaBlock;

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef BaseModelForConditionalGeneration Base;
    typedef Model<Config, Embedding, RMSNorm, InstellaBlock, int, int, int, int, int> ModelClass;
public:
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_INSTELLA)
        : Base(type, config, runtime_config, 4096 * 2), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * (14);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(
                            &w_ctx_, config, false,
                            config.hidden_size, config.num_attention_heads,
                            config.intermediate_size, config.num_key_value_heads, config.max_length);

        auto transformer = Base::get_typed_transformer<ModelClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.rope_mode    = RoPEMode::Original;
            attention.freq_base    = config.rope_theta;
        }
    }

    void load(ModelLoader &loader) override
    {
        auto transformer = get_typed_transformer<ModelClass>();
        loader.read_tensor("model.embed_tokens.weight", transformer->word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);

            loader.read_tensor(layer_prefix + "pre_attention_layernorm.weight",
                            transformer->layers[i].pre_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "pre_feedforward_layernorm.weight",
                            transformer->layers[i].pre_mlp_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);

            loader.read_tensor(layer_prefix + "self_attn.q_norm.weight", transformer->layers[i].attention.q_norm.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_norm.weight", transformer->layers[i].attention.k_norm.weight);
        }
        loader.read_tensor("model.norm.weight", transformer->final_layernorm.weight);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

public:
    Config config;
};