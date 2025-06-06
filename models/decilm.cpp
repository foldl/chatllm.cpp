#define MAX_LAYERS 100

struct LayerConfig
{
    int n_heads_in_group;
    int intermediate_size;
};

struct Config : public llama::v3_1::Config
{
    LayerConfig layer_configs[MAX_LAYERS];
};

typedef llama::v3_1::Tokenizer Tokenizer;

typedef LMNoAttnBlock<RMSNorm, SiLUMLP> DeciNoAttnBlock;

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef BaseModelForConditionalGeneration Base;
    typedef HeterogeneousModel ModelClass;

public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_DECILM)
    : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
        config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        size_t num_tensors = 3;
        for (int i = 0; i < config.num_hidden_layers; i++)
            num_tensors += config.layer_configs[i].n_heads_in_group > 0 ? 13 : 4;

        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        llama::v3_1::init_llama3_freq_factors(freq_factors_value, config.hidden_size / config.num_attention_heads,
            config.rope_theta,
            config.rope_scaling_factor,
            config.rope_scaling_low_freq_factor,
            config.rope_scaling_high_freq_factor,
            config.rope_scaling_original_max_position_embeddings);

        auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
            if (config.layer_configs[layer_index].n_heads_in_group > 0)
            {
                auto layer = new Llama31Block(ctx,
                    config.hidden_size, config.num_attention_heads,
                    config.layer_configs[layer_index].intermediate_size,
                    config.layer_configs[layer_index].n_heads_in_group, config.max_length);

                auto &attention = layer->attention;
                attention.freq_base = config.rope_theta;
                Backend::write_tensor_data(attention.freq_factors, freq_factors_value.data());

                return layer;
            }
            else
            {
                return new DeciNoAttnBlock(ctx, config.hidden_size, config.layer_configs[layer_index].intermediate_size);
            }
        };

        transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                    create_embedding<Embedding>(&w_ctx_, config),
                    create_final_norm<RMSNorm>(&w_ctx_, config),
                    create_lm_head(&w_ctx_, config, false), create_layer);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " vs "
            << w_ctx_.get_mem_size() / ggml_tensor_overhead();
    }

    void load(ModelLoader &loader) override
    {
        auto transformer = get_typed_transformer<ModelClass>();
        transformer->word_embeddings->load("model.embed_tokens.", &loader);

        #define LOAD_MLP()                                                                              \
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", layer->mlp.down_proj.weight);     \
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", layer->mlp.gate_proj.weight);     \
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight", layer->mlp.up_proj.weight);         \
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

            if (config.layer_configs[i].n_heads_in_group > 0)
            {
                auto layer = (Llama31Block *)(transformer->get_layer(i));

                loader.read_tensor(layer_prefix + "input_layernorm.weight", layer->input_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", layer->attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", layer->attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", layer->attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", layer->attention.v_proj.weight);

                LOAD_MLP();
            }
            else
            {
                auto layer = (DeciNoAttnBlock *)(transformer->get_layer(i));
                LOAD_MLP();
            }

        }
        transformer->final_layernorm->load("model.norm.", &loader);

        #undef LOAD_MLP

        if (transformer->lm_head)
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);
    }

protected:
    std::vector<float> freq_factors_value;
    Config config;
};