#include "llama.h"

namespace chatllm::stablelm
{
    struct Config : public llama::v2::Config
    {
        int num_key_value_heads;
        int rope_dim;
        float rope_theta;
        float rope_scaling;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, nullptr)
        {
        }

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);
            tp->EnableReturnSpecialToken(true);
            return size;
        }
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, LayerNorm, StableLMBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_STABLELM)
            : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 4 + config.num_hidden_layers * 14;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config, false,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_key_value_heads, config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
                attention.freq_base = config.rope_theta;
                attention.rope_dim  = config.rope_dim;
                attention.freq_scale = 1 / config.rope_scaling;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
                loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
                loader.read_tensor(layer_prefix + "input_layernorm.bias", transformer->layers[i].input_layernorm.bias);
                loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",transformer->layers[i].mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer->layers[i].mlp.up_proj.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);
                loader.read_tensor(layer_prefix + "post_attention_layernorm.bias", transformer->layers[i].post_attention_layernorm.bias);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight",transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };

    REGISTER_MODEL_LOADER(STABLELM,              stablelm, 1);
}