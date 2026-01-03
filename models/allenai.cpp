#include "allenai.h"
#include "../src/chat_encoders.h"

namespace chatllm::allenai::moe
{
    class ChatHistoryEncoder : public HistoryEncoderBracketRole
    {
    protected:
        void append_ai_ending(int round_idx, std::vector<int> &ids) const override
        {
            ids.push_back(tokenizer->eos_token_id);
            tokenizer->encode("\n", ids);
        }
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
            BaseHistoryEncoder *qa_encoder,
            BaseHistoryEncoder *completion_encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();

        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        bos_token_id = eos_token_id;

        return size;
    }
}

namespace chatllm::allenai::dense
{
    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
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

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();
        transformer->word_embeddings->load("model.embed_tokens.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);

            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                            transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "post_feedforward_layernorm.weight",
                            transformer->layers[i].post_mlp_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);

            loader.read_tensor(layer_prefix + "self_attn.q_norm.weight", transformer->layers[i].attention.q_norm.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_norm.weight", transformer->layers[i].attention.k_norm.weight);
        }
        transformer->final_layernorm->load("model.norm.", &loader);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(OLMoE,                 allenai::moe, 1);
    REGISTER_MODEL_LOADER(OLMo2,                 allenai::dense, 1);
}