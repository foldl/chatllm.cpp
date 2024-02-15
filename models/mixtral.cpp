struct Config : public mistral::Config
{
    int num_experts_per_tok;
    int num_local_experts;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public mistral::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : mistral::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }
};

const int NUM_EXPERTS                   =  8;
const int EXPERTS_PER_TOK               =  2;

// make it easy to test with different number of experts.
#define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

class ConditionalGeneration : public BaseModelForConditionalGeneration<Model<Config, Embedding, RMSNorm,
    MixtralBlock<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;

    ConditionalGeneration(const Config &config);

    void load(ModelLoader &loader) override;

public:
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 244ull * 1024 * 1024;

    Config config;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_user(round_idx, user, ids);

    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    std::ostringstream oss_prompt;

    if (round_idx == 0)
        ids.push_back(tok->bos_token_id);

    oss_prompt << "[INST] " << user << " [/INST]";
    tok->encode(oss_prompt.str(), ids, false, false);
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_MIXTRAL, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * (11 + config.num_local_experts * 3);
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    CHATLLM_CHECK((NUM_EXPERTS == config.num_local_experts) && (EXPERTS_PER_TOK == config.num_experts_per_tok))
        << "unsupported MoE param";

    CHATLLM_CHECK((mistral::SLIDING_WINDOW_LEN == config.sliding_window) || (config.sliding_window <= 0))
        << "sliding_window (" << config.sliding_window << ") must equal to " << mistral::SLIDING_WINDOW_LEN;

    GRAPH_SIZE = 4096;

    transformer = Model<Config, Embedding, RMSNorm, MixtralBlock<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK, mistral::SLIDING_WINDOW_LEN>, int, int, int, int, int>(
                        &w_ctx_, config, false,
                        config.hidden_size, config.num_attention_heads,
                        config.intermediate_size, config.num_key_value_heads, config.max_length);

    batch_input = false;
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(i) + '.';

        for (int j = 0; j < config.num_local_experts; j++)
        {
            std::string prefix = layer_prefix + "block_sparse_moe.experts." + std::to_string(j) + '.';
            loader.read_tensor(prefix + "w1.weight", transformer.layers[i].mlp.experts[j].gate_proj.weight);
            loader.read_tensor(prefix + "w2.weight", transformer.layers[i].mlp.experts[j].down_proj.weight);
            loader.read_tensor(prefix + "w3.weight", transformer.layers[i].mlp.experts[j].up_proj.weight);
        }

        loader.read_tensor(layer_prefix + "block_sparse_moe.gate.weight",
                           transformer.layers[i].mlp.gate.weight);

        loader.read_tensor(layer_prefix + "input_layernorm.weight",
                           transformer.layers[i].input_layernorm.weight);

        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                           transformer.layers[i].post_attention_layernorm.weight);

        loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer.layers[i].attention.o_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
        loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
    }
    loader.read_tensor("model.norm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}