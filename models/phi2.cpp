class QAHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
};

void QAHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    append_user(round_idx, user, ids);

    std::ostringstream oss_prompt;
    oss_prompt << ai << "\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void QAHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "Instruct: " << user << "\n"
               << "Output: ";

    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
{
    append_user(round_idx, user, ids);

    std::ostringstream oss_prompt;
    oss_prompt << ai << "\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "Alice: " << user << "\n"
            << "Bob: ";

    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

static QAHistoryEncoder _qa_encoder;
static ChatHistoryEncoder _chat_encoder;

class Phi2Tokenizer : public BaseTokenizer
{
public:
    Phi2Tokenizer(const BaseConfig &config)
        : BaseTokenizer::BaseTokenizer(config, &_chat_encoder, &_qa_encoder),
          qa_seq_max_len(0)
    {
    }

    Phi2Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder),
          qa_seq_max_len(0)
    {
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

public:
    std::vector<int> qa_terminate_seq1;
    std::vector<int> qa_terminate_seq2;
    std::vector<int> qa_terminate_seq3;
    std::vector<int> qa_terminate_seq4;
    int qa_seq_max_len;
    std::vector<int> chat_terminate_seq;
};

class Phi2ConditionalGeneration : public BaseModelForConditionalGeneration<
                                    Model<BaseConfig, Embedding, LayerNorm, Phi2Block, int, int, int, int, int>>
{
public:
    Phi2ConditionalGeneration() = default;
    Phi2ConditionalGeneration(const BaseConfig &config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, MEM_SIZE, SCRATCH_SIZE), config(config)
    {
        constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        const size_t num_tensors = 5 + config.num_hidden_layers * 17;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = Model<BaseConfig, Embedding, LayerNorm, Phi2Block, int, int, int, int, int>(&w_ctx_, config, true,
                            config.hidden_size, config.num_attention_heads,
                            config.intermediate_size, config.num_attention_heads, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            transformer.layers[i].set_id(i);
            transformer.layers[i].attention.set_id(i);
            transformer.layers[i].attention.set_prec(ggml_prec::GGML_PREC_F32);
        }
    }

public:
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 84ull * 1024 * 1024;

    BaseConfig config;

protected:
    bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override;

protected:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
};

bool Phi2ConditionalGeneration::is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output)
{
    if (output_ids.size() < 1) return false;

    Phi2Tokenizer *tokenizer = dynamic_cast<Phi2Tokenizer *>(this->tokenizer);
    int len = 0;

    switch (tokenizer->get_chat_format())
    {
    case ChatFormat::QA:
        if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq1))
        {
            pop_output = (int)tokenizer->qa_terminate_seq1.size();
            return true;
        }
        else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq2))
        {
            pop_output = (int)tokenizer->qa_terminate_seq2.size();
            return true;
        }
        else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq3))
        {
            pop_output = (int)tokenizer->qa_terminate_seq3.size();
            return true;
        }
        else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq4))
        {
            pop_output = (int)tokenizer->qa_terminate_seq4.size();
            return true;
        }
        len = tokenizer->qa_seq_max_len;
        break;
    case ChatFormat::CHAT:
        if (match_output_sequence(output_ids, tokenizer->chat_terminate_seq))
        {
            pop_output = (int)tokenizer->chat_terminate_seq.size();
            return true;
        }
        len = (int)tokenizer->chat_terminate_seq.size();
        break;
    default:
        ;
    }

    if (BaseModelForConditionalGeneration::is_output_terminated(output_ids, keep_idx, pop_output))
        return true;

    keep_idx = (int)(output_ids.size()) - len + 1;
    return false;
}

#define MAX_IT(x)  if (qa_seq_max_len < (int)x.size()) qa_seq_max_len = (int)x.size()

size_t Phi2Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor();
    size_t size = tp->Load(buffer, n_vocab);
    tp->Encode("\nInstruct:", &qa_terminate_seq1);
    tp->Encode("\nInstruction:", &qa_terminate_seq2);
    tp->Encode("\nUser:", &qa_terminate_seq3);
    tp->Encode("\nINPUT:", &qa_terminate_seq4);
    tp->Encode("\nAlice", &chat_terminate_seq);
    bos_token_id = eos_token_id = tp->PieceToId("<|endoftext|>");

    MAX_IT(qa_terminate_seq1);
    MAX_IT(qa_terminate_seq2);
    MAX_IT(qa_terminate_seq3);
    MAX_IT(qa_terminate_seq4);

    return size;
}

bool Phi2Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id);
}

namespace v1
{
    struct Config : public BaseConfig
    {
    };

    class Tokenizer : public Phi2Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : Phi2Tokenizer(config)
        {
        }
    };

    class ConditionalGeneration : public Phi2ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config);
        ConditionalGeneration(const Config &config, ModelType type);

        void load(ModelLoader &loader) override;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config)
        : ConditionalGeneration::ConditionalGeneration(config, ModelType::MODEL_TYPE_PHI2)
    {

    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, ModelType type)
        : Phi2ConditionalGeneration(config, type)
    {

    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.read_tensor("transformer.embd.wte.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "transformer.h." + std::to_string(i) + '.';

            loader.read_tensor(layer_prefix + "ln.bias",   transformer.layers[i].input_layernorm.bias);
            loader.read_tensor(layer_prefix + "ln.weight", transformer.layers[i].input_layernorm.weight);

            loader.read_tensor(layer_prefix + "mixer.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "mixer.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "mixer.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "mixer.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "mixer.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "mixer.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "mixer.out_proj.bias",   transformer.layers[i].attention.o_proj.bias);
            loader.read_tensor(layer_prefix + "mixer.out_proj.weight", transformer.layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "mlp.fc1.bias",   transformer.layers[i].mlp.fc0.bias);
            loader.read_tensor(layer_prefix + "mlp.fc1.weight", transformer.layers[i].mlp.fc0.weight);
            loader.read_tensor(layer_prefix + "mlp.fc2.bias",   transformer.layers[i].mlp.fc1.bias);
            loader.read_tensor(layer_prefix + "mlp.fc2.weight", transformer.layers[i].mlp.fc1.weight);
        }

        loader.read_tensor("lm_head.ln.bias", transformer.final_layernorm.bias);
        loader.read_tensor("lm_head.ln.weight", transformer.final_layernorm.weight);

        loader.read_tensor("lm_head.linear.bias", dynamic_cast<Linear *>(transformer.lm_head)->bias);
        loader.read_tensor("lm_head.linear.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
            << "corrupted model weights";
    }
}

namespace v2
{
    struct Config : public BaseConfig
    {
        int rope_dim;
        float rope_theta;
    };

    class Tokenizer : public Phi2Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : Phi2Tokenizer(config)
        {
        }
    };

    class ConditionalGeneration : public Phi2ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config);

        void load(ModelLoader &loader) override;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config)
        : Phi2ConditionalGeneration(config, ModelType::MODEL_TYPE_PHI2)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            transformer.layers[i].attention.rope_dim = config.rope_dim;
            transformer.layers[i].attention.freq_base = config.rope_theta;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.read_tensor("model.embed_tokens.weight", transformer.word_embeddings.weight);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';

            loader.read_tensor(layer_prefix + "input_layernorm.bias",   transformer.layers[i].input_layernorm.bias);
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer.layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer.layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer.layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer.layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.dense.bias",   transformer.layers[i].attention.o_proj.bias);
            loader.read_tensor(layer_prefix + "self_attn.dense.weight", transformer.layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "mlp.fc1.bias",   transformer.layers[i].mlp.fc0.bias);
            loader.read_tensor(layer_prefix + "mlp.fc1.weight", transformer.layers[i].mlp.fc0.weight);
            loader.read_tensor(layer_prefix + "mlp.fc2.bias",   transformer.layers[i].mlp.fc1.bias);
            loader.read_tensor(layer_prefix + "mlp.fc2.weight", transformer.layers[i].mlp.fc1.weight);
        }

        loader.read_tensor("model.final_layernorm.bias", transformer.final_layernorm.bias);
        loader.read_tensor("model.final_layernorm.weight", transformer.final_layernorm.weight);

        loader.read_tensor("lm_head.bias", dynamic_cast<Linear *>(transformer.lm_head)->bias);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer.lm_head)->weight);

        CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
            << "corrupted model weights";
    }
}