struct Config : public BaseConfig
{
};

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

static QAHistoryEncoder _qa_encoder;
static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : BaseTokenizer::BaseTokenizer(config, &_chat_encoder, &_qa_encoder)
    {
    }

    size_t load(const char *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

public:
    std::vector<int> qa_terminate_seq1;
    std::vector<int> qa_terminate_seq2;
    std::vector<int> chat_terminate_seq;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration<
                                  Model<Config, LayerNorm, Phi2Block, int, int, int, int, int>>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config);

    void load(ModelLoader &loader) override;

public:
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 84ull * 1024 * 1024;

    Config config;

protected:
    bool is_output_terminated(int next_token_id, const std::vector<int> &output_ids, int &pop_output) override;

private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
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

size_t Tokenizer::load(const char *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor();
    size_t size = tp->Load(buffer, n_vocab);
    tp->Encode("\nInstruct", &qa_terminate_seq1);
    tp->Encode("\nInstruction:", &qa_terminate_seq2);
    tp->Encode("\nAlice", &chat_terminate_seq);
    bos_token_id = eos_token_id = tp->PieceToId("<|endoftext|>");
    return size;
}

bool Tokenizer::is_special_id(int id) const
{
    return (id == pad_token_id);
}

ConditionalGeneration::ConditionalGeneration(const Config &config)
    : BaseModelForConditionalGeneration(ModelType::MODEL_TYPE_PHI2, config, MEM_SIZE, SCRATCH_SIZE), config(config)
{
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 5 + config.num_hidden_layers * 17;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = Model<Config, LayerNorm, Phi2Block, int, int, int, int, int>(&w_ctx_, config, false, true,
                        config.hidden_size, config.num_attention_heads,
                        config.intermediate_size, config.num_attention_heads, config.max_length);

    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        transformer.layers[i].set_id(i);
        transformer.layers[i].attention.set_id(i);
        transformer.layers[i].attention.set_prec(ggml_prec::GGML_PREC_F32);
    }
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

    loader.read_tensor("lm_head.linear.bias", transformer.lm_head.bias);
    loader.read_tensor("lm_head.linear.weight", transformer.lm_head.weight);

    CHATLLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";
}

bool ConditionalGeneration::is_output_terminated(int next_token_id, const std::vector<int> &output_ids, int &pop_output)
{
    if (output_ids.size() < 1) return false;

    Tokenizer *tokenizer = dynamic_cast<Tokenizer *>(this->tokenizer);

    switch (tokenizer->get_chat_format())
    {
    case ChatFormat::QA:
        if (match_output_sequence(next_token_id, output_ids, tokenizer->qa_terminate_seq1))
        {
            pop_output = (int)(tokenizer->qa_terminate_seq1.size() - 1);
            return true;
        }
        else if (match_output_sequence(next_token_id, output_ids, tokenizer->qa_terminate_seq2))
        {
            pop_output = (int)(tokenizer->qa_terminate_seq2.size() - 1);
            return true;
        }
        break;
    case ChatFormat::CHAT:
        if (match_output_sequence(next_token_id, output_ids, tokenizer->chat_terminate_seq))
        {
            pop_output = (int)(tokenizer->chat_terminate_seq.size() - 1);
            return true;
        }
        break;
    default:
        break;
    }

    return BaseModelForConditionalGeneration::is_output_terminated(next_token_id, output_ids, pop_output);
}