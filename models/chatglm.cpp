namespace v1
{
struct Config : public BaseConfig
{
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config) : BaseTokenizer::BaseTokenizer(config, &_chat_encoder) {}

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    void encode(const std::string &text, std::vector<int> &ids) const override;

protected:
    std::string preprocess(const std::string &text) const override;
    std::string postprocess(const std::string &text) const override;

public:
    int mask_token_id;
    int gmask_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<Config, Embedding, LayerNorm, GLMBlock, int, int, int, int> TransformerClass;
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

    void load(ModelLoader &loader) override;

public:
    Config config;
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);

    bos_token_id = tp->PieceToId("<sop>");
    eos_token_id = tp->PieceToId("<eop>");
    mask_token_id = tp->PieceToId("[MASK]");
    gmask_token_id = tp->PieceToId("[gMASK]");
    pad_token_id = tp->PieceToId("<pad>");
    return size;
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
{
    ids.insert(ids.end(), {gmask_token_id, bos_token_id});
    BaseTokenizer::encode(text, ids);
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "答：" << ai << "\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "[Round " << round_idx + 1 << "]\n问：" << user << "\n";
    tokenizer->encode(oss_prompt.str(), ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "答：";
    tokenizer->encode(oss_prompt.str(), ids);
}

std::string Tokenizer::preprocess(const std::string &text) const
{
    std::string output;

    // newline token
    {
        static const std::regex newline_regex("\n");
        output = std::regex_replace(text, newline_regex, "<n>");
    }
    // tab token
    {
        static const std::regex tab_regex("\t");
        output = std::regex_replace(output, tab_regex, "<|tab|>");
    }
    // blank tokens
    {
        static const std::regex pattern(R"([ ]{2,80})");
        output = regex_replace(output, pattern, [](const std::smatch &sm)
            {
                std::ostringstream oss;
                oss << "<|blank_" << sm.str().size() << "|>";
                return oss.str();
            });
    }

    return output;
}

std::string Tokenizer::postprocess(const std::string &text) const
{
    std::string output;

    // newline token
    {
        static const std::regex pattern(R"(<n>)");
        output = std::regex_replace(text, pattern, "\n");
    }
    // tab token
    {
        static const std::regex pattern(R"(<\|tab\|>)");
        output = std::regex_replace(output, pattern, "\t");
    }
    // blank tokens
    {
        static const std::regex pattern(R"(<\|blank_(\d+)\|>)");
        output = regex_replace(output, pattern,
                                [](const std::smatch &sm)
                                { return std::string(std::stoi(sm[1].str()), ' '); });
    }

    // replace punctuations
    // reference: https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
    {
        static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        static const std::vector<std::pair<std::wregex, std::wstring>> punct_map{
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]),)")), converter.from_bytes("$1，")},
            {std::wregex(converter.from_bytes(R"(,([\u4e00-\u9fff]))")), converter.from_bytes("，$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])!)")), converter.from_bytes("$1！")},
            {std::wregex(converter.from_bytes(R"(!([\u4e00-\u9fff]))")), converter.from_bytes("！$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]):)")), converter.from_bytes("$1：")},
            {std::wregex(converter.from_bytes(R"(:([\u4e00-\u9fff]))")), converter.from_bytes("：$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]);)")), converter.from_bytes("$1；")},
            {std::wregex(converter.from_bytes(R"(;([\u4e00-\u9fff]))")), converter.from_bytes("；$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])\?)")), converter.from_bytes("$1？")},
            {std::wregex(converter.from_bytes(R"(\?([\u4e00-\u9fff]))")), converter.from_bytes("？$1")},
        };
        std::wstring w_output = converter.from_bytes(output);
        for (const auto &punct_pair : punct_map)
        {
            w_output = std::regex_replace(w_output, punct_pair.first, punct_pair.second);
        }
        output = converter.to_bytes(w_output);
    }

    return output;
}

ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_CHATGLM, config, runtime_config), config(config)
{
    const size_t tensor_ovhd = ggml_tensor_overhead();
    const size_t num_tensors = 3 + config.num_hidden_layers * 15;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    CHATLLM_CHECK(false) << "regression: ChatGLM is not available. Bye.";

    transformer = new TransformerClass(&w_ctx_, config, nullptr,
                            config.hidden_size, config.num_attention_heads, config.num_hidden_layers,
                            config.max_length);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    TransformerClass *transformer = dynamic_cast<TransformerClass *>(this->transformer);

    loader.read_tensor("transformer.word_embeddings.weight", transformer->word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "transformer.layers." + std::to_string(layer_ids[i]) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "input_layernorm.bias", transformer->layers[i].input_layernorm.bias);
        loader.read_tensor(layer_prefix + "attention.query_key_value.weight",
                            transformer->layers[i].attention.query_key_value.weight);
        loader.read_tensor(layer_prefix + "attention.query_key_value.bias",
                            transformer->layers[i].attention.query_key_value.bias);
        loader.read_tensor(layer_prefix + "attention.dense.weight", transformer->layers[i].attention.dense.weight);
        loader.read_tensor(layer_prefix + "attention.dense.bias", transformer->layers[i].attention.dense.bias);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                            transformer->layers[i].post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",
                            transformer->layers[i].post_attention_layernorm.bias);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer->layers[i].mlp.fc0.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.bias", transformer->layers[i].mlp.fc0.bias);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer->layers[i].mlp.fc1.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.bias", transformer->layers[i].mlp.fc1.bias);
    }
    loader.read_tensor("transformer.final_layernorm.weight", transformer->final_layernorm.weight);
    loader.read_tensor("transformer.final_layernorm.bias", transformer->final_layernorm.bias);

    CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
        << "corrupted model weights";
}
}

namespace v2
{
struct Config : public BaseConfig
{
    int num_kv_heads;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config) : BaseTokenizer::BaseTokenizer(config, &_chat_encoder) {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder) : BaseTokenizer::BaseTokenizer(config, encoder) {}

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    void encode(const std::string &text, std::vector<int> &ids) const override;

    bool is_special_id(int id) const override;

public:
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    typedef Model<Config, Embedding, RMSNorm, GLM2Block, int, int, int, int, int> TransformerClass;
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_CHATGLM2);

    void load(ModelLoader &loader) override;

public:
    Config config;
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    tp = new tokenizer::BPEProcessor1();
    size_t size = tp->Load(buffer, n_vocab);

    int special_id = tp->GetPieceSize();
    mask_token_id = special_id++;
    gmask_token_id = special_id++;
    smask_token_id = special_id++;
    sop_token_id = special_id++;
    eop_token_id = special_id++;

    return size;
}

void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
{
    ids.insert(ids.end(), {gmask_token_id, sop_token_id}); // special prefix
    BaseTokenizer::encode(text, ids);
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "答：" << ai << "\n\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "[Round " << round_idx + 1 << "]\n\n问：" << user << "\n\n";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    std::ostringstream oss_prompt;

    oss_prompt << "答：";
    auto text = oss_prompt.str();
    tokenizer->encode(text, ids);
}

bool Tokenizer::is_special_id(int id) const
{
    return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
            id == eop_token_id;
}

ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
    : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
{
    const size_t tensor_ovhd = ggml_tensor_overhead();
    const size_t num_tensors = 3 + config.num_hidden_layers * 10;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    w_ctx_.dtype = config.dtype;

    transformer = new TransformerClass(&w_ctx_, config, false,
                            config.hidden_size, config.num_attention_heads, config.num_kv_heads,
                            config.intermediate_size, config.max_length);
}

void ConditionalGeneration::load(ModelLoader &loader)
{
    TransformerClass *transformer = dynamic_cast<TransformerClass *>(this->transformer);
    loader.read_tensor("transformer.embedding.word_embeddings.weight", transformer->word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++)
    {
        std::string layer_prefix = "transformer.encoder.layers." + std::to_string(layer_ids[i]) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "self_attention.query_key_value.weight",
                            transformer->layers[i].attention.query_key_value.weight);
        loader.read_tensor(layer_prefix + "self_attention.query_key_value.bias",
                            transformer->layers[i].attention.query_key_value.bias);
        loader.read_tensor(layer_prefix + "self_attention.dense.weight", transformer->layers[i].attention.dense.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                            transformer->layers[i].post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer->layers[i].mlp.dense_h_to_4h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer->layers[i].mlp.dense_4h_to_h.weight);
    }
    loader.read_tensor("transformer.encoder.final_layernorm.weight", transformer->final_layernorm.weight);
    loader.read_tensor("transformer.output_layer.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

    CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
        << "corrupted model weights";
}
}

namespace v3
{
struct Config : public v2::Config
{
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

class Tokenizer : public v2::Tokenizer
{
public:
    Tokenizer(const Config &config) : Tokenizer(config, &_chat_encoder) {}

    Tokenizer(const Config &config, BaseHistoryEncoder *_chat_encoder) : v2::Tokenizer::Tokenizer(config, _chat_encoder) {}

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    bool is_special_id(int id) const override;

public:
    int system_token_id;
    int user_token_id;
    int assistant_token_id;
    int observation_token_id;
};

class GLMInterceptor : public ChunkInterceptor
{
public:
    GLMInterceptor() : ChunkInterceptor(), is_first(true), find_meta(false), found_tool_call(false)
    {}

    void put_chunk(bool _first, const std::string &chunk) override
    {
        if (is_first && found_tool_call)
        {
            found_tool_call = false;
            is_first = false;
            goto handle_data;
        }

        if (is_first)
        {
            std::string s = trim(chunk);
            if (s.size() < 1) return;

            _first = true;

            if (s[0] != '\n')
            {
                find_meta = true;
                found_tool_call = true;
            }

            is_first = false;
        }

    handle_data:
        if (find_meta)
            oss << chunk;
        else
            next->put_chunk(_first, chunk);
    }

    void end() override
    {
        if (find_meta)
            streamer->putln(oss.str(), BaseStreamer::TextType::TOOL_CALLING);

        oss.str("");
        find_meta = false;
        is_first = true;

        ChunkInterceptor::end();
    }
protected:
    std::ostringstream oss;
    bool is_first;
    bool find_meta;
    bool found_tool_call;
};

static GLMInterceptor interceptor;

class ConditionalGeneration : public v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_CHATGLM3)
    {
    }

    ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
};

size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
{
    size_t size = v2::Tokenizer::load(buffer, n_vocab);

    int token_id = eop_token_id + 1;
    system_token_id = token_id++;
    user_token_id   = token_id++;
    assistant_token_id  = token_id++;
    observation_token_id= token_id++;

    terminate_ids.insert(eos_token_id);
    terminate_ids.insert(user_token_id);
    terminate_ids.insert(observation_token_id);
    return size;
}

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    if (tok->get_system_prompt().size() > 0)
    {
        ids.push_back(tok->system_token_id);
        tok->encode(tok->get_system_prompt(), ids);
    }
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->user_token_id);
    tok->encode(user, ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->assistant_token_id);
}

bool Tokenizer::is_special_id(int id) const
{
    return v2::Tokenizer::is_special_id(id)
        || (id == system_token_id)
        || (id == user_token_id)
        || (id == assistant_token_id)
        || (id == observation_token_id);
}
}

namespace v4
{
struct Config : public v3::Config
{
    float rope_ratio;
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public v3::Tokenizer
{
public:
    Tokenizer(const Config &config) : v3::Tokenizer(config, &_chat_encoder)
    {}

    void encode(const std::string &text, std::vector<int> &ids) const override
    {
        BaseTokenizer::encode(text, ids);
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        int special_id = eos_token_id + 1;
        mask_token_id = special_id++;
        gmask_token_id = special_id++;
        smask_token_id = special_id++;
        sop_token_id = special_id++;
        eop_token_id = special_id++;
        system_token_id = special_id++;
        user_token_id   = special_id++;
        assistant_token_id  = special_id++;
        observation_token_id= special_id++;

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        terminate_ids.insert(eos_token_id);
        terminate_ids.insert(user_token_id);
        terminate_ids.insert(observation_token_id);

        return size;
    }
public:
    int nl_token_id;
};

class GLMInterceptor : public ChunkInterceptor
{
public:
    GLMInterceptor() : ChunkInterceptor(), find_meta(false)
    {}

    void put_chunk(bool first, const std::string &chunk) override
    {
        if (first && (chunk[0] != '\n'))
        {
            find_meta = true;
        }

        if (find_meta)
        {
            str = str + chunk;

            if (confirmed) return;

            size_t pos = str.find('\n');
            std::string first = str;
            if (pos != std::string::npos)
            {
                first = str.substr(0, pos);
                if (first.find(' ') == std::string::npos)
                    confirmed = true;
            }

            if (!confirmed && (first.find(' ') != std::string::npos))
            {
                next->put_chunk(true, str);
                str = "";
                find_meta = false;
            }
        }
        else
            next->put_chunk(first, chunk);
    }

    void end() override
    {
        std::string s = str;
        str = "";
        find_meta = false;
        confirmed = false;

        ChunkInterceptor::end();

        if (s.size() > 0)
        {
            streamer->putln(s, BaseStreamer::TextType::TOOL_CALLING);
        }
    }
protected:
    std::string str;
    bool confirmed;
    bool find_meta;
};

static GLMInterceptor interceptor;

class ConditionalGeneration : public v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GLM4)
        : v2::ConditionalGeneration(config, runtime_config, type)
    {
        auto transformer = get_typed_transformer<TransformerClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base *= config.rope_ratio;
        }
    }

    ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
};

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.insert(ids.end(), {tok->gmask_token_id, tok->sop_token_id});

    if (tok->get_system_prompt().size() > 0)
    {
        ids.insert(ids.end(), {tok->system_token_id, tok->nl_token_id});
        tok->encode(tok->get_system_prompt(), ids);
    }
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.insert(ids.end(), {tok->user_token_id, tok->nl_token_id});
    tok->encode(user, ids);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

    ids.push_back(tok->assistant_token_id);
}
}