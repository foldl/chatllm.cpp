#include "chatglm.h"
#include <regex>
#include <codecvt>
#include "qwen.h"
#include "llama.h"
#include "../src/vision_process.h"
#include "../src/chat_encoders.h"
#include "../src/audio_process.h"

namespace chatllm::glm::v1
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config) : BaseTokenizer::BaseTokenizer(config, &_chat_encoder)
    {}

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

    static std::string regex_replace(const std::string &input, const std::regex &regex,
                                     std::function<std::string(const std::smatch &)> format)
    {
        std::ostringstream oss;
        size_t last_index = 0;
        for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++)
        {
            oss << it->prefix() << format(*it);
            last_index = it->position() + it->length();
        }
        oss << input.substr(last_index);
        return oss.str();
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

        transformer->word_embeddings->load("transformer.word_embeddings.", &loader);
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
        transformer->final_layernorm->load("transformer.final_layernorm.", &loader);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace chatllm::glm::v2
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : BaseTokenizer::BaseTokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {}

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
        transformer->word_embeddings->load("transformer.embedding.word_embeddings.", &loader);
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
        transformer->final_layernorm->load("transformer.encoder.final_layernorm.", &loader);
        loader.read_tensor("transformer.output_layer.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace chatllm::glm::v3
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *_chat_encoder)
        : v2::Tokenizer::Tokenizer(config, _chat_encoder)
    {}

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

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_CHATGLM3)
    {
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &interceptor;
    }

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

namespace chatllm::glm::v4
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config) : v3::Tokenizer(config, &_chat_encoder)
    {}

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        BaseTokenizer::encode(text, ids);
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        return do_load(buffer, n_vocab,
            {
                //[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
                "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
    }

    size_t Tokenizer::do_load(tokenizer::DataReader *buffer, int n_vocab, std::vector<std::string> regex_exprs)
    {
        tp = new tokenizer::BPEProcessor2(regex_exprs);
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

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v2::ConditionalGeneration(config, runtime_config, type)
    {
        auto transformer = get_typed_transformer<TransformerClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base *= config.rope_ratio;
        }
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &interceptor;
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
        ids.push_back(tok->nl_token_id);
    }
}

namespace chatllm::glm::glm4_0414
{
    Tokenizer::Tokenizer(const Config &config)
        : v4::Tokenizer(config)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = do_load(buffer, n_vocab,
            {
                // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );

        think_open_token_id = tp->PieceToId("<think>");
        think_close_token_id = tp->PieceToId("</think>");
        if (think_open_token_id >= 0)
        {
            tp->OverrideTokenDecoding(think_open_token_id, "<think>");
            tp->OverrideTokenDecoding(think_close_token_id, "</think>");
        }
        return r;
    }

    class GLM4SelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        GLM4SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention<BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
        }
    };

    class GLM4Block : public LMBlock4<RMSNorm, GLM4SelfAttention, RMSNorm, RMSNorm, SiLUMLP, RMSNorm>
    {
    public:
        GLM4Block(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
                  int max_length, bool qkv_bias, bool o_bias)
            : LMBlock4(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, qkv_bias, o_bias)
        {
            mlp.set_prec(ggml::prec::GGML_PREC_F32);
        }
    };

    typedef Model<Config, Embedding, RMSNorm, GLM4Block, int, int, int, int, int, bool, bool> ModelClass;

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * (14 + (config.use_attention_bias != 0 ? 3 : 0));
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(
                            &w_ctx_, config, false,
                            config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                            config.intermediate_size, config.max_length, config.use_attention_bias != 0, false);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base    = config.rope_theta;
            layer.attention.rope_dim     = config.rope_dim;
        }

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights: " << w_ctx_.get_used_mem() / ggml_tensor_overhead() << " != " << w_ctx_.get_mem_size() / ggml_tensor_overhead();
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        // pay attention to the order
        loader.add_tensor_name_translations({
            {".pre_attention_layernorm.",           ".input_layernorm."},
            {".post_attention_layernorm.",          ".post_self_attn_layernorm."},
            {".pre_mlp_layernorm.",                 ".post_attention_layernorm."},
        });

        BaseModelForConditionalGeneration::load(loader);
    }
}

namespace chatllm::glm::vit
{
    struct Config
    {
        ggml::type dtype;
        int max_length;
        int image_ref_size;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int out_hidden_size;
        int spatial_merge_size;
        int temporal_patch_size;

        float image_mean[3];
        float image_std[3];
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, const Config &config):
            merge_size(config.spatial_merge_size),
            max_patches(config.max_length), ref_w(config.image_ref_size / config.patch_size), ref_h(config.image_ref_size / config.patch_size),
            proj0(ctx, 3, config.hidden_size, config.patch_size, config.patch_size, 0, 1, 1, false),
            proj1(ctx, 3, config.hidden_size, config.patch_size, config.patch_size, 0, 1, 1, false),
            proj_bias(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size)),
            post_conv_layernorm(ctx, config.hidden_size),
            position_embedding(ctx, ggml::type::GGML_TYPE_F32, ref_w * ref_h, config.hidden_size)
        {
            CHATLLM_CHECK(config.temporal_patch_size == 2);
            CHATLLM_CHECK(merge_size == 2);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input0, ggml::tensor *input1, int grid_h, int grid_w)
        {
            ggml::tensor *x0 = proj0.forward(ctx, input0);
            ggml::tensor *x1 = proj1.forward(ctx, input1);
            ggml::tensor *x  = ggml::add(ctx, x0, x1);
            auto bias_view = ggml::reshape_3d(ctx, proj_bias, 1, 1, ggml::get_dim(proj_bias, 0));
            x = ggml::add(ctx, x, bias_view);
            x = ggml::reshape_2d(ctx, x, ggml::get_dim(x, 2), grid_h * grid_w);

            x = post_conv_layernorm.forward(ctx, x);

            {
                //input: ggml[dim, w, h]
                CHATLLM_CHECK(ggml::get_dim(x, 3) == 1);

                ggml::tensor * interp = nullptr;
                auto pos_emb = position_embedding.weight;

                const int hidden_size = ggml::get_dim(pos_emb, 0);

                if ((ref_w == grid_w) && (ref_h == grid_h))
                {
                    interp = ggml::reshape_3d(ctx, pos_emb, hidden_size, ref_w, ref_h);
                }
                else
                {
                    auto permuted = ggml::reshape_3d(ctx, pos_emb, hidden_size, ref_w, ref_h);
                    permuted = ggml::permute(ctx, permuted, 2, 0, 1);
                    interp = ggml::interpolate(ctx, permuted, ggml::InterpolateMode::Bicubic,
                        grid_w, grid_h, hidden_size, 1);

                    interp = ggml::permute(ctx, interp, 1, 2, 0); // -> [dim, w, h]
                    interp = ggml::cont(ctx, interp);
                }

                // rearrange for `merge_size` == 2
                interp = ggml::reshape(ctx, interp, hidden_size * merge_size, grid_w / merge_size, merge_size, grid_h / merge_size);
                interp = ggml::permute(ctx, interp, 0, 2, 1, 3); // -> [dim * 2, 2, w / 2, h / 2]
                interp = ggml::cont(ctx, interp);
                interp = ggml::reshape_2d(ctx, interp, hidden_size, grid_h * grid_w);

                x = ggml::add(ctx, x, interp);
            }

            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += proj0.get_param_num(effective_only);
            r += proj1.get_param_num(effective_only);
            r += post_conv_layernorm.get_param_num(effective_only);
            r += ggml::nelements(proj_bias);
            r += position_embedding.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            proj0.load(path + "patch_embed.proj.0.", loader);
            proj1.load(path + "patch_embed.proj.1.", loader);
            loader->read_tensor(path + "patch_embed.proj.bias", proj_bias);

            post_conv_layernorm.load(path + "post_conv_layernorm.", loader);
            position_embedding.load(path + "embeddings.position_embedding.", loader);
        }
    public:
        const int merge_size;
        const int max_patches;
        const int ref_w;
        const int ref_h;
        Conv2D                  proj0;
        Conv2D                  proj1;
        ggml::tensor           *proj_bias;
        RMSNorm                 post_conv_layernorm;
        Embedding               position_embedding;
    };

    using qwen::vit::TensorPosHelper;
    using qwen::vit::MLP;

    class ViTSelfAttention : public qwen::vit::ViTSelfAttention
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length):
            qwen::vit::ViTSelfAttention(ctx, hidden_size, num_attention_heads, max_length, false)
        {}
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, int hidden_size, int intermediate_size, int spatial_merge_size, int lm_hidden_size):
            spatial_merge_size(spatial_merge_size),
            hidden_size(hidden_size * spatial_merge_size * spatial_merge_size),
            post_layernorm(ctx, hidden_size),
            downsample(ctx, hidden_size, lm_hidden_size, spatial_merge_size, spatial_merge_size),
            proj(ctx, lm_hidden_size, lm_hidden_size, false),
            post_projection_norm(ctx, lm_hidden_size),
            mlp(ctx, lm_hidden_size, intermediate_size)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w) override
        {
            auto output = post_layernorm.forward(ctx, image_features);
            output = ggml::reshape(ctx, output, ggml::get_dim(output, 0), spatial_merge_size, spatial_merge_size,
                ggml::get_dim(output, 1) / spatial_merge_size / spatial_merge_size);
            output = ggml::permute(ctx, output, 2, 0, 1, 3);
            output = ggml::cont(ctx, output);
            output = downsample.forward(ctx, output);
            output = ggml::reshape(ctx, output, ggml::get_dim(output, 2), ggml::get_dim(output, 3));
            output = proj.forward(ctx, output);
            output = post_projection_norm.forward(ctx, output);
            output = ggml::act(ctx, ActFunc::GELU, output);
            output = mlp.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r +=       post_layernorm.get_param_num(effective_only);
            r +=           downsample.get_param_num(effective_only);
            r += post_projection_norm.get_param_num(effective_only);
            r +=                 proj.get_param_num(effective_only);
            r +=                  mlp.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
                  post_layernorm.load(path + "post_layernorm.", loader);
                      downsample.load(path + "downsample.", loader);
            post_projection_norm.load(path + "merger.post_projection_norm.", loader);
                            proj.load(path + "merger.proj.",  loader);
                             mlp.load(path + "merger.",  loader);
        }

    public:
        const int spatial_merge_size;
        const int hidden_size;
        RMSNorm     post_layernorm;
        Conv2D      downsample;
        Linear      proj;
        LayerNorm   post_projection_norm;
        SiLUMLP     mlp;
    };

    class VitSiLUMLP : public BaseMLP
    {
    public:
        VitSiLUMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        :  BaseMLP(ctx, hidden_size, intermediate_size, ActFunc::SILU, true, true, true)
        {}
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<RMSNorm, ViTSelfAttention, RMSNorm, SiLUMLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size)
            : embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog),
                config.hidden_size, config.intermediate_size, config.spatial_merge_size, lm_hidden_size),
            loaded(false)
        {
            const int max_length = config.max_length;
            pos_helper.reset(new TensorPosHelper(max_length, 0, config.patch_size, config.spatial_merge_size));

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.out_hidden_size, max_length);
                layer->set_id(layer_id);
                layer->attention.set_pos_helper(pos_helper.get());
                layers.emplace_back(layer);
            }
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += embeddings.get_param_num(effective_only);
            r += multi_modal_projector.get_param_num(effective_only);
            for (size_t i = 0; i < layers.size(); i++)
                r += layers[i]->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            if (!loader->has_tensor(path + "patch_embed.proj.bias")) return;

            embeddings.load(path, loader);
            multi_modal_projector.load(path, loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "layers." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            loaded = true;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            pos_helper->prepare(grid_h, grid_w);

            auto output = embeddings.forward(ctx, input, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->attention.grid_h = grid_h;
                layers[i]->attention.grid_w = grid_w;
                output = layers[i]->forward(ctx, output, 0);
            }
            output = multi_modal_projector.forward(ctx, output, grid_h, grid_w);
            return output;
        }

        bool is_loaded(void) const
        {
            return loaded;
        }
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        MultiModalProjector multi_modal_projector;
        std::unique_ptr<TensorPosHelper> pos_helper;
    protected:
        bool loaded;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096):
            max_llm_tokens(max_llm_tokens),
            eval(runtime_config, "vis", GRAPH_SIZE),
            _ctx(eval.get_backend_context())
        {
            _ctx.cache_dtype = runtime_config.cache_type;
        }

        bool load(ModelLoader &loader)
        {
            if (vis_model.get())
            {
                loader.push_allocator_manager(eval.get_layer_allocators());
                vis_model->load("visual.", &loader);
                loader.pop_allocator_manager();
                return vis_model->is_loaded();
            }
            else
                return false;
        }

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
        {
            const auto vis_cfg = config["config.json"]["vision_config"];
            if (!vis_cfg.IsObject()) return false;

            vis_config.dtype = dtype;

            vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
            vis_config.image_ref_size       = (int)vis_cfg["image_size"].ToInt();
            vis_config.num_attention_heads  = (int)vis_cfg["num_heads"].ToInt();
            vis_config.num_hidden_layers    = (int)vis_cfg["depth"].ToInt();
            vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
            vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
            vis_config.out_hidden_size      = (int)vis_cfg["out_hidden_size"].ToInt();
            vis_config.spatial_merge_size   = (int)vis_cfg["spatial_merge_size"].ToInt();
            vis_config.temporal_patch_size  = (int)vis_cfg["temporal_patch_size"].ToInt();

            auto pp_cfg = config["preprocessor_config.json"];
            if (pp_cfg.IsObject())
            {
                pp_cfg = config["preprocessor_config.json"];
                auto image_mean = pp_cfg["image_mean"];
                auto image_std = pp_cfg["image_std"];
                CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
                CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

                vis_config.image_mean[0] = (float)image_mean[0].ToFloat();
                vis_config.image_mean[1] = (float)image_mean[1].ToFloat();
                vis_config.image_mean[2] = (float)image_mean[2].ToFloat();
                vis_config.image_std[0] = (float)image_std[0].ToFloat();
                vis_config.image_std[1] = (float)image_std[1].ToFloat();
                vis_config.image_std[2] = (float)image_std[2].ToFloat();
            }
            else
                return false;

            vis_config.max_length = max_llm_tokens * vis_config.spatial_merge_size * vis_config.spatial_merge_size;

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 14 + vis_config.num_hidden_layers * 11;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            _ctx.dtype = dtype;

            vis_model.reset(new VisionTransformer(&_ctx, vis_config, lm_hidden_size));

            _ctx.check_used_mem_size(true);

            return true;
        }

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
        {
            if ((vis_model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
            if (!vis_model->is_loaded()) return;

            for (auto &image : tok->media_emb)
            {
                run_model(gen_config, tok, dtype, image, buf);
            }
        }

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
        {
            ggml::tensor *media_emb = nullptr;
            const auto make_graph = [this, &media_emb, &image](ComputeContext *ctx) -> ggml::tensor * {
                media_emb = ggml::new_tensor_4d(ctx, ggml::type::GGML_TYPE_F32, vis_config.patch_size, vis_config.patch_size, 3, image.grid_width * image.grid_height);
                auto r = vis_model->forward(ctx, media_emb, image.grid_height, image.grid_width);
                return r;
            };
            const auto write_input_data = [&media_emb, &image](ComputeContext *ctx) {
                Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            };

            std::vector<int64_t> shape;
            eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
            return true;
        }

    protected:
        const int max_llm_tokens;
        std::unique_ptr<VisionTransformer> vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };
}

namespace chatllm::glm::v4v
{
    struct Config : glm::glm4_0414::Config
    {
        int mrope_section[qwen::v2_5_vl::MROPE_SECTION_MAX];
    };

    class ChatHistoryEncoder : public v4::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;

    public:
        const vit::Config *vis_config = nullptr;
        bool vit_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public glm::glm4_0414::Tokenizer
    {
    public:
        Tokenizer(const Config &config) : glm::glm4_0414::Tokenizer(config, &_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        int begin_of_image_token_id;
        int end_of_image_token_id;

        int     video_max_frames = 20;
        bool    native_resolution = false;
        double  fps = 1.0;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = glm::glm4_0414::Tokenizer::load(buffer, n_vocab);

        begin_of_image_token_id = tp->PieceToId("<|begin_of_image|>");
        end_of_image_token_id   = tp->PieceToId("<|end_of_image|>");

        return r;
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(begin_of_image_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(end_of_image_token_id);
    }

    class ConditionalGeneration;

    class TensorPosHelper3D : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper3D(int max_length, int image_id_start, ConditionalGeneration *gen)
            : BaseTensorPosHelper(max_length * 4),
              original_length(max_length), image_id_start(image_id_start),
              gen(gen)
        {
        }

        ggml::tensor *allocate_pos_tensor(InitContext *ctx) override
        {
            ggml::tensor *r = ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length);
            ctx->get_allocator()->alloc(r);
            return r;
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override;
    protected:
        const int original_length;
        const int image_id_start;
        ConditionalGeneration *gen;
    };

    #define MAX_PROJECTED_TOKENS    2048

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int size) : pad_arg(new BlockParams::PadEmbedding(size, size)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public TensorPosHelperPrelude, public ExtendEmbedding, public glm::glm4_0414::ConditionalGeneration
    {
    public:
        typedef glm::glm4_0414::ConditionalGeneration Base;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GLM4V);

        void load(ModelLoader &loader) override;
        bool load_more(const json::JSON &config) override;
        void before_generate(const GenerationConfig &gen_config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
    protected:
        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override;
    public:
        const Config config;
        std::vector<int> v_pos;
    protected:
        std::vector<qwen::v2_5_vl::ImageGridSize> images_grid;
        int token_time;
    protected:
        vit::VisualEmbeddingGeneration visual;
    };

    void TensorPosHelper3D::prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen)
    {
        pos->ne[0] = (int)(gen->v_pos.size());
        Backend::write_tensor_data(pos, gen->v_pos.data(), 0, gen->v_pos.size() * sizeof(gen->v_pos[0]));
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type) :
        TensorPosHelperPrelude(new TensorPosHelper3D(config.max_length, config.vocab_size, this)),
        ExtendEmbedding(MAX_PROJECTED_TOKENS),
        glm::glm4_0414::ConditionalGeneration(config, runtime_config, type),
        config(config),
        token_time(0),
        visual(runtime_config, MAX_PROJECTED_TOKENS)
    {
        delete pad_arg;
        pad_arg = nullptr;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<glm::glm4_0414::ModelClass>()->layers[i];
            layer.attention.mrope_sections = this->config.mrope_section;
            layer.attention.rope_mode = RoPEMode::MROPE;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        glm::glm4_0414::ConditionalGeneration::load(loader);

        loader.add_tensor_name_translations({
            {".self_attn.",                         ".attn."},
            {".input_layernorm.",                   ".norm1."},
            {".post_self_attn_layernorm.",          ".norm2."},
        });

        _chat_encoder.vit_loaded = visual.load(loader);
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        glm::glm4_0414::ConditionalGeneration::load_more(config);
        bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config);
        if (r)
        {
            _chat_encoder.vis_config = &visual.vis_config;
        }
        return r;
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->video_max_frames       = utils::get_opt(args, "video_max_frames", tok->video_max_frames);
        tok->native_resolution      = utils::get_opt(args, "native_resolution", tok->native_resolution);
        tok->fps                    = utils::get_opt(args, "fps", tok->fps);
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        images_grid.clear();
        for (auto &mm : tokenizer->media_emb)
        {
            images_grid.emplace_back(mm.grid_width / visual.vis_config.spatial_merge_size,
                                     mm.grid_height / visual.vis_config.spatial_merge_size);
        }

        auto emb = dynamic_cast<Embedding *>(transformer->word_embeddings);
        visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    bool ConditionalGeneration::generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits)
    {
        const int image_id_start = config.vocab_size;
        const int length = (int)input_ids.size();

        // TODO:
        int token_n_inc = 1;

        v_pos.clear();
        v_pos.resize(length * 4, 0);
        int *p_t = &v_pos[length * 0];
        int *p_h = &v_pos[length * 1];
        int *p_w = &v_pos[length * 2];

        if ((n_past == 0) && (n_past_offset == 0))
            token_time = 0;

        int t = token_time;
        int mm_index = 0;

        int i = 0;
        while (i < length)
        {
            if (input_ids[i] < image_id_start)
            {
                p_t[i] = t;
                p_h[i] = t;
                p_w[i] = t;
                i++;
                t++;
                continue;
            }

            CHATLLM_CHECK(mm_index < (int)images_grid.size());

            auto &dim = images_grid[mm_index++];
            for (int f = 0; f < dim.frame_num; f++, t += token_n_inc)
            {
                for (int h = 0; h < dim.h; h++)
                {
                    for (int w = 0; w < dim.w; w++)
                    {
                        CHATLLM_CHECK(input_ids[i] >= image_id_start);
                        p_t[i] = t;
                        p_h[i] = t + h;
                        p_w[i] = t + w;
                        i++;
                    }
                }
            }
            t = std::max(p_h[i - 1], p_w[i - 1]) + 1;
        }

        token_time = t;
        auto r = Base::generate_next_token(input_ids, gen_config, lm_logits);

        return r;
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.insert(ids.end(), {tok->user_token_id, tok->nl_token_id});

        std::unique_ptr<vision::Resize> resize;

        if (!tok->native_resolution)
            resize.reset(new vision::Resize(vis_config->image_ref_size, vis_config->image_ref_size));

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                int w, h;
                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->patch_size;

                vision::MaxPatchNum     param1(vis_config->max_length);
                vision::MergeKernel     param2(vis_config->spatial_merge_size, vis_config->spatial_merge_size);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
                if ((w <= 0) || (h <= 0)) continue;

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_MergeN_ChannelsRGB_PixelsLeftRightDown);

                const int merge_length = vis_config->spatial_merge_size * vis_config->spatial_merge_size;
                image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_media("image", ids, id_start, image.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
    }
}

namespace chatllm::glm::audio_tower
{
    struct Config
    {
        ggml::type dtype;
        int num_hidden_layers;
        int num_attention_heads;
        int num_key_value_heads;
        int intermediate_size;
        int hidden_size;
        int max_position_embeddings;
        int rope_dim;
        float rope_theta;

        // mel features
        int num_mel_bins;
        int chunk_length;
        int feature_size;
        int hop_length;
        int n_fft;
        int n_samples;
        int nb_max_frames;
        int sampling_rate;
    };

    class AudioSelfAttention : public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        AudioSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length);
    };

    AudioSelfAttention::AudioSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
        RoPESelfAttention<BaseCachelessAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, true, true)
    {
        rope_mode = RoPEMode::Original;
        causal = false;
    }

    class MultiModalProjector : public TheMLP
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size);
    };

    MultiModalProjector::MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size):
        TheMLP(ctx, config.intermediate_size, lm_hidden_size * 2, lm_hidden_size, chatllm::ActFunc::GELU, true)
    {
    }

    class AudioEmbedding : public Block
    {
    public:
        AudioEmbedding(InitContext *ctx, const Config &config);

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;
    public:
        Conv1D    conv1;
        Conv1D    conv2;
    };

    class AudioTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, AudioSelfAttention, LayerNorm, TheBiasedGELUMLP> LayerBlock;

        AudioTransformer(InitContext *ctx, const Config &config, int lm_hidden_size);

        int64_t get_param_num(bool effective_only) const override;

        void load(const std::string &path, TensorLoader *loader) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override;

        bool is_loaded(void) const;

    protected:
        const int intermedia_size;
        AudioEmbedding          embed;
        LayerNorm               norm;
        MultiModalProjector     multi_modal_projector;
        std::vector<std::unique_ptr<LayerBlock>> layers;
    protected:
        bool loaded;
    };

    class AudioEmbeddingGeneration
    {
    public:
        AudioEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);

        bool load(ModelLoader &loader);

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &json_config);

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf);

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &audio, std::vector<uint8_t> &buf);
    protected:
        const int max_llm_tokens;
        std::unique_ptr<AudioTransformer> model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config config;
    };

    AudioEmbedding::AudioEmbedding(InitContext *ctx, const Config &config):
        conv1(ctx, config.num_mel_bins, config.hidden_size, 3, 1, 1),
        conv2(ctx, config.hidden_size, config.hidden_size, 3, 2, 1)
    {}

    int64_t AudioEmbedding::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += conv1.get_param_num(effective_only);
        r += conv2.get_param_num(effective_only);
        return r;
    }

    void AudioEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        conv1.load(path + "conv1.", loader);
        conv2.load(path + "conv2.", loader);
    }

    ggml::tensor *AudioEmbedding::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto output = conv1.forward(ctx, input);
        output      = ggml::act(ctx, ActFunc::GELU, output);
        output      = conv2.forward(ctx, output);
        output      = ggml::act(ctx, ActFunc::GELU, output);
        output      = ggml::permute(ctx, output, 1, 0, 2, 3);

        return output;
    }

    AudioTransformer::AudioTransformer(InitContext *ctx, const Config &config, int lm_hidden_size) :
        intermedia_size(config.intermediate_size),
        embed(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
        norm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
        multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog),
            config, lm_hidden_size),
        loaded(false)
    {
        BlockParams::OverrideKProjBiased k_proj_biased(false);

        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, config.num_key_value_heads,
                config.hidden_size / config.num_attention_heads, config.max_position_embeddings);
            layer->set_id(layer_id);
            layer->attention.rope_dim = config.rope_dim;
            layers.emplace_back(layer);
        }
    }

    int64_t AudioTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += norm.get_param_num(effective_only);
        r += embed.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void AudioTransformer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "conv1.bias")) return;

        norm.load(path + "norm.", loader);
        multi_modal_projector.load("multi_modal_projector.", loader);
        embed.load(path, loader);

        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        loaded = true;
    }

    ggml::tensor *AudioTransformer::forward(ComputeContext *ctx, ggml::tensor *input)
    {
        auto output = embed.forward(ctx, input);

        for (size_t i = 0; i < layers.size(); i++)
        {
            output = layers[i]->forward(ctx, output, 0);
        }
        output = norm.forward(ctx, output);

        int multi = intermedia_size /ggml::get_dim(output, 0);

        output = ggml::reshape(ctx, output, intermedia_size, ggml::get_dim(output, 1) / multi,
                 ggml::get_dim(output, 2), ggml::get_dim(output, 3));
        output = multi_modal_projector.forward(ctx, output);
        return output;
    }

    bool AudioTransformer::is_loaded(void) const
    {
        return loaded;
    }

    AudioEmbeddingGeneration::AudioEmbeddingGeneration(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE):
        max_llm_tokens(max_llm_tokens),
        eval(runtime_config, "aud", GRAPH_SIZE),
        _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
    }

    bool AudioEmbeddingGeneration::load(ModelLoader &loader)
    {
        if (model.get())
        {
            loader.push_allocator_manager(eval.get_layer_allocators());
            model->load("audio.", &loader);
            loader.pop_allocator_manager();
            return model->is_loaded();
        }
        else
            return false;
    }

    bool AudioEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &json_config)
    {
        const auto _cfg = json_config["config.json"]["audio_config"];
        if (!_cfg.IsObject()) return false;

        config.dtype = dtype;

        config.num_hidden_layers        = (int)_cfg["num_hidden_layers"].ToInt();
        config.num_attention_heads      = (int)_cfg["num_attention_heads"].ToInt();
        config.num_key_value_heads      = (int)_cfg["num_key_value_heads"].ToInt();
        config.intermediate_size        = (int)_cfg["intermediate_size"].ToInt();
        config.hidden_size              = (int)_cfg["hidden_size"].ToInt();
        config.max_position_embeddings  = (int)_cfg["max_position_embeddings"].ToInt();
        config.rope_dim                 = (int)(_cfg["head_dim"].ToInt() * _cfg["partial_rotary_factor"].ToFloat());
        config.num_mel_bins             = (int)_cfg["num_mel_bins"].ToInt();
        config.rope_theta               = (float)_cfg["rope_parameters"]["rope_theta"].ToFloat();

        auto pp_cfg = json_config["processor_config.json"]["feature_extractor"];
        if (!pp_cfg.IsObject()) return false;

        config.chunk_length     = (int  )pp_cfg["chunk_length"].ToInt();
        config.feature_size     = (int  )pp_cfg["feature_size"].ToInt();
        config.hop_length       = (int  )pp_cfg["hop_length"].ToInt();
        config.n_fft            = (int  )pp_cfg["n_fft"].ToInt();
        config.n_samples        = (int  )pp_cfg["n_samples"].ToInt();
        config.nb_max_frames    = (int  )pp_cfg["nb_max_frames"].ToInt();
        config.sampling_rate    = (int  )pp_cfg["sampling_rate"].ToInt();

        const size_t tensor_ovhd = ggml::tensor_overhead();
        const size_t num_tensors = 10 + config.num_hidden_layers * 16;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = dtype;

        model.reset(new AudioTransformer(&_ctx, config, lm_hidden_size));

        _ctx.check_used_mem_size(true);

        return true;
    }

    void AudioEmbeddingGeneration::generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
    {
        if ((model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!model->is_loaded()) return;

        for (auto &media : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, media, buf);
        }
    }

    bool AudioEmbeddingGeneration::run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &audio, std::vector<uint8_t> &buf)
    {
        ggml::tensor *media_emb = nullptr;
        const auto make_graph = [this, &media_emb, &audio](ComputeContext *ctx) -> ggml::tensor * {
            media_emb = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, config.max_position_embeddings * 2, config.feature_size);;
            auto r = model->forward(ctx, media_emb);
            return r;
        };
        const auto write_input_data = [&media_emb, &audio](ComputeContext *ctx) {
            Backend::write_tensor_data(media_emb, audio.data.data(), 0, audio.data.size() * sizeof(audio.data[0]));
            //print_tensor(media_emb, 0, true);
        };

        std::vector<int64_t> shape;
        eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
        return true;
    }
}

namespace chatllm::glm::asr
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public HistoryEncoderBracketRole
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;

    protected:
        void append_ending(std::vector<int> &ids) const
        {
            tokenizer->encode("\n", ids);
        }
        void append_sys_ending(std::vector<int> &ids) const override
        {
            append_ending(ids);
        }
        void append_ai_ending(int round_idx, std::vector<int> &ids) const override
        {
            append_ending(ids);
        }
        void append_user_ending(int round_idx, std::vector<int> &ids) const override
        {
            append_ending(ids);
        }
    public:
        const audio_tower::Config *aud_config = nullptr;
        bool aud_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config) : BaseTokenizer(config, &_chat_encoder)
        {}
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);
            eos_token_id = tp->PieceToId("<|endoftext|>");
            terminate_ids.insert(tp->PieceToId("<|user|>"));
            terminate_ids.insert(tp->PieceToId("<|assistant|>"));

            audio_bos_token_id = tp->PieceToId("<|begin_of_audio|>");
            audio_eos_token_id = tp->PieceToId("<|end_of_audio|>");
            user_token_id      = tp->PieceToId("<|user|>");
            return size;
        }

        void inject_audio_ids(std::vector<int> &ids, const int  ids_to_inject_start, const int ids_to_inject_count)
        {
            ids.push_back(audio_bos_token_id);
            for (int i = 0; i < ids_to_inject_count; i++)
                ids.push_back(i + ids_to_inject_start);
            ids.push_back(audio_eos_token_id);
            ids.push_back(user_token_id);
            encode("\n", ids);
        }
    public:
        int audio_bos_token_id;
        int audio_eos_token_id;
        int user_token_id;
        std::string transcription_prompt = "Please transcribe this audio into text";
    };

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int extra_ids) : pad_arg(new BlockParams::PadEmbedding(extra_ids, extra_ids)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_GLM_ASR);

        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void before_generate(const GenerationConfig &gen_config) override;
    public:
        audio_tower::AudioEmbeddingGeneration audio;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        ExtendEmbedding(1500),
        llama::v3::ConditionalGeneration(config, runtime_config, type),
        audio(runtime_config)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);
        bool r = audio.load_more(this->config.dtype, this->config.hidden_size, config);
        if (r)
        {
            _chat_encoder.aud_config = &audio.config;
        }
        return r;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {"multi_modal_projector.fc0.",                      "multi_modal_projector.linear_1."},
            {"multi_modal_projector.fc1.",                      "multi_modal_projector.linear_2."},
            {".mlp.fc1.",                                       ".mlp.fc2."},
            {".mlp.fc0.",                                       ".mlp.fc1."},
        });

        _chat_encoder.aud_loaded = audio.load(loader);
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        audio.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        ids.push_back(tok->user_token_id);
        tok->encode("\n", ids);

        bool found_prompt = false;

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                if (piece.content.size() > 0)
                {
                    tok->encode(piece.content, ids);
                    found_prompt = true;
                }
            }
            else if (piece.type == ContentPiece::Type::Audio)
            {
                CHATLLM_CHECK(aud_loaded) << "Audio model not loaded";

                std::vector<float>          pcm_samples;
                std::vector<audio::mel>     mel_chunks;

                if (!audio::load(piece.content.c_str(), pcm_samples, aud_config->sampling_rate)) continue;

                audio::mel_spectrogram(pcm_samples.data(), pcm_samples.size(),
                    aud_config->n_samples,
                    aud_config->sampling_rate,
                    aud_config->feature_size,
                    aud_config->n_fft,
                    aud_config->hop_length,
                    mel_chunks);

                auto &mel = mel_chunks[0];
                CHATLLM_CHECK(mel.n_len == aud_config->max_position_embeddings * 2);

                tok->media_emb.push_back({.emb_vec_number = aud_config->max_position_embeddings / 2, .data = {}});

                auto &media = tok->media_emb.back();
                media.data = std::move(mel.data);

                //audio::load_text_data_file("C:\\projects\\chatllm.cpp\\dump\\dump\\GlmAsrForConditionalGeneration_debug0_Identity", media.data);

                const int id_start = tok->get_image_total_emb_vectors() - media.emb_vec_number + tok->vocab_size;
                tok->inject_audio_ids(ids, id_start, media.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        if (!found_prompt)
            tokenizer->encode(tok->transcription_prompt, ids);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(CHATGLM,               glm::v1, 1);
    REGISTER_MODEL_LOADER(CHATGLM2,              glm::v2, 1);
    REGISTER_MODEL_LOADER(CHATGLM3,              glm::v3, 1);
    REGISTER_MODEL_LOADER(GLM4,                  glm::v4, 1);
    REGISTER_MODEL_LOADER(GLM4_0414,             glm::glm4_0414, 1);
    REGISTER_MODEL_LOADER(GLM4V,                 glm::v4v, 1);
    REGISTER_MODEL_LOADER(GLM_ASR,               glm::asr, 1);
}