#include "mistral.h"

namespace chatllm::mistral::mistral
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const llama::v2::Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const llama::v2::Config &config, BaseHistoryEncoder *encoder)
        : llama::v2::Tokenizer::Tokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = llama::v2::Tokenizer::load(buffer, n_vocab);
        if (tp->GetPieceSize() == 32768)
        {
            // Mistral v0.3
            start_inst_token_id             = 3;
            end_inst_token_id               = 4;
            tool_calls_token_id             = 5;
            start_avail_tools_token_id      = 6;
            end_avail_tools_token_id        = 7;
            start_tool_results_token_id     = 8;
            end_tool_results_token_id       = 9;
            tp->AddAddedToken("[INST]",             start_inst_token_id);
            tp->AddAddedToken("[/INST]",            end_inst_token_id);
            tp->AddAddedToken("[TOOL_CALLS]",       tool_calls_token_id);
            tp->AddAddedToken("[AVAILABLE_TOOLS]",  start_avail_tools_token_id);
            tp->AddAddedToken("[/AVAILABLE_TOOLS]", end_avail_tools_token_id);
            tp->AddAddedToken("[TOOL_RESULTS]",     start_tool_results_token_id);
            tp->AddAddedToken("[/TOOL_RESULTS]",    end_tool_results_token_id);
        }
        else
        {
            start_inst_token_id             = tp->PieceToId("[INST]");
            end_inst_token_id               = tp->PieceToId("[/INST]");
            tool_calls_token_id             = tp->PieceToId("[TOOL_CALLS]");
            start_avail_tools_token_id      = tp->PieceToId("[AVAILABLE_TOOLS]");
            end_avail_tools_token_id        = tp->PieceToId("[/AVAILABLE_TOOLS]");
            start_tool_results_token_id     = tp->PieceToId("[TOOL_RESULTS]");
            end_tool_results_token_id       = tp->PieceToId("[/TOOL_RESULTS]");
        }
        return r;
    }

    class MistralInterceptor : public ChunkInterceptor
    {
    public:
        MistralInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(streamer->tokenizer);
            if (tok->start_avail_tools_token_id < 0)
            {
                next->put_chunk(first, chunk);
                return;
            }

            if (first)
            {
                found_tool_call = chunk.starts_with("[TOOL_CALLS]");
                if (found_tool_call) return;
            }

            if (found_tool_call)
                oss << chunk;
            else
                next->put_chunk(first, chunk);
        }

        void end() override
        {
            if (found_tool_call)
                streamer->putln(oss.str(), BaseStreamer::TextType::TOOL_CALLING);

            oss.str("");
            found_tool_call = false;

            ChunkInterceptor::end();
        }
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };

    static MistralInterceptor interceptor;

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : llama::v2::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>(config, runtime_config, type,
            config.num_key_value_heads, config.max_length, 13)
    {
        CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == SLIDING_WINDOW_LEN))
            << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        batch_input = false;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : ConditionalGeneration(config, runtime_config, MODEL_TYPE_MISTRAL)
    {
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &interceptor;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[TOOL_RESULTS]" << content << "[/TOOL_RESULTS]";

        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        if (tok->start_avail_tools_token_id >= 0)
        {
            if (user.starts_with("[TOOL_RESULTS]"))
            {
                oss_prompt << user;
            }
            else
            {
                std::string user_input = user;
                const std::string tag_tools = "[/AVAILABLE_TOOLS]";
                size_t pos = user.find(tag_tools);
                if (pos != std::string::npos)
                {
                    oss_prompt <<  user.substr(0, pos + tag_tools.size());
                    user_input = user.substr(pos + tag_tools.size());
                }

                oss_prompt << "[INST] " << user_input << " [/INST]";
                tok->encode(oss_prompt.str(), ids, false, false);
            }
        }
        else
        {
            oss_prompt << "[INST] " << user << " [/INST]";
        }

        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }
}

namespace chatllm::mistral::mixtral
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

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : mistral::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[INST] " << user << " [/INST]";
        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }
}

namespace chatllm::mistral::mistral2
{
    Tokenizer::Tokenizer(const Config &config)
        : mistral::Tokenizer(config, &mistral::_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);

        start_inst_token_id             = tp->PieceToId("[INST]");
        end_inst_token_id               = tp->PieceToId("[/INST]");
        tool_calls_token_id             = tp->PieceToId("[TOOL_CALLS]");
        start_avail_tools_token_id      = tp->PieceToId("[AVAILABLE_TOOLS]");
        end_avail_tools_token_id        = tp->PieceToId("[/AVAILABLE_TOOLS]");
        start_tool_results_token_id     = tp->PieceToId("[TOOL_RESULTS]");
        end_tool_results_token_id       = tp->PieceToId("[/TOOL_RESULTS]");
        return size;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : llama::v2::GenericConditionalGeneration<MistralBlock<mistral::SLIDING_WINDOW_LEN>>(config, runtime_config, type,
            config.num_key_value_heads, config.head_dim, config.max_length, 13, false)
    {
        CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == mistral::SLIDING_WINDOW_LEN))
            << "sliding_window (" << config.sliding_window << ") must be " << mistral::SLIDING_WINDOW_LEN;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass2>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        batch_input = false;
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &mistral::interceptor;
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(MISTRAL,               mistral::mistral, 1);
    REGISTER_MODEL_LOADER(MIXTRAL,               mistral::mixtral, 1);
    REGISTER_MODEL_LOADER(MISTRAL2,              mistral::mistral2, 1);
}