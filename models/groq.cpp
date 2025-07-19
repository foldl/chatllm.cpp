#include "llama.h"

namespace chatllm::groq
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public llama::v3::ChatHistoryEncoder
    {
    public:
        void append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const override;
    };

    static class ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public llama::v3::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : llama::v3::Tokenizer(config, &_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t size = llama::v3::Tokenizer::load(buffer, n_vocab);

            tool_call_start_token_id            = tp->PieceToId("<tool_call>");
            tool_call_end_token_id              = tp->PieceToId("</tool_call>");
            tools_start_token_id                = tp->PieceToId("<tools>");
            tools_end_token_id                  = tp->PieceToId("</tools>");
            tool_response_start_token_id        = tp->PieceToId("<tool_response>");
            tool_response_end_token_id          = tp->PieceToId("</tool_response>");

            tp->EnableReturnSpecialToken(true);

            return size;
        }

    public:
        int tool_call_start_token_id;
        int tool_call_end_token_id;
        int tools_start_token_id;
        int tools_end_token_id;
        int tool_response_start_token_id;
        int tool_response_end_token_id;
    };

    void ChatHistoryEncoder::append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        tok->encode_header("tool", ids);
        ids.push_back(tok->tool_response_start_token_id);
        tok->encode_content(content, ids);
        ids.push_back(tok->tool_response_end_token_id);
    }

    class ToolCallingInterceptor : public ChunkInterceptor
    {
    public:
        ToolCallingInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            if (!found_tool_call)
            {
                if (chunk == "<tool_call>")
                {
                    found_tool_call = true;
                }
                else
                    next->put_chunk(first, chunk);
            }
            else
            {
                if (chunk == "</tool_call>")
                    found_tool_call = false;
                else
                    oss << chunk;
            }
        }

        void end() override
        {
            found_tool_call = false;
            auto s = oss.str();
            oss.str("");
            if (s.size() > 0)
                streamer->putln(s.c_str(), BaseStreamer::TextType::TOOL_CALLING);

            ChunkInterceptor::end();
        }
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };

    static class ToolCallingInterceptor interceptor;

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_LLAMA3_GROQ_TOOL)
        {}

        ChunkInterceptor *get_interceptor(void) override { return &interceptor; }
    };

    REGISTER_MODEL_LOADER(LLAMA3_GROQ_TOOL,      groq, 1);
}