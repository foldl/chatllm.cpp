#include "tool_interceptors.h"

namespace chatllm
{
    void ToolCallingInterceptor::put_chunk(bool first, const std::string &chunk)
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

    void ToolCallingInterceptor::end()
    {
        found_tool_call = false;
        auto s = oss.str();
        oss.str("");
        if (s.size() > 0)
            streamer->putln(s.c_str(), BaseStreamer::TextType::TOOL_CALLING);

        ChunkInterceptor::end();
    }
}