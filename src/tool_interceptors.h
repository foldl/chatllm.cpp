#pragma once

#include "chat.h"

namespace chatllm
{
    class ToolCallingInterceptor : public ChunkInterceptor
    {
    public:
        ToolCallingInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override;
        void end() override;
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };
}