#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdint.h>

#include "tokenizer.h"
#include "chat.h"

namespace chatllm
{
    std::string to_string(ModelPurpose purpose);
    std::string format_model_capabilities(uint32_t model_type);
} // namespace chatllm
