#pragma once

#include "chat.h"

namespace chatllm
{
    class HistoryEncoderImStartImEnd: public BaseHistoryEncoder
    {
    public:
        static HistoryEncoderImStartImEnd *get();

        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_tool(int round_idx, const std::string &tool, std::vector<int> &ids) const override;

        void append_ai_opening(int round_idx, std::vector<int> &ids) const;
        void append_user_opening(int round_idx, std::vector<int> &ids) const;

        void set_tokenizer(BaseTokenizer *tokenizer) override;
    protected:
        HistoryEncoderImStartImEnd();
        void encode(const std::string &role, const std::string &text, std::vector<int> &ids) const;
        void encode(const std::string &role, std::vector<int> &ids) const;
    public:
        bool append_nl_after_end_tok = true;
    protected:
        int im_start_token_id   = -1;
        int im_end_token_id     = -1;
        int nl_token_id         = -1;
    };
}