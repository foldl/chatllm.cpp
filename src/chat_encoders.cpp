#include "chat_encoders.h"

namespace chatllm
{
    HistoryEncoderImStartImEnd::HistoryEncoderImStartImEnd() :
        BaseHistoryEncoder()
    {}

    void HistoryEncoderImStartImEnd::append_sys_prompt(std::vector<int> &ids) const
    {
        if (tokenizer->get_system_prompt().size() > 0)
            encode("system", tokenizer->get_system_prompt(), ids);
    }

    void HistoryEncoderImStartImEnd::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        encode("user", user, ids);
    }

    void HistoryEncoderImStartImEnd::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        encode("assistant", ai, ids);
    }

    void HistoryEncoderImStartImEnd::append_tool(int round_idx, const std::string &tool, std::vector<int> &ids) const
    {
        encode("tool", tool, ids);
    }

    void HistoryEncoderImStartImEnd::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        encode("assistant", ids);
    }

    void HistoryEncoderImStartImEnd::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        encode("user", ids);
    }

    void HistoryEncoderImStartImEnd::set_tokenizer(BaseTokenizer *tokenizer)
    {
        im_start_token_id   = -1;
        im_end_token_id     = -1;
        nl_token_id         = -1;
        BaseHistoryEncoder::set_tokenizer(tokenizer);
        if (nullptr == tokenizer) return;

        std::vector<int> ids;
        tokenizer->tp->Encode("<|im_start|><|im_end|>\n", &ids);

        if (ids.size() == 3)
        {
            im_start_token_id = ids[0];
            im_end_token_id   = ids[1];
            nl_token_id       = ids[2];
        }
    }

    void HistoryEncoderImStartImEnd::encode(const std::string &role, std::vector<int> &ids) const
    {
        if (im_start_token_id >= 0)
        {
            ids.push_back(im_start_token_id);
            tokenizer->encode(role, ids);
            ids.push_back(nl_token_id);
        }
        else
        {
            std::ostringstream oss;
            oss << "<|im_start|>"
                << role
                << "\n";
            tokenizer->encode(oss.str(), ids);
        }
    }

    void HistoryEncoderImStartImEnd::encode(const std::string &role, const std::string &text, std::vector<int> &ids) const
    {
        if (im_start_token_id >= 0)
        {
            ids.push_back(im_start_token_id);
            tokenizer->encode(role, ids);
            ids.push_back(nl_token_id);
            tokenizer->encode(text, ids);
            ids.push_back(im_end_token_id);
            if (append_nl_after_end_tok)
                ids.push_back(nl_token_id);
        }
        else
        {
            std::ostringstream oss;
            oss << "<|im_start|>"
                << role
                << "\n"
                << text
                << "<|im_end|>";
            if (append_nl_after_end_tok)
                oss << "\n";
            tokenizer->encode(oss.str(), ids);
        }
    }

    HistoryEncoderImStartImEnd *HistoryEncoderImStartImEnd::get()
    {
        static HistoryEncoderImStartImEnd encoder;
        return &encoder;
    }

    void HistoryEncoderBracketRole::append_sys_ending(std::vector<int> &ids) const
    {
    }

    void HistoryEncoderBracketRole::append_sys_prompt(std::vector<int> &ids) const
    {
        if (auto_add_bos  && (tokenizer->bos_token_id >= 0))
            ids.push_back(tokenizer->bos_token_id);

        if (tokenizer->get_system_prompt().size() > 0)
        {
            std::ostringstream oss;
            oss << "'<|system|>\n" << tokenizer->get_system_prompt();
            tokenizer->encode(oss.str(), ids);
            append_sys_ending(ids);
        }
    }

    void HistoryEncoderBracketRole::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        std::ostringstream oss;
        oss << "'<|assistant|>\n" << ai;
        tokenizer->encode(oss.str(), ids);
        append_ai_ending(round_idx, ids);
    }

    void HistoryEncoderBracketRole::append_user_ending(int round_idx, std::vector<int> &ids) const
    {
    }

    void HistoryEncoderBracketRole::append_ai_ending(int round_idx, std::vector<int> &ids) const
    {
    }

    void HistoryEncoderBracketRole::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss;
        oss << "'<|user|>\n" << user;
        tokenizer->encode(oss.str(), ids);
        append_user_ending(round_idx, ids);
    }

    void HistoryEncoderBracketRole::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        std::ostringstream oss;
        oss << "'<|assistant|>\n";
        tokenizer->encode(oss.str(), ids);
    }
}
