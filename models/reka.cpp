#include "llama.h"
namespace chatllm::reka::flash
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override
        {
        }
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "assistant: " << ai << " <sep> ";
            tokenizer->encode(oss.str(), ids);
        }

        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "human: ";
            if ((0 == round_idx) && (tokenizer->get_system_prompt().size() > 0))
            {
                oss << tokenizer->get_system_prompt() << " ";
            }
            oss << user << " <sep> ";
            tokenizer->encode(oss.str(), ids);
        }

        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "assistant: ";
            tokenizer->encode(oss.str(), ids);
        }
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr)
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            tp->Encode(" <sep>", &chat_terminate_seq);

            return size;
        }
    public:
        std::vector<int> chat_terminate_seq;
    };

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_REKA_FLASH3)
        {}
    protected:
        bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override
        {
            if (output_ids.size() < 1) return false;

            Tokenizer *tokenizer = dynamic_cast<Tokenizer *>(this->tokenizer);
            int len = 0;

            switch (tokenizer->get_chat_format())
            {
            case ChatFormat::CHAT:
                if (match_output_sequence(output_ids, tokenizer->chat_terminate_seq))
                {
                    pop_output = (int)tokenizer->chat_terminate_seq.size();
                    return true;
                }
                len = (int)tokenizer->chat_terminate_seq.size();
                break;
            default:
                ;
            }

            if (BaseModelForConditionalGeneration::is_output_terminated(output_ids, keep_idx, pop_output))
                return true;

            keep_idx = (int)(output_ids.size()) - len + 1;
            return false;
        }
    };

    REGISTER_MODEL_LOADER(REKA_FLASH3,           reka::flash, 1);
}