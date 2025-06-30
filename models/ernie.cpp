#include "ernie.h"

namespace chatllm::ernie::dense
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
        : chatllm::llama::v2::Tokenizer(config, &_chat_encoder)
    {}

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            oss_prompt << tok->get_system_prompt() << "\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "User:  " + user << "\n";
        auto text = oss_prompt.str();
        tok->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("Assistant:  ", ids);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : chatllm::llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type,
            config.num_key_value_heads, config.head_dim, config.max_length, 12, config.tie_word_embeddings != 0)
    {
        auto transformer = Base::get_typed_transformer<ModelClass2>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
}