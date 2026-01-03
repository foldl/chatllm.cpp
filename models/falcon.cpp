#include "llama.h"
#include "../src/chat_encoders.h"

namespace chatllm::falcon::v3
{
    typedef llama::v3::Config Config;

    class ChatHistoryEncoder : public HistoryEncoderBracketRole
    {
    protected:
        void append_ai_ending(int round_idx, std::vector<int> &ids) const override
        {
            ids.push_back(tokenizer->eos_token_id);
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
            tp = new tokenizer::BPEProcessor2();
            size_t size = tp->Load(buffer, n_vocab);

            int nl = tp->PieceToId("Ċ");
            tp->OverrideTokenDecoding(nl, "\n");

            return size;
        }
    };

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_FALCON3)
        {}
    };

    REGISTER_MODEL_LOADER(FALCON3,               falcon::v3, 1);
}