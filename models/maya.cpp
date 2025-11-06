#include <ostream>
#include "orpheus.h"

namespace chatllm::maya::v1
{
    typedef orpheus::tts::Config Config;


    class Tokenizer : public  orpheus::tts::Tokenizer
    {
    public:
        using  orpheus::tts::Tokenizer::Tokenizer;

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            auto r =  orpheus::tts::Tokenizer::load(buffer, n_vocab);

            return r;
        }

        void encode(const std::string &text, std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            oss << "<description=\"";
            oss << (!voice.empty() ? voice : "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.");
            oss << "\"> ";
            oss << text;

            ids.push_back(bos_token_id);
            BaseTokenizer::encode(oss.str(), ids);
            ids.push_back(128009);
        }
    };

    class ConditionalGeneration : public orpheus::tts::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config):
            orpheus::tts::ConditionalGeneration(config, runtime_config, MODEL_TYPE_MAYA1, 128256, 156938) // 128266, 156937)
        {
        }
    };


    REGISTER_MODEL_LOADER(MAYA1,                maya::v1, 1);
}