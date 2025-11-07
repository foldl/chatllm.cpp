#include <ostream>
#include "orpheus.h"

namespace chatllm::maya::v1
{
    typedef orpheus::tts::Config Config;


    class Tokenizer : public orpheus::tts::Tokenizer
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

            // https://github.com/MayaResearch/maya1-fastapi/blob/63099d92bce2431a301982e01d52cd018d69a99b/transformers_inference.py#L25-L30
            // which is exactly the same as Orpheus
            const int SOH_ID = 128259;
            const int EOH_ID = 128260;
            const int SOA_ID = 128261;
            const int BOS_ID = 128000;
            const int TEXT_EOT_ID = 128009;
            const int CODE_START_TOKEN_ID = 128257;

            ids.push_back(SOH_ID);
            ids.push_back(bos_token_id);
            BaseTokenizer::encode(oss.str(), ids);
            ids.push_back(TEXT_EOT_ID);
            ids.push_back(EOH_ID);
            ids.push_back(SOA_ID);
            ids.push_back(CODE_START_TOKEN_ID);
        }
    };

    class ConditionalGeneration : public orpheus::tts::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config):
            orpheus::tts::ConditionalGeneration(config, runtime_config, MODEL_TYPE_MAYA1, 128266, 156937)
        {
        }
    protected:
        void decoder_push_llm_tok_id(const GenerationConfig &gen_config, int id, std::vector<float> &pcm_samples) override
        {
            id = id % codec_config.codebook_size;

            pcm_samples.clear();

            vocoder_ids.push_back(id);
            auto count = vocoder_ids.size();
            if (((count % orpheus::snac::FRAME_SIZE) == 0) && (count >= 28))
            {
                std::vector<int> multiframe(vocoder_ids.end() - 28, vocoder_ids.end());
                codec->decode_frame(gen_config, backend_context, multiframe, count, pcm_samples);
            }
        }
    };


    REGISTER_MODEL_LOADER(MAYA1,                maya::v1, 1);
}