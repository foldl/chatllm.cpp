#include "gemma.h"
#include "llama.h"

using namespace chatllm::gemma;

namespace chatllm::rnj::v1
{
    struct Config : public v3::Config
    {
        float attn_factor;
        float beta_fast;
        float beta_slow;
        float extrapolation_factor;
        float factor;
        int   original_max_position_embeddings;
        float final_logit_soft_capping;
        float attn_logit_soft_capping;
    };

    class Tokenizer : public chatllm::llama::v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config):
            chatllm::llama::v3::Tokenizer(config)
        {
            sys_prompt = "You are rnj-1, a foundation model trained by Essential AI.\n";
            double_nl_token = false;
        }
    };

    template <class Layer> static void setup_layer(Block *block, const Config &config, Block *attn_scores_pp)
    {
        auto layer = dynamic_cast<Layer *>(block);
        auto &attention = layer->attention;
        attention.setup_yarn(config.rope_theta,
            config.original_max_position_embeddings, config.beta_fast, config.beta_slow, config.factor);
        attention.attn_factor = config.attn_factor;
        attention.ext_factor  = config.extrapolation_factor;
        attention.attn_scores_pp = attn_scores_pp;
    }

    class ConditionalGeneration : public v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_RNJ_1) :
            v3::ConditionalGeneration(config, runtime_config, type),
            logits_pp(1.0f / config.final_logit_soft_capping / sqrtf((float)config.hidden_size), config.final_logit_soft_capping),
            attn_scores_pp(1.0f / config.attn_logit_soft_capping, config.attn_logit_soft_capping)
        {
            Block *attn_pp = config.attn_logit_soft_capping > 0.0f ? &attn_scores_pp : nullptr;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_sliding(i))
                {
                    switch (sliding_window)
                    {
                    case 512:
                        setup_layer<v3::Gemma3SWABlock512>(get_typed_transformer<ModelClass>()->get_layer(i), config, attn_pp);
                        break;
                    default:
                        setup_layer<v3::Gemma3SWABlock1024>(get_typed_transformer<ModelClass>()->get_layer(i), config, attn_pp);
                        break;
                    }
                }
                else
                {
                    setup_layer<v3::Gemma3FullBlock>(get_typed_transformer<ModelClass>()->get_layer(i), config, attn_pp);
                }
            }

            if (config.final_logit_soft_capping > 0.0f)
                get_typed_transformer<ModelClass>()->logits_pp = &logits_pp;
        }
    public:
        v2::TanhScaling logits_pp;
        v2::TanhScaling attn_scores_pp;
    };
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(RNJ_1,                rnj::v1, 1);
}
