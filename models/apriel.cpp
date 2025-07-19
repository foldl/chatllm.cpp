#include "llama.h"

namespace chatllm::apriel
{
    struct Config : public llama::v3::Config
    {
        int head_dim;
        int rope_scaling_original_max_position_embeddings;
        float rope_scaling_beta_fast;
        float rope_scaling_beta_slow;
        float rope_scaling_factor;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override
        {
            std::ostringstream oss;
            ids.push_back(tokenizer->bos_token_id);
            oss << "<|system|>\n" << tokenizer->get_system_prompt() << "\n<|end|>\n";
            tokenizer->encode(oss.str(), ids);
        }
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {
            append_ai_opening(round_idx, ids);
            tokenizer->encode(ai, ids);
            tokenizer->encode("\n<|end|>\n", ids);
        }

        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
        {
            append_user_opening(round_idx, ids);
            tokenizer->encode(user, ids);
            tokenizer->encode("\n<|end|>\n", ids);
        }

        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {
            tokenizer->encode("<|assistant|>\n", ids);
        }

        void append_user_opening(int round_idx, std::vector<int> &ids) const override
        {
            tokenizer->encode("<|user|>\n", ids);
        }
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
            : BaseTokenizer::BaseTokenizer(config, encoder)
        {
            sys_prompt = "You are a helpful AI assistant that provides accurate and concise information.";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            return size;
        }
    };

    class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_APRIEL)
            : llama::v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, config.num_key_value_heads, config.head_dim, config.max_length, 12, false)
        {
            auto transformer = Base::get_typed_transformer<ModelClass2>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.freq_base = config.rope_theta;

                attention.n_original_ctx = config.rope_scaling_original_max_position_embeddings;
                attention.beta_fast = config.rope_scaling_beta_fast;
                attention.beta_slow = config.rope_scaling_beta_slow;

                attention.freq_scale  = 1 / config.rope_scaling_factor;
                attention.attn_factor = 1.0f;
                attention.ext_factor  = 1.0f;
            }
        }
    };

    REGISTER_MODEL_LOADER(APRIEL,                apriel, 1);
}