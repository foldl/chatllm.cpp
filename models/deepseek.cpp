#include "deepseek.h"

namespace chatllm::deepseek::v1
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder)
        : llama::v2::Tokenizer::Tokenizer(config, chat_encoder)
    {
        sys_prompt = "";
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_DEEPSEEK)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            std::ostringstream oss_prompt;
            oss_prompt << tok->get_system_prompt() << "\n\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids, false, false);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "User: " << user << "\n\n";

        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "Assistant: ";
        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id);
    }
}

namespace chatllm::deepseek::coder
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
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const llama::v2::Config &config, BaseHistoryEncoder *chat_encoder)
        : llama::v2::Tokenizer::Tokenizer(config, chat_encoder)
    {
        sys_prompt = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.";
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_DEEPSEEK_CODER),
        config(config)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "[\r\n]",
                "\\s?\\p{L}+",
                "\\s?\\p{P}+",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);

        std::vector<int> ids;
        tp->Encode("<｜fim▁hole｜><｜fim▁begin｜><｜fim▁end｜><|User|><|Assistant|><|EOT|>", &ids);

        CHATLLM_CHECK(ids.size() == 6) << "tokenizer error";

        fim_hole_token_id = ids[0];
        fim_begin_token_id = ids[1];
        fim_end_token_id = ids[2];
        user_token_id = ids[3];
        assistant_token_id = ids[4];
        eot_token_id = ids[5];
        return size;
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids, true, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        append_ai_opening(round_idx, ids);

        oss_prompt << ai << "\n<|EOT|>";

        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "\n### Instruction:\n" << user;

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "\n### Response:\n";

        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == eot_token_id);
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        llama::v2::ConditionalGeneration::load(loader);

        auto transformer = get_typed_transformer<ModelClass>();

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
            attention.freq_scale = 1 / config.rope_scaling;
        }
    }

    REGISTER_MODEL_LOADER(DEEPSEEK_CODER,        coder, 1);
}

namespace chatllm::deepseek::v1_moe
{
}

namespace chatllm::deepseek::v2_light
{
    QProj::QProj(InitContext *ctx, int hidden_size, int num_attention_heads,
        int q_lora_rank, int rope_dim, int qk_nope_head_dim, bool use_bias)
        : d_q_proj(q_lora_rank > 0 ? new Linear(ctx, hidden_size, q_lora_rank, use_bias) : nullptr),
            u_q_proj(q_lora_rank > 0 ? new Linear(ctx, q_lora_rank, (qk_nope_head_dim + rope_dim) * num_attention_heads, false) : nullptr),
            norm(q_lora_rank > 0 ? new RMSNorm(ctx, q_lora_rank) : nullptr),
            q_proj(q_lora_rank <= 0 ? new Linear(ctx, hidden_size, (qk_nope_head_dim + rope_dim) * num_attention_heads, use_bias) : nullptr)
    {}

    int64_t QProj::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (q_proj)
        {
            r += q_proj->get_param_num(effective_only);
        }
        else
        {
            r += d_q_proj->get_param_num(effective_only);
            r += u_q_proj->get_param_num(effective_only);
            r += norm->get_param_num(effective_only);
        }
        return r;
    }

    ggml::tensor *QProj::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *tmpq = nullptr;

        if (q_proj)
        {
            tmpq = q_proj->forward(ctx, hidden_states);
        }
        else
        {
            ggml::tensor *q_lora = d_q_proj->forward(ctx, hidden_states);
            q_lora = norm->forward(ctx, q_lora);
            tmpq = u_q_proj->forward(ctx, q_lora);
        }

        return tmpq;
    }

    void QProj::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);

        if (q_proj)
        {
            q_proj->load(path + "q_proj.", loader);
        }
        else
        {
            d_q_proj->load(path + "d_q_proj.", loader);
            u_q_proj->load(path + "u_q_proj.", loader);
            norm->load(path + "q_norm.", loader);
        }
    }
}

namespace chatllm::deepseek::v2
{
    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : v2_light::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>(config, runtime_config, MODEL_TYPE_DEEPSEEK_V2, config.q_lora_rank)
    {
        CHATLLM_CHECK(config.topk_group == 1) << "unsupported MoE param";
    }
}

namespace chatllm::deepseek::v3_light
{
    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v2_light::ConditionalGeneration(config, runtime_config, type, -1, BaseSparseMLP::ScoreFunc::Sigmoid, true, true)
    {
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(DEEPSEEK,              deepseek::v1, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V2_LIGHT,     deepseek::v2_light, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V2,           deepseek::v2, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V3_LIGHT,     deepseek::v3_light, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V1_MoE,       deepseek::v1_moe, 1);
}