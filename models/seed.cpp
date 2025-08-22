#include "../src/models.h"
#include "../src/models_priv.h"

#define MODEL_TYPE_SEED_OSS     (MODEL_TYPE_SEED + 0)

namespace chatllm::seed::oss
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        float rope_theta;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
        void append_user_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : Tokenizer(config, &_chat_encoder)

        {}

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder,
                BaseHistoryEncoder *qa_encoder = nullptr,
                BaseHistoryEncoder *completion_encoder = nullptr)
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder),
              thinking_budget(-1), budget_reflections(-1)
        {
            sys_prompt = "";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    public:
        void encode_role(std::vector<int> &ids, const std::string &role) const;
        void encode(std::vector<int> &ids, const std::string &role, const std::string &content) const;

    public:
        int toolcall_begin_token_id;
        int toolcall_end_token_id;
        int think_begin_token_id;
        int think_end_token_id;
        int budget_begin_token_id;
        int budget_end_token_id;
        int nl_token_id;
    public:
        int thinking_budget;
        int budget_reflections;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                "[^\r\n\\p{L}\\p{N}]?\\p{L}+",
                "\\p{N}{1}",
                " ?[^\\s\\p{L}\\p{N}\r\n]+",
                "\\s*[\r\n]+",
                "\\s+(?!\\S)",
                "\\s+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);

        toolcall_begin_token_id   = tp->PieceToId("<seed:tool_call>");
        toolcall_end_token_id     = tp->PieceToId("</seed:tool_call>");
        think_begin_token_id      = tp->PieceToId("<seed:think>");
        think_end_token_id        = tp->PieceToId("</seed:think>");
        budget_begin_token_id     = tp->PieceToId("<seed:cot_budget_reflect>");
        budget_end_token_id       = tp->PieceToId("</seed:cot_budget_reflect>");

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        tp->OverrideTokenDecoding(think_begin_token_id, "<think>");
        tp->OverrideTokenDecoding(think_end_token_id, "</think>");

        return size;
    }

    void Tokenizer::encode_role(std::vector<int> &ids, const std::string &role) const
    {
        ids.push_back(bos_token_id);
        BaseTokenizer::encode(role, ids);
        ids.push_back(nl_token_id);
    }

    void Tokenizer::encode(std::vector<int> &ids, const std::string &role, const std::string &content) const
    {
        ids.push_back(bos_token_id);
        BaseTokenizer::encode(role, ids);
        ids.push_back(nl_token_id);
        BaseTokenizer::encode(content, ids);
        ids.push_back(eos_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        auto s = tok->get_system_prompt();
        if (s.size() > 0)
        {
            tok->encode(ids, "system", s);
        }

        if (tok->thinking_budget == 0)
        {
            tok->encode(ids, "system", "You are an intelligent assistant that can answer questions in one step without the need for reasoning and thinking, that is, your thinking budget is 0. Next, please skip the thinking process and directly start answering the user's questions.");
        }
        else if (tok->thinking_budget > 0)
        {
            const static std::vector<std::pair<int, int>> table =
            {
                {0,      0},
                {512,    128},
                {1024,   256},
                {2048,   512},
                {4096,   512},
                {8192,   1024},
                {16384,  1024},
            };
            for (const auto &t : table)
            {
                if (t.first >= tok->think_begin_token_id)
                {
                    tok->budget_reflections = t.second;
                    break;
                }
            }

            if (tok->budget_reflections < 0)
               tok->budget_reflections = table.back().second;

            std::ostringstream oss;
            oss << "You are an intelligent assistant with reflective ability. In the process of thinking and reasoning, you need to strictly follow the thinking budget, which is "
                << "\"" << tok->thinking_budget << "\"."
                << "That is, you need to complete your thinking within "
                << tok->thinking_budget
                << " tokens and start answering the user's questions. You will reflect on your thinking process every "
                << tok->budget_reflections
                << " tokens, stating how many tokens have been used and how many are left.";
            tok->encode(ids, "system", oss.str());
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode(ids, "assistant", ai);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode(ids, "user", user);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role(ids, "assistant");

        if (tok->thinking_budget == 0)
        {
            ids.push_back(tok->think_begin_token_id);
            ids.push_back(tok->budget_begin_token_id);
        }
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role(ids, "user");
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, QWen2Block, int, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_SEED_OSS);

        void set_additional_args(const std::map<std::string, std::string> &args) override;
    public:
        Config config;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
        config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * 15;
        const size_t ctx_size = num_tensors * tensor_ovhd;

        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, false,
                                        config.hidden_size, config.num_attention_heads,
                                        config.intermediate_size, config.num_key_value_heads,
                                        config.head_dim,
                                        config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base = config.rope_theta;
        }

        w_ctx_.check_used_mem_size(true);
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->thinking_budget            = utils::get_opt(args, "thinking_budget", tok->thinking_budget);
    }

    REGISTER_MODEL_LOADER(SEED_OSS,              seed::oss, 1);
}