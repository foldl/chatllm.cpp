#include "deepseek.h"
#include "qwen.h"

namespace chatllm::bailing::moe
{
    struct Config : public deepseek::v1_moe::Config
    {
        int head_dim;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config)
            : BaseTokenizer(config, &_chat_encoder)
        {
            sys_prompt = "You are Ling, an assistant created by inclusionAI";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    // '(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
                    "\\p{N}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
                    "\\s*[\\r\\n]",
                    "\\s+(?!\\S)",
                    "\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            role_open_token_id = tp->PieceToId("<role>");

            if (role_open_token_id >= 0)
                terminate_ids.insert(role_open_token_id);

            int t = tp->PieceToId("<think>");
            if (t >= 0)
            {
                tp->OverrideTokenDecoding(t, "<think>");
                sys_prompt = "You are Ring, an assistant created by inclusionAI";
            }
            t = tp->PieceToId("</think>");
            if (t >= 0)
                tp->OverrideTokenDecoding(t, "</think>");

            return size;
        }

    protected:
        int role_open_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);
        tokenizer->encode(ai, ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        tokenizer->encode("<role>HUMAN</role>", ids);
        tokenizer->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        tokenizer->encode("<role>ASSISTANT</role>", ids);
    }

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    class ConditionalGeneration : public deepseek::v1_moe::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : deepseek::v1_moe::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>(config, runtime_config, MODEL_TYPE_BAILINGMOE, config.head_dim)
        {}
    };
}

namespace chatllm::bailing2::moe
{
    struct Config : public bailing::moe::Config
    {
        int rope_dim;
        int n_group;
        int topk_group;
        float routed_scaling_factor;
    };

    typedef bailing::moe::Tokenizer Tokenizer;

    const int NUM_EXPERTS                   =  256;
    const int EXPERTS_PER_TOK               =  8;

    class BailingSparseMoE : public BaseSparseMLP
    {
    public:
        BailingSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size, int num_experts = NUM_EXPERTS, int experts_per_tok = EXPERTS_PER_TOK)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, num_experts, experts_per_tok, ActFunc::SILU, true),
              n_group(-1), topk_group(-1)
        {
            score_func = ScoreFunc::Sigmoid;
            always_scaling = true;
        }
    protected:
        ggml::tensor *select_experts(ComputeContext *ctx, ggml::tensor *corrected_score) override;

    public:
        int n_group;
        int topk_group;
    };

    ggml::tensor *BailingSparseMoE::select_experts(ComputeContext *ctx, ggml::tensor *corrected_score)
    {
        const int n_expert = num_local_experts;
        const int experts_per_group = n_expert / n_group;
        CHATLLM_CHECK(ggml::get_dim(corrected_score, 2) == 1);

        ggml::tensor * selected_experts = nullptr;

        ggml::tensor *grouped_scores = ggml::reshape_4d(ctx, corrected_score, experts_per_group, num_experts_per_tok,
                                                        ggml::get_dim(corrected_score, 1), ggml::get_dim(corrected_score, 2));
        selected_experts = ggml::top_k(ctx, grouped_scores, topk_group);

        ggml::tensor *selected_experts_i64 = ggml::cast_int_to_i64(ctx, selected_experts);

        CHATLLM_CHECK(ggml::get_dim(grouped_scores, 3) == 1);
        grouped_scores                      = ggml::reshape_4d(ctx, grouped_scores, 1, ggml::get_dim(grouped_scores, 0), ggml::get_dim(grouped_scores, 1), ggml::get_dim(grouped_scores, 2));
        ggml::tensor *selected_group_scores = ggml::scale(ctx, grouped_scores, 0.0f);
        grouped_scores        = ggml::get_rows(ctx, grouped_scores, selected_experts);
        selected_group_scores = ggml::set_rows(ctx, selected_group_scores, selected_experts_i64, grouped_scores);

        selected_group_scores = ggml::reshape_3d(ctx, selected_group_scores,
            ggml::get_dim(corrected_score, 0), ggml::get_dim(corrected_score, 1), ggml::get_dim(corrected_score, 2));

        selected_experts = ggml::top_k(ctx, selected_group_scores, num_experts_per_tok);

        return selected_experts;
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<BailingSparseMoE, SiLUMLP> BailingMoEMLP;
        typedef LMBlock1<RMSNorm, qwen::v3::QWen3SelfAttention, RMSNorm, BailingMoEMLP> BailingMoEBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_BAILING_MOE2)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3
                                + moe_layer_num * (12 + 7)
                                + dense_layer_num * 14;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok))
                << "unsupported MoE param";

            #define config_rope(attention)     do { \
                    attention.freq_base      = config.rope_theta;                           \
                    attention.rope_dim       = config.rope_dim;                             \
                } while (false)

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new BailingMoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                        config.num_key_value_heads,
                        config.head_dim,
                        config.max_length);
                    layer->mlp.mlp1.norm_topk_prob          = config.norm_topk_prob != 0;
                    layer->mlp.mlp1.routed_scaling_factor   = config.routed_scaling_factor;
                    layer->mlp.mlp1.n_group                 = config.n_group;
                    layer->mlp.mlp1.topk_group              = config.topk_group;
                    config_rope(layer->attention);
                    return layer;
                }
                else
                {
                    auto layer = new qwen::v3::QWen3Block(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.head_dim, config.max_length);
                    config_rope(layer->attention);
                    return layer;
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);

            Base::transformer = transformer;

            #undef config_rope

            w_ctx_.check_used_mem_size(true);
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".mlp2.",              ".shared_experts."},
                {".mlp1.gate.",         ".gate."},
                {".mlp1.experts.",      ".experts."},
                {".mlp1.gate_score_correction_bias",      ".gate.expert_bias"},
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:
        const Config config;

        bool is_layer_moe(int layer_index)
        {
            return (layer_index >= config.first_k_dense_replace) && (layer_index % config.moe_layer_freq == 0);
        }

        int get_moe_layer_num()
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                    r++;
            }
            return r;
        }
    };
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(BAILINGMOE,            bailing::moe,  1);
    REGISTER_MODEL_LOADER(BAILING_MOE2,          bailing2::moe, 1);
}