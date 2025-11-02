#include "llama.h"
#include "deepseek.h"

namespace chatllm::megrez::chat
{
    typedef llama::v3::Config Config;

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
            : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
        {
            sys_prompt = "你是Megrez-3B-Instruct，将针对用户的问题给出详细的、积极的回答。";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

    public:
        void encode(const std::string &text, std::vector<int> &ids, bool add_role, bool add_turn_end) const;

    public:
        int role_start_token_id;
        int role_end_token_id;
        int turn_end_token_id;
        int function_start_token_id;
        int function_end_token_id;
    };

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        role_start_token_id           = tp->PieceToId("<|role_start|>");
        role_end_token_id             = tp->PieceToId("<|role_end|>");
        function_start_token_id       = tp->PieceToId("<|function_start|>");
        function_end_token_id         = tp->PieceToId("<|function_end|>");

        if (eos_token_id < 0)
            eos_token_id              = tp->PieceToId("<|eos|>");

        turn_end_token_id             = tp->PieceToId("<|turn_end|>");
        terminate_ids.insert(turn_end_token_id);

        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_role, bool add_turn_end) const
    {
        if (add_role)
            ids.push_back(role_start_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_role)
            ids.push_back(role_end_token_id);
        if (add_turn_end)
            ids.push_back(turn_end_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false);
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

        tok->encode("system", ids, true, false);
        tok->encode(tok->get_system_prompt(), ids, false, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        tok->encode("user", ids, true, false);
        tok->encode(user, ids, false, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("assistant", ids, true, false);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("user", ids, true, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == role_start_token_id) || (id == role_end_token_id);
    }

    class ConditionalGeneration : public llama::v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : llama::v3::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_MEGREZ)
        {}
    };

}

namespace chatllm::megrez::moe
{
    struct Config : public deepseek::v1_moe::Config
    {
        int experts_shared_frequency;
        int n_group;
        int topk_group;
        float routed_scaling_factor;
    };

    typedef chat::Tokenizer Tokenizer;

    class SparseMoEWithExperts : public GenericGroupedSparseMoE
    {
    public:
        SparseMoEWithExperts(InitContext *ctx, int hidden_size, int intermediate_size):
            GenericGroupedSparseMoE(ctx, hidden_size, BlockParams::MoE::num_experts, BlockParams::MoE::experts_per_tok, true, false, false, false),
            experts(ctx, hidden_size, intermediate_size, BlockParams::MoE::num_experts, BlockParams::MoE::experts_per_tok, ActFunc::SILU, false)
        {
            set_experts(&experts);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = GenericSparseMLP::get_param_num(effective_only);
            r += experts.get_param_num(effective_only);
            return r;
        }
        void load(const std::string &path, TensorLoader *loader) override
        {
            GenericSparseMLP::load(path, loader);
            experts.load(path + "experts.", loader);
        }
    public:
        MultiMLP experts;
    };

    class SparseMoESharedExperts : public GenericGroupedSparseMoE
    {
    public:
        SparseMoESharedExperts(InitContext *ctx, int hidden_size, int intermediate_size):
            GenericGroupedSparseMoE(ctx, hidden_size, BlockParams::MoE::num_experts, BlockParams::MoE::experts_per_tok, true, false, false, false)
        {
        }
        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = GenericSparseMLP::get_param_num(effective_only);
            if (effective_only)
                r += p_experts->get_param_num(effective_only);
            return r;
        }
    };

    class Prelude
    {
    public:
        Prelude(const Config &config)
        {
            BlockParams::MoE::num_experts     = config.n_routed_experts;
            BlockParams::MoE::experts_per_tok = config.num_experts_per_tok;
        }
    };

    class ConditionalGeneration;

    class MoEModel : public HeterogeneousModel
    {
    public:
        using HeterogeneousModel::HeterogeneousModel;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override;
    public:
        ConditionalGeneration *gen;
    };

    class ConditionalGeneration : public Prelude, public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<SparseMoEWithExperts, SiLUMLP> MoEMLP;
        typedef LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, MoEMLP> MoEBlock;
        typedef CombinedMLP<SparseMoESharedExperts, SiLUMLP> SharedMoEMLP;
        typedef LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, SharedMoEMLP> MoEBlockShared;

        enum LayerType
        {
            Dense,
            MoE,
            SharedMoE,
        };
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_MEGREZ_MOE)
            : Prelude(config),
              BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3
                                + count_layer_type(LayerType::MoE) * (9 + 8)
                                + count_layer_type(LayerType::SharedMoE) * (9 + 5)
                                + count_layer_type(LayerType::Dense) * 12;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            const int head_dim = config.hidden_size / config.num_attention_heads;
            MoEBlock *last_moe_block = nullptr;

            #define config_layer(layer)     do { \
                    layer->mlp.mlp1.norm_topk_prob          = config.norm_topk_prob != 0;   \
                    layer->mlp.mlp1.routed_scaling_factor   = config.routed_scaling_factor; \
                    layer->mlp.mlp1.n_group                 = config.n_routed_experts / config.n_group; \
                    layer->mlp.mlp1.topk_group              = config.topk_group;            \
                    layer->mlp.mlp1.always_scaling          = true;                         \
                } while (false)

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                switch (get_layer_type(layer_index))
                {
                case LayerType::Dense:
                    {
                        auto layer = new LlamaBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length);
                        layer->attention.freq_base      = config.rope_theta;
                        return layer;
                    }
                case LayerType::MoE:
                    {
                        auto layer = new MoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                            config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                            config.num_key_value_heads,
                            head_dim,
                            config.max_length);
                        layer->attention.freq_base              = config.rope_theta;
                        config_layer(layer);

                        last_moe_block = layer;
                        return layer;
                    }
                case LayerType::SharedMoE:
                    {
                        auto layer = new MoEBlockShared(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                            config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                            config.num_key_value_heads,
                            head_dim,
                            config.max_length);
                        layer->attention.freq_base              = config.rope_theta;
                        config_layer(layer);

                        CHATLLM_CHECK(last_moe_block);
                        layer->mlp.mlp1.set_experts(&last_moe_block->mlp.mlp1.experts);
                        return layer;
                    }
                default:
                    return nullptr;
                }
            };

            #undef config_layer

            auto model = new MoEModel(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);
            model->gen = this;
            transformer = model;

            w_ctx_.check_used_mem_size(true);
        }

        void load(ModelLoader &loader) override
        {
            loader.add_tensor_name_translations({
                {".mlp2.",              ".shared_experts."},
                {".mlp1.gate.",         ".gate."},
                {".mlp1.experts.",      ".experts."},
                {".mlp1.gate_score_correction_bias",     ".gate.e_score_correction_bias"}
            });

            BaseModelForConditionalGeneration::load(loader);
        }

    public:
        Config config;

        LayerType get_layer_type(int layer_index)
        {
            if (layer_index < config.first_k_dense_replace) return LayerType::Dense;
            if (layer_index % config.moe_layer_freq != 0) return LayerType::Dense;

            const bool init_experts = ((layer_index - config.first_k_dense_replace) % config.experts_shared_frequency) == 0;
            return init_experts ? LayerType::MoE : LayerType::SharedMoE;
        }

        int count_layer_type(LayerType t)
        {
            int r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (get_layer_type(i) == t)
                    r++;
            }
            return r;
        }
    };

    ggml::tensor *MoEModel::forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past)
    {
        before_forward(ctx, input_ids, n_past);

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        ggml::tensor *hidden_states = custom_embedding ? custom_embedding(ctx, input_ids) :  word_embeddings->forward(ctx, input_ids);
        ggml::tensor *pre_gate_hidden_states = nullptr;
        for (auto &layer : layers)
        {
            ctx->move_to_layer(layer->get_id());

            switch (gen->get_layer_type(layer->get_id()))
            {
            case ConditionalGeneration::LayerType::Dense:
                {
                    auto typed_layer = dynamic_cast<LlamaBlock *>(layer);
                    hidden_states = typed_layer->forward(ctx, hidden_states, n_past);
                    pre_gate_hidden_states = typed_layer->last_result_post_attn_norm;
                }
                break;
            case ConditionalGeneration::LayerType::MoE:
                {
                    auto typed_layer = dynamic_cast<ConditionalGeneration::MoEBlock *>(layer);
                    hidden_states = typed_layer->forward(ctx, hidden_states, pre_gate_hidden_states, n_past);
                    pre_gate_hidden_states = typed_layer->last_result_post_attn_norm;
                }
                break;
            case ConditionalGeneration::LayerType::SharedMoE:
                {
                    auto typed_layer = dynamic_cast<ConditionalGeneration::MoEBlockShared *>(layer);
                    hidden_states = typed_layer->forward(ctx, hidden_states, pre_gate_hidden_states, n_past);
                    pre_gate_hidden_states = typed_layer->last_result_post_attn_norm;
                }
                break;
            default:
                break;
            }
        }

        last_hidden_state = hidden_states;

        ctx->move_to_layer(LayerAllocatorManager::Epilog);
        return final_steps->forward(this, ctx, input_ids, hidden_states);
    }
}

namespace chatllm::megrez
{
    REGISTER_MODEL_LOADER(MEGREZ,                chat, 1);
    REGISTER_MODEL_LOADER(MEGREZ_MOE,            moe, 1);
}