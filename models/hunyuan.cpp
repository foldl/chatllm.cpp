#include "hunyuan.h"
namespace chatllm::hunyuan::dense
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : BaseTokenizer(config, &_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);

        int SPECIAL_START_ID = tp->GetPieceSize();
        end_of_text_token_id    = SPECIAL_START_ID++;
        start_of_text_token_id  = SPECIAL_START_ID++;
        bos_token_id            = SPECIAL_START_ID++;
        eos_token_id            = SPECIAL_START_ID++;
        pad_token_id            = SPECIAL_START_ID++;

            terminate_ids.insert(end_of_text_token_id);

        for (int i = 0; i < (int)(sizeof(extra_token_ids) / sizeof(extra_token_ids[0])); i++)
            extra_token_ids[i] = SPECIAL_START_ID++;

        return size;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->end_of_text_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
        {
            ids.push_back(tok->start_of_text_token_id);
            tok->encode(tok->get_system_prompt(), ids);
            ids.push_back(tok->extra_token_ids[4]);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        if ((round_idx > 0) || (tok->get_system_prompt().size() == 0))
            ids.push_back(tok->start_of_text_token_id);

        tok->encode(user, ids);
        ids.push_back(tok->extra_token_ids[0]);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int head_dim)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 2 + config.num_hidden_layers * 14;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, nullptr,
                                    config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, config.num_key_value_heads,
                                    head_dim,
                                    config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base = config.rope_theta;
            layer.attention.post_norm = true;
        }
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.hidden_size / config.num_attention_heads)
    {
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();

        transformer->word_embeddings->load("model.embed_tokens.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "self_attn.key_layernorm.weight",  transformer->layers[i].attention.k_layernorm.weight);
            loader.read_tensor(layer_prefix + "self_attn.query_layernorm.weight",  transformer->layers[i].attention.q_layernorm.weight);

            loader.read_tensor(layer_prefix + "input_layernorm.weight",          transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
        }
        transformer->final_layernorm->load("model.norm.", &loader);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }
}

namespace chatllm::hunyuan::dense_v1
{
    struct Config : dense::Config
    {
        int head_dim;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : BaseTokenizer(config, &_chat_encoder)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            hy_User_token_id        = tp->PieceToId("<｜hy_User｜>");
            hy_Assistant_token_id   = tp->PieceToId("<｜hy_Assistant｜>");
            bos_token_id            = tp->PieceToId("<｜hy_begin▁of▁sentence｜>");
            eos_token_id            = tp->PieceToId("<｜hy_place▁holder▁no▁2｜>");

            terminate_ids.insert(eos_token_id);

            tp->OverrideTokenDecoding(tp->PieceToId("<think>"),  "<think>");
            tp->OverrideTokenDecoding(tp->PieceToId("</think>"), "</think>");

            return size;
        }

    public:
        int hy_User_token_id;
        int hy_Assistant_token_id;
    };

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);

        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->hy_User_token_id);
        tok->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->hy_Assistant_token_id);
    }

    class ConditionalGeneration : public dense::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : dense::ConditionalGeneration(config, runtime_config, MODEL_TYPE_HUNYUAN_DENSE_V1, config.head_dim)
        {}
    };
}

namespace chatllm::hunyuan::moe_v1
{
    template <class HunyuanMoEMLP> class HunyuanMoEBlock : public LMBlock1<RMSNorm, dense::HunyuanSelfAttention, RMSNorm, HunyuanMoEMLP>
    {
    public:
        HunyuanMoEBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int mlp_intermediate_size1, int mlp_intermediate_size2,
                int num_kv_heads,
                int head_dim, int max_length)
            : LMBlock1<RMSNorm, dense::HunyuanSelfAttention, RMSNorm, HunyuanMoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
            num_kv_heads, head_dim, max_length)
        {}
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK, class MoEBlock> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, MoEBlock, int, int, int, int, int, int, int, int> ModelClass;
    public:
        GenericConditionalGeneration() = default;

        GenericConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : BaseModelForConditionalGeneration(MODEL_TYPE_HUNYUAN_MOE_V1, config, runtime_config, 4096 * 4),
            config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 2 + config.num_hidden_layers * (15 + 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            Base::transformer = new ModelClass(
                &w_ctx_, config, nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.moe_intermediate_size, config.intermediate_size * config.num_shared_expert,
                config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                config.max_length);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = Base::get_typed_transformer<ModelClass>()->layers[i];
                layer.attention.freq_base = config.rope_theta;
                layer.mlp.mlp1.norm_topk_prob = true;
            }

            w_ctx_.check_used_mem_size(true);
        }

    public:
        Config config;
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class HunyuanSparseMoE : public BaseSparseMLP
    {
    public:
        HunyuanSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
        }
    };

    template <const int NUM_EXPERTS, const int EXPERTS_PER_TOK, const int EFFECTIVE_EXPERTS_PER_TOK> class ClassConditionalGeneration
    {
    public:
        typedef CombinedMLP<HunyuanSparseMoE<NUM_EXPERTS, EXPERTS_PER_TOK>, SiLUMLP> HunyuanMoEMLP;

        typedef HunyuanMoEBlock<HunyuanMoEMLP> MoEBlock;

        class ConditionalGeneration : public GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK, MoEBlock>
        {
        public:
            ConditionalGeneration() = default;

            ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
                : GenericConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK, MoEBlock>(config, runtime_config) {}
        };

        static AbstractModel *create(const Config &config, const RuntimeConfig &runtime_config)
        {
            return new ConditionalGeneration(config, runtime_config);
        }
    };

    namespace experts_64
    {
        const int NUM_EXPERTS                   =  64;
        const int EXPERTS_PER_TOK               =  8;

        // make it easy to test with different number of experts.
        const int EFFECTIVE_EXPERTS_PER_TOK     =  EXPERTS_PER_TOK;

        typedef ClassConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config) : ModelProxy()
    {
        switch (config.num_experts)
        {
        case experts_64::NUM_EXPERTS:
            CHATLLM_CHECK(config.moe_topk == experts_64::EXPERTS_PER_TOK);
            set_proxy_model(experts_64::ConditionalGeneration::create(config, runtime_config));
            break;
        default:
            CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.num_experts;
            break;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        loader.add_tensor_name_translations({
            {".k_norm.",                   ".key_layernorm."},
            {".q_norm.",                   ".query_layernorm."},
            {".mlp.mlp1.gate.",            ".mlp.gate."},
            {".mlp.mlp1.experts.",         ".mlp.experts."},
            {".mlp2.",                     ".shared_expert."},
        });

        ModelProxy::load(loader);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(HUNYUAN_DENSE,         hunyuan::dense, 1);
    REGISTER_MODEL_LOADER(HUNYUAN_DENSE_V1,      hunyuan::dense_v1, 1);
    REGISTER_MODEL_LOADER(HUNYUAN_MOE_V1,        hunyuan::moe_v1, 1);
}