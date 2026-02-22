#include "hunyuan.h"
#include <numeric>
#include "qwen.h"
#include "deepseek.h"
#include "../src/vision_process.h"

namespace chatllm
{
    const int MODEL_TYPE_WEDLM            = 0x1f03;
    const int MODEL_TYPE_YOUTU            = 0x1f04;
}

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

    class ChatHistoryEncoderExtra04 : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder           _chat_encoder;
    static ChatHistoryEncoderExtra04    _chat_encoder_extra04;

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
            if (hy_User_token_id >= 0)
            {
                hy_Assistant_token_id   = tp->PieceToId("<｜hy_Assistant｜>");
                bos_token_id            = tp->PieceToId("<｜hy_begin▁of▁sentence｜>");
                eos_token_id            = tp->PieceToId("<｜hy_place▁holder▁no▁2｜>");

                terminate_ids.insert(eos_token_id);
            }
            else
            {
                extra_0_token_id        = tp->PieceToId("<|extra_0|>");
                extra_4_token_id        = tp->PieceToId("<|extra_4|>");
                set_chat_encoder(&_chat_encoder_extra04);
            }

            tp->OverrideTokenDecoding(tp->PieceToId("<think>"),  "<think>");
            tp->OverrideTokenDecoding(tp->PieceToId("</think>"), "</think>");

            return size;
        }

    public:
        int hy_User_token_id = -1;
        int hy_Assistant_token_id = -1;

        int extra_0_token_id = -1;
        int extra_4_token_id = -1;
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

    void ChatHistoryEncoderExtra04::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);

        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode(tok->get_system_prompt(), ids);
            ids.push_back(tok->extra_4_token_id);
        }
    }

    void ChatHistoryEncoderExtra04::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoderExtra04::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(user, ids);
        ids.push_back(tok->extra_0_token_id);
    }

    void ChatHistoryEncoderExtra04::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
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

namespace chatllm::hunyuan::wedlm
{
    typedef chatllm::qwen::v3::Config Config;
    typedef chatllm::qwen::v3::Tokenizer Tokenizer;

    // entropy algo: https://github.com/Tencent/WeDLM/blob/d4481cab821044b8ebd5f78bc37f23787a6275ed/wedlm/engine/sampler.py#L169
    // prob    algo: https://huggingface.co/tencent/WeDLM-8B-Instruct/blob/main/modeling_wedlm.py#L694
    // custom  algo: sampling + prob

    enum AcceptAlgo
    {
        Entropy,
        Prob,
        Custom,

        NUM
    };

    class FinalPP : public Block
    {
    public:
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input) override
        {
            if (isolated)
            {
                aux_output = nullptr;
                return input;
            }
            if (use_entropy)
            {
                aux_output = ggml::categorical_entropy(ctx, nullptr, input);
                ggml::set_output(aux_output);
            }
            else
            {
                aux_output = temperature > 1e-6f ? ggml::scale(ctx, input, 1.0f / temperature) : input;
                aux_output = ggml::soft_max(ctx, aux_output);
            }
            ggml::build_forward_expand(ctx, aux_output);
            return input;
        }
    public:
        ggml::tensor *aux_output = nullptr;
        float temperature   = 0.0f;
        bool isolated       = false;
        bool use_entropy    = false;
    };

    class TensorPosHelper : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper(int max_length): BaseTensorPosHelper(max_length)
        {}
        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override
        {
            if (isolated)
            {
                BaseTensorPosHelper::prepare_pos_tensor(ctx, pos, n_past, qlen);
                return;
            }
            CHATLLM_CHECK(v_pos.size() == qlen);
            ggml::set_dim(pos, 0, qlen);
            Backend::write_tensor_data(pos, v_pos.data(), 0, qlen * sizeof(v_pos[0]));
        }
    public:
        using BaseTensorPosHelper::v_pos;
        bool isolated = false;
    };

    class ConditionalGeneration : public TensorPosHelperPrelude, public chatllm::qwen::v3::ConditionalGeneration
    {
    public:
        typedef chatllm::qwen::v3::ConditionalGeneration Base;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_WEDLM):
            TensorPosHelperPrelude(new TensorPosHelper(config.max_length)),
            Base(config, runtime_config, type),
            logits_pp()
        {
            pos_helper = (TensorPosHelper *)TensorPosHelperParam::get(config.max_length);
            TensorPosHelperPrelude::done();


            final_steps = dynamic_cast<LMFinalSteps *>(transformer->get_final_steps());
            transformer->logits_pp = &logits_pp;
        }

        bool load_more(const json::JSON &config) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  BaseStreamer *streamer = nullptr) override;
    protected:
        bool generate_next_block(const int *input_ids, const int ids_count, const GenerationConfig &gen_config,
            std::vector<float> *logits_output,
            std::vector<int>   *logits_indices,
            std::vector<float> *aux_output,
            const int batch_size = 1);
        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> *logits_output,
                            std::vector<float> *aux_output,
                            const int read_last_n, const int batch_size);
    protected:
        int   mask_tok_id   = -1;
        int   block_size    = 16;
        float threshold     = 0.7f;
        float pos_penalty_factor = 0.02f;
        AcceptAlgo accept_algo = AcceptAlgo::Custom;
        LMFinalSteps *final_steps;
        FinalPP logits_pp;
        TensorPosHelper *pos_helper;
    };

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        block_size              =        utils::get_opt(args, "block_size", block_size);
        int algo                =        utils::get_opt(args, "accept_algo", (int)accept_algo);
        threshold               = (float)utils::get_opt(args, "threshold", threshold);
        pos_penalty_factor      = (float)utils::get_opt(args, "pos_penalty_factor", pos_penalty_factor);
        pos_helper->isolated    = block_size <= 1;

        if ((0 <= algo) && (algo < AcceptAlgo::NUM))
            accept_algo = (AcceptAlgo)algo;

        logits_pp.use_entropy   = accept_algo == AcceptAlgo::Entropy;
        logits_pp.isolated      = (block_size <= 1) || (accept_algo == AcceptAlgo::Custom);
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        bool r = Base::load_more(config);
        auto id = config["config.json"]["mask_token_id"];
        mask_tok_id = id.IsIntegral() ? (int)id.ToInt() : 151665;
        return r;
    }

    bool ConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> *logits_output,
                            std::vector<float> *aux_output,
                            const int read_last_n, const int batch_size)
    {
        CHATLLM_CHECK(batch_size == 1);

        if (!initial_run)
        {
            initial_run = true;
            int past = gen_config.max_length / transformer->get_reserved_batch_size() - ids_count;
            if (past < 0) past = 0;
            if (!before_initial_run(ids_count, gen_config, past))
                return false;
        }

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;
        LMFinalStepsDisabler disabler(final_steps, logits_output == nullptr);

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        if (logits_output)
        {
            CHATLLM_CHECK(read_last_n <= ids_count);
            final_steps->set_read_last_n(read_last_n);
        }

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);

        ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

        if (logits_output)
            logits_output->resize(ggml::nelements(r));
        if (aux_output && logits_pp.aux_output)
            aux_output->resize(ggml::nelements(logits_pp.aux_output));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, input_ids);

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        ctx.compute();

        if (logits_output)
            Backend::read_tensor_data(r, logits_output->data());
        if (aux_output && logits_pp.aux_output)
            Backend::read_tensor_data(logits_pp.aux_output, aux_output->data());

        ctx.reset();

        return true;
    }

    bool ConditionalGeneration::generate_next_block(const int *input_ids, const int ids_count, const GenerationConfig &gen_config,
        std::vector<float> *logits_output,
        std::vector<int>   *logits_indices,
        std::vector<float> *aux_output,
        const int batch_size)
    {
        int batch = batch_input > 1 ? batch_input : 1;
        CHATLLM_CHECK(batch >= block_size);

        int remain = ids_count;
        int past = n_past + n_past_offset;

        std::vector<int> rearranged_ids;
        std::vector<int> masked_ori_indices; // index in original `input_ids`
        std::vector<int> rearranged_ori_ids;

        for (int i = 0; i < ids_count; i++)
        {
            if (input_ids[i] != mask_tok_id)
            {
                rearranged_ids.push_back(input_ids[i]);
                rearranged_ori_ids.push_back(past + i);
            }
            else
            {
                masked_ori_indices.push_back(i);
            }
        }
        for (int i = 0; i < (int)masked_ori_indices.size(); i++)
        {
            rearranged_ids.push_back(mask_tok_id);
            rearranged_ori_ids.push_back(past + masked_ori_indices[i]);
        }

        if (logits_output)
        {
            if (masked_ori_indices.size() == 0)
                masked_ori_indices.clear();
            CHATLLM_CHECK(masked_ori_indices.size() > 0);
            CHATLLM_CHECK(logits_indices);

            logits_indices->clear();
            logits_indices->insert(logits_indices->end(), masked_ori_indices.begin(), masked_ori_indices.end());
        }
        else
            CHATLLM_CHECK(masked_ori_indices.size() == 0);

        const int *p = rearranged_ids.data();
        const int *p_id = rearranged_ori_ids.data();
        const int eval_size = masked_ori_indices.size() > 0 ? (int)masked_ori_indices.size() : 1;

        while ((remain > eval_size) && !aborted)
        {
            int size = std::min(batch, remain - eval_size);
            pos_helper->v_pos.clear();
            pos_helper->v_pos.insert(pos_helper->v_pos.end(), p_id, p_id + size);

            if (!run_model(p, size, gen_config, past, nullptr, nullptr, -1, batch_size))
                return false;

            p       += size;
            p_id    += size;
            remain  -= size;
            past    += size;
        }

        if (aborted) return false;

        pos_helper->v_pos.clear();
        pos_helper->v_pos.insert(pos_helper->v_pos.end(), p_id, p_id + remain);
        auto r = run_model(p, remain, gen_config, past, logits_output, aux_output, remain, batch_size);

        return r;
    }

    std::vector<int> ConditionalGeneration::generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  BaseStreamer *streamer)
    {
        if (block_size <= 1)
        {
            return Base::generate(input_ids, gen_config, continuous, completed, performance, streamer);
        }

        CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
            << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
            << config_.max_length << ")";

        const int mask_id = mask_tok_id;

        std::unique_ptr<Sampler> sampler =
            std::unique_ptr<Sampler>(SamplerFactory::Create(gen_config));
        logits_pp.temperature = gen_config.temperature;

        aborted = false;

        std::vector<int> curr_input_ids(input_ids);
        curr_input_ids.insert(curr_input_ids.end(), input_ids.begin(), input_ids.end());

        std::vector<int> output_ids;
        output_ids.reserve(gen_config.max_length);

        if (!continuous)
        {
            n_past = 0;
            n_past_offset = 0;
        }

        completed = false;

        transformer->set_ctx((int)curr_input_ids.size());
        int next_output_idx = 0;

        int gen_max_tokens = gen_config.max_new_tokens;
        if (gen_max_tokens > 0)
            gen_max_tokens = n_past + (int)curr_input_ids.size() + gen_max_tokens;

        if (performance)
            performance->Reset();

        before_generate(gen_config);

        std::vector<int> block_result;

        auto emit_token = [&output_ids, &next_output_idx, &completed, this, streamer](int next_token_id) {
            output_ids.push_back(next_token_id);

            int pop_output = 0;
            int keep_idx = 0;
            if (is_output_terminated(output_ids, keep_idx, pop_output))
            {
                while (pop_output-- > 0)
                    output_ids.pop_back();
                keep_idx = (int)output_ids.size();
                completed = true;
            }

            if (streamer)
            {
                if (keep_idx > (int)output_ids.size())
                    keep_idx = (int)output_ids.size();
                for (; next_output_idx < keep_idx; next_output_idx++)
                    streamer->put({output_ids[next_output_idx]});
            }
        };

        //block_result.push_back(curr_input_ids.back());
        int next_pos_to_add = 0;

        // pre-fill
        {
            const int prefill_len = (int)curr_input_ids.size();
            if (prefill_len > 0)
            {
                generate_next_block(curr_input_ids.data(), prefill_len, gen_config, nullptr, nullptr, nullptr);
                curr_input_ids.clear();
                n_past  += prefill_len;

                if (performance)
                    performance->Accumulate(ModelPerfInfo::Type::Prompt, prefill_len);
            }
        }

        while (!aborted && !completed)
        {
            // Note: we have to run a whole block again and again.
            std::vector<float> lm_logits;
            std::vector<int>   logits_indices;
            std::vector<float> aux_output;

            block_result.resize(block_size, mask_id);
            if (next_pos_to_add == (int)block_result.size())
                block_result.resize(block_size * 2, mask_id);
            generate_next_block(block_result.data(), (int)block_result.size(), gen_config, &lm_logits, &logits_indices, &aux_output);

            // tokens that are settled and prefilled
            const int consumed = next_pos_to_add;

            struct candidate
            {
                int pos;
                int token_id;
                float prob;

                candidate(int pos, int token_id, float prob) :
                    pos(pos), token_id(token_id), prob(prob)
                {}
            };

            int transferred = 0;
            std::vector<candidate> candidates;

            float *logits = lm_logits.data();
            for (int i = 0; i < (int)logits_indices.size(); i++, logits += config_.vocab_size)
            {
                int pos = logits_indices[i];
                float prob = 0.0f;
                bool reject = true;
                int next_token_id = -1;
                switch (accept_algo)
                {
                case AcceptAlgo::Entropy:
                    {
                        const float adjusted_entropy = aux_output[i] + pos_penalty_factor * (pos - next_pos_to_add);
                        reject = adjusted_entropy >= threshold;
                        prob = -adjusted_entropy;
                        next_token_id = sampler->sampling(logits,  config_.vocab_size, &prob);
                    }
                    break;

                case AcceptAlgo::Prob:
                    {
                        const float *probs = &aux_output[(int64_t)i * config_.vocab_size];
                        auto max_ele = std::max_element(probs, probs + config_.vocab_size);
                        prob = *max_ele;
                        reject = prob <= threshold;
                        next_token_id = (int)(max_ele - probs);
                    }
                default:
                    {
                        next_token_id = sampler->sampling(logits,  config_.vocab_size, &prob);
                        reject = prob <= threshold;
                    }
                    break;
                }

                if (reject)
                {
                    candidates.emplace_back(pos, next_token_id, prob);
                    continue;
                }

                block_result[pos] = next_token_id;
                transferred += 1;
            }

            if (transferred < 1)
            {
                auto m = std::max_element(candidates.begin(), candidates.end(), [](const candidate& a, const candidate& b) { return a.prob < b.prob; });

                int pos = m->pos;

                block_result[pos] = m->token_id;
                transferred = 1;
            }

            if (performance)
                performance->Accumulate(ModelPerfInfo::Type::Generation, transferred);

            for (int i = next_pos_to_add; !completed && (i < (int)block_result.size()); i++)
            {
                if (mask_id == block_result[i]) break;

                next_pos_to_add += 1;
                emit_token(block_result[i]);
            }

            // these tokens are already consumed
            int settle = consumed;
            block_result.erase(block_result.begin(), block_result.begin() + settle);
            next_pos_to_add -= settle;
            n_past  += settle;

            if (completed)
                break;

            if ((gen_max_tokens > 0) && (n_past + block_size > gen_max_tokens))
            {
                aborted = true;
                break;
            }
        }

        if (aborted && !completed)
            completed = true;

        if (next_pos_to_add > 0)
        {
            // finalize these tokens
            generate_next_block(block_result.data(), next_pos_to_add, gen_config, nullptr, nullptr, nullptr);
        }

        after_generate();

        return output_ids;
    }
}

namespace chatllm::hunyuan::youtu::llm
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int kv_lora_rank;
        int q_lora_rank;
        int qk_nope_head_dim;
        int qk_rope_head_dim;
        int v_head_dim;
        int tie_word_embeddings;

        float rope_theta;
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
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

    public:
        int user_token_id;
        int assistant_token_id;
    };

    Tokenizer::Tokenizer(const Config &config):
        BaseTokenizer(config, &_chat_encoder)
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
        tp->Encode("<|User|><|Assistant|><think></think>", &ids);

        CHATLLM_CHECK(ids.size() == 4) << "tokenizer error";

        user_token_id = ids[0];
        assistant_token_id = ids[1];
        tp->OverrideTokenDecoding(ids[2], "<think>");
        tp->OverrideTokenDecoding(ids[3], "</think>");
        return size;
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids);
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
        ids.push_back(tok->user_token_id);
        tok->encode(user, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
    }

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    protected:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, deepseek::v2_light::DeepSeek2Block, int, int, int, int, int, int, int, int, int, int, bool> ModelClass;
    public:

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_YOUTU)
            : ConditionalGeneration(config, runtime_config, type, config.q_lora_rank)
        {}

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int q_lora_rank)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
              config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 2 + config.num_hidden_layers * 17;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config, nullptr,
                                        config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                        config.num_key_value_heads, config.max_length,
                                        q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                        false);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto layer = &get_typed_transformer<ModelClass>()->layers[i];
                layer->attention.freq_base = config.rope_theta;
                layer->attention.rope_mode = RoPEMode::Interleaved;
                layer->attention.set_prec(ggml::prec::GGML_PREC_F32);
            }

            w_ctx_.check_used_mem_size(true);
        }

    public:
        Config config;
    };
}

namespace chatllm::hunyuan::youtu::vit
{
    const int VIT_MAX_LAYERS   = 128;

    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int max_patches;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int out_hidden_size;
        int spatial_merge_size;
        int window_size;
        int tokens_per_second;
        int num_channels;

        float image_mean[3];
        float image_std[3];

        bool fullatt_block_indices[VIT_MAX_LAYERS];
    };

    class PatchEmbedding : public Linear
    {
    public:
        PatchEmbedding(InitContext *ctx, const Config &config);
        void load(const std::string &path, TensorLoader *loader) override;
    };

    PatchEmbedding::PatchEmbedding(InitContext *ctx, const Config &config):
        Linear(ctx, config.patch_size * config.patch_size * config.num_channels, config.hidden_size)
    {
    }

    void PatchEmbedding::load(const std::string &path, TensorLoader *loader)
    {
        Linear::load(path + "patch_embedding.", loader);
    }

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, int hidden_size, int spatial_merge_size, int lm_hidden_size):
            hidden_size(hidden_size * spatial_merge_size * spatial_merge_size),
            pre_norm(ctx, hidden_size),
            mlp(ctx, this->hidden_size, this->hidden_size, lm_hidden_size)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features, int grid_h, int grid_w) override
        {
            auto output = pre_norm.forward(ctx, image_features);
            output = ggml::reshape(ctx, output, hidden_size, -1);
            output = mlp.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += pre_norm.get_param_num(effective_only);
            r +=      mlp.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            pre_norm.load(path + "ln_q.",  loader);
             mlp.fc0.load(path + "mlp.0.", loader);
             mlp.fc1.load(path + "mlp.2.", loader);
        }

    public:
        const int hidden_size;
        RMSNorm             pre_norm;
        qwen::vit::MLP      mlp;
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<LayerNorm, qwen::vit::ViTSelfAttention, LayerNorm, qwen::vit::MLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override;
        bool is_loaded(void) const;
    public:
        const Config config;
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        MultiModalProjector multi_modal_projector;
        LayerNorm post_layernorm;
        qwen::vit::TensorPosHelper pos_helper;
        ggml::tensor *window_id;
        ggml::tensor *reverse_id;
    protected:
        bool loaded = false;
    };

    VisionTransformer::VisionTransformer(InitContext *ctx, const Config &config):
        Block(),
        config(config),
        embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config),
        multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.spatial_merge_size, config.out_hidden_size),
        post_layernorm(ctx, config.hidden_size),
        pos_helper(config.max_patches, config.window_size, config.patch_size, config.spatial_merge_size),
        window_id(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, config.max_patches)),
        reverse_id(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, config.max_patches))
    {
        const int max_length = config.max_patches;

        ctx->get_allocator()->alloc(window_id);
        ctx->get_allocator()->alloc(reverse_id);

        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
        {
            qwen::vit::ViTParams::set_full_attention(config.fullatt_block_indices[layer_id]);
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
            layer->set_id(layer_id);
            layer->attention.set_pos_helper(&pos_helper);
            layers.emplace_back(layer);
        }
    }

    int64_t VisionTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embeddings.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        r += post_layernorm.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void VisionTransformer::load(const std::string &path, TensorLoader *loader)
    {
        if (!loader->has_tensor(path + "embeddings.patch_embedding.weight")) return;

        embeddings.load(path + "embeddings.", loader);
        multi_modal_projector.load("merger.", loader);
        post_layernorm.load(path + "post_layernorm.", loader);
        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        loaded = true;
    }

    ggml::tensor *VisionTransformer::forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
    {
        pos_helper.prepare(grid_h, grid_w);
        pos_helper.write_mapping_tensors(ctx, window_id, reverse_id);

        auto output = embeddings.forward(ctx, input);

        output = ggml::get_rows(ctx, output, window_id);

        for (size_t i = 0; i < layers.size(); i++)
        {
            auto l = layers[i].get();
            if (l->attention.mask)
                pos_helper.write_mask_tensor(ctx, l->attention.mask);
            auto attn = &l->attention;

            attn->grid_h = grid_h;
            attn->grid_w = grid_w;
            output = l->forward(ctx, output, 0);
        }

        output = post_layernorm.forward(ctx, output);
        output = multi_modal_projector.forward(ctx, output, grid_h, grid_w);
        output = ggml::get_rows(ctx, output, reverse_id);
        return output;
    }

    bool VisionTransformer::is_loaded(void) const
    {
        return loaded;
    }
}

namespace chatllm::hunyuan::youtu::vl
{
    typedef llm::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    public:
        const vit::Config *vis_config;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count);
    public:
        void append_role(const std::string &role, std::vector<int> &ids);
        int vision_start_token_id;
        int vision_end_token_id;
    };

    Tokenizer::Tokenizer(const Config &config):
        BaseTokenizer(config, &_chat_encoder)
    {
        sys_prompt = "You are a helpful assistant.";
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

        tp->EnableReturnSpecialToken(true);

        vision_start_token_id = tp->PieceToId("<|vision_start|>");
        vision_end_token_id   = tp->PieceToId("<|vision_end|>");
        return size;
    }

    void Tokenizer::append_role(const std::string &role, std::vector<int> &ids)
    {
        ids.push_back(bos_token_id);
        encode(role + "\n", ids);
    }

    void Tokenizer::inject_media(const std::string &media_type, std::vector<int> &ids, const int ids_to_inject_start, const int ids_to_inject_count)
    {
        ids.push_back(vision_start_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(vision_end_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->append_role("system", ids);
        tok->encode(tok->get_system_prompt(), ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->append_role("assistant", ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->append_role("user", ids);
        tok->encode(user, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->append_role("assistant", ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->append_role("user", ids);

        tok->media_emb.clear();

        std::unique_ptr<vision::Resize> resize;

        for (auto &piece : user.pieces)
        {
            if (piece.type != ContentPiece::Type::Image) continue;

            CHATLLM_CHECK(vis_config) << "Vision model not loaded";

            int w, h;
            std::vector<uint8_t> pixels;
            const int patch_size = vis_config->patch_size;

            vision::MaxPatchNum     param1(vis_config->max_patches);
            vision::MergeKernel     param2(vis_config->spatial_merge_size, vis_config->spatial_merge_size);

            vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);
            if ((w <= 0) || (h <= 0)) continue;

            std::vector<float> scaled;
            vision::image_rescale(pixels, scaled);

            vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

            tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

            auto &image = tok->media_emb.back();

            vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::PatchesLeftRightDown_MergeN_PixelsLeftRightDown_ChannelsRGB);

            const int merge_length = vis_config->spatial_merge_size * vis_config->spatial_merge_size;
            image.emb_vec_number = image.grid_width * image.grid_height / merge_length;

            const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
            tok->inject_media("image", ids, id_start, image.emb_vec_number);
        }

        for (auto &piece : user.pieces)
        {
            if (piece.type != ContentPiece::Type::Video) continue;

            CHATLLM_THROW << "TODO: content type: " << (int)piece.type;
        }

        for (auto &piece : user.pieces)
        {
            if (piece.type != ContentPiece::Type::Text) continue;

            tok->encode(piece.content, ids);
        }

        ids.push_back(tok->eos_token_id);
    }

    class ExtendEmbedding
    {
    public:
        ExtendEmbedding(int n) : pad_arg(new BlockParams::PadEmbedding(n, n)) {}
    public:
        BlockParams::PadEmbedding *pad_arg = nullptr;
    };

    class ConditionalGeneration : public ExtendEmbedding, public llm::ConditionalGeneration
    {
    public:
        typedef llm::ConditionalGeneration Base;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_YOUTU_VL);
        bool load_more(const json::JSON &config) override;
        void load(ModelLoader &loader) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void before_generate(const GenerationConfig &gen_config) override;
        int64_t get_param_num(bool effective_only) const;
    protected:
        const int max_patches;
        std::unique_ptr<vit::VisionTransformer> vision;
        TensorGraphEvaluator vis_eval;
        InitContext vis_ctx;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type):
        ExtendEmbedding(14400),
        Base(config, runtime_config, type),
        max_patches(pad_arg->get()),
        vis_eval(runtime_config, "vis"),
        vis_ctx(&backend_context)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON &config)
    {
        Base::load_more(config);

        const auto vis_cfg = config["config.json"]["vision_config"];
        if (!vis_cfg.IsObject()) return false;

        vit::Config vis_config;
        memset(&vis_config, 0, sizeof(vis_config));

        vis_config.dtype = this->config.dtype;

        vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
        vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
        vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
        vis_config.spatial_merge_size   = 2;

        vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
        vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
        vis_config.window_size          = (int)vis_cfg["window_size"].ToInt();
        vis_config.tokens_per_second    = (int)vis_cfg["tokens_per_second"].ToInt();
        vis_config.out_hidden_size      = (int)vis_cfg["out_hidden_size"].ToInt();
        vis_config.num_channels         = (int)vis_cfg["num_channels"].ToInt();

        CHATLLM_CHECK(vis_config.out_hidden_size == this->config.hidden_size);
        CHATLLM_CHECK(vis_config.num_channels    == 3);

        auto indices = vis_cfg["fullatt_block_indexes"];
        int full_cnt = 0;
        CHATLLM_CHECK((int)indices.length() <= vit::VIT_MAX_LAYERS);
        for (int i = 0; i < (int)indices.length(); i++)
        {
            vis_config.fullatt_block_indices[indices[i].ToInt()] = true;
            full_cnt += 1;
        }

        auto pp_cfg = config["preprocessor_config.json"];
        if (pp_cfg.IsObject())
        {
            auto image_mean = pp_cfg["image_mean"];
            auto image_std  = pp_cfg["image_std"];
            CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
            CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

            vis_config.image_mean[0]    = (float)image_mean[0].ToFloat();
            vis_config.image_mean[1]    = (float)image_mean[1].ToFloat();
            vis_config.image_mean[2]    = (float)image_mean[2].ToFloat();
            vis_config.image_std[0]     = (float)image_std[0].ToFloat();
            vis_config.image_std[1]     = (float)image_std[1].ToFloat();
            vis_config.image_std[2]     = (float)image_std[2].ToFloat();

            vis_config.max_patches      = max_patches;
        }

        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 11 + vis_config.num_hidden_layers * 18 - full_cnt;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        vis_ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        vis_ctx.dtype = vis_config.dtype;

        vision.reset(new vit::VisionTransformer(&vis_ctx, vis_config));

        vis_ctx.check_used_mem_size(true);

        return true;
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {".input_layernorm.",           ".layer_norm1."},
            {".post_attention_layernorm.",  ".layer_norm2."},
        });

        if (vision.get() == nullptr) return;

        vision->load("visual.", &loader);
        if (vision->is_loaded())
        {
            _chat_encoder.vis_config = &vision->config;
        }
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
    }

    int64_t ConditionalGeneration::get_param_num(bool effective_only) const
    {
        int64_t r = Base::get_param_num(effective_only);
        if (vision.get() != nullptr)       r += vision->get_param_num(effective_only);
        return r;
    }

    void ConditionalGeneration::before_generate(const GenerationConfig &gen_config)
    {
        std::vector<uint8_t> buf;
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (vision.get() == nullptr) return;
        if (!vision->is_loaded()) return;

        auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
        const vit::Config vis_config = vision->config;

        for (auto &image : tok->media_emb)
        {
            ggml::tensor *media_emb = nullptr;
            const auto make_graph = [this, &media_emb, &image, &vis_config](ComputeContext *ctx) -> ggml::tensor * {
                media_emb = ggml::new_tensor_2d(ctx,
                    ggml::type::GGML_TYPE_F32, vis_config.patch_size * vis_config.patch_size * 3, image.grid_width * image.grid_height);
                auto r = vision->forward(ctx, media_emb, image.grid_height, image.grid_width);
                return r;
            };
            const auto write_input_data = [&media_emb, &image](ComputeContext *ctx) {
                Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            };

            std::vector<int64_t> shape;
            vis_eval.evaluate(gen_config, make_graph, write_input_data, ggml::type_of(emb->weight),
                shape, buf);
        }

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(HUNYUAN_DENSE,         hunyuan::dense, 1);
    REGISTER_MODEL_LOADER(HUNYUAN_DENSE_V1,      hunyuan::dense_v1, 1);
    REGISTER_MODEL_LOADER(HUNYUAN_MOE_V1,        hunyuan::moe_v1, 1);
    REGISTER_MODEL_LOADER(WEDLM,                 hunyuan::wedlm, 1);
    REGISTER_MODEL_LOADER(YOUTU,                 hunyuan::youtu::llm, 1);
    REGISTER_MODEL_LOADER(YOUTU_VL,              hunyuan::youtu::vl, 1);
}