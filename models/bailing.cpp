#include <algorithm>
#include <numeric>
#include <cstring>
#include <functional>
#include "deepseek.h"

namespace chatllm::bailing
{
    enum AdditionalModelType
    {
        MODEL_TYPE_LLADA2 = ModelType::MODEL_TYPE_BAILING_MOE2 + 1,
    };
}

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
        Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder = &_chat_encoder)
            : BaseTokenizer(config, chat_encoder)
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

namespace chatllm::bailing::moe2
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

    class BailingSparseMoE : public GenericGroupedSparseMoE
    {
    public:
        BailingSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size, int num_experts = NUM_EXPERTS, int experts_per_tok = EXPERTS_PER_TOK):
            GenericGroupedSparseMoE(ctx, hidden_size, num_experts, experts_per_tok, true, false, false, false),
            experts(ctx, hidden_size, intermediate_size, num_experts, experts_per_tok, ActFunc::SILU, false)
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

    class AttnParams
    {
    public:
        static int custom_mask;
    };

    int AttnParams::custom_mask = false;

    class SelfAttention : public QKNormedAttention<RMSNorm, BaseAttention>
    {
    public:
        SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length):
            QKNormedAttention<RMSNorm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false),
            mask(nullptr)
        {
            if (AttnParams::custom_mask)
            {
                // reverse some data
                mask = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F16, max_length, 32);
                ctx->get_allocator()->alloc(mask);
            }
        }

        ggml::tensor *attn_scores_to_probs(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
            ggml::tensor *attn_scores) override
        {
            if (nullptr == mask)
            {
                return QKNormedAttention<RMSNorm, BaseAttention>::attn_scores_to_probs(ctx, hidden_size, n_past, qlen, attn_scores);
            }

            const int head_size = hidden_size / num_attention_heads;

            ggml::tensor * sub_mask = ggml::view_2d(ctx, mask, n_past + qlen, qlen, (n_past + qlen) * ggml::element_size(mask), 0);

            // attn_probs = soft_max(attn_masked)
            ggml::tensor * attn_probs = ggml::soft_max_ext(ctx, attn_scores, sub_mask, 1.f / sqrtf((float)head_size), 0.0f);

            return attn_probs;
        }
    public:
        ggml_tensor *mask;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<BailingSparseMoE, SiLUMLP> BailingMoEMLP;
        typedef LMBlock1<RMSNorm, SelfAttention, RMSNorm, BailingMoEMLP> BailingMoEBlock;
        typedef LMBlock1<RMSNorm, SelfAttention, RMSNorm, SiLUMLP> BailingDenseBlock;
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
                                + dense_layer_num * 14
                                + (AttnParams::custom_mask ? config.num_hidden_layers : 0);
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
                    auto layer = new BailingDenseBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
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

        SelfAttention *get_attn_of_layer(int layer_index)
        {
            if (is_layer_moe(layer_index))
            {
                auto layer = dynamic_cast<BailingMoEBlock *>(transformer->get_layer(layer_index));
                return &layer->attention;
            }
            else
            {
                auto layer = dynamic_cast<BailingDenseBlock *>(transformer->get_layer(layer_index));
                return &layer->attention;
            }
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

namespace chatllm::bailing::llada
{
    typedef moe2::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public moe::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : moe::Tokenizer(config, &_chat_encoder)
        {
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = moe::Tokenizer::load(buffer, n_vocab);

            eos_token_id  = tp->PieceToId("<|role_end|>");
            mask_token_id = tp->PieceToId("<|mask|>");

            // LlaDA might generate lots of PAD
            if (mask_token_id >= 0)
                terminate_ids.insert(pad_token_id);

            return r;
        }
    public:
        int mask_token_id;
    };

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        if (tokenizer->get_system_prompt().size() > 0)
        {
            tokenizer->encode("<role>SYSTEM</role>", ids);
            tokenizer->encode(tokenizer->get_system_prompt(), ids);
            ids.push_back(tokenizer->eos_token_id);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);
        tokenizer->encode(ai, ids);
        ids.push_back(tokenizer->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        tokenizer->encode("<role>HUMAN</role>", ids);
        tokenizer->encode(user, ids);
        ids.push_back(tokenizer->eos_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        tokenizer->encode("<role>ASSISTANT</role>", ids);
    }

    class Prelude
    {
    public:
        Prelude()
        {
            moe2::AttnParams::custom_mask = true;
        }
    };

    class ConditionalGeneration : public Prelude, public moe2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_LLADA2):
            Prelude(),
            moe2::ConditionalGeneration(config, runtime_config, type)
        {
            final_steps = dynamic_cast<LMFinalSteps *>(transformer->get_final_steps());
        }
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  BaseStreamer *streamer = nullptr) override;
    protected:
        bool generate_next_block(const int *input_ids, const int ids_count, const GenerationConfig &gen_config,
            std::vector<float> *logits_output, const int batch_size = 1);
        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> *logits_output, const int batch_size);

        void update_mask(int past, int qlen);
    public:
        int block_length    = 32;
        int steps           = 32;
        int minimal_topk    = 1;
        float threshold     = 0.95f;
    protected:
        std::vector<float> mask;
        LMFinalSteps *final_steps;
        std::vector<int> fraction_ids;
    };

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        block_length        = utils::get_opt(args, "block_length", block_length);
        steps               = utils::get_opt(args, "steps", steps);
        minimal_topk        = utils::get_opt(args, "minimal_topk", minimal_topk);
        threshold    = (float)utils::get_opt(args, "threshold", threshold);
        final_steps->set_read_last_n(block_length);
        steps               = std::min(steps, block_length);
    }

    void ConditionalGeneration::update_mask(int past, int qlen)
    {
        CHATLLM_CHECK((past % block_length) == 0);
        CHATLLM_CHECK((qlen % block_length) == 0);

        std::vector<float> mask;
        const int col_num = qlen + past;
        mask.resize(qlen * col_num);
        for (int i = 0; i < qlen / block_length; i++)
        {
            for (int j = 0; j < col_num / block_length; j++)
            {
                const float v = i + (past / block_length) >= j ? 0.0f : -INFINITY;
                for (int ii = 0; ii < block_length; ii++)
                {
                    const int row_index = i * block_length + ii;
                    for (int jj = 0; jj < block_length; jj++)
                    {
                        const int col_index = j * block_length + jj;
                        const int index = row_index * col_num + col_index;
                        mask[index] = v;
                    }
                }
            }
        }

        std::vector<uint8_t> buf;
        buf.resize(ggml::element_size(get_attn_of_layer(0)->mask) * mask.size());
        ggml::from_float(ggml::type_of(get_attn_of_layer(0)->mask), mask.data(), buf.data(), mask.size(), 1);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto m = get_attn_of_layer(i)->mask;
            ggml::set_dim(m, 0, col_num);
            ggml::set_dim(m, 1, qlen);
            Backend::write_tensor_data(m, buf.data(), 0, buf.size());
        }
    }

    bool ConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> *logits_output, const int batch_size)
    {
        CHATLLM_CHECK(batch_size == 1);
        CHATLLM_CHECK((ids_count % block_length) == 0);

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

        update_mask(past, ids_count);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);

        ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

        if (logits_output)
            logits_output->resize(ggml::nelements(r));

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

        ctx.reset();

        return true;
    }

    bool ConditionalGeneration::generate_next_block(const int *input_ids, const int ids_count, const GenerationConfig &gen_config,
        std::vector<float> *logits_output, const int batch_size)
    {
        int batch = batch_input > 1 ? batch_input : 1;
        batch = (batch / block_length) * block_length;
        CHATLLM_CHECK(batch >= block_length);

        const int *p = input_ids;
        int remain = ids_count;
        int past = n_past + n_past_offset;

        for (; (remain > batch) && !aborted; p += batch, remain -= batch, past += batch)
        {
            if (!run_model(p, batch, gen_config, past, nullptr, batch_size))
                return false;
        }

        return run_model(p, remain, gen_config, past, logits_output, batch_size);
    }

    std::vector<int> ConditionalGeneration::generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  BaseStreamer *streamer)
    {
        CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
            << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
            << config_.max_length << ")";

        const int mask_id = dynamic_cast<Tokenizer *>(tokenizer)->mask_token_id;
        std::vector<int> num_transfer_tokens_schedule;
        for (int i = 0; i < steps; i++) num_transfer_tokens_schedule.push_back(block_length / steps);
        const int remain = block_length % steps;
        for (int i = 0; i < remain; i++) num_transfer_tokens_schedule[steps - 1 - i]++;

        GenerationConfig _gen_config(gen_config);
        _gen_config.do_sample = true;
        _gen_config.sampling = "top_p";
        std::unique_ptr<Sampler> sampler = std::unique_ptr<Sampler>(SamplerFactory::Create(_gen_config, get_seed()));

        aborted = false;

        std::vector<int> curr_input_ids(fraction_ids);
        curr_input_ids.insert(curr_input_ids.end(), input_ids.begin(), input_ids.end());
        fraction_ids.clear();

        std::vector<int> output_ids;
        output_ids.reserve(gen_config.max_length);

        if (!continuous)
        {
            n_past = 0;
            n_past_offset = 0;
        }

        CHATLLM_CHECK(n_past_offset == 0) << "shifting not supported yet";

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
        block_result.resize(block_length);
        std::fill(block_result.begin(), block_result.end(), mask_id);

        auto get_num_of_valid_tok = [&block_result, mask_id]() -> int {
            return std::accumulate(block_result.begin(), block_result.end(), 0, [mask_id](const int &acc, const int &id) { return acc + (id != mask_id ? 1 : 0); });
        };

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

        // pre-fill
        {
            const int prefill_block_num = (int)curr_input_ids.size() / block_length;
            const int prefill_len = prefill_block_num * block_length;
            if (prefill_len > 0)
            {
                generate_next_block(curr_input_ids.data(), prefill_len, gen_config, nullptr);
                n_past += prefill_len;
                curr_input_ids.erase(curr_input_ids.begin(), curr_input_ids.begin() + prefill_len);

                if (performance)
                    performance->Accumulate(ModelPerfInfo::Type::Prompt, prefill_len);
            }
        }

        memcpy(block_result.data(), curr_input_ids.data(), curr_input_ids.size() * sizeof(curr_input_ids[0]));
        int next_pos_to_add = (int)(curr_input_ids.size());
        int block_prefilled_size = next_pos_to_add;
        curr_input_ids.clear();

        while (!aborted && !completed)
        {
            for (int step = 0; !completed && (step < steps); step++)
            {
                // Note: we have to run a whole block again and again.
                std::vector<float> lm_logits;
                generate_next_block(block_result.data(), block_length, gen_config, &lm_logits);

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
                for (int i = 0; i < (int)block_result.size(); i++, logits += config_.vocab_size)
                {
                    if (block_result[i] != mask_id) continue;
                    float prob;
                    int next_token_id = sampler->sampling(logits,  config_.vocab_size, &prob);
                    if (prob <= threshold)
                    {
                        candidates.emplace_back(i, next_token_id, logits[next_token_id]);
                        continue;
                    }

                    block_result[i] = next_token_id;
                    transferred += 1;
                }

                int remain = block_length - get_num_of_valid_tok();
                if (remain < 1) break;

                const int num_to_transfer = std::min(num_transfer_tokens_schedule[step] - transferred, remain);
                if (num_to_transfer > 0)
                {
                    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) { return a.prob >= b.prob; });
                    for (int i = 0; i < num_to_transfer; i++)
                    {
                        block_result[candidates[i].pos] = candidates[i].token_id;
                        transferred += 1;
                    }
                }

                for (int i = next_pos_to_add; !completed && (i < block_length); i++)
                {
                    if (mask_id == block_result[i]) break;

                    next_pos_to_add += 1;
                    emit_token(block_result[i]);
                }

                remain = block_length - get_num_of_valid_tok();
                if (remain < 1) break;
            }

            for (int i = next_pos_to_add; !completed && (i < block_length); i++)
            {
                next_pos_to_add += 1;
                emit_token(block_result[i]);
            }

            // block is now finalized
            if (next_pos_to_add == block_length)
            {
                generate_next_block(block_result.data(), next_pos_to_add, gen_config, nullptr);
                n_past += next_pos_to_add;
            }

            if (performance)
                performance->Accumulate(ModelPerfInfo::Type::Generation, block_length - block_prefilled_size);

            if (completed)
                break;

            if ((gen_max_tokens > 0) && (n_past + block_length >= gen_max_tokens))
            {
                aborted = true;
                break;
            }

            next_pos_to_add         = 0;
            block_prefilled_size    = 0;
            std::fill(block_result.begin(), block_result.end(), mask_id);
        }

        if (aborted && !completed)
            completed = true;

        after_generate();

        if (next_pos_to_add != block_length)
        {
            fraction_ids.insert(fraction_ids.end(), block_result.begin(), block_result.begin() + next_pos_to_add);
        }

        return output_ids;
    }
}

namespace chatllm::bailing
{
    REGISTER_MODEL_LOADER(BAILINGMOE,            moe,  1);
    REGISTER_MODEL_LOADER(BAILING_MOE2,          moe2, 1);
    REGISTER_MODEL_LOADER(LLADA2,                llada, 1);
}