#include <algorithm>
#include <numeric>
#include <cstring>
#include <functional>
#include "deepseek.h"
#include "qwen.h"

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
        void append_sys_prompt(std::vector<int> &ids) const override;
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
            eos_token_id  = tp->PieceToId("<|role_end|>");
            mask_token_id = tp->PieceToId("<|mask|>");

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
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_BAILING_MOE2, bool causal = true)
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
                    layer->attention.causal = causal;
                    config_rope(layer->attention);
                    return layer;
                }
                else
                {
                    auto layer = new qwen::v3::QWen3Block(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.head_dim, config.max_length);
                    layer->attention.causal = causal;
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

namespace chatllm::bailing::llada
{
    typedef moe2::Config Config;
    typedef moe2::Tokenizer Tokenizer;

    class ConditionalGeneration : public moe2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_LLADA2):
            moe2::ConditionalGeneration(config, runtime_config, type, false)
        {
            final_steps = dynamic_cast<LMFinalSteps *>(transformer->get_final_steps());
        }
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  int gen_max_tokens,
                                  BaseStreamer *streamer = nullptr) override;
    protected:
        bool generate_next_block(const int *input_ids, const int ids_count, const GenerationConfig &gen_config,
            std::vector<float> *logits_output,
            std::vector<int>   *logits_orderring, const int batch_size = 1);
        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> *logits_output,
                            std::vector<int>   *logits_orderring, const int batch_size);
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

    bool ConditionalGeneration::run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> *logits_output,
                            std::vector<int>   *logits_orderring, const int batch_size)
    {
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

        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor *input_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ids_count, batch_size);

        final_steps->set_do_orderring(logits_orderring != nullptr);
        ggml::tensor *r = transformer->forward(&ctx, input_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (logit_scale > 0)
            r = ggml::scale(&ctx, r, logit_scale);

        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

        if (logits_output)
            logits_output->resize(ggml::nelements(r));
        if (logits_orderring)
            logits_orderring->resize(ggml::nelements(final_steps->get_orderring_result()));

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
        if (logits_orderring)
            Backend::read_tensor_data(final_steps->get_orderring_result(), logits_orderring->data());

        ctx.reset();

        return true;
    }

    bool ConditionalGeneration::generate_next_block(const int *input_ids, const int ids_count, const GenerationConfig &gen_config,
        std::vector<float> *logits_output,
        std::vector<int>   *logits_orderring, const int batch_size)
    {
        int batch = batch_input > 1 ? batch_input : 1;
        batch = (batch / block_length) * block_length;
        CHATLLM_CHECK(batch >= block_length);

        const int *p = input_ids;
        int remain = ids_count;
        int past = n_past + n_past_offset;

        for (; (remain > batch) && !aborted; p += batch, remain -= batch, past += batch)
        {
            if (!run_model(p, batch, gen_config, past, nullptr, nullptr, 1))
                return false;
        }

        return run_model(p, remain, gen_config, past, logits_output, logits_orderring, 1);
    }

    std::vector<int> ConditionalGeneration::generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  int gen_max_tokens,
                                  BaseStreamer *streamer)
    {
        CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
            << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
            << config_.max_length << ")";

        const int mask_id = dynamic_cast<bailing::moe::Tokenizer *>(tokenizer)->mask_token_id;
        std::vector<int> num_transfer_tokens_schedule;
        for (int i = 0; i < steps; i++) num_transfer_tokens_schedule.push_back(block_length / steps);
        const int remain = block_length % steps;
        for (int i = 0; i < remain; i++) num_transfer_tokens_schedule[steps - 1 - i]++;

        GenerationConfig _gen_config(gen_config);
        _gen_config.do_sample = true;
        _gen_config.sampling = "top_p";
        std::unique_ptr<Sampler> sampler = std::unique_ptr<Sampler>(SamplerFactory::Create(_gen_config, _seed));

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
                generate_next_block(curr_input_ids.data(), prefill_len, gen_config, nullptr, nullptr);
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
                generate_next_block(block_result.data(), block_length, gen_config, &lm_logits, nullptr);

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
                    int next_token_id = sampler->sampling(logits,  config_.vocab_size);
                    if (logits[next_token_id] <= threshold)
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
                generate_next_block(block_result.data(), next_pos_to_add, gen_config, nullptr, nullptr);
            n_past += next_pos_to_add;

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