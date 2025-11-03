#include "qwen.h"

namespace chatllm::ouro
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_theta;
    };

    typedef qwen::v2::Tokenizer Tokenizer;

    typedef LMBlock4<RMSNorm,
                 LlamaSelfAttention,
                 RMSNorm,
                 RMSNorm,
                 SiLUMLP,
                 RMSNorm> OuroBlock;

    class ModelClass : public HeterogeneousModel
    {
    public:
        struct OneRunResult
        {
            ggml::tensor *hidden_states;
            ggml::tensor *gate_result;

            OneRunResult(ggml::tensor *hidden_states, ggml::tensor *gate_result):
                hidden_states(hidden_states), gate_result(gate_result)
            {}
        };

        ModelClass(InitContext *ctx, const Config &config, int total_ut_steps, float exit_threshold, int max_qlen):
            HeterogeneousModel(ctx, config.num_hidden_layers * total_ut_steps, config.hidden_size,
                create_embedding<Embedding>(ctx, config),
                create_final_norm<RMSNorm>(ctx, config),
                create_lm_head(ctx, config, false),
                [&](InitContext *ctx, int layer_index) {
                    ctx->move_to_layer(layer_index % config.num_hidden_layers);
                    return new OuroBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                config.num_key_value_heads, config.max_length);
                }),
            gate(ctx, config.hidden_size, 1, true),
            config(config),
            total_ut_steps(total_ut_steps), exit_threshold(exit_threshold)
        {
            CHATLLM_CHECK(total_ut_steps >= 1) << "total_ut_steps must >= 1";

            for (int step = 0; step < total_ut_steps; step++)
            {
                for (int i = 0; i < config.num_hidden_layers; i++)
                {
                    auto layer = dynamic_cast<OuroBlock *>(layers[step * config.num_hidden_layers + i]);
                    layer->attention.freq_base    = config.rope_theta;
                }
            }
        }

        int64_t get_param_num_of_layers(bool effective_only) const override
        {
            int64_t r = 0;
            for (int i = 0; i < config.num_hidden_layers; i++)
                r += layers[i]->get_param_num(effective_only);
            if (effective_only) r *= total_ut_steps;
            return r;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = HeterogeneousModel::get_param_num(effective_only);
            r += gate.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids) override
        {
            HeterogeneousModel::load(path, loader, layer_ids);
            gate.load(path + "early_exit_gate.", loader);
        }

        void forward_hidden_states(ComputeContext *ctx, ggml::tensor *input_ids, int n_past,
            std::vector<ggml::tensor *> &step_norms,
            std::vector<ggml::tensor *> &step_confidences)
        {
            before_forward(ctx, input_ids, n_past);
            ctx->move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor* hidden_states = custom_embedding ? custom_embedding(ctx, input_ids) :  word_embeddings->forward(ctx, input_ids);

            const int qlen = ggml::get_dim(hidden_states, 1);

            int layer_id = 0;
            for (int step = 0; step < total_ut_steps; step++)
            {
                for (int i = 0; i < config.num_hidden_layers; i++, layer_id++)
                {
                    auto *layer = layers[layer_id];
                    ctx->move_to_layer(layer->get_id() % config.num_hidden_layers);
                    hidden_states = layer->forward(ctx, hidden_states, n_past);
                }

                ctx->move_to_layer(LayerAllocatorManager::Epilog);
                hidden_states = final_layernorm->forward(ctx, hidden_states);

                auto step_result = ggml::view_1d(ctx, hidden_states, config.hidden_size,
                    (qlen - 1) * ggml::row_size(hidden_states));

                ggml::set_output(step_result);
                step_norms.push_back(step_result);

                auto confidence = gate.forward(ctx, step_result);
                confidence = ggml::sigmoid(ctx, confidence);
                ggml::set_output(confidence);
                step_confidences.push_back(confidence);

                ggml::build_forward_expand(ctx, confidence);
            }
        }

        ggml::tensor *forward_lm_head(ComputeContext *ctx, ggml::tensor *final_norm_result)
        {
            auto hidden_states = final_norm_result;
            const int qlen  = ggml::get_dim(hidden_states, 1);

            ctx->move_to_layer(LayerAllocatorManager::Epilog);
            auto transformer_outputs = ggml::view_2d(ctx, hidden_states, hidden_size, 1,
                    ggml::row_size(hidden_states),
                    (qlen - 1) * ggml::row_size(hidden_states));

            ggml::tensor *lm_logits = lm_head ? lm_head->forward(ctx, transformer_outputs) :
                                                word_embeddings->forward(ctx, transformer_outputs);

            if (logits_pp)
                lm_logits = logits_pp->forward(ctx, lm_logits);

            return lm_logits;
        }
    public:
        Linear gate;
    protected:
        const Config config;
        const int total_ut_steps;
        const float exit_threshold;
    };

    float sigmoid(float v)
    {
        return 1.0f / (1.0f + std::expf(-v));
    }

    class Prelude
    {
    public:
        Prelude()
        {
            BlockParams::Epsilon::rms_norm = 1e-6f;
        }
    };

    class ConditionalGeneration : public Prelude, public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_OURO):
            Prelude(),
            BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 10),
            total_ut_steps(utils::get_opt(runtime_config.additional, "total_ut_steps", 4)),
            exit_threshold((float)utils::get_opt(runtime_config.additional, "exit_threshold", 1.0f))
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 5 + config.num_hidden_layers * 14 * total_ut_steps;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            model = new ModelClass(&w_ctx_, config, total_ut_steps, exit_threshold, runtime_config.batch_input_size);
            transformer = model;

            w_ctx_.check_used_mem_size(true);

            layer_ids.clear();
            for (int i = 0; i < total_ut_steps; i++)
            {
                for (int j = 0; j < config.num_hidden_layers; j++)
                    layer_ids.push_back(j);
            }
        }

        void load(ModelLoader &loader) override
        {
            // pay attention to the order
            loader.add_tensor_name_translations({
                {".pre_attention_layernorm.",           ".input_layernorm."},
                {".post_attention_layernorm.",          ".input_layernorm_2."},
                {".pre_mlp_layernorm.",                 ".post_attention_layernorm."},
                {".post_mlp_layernorm.",                ".post_attention_layernorm_2."},
            });

            BaseModelForConditionalGeneration::load(loader);
        }

        bool run_model_steps(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output)
        {
            ForwardContext ctx(&backend_context);
            ctx.user_options = w_ctx_.user_options;

            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            set_dbg_ctx(&ctx);

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
            ggml::tensor *input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);

            std::vector<ggml::tensor *> step_norms;
            std::vector<ggml::tensor *> step_confidences;
            model->forward_hidden_states(&ctx, input_ids_tensor, past, step_norms, step_confidences);

            if (!ctx.allocate()) return false;

            Backend::write_tensor_data(input_ids_tensor, input_ids);

            ctx.compute();

            std::vector<float> pdf;
            std::vector<float> cdf;
            pdf.resize(total_ut_steps, 0.0f);
            cdf.resize(total_ut_steps, 0.0f);
            float remaining_prob = 1.0f;

            int step = 0;
            for (step = 0; step < total_ut_steps; step++)
            {
                CHATLLM_CHECK(ggml::type_of(step_confidences[step]) == ggml::type::GGML_TYPE_F32);
                CHATLLM_CHECK(ggml::nelements(step_confidences[step]) == 1);

                float lambda_i = 0.0f;
                Backend::read_tensor_data(step_confidences[step], &lambda_i);

                pdf[step] = step < total_ut_steps - 1 ? lambda_i * remaining_prob : remaining_prob;
                remaining_prob -= pdf[step];

                cdf[step] += step > 0 ? pdf[step] + cdf[step - 1] : pdf[step];
                if (cdf[step] >= exit_threshold)
                    break;
            }

            if (step >= total_ut_steps) step = total_ut_steps - 1;

            CHATLLM_CHECK(ggml::type_of(step_norms[step]) == ggml::type::GGML_TYPE_F32);
            output.resize(ggml::nelements(step_norms[step]));
            Backend::read_tensor_data(step_norms[step], output.data());

            ctx.reset();

            return true;
        }

        bool run_final(const GenerationConfig &gen_config,
                       std::vector<float> &output)
        {
            ForwardContext ctx(&backend_context);
            ctx.user_options = w_ctx_.user_options;

            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);
            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

            set_dbg_ctx(&ctx);

            ggml::tensor *final_norm = ggml::new_tensor_1d(&ctx, GGML_TYPE_F32, output.size());

            ggml::tensor *r = model->forward_lm_head(&ctx, final_norm);

            ggml::build_forward_expand(&ctx, r);

            CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

            output.resize(ggml::nbytes(r) / sizeof(output[0]));

            if (!ctx.allocate()) return false;

            Backend::write_tensor_data(final_norm, output.data());

            ctx.compute();

            Backend::read_tensor_data(r, output.data());

            ctx.reset();

            return true;
        }

        bool run_model(const int *input_ids, const int ids_count,
                            const GenerationConfig &gen_config,
                            int past,
                            std::vector<float> &output)
        {
            if (!initial_run)
            {
                initial_run = true;
                int past = gen_config.max_length / transformer->get_reserved_batch_size() - ids_count;
                if (past < 0) past = 0;
                if (!before_initial_run(ids_count, gen_config, past))
                    return false;
            }

            run_model_steps(input_ids, ids_count, gen_config, past, output);

            run_final(gen_config, output);

            return true;
        }

        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override
        {
            int batch = batch_input > 1 ? batch_input : 1;

            const int *p = input_ids.data();
            int remain = (int)input_ids.size();
            int past = n_past + n_past_offset;

            for (; (remain > batch) && !aborted; p += batch, remain -= batch, past += batch)
            {
                if (!run_model(p, batch, gen_config, past, lm_logits))
                    return false;
            }

            return run_model(p, remain, gen_config,past, lm_logits);
        }
    protected:
        ModelClass *model;
        const int total_ut_steps;
        const float exit_threshold;
    };

    REGISTER_MODEL_LOADER(OURO,                ouro, 1);
}