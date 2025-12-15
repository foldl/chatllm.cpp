#include "mistral.h"
#include "../src/vision_process.h"

namespace chatllm::mistral::mistral
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : llama::v2::Tokenizer::Tokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = llama::v2::Tokenizer::load(buffer, n_vocab);
        if (tp->GetPieceSize() == 32768)
        {
            // Mistral v0.3
            start_inst_token_id             = 3;
            end_inst_token_id               = 4;
            tool_calls_token_id             = 5;
            start_avail_tools_token_id      = 6;
            end_avail_tools_token_id        = 7;
            start_tool_results_token_id     = 8;
            end_tool_results_token_id       = 9;
            tp->AddAddedToken("[INST]",             start_inst_token_id);
            tp->AddAddedToken("[/INST]",            end_inst_token_id);
            tp->AddAddedToken("[TOOL_CALLS]",       tool_calls_token_id);
            tp->AddAddedToken("[AVAILABLE_TOOLS]",  start_avail_tools_token_id);
            tp->AddAddedToken("[/AVAILABLE_TOOLS]", end_avail_tools_token_id);
            tp->AddAddedToken("[TOOL_RESULTS]",     start_tool_results_token_id);
            tp->AddAddedToken("[/TOOL_RESULTS]",    end_tool_results_token_id);
        }
        else
        {
            start_inst_token_id             = tp->PieceToId("[INST]");
            end_inst_token_id               = tp->PieceToId("[/INST]");
            tool_calls_token_id             = tp->PieceToId("[TOOL_CALLS]");
            start_avail_tools_token_id      = tp->PieceToId("[AVAILABLE_TOOLS]");
            end_avail_tools_token_id        = tp->PieceToId("[/AVAILABLE_TOOLS]");
            start_tool_results_token_id     = tp->PieceToId("[TOOL_RESULTS]");
            end_tool_results_token_id       = tp->PieceToId("[/TOOL_RESULTS]");
        }
        return r;
    }

    class MistralInterceptor : public ChunkInterceptor
    {
    public:
        MistralInterceptor() : ChunkInterceptor(), found_tool_call(false)
        {}

        void put_chunk(bool first, const std::string &chunk) override
        {
            Tokenizer *tok = dynamic_cast<Tokenizer *>(streamer->tokenizer);
            if (tok->start_avail_tools_token_id < 0)
            {
                next->put_chunk(first, chunk);
                return;
            }

            if (first)
            {
                found_tool_call = chunk.starts_with("[TOOL_CALLS]");
                if (found_tool_call) return;
            }

            if (found_tool_call)
                oss << chunk;
            else
                next->put_chunk(first, chunk);
        }

        void end() override
        {
            if (found_tool_call)
                streamer->putln(oss.str(), BaseStreamer::TextType::TOOL_CALLING);

            oss.str("");
            found_tool_call = false;

            ChunkInterceptor::end();
        }
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };

    static MistralInterceptor interceptor;

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : llama::v2::GenericConditionalGeneration<MistralBlock<SLIDING_WINDOW_LEN>>(config, runtime_config, type,
            config.num_key_value_heads, config.max_length, 13)
    {
        CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == SLIDING_WINDOW_LEN))
            << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        batch_input = false;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : ConditionalGeneration(config, runtime_config, MODEL_TYPE_MISTRAL)
    {
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &interceptor;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true);
    }

    void ChatHistoryEncoder::append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[TOOL_RESULTS]" << content << "[/TOOL_RESULTS]";

        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if ((tok->start_system_prompt_token_id >= 0) && (tok->get_system_prompt().size() > 0))
        {
            ids.push_back(tok->start_system_prompt_token_id);
            tok->encode(tok->get_system_prompt(), ids, false, false);
            ids.push_back(tok->end_system_prompt_token_id);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        if (tok->start_avail_tools_token_id >= 0)
        {
            if (user.starts_with("[TOOL_RESULTS]"))
            {
                oss_prompt << user;
            }
            else
            {
                std::string user_input = user;
                const std::string tag_tools = "[/AVAILABLE_TOOLS]";
                size_t pos = user.find(tag_tools);
                if (pos != std::string::npos)
                {
                    oss_prompt <<  user.substr(0, pos + tag_tools.size());
                    user_input = user.substr(pos + tag_tools.size());
                }

                oss_prompt << "[INST] " << user_input << " [/INST]";
                tok->encode(oss_prompt.str(), ids, false, false);
            }
        }
        else
        {
            oss_prompt << "[INST] " << user << " [/INST]";
        }

        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }
}

namespace chatllm::mistral::mixtral
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

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : mistral::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
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
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << "[INST] " << user << " [/INST]";
        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }
}

namespace chatllm::mistral::mistral2
{
    Tokenizer::Tokenizer(const BaseConfig &config)
        : mistral::Tokenizer(config, &mistral::_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : mistral::Tokenizer(config, encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);

        start_inst_token_id             = tp->PieceToId("[INST]");
        end_inst_token_id               = tp->PieceToId("[/INST]");
        tool_calls_token_id             = tp->PieceToId("[TOOL_CALLS]");
        start_avail_tools_token_id      = tp->PieceToId("[AVAILABLE_TOOLS]");
        end_avail_tools_token_id        = tp->PieceToId("[/AVAILABLE_TOOLS]");
        start_tool_results_token_id     = tp->PieceToId("[TOOL_RESULTS]");
        end_tool_results_token_id       = tp->PieceToId("[/TOOL_RESULTS]");
        start_system_prompt_token_id    = tp->PieceToId("[SYSTEM_PROMPT]");
        end_system_prompt_token_id      = tp->PieceToId("[/SYSTEM_PROMPT]");
        return size;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : llama::v2::GenericConditionalGeneration<MistralBlock<mistral::SLIDING_WINDOW_LEN>>(config, runtime_config, type,
            config.num_key_value_heads, config.head_dim, config.max_length, 13, false)
    {
        CHATLLM_CHECK((config.sliding_window <= 0) || (config.sliding_window == mistral::SLIDING_WINDOW_LEN))
            << "sliding_window (" << config.sliding_window << ") must be " << mistral::SLIDING_WINDOW_LEN;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass2>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        batch_input = false;
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &mistral::interceptor;
    }
}

namespace chatllm::mistral::pixtral
{
    struct Config
    {
        ggml::type dtype;
        int patch_size;
        int max_length;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int longest_edge;
        int spatial_merge_size;

        float rope_theta;
        float image_mean[3];
        float image_std[3];
    };

    class PatchEmbedding : public Block
    {
    public:
        PatchEmbedding(InitContext *ctx, int out_dim, int patch_size, int in_dim = 3)
            : proj(ctx, in_dim, out_dim, patch_size, patch_size, 0, 1, 1, false),
              ln_pre(ctx, out_dim)
        {
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            ggml::tensor *x = proj.forward(ctx, input);
            x = ggml::reshape(ctx, x, ggml::get_dim(x, 0) * ggml::get_dim(x, 1), ggml::get_dim(x, 2), ggml::get_dim(x, 3));
            x = ggml::transpose(ctx, x);
            x = ln_pre.forward(ctx, x);
            x = ggml::reshape_2d(ctx, x, ggml::get_dim(x, 0), grid_h * grid_w);
            return x;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += proj.get_param_num(effective_only);
            r += ln_pre.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            proj.load  (path + "patch_conv.", loader);
            ln_pre.load(path + "ln_pre.", loader);
        }

    public:
        Conv2D                  proj;
        RMSNorm                 ln_pre;
    };

    class TensorPosHelper
    {
    public:
        TensorPosHelper(int max_length, int patch_size)
            : max_length(max_length * 4), original_length(max_length), patch_size(patch_size)
        {
            v_pos.resize(this->max_length);
        }

        ggml::tensor *allocate_pos_tensor(InitContext *ctx)
        {
            ggml::tensor *r = ggml::new_tensor_1d(ctx, GGML_TYPE_I32, max_length);
            ctx->get_allocator()->alloc(r);
            return r;
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen)
        {
            ggml::set_dim(pos, 0, length * 4);
            Backend::write_tensor_data(pos, v_pos.data(), 0, length * 2 * sizeof(v_pos[0]));
        }

        void prepare(int grid_h, int grid_w)
        {
            length = grid_h * grid_w;

            v_pos.clear();
            v_pos.resize(length * 2, 0);
            int *p_h = &v_pos[length * 0];
            int *p_w = &v_pos[length * 1];
            int index = 0;

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    p_w[index] = j;
                    p_h[index] = i;
                    index++;
                }
            }
        }
    public:
        const int max_length;
        const int original_length;
        const int patch_size;
        int length;
        std::vector<int> v_pos;
    };

    class TensorPosHelper2D : public BaseTensorPosHelper
    {
    public:
        TensorPosHelper2D(int max_length)
            : BaseTensorPosHelper(max_length * 4)
        {
        }

        void prepare_pos_tensor(ComputeContext *ctx, ggml::tensor *pos, const int n_past, const int qlen) override
        {
            helper->prepare_pos_tensor(ctx, pos, n_past, qlen);
        }
    public:
        TensorPosHelper *helper;
    };

    class ViTSelfAttention : public TensorPosHelperPrelude, public RoPESelfAttention<BaseCachelessAttention>
    {
    public:
        ViTSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : TensorPosHelperPrelude(new TensorPosHelper2D(max_length)),
            RoPESelfAttention<BaseCachelessAttention>(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, false, false)
        {
            TensorPosHelperPrelude::done();
            causal = false;
            CHATLLM_CHECK(rope_dim % 4 == 0);
        }
        void set_pos_helper(TensorPosHelper *helper)
        {
            TensorPosHelper2D *h = (TensorPosHelper2D *)pos_helper.get();
            h->helper = helper;
        }
    protected:
        ggml::tensor *apply_2d_rope(ComputeContext *ctx, ggml::tensor *hidden, int hidden_size) const
        {
            // ggml shape of hidden: [head_size, heads, qlen]
            int sections[4] = {rope_dim / 4, rope_dim / 4, 0, 0};

            hidden = ggml::rope_ext_inplace(ctx, hidden, pos, freq_factors, rope_dim / 2, GGML_ROPE_TYPE_VISION, n_original_ctx,
                                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
                                sections);

            return hidden;
        }
        ggml::tensor *apply_pos_embedding_k(ComputeContext *ctx, ggml::tensor *k, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            k = apply_2d_rope(ctx, k, hidden_size);
            return k;
        }
        ggml::tensor *apply_pos_embedding_q(ComputeContext *ctx, ggml::tensor *q, int hidden_size, int qlen, ggml::tensor * past) const override
        {
            q = apply_2d_rope(ctx, q, hidden_size);
            return q;
        }
    public:
        int grid_w = 0;
        int grid_h = 0;
    };

    class MultiModalProjector : public Block
    {
    public:
        MultiModalProjector(InitContext *ctx, const Config &config, int lm_hidden_size, int num_feature_layers = 1)
            :
            hidden_size(config.hidden_size * config.spatial_merge_size * config.spatial_merge_size),
            norm(ctx, config.hidden_size),
            linear_1(ctx, config.hidden_size * num_feature_layers, lm_hidden_size, false),
            linear_2(ctx, lm_hidden_size, lm_hidden_size, false),
            patch_merger(ctx, hidden_size, config.hidden_size, false)
        {
            CHATLLM_CHECK(num_feature_layers == 1);
            merge_param.merge_kernel_size[0] = merge_param.merge_kernel_size[1] = config.spatial_merge_size;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *image_features) override
        {
            const int merge_size = merge_param.merge_kernel_size[0] * merge_param.merge_kernel_size[1];
            auto output = norm.forward(ctx, image_features);
            output = ggml::merge_patch(ctx, output, &merge_param);
            output = ggml::reshape(ctx, output, hidden_size / merge_size, merge_size, -1);
            output = ggml::transpose(ctx, output);
            output = ggml::reshape(ctx, output, hidden_size, -1);
            output = patch_merger.forward(ctx, output);
            output = linear_1.forward(ctx, output);
            output = ggml::act(ctx, ActFunc::GELU, output);
            output = linear_2.forward(ctx, output);
            return output;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += norm.get_param_num(effective_only);
            r += linear_1.get_param_num(effective_only);
            r += linear_2.get_param_num(effective_only);
            r += patch_merger.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
                norm.load(path + "norm.", loader);
            linear_1.load(path + "linear_1.", loader);
            linear_2.load(path + "linear_2.", loader);
            patch_merger.load(path + "patch_merger.merging_layer.", loader);
        }

    public:
        const int hidden_size;
        RMSNorm   norm;
        Linear    linear_1;
        Linear    linear_2;
        Linear    patch_merger;
        ggml::merge_patch_param merge_param;
    };

    class VisionTransformer : public Block
    {
    public:
        typedef LMBlock1<RMSNorm, ViTSelfAttention, RMSNorm, SiLUMLP> LayerBlock;
        VisionTransformer(InitContext *ctx, const Config &config, int lm_hidden_size)
            : embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
                        config.hidden_size, config.patch_size),
            multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config, lm_hidden_size),
            loaded(false)
        {
            const int max_length = config.max_length;
            pos_helper.reset(new TensorPosHelper(max_length, config.patch_size));

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = new LayerBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size, max_length);
                layer->set_id(layer_id);
                layer->attention.set_pos_helper(pos_helper.get());
                layers.emplace_back(layer);
            }
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += embeddings.get_param_num(effective_only);
            r += multi_modal_projector.get_param_num(effective_only);
            for (size_t i = 0; i < layers.size(); i++)
                r += layers[i]->get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            if (!loader->has_tensor(path + "patch_conv.weight")) return;

            embeddings.load(path, loader);
            multi_modal_projector.load("multi_modal_projector.", loader);
            for (size_t i = 0; i < layers.size(); i++)
            {
                std::string block_path = path + "layers." + std::to_string(i) + ".";
                layers[i]->load(block_path, loader);
            }
            loaded = true;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w) override
        {
            multi_modal_projector.merge_param.grid_h = grid_h;
            multi_modal_projector.merge_param.grid_w = grid_w;
            pos_helper->prepare(grid_h, grid_w);

            auto output = embeddings.forward(ctx, input, grid_h, grid_w);

            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->attention.grid_h = grid_h;
                layers[i]->attention.grid_w = grid_w;
                output = layers[i]->forward(ctx, output, 0);
            }
            output = multi_modal_projector.forward(ctx, output);
            return output;
        }

        bool is_loaded(void) const
        {
            return loaded;
        }
    public:
        PatchEmbedding embeddings;
        std::vector<std::unique_ptr<LayerBlock>> layers;
        MultiModalProjector multi_modal_projector;
        std::unique_ptr<TensorPosHelper> pos_helper;
    protected:
        bool loaded;
    };

    class VisualEmbeddingGeneration
    {
    public:
        VisualEmbeddingGeneration(const RuntimeConfig &runtime_config, int max_llm_tokens, size_t GRAPH_SIZE = 4096):
            max_llm_tokens(max_llm_tokens),
            eval(runtime_config, "vis", GRAPH_SIZE),
            _ctx(eval.get_backend_context())
        {
            _ctx.cache_dtype = runtime_config.cache_type;
        }

        bool load(ModelLoader &loader)
        {
            if (vis_model.get())
            {
                loader.push_allocator_manager(eval.get_layer_allocators());
                vis_model->load("vision_model.", &loader);
                loader.pop_allocator_manager();
                return vis_model->is_loaded();
            }
            else
                return false;
        }

        bool load_more(ggml::type dtype, int lm_hidden_size, const json::JSON &config)
        {
            const auto vis_cfg = config["config.json"]["vision_config"];
            if (!vis_cfg.IsObject()) return false;

            vis_config.dtype = dtype;

            vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
            vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
            vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
            vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
            vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();

            auto pp_cfg = config["processor_config.json"];
            if (pp_cfg.IsObject())
            {
                vis_config.spatial_merge_size = (int)pp_cfg["spatial_merge_size"].ToInt();

                pp_cfg = config["processor_config.json"]["image_processor"];
                auto image_mean = pp_cfg["image_mean"];
                auto image_std = pp_cfg["image_std"];
                CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
                CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

                vis_config.image_mean[0] = (float)image_mean[0].ToFloat();
                vis_config.image_mean[1] = (float)image_mean[1].ToFloat();
                vis_config.image_mean[2] = (float)image_mean[2].ToFloat();
                vis_config.image_std[0] = (float)image_std[0].ToFloat();
                vis_config.image_std[1] = (float)image_std[1].ToFloat();
                vis_config.image_std[2] = (float)image_std[2].ToFloat();

                vis_config.longest_edge = (int)pp_cfg["size"]["longest_edge"].ToInt();
            }
            else
                return false;

            vis_config.max_length = max_llm_tokens * vis_config.spatial_merge_size * vis_config.spatial_merge_size;

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 6 + vis_config.num_hidden_layers * 10;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            _ctx.dtype = dtype;

            vis_model.reset(new VisionTransformer(&_ctx, vis_config, lm_hidden_size));

            _ctx.check_used_mem_size(true);

            return true;
        }

        void generate(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, std::vector<uint8_t> &buf)
        {
            if ((vis_model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
            if (!vis_model->is_loaded()) return;

            for (auto &image : tok->media_emb)
            {
                run_model(gen_config, tok, dtype, image, buf);
            }
        }

    protected:
        bool run_model(const GenerationConfig &gen_config, BaseTokenizer *tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector &image, std::vector<uint8_t> &buf)
        {
            ggml::tensor *media_emb = nullptr;
            const auto make_graph = [this, &media_emb, &image](ComputeContext *ctx) -> ggml::tensor * {
                media_emb = ggml::new_tensor_3d(ctx,
                    ggml::type::GGML_TYPE_F32, vis_config.patch_size * image.grid_width, vis_config.patch_size * image.grid_height, 3);
                auto r = vis_model->forward(ctx, media_emb, image.grid_height, image.grid_width);
                return r;
            };
            const auto write_input_data = [&media_emb, &image](ComputeContext *ctx) {
                Backend::write_tensor_data(media_emb, image.data.data(), 0, image.data.size() * sizeof(image.data[0]));
            };

            std::vector<int64_t> shape;
            eval.evaluate(gen_config, make_graph, write_input_data, dtype, shape, buf);
            return true;
        }

    protected:
        const int max_llm_tokens;
        std::unique_ptr<VisionTransformer> vis_model;
        TensorGraphEvaluator eval;
        InitContext _ctx; // weight context
    public:
        Config vis_config;
    };
}

namespace chatllm::mistral::mistral3
{
    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int sliding_window;
        int tie_word_embeddings;
        int head_dim;

        float beta_fast;
        float beta_slow;
        float factor;
        float llama_4_scaling_beta;
        float mscale;
        float mscale_all_dim;
        int   original_max_position_embeddings;
        float rope_theta;
    };

    class ChatHistoryEncoder : public mistral::ChatHistoryEncoder
    {
    public:
        void append_user(int round_idx, const Content &user, std::vector<int> &ids) const override;

    public:
        const pixtral::Config *vis_config = nullptr;
        bool vit_loaded = false;
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public mistral2::Tokenizer
    {
    public:
        Tokenizer(const Config &config) : mistral2::Tokenizer(config, &_chat_encoder)
        {
            sys_prompt = R"""(You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.

When you're not sure about some information or when the user's request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid making up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos.

# TOOL CALLING INSTRUCTIONS

You may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:

1. When the request requires up-to-date information.
2. When the request requires specific data that you do not have in your knowledge base.
3. When the request involves actions that you cannot perform without tools.

Always prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.)""";

            sys_prompt = utils::replace_all(sys_prompt, "{today}", utils::now());
            sys_prompt = utils::replace_all(sys_prompt, "{yesterday}", utils::now(-1));
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = mistral2::Tokenizer::load(buffer, n_vocab);
            bos_token_id = tp->PieceToId("<s>");
            eos_token_id = tp->PieceToId("</s>");
            image_break_token_id    = tp->PieceToId("[IMG_BREAK]");
            image_end_token_id      = tp->PieceToId("[IMG_END]");
            think_start_token_id    = tp->PieceToId("[THINK]");
            think_end_token_id      = tp->PieceToId("[/THINK]");
            if (think_start_token_id >= 0)
            {
                tp->OverrideTokenDecoding(think_start_token_id, "<think>");
                tp->OverrideTokenDecoding(think_end_token_id,   "</think>");
            }
            return r;
        }

        void inject_image(std::vector<int> &ids, const int ids_to_inject_start, const int width, const int height)
        {
            int id = ids_to_inject_start;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    ids.push_back(id++);
                }
                ids.push_back(image_break_token_id);
            }
            ids.push_back(image_end_token_id);
        }
    public:
        int image_break_token_id;
        int image_end_token_id;
        int think_start_token_id;
        int think_end_token_id;
    };

    class ScaledSelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        ScaledSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : RoPESelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            attn_scale = ggml::new_tensor_1d(ctx, GGML_TYPE_F32, max_length);
            ctx->get_allocator()->alloc(attn_scale);
        }

        ggml::tensor *calc_attn_scores(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
            ggml::tensor *key_layer, ggml::tensor *query_layer, ggml::tensor *value_layer) override
        {
            auto scale = ggml::view_2d(ctx, attn_scale, 1, qlen, ggml::element_size(attn_scale), n_past * ggml::element_size(attn_scale));
            query_layer = ggml::mul(ctx, query_layer, scale);

            return RoPESelfAttention<BaseAttention>::calc_attn_scores(ctx, hidden_size, n_past, qlen,
                key_layer, query_layer, value_layer);
        }

        void write_attn_scale(const float *data)
        {
            Backend::write_tensor_data(attn_scale, data);
        }
    public:
        ggml::tensor *attn_scale;
    };

    static float yarn_get_mscale(float scale = 1.0f, float mscale = 1.0f)
    {
        if (scale <= 1.0f)
            return 1.0f;
        return 0.1f * mscale * logf(scale) + 1.0f;
    }

    static float get_llama_4_attn_scale(int pos, float beta, int max_position_embeddings)
    {
        return 1.0f + beta * logf(1.0f + floorf((float)pos / max_position_embeddings));
    }

    #define MAX_PROJECTED_TOKENS    2048

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef LMBlock1<RMSNorm, ScaledSelfAttention, RMSNorm, SiLUMLP> LayerBlock;
        typedef Model<BaseConfig, Embedding, RMSNorm, LayerBlock, int, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_MISTRAL3) :
            BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
            config(config),
            visual(runtime_config, MAX_PROJECTED_TOKENS)
        {
            BlockParams::PadEmbedding param(MAX_PROJECTED_TOKENS, MAX_PROJECTED_TOKENS);

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = (config.tie_word_embeddings ? 2 : 3)
                                + config.num_hidden_layers * 13;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(&w_ctx_, config,
                (0 == config.tie_word_embeddings) ? create_embedding<Embedding>(&w_ctx_, config) : nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads,
                config.head_dim, config.max_length);

            float m = 1.0f;
            float attn_scaling_factor = -1.0f;
            if (config.original_max_position_embeddings > 0)
            {
                m = yarn_get_mscale(config.factor, config.mscale) / yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor = 1 / sqrtf((float)config.head_dim);
                float mscale = yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor *= mscale * mscale;
                m /= 1.0f + 0.1f * logf(config.factor);
            }

            #define config_rope(attention)     do { \
                    attention.rope_mode      = RoPEMode::Original;                          \
                    attention.freq_base      = config.rope_theta;                           \
                    if (config.original_max_position_embeddings > 0)                        \
                    {                                                                       \
                    attention.n_ctx          = config.max_length;                           \
                    attention.n_original_ctx = config.original_max_position_embeddings;     \
                    attention.freq_scale     = 1 / config.factor;                           \
                    attention.beta_fast      = config.beta_fast;                            \
                    attention.beta_slow      = config.beta_slow;                            \
                    attention.ext_factor               = 1.0f;                              \
                    attention.attn_factor              = m;                                 \
                    attention.attn_scaling_factor      = attn_scaling_factor;               \
                    }                                                                       \
                } while (false)


            std::vector<float> attn_scale;
            attn_scale.resize(config.max_length);
            for (int i = 0; i < config.max_length; i++)
            {
                attn_scale[i] = get_llama_4_attn_scale(i, config.llama_4_scaling_beta, config.original_max_position_embeddings);
            }

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &layer = dynamic_cast<ModelClass *>(transformer)->layers[i];
                config_rope(layer.attention);
                layer.attention.write_attn_scale(attn_scale.data());
            }

            #undef config_rope

            w_ctx_.check_used_mem_size(true);
        }

        void load(ModelLoader &loader) override
        {
            BaseModelForConditionalGeneration::load(loader);

            loader.add_tensor_name_translations({
                {".self_attn.",                         ".attn."},
                {".input_layernorm.",                   ".attention_norm."},
                {".post_attention_layernorm.",          ".ffn_norm."},
            });

            _chat_encoder.vit_loaded = visual.load(loader);
        }

        bool load_more(const json::JSON &config) override
        {
            BaseModelForConditionalGeneration::load_more(config);
            bool r = visual.load_more(this->config.dtype, this->config.hidden_size, config);
            if (r)
            {
                _chat_encoder.vis_config = &visual.vis_config;
            }
            return r;
        }

        void before_generate(const GenerationConfig &gen_config) override
        {
            std::vector<uint8_t> buf;
            auto emb = dynamic_cast<Embedding *>(dynamic_cast<ModelClass *>(transformer)->word_embeddings);
            visual.generate(gen_config, dynamic_cast<Tokenizer *>(tokenizer), ggml::type_of(emb->weight), buf);
            if (buf.size() < 1) return;

            size_t offset = emb->get_base_nbytes();
            Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
        }

    public:
        const Config config;
    protected:
        pixtral::VisualEmbeddingGeneration visual;
    };

    void ChatHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->start_inst_token_id);

        for (auto &piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Image)
            {
                CHATLLM_CHECK(vit_loaded) << "Vision model not loaded";

                int w, h;
                std::vector<uint8_t> pixels;
                const int patch_size = vis_config->patch_size;

                vision::MaxPatchNum param2(vis_config->max_length);

                vision::image_load(piece.content.c_str(), pixels, w, h, patch_size, vision::PaddingMode::Black);

                std::vector<float> scaled;
                vision::image_rescale(pixels, scaled);

                vision::image_normalize(scaled, vis_config->image_mean, vis_config->image_std);

                tok->media_emb.push_back({.grid_width = w / patch_size, .grid_height = h / patch_size, .patch_size = patch_size, .data = {}});

                auto &image = tok->media_emb.back();

                vision::image_arrange(scaled, w, patch_size, image.data, vision::PatchesFormat::ChannelsRGB_PixelsLeftRightDown);

                image.emb_vec_number = image.grid_width * image.grid_height;

                const int id_start = tok->get_image_total_emb_vectors() - image.emb_vec_number + tok->vocab_size;
                tok->inject_image(ids, id_start, image.grid_width / vis_config->spatial_merge_size, image.grid_height / vis_config->spatial_merge_size);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }
        ids.push_back(tok->end_inst_token_id);
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(MISTRAL,               mistral::mistral, 1);
    REGISTER_MODEL_LOADER(MIXTRAL,               mistral::mixtral, 1);
    REGISTER_MODEL_LOADER(MISTRAL2,              mistral::mistral2, 1);
    REGISTER_MODEL_LOADER(MISTRAL3,              mistral::mistral3, 1);
}