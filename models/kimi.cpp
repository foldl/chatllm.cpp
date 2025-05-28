namespace vl
{
    struct Config : public deepseek::v3_light::Config
    {
    };

    struct VisualConfig
    {
        ggml::type dtype;
        int patch_size;
        int num_attention_heads;
        int num_hidden_layers;
        int hidden_size;
        int intermediate_size;
        int init_pos_emb_height;
        int init_pos_emb_width;
        int merge_kernel_size[2];

        int   in_token_limit;
        float image_mean[3];
        float image_std[3];
        bool  pad_input;
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
        Tokenizer(const Config &config)
            : Tokenizer(config, &_chat_encoder)
        {}

        Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder)
            : BaseTokenizer(config, chat_encoder, nullptr, nullptr)
        {
            sys_prompt = "You are a helpful assistant";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor2(
                {
                    "[\\p{Han}]+",
                    // FIXME: support &&
                    //"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    //"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\+",
                }
            );
            size_t size = tp->Load(buffer, n_vocab);

            return size;
        }

        bool load_config(const json::JSON &config) override
        {
            #define check_token(tok) {std::string("<|" #tok "|>"), &(tok ## _token_id)}

            load_added_tokens(config, {
                check_token(im_end),
                check_token(im_user),
                check_token(im_assistant),
                check_token(im_system),
                check_token(im_middle),
                check_token(media_start),
                check_token(media_content),
                check_token(media_end),
                check_token(media_pad),
            });
            #undef check_token

            if (im_end_token_id >= 0)
                terminate_ids.insert(im_end_token_id);

            return true;
        }
    public:
        int im_end_token_id;
        int im_user_token_id;
        int im_assistant_token_id;
        int im_system_token_id;
        int im_middle_token_id;
        int media_start_token_id;
        int media_content_token_id;
        int media_end_token_id;
        int media_pad_token_id;
    };

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->im_end_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->im_system_token_id);
        tok->encode("system", ids);
        ids.push_back(tok->im_middle_token_id);
        tok->encode(tok->get_system_prompt(), ids);
        ids.push_back(tok->im_end_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_user_opening(round_idx, ids);
        tok->encode(user, ids);
        ids.push_back(tok->im_end_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->im_assistant_token_id);
        tok->encode("assistant", ids);
        ids.push_back(tok->im_middle_token_id);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->im_user_token_id);
        tok->encode("user", ids);
        ids.push_back(tok->im_middle_token_id);
    }

    class PatchEmb : public Block
    {
    public:
        PatchEmb() {}
        PatchEmb(InitContext *ctx, int out_dim, int patch_size, int pos_emb_height, int pos_emb_width, int in_dim = 3)
            : proj(ggml::new_tensor_4d(ctx, ctx->dtype, patch_size, patch_size, in_dim, out_dim)),
              pos_emb(ggml::new_tensor_3d(ctx, ctx->dtype, pos_emb_height, pos_emb_width, out_dim))
        {
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return ggml::nelements(proj) + ggml::nelements(pos_emb);
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input, int grid_h, int grid_w)
        {
           //  ggml::tensor *x = ggml::conv_2d(ctx, proj, input);

            return nullptr;
        }
    public:
        ggml::tensor *proj;
        ggml::tensor *pos_emb;
    };

    class VisualModel
    {
    public:
        VisualModel(const VisualConfig &config, const RuntimeConfig &runtime_config, BackendContext *backend_context, size_t GRAPH_SIZE = 2048)
            :
            backend_context(backend_context), GRAPH_SIZE(GRAPH_SIZE), w_ctx_(backend_context)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * (11 + 3);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;
            InitContext *ctx = &w_ctx_;

            auto create_layer = [&](InitContext *ctx, int layer_index) {
                return new Block();
            };

            layers.reserve(config.num_hidden_layers);
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = create_layer(ctx, layer_id);
                layers.emplace_back(layer);

                layer->set_id(layer_id);
            }
        }
    protected:
        std::vector<Block *> layers;
        LayerNorm final_layernorm;
        BackendContext *backend_context;
        const size_t GRAPH_SIZE;
        InitContext w_ctx_; // weight context
    };

    class ConditionalGeneration : public deepseek::v3_light::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_KIMI_VL)
            : deepseek::v3_light::ConditionalGeneration(config, runtime_config, type)
        {
            vis_config.dtype = config.dtype;
        }

        bool load_more(const json::JSON &config) override
        {
            const auto vis_cfg = config["config.json"]["vision_config"];
            auto pp_cfg = config["preprocessor_config.json"];
            if (!vis_cfg.IsObject() || !pp_cfg.IsObject()) return false;

            vis_config.patch_size           = (int)vis_cfg["patch_size"].ToInt();
            vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
            vis_config.num_attention_heads  = (int)vis_cfg["num_attention_heads"].ToInt();
            vis_config.num_hidden_layers    = (int)vis_cfg["num_hidden_layers"].ToInt();
            vis_config.hidden_size          = (int)vis_cfg["hidden_size"].ToInt();
            vis_config.intermediate_size    = (int)vis_cfg["intermediate_size"].ToInt();
            vis_config.init_pos_emb_height  = (int)vis_cfg["init_pos_emb_height"].ToInt();
            vis_config.init_pos_emb_width   = (int)vis_cfg["init_pos_emb_width"].ToInt();

            auto size = vis_cfg["merge_kernel_size"];
            CHATLLM_CHECK(size.length() == 2) << "invalid merge_kernel_size";
            vis_config.merge_kernel_size[0]   = (int)size[0].ToInt();
            vis_config.merge_kernel_size[1]   = (int)size[1].ToInt();

            auto image_mean = pp_cfg["image_mean"];
            auto image_std  = pp_cfg["image_std"];
            CHATLLM_CHECK(image_mean.length() == 3) << "invalid image_mean";
            CHATLLM_CHECK(image_std.length() == 3) << "invalid image_std";

            vis_config.in_token_limit   = (int  )pp_cfg["in_token_limit"].ToInt();
            vis_config.pad_input        =        pp_cfg["pad_input"].ToBool();
            vis_config.image_mean[0]    = (float)image_mean[0].ToFloat();
            vis_config.image_mean[1]    = (float)image_mean[1].ToFloat();
            vis_config.image_mean[2]    = (float)image_mean[2].ToFloat();
            vis_config.image_std[0]     = (float)image_std[0].ToFloat();
            vis_config.image_std[1]     = (float)image_std[1].ToFloat();
            vis_config.image_std[2]     = (float)image_std[2].ToFloat();

            return true;
        }
    public:
        VisualConfig vis_config;
    };
}