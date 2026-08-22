#include "allenai.h"
#include "deepseek.h"

namespace chatllm::instella
{
    const int MODEL_TYPE_INSTELLA_MoE = MODEL_TYPE_INSTELLA + 1;
}

namespace chatllm::instella
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        float rope_theta;
    };

    typedef allenai::moe::Tokenizer Tokenizer;

    typedef LMBlock4<RMSNorm,
                    allenai::moe::OLSelfAttention,
                    Identity,
                    RMSNorm,
                    SiLUMLP,
                    Identity> InstellaBlock;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<Config, Embedding, RMSNorm, InstellaBlock, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_INSTELLA)
            : Base(type, config, runtime_config, 4096 * 2), config(config)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = 3 + config.num_hidden_layers * (14);
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            transformer = new ModelClass(
                                &w_ctx_, config, false,
                                config.hidden_size, config.num_attention_heads,
                                config.intermediate_size, config.num_key_value_heads, config.max_length);

            auto transformer = Base::get_typed_transformer<ModelClass>();
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                auto &attention = transformer->layers[i].attention;
                attention.rope_mode    = RoPEMode::Original;
                attention.freq_base    = config.rope_theta;
            }
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(Base::layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
                loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   transformer->layers[i].mlp.up_proj.weight);

                loader.read_tensor(layer_prefix + "pre_attention_layernorm.weight",
                                transformer->layers[i].pre_attention_layernorm.weight);

                loader.read_tensor(layer_prefix + "pre_feedforward_layernorm.weight",
                                transformer->layers[i].pre_mlp_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);

                loader.read_tensor(layer_prefix + "self_attn.q_norm.weight", transformer->layers[i].attention.q_norm.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_norm.weight", transformer->layers[i].attention.k_norm.weight);
            }
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }

    public:
        Config config;
    };
}

namespace chatllm::instella::moe
{
    struct Config : public deepseek::v3_light::Config
    {
        int farskip;
        int first_k_dense_replace;
        int farskip_start_idx;
        int farskip_end_idx;
    };

    typedef deepseek::v3_light::Tokenizer Tokenizer;

    class MLAGatedAttention : public deepseek::v2_light::MLAttention
    {
    public:
        MLAGatedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias);

        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;

    protected:
        ggml::tensor *output_project(ComputeContext *ctx, ggml::tensor *scores) override;
        ggml::tensor *rt_gate = nullptr;

    public:
        static bool gated_attention;
    public:
        std::unique_ptr<Linear> gate_proj;
    };

    bool MLAGatedAttention::gated_attention = false;

    MLAGatedAttention::MLAGatedAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                      int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                      bool use_bias):
        deepseek::v2_light::MLAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
            q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
            use_bias)
    {
        if (gated_attention)
        {
            gate_proj.reset(new Linear(ctx, hidden_size, num_attention_heads * v_head_dim, false));
        }
    }

    ggml::tensor *MLAGatedAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        rt_gate = nullptr;
        if (gate_proj.get())
        {
            rt_gate = gate_proj->forward(ctx, hidden_states);
            rt_gate = ggml::sigmoid(ctx, rt_gate);
        }

        ggml::tensor * r = deepseek::v2_light::MLAttention::forward(ctx, hidden_states, n_past);
        return r;
    }

    ggml::tensor *MLAGatedAttention::output_project(ComputeContext *ctx, ggml::tensor *scores)
    {
        if (rt_gate != nullptr)
        {
            auto gate = gate_proj->forward(ctx, scores);
            scores = ggml::mul(ctx, scores, rt_gate);
        }
        ggml::tensor * r = deepseek::v2_light::MLAttention::output_project(ctx, scores);
        return r;
    }

    int64_t MLAGatedAttention::get_param_num(bool effective_only) const
    {
        int64_t r = deepseek::v2_light::MLAttention::get_param_num(effective_only);
        if (gate_proj.get())
            r += gate_proj->get_param_num(effective_only);
        return r;
    }

    void MLAGatedAttention::load(const std::string &path, TensorLoader *loader)
    {
        deepseek::v2_light::MLAttention::load(path, loader);
        if (gate_proj.get())
        {
            gate_proj->load(path + "gate_proj.", loader);
        }
    }

    class LMBlock1ForwardFar : public LMBlock1Forward
    {
    public:
        using LMBlock1Forward::LMBlock1Forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *input_to_attn, ggml::tensor *input_to_mlp, int n_past);
    public:
        ggml::tensor *rt_residual_no_routed = nullptr;
    };

    ggml::tensor *LMBlock1ForwardFar::forward(ComputeContext *ctx, ggml::tensor *hidden_states, ggml::tensor *input_to_attn, ggml::tensor *input_to_mlp, int n_past)
    {
        ggml::tensor *residual = hidden_states;

        input_to_attn = input_layernorm->forward(ctx, input_to_attn);
        auto attn_output = attention->forward(ctx, input_to_attn, n_past);

        residual = ggml::add(ctx, attn_output, residual);

        if (nullptr == input_to_mlp) input_to_mlp = residual;
        input_to_mlp = post_attention_layernorm->forward(ctx, input_to_mlp);
        last_result_post_attn_norm = input_to_mlp;

        auto mlp_output = mlp->forward(ctx, input_to_mlp);

        hidden_states = ggml::add(ctx, mlp_output, residual);

        auto combined_mlp = dynamic_cast<BaseCombinedMLP *>(mlp);
        if (combined_mlp != nullptr)
        {
            rt_residual_no_routed = ggml::add(ctx, combined_mlp->rt_mlp2_output, residual);
        }

        return hidden_states;
    }

    template <class MoEMLP> class FarSkipDecoderLayer : public LMBlock1<RMSNorm, MLAGatedAttention, RMSNorm, MoEMLP>
    {
    public:
        typedef LMBlock1<RMSNorm, MLAGatedAttention, RMSNorm, MoEMLP> Base;
        using LMBlock1<RMSNorm, MLAGatedAttention, RMSNorm, MoEMLP>::LMBlock1;

        FarSkipDecoderLayer(InitContext *ctx, bool farskip, ggml_tensor **prev_rt_residual_no_routed, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias, bool gate_use_bias)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size,
                  mlp_intermediate_size1, mlp_intermediate_size2,
                  num_kv_heads, max_length,
                  q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim, use_bias, gate_use_bias),
            farskip(farskip),
            prev_rt_residual_no_routed(prev_rt_residual_no_routed)
        {}

        FarSkipDecoderLayer(InitContext *ctx, bool farskip, ggml_tensor **prev_rt_residual_no_routed, int hidden_size, int num_attention_heads, int intermediate_size,
                  int num_kv_heads, int max_length,
                  int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                  bool use_bias)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size,
                  num_kv_heads, max_length,
                  q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim, use_bias),
            farskip(farskip),
            prev_rt_residual_no_routed(prev_rt_residual_no_routed)
        {}

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            LMBlock1ForwardFar eval(&(Base::input_layernorm), &(Base::attention), &(Base::post_attention_layernorm),
                &(Base::mlp), Base::get_id(), Base::scale_depth);

            if (farskip)
            {
                if (prev_rt_residual_no_routed && *prev_rt_residual_no_routed)
                {
                    hidden_states = eval.forward(ctx, hidden_states, *prev_rt_residual_no_routed, hidden_states, n_past);
                }
                else
                {
                    hidden_states = eval.forward(ctx, hidden_states, hidden_states, hidden_states, n_past);
                }
            }
            else
            {
                hidden_states = eval.forward(ctx, hidden_states, hidden_states, nullptr, n_past);
            }

            rt_residual_no_routed = eval.rt_residual_no_routed;
            Base::last_result_post_attn_norm = eval.last_result_post_attn_norm;
            return hidden_states;
        }
    public:
        const bool farskip;
        ggml::tensor *rt_residual_no_routed = nullptr;
    protected:
        ggml_tensor ** const prev_rt_residual_no_routed;
    };

    using deepseek::v2_light::DeepSeekSparseMoE;
    using deepseek::v2_light::yarn_get_mscale;

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<DeepSeekSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> MoEMLP;
        typedef FarSkipDecoderLayer<MoEMLP>  MoEBlock;
        typedef FarSkipDecoderLayer<SiLUMLP> DenseBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config) : ConditionalGeneration0(config, runtime_config, (ModelType)MODEL_TYPE_INSTELLA_MoE, -1)
        {}

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int q_lora_rank, BaseSparseMLP::ScoreFunc score_func = BaseSparseMLP::ScoreFunc::Softmax,
            bool gate_use_bias = false, bool always_scaling = false)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            MLAGatedAttention::gated_attention = true;
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3
                                + moe_layer_num * (16 + 3 + (gate_use_bias ? 1 : 0))
                                + dense_layer_num * 15
                                + (q_lora_rank > 0 ? config.num_hidden_layers * 2 : 0)
                                + config.num_hidden_layers * 1;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK)
                            && (config.n_group == 1))
                << "unsupported MoE param";

            ggml_tensor **prev_rt_residual_no_routed = nullptr;

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new MoEBlock(ctx, is_layer_farskip(layer_index), prev_rt_residual_no_routed, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.moe_intermediate_size, config.moe_intermediate_size * config.n_shared_experts,
                        config.num_key_value_heads, config.max_length,
                        q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                        false, gate_use_bias);
                    auto sparse = dynamic_cast<BaseSparseMLP *>(&layer->mlp.mlp1);
                    sparse->score_func = score_func;
                    sparse->routed_scaling_factor = config.routed_scaling_factor;
                    sparse->always_scaling = always_scaling;

                    prev_rt_residual_no_routed = &layer->rt_residual_no_routed;
                    return layer;
                }
                else
                {
                    auto layer = new DenseBlock(ctx, is_layer_farskip(layer_index), prev_rt_residual_no_routed, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                                                config.num_key_value_heads, config.max_length,
                                                q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim,
                                                false);
                    prev_rt_residual_no_routed = &layer->rt_residual_no_routed;
                    return layer;
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);
            Base::transformer = transformer;

            float m = 1.0f;
            float attn_scaling_factor = -1.0f;
            if (config.original_max_position_embeddings > 0)
            {
                m = deepseek::v2_light::yarn_get_mscale(config.factor, config.mscale) / yarn_get_mscale(config.factor, config.mscale_all_dim);
                attn_scaling_factor = 1 / sqrtf((float)(config.qk_rope_head_dim + config.qk_nope_head_dim));
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

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<MoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                    layer->mlp.mlp1.norm_topk_prob = config.norm_topk_prob != 0;
                }
                else
                {
                    auto *layer = dynamic_cast<DenseBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope
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

        bool is_layer_moe(int layer_index)
        {
            return layer_index >= config.first_k_dense_replace;
        }

        bool is_layer_farskip(int layer_index)
        {
            return (config.farskip_start_idx <= layer_index) && (layer_index <= config.farskip_end_idx);
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

    const int NUM_EXPERTS                   =  64;
    const int EXPERTS_PER_TOK               =  6;
    typedef ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK> ConditionalGeneration;
}

namespace chatllm::instella
{
    REGISTER_MODEL_LOADER(INSTELLA,              instella,      1);
    REGISTER_MODEL_LOADER(INSTELLA_MoE,          instella::moe,  1);
}