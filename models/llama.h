#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::llama::v2
{
    struct Config : public BaseConfig
    {
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

    public:
        virtual void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const;
    };

    template <class LayerBlock> class GenericConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef Model<BaseConfig, Embedding, RMSNorm, LayerBlock, int, int, int, int, int> ModelClass;
        typedef Model<BaseConfig, Embedding, RMSNorm, LayerBlock, int, int, int, int, int, int> ModelClass2;
    public:
        GenericConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type, int num_key_value_heads, int max_length, int tensors_per_layer = 12, bool tie_lm_head = false, int additional_tensor = 0)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config), type_class(1)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * tensors_per_layer + additional_tensor;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (tie_lm_head)
                Base::transformer = new ModelClass(&w_ctx_, config, nullptr,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, max_length);
            else
                Base::transformer = new ModelClass(&w_ctx_, config, false,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, max_length);
        }

        GenericConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type, int num_key_value_heads, int head_dim, int max_length, int tensors_per_layer, bool tie_lm_head)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config), type_class(2)
        {
            const size_t tensor_ovhd = ggml_tensor_overhead();
            const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * tensors_per_layer;
            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            if (tie_lm_head)
                Base::transformer = new ModelClass2(&w_ctx_, config, nullptr,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, head_dim, max_length);
            else
                Base::transformer = new ModelClass2(&w_ctx_, config, false,
                                                    config.hidden_size, config.num_attention_heads,
                                                    config.intermediate_size, num_key_value_heads, head_dim, max_length);
        }

    public:
        BaseConfig config;

    protected:
        const int type_class;
    };

    class ConditionalGeneration : public GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA2);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length);
    };
}

namespace chatllm::llama::v3
{
    struct Config : public v2::Config
    {
        int num_key_value_heads;
        float rope_theta;
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

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        bool load_config(const json::JSON &config) override;
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

        void encode_header(const std::string &text, std::vector<int> &ids) const;

        void encode_content(const std::string &text, std::vector<int> &ids) const;
    public:
        int start_header_id;
        int end_header_id;
        int eot_id;
        int nl_token_id;

    private:
        bool is_seed_coder;
    };

    class ConditionalGeneration : public v2::GenericConditionalGeneration<LlamaBlock>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA3);
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length);
    };
}

namespace chatllm::llama::v3_1
{
    struct Config : public v3::Config
    {
        int rope_scaling_original_max_position_embeddings;
        float rope_scaling_factor;
        float rope_scaling_low_freq_factor;
        float rope_scaling_high_freq_factor;
    };

    class ChatHistoryEncoder : public v3::ChatHistoryEncoder
    {
    public:
        void append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const override;
    };

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int eom_id;
        int python_tag_id;
    };

    class ToolCallingInterceptor : public ChunkInterceptor
    {
    public:
        ToolCallingInterceptor();

        void put_chunk(bool first, const std::string &chunk) override;

        void end() override;
    protected:
        std::ostringstream oss;
        bool found_tool_call;
    };


    void init_llama3_freq_factors(std::vector<float> &freq_factors, int rope_dim,
                float theta, float scaling_factor, float scaling_low_freq_factor, float scaling_high_freq_factor, int original_max_position_embeddings);

    class ConditionalGeneration : public v2::GenericConditionalGeneration<Llama31Block>
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA3_1);

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                            int num_key_value_heads, int max_length, int tensors_per_layer = 13, bool tie_lm_head = false, int additional_tensor = 0);

        ChunkInterceptor *get_interceptor(void) override;
    };
}

namespace chatllm::llama::v3_2
{
    struct Config : public v3_1::Config
    {
        int tie_word_embeddings;
    };

    typedef v3_1::Tokenizer Tokenizer;

    class ConditionalGeneration : public v3_1::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA3_2);
    };
}

namespace chatllm::llama::v2_plus
{
    typedef v3::Config Config;
    typedef v2::Tokenizer Tokenizer;

    class ConditionalGeneration : public v3::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_LLAMA2);
    };
}

namespace chatllm::llama::multi
{
    struct Config : public v3::Config
    {
        int n_future_tokens;
    };

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    class LlamaBlock : public LMBlock1<RMSNorm, LlamaSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        LlamaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length)
        {}
    };

    // MultiPredModel seems better to inherit from HeterogeneousModel (or move into it)
    // If there are more models use this technique, then DO IT.
    class MultiPredModel : public Model<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int>
    {
    public:
        typedef Model<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int> Base;
        MultiPredModel(InitContext *ctx, const Config &config, int hidden_size, int num_attention_heads, int intermediate_size,
                       int num_key_value_heads, int max_length, int n_future_tokens);

        ~MultiPredModel();

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override;

        int set_n_future_tokens(int n);

    protected:
        int64_t get_param_num_of_layers(bool effective_only) const override;

        ggml::tensor *calc_logits(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states);
    public:
        Config config;
        const int n_future_tokens;
        std::vector<LlamaBlock *> extra_heads;
        std::vector<Block *> prediction_heads;
    private:
        int effective_n;
    };


    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    private:
        typedef BaseModelForConditionalGeneration Base;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_LLAMA_MULTI,
                                     int tensors_per_layer = 12);

        void load(ModelLoader &loader) override;

        void set_additional_args(const std::map<std::string, std::string> &args) override;
    public:
        Config config;
    };
}

namespace chatllm::llama::ds_r1_distill
{
    typedef v3_2::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);

        Tokenizer(const Config &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int user_token_id;
        int assistant_token_id;
        int nl_token_id;
    };

    class ConditionalGeneration : public v3_2::ConditionalGeneration
    {
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA);
    };
}

namespace chatllm::llama::v4
{
    struct Config : public v2::Config
    {
        int num_key_value_heads;
        int attention_chunk_size;
        int head_dim;
        int interleave_moe_layer_step;
        int intermediate_size_mlp;
        int num_experts_per_tok;
        int n_routed_experts;
        int use_qk_norm;

        float router_aux_loss_coef;
        float rope_theta;
        int   rope_scaling_original_max_position_embeddings;
        float rope_scaling_factor;
        float rope_scaling_low_freq_factor;
        float rope_scaling_high_freq_factor;
    };

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK> class LlamaSparseMoE : public BaseSparseMLP
    {
    public:
        LlamaSparseMoE(InitContext *ctx, int hidden_size, int intermediate_size)
            : BaseSparseMLP(ctx, hidden_size, intermediate_size, NUM_EXPERTS, EXPERTS_PER_TOK, ActFunc::SILU, false)
        {
            score_func = ScoreFunc::Sigmoid;
            norm_topk_prob = false;
            pre_weighting  = true;
        }
    };

    class LlamaNormedSelfAttention : public QKNormedAttention<L2Norm, BaseAttention>
    {
    public:
        LlamaNormedSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<L2Norm, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            post_norm = true;
        }
    };

    template <class SelfAttention, class MoEMLP> class LlamaMoEBlock_ : public LMBlock1<RMSNorm, SelfAttention, RMSNorm, MoEMLP>
    {
    public:
        LlamaMoEBlock_(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                  int mlp_intermediate_size1, int mlp_intermediate_size2,
                  int num_kv_heads,
                  int head_dim, int max_length)
            : LMBlock1<RMSNorm, SelfAttention, RMSNorm, MoEMLP>(ctx, hidden_size, num_attention_heads, intermediate_size, mlp_intermediate_size1, mlp_intermediate_size2,
              num_kv_heads, head_dim, max_length)
        {}
    };

    template <int NUM_EXPERTS, int EXPERTS_PER_TOK, int EFFECTIVE_EXPERTS_PER_TOK, class SelfAttention> class ConditionalGeneration0 : public BaseModelForConditionalGeneration
    {
    public:
        typedef CombinedMLP<LlamaSparseMoE<NUM_EXPERTS, EFFECTIVE_EXPERTS_PER_TOK>, SiLUMLP> LlamaMoEMLP;
        typedef LlamaMoEBlock_<SelfAttention, LlamaMoEMLP> LlamaMoEBlock;
        typedef LMBlock1<RMSNorm, SelfAttention, RMSNorm, SiLUMLP> LlamaDenseBlock;
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
    public:
        ConditionalGeneration0() = default;

        ConditionalGeneration0(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_LLAMA4)
            : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
              config(config)
        {
            init_moe_vector();

            const size_t tensor_ovhd = ggml_tensor_overhead();
            const int moe_layer_num = get_moe_layer_num();
            const int dense_layer_num = config.num_hidden_layers - moe_layer_num;
            const size_t num_tensors = 3 + moe_layer_num * (13 + 3) + dense_layer_num * 13 +
                (config.rope_scaling_original_max_position_embeddings > 0 ?  config.num_hidden_layers : 0);

            const size_t ctx_size = num_tensors * tensor_ovhd;
            w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
            w_ctx_.dtype = config.dtype;

            CHATLLM_CHECK((NUM_EXPERTS == config.n_routed_experts)
                            && (EXPERTS_PER_TOK == config.num_experts_per_tok)
                            && (EFFECTIVE_EXPERTS_PER_TOK <= EXPERTS_PER_TOK))
                << "unsupported MoE param";

            CHATLLM_CHECK(config.max_length <= 8192)
                << "TODO: support longer context (attn_temperature_tuning)." << std::endl
                << "use `--max_length 8192` to specify a shorter context.";

            auto create_layer = [&](InitContext *ctx, int layer_index) -> Block * {
                if (is_layer_moe(layer_index))
                {
                    auto layer = new LlamaMoEBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size,
                        config.intermediate_size, config.intermediate_size,
                        config.num_key_value_heads, config.head_dim, config.max_length);
                    return layer;
                }
                else
                {
                    return new LlamaDenseBlock(ctx, config.hidden_size, config.num_attention_heads, config.intermediate_size_mlp,
                                                config.num_key_value_heads, config.head_dim, config.max_length);
                }
            };

            auto transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false), create_layer);
            Base::transformer = transformer;

            std::vector<float> freq_factors_value;
            if (config.rope_scaling_original_max_position_embeddings > 0)
            {
                v3_1::init_llama3_freq_factors(freq_factors_value, config.head_dim,
                    config.rope_theta,
                    config.rope_scaling_factor,
                    config.rope_scaling_low_freq_factor,
                    config.rope_scaling_high_freq_factor,
                    config.rope_scaling_original_max_position_embeddings);
            }

            #define config_rope(attention)     do { \
                    attention.use_rope     = (i + 1) % 4 != 0;      \
                    attention.freq_base    = config.rope_theta;     \
                    attention.rope_mode = RoPEMode::Original;       \
                    if (config.rope_scaling_original_max_position_embeddings > 0) { \
                        attention.freq_factors = ggml::new_tensor_1d(&w_ctx_, GGML_TYPE_F32, config.head_dim / 2);  \
                        w_ctx_.get_allocator()->alloc(attention.freq_factors);                                      \
                        Backend::write_tensor_data(attention.freq_factors, freq_factors_value.data());  \
                    }                                               \
                } while (false)

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<LlamaMoEBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
                else
                {
                    auto *layer = dynamic_cast<LlamaDenseBlock *>(transformer->get_layer(i));
                    config_rope(layer->attention);
                }
            }

            #undef config_rope

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                    << "corrupted model weights: " << w_ctx_.get_used_mem() / tensor_ovhd << " vs " << w_ctx_.get_mem_size() / tensor_ovhd;
        }

        void load(ModelLoader &loader) override
        {
            auto transformer = get_typed_transformer<ModelClass>();

            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            transformer->final_layernorm->load("model.norm.", &loader);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            SelfAttention *attention = nullptr;

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

                if (is_layer_moe(i))
                {
                    auto *layer = dynamic_cast<LlamaMoEBlock *>(transformer->get_layer(i));
                    attention = &layer->attention;

                    loader.read_tensor(layer_prefix + "mlp.mlp1.experts_down.weight", layer_prefix + "mlp.experts.", config.n_routed_experts, ".down_proj.weight", layer->mlp.mlp1.experts_down.weight);
                    loader.read_tensor(layer_prefix + "mlp.mlp1.experts_gate.weight", layer_prefix + "mlp.experts.", config.n_routed_experts, ".gate_proj.weight", layer->mlp.mlp1.experts_gate.weight);
                    loader.read_tensor(layer_prefix + "mlp.mlp1.experts_up.weight",   layer_prefix + "mlp.experts.", config.n_routed_experts, ".up_proj.weight",   layer->mlp.mlp1.experts_up.weight);

                    loader.read_tensor(layer_prefix + "mlp.gate.weight", layer->mlp.mlp1.gate.weight);

                    loader.read_tensor(layer_prefix + "mlp.shared_expert.down_proj.weight", layer->mlp.mlp2.down_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.shared_expert.gate_proj.weight", layer->mlp.mlp2.gate_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.shared_expert.up_proj.weight",   layer->mlp.mlp2.up_proj.weight);

                    loader.read_tensor(layer_prefix + "input_layernorm.weight",          layer->input_layernorm.weight);
                    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);
                }
                else
                {
                    auto *layer = dynamic_cast<LlamaDenseBlock *>(transformer->get_layer(i));
                    attention = &layer->attention;

                    loader.read_tensor(layer_prefix + "mlp.down_proj.weight", layer->mlp.down_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", layer->mlp.gate_proj.weight);
                    loader.read_tensor(layer_prefix + "mlp.up_proj.weight",   layer->mlp.up_proj.weight);

                    loader.read_tensor(layer_prefix + "input_layernorm.weight",          layer->input_layernorm.weight);
                    loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", layer->post_attention_layernorm.weight);
                }

                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", attention->k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", attention->o_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", attention->q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", attention->v_proj.weight);
            }
        }

    private:
        void init_moe_vector()
        {
            layer_is_moe.resize(config.num_hidden_layers, false);
            for (int i = config.interleave_moe_layer_step - 1; i < config.num_hidden_layers; i += config.interleave_moe_layer_step)
                layer_is_moe[i] = true;
        }

    public:
        Config config;
        std::vector<bool> layer_is_moe;

        bool is_layer_moe(int layer_index)
        {
            return layer_is_moe[layer_index];
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

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);
    };
}