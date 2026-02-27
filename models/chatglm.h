#pragma once
#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::glm::v1
{
    struct Config : public BaseConfig
    {
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const Config &config);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

    protected:
        std::string preprocess(const std::string &text) const override;
        std::string postprocess(const std::string &text) const override;

    public:
        int mask_token_id;
        int gmask_token_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, LayerNorm, GLMBlock, int, int, int, int> TransformerClass;
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        void load(ModelLoader &loader) override;

    public:
        Config config;
    };
}

namespace chatllm::glm::v2
{
    struct Config : public BaseConfig
    {
        int num_kv_heads;
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        void encode(const std::string &text, std::vector<int> &ids) const override;

        bool is_special_id(int id) const override;

    public:
        int mask_token_id;
        int gmask_token_id;
        int smask_token_id;
        int sop_token_id;
        int eop_token_id;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, GLM2Block, int, int, int, int, int> TransformerClass;
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_CHATGLM2);

        void load(ModelLoader &loader) override;

    public:
        Config config;
    };
}

namespace chatllm::glm::v3
{
    struct Config : public v2::Config
    {
    };

    class Tokenizer : public v2::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);

        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *_chat_encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;

        bool is_special_id(int id) const override;

    public:
        int system_token_id;
        int user_token_id;
        int assistant_token_id;
        int observation_token_id;
    };

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        ChunkInterceptor *get_interceptor(void);
    };
}

namespace chatllm::glm::v4
{
    struct Config : public v3::Config
    {
        float rope_ratio;
    };

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    protected:
        void append_role_tok(std::vector<int> &ids, int role_tok) const;
    public:
        bool add_nl_token = true;
    };

    class Tokenizer : public v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *_chat_encoder);

        void encode(const std::string &text, std::vector<int> &ids) const override;
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int nl_token_id;

    protected:
        size_t do_load(tokenizer::DataReader *buffer, int n_vocab, std::vector<std::string> regex_exprs);
    };

    class ConditionalGeneration : public v2::ConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GLM4);
        ChunkInterceptor *get_interceptor(void) override;
    };
}

namespace chatllm::glm::glm4_0414
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int use_attention_bias;
        int rope_dim;
        float rope_theta;
    };

    class Tokenizer : public v4::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
        Tokenizer(const BaseConfig &config, BaseHistoryEncoder *_chat_encoder);

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int think_open_token_id;
        int think_close_token_id;
    };

    class GLM4SelfAttention : public RoPESelfAttention<BaseAttention>
    {
    public:
        GLM4SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : RoPESelfAttention<BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
        {
        }
    };

    class GLM4Block : public LMBlock4<RMSNorm, GLM4SelfAttention, RMSNorm, RMSNorm, SiLUMLP, RMSNorm>
    {
    public:
        GLM4Block(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
                  int max_length, bool qkv_bias, bool o_bias)
            : LMBlock4(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, qkv_bias, o_bias)
        {
            mlp.set_prec(ggml::prec::GGML_PREC_F32);
        }
    };

    typedef Model<Config, Embedding, RMSNorm, GLM4Block, int, int, int, int, int, bool, bool> ModelClass;

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_GLM4);

        void load(ModelLoader &loader) override;

    protected:
        Config config;
    };
}