#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"

namespace chatllm::hunyuan::dense
{
    struct Config : public BaseConfig
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
    };

    class Tokenizer : public BaseTokenizer
    {
    public:
        Tokenizer(const BaseConfig &config);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
    public:
        int end_of_text_token_id;
        int start_of_text_token_id;
        int extra_token_ids[5];
    };

    class HunyuanSelfAttention : public QKNormedAttention<RMSNormInplace, BaseAttention>
    {
    public:
        HunyuanSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : QKNormedAttention<RMSNormInplace, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, false, false) {}
        HunyuanSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : QKNormedAttention<RMSNormInplace, BaseAttention>(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false) {}
    };

    class HunyuanBlock : public LMBlock1<RMSNorm, HunyuanSelfAttention, RMSNorm, SiLUMLP>
    {
    public:
        HunyuanBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int head_dim, int max_length)
            : LMBlock1(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {}
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef Model<Config, Embedding, RMSNorm, HunyuanBlock, int, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_HUNYUAN_DENSE);

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type, int head_dim);

        void load(ModelLoader &loader) override;

    public:
        Config config;
    };
}

namespace chatllm::hunyuan::moe_v1
{
    struct Config : public BaseConfig
    {
        int num_key_value_heads;
        int num_experts;
        int moe_intermediate_size;
        int moe_topk;
        int num_shared_expert;

        float rope_theta;
    };

    typedef dense::Tokenizer Tokenizer;

    class ConditionalGeneration : public ModelProxy
    {
    public:
        ConditionalGeneration() = default;

        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config);

        void load(ModelLoader &loader) override;
    };
}