#include "deepseek.h"

namespace chatllm::deepseek::v1
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *chat_encoder)
        : llama::v2::Tokenizer::Tokenizer(config, chat_encoder)
    {
        sys_prompt = "";
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_DEEPSEEK)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);

        std::ostringstream oss_ai;
        oss_ai << " " << ai;
        tok->encode(oss_ai.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            std::ostringstream oss_prompt;
            oss_prompt << tok->get_system_prompt() << "\n\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids, false, false);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_user_opening(round_idx, ids);

        std::ostringstream oss_prompt;
        oss_prompt << " ";
        oss_prompt << user << "\n\n";
        tok->encode(oss_prompt.str(), ids, false, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("Assistant:", ids, false, false);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("User:", ids, false, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id);
    }
}

namespace chatllm::deepseek::coder
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

    Tokenizer::Tokenizer(const llama::v2::Config &config, BaseHistoryEncoder *chat_encoder)
        : llama::v2::Tokenizer::Tokenizer(config, chat_encoder)
    {
        sys_prompt = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.";
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_DEEPSEEK_CODER),
        config(config)
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
        tp->Encode("<｜fim▁hole｜><｜fim▁begin｜><｜fim▁end｜><|User|><|Assistant|><|EOT|>", &ids);

        CHATLLM_CHECK(ids.size() == 6) << "tokenizer error";

        fim_hole_token_id = ids[0];
        fim_begin_token_id = ids[1];
        fim_end_token_id = ids[2];
        user_token_id = ids[3];
        assistant_token_id = ids[4];
        eot_token_id = ids[5];
        return size;
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids, true, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        append_ai_opening(round_idx, ids);

        oss_prompt << ai << "\n<|EOT|>";

        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "\n### Instruction:\n" << user;

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "\n### Response:\n";

        auto text = oss_prompt.str();
        tok->encode(text, ids, false, false);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == eot_token_id);
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        llama::v2::ConditionalGeneration::load(loader);

        auto transformer = get_typed_transformer<ModelClass>();

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
            attention.freq_scale = 1 / config.rope_scaling;
        }
    }

    REGISTER_MODEL_LOADER(DEEPSEEK_CODER,        coder, 1);
}

namespace chatllm::deepseek::v1_moe
{
}

namespace chatllm::deepseek::v2_light
{
    QProj::QProj(InitContext *ctx, int hidden_size, int num_attention_heads,
        int q_lora_rank, int rope_dim, int qk_nope_head_dim, bool use_bias)
        : d_q_proj(q_lora_rank > 0 ? new Linear(ctx, hidden_size, q_lora_rank, use_bias) : nullptr),
            u_q_proj(q_lora_rank > 0 ? new Linear(ctx, q_lora_rank, (qk_nope_head_dim + rope_dim) * num_attention_heads, false) : nullptr),
            norm(q_lora_rank > 0 ? new RMSNorm(ctx, q_lora_rank) : nullptr),
            q_proj(q_lora_rank <= 0 ? new Linear(ctx, hidden_size, (qk_nope_head_dim + rope_dim) * num_attention_heads, use_bias) : nullptr)
    {}

    int64_t QProj::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (q_proj)
        {
            r += q_proj->get_param_num(effective_only);
        }
        else
        {
            r += d_q_proj->get_param_num(effective_only);
            r += u_q_proj->get_param_num(effective_only);
            r += norm->get_param_num(effective_only);
        }
        return r;
    }

    ggml::tensor *QProj::forward(ComputeContext *ctx, ggml::tensor *hidden_states)
    {
        ggml::tensor *tmpq = nullptr;

        if (q_proj)
        {
            tmpq = q_proj->forward(ctx, hidden_states);
        }
        else
        {
            ggml::tensor *q_lora = d_q_proj->forward(ctx, hidden_states);
            q_lora = norm->forward(ctx, q_lora);
            tmpq = u_q_proj->forward(ctx, q_lora);
        }

        return tmpq;
    }

    void QProj::load(const std::string &path, TensorLoader *loader)
    {
        Block::load(path, loader);

        if (q_proj)
        {
            q_proj->load(path + "q_proj.", loader);
        }
        else
        {
            d_q_proj->load(path + "d_q_proj.", loader);
            u_q_proj->load(path + "u_q_proj.", loader);
            norm->load(path + "q_norm.", loader);
        }
    }

    BaseMLAttention::BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                    int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                    bool use_bias,
                    int cache_length)
        : KVCacheAttention(ctx,
                            num_attention_heads, num_kv_heads,
                            BlockParams::Optimization::speed ? (qk_nope_head_dim + rope_dim) * num_kv_heads : rope_dim * 1,
                            BlockParams::Optimization::speed ? v_head_dim * num_kv_heads : kv_lora_rank,
                            max_length,
                            cache_length),
            opt_speed(BlockParams::Optimization::speed),
            kv_lora_rank(kv_lora_rank),
            rope_dim(rope_dim),
            qk_nope_head_dim(qk_nope_head_dim),
            v_head_dim(v_head_dim),
            d_kv_proj(ctx, hidden_size, kv_lora_rank, nullptr, use_bias),
            k_pe_proj(ctx, hidden_size, rope_dim, nullptr, use_bias),
            u_k_nope_proj(ctx, kv_lora_rank, qk_nope_head_dim * num_kv_heads, nullptr, false),
            u_v_proj(ctx, kv_lora_rank, v_head_dim * num_kv_heads, nullptr, false),
            q_proj(ctx, hidden_size, num_attention_heads, q_lora_rank, rope_dim, qk_nope_head_dim, use_bias),
            o_proj(ctx, v_head_dim * num_attention_heads, hidden_size, use_bias),
            kv_norm(ctx, kv_lora_rank)
    {
    }

    BaseMLAttention::BaseMLAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                    int q_lora_rank, int kv_lora_rank, int rope_dim, int qk_nope_head_dim, int v_head_dim,
                    bool use_bias)
        : BaseMLAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length,
                            q_lora_rank, kv_lora_rank, rope_dim, qk_nope_head_dim, v_head_dim,
                            use_bias,
                            max_length)
    {}

    int64_t BaseMLAttention::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += d_kv_proj.get_param_num(effective_only);
        r += k_pe_proj.get_param_num(effective_only);
        r += u_k_nope_proj.get_param_num(effective_only);
        r += u_v_proj.get_param_num(effective_only);
        r += q_proj.get_param_num(effective_only);
        r += kv_norm.get_param_num(effective_only);
        r += o_proj.get_param_num(effective_only);
        return r;
    }

    ggml::tensor *BaseMLAttention::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        if (opt_speed)
            return forward_speed(ctx, hidden_states, n_past);
        else
            return forward_memory(ctx, hidden_states, n_past);
    }

    void BaseMLAttention::load(const std::string &path, TensorLoader *loader)
    {
        KVCacheAttention::load(path, loader);
        q_proj.load(path + "", loader);

        d_kv_proj.load(path + "d_kv_proj.", loader);
        k_pe_proj.load(path + "k_pe_proj.", loader);
        u_k_nope_proj.load(path + "u_k_nope_proj.", loader);
        u_v_proj.load(path + "u_v_proj.", loader);
        o_proj.load(path + "o_proj.", loader);
        kv_norm.load(path + "kv_norm.", loader);
    }

    ggml::tensor *BaseMLAttention::forward_speed(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int hidden_size = o_proj.in_features();
        const int qlen = (int)hidden_states->ne[1];

        KVCacheAttention::before_forward(ctx, n_past, qlen);

        ggml::tensor *kv_lora = d_kv_proj.forward(ctx, hidden_states);
        kv_lora = kv_norm.forward(ctx, kv_lora);

        ggml::tensor *tmpv = u_v_proj.forward(ctx, kv_lora);

        ggml::tensor *k_nope = u_k_nope_proj.forward(ctx, kv_lora);
        ggml::tensor *k_pe = k_pe_proj.forward(ctx, hidden_states);

        ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);

        ggml::tensor *scores = cross_attention_speed(ctx, hidden_size, n_past, qlen, tmpq, k_nope, k_pe, tmpv);

        ggml::tensor *attn_output = output_project(ctx, scores);

        return attn_output;
    }

    ggml::tensor *BaseMLAttention::cross_attention_speed(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                            ggml::tensor *q, ggml::tensor *k_nope, ggml::tensor *k_pe, ggml::tensor *v)
    {
        // [qlen, heads, head_size]
        k_pe = ggml::reshape_3d(ctx, k_pe, rope_dim, 1, qlen);
        k_pe = apply_pos_embedding_k(ctx, k_pe, rope_dim * 1, qlen, pos);

        k_nope = ggml::reshape_3d(ctx, k_nope, qk_nope_head_dim, num_kv_heads, qlen);
        k_pe   = ggml::repeat(ctx, k_pe, rope_dim, num_kv_heads, qlen);

        auto key_layer = ggml::concat(ctx, k_nope, k_pe, 0);

        // [qlen, heads, head_size]
        ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, qk_nope_head_dim + rope_dim, num_attention_heads, qlen);
        ggml::tensor * q_pe = ggml::view_3d(ctx, query_layer,
                        rope_dim, num_attention_heads, qlen,
                        query_layer->nb[1], query_layer->nb[2],
                        qk_nope_head_dim * ggml::element_size(query_layer));

        if (ctx->is_using_gpu())
        {
            // TODO: optimize (GPU rope requires continuous)
            ggml::tensor * q_pe_cont = ggml::cont(ctx, q_pe);
            q_pe_cont = apply_pos_embedding_q(ctx, q_pe_cont, rope_dim * num_attention_heads, qlen, pos);
            q_pe = ggml::cpy(ctx, q_pe_cont, q_pe);
        }
        else
        {
            q_pe = apply_pos_embedding_q(ctx, q_pe, rope_dim * num_attention_heads, qlen, pos);
        }

        ggml::build_forward_expand(ctx, q_pe);

        ggml::tensor *attn_scores = cross_attention_after_pe(ctx, hidden_size, n_past, qlen, query_layer, key_layer, v);

        return attn_scores;
    }

    ggml::tensor *BaseMLAttention::output_project(ComputeContext *ctx, ggml::tensor *scores)
    {
        ggml::tensor *attn_output = o_proj.forward(ctx, scores);

        return attn_output;
    }

    ggml::tensor *BaseMLAttention::forward_memory(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        const int hidden_size = o_proj.in_features();
        const int qlen = (int)hidden_states->ne[1];

        KVCacheAttention::before_forward(ctx, n_past, qlen);

        ggml::tensor *kv_lora = d_kv_proj.forward(ctx, hidden_states);
        kv_lora = kv_norm.forward(ctx, kv_lora);

        ggml::tensor *k_pe = k_pe_proj.forward(ctx, hidden_states);

        ggml::tensor *tmpq = q_proj.forward(ctx, hidden_states);

        ggml::tensor *scores = cross_attention_memory(ctx, hidden_size, n_past, qlen, tmpq, k_pe, kv_lora);

        ggml::tensor *attn_output = output_project(ctx, scores);

        return attn_output;
    }

    ggml::tensor *BaseMLAttention::cross_attention_memory(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                            ggml::tensor *q, ggml::tensor *k_pe, ggml::tensor *kv_lora)
    {
        // [qlen, heads, head_size]
        k_pe = ggml::reshape_3d(ctx, k_pe, rope_dim, 1, qlen);
        k_pe = apply_pos_embedding_k(ctx, k_pe, rope_dim * 1, qlen, pos);
        k_pe = ggml::reshape_1d(ctx, k_pe, rope_dim * 1 * qlen);

        // [qlen, heads, head_size]
        ggml::tensor * query_layer = ggml::reshape_3d(ctx, q, qk_nope_head_dim + rope_dim, num_attention_heads, qlen);
        ggml::tensor * q_pe = ggml::view_3d(ctx, query_layer,
                        rope_dim, num_attention_heads, qlen,
                        query_layer->nb[1], query_layer->nb[2],
                        qk_nope_head_dim * ggml::element_size(query_layer));

        if (ctx->is_using_gpu())
        {
            // TODO: optimize (GPU rope requires continuous)
            ggml::tensor * q_pe_cont = ggml::cont(ctx, q_pe);
            q_pe_cont = apply_pos_embedding_q(ctx, q_pe_cont, rope_dim * num_attention_heads, qlen, pos);
            q_pe = ggml::cpy(ctx, q_pe_cont, q_pe);
        }
        else
        {
            q_pe = apply_pos_embedding_q(ctx, q_pe, rope_dim * num_attention_heads, qlen, pos);
        }

        ggml::build_forward_expand(ctx, q_pe);

        ggml::tensor *attn_scores = cross_attention_after_pe_memory(ctx, hidden_size, n_past, qlen, query_layer, k_pe, kv_lora);

        return attn_scores;
    }

    ggml::tensor *BaseMLAttention::get_k_pe_from_cache(ComputeContext *ctx, const int n_past, const int qlen)
    {
        ggml::tensor *k_pe = nullptr;

        k_pe = ggml::view_2d(ctx, k_cache, k_hidden_size, n_past + qlen,
                            ggml::row_size(k_cache),
                            0);

        return k_pe;
    }

    ggml::tensor *BaseMLAttention::get_kv_lora_from_cache(ComputeContext *ctx, const int n_past, const int qlen)
    {
        ggml::tensor *kv_lora = nullptr;

        kv_lora = ggml::view_2d(ctx, v_cache, v_hidden_size, n_past + qlen,
                                v_hidden_size * ggml::element_size(v_cache),
                                0);

        return kv_lora;
    }

    void BaseMLAttention::save_lora_to_cache(ComputeContext *ctx, const int n_past, const int qlen,
        ggml::tensor *k_pe, ggml::tensor *kv_lora)
    {
        ggml::tensor * pe_cache_view = ggml::view_1d(ctx, k_cache, qlen * k_hidden_size,
                                    ggml::row_size(k_cache) * n_past);

        ggml::tensor * kv_cache_view = ggml::view_1d(ctx, v_cache, qlen * v_hidden_size,
                                    ggml::element_size(v_cache) * v_hidden_size * n_past);

        ggml::tensor * pe_view = ggml::view_1d(ctx, k_pe,    qlen * k_hidden_size, 0);
        ggml::tensor * kv_view = ggml::view_1d(ctx, kv_lora, qlen * v_hidden_size, 0);

        // important: storing RoPE-ed version of K in the KV cache!
        ggml::build_forward_expand(ctx, ggml::cpy(ctx, pe_view, pe_cache_view));
        ggml::build_forward_expand(ctx, ggml::cpy(ctx, kv_view, kv_cache_view));
    }

    ggml::tensor *BaseMLAttention::cross_attention_after_pe_memory(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen0,
                                            ggml::tensor *query_layer, ggml::tensor *k_pe, ggml::tensor *kv_lora)
    {
        const int head_size = qk_nope_head_dim + rope_dim;

        if (!attn_scaling)
            query_layer = ggml::scale(ctx, query_layer, 1.f / sqrtf((float)head_size));

        query_layer = ggml::permute(ctx, query_layer, 0, 2, 1, 3);                     // [heads, qlen, head_size]

        // store key and value to memory
        save_lora_to_cache(ctx, n_past, qlen0, k_pe, kv_lora);

        ggml::tensor *k_pe_all       = get_k_pe_from_cache(ctx, n_past, qlen0);
        ggml::tensor *kv_lora_all    = get_kv_lora_from_cache(ctx, n_past, qlen0);

        // make ggml ops happy
        kv_lora_all = ggml::cast(ctx, kv_lora_all, ggml::type::GGML_TYPE_F32);

        const int qlen = n_past + qlen0;

        ggml::tensor *k_nope = u_k_nope_proj.forward(ctx, kv_lora_all);

        ggml::tensor *key_layer = nullptr;

        k_nope   = ggml::reshape_3d(ctx, k_nope,     qk_nope_head_dim, num_kv_heads, qlen);
        k_pe_all = ggml::reshape_3d(ctx, k_pe_all,   rope_dim,         1,            qlen);
        k_pe_all = ggml::cast      (ctx, k_pe_all,   ggml::type::GGML_TYPE_F32);
        k_pe_all = ggml::repeat    (ctx, k_pe_all,   rope_dim,         num_kv_heads, qlen);

        key_layer = ggml::concat (ctx, k_nope,     k_pe_all,         0);
        key_layer = ggml::permute(ctx, key_layer,  0, 2, 1, 3); // [qlen, heads, head_size] -> [heads, qlen, head_size]

        ggml::tensor *value_layer = u_v_proj.forward(ctx, kv_lora_all);
        value_layer = ggml::reshape_3d(ctx, value_layer, v_head_dim, num_kv_heads, qlen); // [qlen, heads, head_size]
        value_layer = ggml::permute   (ctx, value_layer, 1, 2, 0, 3); // [qlen, heads, head_size] -> [heads, head_size, qlen]
        value_layer = ggml::cont      (ctx, value_layer);

        ggml::tensor *attn_scores = calc_attn_scores(ctx, hidden_size, n_past, qlen0, key_layer, query_layer, value_layer);
        return attn_scores;
    }
}

namespace chatllm::deepseek::v2
{
    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : v2_light::ConditionalGeneration0<NUM_EXPERTS, EXPERTS_PER_TOK, EXPERTS_PER_TOK>(config, runtime_config, MODEL_TYPE_DEEPSEEK_V2, config.q_lora_rank)
    {
        CHATLLM_CHECK(config.topk_group == 1) << "unsupported MoE param";
    }
}

namespace chatllm::deepseek::v3_light
{
    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v2_light::ConditionalGeneration(config, runtime_config, type, -1, BaseSparseMLP::ScoreFunc::Sigmoid, true, true)
    {
    }
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(DEEPSEEK,              deepseek::v1, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V2_LIGHT,     deepseek::v2_light, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V2,           deepseek::v2, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V3_LIGHT,     deepseek::v3_light, 1);
    REGISTER_MODEL_LOADER(DEEPSEEK_V1_MoE,       deepseek::v1_moe, 1);
}