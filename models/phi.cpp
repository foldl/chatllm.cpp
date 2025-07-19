#include "phi.h"

namespace chatllm::phi::v2
{
    void QAHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);

        std::ostringstream oss_prompt;
        oss_prompt << ai << "\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void QAHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Instruct: " << user << "\n";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void QAHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Output: ";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        append_ai_opening(round_idx, ids);

        std::ostringstream oss_prompt;
        oss_prompt << ai << "\n";
        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Alice: " << user << "\n";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        std::ostringstream oss_prompt;

        oss_prompt << "Bob: ";

        auto text = oss_prompt.str();
        tokenizer->encode(text, ids);
    }

    static QAHistoryEncoder _qa_encoder;
    static ChatHistoryEncoder _chat_encoder;

    Phi2Tokenizer::Phi2Tokenizer(const BaseConfig &config)
        : BaseTokenizer::BaseTokenizer(config, &_chat_encoder, &_qa_encoder),
        qa_seq_max_len(0)
    {
    }

    Phi2Tokenizer::Phi2Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder),
        qa_seq_max_len(0)
    {
    }

    Phi2ConditionalGeneration::Phi2ConditionalGeneration(const BaseConfig &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 5 + config.num_hidden_layers * 17;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        transformer = new ModelClass(&w_ctx_, config, true,
                            config.hidden_size, config.num_attention_heads,
                            config.intermediate_size, config.num_attention_heads, config.max_length);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            get_typed_transformer<ModelClass>()->layers[i].set_id(i);
            get_typed_transformer<ModelClass>()->layers[i].attention.set_id(i);
            get_typed_transformer<ModelClass>()->layers[i].attention.set_prec(ggml::prec::GGML_PREC_F32);
        }
    }


    bool Phi2ConditionalGeneration::is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output)
    {
        if (output_ids.size() < 1) return false;

        Phi2Tokenizer *tokenizer = dynamic_cast<Phi2Tokenizer *>(this->tokenizer);
        int len = 0;

        switch (tokenizer->get_chat_format())
        {
        case ChatFormat::QA:
            if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq1))
            {
                pop_output = (int)tokenizer->qa_terminate_seq1.size();
                return true;
            }
            else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq2))
            {
                pop_output = (int)tokenizer->qa_terminate_seq2.size();
                return true;
            }
            else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq3))
            {
                pop_output = (int)tokenizer->qa_terminate_seq3.size();
                return true;
            }
            else if (match_output_sequence(output_ids, tokenizer->qa_terminate_seq4))
            {
                pop_output = (int)tokenizer->qa_terminate_seq4.size();
                return true;
            }
            len = tokenizer->qa_seq_max_len;
            break;
        case ChatFormat::CHAT:
            if (match_output_sequence(output_ids, tokenizer->chat_terminate_seq))
            {
                pop_output = (int)tokenizer->chat_terminate_seq.size();
                return true;
            }
            len = (int)tokenizer->chat_terminate_seq.size();
            break;
        default:
            ;
        }

        if (BaseModelForConditionalGeneration::is_output_terminated(output_ids, keep_idx, pop_output))
            return true;

        keep_idx = (int)(output_ids.size()) - len + 1;
        return false;
    }

    #define MAX_IT(x)  if (qa_seq_max_len < (int)x.size()) qa_seq_max_len = (int)x.size()

    size_t Phi2Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);
        tp->Encode("\nInstruct:", &qa_terminate_seq1);
        tp->Encode("\nInstruction:", &qa_terminate_seq2);
        tp->Encode("\nUser:", &qa_terminate_seq3);
        tp->Encode("\nINPUT:", &qa_terminate_seq4);
        tp->Encode("\nAlice", &chat_terminate_seq);
        bos_token_id = eos_token_id = tp->PieceToId("<|endoftext|>");

        MAX_IT(qa_terminate_seq1);
        MAX_IT(qa_terminate_seq2);
        MAX_IT(qa_terminate_seq3);
        MAX_IT(qa_terminate_seq4);

        return size;
    }

    bool Phi2Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id);
    }

    namespace v1
    {

        Tokenizer::Tokenizer(const BaseConfig &config)
            : Phi2Tokenizer(config)
        {
        }

        ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : ConditionalGeneration::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_PHI2)
        {

        }

        ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
            : Phi2ConditionalGeneration(config, runtime_config, type)
        {

        }

        void ConditionalGeneration::load(ModelLoader &loader)
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("transformer.embd.wte.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "transformer.h." + std::to_string(layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "ln.bias",   transformer->layers[i].input_layernorm.bias);
                loader.read_tensor(layer_prefix + "ln.weight", transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "mixer.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "mixer.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "mixer.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "mixer.out_proj.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "mixer.out_proj.weight", transformer->layers[i].attention.o_proj.weight);

                loader.read_tensor(layer_prefix + "mlp.fc1.bias",   transformer->layers[i].mlp.fc0.bias);
                loader.read_tensor(layer_prefix + "mlp.fc1.weight", transformer->layers[i].mlp.fc0.weight);
                loader.read_tensor(layer_prefix + "mlp.fc2.bias",   transformer->layers[i].mlp.fc1.bias);
                loader.read_tensor(layer_prefix + "mlp.fc2.weight", transformer->layers[i].mlp.fc1.weight);
            }
            transformer->final_layernorm->load("lm_head.ln.", &loader);

            loader.read_tensor("lm_head.linear.bias", dynamic_cast<Linear *>(transformer->lm_head)->bias);
            loader.read_tensor("lm_head.linear.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }
    }

    namespace v2
    {
        Tokenizer::Tokenizer(const BaseConfig &config)
            : Phi2Tokenizer(config)
        {
        }

        ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
            : Phi2ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_PHI2)
        {
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                get_typed_transformer<ModelClass>()->layers[i].attention.rope_dim = config.rope_dim;
                get_typed_transformer<ModelClass>()->layers[i].attention.freq_base = config.rope_theta;
            }
        }

        void ConditionalGeneration::load(ModelLoader &loader)
        {
            auto transformer = get_typed_transformer<ModelClass>();
            transformer->word_embeddings->load("model.embed_tokens.", &loader);
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

                loader.read_tensor(layer_prefix + "input_layernorm.bias",   transformer->layers[i].input_layernorm.bias);
                loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);

                loader.read_tensor(layer_prefix + "self_attn.q_proj.bias",   transformer->layers[i].attention.q_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.bias",   transformer->layers[i].attention.k_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.bias",   transformer->layers[i].attention.v_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
                loader.read_tensor(layer_prefix + "self_attn.dense.bias",   transformer->layers[i].attention.o_proj.bias);
                loader.read_tensor(layer_prefix + "self_attn.dense.weight", transformer->layers[i].attention.o_proj.weight);

                loader.read_tensor(layer_prefix + "mlp.fc1.bias",   transformer->layers[i].mlp.fc0.bias);
                loader.read_tensor(layer_prefix + "mlp.fc1.weight", transformer->layers[i].mlp.fc0.weight);
                loader.read_tensor(layer_prefix + "mlp.fc2.bias",   transformer->layers[i].mlp.fc1.bias);
                loader.read_tensor(layer_prefix + "mlp.fc2.weight", transformer->layers[i].mlp.fc1.weight);
            }
            transformer->final_layernorm->load("model.final_layernorm.", &loader);

            loader.read_tensor("lm_head.bias", dynamic_cast<Linear *>(transformer->lm_head)->bias);
            loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

            CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
                << "corrupted model weights";
        }
    }
}

namespace chatllm::phi::v3
{
    static ChatHistoryEncoder _chat_encoder;

    Phi3Tokenizer::Phi3Tokenizer(const BaseConfig &config)
        : Phi3Tokenizer(config, &_chat_encoder)
    {
    }

    Phi3Tokenizer::Phi3Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder),
            append_nl_after_end_tok(false)
    {
    }

    size_t Phi3Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        system_token_id     = tp->PieceToId("<|system|>");
        user_token_id       = tp->PieceToId("<|user|>");
        assistant_token_id  = tp->PieceToId("<|assistant|>");
        end_token_id        = tp->PieceToId("<|end|>");
        nl_token_id         = tp->PieceToId("\n");

        if (-1 == system_token_id)
        {
            CHATLLM_CHECK(tp->GetPieceSize() == 32000) << " unsupported tokenizer";
            system_token_id     = 32006;
            user_token_id       = 32010;
            assistant_token_id  = 32001;
            end_token_id        = 32007;
        }

        pad_token_id = eos_token_id;

        terminate_ids.insert(end_token_id);

        return size;
    }

    void Phi3Tokenizer::encode(const std::string &msg, std::vector<int> &ids, int type_token_id, int end_token_id)
    {
        if (type_token_id >= 0)
        {
            ids.push_back(type_token_id);
            if (nl_token_id >= 0)
                ids.push_back(nl_token_id);
        }
        BaseTokenizer::encode(msg, ids);
        if (end_token_id >= 0)
        {
            ids.push_back(end_token_id);
            if (append_nl_after_end_tok)
                ids.push_back(nl_token_id);
        }
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length)
        : llama::v2::GenericConditionalGeneration<Phi3Block4k>(config, runtime_config, type, num_key_value_heads, max_length, 13)
    {
        CHATLLM_CHECK(config.sliding_window == SLIDING_WINDOW_LEN - 1)
            << "sliding_window (" << config.sliding_window << ") must be " << SLIDING_WINDOW_LEN - 1;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }

        batch_input = false;
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids, -1, tok->end_token_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (add_bos)
            ids.push_back(tok->bos_token_id);

        if (tok->get_system_prompt().size() > 0)
            tok->encode(tok->get_system_prompt(), ids, tok->system_token_id, tok->end_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        tok->encode(user, ids, tok->user_token_id, tok->end_token_id);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode("", ids, tok->assistant_token_id, -1);
    }
}

namespace chatllm::phi::v3_su
{
    static int get_factor_len(const Config &config)
    {
        int i = MAX_FACTOR_LEN - 1;
        for (; (i >= 0) && (config.short_factor[i] == 0.0f); i--) ;
        return i + 1;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length, bool tie_lm_head)
        : llama::v2::GenericConditionalGeneration<Phi3SUBlock>(config, runtime_config, type, num_key_value_heads, max_length, 13, tie_lm_head)
    {
        CHATLLM_CHECK(config.sliding_window >= config.max_length)
            << "sliding_window (" << config.sliding_window << ") must >= " << config.max_length;

        CHATLLM_CHECK(config.rope_scaling == 1)
            << "rope_scaling (" << config.rope_scaling << ") must == " << 1;

        float scaling_factor = (float)config.max_position_embeddings / config.original_max_position_embeddings;
        if (scaling_factor <= 1.0f)
            scaling_factor = 1.0f;
        else
            scaling_factor = sqrtf(1.0f + logf(scaling_factor) / logf((float)config.original_max_position_embeddings));

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.config(&w_ctx_, config.original_max_position_embeddings, config.rope_theta,
                                scaling_factor,
                                scaling_factor,
                                get_factor_len(config),
                                config.short_factor,
                                config.long_factor);
        }
    }
}

namespace chatllm::phi::v3_su2
{
    Tokenizer::Tokenizer(const BaseConfig &config) : v3::Tokenizer(config, &v3::_chat_encoder)
    {
        append_nl_after_end_tok = true;
        v3::_chat_encoder.add_bos = false;
    }
}

namespace chatllm::phi::v3_su3
{
    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v3_su2::ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = get_typed_transformer<ModelClass>()->layers[i].attention;
            attention.config(&w_ctx_, config.original_max_position_embeddings, config.rope_theta,
                                config.short_mscale,
                                config.long_mscale,
                                config.hidden_size / config.num_attention_heads / 2,
                                config.short_factor,
                                config.long_factor);
        }
    }
}

namespace chatllm::phi::v3_moe
{
    static void _compute_forward_custom_sparsemixer_f32(ggml::tensor * dst, const ggml::tensor * a, const ggml::tensor * b, const ggml::tensor * c, int ith, int nth, const mixing_param *userdata)
    {
        CHATLLM_CHECK(nth == 1) << "nth must be 1";
    }

    void _compute_forward_custom_sparsemixer(ggml::tensor * dst, const ggml::tensor * a, const ggml::tensor * b, const ggml::tensor * c, int ith, int nth, void *userdata)
    {
        switch (ggml::type_of(c))
        {
            case GGML_TYPE_F32:
                _compute_forward_custom_sparsemixer_f32(dst, a, b, c, ith, nth, (const mixing_param *)userdata);
                break;
            default:
                CHATLLM_CHECK(false) << "_compute_forward_custom_sparsemixer: unsupported type " << ggml::type_of(c);
                break;
        }
    }

    Phi3SUSelfAttentionBiased::Phi3SUSelfAttentionBiased(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
        : Phi3SUSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true)
    {}
}

namespace chatllm::phi::v4
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : BaseTokenizer(config, &_chat_encoder)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);

        im_start_token_id     = tp->PieceToId("<|im_start|>");
        im_sep_token_id       = tp->PieceToId("<|im_sep|>");
        im_end_token_id       = tp->PieceToId("<|im_end|>");

        pad_token_id = eos_token_id;

        terminate_ids.insert(im_end_token_id);

        return size;
    }

    void Tokenizer::encode_role(const std::string &msg, std::vector<int> &ids)
    {
        ids.push_back(im_start_token_id);
        encode(msg, ids, false);
        ids.push_back(im_sep_token_id);
    }

    void Tokenizer::encode(const std::string &msg, std::vector<int> &ids, bool add_end_tok)
    {
        BaseTokenizer::encode(msg, ids);
        if (add_end_tok)
        {
            ids.push_back(im_end_token_id);
        }
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode_role("system", ids);
            tok->encode(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_user_opening(round_idx, ids);
        tok->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role("assistant", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_role("user", ids);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v3::ConditionalGeneration(config, runtime_config, MODEL_TYPE_PHI4)
    {
    }
}

namespace chatllm::phi::v4_mini
{
    Tokenizer::Tokenizer(const BaseConfig &config)
        : v3_su::Tokenizer(config)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2();
        size_t size = tp->Load(buffer, n_vocab);

        system_token_id     = tp->PieceToId("<|system|>");
        user_token_id       = tp->PieceToId("<|user|>");
        assistant_token_id  = tp->PieceToId("<|assistant|>");
        end_token_id        = tp->PieceToId("<|end|>");
        nl_token_id         = -1;

        pad_token_id = eos_token_id;
        terminate_ids.insert(end_token_id);

        return size;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v3_su::ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length, true)
    {}
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(PHI2,                  phi::v2::v1, 1);
    REGISTER_MODEL_LOADER(PHI2_V2,               phi::v2::v2, 1);
    REGISTER_MODEL_LOADER(PHI3,                  phi::v3, 1);
    REGISTER_MODEL_LOADER(PHI3_SU,               phi::v3_su, 1);
    REGISTER_MODEL_LOADER(PHI3_SU2,              phi::v3_su2, 1);
    REGISTER_MODEL_LOADER(PHI3_SU3,              phi::v3_su3, 1);
    REGISTER_MODEL_LOADER(PHI3_MOE,              phi::v3_moe, 1);
    REGISTER_MODEL_LOADER(PHI4,                  phi::v4, 1);
    REGISTER_MODEL_LOADER(PHI4_MINI,             phi::v4_mini, 1);
}