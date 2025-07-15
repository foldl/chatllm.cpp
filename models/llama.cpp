#include "llama.h"
#include <numbers>

namespace chatllm::llama::v2
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = R"""(You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.)""";
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_attention_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length)
        : GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor1();
        size_t size = tp->Load(buffer, n_vocab);

        pad_token_id = tp->PieceToId("<pad>");
        return size;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
    {
        if (add_bos)
            ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            ids.push_back(eos_token_id);
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids, false, false);
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
        std::ostringstream oss_prompt;

        oss_prompt << "<<SYS>>\n" << tok->get_system_prompt() << "\n<</SYS>>\n\n";
        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss_prompt;

        oss_prompt << "[INST] " + user << "[/INST] ";

        auto text = oss_prompt.str();
        tok->encode(text, ids, true, false);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return id == pad_token_id;
    }
}

namespace chatllm::llama::v3
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const BaseConfig &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder), is_seed_coder(false)
    {
        sys_prompt = "";
    }

    bool Tokenizer::load_config(const json::JSON &config)
    {
        auto name = config["model_name"];
        if (name.IsString())
        {
            is_seed_coder = name.ToString() == "Seed-Coder";
            if (is_seed_coder)
                sys_prompt = "You are an AI programming assistant, utilizing the Seed-Coder model, developed by ByteDance Seed, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n";
        }
        return BaseTokenizer::load_config(config);
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);

        start_header_id = tp->PieceToId("<|start_header_id|>");
        end_header_id   = tp->PieceToId("<|end_header_id|>");
        eot_id          = tp->PieceToId("<|eot_id|>");

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        if (eot_id >= 0)
            terminate_ids.insert(eot_id);

        return size;
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == start_header_id) || (id == end_header_id) || (id == eot_id);
    }

    void Tokenizer::encode_header(const std::string &text, std::vector<int> &ids) const
    {
        // compatible with Seed-Coder
        if (start_header_id >= 0)
        {
            ids.push_back(start_header_id);
            encode(text, ids);
            ids.push_back(end_header_id);
            ids.push_back(nl_token_id);
            ids.push_back(nl_token_id);
        }
        else
        {
            ids.push_back(bos_token_id);
            encode(text, ids);
            ids.push_back(nl_token_id);
        }
    }

    void Tokenizer::encode_content(const std::string &text, std::vector<int> &ids) const
    {
        encode(text, ids);
        ids.push_back(eot_id >= 0 ? eot_id : eos_token_id);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        append_ai_opening(round_idx, ids);
        tok->encode(ai, ids);
        ids.push_back(tok->eot_id);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode_header("system", ids);
            tok->encode_content(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        tok->encode_header("user", ids);
        tok->encode_content(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_header("assistant", ids);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        tok->encode_header("user", ids);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length)
        : v2::GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length)
    {
        auto transformer = Base::get_typed_transformer<ModelClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }
}

namespace chatllm::llama::v3_1
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : v3::Tokenizer(config, &_chat_encoder)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t size = v3::Tokenizer::load(buffer, n_vocab);

        eom_id          = tp->PieceToId("<|eom_id|>");
        python_tag_id   = tp->PieceToId("<|python_tag|>");

        terminate_ids.insert(eom_id);

        tp->EnableReturnSpecialToken(true);

        return size;
    }

    void ChatHistoryEncoder::append_tool(int round_idx, const std::string &content, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        std::ostringstream oss;

        tok->encode_header("ipython", ids);
        tok->encode_content(content, ids);
    }

    ToolCallingInterceptor::ToolCallingInterceptor() : ChunkInterceptor(), found_tool_call(false)
    {}

    void ToolCallingInterceptor::put_chunk(bool first, const std::string &chunk)
    {
        if (!found_tool_call)
        {
            if (chunk == "<|python_tag|>")
            {
                found_tool_call = true;
            }
            else
                next->put_chunk(first, chunk);
        }
        else
        {
            oss << chunk;
        }
    }

    void ToolCallingInterceptor::end()
    {
        found_tool_call = false;
        auto s = oss.str();
        oss.str("");
        if (s.size() > 0)
            streamer->putln(s.c_str(), BaseStreamer::TextType::TOOL_CALLING);

        ChunkInterceptor::end();
    }

    static class ToolCallingInterceptor interceptor;

    void init_llama3_freq_factors(std::vector<float> &freq_factors, int rope_dim,
                float theta, float scaling_factor, float scaling_low_freq_factor, float scaling_high_freq_factor, int original_max_position_embeddings)
    {
        std::vector<float > inv_freq;
        const float base = theta;

        for (int i = 0; i < rope_dim; i += 2)
        {
            inv_freq.push_back(1.0f / (powf(base,  (float)i / rope_dim)));
        }

        freq_factors.clear();

        const float factor = scaling_factor;
        const float low_freq_factor  = scaling_low_freq_factor;
        const float high_freq_factor = scaling_high_freq_factor;
        const int   old_context_len  = original_max_position_embeddings;

        const float low_freq_wavelen = old_context_len / low_freq_factor;
        const float high_freq_wavelen = old_context_len / high_freq_factor;

        for (int i = 0; i < (int)inv_freq.size(); i++)
        {
            const float freq = inv_freq[i];
            const float wavelen = 2 * std::numbers::pi_v<float> / freq;
            if (wavelen < high_freq_wavelen)
            {
                freq_factors.push_back(1.0f);
            }
            else if (wavelen > low_freq_wavelen)
            {
                freq_factors.push_back(1.0f / factor);
            }
            else
            {
                const float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                freq_factors.push_back((1 - smooth) / factor + smooth);
            }
        }

        for (int i = 0; i < (int)freq_factors.size(); i++)
        {
            // factor is used as a divider in ggml
            freq_factors[i] = 1 / freq_factors[i];
        }
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length)
    {}

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length, int tensors_per_layer, bool tie_lm_head, int additional_tensor)
        : v2::GenericConditionalGeneration<Llama31Block>(config, runtime_config, type, num_key_value_heads, max_length, tensors_per_layer, tie_lm_head, additional_tensor)
    {
        std::vector<float> freq_factors_value;
        init_llama3_freq_factors(freq_factors_value, config.hidden_size / config.num_attention_heads,
                                    config.rope_theta,
                                    config.rope_scaling_factor,
                                    config.rope_scaling_low_freq_factor,
                                    config.rope_scaling_high_freq_factor,
                                    config.rope_scaling_original_max_position_embeddings);

        auto transformer = Base::get_typed_transformer<ModelClass>();
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base    = config.rope_theta;
            Backend::write_tensor_data(attention.freq_factors, freq_factors_value.data());
        }
    }

    ChunkInterceptor *ConditionalGeneration::get_interceptor(void)
    {
        return &interceptor;
    }
}

namespace chatllm::llama::v3_2
{
    typedef v3_1::Tokenizer Tokenizer;

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v3_1::ConditionalGeneration(config, runtime_config, type, config.num_key_value_heads, config.max_length,
                                        13, config.tie_word_embeddings != 0, 0)
    {}
}

namespace chatllm::llama::v2_plus
{
    typedef v3::Config Config;
    typedef v2::Tokenizer Tokenizer;

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v3::ConditionalGeneration(config, runtime_config, type)
    {}
}

namespace chatllm::llama::multi
{
    Tokenizer::Tokenizer(const Config &config)
        : v2::Tokenizer(config)
    {
        // sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        size_t r = v2::Tokenizer::load(buffer, n_vocab);
        return r;
    }

    MultiPredModel::MultiPredModel(InitContext *ctx, const Config &config, int hidden_size, int num_attention_heads, int intermediate_size,
                    int num_key_value_heads, int max_length, int n_future_tokens)
        : Model<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int>(
                ctx, config, false, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length),
            config(config),
            n_future_tokens(n_future_tokens)
    {
        effective_n = n_future_tokens;

        for (int i = 0; i < n_future_tokens - 1; i++)
        {
            auto layer = new LlamaBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, max_length);
            extra_heads.push_back(layer);
            layer->set_id(config.num_hidden_layers + i);
        }

        prediction_heads.push_back(get_layer(config.num_hidden_layers - 1));
        for (int i = 0; i < n_future_tokens - 1; i++)
        {
            auto layer = extra_heads[i];
            prediction_heads.push_back(layer);

            ctx->move_to_layer(i);
            auto allocator = ctx->get_allocator();
            auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
            layer->set_cache_buffer(buf);
        }
    }

    MultiPredModel::~MultiPredModel()
    {
    }

    ggml::tensor *MultiPredModel::forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past)
    {
        ggml::tensor *hidden_states = word_embeddings->forward(ctx, input_ids);
        for (int i = 0; i <= config.num_hidden_layers - 2; i++)
        {
            hidden_states = get_layer(i)->forward(ctx, hidden_states, n_past);
        }

        auto h_trunk = hidden_states;
        ggml::tensor *lm_logits = ggml::new_tensor_2d(ctx, GGML_TYPE_F32, config.vocab_size, effective_n);

        for (int i = 0; i < effective_n; i++)
        {
            ggml::tensor *tok_states = prediction_heads[i]->forward(ctx, h_trunk, n_past);

            ggml::tensor *logits = calc_logits(ctx, input_ids, tok_states);
            ggml::tensor *view = ggml::view_1d(ctx, lm_logits, config.vocab_size,
                                                i * ggml::nbytes(logits));
            ggml::build_forward_expand(ctx, ggml::cpy(ctx, logits, view));
        }

        return lm_logits;
    }

    int MultiPredModel::set_n_future_tokens(int n)
    {
        if ((1 <= n) && (n <= n_future_tokens))
            effective_n = n;
        return effective_n;
    }

    int64_t MultiPredModel::get_param_num_of_layers(bool effective_only) const
    {
        int64_t r = Base::get_param_num_of_layers(effective_only);
        if (extra_heads.size() > 0)
            r += extra_heads[0]->get_param_num(effective_only) * extra_heads.size();
        return r;
    }

    ggml::tensor *MultiPredModel::calc_logits(ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
    {
        // NOTE: only compute next_token_logits for the last token
        hidden_states = ggml::view_2d(ctx, hidden_states, config.hidden_size, 1,
                                    config.hidden_size * ggml::element_size(hidden_states),
                                    (input_ids->ne[0] - 1) * config.hidden_size * ggml::element_size(hidden_states));

        ggml::tensor *transformer_outputs = final_layernorm->forward(ctx, hidden_states);

        transformer_outputs =
                ggml::view_1d(ctx, transformer_outputs, config.hidden_size, 0);

        ggml::tensor *logits = lm_head ? lm_head->forward(ctx, transformer_outputs)
                                            : word_embeddings->forward(ctx, transformer_outputs);

        if (logits_pp)
            logits = logits_pp->forward(ctx, logits);
        return logits;
    }


    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                                    int tensors_per_layer)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + (config.num_hidden_layers + config.n_future_tokens - 1) * tensors_per_layer;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        Base::transformer = new MultiPredModel(&w_ctx_, config,
                                                config.hidden_size, config.num_attention_heads,
                                                config.intermediate_size, config.num_key_value_heads, config.max_length,
                                                config.n_future_tokens);
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        auto transformer = Base::get_typed_transformer<MultiPredModel>();
        transformer->word_embeddings->load("model.embed_tokens.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight", transformer->layers[i].mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight", transformer->layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight", transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
        }

        for (int i = 0; i < (int)transformer->extra_heads.size(); i++)
        {
            std::string layer_prefix = "model.extra_heads." + std::to_string(i) + '.';
            loader.read_tensor(layer_prefix + "input_layernorm.weight",             transformer->extra_heads[i]->input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.down_proj.weight",               transformer->extra_heads[i]->mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.gate_proj.weight",               transformer->extra_heads[i]->mlp.gate_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.up_proj.weight",                 transformer->extra_heads[i]->mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",    transformer->extra_heads[i]->post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->extra_heads[i]->attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->extra_heads[i]->attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->extra_heads[i]->attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->extra_heads[i]->attention.v_proj.weight);
        }
        transformer->final_layernorm->load("model.norm.", &loader);
        loader.read_tensor("lm_head.weight", dynamic_cast<Linear *>(transformer->lm_head)->weight);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        #define get_value(n) do {                       \
            auto it = args.find(#n);                    \
            if (it != args.end()) n = it->second;       \
        } while (0)

        std::string n_future_tokens("");
        get_value(n_future_tokens);
        if (n_future_tokens.size() > 0)
            dynamic_cast<MultiPredModel *>(Base::transformer)->set_n_future_tokens(std::atoi(n_future_tokens.c_str()));
    }
}

namespace chatllm::llama::ds_r1_distill
{
    typedef v3_2::Config Config;

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
        );

        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        user_token_id           = tp->PieceToId("<｜User｜>");
        assistant_token_id      = tp->PieceToId("<｜Assistant｜>");

        std::vector<int> ids;
        tp->Encode("\n", &ids);

        nl_token_id = -1;

        if (ids.size()  == 1)
            nl_token_id = ids[0];

        bos_token_id            = tp->PieceToId("<｜begin▁of▁sentence｜>");
        eos_token_id            = tp->PieceToId("<｜end▁of▁sentence｜>");

        return size;
    }


    void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->user_token_id);
        tok->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        ids.push_back(tok->assistant_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : v3_2::ConditionalGeneration(config, runtime_config, type)
    {}
}

namespace chatllm::llama::v4
{
    Tokenizer::Tokenizer(const Config &config)
        : v3::Tokenizer(config)
    {}

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        // TODO: FIX regexp
        size_t size = v3::Tokenizer::load(buffer, n_vocab);

        start_header_id = tp->PieceToId("<|header_start|>");
        end_header_id   = tp->PieceToId("<|header_end|>");
        eot_id          = tp->PieceToId("<|eot|>");

        terminate_ids.insert(eot_id);

        return size;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config) : ModelProxy()
    {
        if (config.use_qk_norm)
        {
            switch (config.n_routed_experts)
            {
            case 16:
                set_proxy_model(new ConditionalGeneration0<16, 1, 1, LlamaNormedSelfAttention>(config, runtime_config));
                break;
            default:
                CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.n_routed_experts;
                break;
            }
        }
        else
        {
            switch (config.n_routed_experts)
            {
            case 128:
                set_proxy_model(new ConditionalGeneration0<128, 1, 1, LlamaSelfAttention>(config, runtime_config));
                break;
            default:
                CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.n_routed_experts;
                break;
            }
        }
    }
}