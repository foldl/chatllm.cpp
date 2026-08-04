#include "llama.h"
#include "../src/chat_encoders.h"

namespace chatllm::nanbeige
{
    struct Config : BaseConfig
    {
        int num_key_value_heads;
        int head_dim;
        int num_loops;
        int skip_loop_final_norm;
        int tie_word_embeddings;
        float rope_theta;
    };

    static HistoryEncoderImStartImEnd _chat_encoder;

    class Tokenizer : public BaseTokenizer
    {
    public:

        Tokenizer(const BaseConfig &config):
            BaseTokenizer(config, nullptr)
        {
            sys_prompt = "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。";
        }

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            tp = new tokenizer::BPEProcessor1();
            size_t size = tp->Load(buffer, n_vocab);
            return size;
        }

        bool load_config(const json::JSON &config) override
        {
            auto cfg = config["tokenizer_config.json"]["added_tokens_decoder"];
            for (auto &tok : cfg.ObjectRange())
            {
                tp->AddAddedToken(tok.second["content"].ToString(), (int)std::atoi(tok.first.c_str()));
            }

            set_chat_encoder(&_chat_encoder);

            return true;
        }
    };

    class LoopedLayer : public Block
    {
    public:
        LoopedLayer(InitContext *ctx, Block *layer_onto, KVCacheAttention *attention);

        int64_t get_param_num(bool effective_only) const override;

        size_t get_cache_size(void) const override
        {
            size_t r = 0;
            if (k_cache)
                r += ggml::nbytes(k_cache);
            if (v_cache)
                r += ggml::nbytes(v_cache);
            return r;
        }

        void  set_cache_buffer(BackendBuffer *buffer) override
        {
            size_t offset = 0;
            if (k_cache)
            {
                buffer->assign_to(k_cache, offset);
                offset += ggml::nbytes(k_cache);
            }
            if (v_cache)
            {
                buffer->assign_to(v_cache, offset);
            }
        }

        size_t read_cache_data(void *buffer, size_t buffer_size) const override;
        size_t write_cache_data(const void *buffer, size_t buffer_size) override;

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override;
    public:
        ggml::tensor *k_cache;
        ggml::tensor *v_cache;
    protected:
        Block * const onto;
        KVCacheAttention * const attention;
    };

    LoopedLayer::LoopedLayer(InitContext *ctx, Block *layer_onto, KVCacheAttention *attention):
        Block(),
        k_cache(ggml::new_tensor_like(ctx, attention->k_cache)),
        v_cache(ggml::new_tensor_like(ctx, attention->v_cache)),
        onto(layer_onto),
        attention(attention)
    {
    }

    int64_t LoopedLayer::get_param_num(bool effective_only) const
    {
        return effective_only ? onto->get_param_num(true) : 0;
    }

    ggml::tensor *LoopedLayer::forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past)
    {
        ggml::tensor *k_bak = attention->k_cache;
        ggml::tensor *v_bak = attention->v_cache;

        attention->k_cache = k_cache;
        attention->v_cache = v_cache;

        auto r = onto->forward(ctx, hidden_states, n_past);

        attention->k_cache = k_bak;
        attention->v_cache = v_bak;

        return r;
    }

    size_t LoopedLayer::read_cache_data(void *buffer, size_t buffer_size) const
    {
        size_t r = 0;
        uint8_t *p = (uint8_t *)buffer;
        if (k_cache)
        {
            size_t s = ggml::nbytes(k_cache) <= buffer_size ? ggml::nbytes(k_cache) : buffer_size;
            Backend::read_tensor_data(k_cache, p, 0, s);
            r += s;
            buffer_size -= s;
            p += s;
        }
        if (v_cache && (buffer_size > 0))
        {
            size_t s = ggml::nbytes(v_cache) <= buffer_size ? ggml::nbytes(v_cache) : buffer_size;
            Backend::read_tensor_data(v_cache, p, 0, s);
            r += s;
        }
        return r;
    }

    size_t LoopedLayer::write_cache_data(const void *buffer, size_t buffer_size)
    {
        size_t r = 0;
        const uint8_t *p = (const uint8_t *)buffer;
        if (k_cache)
        {
            size_t s = ggml::nbytes(k_cache) <= buffer_size ? ggml::nbytes(k_cache) : buffer_size;
            Backend::write_tensor_data(k_cache, p, 0, s);
            r += s;
            buffer_size -= s;
            p += s;
        }
        if (v_cache && (buffer_size > 0))
        {
            size_t s = ggml::nbytes(v_cache) <= buffer_size ? ggml::nbytes(v_cache) : buffer_size;
            Backend::write_tensor_data(v_cache, p, 0, s);
            r += s;
        }
        return r;
    }

    template <class Config, class Embedding, class FinalNorm, class LayerBlock, typename... _Types> class LoopModel :
        public HeterogeneousModel
    {
    private:
        typedef HeterogeneousModel Base;
    protected:
        class Accessor
        {
        friend LoopModel;
        protected:
            Accessor() : m(nullptr) {}
        public:
            LayerBlock & operator[](int index)
            {
                if (nullptr == m)
                {
                    uintptr_t offset = (uintptr_t)&(((LoopModel *)(nullptr))->layers);
                    m = (LoopModel *)(uintptr_t(this) - offset);
                }
                return *(dynamic_cast<LayerBlock *>((m->Base::layers)[index])); // .get()));
            }
        private:
            LoopModel *m;
        };
    public:
        LoopModel() = default;
        LoopModel(InitContext *ctx, const Config &config, int num_loops, bool lm_head_bias, _Types... layer_args)
            : LoopModel(ctx, config, num_loops,
                create_lm_head(ctx, config, lm_head_bias),
                std::forward<_Types>(layer_args)...)
        {}

        LoopModel(InitContext *ctx, const Config &config, int num_loops, Block *lm_head, _Types... layer_args) :
            HeterogeneousModel(ctx, config.num_hidden_layers, config.hidden_size,
                            create_embedding<Embedding>(ctx, config),
                            create_final_norm<FinalNorm>(ctx, config),
                            lm_head,
                            [&](InitContext *ctx, int layer_index) {
                                return new LayerBlock(ctx, std::forward<_Types>(layer_args)...);
                            }),
            num_loops(num_loops),
            num_hidden_layers(config.num_hidden_layers)
        {
            if (num_loops <= 1) return;
            for (int i = 1; i < num_loops; i++)
            {
                for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
                {
                    ctx->move_to_layer(layer_id);
                    {
                        auto layer = (LayerBlock *)(Base::layers[layer_id]);
                        looped_layers.emplace_back(ctx, layer, &layer->attention);
                    }

                    auto layer = &looped_layers.back();
                    layer->set_id(i * num_hidden_layers + layer_id);
                    cache_size += layer->get_cache_size();

                    auto allocator = ctx->get_allocator();
                    auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
                    layer->set_cache_buffer(buf);
                }
            }
        }

        void before_eval(ComputeContext *ctx) override
        {
            Base::before_eval(ctx);
            for (auto &layer : looped_layers)
            {
                layer.before_eval(ctx);
            }
        }

        void set_ctx(int n_ctx) override
        {
            Base::set_ctx(n_ctx);
            for (auto &layer : looped_layers)
                layer.set_ctx(n_ctx);
        }

        void shift_cache(int shift, int total) override
        {
            Base::shift_cache(shift, total);
            for (auto &layer : looped_layers)
                layer.shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = Base::get_param_num(effective_only);
            for (auto &layer : looped_layers)
                r += layer.get_param_num(effective_only);
            return r;
        }

        int    get_layer_num(void) const
        {
            int r = Base::get_layer_num();
            r += (int)looped_layers.size();
            return r;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override
        {
            before_forward(ctx, input_ids, n_past);
            prepare_for_lens(ctx);

            ctx->move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor *hidden_states = custom_embedding ? custom_embedding(ctx, input_ids) :  word_embeddings->forward(ctx, input_ids);
            for (auto &layer : Base::layers)
            {
                ctx->move_to_layer(layer->get_id());
                if (layer_preprocess.get())
                {
                    auto t = layer_preprocess->forward(this, ctx, hidden_states, layer->get_id());
                    if (t) hidden_states = t;
                }

                hidden_states = layer->forward(ctx, hidden_states, n_past);

                attach_lens(ctx, hidden_states, layer->get_id());
            }

            for (int i = 0; i < num_loops - 1; i++)
            {
                if (!skip_loop_final_norm)
                {
                    hidden_states = final_layernorm->forward(ctx, hidden_states);
                }

                for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
                {
                    ctx->move_to_layer(layer_id);
                    auto layer = &looped_layers[i * num_hidden_layers + layer_id];

                    if (layer_preprocess.get())
                    {
                        auto t = layer_preprocess->forward(this, ctx, hidden_states, layer_id);
                        if (t) hidden_states = t;
                    }

                    hidden_states = layer->forward(ctx, hidden_states, n_past);

                    attach_lens(ctx, hidden_states, layer->get_id());
                }
            }

            last_hidden_state = hidden_states;

            ctx->move_to_layer(LayerAllocatorManager::Epilog);
            return final_steps->forward(this, ctx, input_ids, hidden_states);
        }

    public:
        Accessor layers;
        bool skip_loop_final_norm = false;
    protected:
        const int num_loops;
        const int num_hidden_layers;
        std::vector<LoopedLayer> looped_layers;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef LoopModel<BaseConfig, Embedding, RMSNorm, LlamaBlock, int, int, int, int, int, int> ModelClass;
    public:
        ConditionalGeneration() = default;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_NANBEIGE);
    protected:
        const Config config;
    };

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2), config(config)
    {
        const int    num_loops   = std::atoi(utils::get_opt(runtime_config.additional, "num_loops", utils::sprintf("%d", config.num_loops)).c_str());
        const bool   tie_lm_head = config.tie_word_embeddings > 0;
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = (tie_lm_head ? 2 : 3) + config.num_hidden_layers * 12 + (2 * (num_loops - 1) * config.num_hidden_layers);
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        if (tie_lm_head)
            transformer = new ModelClass(&w_ctx_, config, num_loops, nullptr,
                                                config.hidden_size, config.num_attention_heads,
                                                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);
        else
            transformer = new ModelClass(&w_ctx_, config, num_loops, false,
                                                config.hidden_size, config.num_attention_heads,
                                                config.intermediate_size, config.num_key_value_heads, config.head_dim, config.max_length);

        auto transformer = get_typed_transformer<ModelClass>();
        transformer->skip_loop_final_norm = config.skip_loop_final_norm > 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = transformer->layers[i].attention;
            attention.freq_base = config.rope_theta;
        }
    }

    REGISTER_MODEL_LOADER(NANBEIGE,           nanbeige, 1);
}
