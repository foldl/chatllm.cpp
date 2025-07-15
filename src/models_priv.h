#pragma once

#include <stdint.h>
#include "chat.h"

namespace chatllm
{
    #define MAKE_TYPE_TAG(v)            (((uint32_t)(v) >> 1) << 24)
    #define MODEL_TYPE_TAG_ChatImageIn                              MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::ImageInput)
    #define MODEL_TYPE_TAG_ChatAudioIn                              MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::AudioInput)
    #define MODEL_TYPE_TAG_ChatImageInVideoIn                       MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::ImageInput + ChatModelAccessPoint::VideoInput)
    #define MODEL_TYPE_TAG_ChatImageInVideoInAudioInAudioOut        MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::ImageInput + ChatModelAccessPoint::VideoInput + ChatModelAccessPoint::AudioInput + ChatModelAccessPoint::AudioOutput)

    enum ModelType
    {
        MODEL_TYPE_CHATGLM  = 1,
        MODEL_TYPE_CHATGLM2 = 2,
        MODEL_TYPE_CHATGLM3 = 3,
        MODEL_TYPE_CODEGEEX2 = 4,
        MODEL_TYPE_CHARACTERGLM = 5,
        MODEL_TYPE_GLM4 = 6,
        MODEL_TYPE_CODEGEEX4 = 7,
        MODEL_TYPE_GLM4_0414 = 8,

        MODEL_TYPE_INTERNLM     = 0x100,
        MODEL_TYPE_INTERNLM2    = 0x101, // extended model, supporting 7B & 20B
        MODEL_TYPE_INTERNLM2_1  = 0x102,
        MODEL_TYPE_INTERNLM3    = 0x103,

        MODEL_TYPE_LLAMA2   = 0x150,
        MODEL_TYPE_CODELLAMA= 0x151,
        MODEL_TYPE_WIZARDCODER      = 0x152,
        MODEL_TYPE_WIZARDLM         = 0x153,
        MODEL_TYPE_WIZARDMATH       = 0x154,
        MODEL_TYPE_TIGERBOT         = 0x155,
        MODEL_TYPE_LLAMA2PLUS       = 0x156,
        MODEL_TYPE_MEGREZ           = 0x157,
        MODEL_TYPE_FALCON3          = 0x158,
        MODEL_TYPE_REKA_FLASH3      = 0x159,

        MODEL_TYPE_BAICHUANLLAMA    = 0x200,
        MODEL_TYPE_BAICHUAN         = 0x201,
        MODEL_TYPE_BAICHUAN_M1      = 0x202,

        MODEL_TYPE_DEEPSEEK = 0x300,
        MODEL_TYPE_DEEPSEEK_CODER   = 0x301,
        MODEL_TYPE_CODEFUSE_DEEPSEEK = 0x302,
        MODEL_TYPE_NUMINAMATH        = 0x303,
        MODEL_TYPE_DEEPSEEK_V2_LIGHT = 0x320,
        MODEL_TYPE_DEEPSEEK_V2       = 0x321,
        MODEL_TYPE_DEEPSEEK_V3_LIGHT = 0x322,   // DOES NOT EXIST
        MODEL_TYPE_DEEPSEEK_V3       = 0x323,
        MODEL_TYPE_DEEPSEEK_V1_MoE   = 0x324,
        MODEL_TYPE_GIGACHAT          = 0x325,
        MODEL_TYPE_BAILINGMOE        = 0x326,
        MODEL_TYPE_XVERSEMOE         = 0x327,

        MODEL_TYPE_YI       = 0x400,
        MODEL_TYPE_MAP_NEO  = 0x401,

        MODEL_TYPE_PHI2     = 0x500,
        MODEL_TYPE_PHI2_V2  = 0x501,
        MODEL_TYPE_PHI3     = 0x520,
        MODEL_TYPE_PHI3_SU  = 0x521,
        MODEL_TYPE_PHI3_SU2 = 0x522,
        MODEL_TYPE_PHI3_SU3 = 0x523,
        MODEL_TYPE_PHI3_MOE = 0x530,
        MODEL_TYPE_PHI4     = 0x531,
        MODEL_TYPE_PHI4_MINI= 0x532,

        MODEL_TYPE_DOLPHINPHI2      = 0x510,
        MODEL_TYPE_DOLPHINPHI2_V2   = 0x511,

        MODEL_TYPE_MISTRAL  = 0x600,
        MODEL_TYPE_MIXTRAL  = 0x601,
        MODEL_TYPE_OPENCHAT = 0x602,
        MODEL_TYPE_NEURALBEAGLE = 0x603,
        MODEL_TYPE_STARLING     = 0x604,
        MODEL_TYPE_WIZARDLM2_MOE  = 0x605,
        MODEL_TYPE_MISTRAL2       = 0x606,
        MODEL_TYPE_DEEPHERMES3_MISTRAL = 0x607,

        MODEL_TYPE_QWEN     = 0x700,
        MODEL_TYPE_QWEN2    = 0x710,
        MODEL_TYPE_QWEN2TIE = 0x711,
        MODEL_TYPE_QWEN2MoE = 0x750,
        MODEL_TYPE_MARCO_O1 = 0x751,
        MODEL_TYPE_QWQ      = 0x752,
        MODEL_TYPE_READERLM2= 0x753,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN     = 0x754,
        MODEL_TYPE_QWEN3                        = 0x755,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN3    = 0x756,

        MODEL_TYPE_BLUELM   = 0x800,

        MODEL_TYPE_STABLELM = 0x900,

        MODEL_TYPE_ORION    = 0x1000,

        MODEL_TYPE_MINICPM  = 0x1100,
        MODEL_TYPE_MINICPM2 = 0x1101,
        MODEL_TYPE_MINICPM_MoE = 0x1102,
        MODEL_TYPE_MINICPM3 = 0x1110,
        MODEL_TYPE_MINICPM4 = 0x1111,

        MODEL_TYPE_PERSIMMON= 0x1200,
        MODEL_TYPE_FUYU     = 0x1201,

        MODEL_TYPE_GEMMA    = 0x1300,
        MODEL_TYPE_GEMMA2   = 0x1301,
        MODEL_TYPE_GEMMA3   = 0x1302,

        MODEL_TYPE_COHERE_COMMAND_R     = 0x1400,
        MODEL_TYPE_COHERE_AYA_23        = 0x1401,
        MODEL_TYPE_COHERE_COMMAND_R7B   = 0x1402,

        MODEL_TYPE_GROK_1           = 0x1500,

        MODEL_TYPE_ZHINAO           = 0x1600,

        MODEL_TYPE_LLAMA3           = 0x1700,
        MODEL_TYPE_SMOLLM           = 0x1701,
        MODEL_TYPE_LLAMA3_GROQ_TOOL = 0x1702,
        MODEL_TYPE_LLAMA3_1         = 0x1703,
        MODEL_TYPE_LLAMA3_2         = 0x1704,
        MODEL_TYPE_EXAONE           = 0x1705,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA = 0x1706,
        MODEL_TYPE_AQUILA2                   = 0x1707,
        MODEL_TYPE_ERNIE_DENSE               = 0x1708,

        MODEL_TYPE_STARCODER2       = 0x1800,

        MODEL_TYPE_XVERSE           = 0x1900,

        MODEL_TYPE_INDEX            = 0x1a00,

        MODEL_TYPE_OLMoE            = 0x1b00,
        MODEL_TYPE_OLMo2            = 0x1b01,

        MODEL_TYPE_ALPHAGEO_LM      = 0x1c00,

        MODEL_TYPE_GRANITE_MoE      = 0x1d00,
        MODEL_TYPE_GRANITE          = 0x1d01,

        MODEL_TYPE_TELECHAT2        = 0x1e00,

        MODEL_TYPE_HUNYUAN_DENSE    = 0x1f00,
        MODEL_TYPE_HUNYUAN_MOE_V1   = 0x1f01,

        MODEL_TYPE_MOONLIGHT        = 0x2000,

        MODEL_TYPE_INSTELLA         = 0x2100,

        MODEL_TYPE_DECILM           = 0x2200,

        MODEL_TYPE_SOLARPRO         = 0x2300,

        MODEL_TYPE_APRIEL           = 0x2400,

        MODEL_TYPE_ERNIE_MOE        = 0x2500,

        MODEL_TYPE_PANGU_MOE        = 0x2600,

        MODEL_TYPE_SMOLLM3          = 0x2700,

        MODEL_TYPE_EXAONE4          = 0x2800,

        MODEL_TYPE_BCE_Embedding = 0x10000100,
        MODEL_TYPE_BCE_ReRanker  = 0x10000101,
        MODEL_TYPE_BGE_M3        = 0x10000102,
        MODEL_TYPE_BGE_ReRanker_M3  = 0x10000103,
        MODEL_TYPE_MiniCPM_Embedding_Light  = 0x10000104,
        MODEL_TYPE_MiniCPM_ReRanker_Light   = 0x10000105,
        MODEL_TYPE_ORPHEUS_TTS              = 0x10000106,
        MODEL_TYPE_OUTE_TTS_LLAMA           = 0x10000107,
        MODEL_TYPE_OUTE_TTS_QWEN3           = 0x10000108,
        MODEL_TYPE_QWEN3_Embedding          = 0x10000109,
        MODEL_TYPE_QWEN3_ReRanker           = 0x1000010A,

        MODEL_TYPE_LLAMA_MULTI      = 0x20000001,

        MODEL_TYPE_LLAMA4           = MODEL_TYPE_TAG_ChatImageIn + 0x0000001,
        MODEL_TYPE_GEMMA3Vis        = MODEL_TYPE_TAG_ChatImageIn + 0x0000011,

        MODEL_TYPE_QWEN2_AUDIO      = MODEL_TYPE_TAG_ChatAudioIn + 0x0000001,

        MODEL_TYPE_QWEN2_5_VL       = MODEL_TYPE_TAG_ChatImageInVideoIn + 0x0000001,
        MODEL_TYPE_KIMI_VL          = MODEL_TYPE_TAG_ChatImageInVideoIn + 0x0000100,
        MODEL_TYPE_SMOL_VLM         = MODEL_TYPE_TAG_ChatImageInVideoIn + 0x0000200,
    };

    struct RuntimeConfig
    {
        bool moe_on_cpu;
        int n_threads;
        int batch_input_size;
        ggml::type cache_type;
        std::map<std::string, std::string> model_gpu_layers;
        RuntimeConfig(bool moe_on_cpu, int n_threads, int batch_input_size, ggml::type cache_type):
            moe_on_cpu(moe_on_cpu), n_threads(n_threads), batch_input_size(batch_input_size), cache_type(cache_type)
        {}
    };

    template <class Config>
    void load_config(ModelLoader &loader, Config &config, const ModelObject::extra_args &args)
    {
        if (0 == loader.offset_config)
            loader.offset_config = loader.tell();
        else
            loader.seek(loader.offset_config, SEEK_SET);

        // load config
        config = loader.read_basic<Config>();
        if (args.max_length > 0)
            config.max_length = args.max_length;
        if (args.re_quantize >= 0)
            config.dtype = (ggml::type)args.re_quantize;

        loader.offset_tokenizer = loader.tell();
    }

    template <class Config, class Tokenizer>
    Tokenizer *load_tokenizer(ModelLoader &loader, Config &config)
    {
        loader.seek(loader.offset_tokenizer, SEEK_SET);

        // load tokenizer
        Tokenizer *tokenizer = new Tokenizer(config);
        tokenizer->load(loader.get_reader(), config.vocab_size);
        tokenizer->load_config(loader.meta_json);
        loader.load_all_tensors();

        return tokenizer;
    }

    template <class Config, class ConditionalGeneration>
    ConditionalGeneration *load_model(ModelLoader &loader, Config &config, const ModelObject::extra_args &args)
    {
        std::vector<int> layers;
        if (args.layer_spec.size() > 0)
        {
            utils::parse_int_lists(layers, args.layer_spec, config.num_hidden_layers);
            config.num_hidden_layers = (int)layers.size();
        }

        RuntimeConfig rt_config(args.moe_on_cpu, args.n_threads, args.batch_size, (ggml::type)args.cache_type);
        rt_config.model_gpu_layers = args.model_n_gpu_layers;

        // load model
        ConditionalGeneration *model = new ConditionalGeneration(config, rt_config);
        model->set_type(loader.model_type);
        model->set_names(loader.model_name, loader.model_native_name);
        if (layers.size() > 0)
            model->set_layer_ids(layers);

        loader.push_allocator_manager(model->get_alloc_manager());
        model->load_more(loader.meta_json);
        model->load(loader);

        return model;
    }

    template <class Config, class ConditionalGeneration>
    ConditionalGeneration *load_model(ModelLoader &loader, const ModelObject::extra_args &args)
    {
        Config config;

        load_config<Config>(loader, config, args);

        return load_model<Config, ConditionalGeneration>(loader, config, args);
    }

    template <class Config, class Tokenizer, class ConditionalGeneration>
    bool load_model(ModelLoader &loader, ModelFactory::Result &result, const ModelObject::extra_args &args)
    {
        // load config
        Config config;

        load_config<Config>(loader, config, args);

        // load tokenizer
        result.tokenizer = std::unique_ptr<BaseTokenizer>(load_tokenizer<Config, Tokenizer>(loader, config));

#if (0)
        // test tokenizer
        std::vector<int> ids = result.tokenizer->encode("\nAlice:");
        for (auto x : ids) std::cout << x << ", ";
        std::cout << std::endl;

        //ids = {0,1,2,195,196};
        std::cout << result.tokenizer->decode(ids) << std::endl;
        exit(-1);
#endif
        // load model
        result.model = std::unique_ptr<AbstractModel>(load_model<Config, ConditionalGeneration>(loader, config, args));

        result.model->set_tokenizer(result.tokenizer.get());

        return true;
    }

    class BaseImplModelLoader
    {
    public:
        BaseImplModelLoader(int model_type);
        virtual AbstractModel *load_model(ModelLoader &loader, const ModelObject::extra_args &args) = 0;
        virtual bool load_model(ModelLoader &loader, ModelFactory::Result &result, const ModelObject::extra_args &args) = 0;
    };

    template <int model_type, class Config, class Tokenizer, class ConditionalGeneration> class ImplModelLoader : public BaseImplModelLoader
    {
    public:
        ImplModelLoader() : BaseImplModelLoader(model_type) {}
        AbstractModel *load_model(ModelLoader &loader, const ModelObject::extra_args &args) override
        {
            return chatllm::load_model<Config, ConditionalGeneration>(loader, args);
        }

        bool load_model(ModelLoader &loader, ModelFactory::Result &result, const ModelObject::extra_args &args) override
        {
            return chatllm::load_model<Config, Tokenizer, ConditionalGeneration>(loader, result, args);
        }
    };

    class ForwardContext : public ComputeContext
    {
    public:
        ForwardContext(BackendContext *backend_context) : ComputeContext(backend_context)
        {
        }

        struct ggml_context *get_ctx() override { return gctx.get(); }
        ggml_cgraph *get_cgraph(void) override { return gf; }

    public:
        GGMLContext gctx;
        ggml_cgraph *gf;
    };

    void set_dbg_ctx(ForwardContext *c);

    class BaseModelForConditionalGeneration : public BaseModel
    {
    public:
        BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);
        virtual ~BaseModelForConditionalGeneration() = default;

        void set_layer_ids(const std::vector<int> &ids) override;
        int get_max_length(void) override;
        void shift_memory(int keep) override;
        int64_t get_param_num(bool effective_only) const override;
        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                  const bool continuous,
                                  bool &completed,
                                  ModelPerfInfo *performance,
                                  int gen_max_tokens,
                                  BaseStreamer *streamer = nullptr);

        void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding) override;
        float qa_rank(const GenerationConfig &gen_config, const std::vector<int> &input_ids) override;
        bool generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, std::vector<float> &lm_logits) override;
        int save_session(FILE *f) const override;
        int load_session(FILE *f) override;
        int save_session(ModelSessionMemory &session) const override;
        int load_session(ModelSessionMemory &session) override;
        void prepare(const RuntimeConfig &rt_config);
        LayerAllocatorManager *get_alloc_manager(void) override;

        void load(ModelLoader &loader) override;

    protected:
        virtual void before_generate(const GenerationConfig &gen_config);
        virtual void after_generate(void);
        virtual void do_build_graph(ForwardContext &ctc, const std::vector<int> &input_ids,
                                       const GenerationConfig &gen_config,
                                       int past);
        virtual bool before_initial_run(const int ids_count,
                                       const GenerationConfig &gen_config,
                                       int past);

        bool run_model(const std::vector<int> &input_ids,
            const GenerationConfig &gen_config,
            int past,
            std::vector<float> &output);

        virtual bool run_model(const int *input_ids, const int ids_count,
                                const GenerationConfig &gen_config,
                                int past,
                                std::vector<float> &output);

        virtual bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output);

        bool match_output_sequence(const std::vector<int> &output_ids, const std::vector<int> &pattern);

        template <class T> T *get_typed_transformer(void) const
        {
            return dynamic_cast<T *>(transformer);
        }

    protected:
        ModelBlock *transformer;
        const size_t GRAPH_SIZE;
        int batch_input;
        float logit_scale;
        std::vector<int> layer_ids;
        BackendContext backend_context;
        InitContext w_ctx_; // weight context
    private:
        BaseConfig config_;
        bool initial_run = false;
    };

    template <class Config, class Embedding, class FinalNorm, class LayerBlock, typename... _Types> class Model :
        public HeterogeneousModel
    {
    private:
        typedef HeterogeneousModel Base;
    protected:
        class Accessor
        {
        friend Model;
        protected:
            Accessor() : m(nullptr) {}
        public:
            LayerBlock & operator[](int index)
            {
                if (nullptr == m)
                {
                    uintptr_t offset = (uintptr_t)&(((Model *)(nullptr))->layers);
                    m = (Model *)(uintptr_t(this) - offset);
                }
                return *(dynamic_cast<LayerBlock *>((m->Base::layers)[index])); // .get()));
            }
        private:
            Model *m;
        };
    public:
        Model() = default;
        Model(InitContext *ctx, const Config &config, bool lm_head_bias, _Types... layer_args)
            : Model(ctx, config,
                create_lm_head(ctx, config, lm_head_bias),
                std::forward<_Types>(layer_args)...)
        {}

        Model(InitContext *ctx, const Config &config, Block *lm_head, _Types... layer_args)
        : HeterogeneousModel(ctx, config.num_hidden_layers, config.hidden_size,
                            create_embedding<Embedding>(ctx, config),
                            create_final_norm<FinalNorm>(ctx, config),
                            lm_head,
                            [&](InitContext *ctx, int layer_index) {
                                return new LayerBlock(ctx, std::forward<_Types>(layer_args)...);
                            })
        {
        }

    protected:
        int64_t get_param_num_of_layers(bool effective_only) const override
        {
            int64_t r = 0;
            if (Base::layers.size() > 0)
                r += Base::layers[0]->get_param_num(effective_only) * Base::layers.size();
            return r;
        }
    public:
        Accessor layers;
    };

    template <class Config, class Embedding, class LayerBlock, class FinalBlock, typename... _Types> class EmbeddingModel : public Model
        <Config, Embedding, FinalBlock, LayerBlock, _Types...>
    {
    public:
        typedef Model<Config, Embedding, FinalBlock, LayerBlock, _Types...> Base;
        typedef HeterogeneousModel BaseBase;

        EmbeddingModel() = default;

        EmbeddingModel(InitContext *ctx, const Config &config, _Types... layer_args)
        : Base(ctx, config, nullptr, std::forward<_Types>(layer_args)...)
        {
            Base::set_final_steps(std::make_unique<EmbeddingPoolingFinalSteps>());
        }
    };

    template <class Embedding> Block *create_embedding(InitContext *ctx, const BaseConfig &config)
    {
        return new Embedding(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config.vocab_size, config.hidden_size, config.max_length);
    }

    template <class FinalNorm> Block *create_final_norm(InitContext *ctx, const BaseConfig &config)
    {
        return new FinalNorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size);
    }

    Block *create_lm_head(InitContext *ctx, const BaseConfig &config, bool bias = false);
}