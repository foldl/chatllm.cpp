#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <set>
#include <vector>
#include <random>
#include "basics.h"
#include "tokenizer.h"
#include "vectorstore.h"

namespace chatllm
{
    enum ChatFormat
    {
        CHAT = 0,
        COMPLETION,
        QA,
    };

    enum ModelPurpose
    {
        Chat = 0,
        TextEmbedding,
        Ranker,
    };

    std::string to_string(ggml_tensor *tensor, bool with_data = true);

    struct BaseConfig
    {
        // common attributes
        ggml_type dtype;
        int vocab_size;
        int hidden_size;
        int num_attention_heads;
        int num_hidden_layers;
        int intermediate_size;
        // for sequence generation
        int max_length;
        // for tokenizer
        int bos_token_id;
        int eos_token_id;
        int pad_token_id;
        int sep_token_id;
    };

    std::string trim(std::string str, const char *spaces = " \t");

    class BaseHistoryEncoder;

    class BaseTokenizer
    {
    public:
        BaseTokenizer(const BaseConfig &config,
                        BaseHistoryEncoder *chat_encoder,
                        BaseHistoryEncoder *qa_encoder = nullptr,
                        BaseHistoryEncoder *completion_encoder = nullptr);

        virtual ~BaseTokenizer() = default;

        virtual size_t load(const char *buffer, int n_vocab) = 0;

        virtual void encode(const std::string &text, std::vector<int> &ids) const;
        virtual std::vector<int> encode(const std::string &text) const;

        virtual void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const;

        virtual std::string decode(const std::vector<int> &ids) const;

        virtual std::vector<int> encode_history(const std::vector<std::string> &history, int max_length, const bool incremental = false);
        virtual std::vector<int> encode_history(BaseHistoryEncoder *encoder, const std::vector<std::string> &history, int max_length, const bool incremental = false);

        void set_system_prompt(const std::string &prompt) { sys_prompt = prompt; }
        const std::string &get_system_prompt(void) { return sys_prompt; }
        virtual void set_additional_args(const std::map<std::string, std::string> &args) {}

        virtual int get_terminate_token_id(void) const { return -1000; }
        virtual bool is_special_id(int id) const { return false; }

        void set_chat_format(ChatFormat format) { this->format = format; }
        ChatFormat get_chat_format(void) const { return format; }

        int bos_token_id;
        int eos_token_id;
        int pad_token_id;
        int sep_token_id;

    protected:
        virtual int get_history_start(const std::vector<std::string> &history, int max_length) const;

        virtual std::string preprocess(const std::string &text) const;
        virtual std::string postprocess(const std::string &text) const;

    public:
        tokenizer::Processor *tp;
    protected:
        std::string sys_prompt;
        int history_offset;
        const int max_length;
        ChatFormat format;
        BaseHistoryEncoder *chat_encoder;
        BaseHistoryEncoder *completion_encoder;
        BaseHistoryEncoder *qa_encoder;
    };

    class BaseHistoryEncoder
    {
    public:
        BaseHistoryEncoder() : tokenizer(nullptr) {};

        virtual void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const {};
        virtual void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
        {
            tokenizer->encode(user, ids);
        }

        void set_tokenizer(BaseTokenizer *tokenizer)
        {
            this->tokenizer = tokenizer;
        }
    protected:
        BaseTokenizer *tokenizer;
    };

    class GGMLContext
    {
    public:
        GGMLContext() : gctx_(nullptr) {}

        GGMLContext(ggml_init_params params) : gctx_(ggml_init(params))
        {
            CHATLLM_CHECK(gctx_) << "failed to init ggml context";
        }
        // copy constructor
        GGMLContext(const GGMLContext &) = delete;
        // move constructor
        GGMLContext(GGMLContext &&other) : gctx_(other.gctx_) { other.gctx_ = nullptr; }

        ~GGMLContext() { reset(); }

        // copy assignment
        GGMLContext &operator=(const GGMLContext &) = delete;
        // move assignment
        GGMLContext &operator=(GGMLContext &&other)
        {
            reset();
            gctx_ = other.gctx_;
            other.gctx_ = nullptr;
            return *this;
        }

        ggml_context *get() const { return gctx_; }

        void reset()
        {
            if (gctx_)
            {
                ggml_free(gctx_);
                gctx_ = nullptr;
            }
        }

    private:
        ggml_context *gctx_;
    };

    struct InitContext
    {
        GGMLContext gctx;
        ggml_type dtype;
    };

    struct ForwardContext
    {
        GGMLContext gctx;
        ggml_cgraph *gf;
        ggml_scratch scratch;
    };

    class BaseStreamer
    {
    public:
        virtual ~BaseStreamer() = default;
        virtual void put(const std::vector<int> &output_ids) = 0;
        virtual void putln(const std::string &line) = 0;
        virtual void end() = 0;
    };

    class MappedFile
    {
    public:
        MappedFile(const std::string &path);
        ~MappedFile();

    public:
        char *data;
        size_t size;
    };

    class ModelLoader
    {
    public:
        ModelLoader(const std::string &path)
            : ModelLoader(new MappedFile(path))
        {
        }

        int64_t tell() const { return ptr - data; }

        void seek(int64_t offset, int whence);

        template <typename T>
        T read_basic()
        {
            T obj = *(T *)ptr;
            ptr += sizeof(T);
            return obj;
        }

        std::string read_string(size_t length);

        void read_tensor(const std::string &name, ggml_tensor *tensor);

    private:
        ModelLoader(MappedFile *mapped_file)
            : mapped_file(std::unique_ptr<MappedFile>(mapped_file)),
              data(mapped_file->data), size(mapped_file->size), ptr(mapped_file->data)
        {
        }

        std::unique_ptr<MappedFile> mapped_file;

    public:
        const char *const data;
        size_t size;
        const char *ptr;
    };

    // ===== generation =====

    struct GenerationConfig
    {
        int max_length;
        int max_context_length;
        bool do_sample;
        int top_k;
        float top_p;
        float temperature;
        int num_threads;

        GenerationConfig()
        {
        }

        GenerationConfig(int max_length, int max_context_length, bool do_sample, int top_k,
                         float top_p, float temperature, int num_threads)
            : max_length(max_length), max_context_length(max_context_length), do_sample(do_sample), top_k(top_k),
              top_p(top_p), temperature(temperature), num_threads(num_threads) {}
    };

    class BaseModel
    {
    public:
        BaseModel(int type, std::string name, std::string native_name, ModelPurpose purpose) :
            type_(type), name_(name), native_name_(native_name), gen(0x123), n_past(0),
            n_past_offset(0), tokenizer(nullptr),
            purpose(purpose), aborted(false), terminate_token_id(-1000)
        {}

        virtual ~BaseModel()
        {}

        virtual std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                            const bool continuous,
                                            bool &completed,
                                            BaseStreamer *streamer = nullptr)
        {
            std::vector<int> r;
            return r;
        };

        virtual void abort_generation(void)
        {
            aborted = true;
        }

        virtual void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding)
        {}

        virtual float qa_rank(const GenerationConfig &gen_config,
                              const std::vector<int> &input_ids)
        {
            return 0.0f;
        }

        virtual int get_text_embedding_dim(void) const { return -1; }

        std::string type_name() const { return name_; }
        std::string native_name() const { return native_name_; }
        ModelPurpose get_purpose() const { return purpose; }

        virtual void load(ModelLoader &loader) = 0;

        void set_tokenizer(BaseTokenizer *tokenizer)
        {
            this->tokenizer = tokenizer;
        }

        virtual void set_ctx(int n_ctx) {}

        void seed(int x) { gen.seed((unsigned int)x); }

        virtual int get_max_length(void) = 0;

        int get_n_past(void) { return n_past; }

        virtual void shift_memory(int keep)
        {
            CHATLLM_CHECK(n_past >= keep) << "length of kept should not exceeds history";

            n_past_offset += n_past - keep;
            n_past = keep;
        }

        virtual int64_t get_param_num(bool effective_only) const
        {
            return 0;
        }

    protected:
        int type_;
        std::string name_;
        std::string native_name_;
        std::mt19937 gen;
        int n_past;
        int n_past_offset;
        BaseTokenizer *tokenizer;
        ModelPurpose purpose;
        bool aborted;
    public:
        int terminate_token_id; // when LLM uses another token as end indicator
    };

    class ModelFactory
    {
    public:
        struct Result
        {
            std::unique_ptr<BaseTokenizer> tokenizer;
            std::unique_ptr<BaseModel> model;
        };

        static bool load(ModelLoader &loader, Result &result, int max_length);
    private:
        static bool load(int model_type, int version, ModelLoader &loader, Result &result, int max_length);
    };

    class ModelObject
    {
    public:
        struct extra_args
        {
            int   max_length;
            extra_args(int max_length) : max_length(max_length) {}
            extra_args() : extra_args(-1) {}
        };

        ModelObject(const std::string &path);
        ModelObject(const std::string &path, const extra_args &args);

        void text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result);

    public:
        std::unique_ptr<BaseTokenizer> tokenizer;
        std::unique_ptr<BaseModel> model;
        std::unique_ptr<ModelLoader> loader;
        const bool loaded;
    };

    class Pipeline
    {
    public:
        enum ExtendingMethod
        {
            Shift,
            Restart,
        };

        Pipeline(const std::string &path);
        Pipeline(const std::string &path, const ModelObject::extra_args &args);

        std::string chat(std::vector<std::string> &history, const GenerationConfig &gen_config,
                         BaseStreamer *streamer = nullptr);
        virtual void abort_generation(void);

        void set_system_prompt(const std::string &prompt);
        void set_extending_method(ExtendingMethod method);
        virtual void set_additional_args(const std::map<std::string, std::string> &args);

        void text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result);
        float qa_rank(const std::string &q, const std::string &a, const GenerationConfig &gen_config);

        int get_text_embedding_dim(void);

        virtual std::string get_additional_description(void) const;

        bool is_loaded(void) const { return modelobj.loaded; }

    public:
        BaseTokenizer *tokenizer;
        BaseModel *model;

    protected:
        bool initializing;
        ExtendingMethod extending;
        ModelObject modelobj;

        std::string chat_with_restart(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                         BaseStreamer *streamer);
        std::string chat_with_shift(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                         BaseStreamer *streamer);

        virtual void before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer);
        virtual void post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer);
    };

    class AugmentedQueryComposer
    {
    public:
        AugmentedQueryComposer();
        std::string compose_augmented_query(const std::string &query, const std::vector<std::string> augments) const;

        void set_prompt_template(const std::string &s);
        void set_context_sep(const std::string &s);
    protected:
        std::string prompt_template;
        std::string context_sep;
    };

    class RAGPipeline : public Pipeline
    {
    public:
        RAGPipeline(const std::string &path, const ModelObject::extra_args &args,
                    DistanceStrategy vec_cmp, const std::vector<std::string> &vector_stores,
                    const std::string &embedding_model, const std::string &reranker_model = "");

        std::string get_additional_description(void) const override;

        AugmentedQueryComposer composer;
        std::string reference_tag;
        bool    hide_reference;
        int     retrieve_top_n;
        int     rerank_top_n;
        bool    dump;
        float   rerank_score_threshold;
        int     rag_post_extending;

    protected:
        void before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer) override;
        void post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer) override;

        CVectorStore vs;
        ModelObject embedding;
        ModelObject *reranker;
        std::vector<std::string> metainfo;

    private:
        void rerank(const std::string &query, std::vector<int64_t> &candidates, const GenerationConfig &gen_config, int top_n = 3);
    };

} // namespace chatllm
