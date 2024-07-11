#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <random>
#include <chrono>
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

        virtual size_t load(tokenizer::DataReader *buffer, int n_vocab) = 0;

        virtual void encode(const std::string &text, std::vector<int> &ids) const;
        virtual std::vector<int> encode(const std::string &text) const;

        virtual void encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const;

        virtual std::string decode(const std::vector<int> &ids) const;

        virtual std::vector<int> encode_history(const std::vector<std::string> &history, int max_length, const bool incremental = false);
        virtual std::vector<int> encode_history(BaseHistoryEncoder *encoder, const std::vector<std::string> &history, int max_length, const bool incremental = false);
        virtual std::vector<int> encode_sys_prompt(void);

        void set_system_prompt(const std::string &prompt) { sys_prompt = prompt; }
        const std::string &get_system_prompt(void) { return sys_prompt; }
        virtual void set_additional_args(const std::map<std::string, std::string> &args) {}

        virtual bool is_terminate_token_id(int id) const;
        virtual bool is_special_id(int id) const { return false; }

        void set_chat_format(ChatFormat format) { this->format = format; }
        ChatFormat get_chat_format(void) const { return format; }

        virtual void set_skip_sys_prompt(bool skip);

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
        bool auto_add_bos;
        std::set<int> terminate_ids;
    };

    class BaseHistoryEncoder
    {
    public:
        BaseHistoryEncoder() : skip_sys_prompt(false), tokenizer(nullptr) {}

        virtual void append_sys_prompt(std::vector<int> &ids) const;

        virtual void append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const;
        virtual void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const;
        virtual void do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const;

        void set_tokenizer(BaseTokenizer *tokenizer)
        {
            this->tokenizer = tokenizer;
        }
    public:
        bool skip_sys_prompt;
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

    class ChunkInterceptor;

    class BaseStreamer
    {
    public:
        enum TextType
        {
            META            = 1,    // print a whole line: general information
            ERR             = 2,    // print a whole line: error message
            REF             = 3,    // print a whole line: reference
            REWRITTEN_QUERY = 4,    // print a whole line: rewritten query
            HISTORY_USER    = 5,    // print a whole line: user input history
            HISTORY_AI      = 6,    // print a whole line: AI output history
            TOOL_CALLING    = 7,
            EMBEDDING       = 8,
            RANKING         = 9,
        };
        BaseStreamer(BaseTokenizer *tokenizer);
        virtual ~BaseStreamer() = default;
        virtual void put(const std::vector<int> &output_ids);
        virtual void put_chunk(bool first, const std::string &chunk) = 0;
        virtual void putln(const std::string &line, TextType type = TextType::META) = 0;
        virtual void end();

        virtual void set_interceptor(ChunkInterceptor *interceptor);
        virtual void remove_interceptor(void);

        // used for RAG
        void put_reference(const std::string &line)
        {
            putln(line, TextType::REF);
        }
        void put_rewritten_query(const std::string &line)
        {
            putln(line, TextType::REWRITTEN_QUERY);
        }
        void put_history_user(const std::string &line)
        {
            putln(line, TextType::HISTORY_USER);
        }
        void put_history_ai(const std::string &line)
        {
            putln(line, TextType::HISTORY_AI);
        }
        void put_tool_calling(const std::string &line)
        {
            putln(line, TextType::TOOL_CALLING);
        }
    public:
        bool is_prompt;
        BaseTokenizer *tokenizer;
    protected:
        bool is_first;
        size_t print_len;
        std::vector<int> token_cache;
        ChunkInterceptor *interceptor;

        virtual void call_put_chunk(bool first, const std::string &chunk);
    };

    class ChunkInterceptor
    {
    public:
        ChunkInterceptor() : streamer(nullptr) {}
        virtual void intercept(BaseStreamer *streamer) { this->streamer = streamer; }
        virtual void put_chunk(bool first, const std::string &chunk) { if (streamer) streamer->put_chunk(first, chunk); }
        virtual void end() { }
    protected:
        BaseStreamer *streamer;
    };

    class MappedFile : public tokenizer::DataReader
    {
    public:
        MappedFile(const std::string &path);
        ~MappedFile();

        int64_t tell() override;

        void seek(int64_t offset, int whence) override;

        size_t read_buffer(void *output, size_t len) override;

    protected:
        char *data;
        const char *ptr;
    };

    class SimpleFile : public tokenizer::DataReader
    {
    public:
        SimpleFile(const std::string &path);
        ~SimpleFile();

        int64_t tell() override;

        void seek(int64_t offset, int whence) override;

        size_t read_buffer(void *output, size_t len) override;
    protected:
        FILE *f;
    };

    class TensorInfo
    {
    public:
        TensorInfo(enum ggml_type type, int n_dim, const int64_t *ne, size_t _offset);
        ~TensorInfo();

        void *load(tokenizer::DataReader *reader);

        size_t aligned_data_start(size_t offset);
        size_t aligned_size(void);

    public:
        ggml_tensor tensor;
        const size_t _offset;
        void *data;
    };

    class ModelLoader
    {
    public:
        ModelLoader(const std::string &path)
            : ModelLoader(new SimpleFile(path))
        {
        }

        int64_t tell() const
        {
            return _file->tell();
        }

        void seek(int64_t offset, int whence)
        {
            _file->seek(offset, whence);
        }

        template <typename T> T read_basic()
        {
            return _file->read_basic<T>();
        }

        std::string read_string(size_t length)
        {
            std::string s;
            s.resize(length);
            _file->read_buffer(s.data(), length);
            return s;
        }

        void read_tensor(const std::string &name, ggml_tensor *tensor);
        void load_all_tensors(void);

        tokenizer::DataReader *get_reader()
        {
            return _file.get();
        }

    private:
        ModelLoader(tokenizer::DataReader *mapped_file)
            : _file(std::unique_ptr<tokenizer::DataReader>(mapped_file)),
              offset_config(0),
              offset_tokenizer(0),
              model_type(-1), version(-1)
        {
        }

        std::unique_ptr<tokenizer::DataReader> _file;

    public:
        size_t offset_config;
        size_t offset_tokenizer;
        int model_type;
        int version;
        std::map<std::string, TensorInfo> tensor_dict;
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
        float presence_penalty;
        float tfs_z;
        std::string sampling;

        GenerationConfig()
        {
        }

        GenerationConfig(int max_length, int max_context_length, bool do_sample, int top_k,
                         float top_p, float temperature, int num_threads, const std::string sampling, float presence_penalty, float tfs_z)
            : max_length(max_length), max_context_length(max_context_length), do_sample(do_sample), top_k(top_k),
              top_p(top_p), temperature(temperature), num_threads(num_threads), presence_penalty(presence_penalty), tfs_z(tfs_z),
              sampling(sampling) {}
    };

    class ModelPerfInfo
    {
    public:
        enum Type
        {
            Prompt = 0,
            Generation,
            NUM
        };

        struct Performance
        {
            size_t tok_count;
            double duration_ms;
        };

        ModelPerfInfo();

        void Reset(void);
        double Elapsed(void);

        void Accumulate(Type type, size_t tok_count);

        Performance timings[Type::NUM];

    private:
        using Clock = std::chrono::steady_clock;
        using MilliSecond = std::chrono::duration<double, std::ratio<1, 1000>>;

        std::chrono::time_point<Clock> m_beg { Clock::now() };
    };

    class AbstractModel
    {
    public:
        virtual ~AbstractModel()
        {}

        virtual void set_layer_ids(const std::vector<int> &ids) = 0;

        virtual std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                            const bool continuous,
                                            bool &completed,
                                            ModelPerfInfo *performance,
                                            int gen_max_tokens,
                                            BaseStreamer *streamer = nullptr) = 0;

        virtual void abort_generation(void) = 0;

        virtual void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding) = 0;

        virtual float qa_rank(const GenerationConfig &gen_config,
                              const std::vector<int> &input_ids) = 0;

        virtual int get_text_embedding_dim(void) const = 0;

        virtual std::string type_name() const = 0;
        virtual std::string native_name() const = 0;
        virtual ModelPurpose get_purpose() const = 0;

        virtual void load(ModelLoader &loader) = 0;

        virtual void set_tokenizer(BaseTokenizer *tokenizer) = 0;

        virtual void set_ctx(int n_ctx)= 0;

        virtual void seed(int x) = 0;

        virtual int get_max_length(void) = 0;

        virtual int get_n_past(void) = 0;
        virtual void set_n_past(int n_past) = 0;

        virtual void shift_memory(int keep) = 0;

        virtual int save_session(FILE *f) const = 0;
        virtual int load_session(FILE *f) = 0;

        virtual int64_t get_param_num(bool effective_only) const = 0;

        virtual ChunkInterceptor *get_interceptor(void) { return nullptr; }

        virtual void set_additional_args(const std::map<std::string, std::string> &args) {}
    };

    class ModelProxy : public AbstractModel
    {
    public:
        ModelProxy() : AbstractModel(), model(nullptr) {}

        virtual ~ModelProxy()
        {
            delete model;
        }

        void set_layer_ids(const std::vector<int> &ids) override { return model->set_layer_ids(ids); }

        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                            const bool continuous,
                                            bool &completed,
                                            ModelPerfInfo *performance,
                                            int gen_max_tokens,
                                            BaseStreamer *streamer = nullptr) override
        {
            return model->generate(input_ids, gen_config, continuous, completed, performance, gen_max_tokens, streamer);
        }

        void abort_generation(void) override { model->abort_generation(); }

        void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding) override
        {
            model->text_embedding(gen_config, input_ids, embedding);
        }

        float qa_rank(const GenerationConfig &gen_config,
                              const std::vector<int> &input_ids) override { return model->qa_rank(gen_config, input_ids); }

        int get_text_embedding_dim(void) const override { return model->get_text_embedding_dim(); }

        std::string type_name() const  override { return model->type_name(); }
        std::string native_name() const  override { return model->native_name(); }
        ModelPurpose get_purpose() const  override { return model->get_purpose(); }

        void load(ModelLoader &loader) override { return model->load(loader); }

        void set_tokenizer(BaseTokenizer *tokenizer) override { model->set_tokenizer(tokenizer); }

        void set_ctx(int n_ctx) override { model->set_ctx(n_ctx); }

        void seed(int x) override { model->seed(x); }

        int get_max_length(void) override { return model->get_max_length(); }

        int get_n_past(void) override { return model->get_n_past(); }
        void set_n_past(int n_past) override { model->set_n_past(n_past); }

        void shift_memory(int keep) override { model->shift_memory(keep); }

        int save_session(FILE *f) const { return model->save_session(f); }
        int load_session(FILE *f) override { return model->load_session(f); }

        int64_t get_param_num(bool effective_only) const override { return model->get_param_num(effective_only); }

        ChunkInterceptor *get_interceptor(void) override { return model->get_interceptor(); }

    protected:
        AbstractModel *model;
        void set_proxy_model(AbstractModel *model) { this->model = model; }
    };

    class BaseModel : public AbstractModel
    {
    public:
        BaseModel(int type, std::string name, std::string native_name, ModelPurpose purpose) :
            type_(type), name_(name), native_name_(native_name), n_past(0),
            n_past_offset(0), tokenizer(nullptr),
            purpose(purpose), aborted(false)
        {}

        virtual ~BaseModel()
        {}

        std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                                            const bool continuous,
                                            bool &completed,
                                            ModelPerfInfo *performance,
                                            int gen_max_tokens,
                                            BaseStreamer *streamer = nullptr) override
        {
            std::vector<int> r;
            return r;
        }

        void abort_generation(void) override
        {
            aborted = true;
        }

        void text_embedding(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                    std::vector<float> &embedding) override
        {}

        float qa_rank(const GenerationConfig &gen_config,
                              const std::vector<int> &input_ids) override
        {
            return 0.0f;
        }

        int get_text_embedding_dim(void) const override { return -1; }

        std::string type_name() const override { return name_; }
        std::string native_name() const override { return native_name_; }
        ModelPurpose get_purpose() const override { return purpose; }

        void set_tokenizer(BaseTokenizer *tokenizer) override
        {
            this->tokenizer = tokenizer;
        }

        void set_ctx(int n_ctx) override {}

        void seed(int x) override { _seed = x; }

        int get_n_past(void) override { return n_past; }

        void set_n_past(int n_past) override { this->n_past = n_past; }

        void shift_memory(int keep) override
        {
            CHATLLM_CHECK(n_past >= keep) << "length of kept should not exceeds history";

            n_past_offset += n_past - keep;
            n_past = keep;
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return 0;
        }

        int save_session(FILE *f) const
        {
            struct state state = {.type = type_, .n_past = n_past, .n_past_offset = n_past_offset};
            if (fwrite(&state, sizeof(state), 1, f) != 1)
                return -1;
            return 0;
        }

        int load_session(FILE *f) override
        {
            struct state state = {0};
            if (fread(&state, sizeof(state), 1, f) != 1)
                return -1;
            if (state.type != type_) return -1;
            n_past = state.n_past;
            n_past_offset = state.n_past_offset;
            return 0;
        }
    private:
        struct state
        {
            int type;
            int n_past;
            int n_past_offset;
        };
    protected:
        const int type_;
        std::string name_;
        std::string native_name_;
        int _seed;
        int n_past;
        int n_past_offset;
        BaseTokenizer *tokenizer;
        ModelPurpose purpose;
        bool aborted;
    };

    class ModelObject
    {
    public:
        struct extra_args
        {
            int   max_length;
            std::string layer_spec;
            extra_args(int max_length, const std::string &layer_spec) : max_length(max_length), layer_spec(layer_spec) {}
            extra_args() : extra_args(-1, "") {}
        };

        ModelObject(const std::string &path);
        ModelObject(const std::string &path, const extra_args &args);

        void text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result);

        AbstractModel *fork_model(const extra_args &args);

    public:
        std::unique_ptr<BaseTokenizer> tokenizer;
        std::unique_ptr<AbstractModel> model;
        std::unique_ptr<ModelLoader> loader;
        const bool loaded;
    };

    class ModelFactory
    {
    public:
        struct Result
        {
            std::unique_ptr<BaseTokenizer> tokenizer;
            std::unique_ptr<AbstractModel> model;
        };

        static bool load(ModelLoader &loader, Result &result, const ModelObject::extra_args &args);

        static AbstractModel *load_model_again(ModelLoader &loader, const ModelObject::extra_args &args);

        static std::string load_info(ModelLoader &loader);

    private:
        static bool load(int model_type, int version, ModelLoader &loader, Result &result, const ModelObject::extra_args &args);
    };

    class Pipeline
    {
    public:
        enum ExtendingMethod
        {
            Shift,
            Restart,
            None,
        };

        Pipeline(const std::string &path);
        Pipeline(const std::string &path, const ModelObject::extra_args &args);

        std::string chat(std::vector<std::string> &history, const GenerationConfig &gen_config,
                         BaseStreamer *streamer = nullptr);
        virtual void abort_generation(void);
        virtual void eval_sys_prompt(const GenerationConfig &gen_config);

        void set_system_prompt(const std::string &prompt);
        void set_extending_method(ExtendingMethod method);
        virtual void set_additional_args(const std::map<std::string, std::string> &args);

        void text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result);
        float qa_rank(const std::string &q, const std::string &a, const GenerationConfig &gen_config);

        int get_text_embedding_dim(void);

        virtual std::string get_additional_description(void) const;

        bool is_loaded(void) const { return modelobj.loaded; }

        virtual void restart(void);
        virtual void rewind(int n_past);

        virtual int save_session(const std::vector<std::string> &history, const std::string &file_name);
        virtual int load_session(std::vector<std::string> &history, const std::string &file_name, BaseStreamer *streamer, int *n_past = nullptr);
    protected:
        const char head_magic[16] = "CHATLLM-SESSION";

        struct file_header
        {
            char magic[16];
            size_t history_len;
        };

    public:
        BaseTokenizer *tokenizer;
        AbstractModel *model;
        ModelPerfInfo performance;
        int gen_max_tokens;

    protected:
        bool initializing;
        ExtendingMethod extending;
        ModelObject modelobj;

        std::string chat_with_restart(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                         BaseStreamer *streamer);
        std::string chat_with_shift(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                         BaseStreamer *streamer);
        std::string chat_without_extending(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer);

        virtual void before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer);
        virtual void post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer);
    };

    class AugmentedQueryComposer
    {
    public:
        AugmentedQueryComposer();
        std::string compose_augmented_query(const std::string &query, const std::vector<std::string> augments) const;
        std::string rewrite_query_for_retrieve(const std::string &query) const;
        std::string parse_rewritten_query_result(const std::string &query) const;
        bool is_rewritten_template_set(void) const;
        void set_prompt_template(const std::string &s);
        void set_context_sep(const std::string &s);
        void set_rewrite_template(const std::string &s);
    protected:
        std::string prompt_template;
        std::string context_sep;
        std::string query_rewritten_template;
    };

    class RAGPipeline : public Pipeline
    {
    public:
        RAGPipeline(const std::string &path, const ModelObject::extra_args &args,
                    DistanceStrategy vec_cmp, const std::vector<std::string> &vector_stores,
                    const std::string &embedding_model, const std::string &reranker_model = "");

        std::string get_additional_description(void) const override;

        AugmentedQueryComposer composer;
        bool    hide_reference;
        int     retrieve_top_n;
        int     rerank_top_n;
        bool    dump;
        float   rerank_score_threshold;
        int     rag_post_extending;
        bool    rerank_rewrite;

    protected:
        void before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer) override;
        void post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer) override;

        CVectorStore vs;
        ModelObject embedding;
        ModelObject *reranker;
        std::vector<std::string> metainfo;

    private:
        void rerank(const std::string &query, std::vector<int64_t> &candidates, const GenerationConfig &gen_config, int top_n = 3);
        std::string rewrite_query(const std::string &prompt, const GenerationConfig &gen_config);
        AbstractModel *rewrite_model;
    };

} // namespace chatllm
