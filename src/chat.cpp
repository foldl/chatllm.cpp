#include "chat.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <regex>
#include <string>
#include <functional>

#include <sys/stat.h>
#include <thread>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <stdio.h>
#endif

#ifdef GGML_USE_CLBLAST
#include "ggml-opencl.h"
#endif

namespace chatllm
{

    std::string trim(std::string str, const char *spaces)
    {
        str.erase(str.find_last_not_of(spaces) + 1);
        str.erase(0,str.find_first_not_of(spaces));
        return str;
    }

    BaseStreamer::BaseStreamer(BaseTokenizer *tokenizer)
        : is_prompt(true), tokenizer(tokenizer), is_first(true), print_len(0),
          interceptor(nullptr)
    {
    }

    void BaseStreamer::put(const std::vector<int> &output_ids)
    {
        is_prompt = false;

        token_cache.insert(token_cache.end(), output_ids.begin(), output_ids.end());
        std::string text = tokenizer->decode(token_cache);
        if (text.empty())
        {
            return;
        }

        std::string printable_text;
        if ((text.back() == '\n') || (text.back() == '\r'))
        {
            // flush the cache after newline
            printable_text = text.substr(print_len);
            token_cache.clear();
            print_len = 0;
        }
        else
        {
            size_t end = tokenizer::get_end_of_valid_utf8(text, print_len);
            if (end > print_len)
            {
                printable_text = text.substr(print_len, end - print_len);
                print_len = end;
            }
        }

        if (printable_text.size() > 0)
        {
            call_put_chunk(is_first, printable_text);
            is_first = false;
        }
    }

    void BaseStreamer::end()
    {
        if (tokenizer)
        {
            std::string text = tokenizer->decode(token_cache);
            size_t end = tokenizer::get_end_of_valid_utf8(text, print_len);
            if (end > print_len)
            {
                call_put_chunk(is_first, text.substr(print_len));
                print_len = end;
            }
        }

        if (interceptor)
        {
            auto it = interceptor;
            interceptor = nullptr;
            it->end();
        }

        is_first = true;
        is_prompt = true;
        token_cache.clear();
        print_len = 0;
    }

    void BaseStreamer::set_interceptor(ChunkInterceptor *interceptor)
    {
        this->interceptor = interceptor;
        if (interceptor) interceptor->intercept(this);
    }

    void BaseStreamer::remove_interceptor(void)
    {
        if (interceptor) interceptor->intercept(nullptr);
        interceptor = nullptr;
    }

    void BaseStreamer::call_put_chunk(bool first, const std::string &chunk)
    {
        if (interceptor)
            interceptor->put_chunk(first, chunk);
        else
            put_chunk(first, chunk);
    }

    BaseTokenizer::BaseTokenizer(const BaseConfig &config,
                        BaseHistoryEncoder *chat_encoder,
                        BaseHistoryEncoder *qa_encoder,
                        BaseHistoryEncoder *completion_encoder) :
        bos_token_id(config.bos_token_id),
        eos_token_id(config.eos_token_id),
        pad_token_id(config.pad_token_id),
        sep_token_id(config.sep_token_id),
        tp(nullptr),
        sys_prompt(""),
        history_offset(0),
        max_length(config.max_length),
        format(ChatFormat::CHAT),
        chat_encoder(chat_encoder),
        completion_encoder(completion_encoder),
        qa_encoder(qa_encoder),
        auto_add_bos(true)
    {
        if (chat_encoder)
            chat_encoder->set_tokenizer(this);
        if (completion_encoder)
            completion_encoder->set_tokenizer(this);
        if (qa_encoder)
            qa_encoder->set_tokenizer(this);
    }

    bool BaseTokenizer::is_terminate_token_id(int id) const
    {
        if (id == eos_token_id) return true;

        return terminate_ids.find(id) != terminate_ids.end();
    }

    std::string BaseTokenizer::preprocess(const std::string &text) const
    {
        return text;
    }

    std::string BaseTokenizer::postprocess(const std::string &text) const
    {
        return text;
    }

    void BaseTokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        std::string input = preprocess(text);
        tp->Encode(input, &ids);
    }

    void BaseTokenizer::encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const
    {
        std::string input = preprocess(q);
        tp->Encode(input, &ids);

        input = preprocess(a);
        tp->Encode(input, &ids);
    }

    std::vector<int> BaseTokenizer::encode(const std::string &text) const
    {
        std::vector<int> ids;
        encode(text, ids);
        return ids;
    }

    std::string BaseTokenizer::decode(const std::vector<int> &ids) const
    {
        // filter out special tokens
        std::vector<int> normal_ids(ids);
        normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(),
                                        [this](int id)
                                        { return is_special_id(id); }),
                        normal_ids.end());
        std::string text;
        tp->Decode(normal_ids, &text);
        text = postprocess(text);
        return text;
    }

    int BaseTokenizer::get_history_start(const std::vector<std::string> &history, int max_length) const
    {
        int start = (int)history.size() - 1;
        size_t total_id_num = encode(history[start]).size();
        start--;
        while (start >= 1)
        {
            total_id_num += encode(history[start]).size();
            total_id_num += encode(history[start - 1]).size();
            if ((int)total_id_num >= max_length)
                break;
            start -= 2;
        }
        return start + 1;
    }

    std::vector<int> BaseTokenizer::encode_history(BaseHistoryEncoder *encoder, const std::vector<std::string> &history, int max_length, const bool incremental)
    {
        CHATLLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();
        std::vector<int> input_ids;

        if (!incremental)
        {
            int start = get_history_start(history, max_length / 2);
            history_offset = start;

            for (; start <= (int)history.size() - 3; start += 2)
            {
                std::string user = trim(history[start]);
                std::string ai = trim(history[start + 1]);
                encoder->append_pair((start - history_offset) / 2, user, ai, input_ids);
            }
        }

        std::string user = trim(history[history.size() - 1]);
        encoder->append_user((int)((history.size() - history_offset - 1) / 2), trim(user), input_ids);

        return input_ids;
    }

    std::vector<int> BaseTokenizer::encode_sys_prompt(void)
    {
        std::vector<int> ids;
        switch (format)
        {
        case ChatFormat::COMPLETION:
            if (completion_encoder)
                completion_encoder->append_sys_prompt(ids);
            else;
            break;

        case ChatFormat::QA:
            if (qa_encoder)
                qa_encoder->append_sys_prompt(ids);
            else if (chat_encoder)
            {
                chat_encoder->append_sys_prompt(ids);
            }
            else;
            break;
        default:
            if (chat_encoder)
                chat_encoder->append_sys_prompt(ids);
            else;
        }

        if (auto_add_bos && (bos_token_id >= 0))
        {
            if ((ids.size() > 0) && (ids[0] != bos_token_id))
                ids.insert(ids.begin(), bos_token_id);
        }
        return ids;
    }

    std::vector<int> BaseTokenizer::encode_history(const std::vector<std::string> &history, int max_length, const bool incremental)
    {
        switch (format)
        {
        case ChatFormat::COMPLETION:
            if (completion_encoder)
                return encode_history(completion_encoder, history, max_length, incremental);
            else;
            break;

        case ChatFormat::QA:
            if (qa_encoder)
                return encode_history(qa_encoder, history, max_length, incremental);
            else if (chat_encoder)
            {
                std::vector<std::string> copied;
                copied.push_back(history[history.size() - 1]);
                return encode_history(chat_encoder, copied, max_length, incremental);
            }
            else;
            break;
        default:
            if (chat_encoder)
                return encode_history(chat_encoder, history, max_length, incremental);
            else;
        }

        std::vector<int> r = encode(history[history.size() - 1]);
        if (auto_add_bos && (bos_token_id >= 0))
        {
            if ((r.size() > 0) && (r[0] != bos_token_id))
                r.insert(r.begin(), bos_token_id);
            else
                r.push_back(bos_token_id);
        }
        return r;
    }

    void BaseTokenizer::set_skip_sys_prompt(bool skip)
    {
        if (chat_encoder)
            chat_encoder->skip_sys_prompt = skip;
        if (completion_encoder)
            completion_encoder->skip_sys_prompt = skip;
        if (qa_encoder)
            qa_encoder->skip_sys_prompt = skip;
    }

    void BaseHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
    }

    void BaseHistoryEncoder::append_pair(int round_idx, const std::string &user, const std::string &ai, std::vector<int> &ids) const
    {
    }

    void BaseHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        if (round_idx == 0)
        {
            if (!skip_sys_prompt)
                append_sys_prompt(ids);
        }
        do_append_user(round_idx, user, ids);
    }

    void BaseHistoryEncoder::do_append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        tokenizer->encode(user, ids);
    }

    static std::string shape_to_string(ggml_tensor *tensor)
    {
        int n_dims = ggml_n_dims(tensor);
        std::ostringstream oss;
        oss << '[';
        for (int i = n_dims - 1; i >= 0; i--)
        {
            oss << tensor->ne[i] << (i > 0 ? ", " : "");
        }
        oss << ']';
        return oss.str();
    }

    static std::string strides_to_string(ggml_tensor *tensor)
    {
        int n_dims = ggml_n_dims(tensor);
        std::ostringstream oss;
        oss << '[';
        for (int i = n_dims - 1; i >= 0; i--)
        {
            oss << tensor->nb[i] << (i > 0 ? ", " : "");
        }
        oss << ']';
        return oss.str();
    }

    std::string to_string(ggml_tensor *tensor, bool with_data)
    {
        std::ostringstream oss;
        int n_dims = ggml_n_dims(tensor);
        oss << "ggml_tensor(";

        if (with_data)
        {
            if (n_dims > 3)
                oss << "[";
            for (int i3 = 0; i3 < tensor->ne[3]; i3++)
            {
                if (n_dims > 2)
                    oss << (i3 > 0 ? ",\n\n[" : "[");
                for (int i2 = 0; i2 < tensor->ne[2]; i2++)
                {
                    if (n_dims > 1)
                        oss << (i2 > 0 ? ",\n\n[" : "[");
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++)
                    {
                        oss << (i1 > 0 ? ",\n[" : "[");
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++)
                        {
                            auto ptr = (char *)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                                       i0 * tensor->nb[0];
                            float val;
                            if (tensor->type == GGML_TYPE_F32)
                            {
                                val = *(float *)ptr;
                            }
                            else if (tensor->type == GGML_TYPE_F16)
                            {
                                val = ggml_fp16_to_fp32(*(ggml_fp16_t *)ptr);
                            }
                            else
                            {
                                CHATLLM_THROW << "unimplemented";
                            }
                            oss << (i0 > 0 ? ", " : "") << std::setw(7) << std::fixed << std::setprecision(4) << val;
                        }
                        oss << "]";
                    }
                    if (n_dims > 1)
                        oss << "]";
                }
                if (n_dims > 2)
                    oss << "]";
            }
            if (n_dims > 3)
                oss << "]";
        }

        oss << ", shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
        return oss.str();
    }

#ifdef _POSIX_MAPPED_FILES
    MappedFile::MappedFile(const std::string &path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        CHATLLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct stat sb;
        CHATLLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
        _size = sb.st_size;

        data = (char *)mmap(nullptr, _size, PROT_READ, MAP_SHARED, fd, 0);
        CHATLLM_CHECK(data != MAP_FAILED) << strerror(errno);

        CHATLLM_CHECK(close(fd) == 0) << strerror(errno);
        ptr = data;
    }

    MappedFile::~MappedFile() { CHATLLM_CHECK(munmap(data, _size) == 0) << strerror(errno); }
#elif defined(_WIN32)
    MappedFile::MappedFile(const std::string &path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        CHATLLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct _stat64 sb;
        CHATLLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
        _size = sb.st_size;

        HANDLE hFile = (HANDLE)_get_osfhandle(fd);

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        CHATLLM_CHECK(hMapping != NULL) << strerror(errno);

        data = (char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMapping);

        CHATLLM_CHECK(data != NULL) << strerror(errno);
        CHATLLM_CHECK(close(fd) == 0) << strerror(errno);
        ptr = data;
    }

    MappedFile::~MappedFile()
    {
        CHATLLM_CHECK(UnmapViewOfFile(data)) << strerror(errno);
    }
#endif

    int64_t MappedFile::tell()
    {
        return ptr - data;
    }

    void MappedFile::seek(int64_t offset, int whence)
    {
        if (whence == SEEK_SET)
            ptr = data + offset;
        else if (whence == SEEK_CUR)
            ptr += offset;
        else if (whence == SEEK_END)
            ptr = data + size() + offset;
        else
            CHATLLM_THROW << "invalid seek mode " << whence;
    }

    size_t MappedFile::read_buffer(void *output, size_t len)
    {
        size_t remain = size() - tell();
        if (len > remain) len = remain;
        memcpy(output, ptr, len);
        ptr += len;
        return len;
    }

    SimpleFile::SimpleFile(const std::string &path)
    {
        f = std::fopen(path.c_str(), "rb");
        CHATLLM_CHECK(f != nullptr) << "cannot open file " << path << ": " << strerror(errno);
        seek(0, SEEK_END);
        _size = tell();
        seek(0, SEEK_SET);
    }

    SimpleFile::~SimpleFile()
    {
        std::fclose(f);
        f = nullptr;
    }

#if defined(_MSC_VER)
#define ftello64    _ftelli64
#define fseeko64    _fseeki64
#endif

    int64_t SimpleFile::tell()
    {
        return ftello64(f);
    }

    void SimpleFile::seek(int64_t offset, int whence)
    {
        CHATLLM_CHECK(fseeko64(f, offset, whence) == 0) << "SimpleFile::seek: fails offset = " << offset << ", whence = " << whence;
    }

    size_t SimpleFile::read_buffer(void *output, size_t len)
    {
        return fread(output, 1, len, f);
    }

    struct ggml_tensor * ggml_init_tensor(struct ggml_tensor *tensor,
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t       * ne)
    {
        struct ggml_tensor *result = tensor;

        *result = ggml_tensor {
            /*.type         =*/ type,
            /*.backend      =*/ GGML_BACKEND_TYPE_CPU,
            /*.buffer       =*/ NULL,
            /*.ne           =*/ { 1, 1, 1, 1 },
            /*.nb           =*/ { 0, 0, 0, 0 },
            /*.op           =*/ GGML_OP_NONE,
            /*.op_params    =*/ { 0 },
            /*.flags        =*/ 0,
            /*.grad         =*/ NULL,
            /*.src          =*/ { NULL },
            /*.perf_runs    =*/ 0,
            /*.perf_cycles  =*/ 0,
            /*.perf_time_us =*/ 0,
            /*.view_src     =*/ NULL,
            /*.view_offs    =*/ 0,
            /*.data         =*/ NULL,
            /*.name         =*/ { 0 },
            /*.extra        =*/ NULL,
            /*.padding      =*/ { 0 },
        };

        for (int i = 0; i < n_dims; i++) {
            result->ne[i] = ne[i];
        }

        result->nb[0] = ggml_type_size(type);
        result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
        for (int i = 2; i < GGML_MAX_DIMS; i++) {
            result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
        }

        return result;
    }

    TensorInfo::TensorInfo(enum ggml_type type, int n_dim, const int64_t *ne, size_t _offset)
        : _offset(_offset), data(nullptr)
    {
        ggml_init_tensor(&tensor, type, n_dim, ne);
    }

    TensorInfo::~TensorInfo()
    {
        if (data)
            free(data);
        data = nullptr;
    }

    size_t TensorInfo::aligned_data_start(size_t offset)
    {
        constexpr size_t MEM_ALIGNED = 16;
        return (offset + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
    }

    size_t TensorInfo::aligned_size()
    {
        return (aligned_data_start(_offset) - _offset) + ggml_nbytes(&tensor);
    }

    void *TensorInfo::load(tokenizer::DataReader *reader)
    {
        if (data) return tensor.data;

        constexpr int64_t MEM_ALIGNED = 16;
        data = malloc(ggml_nbytes(&tensor) + MEM_ALIGNED);
        tensor.data = (void *)aligned_data_start((size_t)data);

        reader->seek(aligned_data_start(_offset), SEEK_SET);
        reader->read_buffer(tensor.data, ggml_nbytes(&tensor));
        return tensor.data;
    }

    void ModelLoader::load_all_tensors(void)
    {
        if (tensor_dict.size() > 0) return;

        while (tell() < _file->size())
        {
            std::string weight_name;

            // read and check tensor name
            {
                int name_size = read_basic<int>();
                weight_name = read_string(name_size);
            }

            int64_t ne[4] = {1,1,1,1};

            // read and check tensor shape
            int ndim = read_basic<int>();
            for (int i = ndim - 1; i >= 0; i--)
            {
                int dim_size = read_basic<int>();
                ne[i] = dim_size;
            }

            // read and check tensor dtype
            ggml_type dtype = (ggml_type)read_basic<int>();

            tensor_dict.emplace(weight_name, TensorInfo(dtype, ndim, ne, tell()));

            TensorInfo &t = tensor_dict.at(weight_name);

            seek(t.aligned_size(), SEEK_CUR);
        }
    }

    void ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor)
    {
        auto search = tensor_dict.find(name);
        CHATLLM_CHECK(search != tensor_dict.end()) << "tensor not exist: " << name;

        ggml_set_name(tensor, name.c_str());
        TensorInfo &t = search->second;

        CHATLLM_CHECK(t.tensor.type == tensor->type)
                << "tensor " << name << " dtype mismatch: expect " << tensor->type << " but got " << t.tensor.type;

        // read and check tensor shape
        {
            int ndim = ggml_n_dims(&t.tensor);
            int n_dims = ggml_n_dims(tensor);

            // a quick fix
            if ((n_dims == 1) && (ndim == 2) && (tensor->ne[1] == 1))
                n_dims = 2;

            CHATLLM_CHECK(ndim == n_dims)
                << "tensor " << name << " ndim mismatch: expect " << n_dims << " but got " << ndim;
            for (int i = 0; i < ndim; i++)
            {
                int64_t dim_size = t.tensor.ne[i];
                CHATLLM_CHECK(dim_size == tensor->ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                                         << ": expect " << tensor->ne[i] << " but got " << dim_size;
            }
        }

        tensor->data = t.load(_file.get());
    }

    ModelObject::ModelObject(const std::string &path)
        : ModelObject(path, ModelObject::extra_args())
    {
    }

    ModelObject::ModelObject(const std::string &path, const extra_args &args)
        : loader(nullptr), loaded(path.size() > 0)
    {
        ModelFactory::Result result = {nullptr, nullptr};
        if (path.size() > 0)
        {
            loader = std::unique_ptr<ModelLoader>(new ModelLoader(path));
            if (!ModelFactory::load(*loader, result, args))
                CHATLLM_THROW << "ModelFactory::load() failed";
        }

        tokenizer = std::move(result.tokenizer);
        model = std::move(result.model);
    }

    void ModelObject::text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result)
    {
        if (!loaded) return;
        std::vector<int> input_ids;
        tokenizer->encode(input, input_ids);
        model->text_embedding(gen_config, input_ids, result);
    }

    AbstractModel *ModelObject::fork_model(const extra_args &args)
    {
        if (!loaded) return nullptr;
        return ModelFactory::load_model_again(*loader, args);
    }

    // ===== pipeline =====

    Pipeline::Pipeline(const std::string &path)
        : Pipeline(path, ModelObject::extra_args())
    {
    }

    Pipeline::Pipeline(const std::string &path, const ModelObject::extra_args &args)
        : gen_max_tokens(-1),
          initializing(true),
          extending(ExtendingMethod::Restart),
          modelobj(path, args)
    {
        model = modelobj.model.get();
        tokenizer = modelobj.tokenizer.get();
    }

    std::string Pipeline::chat_with_restart(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = tokenizer->get_chat_format() == ChatFormat::CHAT;
        bool completed = false;
        std::vector<int> input_ids;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, gen_max_tokens, streamer);
        if (!completed)
        {
            if (continuous)
            {
                streamer->putln("\nRUN OUT OF CONTEXT. Let me forget something and try again ...\n");
                input_ids = tokenizer->encode_history(history, gen_config.max_context_length);
                output_ids = model->generate(input_ids, gen_config, false, completed, &performance, gen_max_tokens, streamer);
            }
            else
                streamer->putln("\nRUN OUT OF CONTEXT. I have to stop now.\n");
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    std::string Pipeline::chat_without_extending(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = tokenizer->get_chat_format() == ChatFormat::CHAT;
        bool completed = false;
        std::vector<int> input_ids;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, gen_max_tokens, streamer);
        if (!completed)
        {
            streamer->putln("\nRUN OUT OF CONTEXT. I have to stop now.\n");
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    std::string Pipeline::chat_with_shift(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = true;
        bool completed = false;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        std::vector<int> input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous);
        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, gen_max_tokens, streamer);

        while (!completed)
        {
            streamer->putln("\nRUN OUT OF CONTEXT. Try to forget something and continue ...\n");
            model->shift_memory(gen_config.max_context_length);
            if (output_ids.size() > 0)
                input_ids = {output_ids[output_ids.size() - 1]};
            else
                input_ids.clear();

            auto ids = model->generate(input_ids, gen_config, true, completed, &performance, gen_max_tokens, streamer);
            output_ids.insert(output_ids.end(), ids.begin(), ids.end());
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    void Pipeline::eval_sys_prompt(const GenerationConfig &gen_config)
    {
        bool completed = false;
        std::vector<int> input_ids = tokenizer->encode_sys_prompt();

        model->generate(input_ids, gen_config, false, completed, &performance, 1, nullptr);

        // just in case that chatting is continued
        tokenizer->set_skip_sys_prompt(true);
        initializing = false;
        return;
    }

    std::string Pipeline::chat(std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        std::string r;

        if (modelobj.loaded && streamer)
        {
            ChunkInterceptor *interceptor = model->get_interceptor();
            if (interceptor)
            {
                streamer->set_interceptor(interceptor);
            }
        }

        before_chat(history, gen_config, streamer);

        if (modelobj.loaded)
        {
            switch (extending)
            {
            case ExtendingMethod::Shift:
                r = chat_with_shift(history, gen_config, streamer);
                break;
            case ExtendingMethod::Restart:
                r = chat_with_restart(history, gen_config, streamer);
                break;
            default:
                r = chat_without_extending(history, gen_config, streamer);
                break;
            }
        }

        post_chat(history, gen_config, streamer);

        if (streamer)
            streamer->end();

        return r;
    }

    void Pipeline::abort_generation(void)
    {
        if (modelobj.loaded)
            modelobj.model->abort_generation();
    }

    void Pipeline::text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result)
    {
        modelobj.text_embedding(input, gen_config, result);
    }

    int Pipeline::get_text_embedding_dim(void)
    {
        if (!modelobj.loaded) return 0;
        return model->get_text_embedding_dim();
    }

    std::string Pipeline::get_additional_description(void) const
    {
        return "";
    }

    void Pipeline::restart(void)
    {
        initializing = true;
    }

    void Pipeline::rewind(int n_past)
    {
        if (!modelobj.loaded) return;
        model->set_n_past(n_past);
    }

    int Pipeline::save_session(const std::vector<std::string> &history, const std::string &file_name)
    {
        if (!modelobj.loaded) return -1000;

        FILE *f = fopen(file_name.c_str(), "wb");
        file_header header = {0};
        memcpy(header.magic, head_magic, sizeof(header.magic));
        header.history_len = history.size();

        if (fwrite(&header, sizeof(header), 1, f) != 1)
            return -1;

        for (auto &s : history)
        {
            size_t len = s.size();
            fwrite(&len, sizeof(len), 1, f);
            fwrite(s.data(), 1, len, f);
        }

        model->save_session(f);

        fclose(f);
        return 0;
    }

    int Pipeline::load_session(std::vector<std::string> &history, const std::string &file_name, BaseStreamer *streamer, int *n_past)
    {
        if (!modelobj.loaded) return -1000;

        int r = -1;
        FILE *f = fopen(file_name.c_str(), "rb");
        file_header header = {0};
        if (fread(&header, sizeof(header), 1, f) != 1)
            return -1;

        history.clear();

        if (memcmp(header.magic, head_magic, sizeof(header.magic)) == 0)
        {
            for (size_t i = 0; i < header.history_len; i++)
            {
                std::string s;
                size_t len;
                if (fread(&len, sizeof(len), 1, f) != 1)
                    return -2;

                s.resize(len);
                if (fread(s.data(), 1, len, f) != len)
                    return -3;
                history.push_back(s);
                if (streamer)
                {
                    if ((history.size() % 2) == 0)
                        streamer->put_history_ai(s);
                    else
                        streamer->put_history_user(s);
                }
            }

            r = model->load_session(f);
            if (r != 0) goto exit;
        }
    exit:
        fclose(f);
        if (r == 0)
        {
            initializing = false;
            tokenizer->set_skip_sys_prompt(true);
            if (n_past != nullptr)
                *n_past = model->get_n_past();
        }
        return r;
    }

    float Pipeline::qa_rank(const std::string &q, const std::string &a, const GenerationConfig &gen_config)
    {
        if (!modelobj.loaded) return -1.0f;
        std::vector<int> input_ids;
        tokenizer->encode_qa(q, a, input_ids);
        return model->qa_rank(gen_config, input_ids);
    }

    void Pipeline::set_system_prompt(const std::string &prompt)
    {
        if (!modelobj.loaded) return;
        tokenizer->set_system_prompt(prompt);
    }

    void Pipeline::set_extending_method(ExtendingMethod method)
    {
        extending = method;
    }

    void Pipeline::set_additional_args(const std::map<std::string, std::string> &args)
    {
        if (!modelobj.loaded) return;
        tokenizer->set_additional_args(args);
    }

    void Pipeline::before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
    }

    void Pipeline::post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
    }

    RAGPipeline::RAGPipeline(const std::string &path, const ModelObject::extra_args &args,
        DistanceStrategy vec_cmp, const std::vector<std::string> &vector_stores,
        const std::string &embedding_model, const std::string &reranker_model)
        : Pipeline(path, args),
          composer(),
          hide_reference(false),
          retrieve_top_n(5), rerank_top_n(3),
          dump(false),
          rerank_score_threshold(0.5f),
          rag_post_extending(0),
          rerank_rewrite(false),
          vs(vec_cmp, vector_stores),
          embedding(embedding_model),
          reranker(nullptr),
          rewrite_model(nullptr)
    {
        if (reranker_model.size() > 0)
            reranker = new ModelObject(reranker_model);
    }

    void RAGPipeline::rerank(const std::string &query, std::vector<int64_t> &candidates, const GenerationConfig &gen_config, int top_n)
    {
        std::vector<float> scores;
        std::vector<size_t> order;
        std::vector<int64_t> result;

        for (size_t i = 0; i < candidates.size(); i++)
        {
            std::vector<int> input_ids;
            std::string c, m;
            vs.GetRecord(candidates[i], c, m);
            reranker->tokenizer->encode_qa(query, c, input_ids);
            scores.push_back(reranker->model->qa_rank(gen_config, input_ids));
        }

        chatllm::ordering(scores, order, true);

        if (top_n > (int)order.size()) top_n = (int)order.size();

        for (int i = 0; i < top_n; i++)
        {
            size_t index = order[i];
            if (scores[index] >= rerank_score_threshold)
                result.push_back(candidates[index]);
            else
                break;
        }

        candidates.clear();
        candidates.insert(candidates.begin(), result.begin(), result.end());
    }

    std::string RAGPipeline::rewrite_query(const std::string &prompt, const GenerationConfig &gen_config)
    {
        if (nullptr == rewrite_model)
        {
            rewrite_model = modelobj.fork_model(ModelObject::extra_args());
            rewrite_model->set_tokenizer(tokenizer);
        }

        std::vector<std::string> history;
        history.push_back(prompt);
        bool completed = false;

        std::vector<int> input_ids = tokenizer->encode_history(history, gen_config.max_context_length, false);
        std::vector<int> output_ids = rewrite_model->generate(input_ids, gen_config, false, completed, &performance, -1, nullptr);

        if (!completed) return "";

        return tokenizer->decode(output_ids);
    }

    void RAGPipeline::before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        size_t index = history.size() - 1;
        std::string query(history.back());
        std::string rewritten_query(query);
        std::vector<float> query_emb;
        std::vector<int64_t> selected;

        metainfo.clear();

        if (modelobj.loaded && composer.is_rewritten_template_set())
        {
            std::string rewritten = composer.rewrite_query_for_retrieve(query);
            rewritten_query = rewrite_query(rewritten, gen_config);
            rewritten_query = composer.parse_rewritten_query_result(rewritten_query);
            if (rewritten_query.size() > 0)
                streamer->put_rewritten_query(rewritten_query);
            else
                rewritten_query = query;
        }

        embedding.text_embedding(rewritten_query, gen_config, query_emb);

        vs.Query(query_emb, selected, retrieve_top_n);

        if (reranker != nullptr)
            rerank(rerank_rewrite ? rewritten_query : query, selected, gen_config, rerank_top_n);

        std::vector<std::string> augments;

        for (auto i : selected)
        {
            std::string c, m;
            vs.GetRecord(i, c, m);
            augments.push_back(c);
            metainfo.push_back(m);

            size_t last = augments.size() - 1;

            if (rag_post_extending > 0)
            {
                for (auto j = i - 1; (j >= 0) && (j >= i - rag_post_extending); j--)
                {
                    std::string c0, m0;
                    vs.GetRecord(j, c0, m0);
                    if (m0 == m)
                        augments[last] = c0 + "\n" + augments[last];
                    else
                        break;
                }

                for (auto j = i + 1; (j >= 0) && (j <= i + rag_post_extending); j++)
                {
                    std::string c0, m0;
                    vs.GetRecord(j, c0, m0);
                    if (m0 == m)
                        augments[last] = augments[last] + "\n" + c0;
                    else
                        break;
                }
            }
        }

        if (augments.size() < 1) return;

        if (dump)
        {
            for (size_t i = 0; i < augments.size(); i++)
            {
                streamer->putln(metainfo[i]);
                streamer->putln(augments[i]);
                streamer->putln("");
            }
        }

        auto composed = composer.compose_augmented_query(query, augments);

        history[index] = composed;
    }

    std::string RAGPipeline::get_additional_description(void) const
    {
        std::ostringstream oss;

        int64_t total_param_num = embedding.model->get_param_num(false);

        oss << "Augmented by " << embedding.model->type_name() << " (" << std::fixed << std::setprecision(1) << (double)total_param_num / 1000000000. << "B)";

        if (reranker != nullptr)
        {
            total_param_num = reranker->model->get_param_num(false);
            oss << " and " << reranker->model->type_name() << " (" << std::fixed << std::setprecision(1) << (double)total_param_num / 1000000000. << "B)";
        }

        oss << ".";

        return oss.str();
    }

    void RAGPipeline::post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        if (hide_reference || (metainfo.size() < 1)) return;

        std::set<std::string> merged;

        for (auto &s : metainfo)
        {
            if (merged.find(s) != merged.end()) continue;

            merged.insert(s);
            streamer->put_reference(s);
        }
    }

    AugmentedQueryComposer::AugmentedQueryComposer()
        : prompt_template("Answer this question based on given information: {question}\n```\n{context}\n```"),
          context_sep("\n```\n")
    {}

    void replace_all(std::string &s, const std::string &sub, const std::string replace)
    {
        size_t pos = 0;
        while (true)
        {
            pos = s.find(sub, pos);
            if (pos == std::string::npos) {
                break;
            }

            s.erase(pos, sub.length());
            s.insert(pos, replace);

            pos += replace.length();
        }
    }

    void unescape_c_sequences(std::string &s)
    {
        replace_all(s, "\\n", "\n");
        replace_all(s, "\\t", "\t");
    }

    std::string AugmentedQueryComposer::compose_augmented_query(const std::string &query, const std::vector<std::string> augments) const
    {
        const static std::regex r(R""([\r\n]+)"");

        if (augments.size() < 1) return query;

        std::ostringstream oss;
        oss << augments[0];
        for (size_t i = 1; i < augments.size(); i++)
            oss << context_sep << augments[i];

        std::string context = oss.str();

        std::string s(prompt_template);
        replace_all(s, "{context}", oss.str());
        replace_all(s, "{question}", query);
        return s;
    }

    bool AugmentedQueryComposer::is_rewritten_template_set(void) const
    {
        return query_rewritten_template.find("{question}", 0) != std::string::npos;
    }

    std::string AugmentedQueryComposer::rewrite_query_for_retrieve(const std::string &query) const
    {
        if (query_rewritten_template.size() < 1)
            return query;

        std::string s(query_rewritten_template);
        replace_all(s, "{question}", query);
        return s;
    }

    // TODO: this might not work because:
    // 1. some LLM does not support this;
    // 2. output format varies.
    std::string AugmentedQueryComposer::parse_rewritten_query_result(const std::string &query) const
    {
        std::string r(query);
        size_t pos = 0;
        tokenizer::TextTrim t;
        const char *delimiters[] = {":", "："};
        bool found = false;

        pos = r.find("\n");
        if (pos != std::string::npos)
        {
            // xxx:
            // ...
            // ...
            r = r.substr(pos + 1);
            std::ostringstream oss;

            while (r.size() > 0)
            {
                std::string part;
                pos = r.find("\n");
                if (pos != std::string::npos)
                {
                    part = r.substr(0, pos);
                    r = r.substr(pos + 1);
                }
                else
                {
                    part = r;
                    r = "";
                }

                pos = part.find(" ");
                if (pos != std::string::npos)
                    part = part.substr(pos + 1);

                part = t.transform(part);

                oss << part << " ";
            }

            r = oss.str();
            r = t.transform(r);

            goto final;
        }

        for (const char *delimiter : delimiters)
        {
            pos = r.find(delimiter);
            if (pos != std::string::npos)
            {
                r = r.substr(pos + strlen(delimiter));
                found = true;
                break;
            }
        }

        if (found)
        {
            replace_all(r, ",", " ");
            replace_all(r, "，", " ");
            replace_all(r, ".", " ");

            r = t.transform(r);

            goto final;
        }

        if ((pos = r.find(",")) != std::string::npos)
        {
            replace_all(r, ",", " ");
            replace_all(r, "，", " ");
            replace_all(r, ".", " ");

            r = t.transform(r);

            goto final;
        }

        return "";

final:
        tokenizer::TextPrepDeleteMultiSpaces del;
        return del.transform(r);
    }

    void AugmentedQueryComposer::set_prompt_template(const std::string &s)
    {
        if (s.size() < 1) return;
        prompt_template = s;
        unescape_c_sequences(prompt_template);
    }

    void AugmentedQueryComposer::set_rewrite_template(const std::string &s)
    {
        if (s.size() < 1) return;
        query_rewritten_template = s;
        unescape_c_sequences(query_rewritten_template);
    }

    void AugmentedQueryComposer::set_context_sep(const std::string &s)
    {
        if (s.size() < 1) return;
        context_sep = s;
        unescape_c_sequences(context_sep);
    }

    ModelPerfInfo::ModelPerfInfo()
    {
        memset(&timings, 0, sizeof(timings));
    }

    void ModelPerfInfo::Accumulate(Type type, size_t tok_count)
    {
        timings[type].tok_count += tok_count;
        timings[type].duration_ms += Elapsed();
    }

    void ModelPerfInfo::Reset(void)
    {
        m_beg = Clock::now();
    }

    double ModelPerfInfo::Elapsed(void)
    {
        double r = std::chrono::duration_cast<MilliSecond>(Clock::now() - m_beg).count();
        Reset();
        return r;
    }

} // namespace chatllm
