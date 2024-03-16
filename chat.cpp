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
        qa_encoder(qa_encoder)
    {
        if (chat_encoder)
            chat_encoder->set_tokenizer(this);
        if (completion_encoder)
            completion_encoder->set_tokenizer(this);
        if (qa_encoder)
            qa_encoder->set_tokenizer(this);
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

    std::vector<int> BaseTokenizer::encode_history(const std::vector<std::string> &history, int max_length, const bool incremental)
    {
        switch (format)
        {
        case ChatFormat::COMPLETION:
            if (completion_encoder)
                return encode_history(completion_encoder, history, max_length, incremental);
            else
                return encode(history[history.size() - 1]);
        case ChatFormat::QA:
            if (qa_encoder)
                return encode_history(qa_encoder, history, max_length, incremental);
            else if (chat_encoder)
            {
                std::vector<std::string> copied;
                copied.push_back(history[history.size() - 1]);
                return encode_history(chat_encoder, copied, max_length, incremental);
            }
            else
                return encode(history[history.size() - 1]);
        default:
            if (chat_encoder)
                return encode_history(chat_encoder, history, max_length, incremental);
            else
                return encode(history[history.size() - 1]);
        }
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

    // ===== streamer =====

    void TextStreamer::putln(const std::string &line)
    {
        std::cout << line << std::endl << std::flush;
    }

    void TextStreamer::put(const std::vector<int> &output_ids)
    {
        if (is_prompt_)
        {
            // skip prompt
            is_prompt_ = false;
            return;
        }

        static const std::vector<char> puncts{',', '!', ':', ';', '?'};

        token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
        std::string text = tokenizer_->decode(token_cache_);
        if (text.empty())
        {
            return;
        }

        std::string printable_text;
        if ((text.back() == '\n') || (text.back() == '\r'))
        {
            // flush the cache after newline
            printable_text = text.substr(print_len_);
            token_cache_.clear();
            print_len_ = 0;
        }
        else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end())
        {
            // last symbol is a punctuation, hold on
        }
        else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0)
        {
            // ends with an incomplete token, hold on
        }
        else
        {
            printable_text = text.substr(print_len_);
            print_len_ = text.size();
        }

        std::cout << printable_text << std::flush;
    }

    void TextStreamer::end()
    {
        std::string text = tokenizer_->decode(token_cache_);
        std::cout << text.substr(print_len_) << std::endl;
        is_prompt_ = true;
        token_cache_.clear();
        print_len_ = 0;
    }

#ifdef _POSIX_MAPPED_FILES
    MappedFile::MappedFile(const std::string &path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        CHATLLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct stat sb;
        CHATLLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
        size = sb.st_size;

        data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        CHATLLM_CHECK(data != MAP_FAILED) << strerror(errno);

        CHATLLM_CHECK(close(fd) == 0) << strerror(errno);
    }

    MappedFile::~MappedFile() { CHATLLM_CHECK(munmap(data, size) == 0) << strerror(errno); }
#elif defined(_WIN32)
    MappedFile::MappedFile(const std::string &path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        CHATLLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct _stat64 sb;
        CHATLLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
        size = sb.st_size;

        HANDLE hFile = (HANDLE)_get_osfhandle(fd);

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        CHATLLM_CHECK(hMapping != NULL) << strerror(errno);

        data = (char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMapping);

        CHATLLM_CHECK(data != NULL) << strerror(errno);

        CHATLLM_CHECK(close(fd) == 0) << strerror(errno);
    }

    MappedFile::~MappedFile()
    {
        CHATLLM_CHECK(UnmapViewOfFile(data)) << strerror(errno);
    }
#endif

    void ModelLoader::seek(int64_t offset, int whence)
    {
        if (whence == SEEK_SET)
            ptr = data + offset;
        else if (whence == SEEK_CUR)
            ptr += offset;
        else if (whence == SEEK_END)
            ptr = data + size + offset;
        else
            CHATLLM_THROW << "invalid seek mode " << whence;
    }

    std::string ModelLoader::read_string(size_t length)
    {
        std::string s(ptr, ptr + length);
        ptr += length;
        return s;
    }

    void ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor)
    {
        // read and check tensor name
        {
            int name_size = read_basic<int>();
            CHATLLM_CHECK(name_size == (int)name.size())
                << "tensor " << name << " name size mismatch: expect " << name.size() << " but got " << name_size
                << "(part of name: " << read_string(name.size()) << ")";
            std::string weight_name = read_string(name_size);
            CHATLLM_CHECK(weight_name == name) << "tensor name mismatch: expect " << name << " but got " << weight_name;
            ggml_set_name(tensor, weight_name.c_str());
        }

        // read and check tensor shape
        {
            int ndim = read_basic<int>();
            int n_dims = ggml_n_dims(tensor);

            // a quick fix
            if ((n_dims == 1) && (ndim == 2) && (tensor->ne[1] == 1))
                n_dims = 2;

            CHATLLM_CHECK(ndim == n_dims)
                << "tensor " << name << " ndim mismatch: expect " << n_dims << " but got " << ndim;
            for (int i = ndim - 1; i >= 0; i--)
            {
                int dim_size = read_basic<int>();
                CHATLLM_CHECK(dim_size == tensor->ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                                         << ": expect " << tensor->ne[i] << " but got " << dim_size;
            }
        }

        // read and check tensor dtype
        {
            ggml_type dtype = (ggml_type)read_basic<int>();
            CHATLLM_CHECK(dtype == tensor->type)
                << "tensor " << name << " dtype mismatch: expect " << tensor->type << " but got " << dtype;
        }

        // map tensor data
        {
            constexpr int64_t MEM_ALIGNED = 16;
            const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
            tensor->data = const_cast<char *const>(data) + data_offset;
            seek(data_offset + ggml_nbytes(tensor), SEEK_SET);
        }
    }

    ModelObject::ModelObject(const std::string &path)
        : ModelObject(path, ModelObject::extra_args())
    {
    }

    ModelObject::ModelObject(const std::string &path, const extra_args &args)
    {
        loader = std::unique_ptr<ModelLoader>(new ModelLoader(path));

        ModelFactory::Result result = {nullptr, nullptr};
        if (!ModelFactory::load(*loader, result, args.max_length))
            CHATLLM_THROW << "ModelFactory::load() failed";

        tokenizer = std::move(result.tokenizer);
        model = std::move(result.model);
    }

    void ModelObject::text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result)
    {
        std::vector<int> input_ids;
        tokenizer->encode(input, input_ids);
        model->text_embedding(gen_config, input_ids, result);
    }

    // ===== pipeline =====

    Pipeline::Pipeline(const std::string &path)
        : Pipeline(path, ModelObject::extra_args())
    {
    }

    Pipeline::Pipeline(const std::string &path, const ModelObject::extra_args &args)
        : initializing(true),
          extending(ExtendingMethod::Restart),
          modelobj(path, args)
    {
        model = modelobj.model.get();
        tokenizer = modelobj.tokenizer.get();
    }

    std::string Pipeline::chat_with_restart(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = true;
        bool completed = false;
        std::vector<int> input_ids;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, streamer);
        if (!completed)
        {
            if (continuous)
            {
                std::cout << std::endl << "RUN OUT OF CONTEXT. Let me forget something and try again ..." << std::endl << std::endl;
                input_ids = tokenizer->encode_history(history, gen_config.max_context_length);
                output_ids = model->generate(input_ids, gen_config, false, completed, streamer);
            }
            else
                std::cout << std::endl << "RUN OUT OF CONTEXT. I have to stop now." << std::endl << std::endl;
        }

        std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
        std::string output = tokenizer->decode(new_output_ids);
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
        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, streamer);

        while (!completed)
        {
            std::cout << std::endl << "RUN OUT OF CONTEXT. Try to forget something and continue ..." << std::endl << std::endl;
            model->shift_memory(gen_config.max_context_length);
            if (output_ids.size() > 0)
                input_ids = {output_ids[output_ids.size() - 1]};
            else
                input_ids.clear();

            auto ids = model->generate(input_ids, gen_config, true, completed, streamer);
            output_ids.insert(output_ids.end(), ids.begin(), ids.end());
        }

        std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
        std::string output = tokenizer->decode(new_output_ids);
        return output;
    }

    std::string Pipeline::chat(std::vector<std::string> &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        std::string r;
        before_chat(history, gen_config);

        switch (extending)
        {
        case ExtendingMethod::Shift:
            r = chat_with_shift(history, gen_config, streamer);
        default:
            r = chat_with_restart(history, gen_config, streamer);
        }
        post_chat(history, gen_config, streamer);
        return r;
    }

    void Pipeline::text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result)
    {
        modelobj.text_embedding(input, gen_config, result);
    }

    int Pipeline::get_text_embedding_dim(void)
    {
        return model->get_text_embedding_dim();
    }

    std::string Pipeline::get_additional_description(void) const
    {
        return "";
    }

    float Pipeline::qa_rank(const std::string &q, const std::string &a, const GenerationConfig &gen_config)
    {
        std::vector<int> input_ids;
        tokenizer->encode_qa(q, a, input_ids);
        return model->qa_rank(gen_config, input_ids);
    }

    void Pipeline::set_system_prompt(const std::string &prompt)
    {
        tokenizer->set_system_prompt(prompt);
    }

    void Pipeline::set_extending_method(ExtendingMethod method)
    {
        extending = method;
    }

    void Pipeline::set_additional_args(const std::map<std::string, std::string> &args)
    {
        tokenizer->set_additional_args(args);
    }

    void Pipeline::before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config)
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
          reference_tag("References:"), hide_reference(false),
          retrieve_top_n(5), rerank_top_n(3),
          vs(vec_cmp, vector_stores),
          embedding(embedding_model),
          reranker(nullptr)
    {
        if (reranker_model.size() > 0)
            reranker = new ModelObject(reranker_model);
    }

    void RAGPipeline::rerank(const std::string &query, std::vector<int64_t> &candidates, const GenerationConfig &gen_config, int top_n)
    {
        std::vector<float> scores;
        std::vector<int> order;
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
            int index = order[i];
            if (reranker->model->qa_acceptable(scores[index]))
                result.push_back(candidates[index]);
            else
                break;
        }

        candidates.clear();
        candidates.insert(candidates.begin(), result.begin(), result.end());
    }

    void RAGPipeline::before_chat(std::vector<std::string> &history, const GenerationConfig &gen_config)
    {
        size_t index = history.size() - 1;
        std::string query(history.back());
        std::vector<float> query_emb;
        std::vector<int64_t> selected;

        metainfo.clear();

        embedding.text_embedding(query, gen_config, query_emb);

        vs.Query(query_emb, selected, retrieve_top_n);

        if (reranker != nullptr)
            rerank(query, selected, gen_config, rerank_top_n);

        std::vector<std::string> augments;

        for (auto i : selected)
        {
            std::string c, m;
            vs.GetRecord(i, c, m);
            augments.push_back(c);
            metainfo.emplace(m);
        }

        if (augments.size() < 1) return;

        auto composed = composer.compose_augmented_query(query, augments);

        history[index] = composed;
    }

    std::string RAGPipeline::get_additional_description(void) const
    {
        std::ostringstream oss;

        int64_t total_param_num = embedding.model->get_param_num(false);

        oss << "Augmented by " << embedding.model->type_name() << " (" << std::fixed << std::setprecision(1) << total_param_num / 1000000000. << "B)";

        if (reranker != nullptr)
        {
            total_param_num = reranker->model->get_param_num(false);
            oss << " and " << reranker->model->type_name() << " (" << std::fixed << std::setprecision(1) << total_param_num / 1000000000. << "B)";
        }

        oss << ".";

        return oss.str();
    }

    void RAGPipeline::post_chat(std::vector<std::string> &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        if (hide_reference || (metainfo.size() < 1)) return;

        streamer->putln("");
        streamer->putln(reference_tag);
        for (auto s : metainfo)
            streamer->putln("1. " + s);
    }

    AugmentedQueryComposer::AugmentedQueryComposer()
        : prompt_template("Answer the question based on given information: {question}\n```\n{context}"),
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

    void AugmentedQueryComposer::set_prompt_template(const std::string &s)
    {
        prompt_template = s;
        unescape_c_sequences(prompt_template);
    }

    void AugmentedQueryComposer::set_context_sep(const std::string &s)
    {
        context_sep = s;
        unescape_c_sequences(context_sep);
    }

} // namespace chatllm
