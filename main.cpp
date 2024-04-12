#include "chat.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <thread>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstring>
#include <random>
#include <map>

#include "vectorstore.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

struct Args
{
    std::string model_path = "";
    std::string embedding_model_path = "";
    std::string reranker_model_path = "";
    std::vector<std::string> vector_store;
    std::string vector_store_in = "";
    std::string system = "";
    std::string prompt = "你好";
    std::string sampling = "top_p";
    std::string extending = "restart";
    std::string test_fn = "";
    std::string rag_template = "";
    std::string rag_context_sep = "";
    std::string retrieve_rewrite_template = "";
    std::map<std::string, std::string> additional;
    int max_length = -1;
    int max_context_length = 512;
    bool interactive = false;
    int top_k = 0;
    float top_p = 0.7f;
    float temp = 0.7f;
    float tfs_z = 0.95f;
    float presence_penalty = 1.0f;
    int num_threads = 0;
    bool multi_line = false;
    int seed;
    chatllm::ChatFormat format = chatllm::ChatFormat::CHAT;
    bool tokenize = false;
    DistanceStrategy vc = DistanceStrategy::MaxInnerProduct;
    int retrieve_top_n = 2;
    int rerank_top_n = 1;
    float rerank_score_thres = 0.35f;
    int rag_post_extending = 0;
    bool hide_reference = false;
    bool rag_dump = false;
    bool show_banner = true;
    bool show_help = false;
    bool rerank_rewrite = false;
};

#define MULTI_LINE_END_MARKER_W  L"\\."
#define MULTI_LINE_END_MARKER     "\\."

void usage(const std::string &prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Basic options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        model path\n"
              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
              << "  -s, --system SYSTEM     system prompt (instruction) (default: model specific)\n"
              << "  -i, --interactive       run in interactive mode\n"
              << "  -l, --max_length N      max total length including prompt and output (default: model specific)\n"
              << "                          generally, this is used to reduce KV cache size.\n"
              << "                          for models that does not show its max context window in `config.json`,\n"
              << "                          use this to enlarge it (use with caution!).\n"
              << "  -n, --threads N         number of threads for inference (default: number of cores)\n"
              << "  -c, --max_context_length N\n"
              << "                          max context length (default: 512)\n"
              << "  --extending EXT         context extending method (EXT = restart | shift) (default: restart)\n"
              << "  --multi                 enabled multiple lines of input\n"
              << "                          when enabled,  `" << MULTI_LINE_END_MARKER << "` marks the end of your input.\n"
              << "  --format FMT            conversion format (model specific, FMT = chat | completion | qa) (default: chat)\n"
              << "Sampling options:\n"
              << "  --sampling ALG          sampling algorithm (ALG = greedy | top_p | tfs) (default: top_p) \n"
              << "                          where, tfs = Tail Free Sampling\n"
              << "  -t, --temp T            temperature (default: 0.7) (Note: `-t 0` also sets sampling algorithm to greedy)\n"
              << "  --top_k N               top-k sampling (default: 0)\n"
              << "  --top_p N               top-p sampling (default: 0.7)\n"
              << "  --tfs_z Z               Z param for TFS (default: 0.95)\n"
              << "  --presence_penalty N    presence repetition penalty (default: 1.0, no penalty)\n"
              << "  --seed N                seed for random generator (default: random)\n"
              << "RAG options:\n"
              << "  --vector_store FILE     append a vector store file (when at lease one is specifed, RAG is enabled)\n"
              << "  --embedding_model PATH  embedding model path (mandatory if RAG is enabled)\n"
              << "  --distance_strategy DS  distance strategy (model dependent, default: MaxInnerProduct)\n"
              << "                          DS = EuclideanDistance | MaxInnerProduct | InnerProduct | CosineSimilarity\n"
              << "  --retrieve_top_n N      number of retrieved items using embedding model (default: 2)\n"
              << "  --retrieve_rewrite_template ...\n"
              << "                          prompt template to ask LLM to rewrite a query for retrieving (optional).\n"
              << "                          (default: \"\", i.e. disabled, the original prompt is used for retrieving)\n"
              << "                          macros: {question}. this may NOT WORK. Example:\n"
              << "                          Extract keywords for querying: {question}\n"
              << "  --reranker_model PATH   reranker model path (optional)\n"
              << "  --rerank_score_thres    reranking score threshold (default: 0.35)\n"
              << "                          items with a lower score are discarded.\n"
              << "  --rerank_top_n N        number of selected items using reranker model (default: 1)\n"
              << "   +rerank_rewrite        reranker use the rewritten query (default: OFF, i.e. use the original user input)\n"
              << "  --hide_reference        do not show references (default: false)\n"
              << "  --rag_template ...      prompt template for RAG (macros: {context}, {question}) (optional).\n"
              << "                          Support some C escape sequences (\\n). Example:\n"
              << "                          Answer the question according to below information:\n"
              << "                          ---\n"
              << "                          {context}\n"
              << "                          ---\n"
              << "                          Question: {question}\n"
              << "  --rag_context_sep       context separator (default: '\\n```\\n')\n"
              << "                          Support some C escape sequences (\\n).\n"
              << "  --rag_post_extending N  extend selected items with pre & post N chunks with same metadata. (default: 0)\n"
              << "                          this may be useful when context length of embedding/reranker models is limited.\n"
              << "   +rag_dump              (debug) dump retrieved/re-ranking results\n"
              << "Misc:\n"
              << "  --init_vs FILE          init vector store file from input\n"
              << "  --tokenize              (debug) tokenize `prompt` and exit\n"
              << "  --test FILE             test against inputs from a file and exit\n"
              << "  --hide_banner           hide banner\n"
              << "Additional key-value args:\n"
              << "  --kv                    start of additional args. all following options are interpreted as k-v pairs\n"
              << "  key value               a key-value pair of args\n"
              << std::endl;
}

static size_t parse_args(Args &args, const std::vector<std::string> &argv)
{
    std::random_device rd;
    args.seed = rd();
    const size_t argc = argv.size();

    #define handle_para0(fmt1, field, f)        \
        else if ((strcmp(arg, fmt1) == 0))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field = f(argv[c].c_str());                            \
        }

    #define handle_param(fmt1, fmt2, field, f)    \
        else if ((strcmp(arg, fmt1) == 0) || (strcmp(arg, fmt2) == 0))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field = f(argv[c].c_str());                            \
        }

    #define append_param(fmt1, field, f)        \
        else if ((strcmp(arg, fmt1) == 0))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field.push_back(f(argv[c].c_str()));                   \
        }

    size_t c = 1;
    while (c < argc)
    {
        const char *arg = argv[c].c_str();
        if ((strcmp(arg, "--help") == 0) || (strcmp(arg, "-h") == 0))
        {
            args.show_help = true;
        }
        else if ((strcmp(arg, "--interactive") == 0) || (strcmp(arg, "-i") == 0))
        {
            args.interactive = true;
        }
        else if (strcmp(arg, "--multi") == 0)
        {
            args.multi_line = true;
        }
        else if (strcmp(arg, "--tokenize") == 0)
        {
            args.tokenize = true;
        }
        else if (strcmp(arg, "--hide_reference") == 0)
        {
            args.hide_reference = true;
        }
        else if (strcmp(arg, "--hide_banner") == 0)
        {
            args.show_banner = false;
        }
        else if (strcmp(arg, "+rag_dump") == 0)
        {
            args.rag_dump = true;
        }
        else if (strcmp(arg, "+rerank_rewrite") == 0)
        {
            args.rerank_rewrite = true;
        }
        else if (strcmp(arg, "--format") == 0)
        {
            c++;
            if (c < argc)
            {
                if (argv[c] == "completion")
                    args.format = chatllm::ChatFormat::COMPLETION;
                else if (argv[c] == "qa")
                    args.format = chatllm::ChatFormat::QA;
                else
                    args.format = chatllm::ChatFormat::CHAT;
            }
        }
        else if (strcmp(arg, "--kv") == 0)
        {
            while (c + 2 < argc)
            {
                args.additional.insert_or_assign(argv[c + 1], argv[c + 2]);
                c += 2;
            }
        }
        handle_param("--model",                 "-m", model_path,           std::string)
        handle_param("--prompt",                "-p", prompt,               std::string)
        handle_param("--system",                "-s", system,               std::string)
        handle_param("--max_length",            "-l", max_length,           std::stoi)
        handle_param("--max_context_length",    "-c", max_context_length,   std::stoi)
        handle_para0("--extending",                   extending,            std::string)
        handle_para0("--sampling",                    sampling,             std::string)
        handle_param("--top_k",                 "-k", top_k,                std::stoi)
        handle_param("--top_p",                 "-q", top_p,                std::stof)
        handle_para0("--tfs_z",                       tfs_z,                std::stof)
        handle_param("--temp",                  "-t", temp,                 std::stof)
        handle_para0("--presence_penalty",            presence_penalty,     std::stof)
        handle_param("--threads",               "-n", num_threads,          std::stoi)
        handle_para0("--seed",                        seed,                 std::stoi)
        handle_para0("--test",                        test_fn,              std::string)
        append_param("--vector_store",                vector_store,         std::string)
        handle_para0("--embedding_model",             embedding_model_path, std::string)
        handle_para0("--distance_strategy",           vc,                   ParseDistanceStrategy)
        handle_para0("--retrieve_top_n",              retrieve_top_n,       std::stoi)
        handle_para0("--reranker_model",              reranker_model_path,  std::string)
        handle_para0("--retrieve_rewrite_template",   retrieve_rewrite_template,  std::string)
        handle_para0("--rerank_score_thres",          rerank_score_thres,   std::stof)
        handle_para0("--rerank_top_n",                rerank_top_n,         std::stoi)
        handle_para0("--rag_post_extending",          rag_post_extending,   std::stoi)
        handle_para0("--rag_template",                rag_template,         std::string)
        handle_para0("--rag_context_sep",             rag_context_sep,      std::string)
        handle_para0("--init_vs",                     vector_store_in,      std::string)
        else
            break;

        c++;
    }

#undef append_param

    return c;
}

#if defined(_WIN32)
static void append_utf8(char32_t ch, std::string &out)
{
    if (ch <= 0x7F)
    {
        out.push_back(static_cast<unsigned char>(ch));
    }
    else if (ch <= 0x7FF)
    {
        out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    }
    else if (ch <= 0xFFFF)
    {
        out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    }
    else if (ch <= 0x10FFFF)
    {
        out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    }
    else
    {
        // Invalid Unicode code point
    }
}

static bool get_utf8_line(std::string &line, bool multi_line)
{
    std::wstring marker(MULTI_LINE_END_MARKER_W);

    do
    {
        std::wstring prompt;
        std::getline(std::wcin, prompt);

        if (multi_line)
        {
            if (prompt == marker)
                return true;
            if (line.size() > 0)
                append_utf8('\n', line);
        }

        for (auto wc : prompt)
            append_utf8(wc, line);
    } while (multi_line);

    return true;
}
#else
static bool get_utf8_line(std::string &line, bool multi_line)
{
    do
    {
        std::string prompt;
        std::getline(std::cin, prompt);

        if (multi_line)
        {
            if (prompt == MULTI_LINE_END_MARKER)
                return true;
            if (line.size() > 0)
                line.push_back('\n');
        }

        line.append(prompt.begin(), prompt.end());
    } while (multi_line);

    return true;
}
#endif

static inline int get_num_physical_cores()
{
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

static void trim(std::string &s)
{
    size_t l = s.size();
    while (l > 0)
    {
        if ((s[l - 1] == '\r') || (s[l - 1] == '\n'))
            l--;
        else
            break;
    }
    s.resize(l);
}

// reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
class TextStreamer : public chatllm::BaseStreamer
{
public:
    TextStreamer(chatllm::BaseTokenizer *tokenizer) :
        tokenizer(tokenizer), is_prompt(true), print_len(0),
        cout(std::cout),
        reference_tag("Reference:"), ref_count(0) {}
    void put(const std::vector<int> &output_ids) override;
    void putln(const std::string &line) override;
    void put_reference(const std::string &line) override;
    void put_rewritten_query(const std::string &line) override;
    void end() override;

private:
    chatllm::BaseTokenizer *tokenizer;
    bool is_prompt;
    std::vector<int> token_cache;
    int print_len;
public:
    std::ostream &cout;
    std::string reference_tag;
    int ref_count;
};

static void run_file(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config)
{
    std::vector<std::string> history;
    std::string input;
    std::ifstream f(args.test_fn);

    if (f.is_open())
    {
        while (std::getline(f, input))
        {
            trim(input);
            streamer.cout << "You  > " << input << std::endl;
            history.emplace_back(std::move(input));

            streamer.cout << "A.I. > " << std::flush;
            std::string output = pipeline.chat(history, gen_config, &streamer);
            history.emplace_back(std::move(output));
        }
    }

    f.close();
    streamer.cout << std::endl << pipeline.model->get_n_past() << " tokens are processed/generated. Bye" << std::endl;
}

static void show_banner(chatllm::Pipeline &pipeline, bool show, chatllm::BaseStreamer *streamer)
{
    std::ostringstream oss;

    if (!show) return;
    if (pipeline.is_loaded())
    {
        #define MODEL_INFO()     "You are served by " << std::left << std::setw(28) << pipeline.model->type_name() + ","
        #define SHOW_NATIVE()    if (pipeline.model->native_name().size() > 0) { oss << "(" << pipeline.model->native_name() << ")"; }

        const int64_t total_param_num = pipeline.model->get_param_num(false);
        const int64_t total_effective_param_num = pipeline.model->get_param_num(true);

        oss     << R"(    ________          __  __    __    __  ___ )"; SHOW_NATIVE(); oss << '\n'
                << R"(   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____  )" << '\n'
                << R"(  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                << R"( / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                << R"( \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n';
        oss     << MODEL_INFO()                               << R"(/_/   /_/       )" << '\n';
        if (total_param_num == total_effective_param_num)
            oss    << "with " << total_param_num << " (" << std::fixed << std::setprecision(1) << (double)total_param_num / 1000000000. << "B) parameters." << '\n';
        else
            oss    << "with " << total_param_num << " (" << std::fixed << std::setprecision(1) << (double)total_effective_param_num / 1000000000. << "B effect.) parameters." << '\n';
    }
    else
    {
        oss     << R"(    ________          __  __    __    __  ___ )" << '\n'
                << R"(   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____  )" << '\n'
                << R"(  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                << R"( / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                << R"( \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n';
        oss     << R"(No LLM is loaded.                             /_/   /_/       )" << '\n';
    }

    auto additional = pipeline.get_additional_description();
    if (additional.size() > 0)
    {
        oss << additional << std::endl;
    }

    streamer->putln(oss.str());
}

static void print_embedding(const std::vector<float> &data, std::ostream &cout)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        if ((i % 8) == 0) cout << std::endl;
        cout << std::setw(14) << std::fixed << std::setprecision(8) << data[i] << "  ";
    }
    cout << std::endl;
}

static void run_text_embedding(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config)
{
    std::vector<float> result;

    if (!args.interactive)
    {
        pipeline.text_embedding(args.prompt, gen_config, result);
        print_embedding(result, streamer.cout);
        return;
    }

    while (1)
    {
        streamer.cout << "Input > " << std::flush;
        std::string input;
        if (!get_utf8_line(input, args.multi_line))
        {
            streamer.cout << "FAILED to read line." << std::endl;
            break;
        }
        if (input.empty()) continue;

        result.clear();
        pipeline.text_embedding(input, gen_config, result);
        streamer.cout << "      > ";

        print_embedding(result, streamer.cout);

    }
    streamer.cout << "Bye\n";
}

static void run_qa_ranker(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config)
{
    while (1)
    {
        streamer.cout << "Answer > " << std::flush;
        std::string answer;
        if (!get_utf8_line(answer, args.multi_line))
        {
            streamer.cout << "FAILED to read line." << std::endl;
            break;
        }
        if (answer.empty()) continue;

        float rank = pipeline.qa_rank(args.prompt, answer, gen_config);
        streamer.cout << std::setw(14) << std::fixed << std::setprecision(8) << rank << std::endl;
    }
    streamer.cout << "Bye\n";
}

#define DEF_GenerationConfig(gen_config, args) chatllm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,    \
                                         args.top_p, args.temp, args.num_threads, args.sampling, args.presence_penalty, args.tfs_z)

void chat(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer)
{
    if (args.system.size() > 0)
        pipeline.set_system_prompt(args.system);

    if (pipeline.is_loaded())
    {
        pipeline.model->seed(args.seed);
        args.max_length = pipeline.model->get_max_length();

        if (args.extending == "shift")
            pipeline.set_extending_method(chatllm::Pipeline::ExtendingMethod::Shift);
        else
            pipeline.set_extending_method(chatllm::Pipeline::ExtendingMethod::Restart);

        pipeline.tokenizer->set_chat_format(args.format);
    }

    if (args.tokenize)
    {
        auto ids = pipeline.tokenizer->encode(args.prompt);
        streamer.cout << "ID: ";
        for (auto x : ids)
            streamer.cout << x << ", ";
        streamer.cout << std::endl;
        return;
    }

    pipeline.set_additional_args(args.additional);

    const std::string ai_prompt   = "A.I.";
    const std::string user_prompt = "You";
    const int prompt_len = 4;

    DEF_GenerationConfig(gen_config, args);
    std::vector<std::string> history;

    show_banner(pipeline, args.interactive && args.show_banner, &streamer);

    if (pipeline.is_loaded())
    {
        switch (pipeline.model->get_purpose())
        {
        case chatllm::ModelPurpose::TextEmbedding:
            run_text_embedding(args, pipeline, streamer, gen_config);
            return;
        case chatllm::ModelPurpose::Ranker:
            run_qa_ranker(args, pipeline, streamer, gen_config);
            return;
        default:
            break;
        }
    }

    if (args.test_fn.size() > 0)
    {
        run_file(args, pipeline, streamer, gen_config);
        return;
    }

    if (!args.interactive)
    {
        history.push_back(args.prompt);
        pipeline.chat(history, gen_config, &streamer);
        return;
    }

    while (1)
    {
        streamer.cout << std::setw(prompt_len) << std::left << user_prompt << " > " << std::flush;
        std::string input;
        if (!get_utf8_line(input, args.multi_line))
        {
            streamer.cout << "FAILED to read line." << std::endl;
            break;
        }
        if (input.empty()) continue;

        history.emplace_back(std::move(input));
        streamer.cout << std::setw(prompt_len) << std::left << ai_prompt << " > " << std::flush;
        std::string output = pipeline.chat(history, gen_config, &streamer);
        history.emplace_back(std::move(output));
    }
    streamer.cout << "Bye\n";
}

void TextStreamer::putln(const std::string &line)
{
    cout << line << std::endl << std::flush;
}

void TextStreamer::put_reference(const std::string &line)
{
    if (ref_count == 0)
    {
        putln("");
        putln(reference_tag);
    }
    ref_count++;
    cout << ref_count << ". " << line << std::endl << std::flush;
}

void TextStreamer::put_rewritten_query(const std::string &line)
{
    cout << "Searching " << line << " ..." << std::endl << std::flush;
}

void TextStreamer::put(const std::vector<int> &output_ids)
{
    if (is_prompt)
        is_prompt = false;

    static const std::vector<char> puncts{',', '!', ':', ';', '?'};

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
        printable_text = text.substr(print_len);
        print_len = (int)text.size();
    }

    cout << printable_text << std::flush;
}

void TextStreamer::end()
{
    if (tokenizer)
    {
        std::string text = tokenizer->decode(token_cache);
        cout << text.substr(print_len) << std::endl;
    }
    is_prompt = true;
    token_cache.clear();
    print_len = 0;
    ref_count = 0;
}

#if defined(_WIN32)
std::string wstr_to_utf8(const wchar_t* wstr)
{
    int s = WideCharToMultiByte(CP_UTF8, 0, wstr, (int)wcslen(wstr), NULL, 0, NULL, NULL);
    std::string str;
    str.resize(s);
    WideCharToMultiByte(CP_UTF8, 0, wstr, (int)wcslen(wstr), LPSTR(str.data()), s, NULL, NULL);
    return str;
}
#endif

#ifndef CHATLLM_SHARED_LIB

static int init_vector_store(Args &args)
{
    chatllm::Pipeline pipeline(args.embedding_model_path);
    args.max_length = pipeline.model->get_max_length();

    DEF_GenerationConfig(gen_config, args);
    std::vector<float> r;

    CVectorStore vs(args.vc, pipeline.get_text_embedding_dim(),
        [&pipeline, &gen_config, &r](const std::string &s, float *emb)
        {
            pipeline.text_embedding(s, gen_config, r);
            CHATLLM_CHECK((int)r.size() == pipeline.get_text_embedding_dim()) << "embedding dim mismatch";
            memcpy(emb, r.data(), r.size() * sizeof(float));
        },
        args.vector_store_in.c_str());
    vs.ExportDB((args.vector_store_in + ".vsdb").c_str());
    printf("Vector store saved to: %s\n", (args.vector_store_in + ".vsdb").c_str());
    return 0;
}

#if defined(_WIN32)
int wmain(int argc, const wchar_t **wargv)
{
    std::vector<std::string> utf_args;
    for (int i = 0; i < argc; i++)
        utf_args.push_back(wstr_to_utf8(wargv[i]));

    _setmode(_fileno(stdin), _O_WTEXT);
    // Set console code page to UTF-8 so console known how to interpret string data
    SetConsoleOutputCP(CP_UTF8);
    // Enable buffering to prevent VS from chopping up UTF-8 byte sequences
    setvbuf(stdout, nullptr, _IOFBF, 1000);

#else
int main(int argc, const char **argv)
{
    std::vector<std::string> utf_args;
    for (int i = 0; i < argc; i++)
        utf_args.push_back(argv[i]);
#endif

    Args args;
    auto count = parse_args(args, utf_args);
    if (args.show_help)
    {
        usage(utf_args[0]);
        return 0;
    }

    if (count < utf_args.size())
    {
        std::cerr << "Unknown arguments:";
        for (auto i = count; i < utf_args.size(); i++)
        {
            std::cerr << " " << utf_args[i];
        }
        std::cerr << std::endl;

        exit(EXIT_FAILURE);
    }

    if (args.num_threads <= 0)
        args.num_threads = get_num_physical_cores();

    if (args.vector_store_in.size() > 0)
        return init_vector_store(args);

    try
    {
        chatllm::ModelObject::extra_args pipe_args(args.max_length);
        if (args.embedding_model_path.size() < 1)
        {
            chatllm::Pipeline pipeline(args.model_path, pipe_args);
            TextStreamer streamer(pipeline.tokenizer);
            chat(args, pipeline, streamer);
        }
        else
        {
            chatllm::RAGPipeline pipeline(args.model_path, pipe_args,
                args.vc, args.vector_store,
                args.embedding_model_path, args.reranker_model_path);
            pipeline.hide_reference = args.hide_reference;
            pipeline.retrieve_top_n = args.retrieve_top_n;
            pipeline.rerank_top_n   = args.rerank_top_n;
            pipeline.dump           = args.rag_dump;
            pipeline.rerank_score_threshold = args.rerank_score_thres;
            pipeline.rag_post_extending     = args.rag_post_extending;
            pipeline.rerank_rewrite         = args.rerank_rewrite;
            pipeline.composer.set_context_sep(args.rag_context_sep);
            pipeline.composer.set_prompt_template(args.rag_template);
            pipeline.composer.set_rewrite_template(args.retrieve_rewrite_template);
            TextStreamer streamer(pipeline.tokenizer);
            chat(args, pipeline, streamer);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

#else // CHATLLM_SHARED_LIB

#ifdef _WIN32
    #define DLL_DECL __declspec(dllexport)
#elif __GNUC__ >= 4
    #define DLL_DECL __attribute__((visibility("default")))
#else
    #define DLL_DECL
#endif

#include "bindings/libchatllm.h"

class Chat
{
public:
    Chat():
        streamer(nullptr), pipeline(nullptr),
        f_print_ref(nullptr),
        f_print_query(nullptr)
    {
        append_param("...");
    }

    void append_param(const char *utf8_str)
    {
        params.push_back(utf8_str);
    }

public:
    std::vector<std::string> params;
    std::vector<std::string> history;
    chatllm::BaseStreamer *streamer;
    chatllm::Pipeline *pipeline;
    chatllm::GenerationConfig gen_config;
    f_chatllm_print f_print_ref;
    f_chatllm_print f_print_query;
};

class FFIStreamer : public chatllm::BaseStreamer
{
public:
    FFIStreamer(chatllm::BaseTokenizer *tokenizer,
        f_chatllm_print f_print,
        f_chatllm_print f_print_ref,
        f_chatllm_print f_print_query,
        f_chatllm_end f_end, void *user_data) :
        tokenizer(tokenizer), is_prompt(true), print_len(0),
        f_print(f_print), f_print_ref(f_print_ref), f_print_query(f_print_query), f_end(f_end), user_data(user_data),
        ref_count(0)
    {
    }

    void put(const std::vector<int> &output_ids) override
    {
        if (is_prompt)
            is_prompt = false;

        static const std::vector<char> puncts{',', '!', ':', ';', '?'};

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
            f_print(user_data, printable_text.c_str());
    }

    void putln(const std::string &line) override
    {
        f_print(user_data, line.c_str());
        f_print(user_data, "\n");
    }

    void put_reference(const std::string &line) override
    {
        if (f_print_ref)
        {
            f_print_ref(user_data, line.c_str());
        }
        else
        {
            std::ostringstream oss;
            ref_count++;
            if (1 == ref_count)
            {
                oss << "\nReference:\n";
            }
            oss << ref_count << ". " << line;
            putln(oss.str());
        }
    }

    void put_rewritten_query(const std::string &line) override
    {
        if (f_print_query)
        {
            f_print_query(user_data, line.c_str());
        }
    }

    void end() override
    {
        if (tokenizer)
        {
            std::string text = tokenizer->decode(token_cache);
            putln(text.substr(print_len));
        }
        is_prompt = true;
        token_cache.clear();
        print_len = 0;
        f_end(user_data);
        ref_count = 0;
    }

public:
    chatllm::BaseTokenizer *tokenizer;
    bool is_prompt;
    std::vector<int> token_cache;
    size_t print_len;
    f_chatllm_print f_print;
    f_chatllm_print f_print_ref;
    f_chatllm_print f_print_query;
    f_chatllm_end f_end;
    void *user_data;
    int ref_count;
};

struct chatllm_obj *chatllm_create(void)
{
    return (chatllm_obj *)new Chat();
}

void chatllm_append_param(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    chat->append_param(utf8_str);
}

static int start_chat(Chat *chat, Args &args, chatllm::Pipeline &pipeline, chatllm::BaseStreamer &streamer)
{
    chat->pipeline = &pipeline;
    chat->streamer = &streamer;

    if (args.system.size() > 0)
        pipeline.set_system_prompt(args.system);

    if (pipeline.is_loaded())
    {
        pipeline.model->seed(args.seed);
        args.max_length = pipeline.model->get_max_length();

        if (args.extending == "shift")
            pipeline.set_extending_method(chatllm::Pipeline::ExtendingMethod::Shift);
        else
            pipeline.set_extending_method(chatllm::Pipeline::ExtendingMethod::Restart);

        pipeline.tokenizer->set_chat_format(args.format);
    }

    pipeline.set_additional_args(args.additional);

    DEF_GenerationConfig(gen_config, args);

    chat->gen_config = gen_config;

    show_banner(pipeline, args.interactive && args.show_banner, &streamer);

    return 0;
}

int chatllm_set_print_reference(struct chatllm_obj *obj, f_chatllm_print f_print)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    chat->f_print_ref = f_print;
    return 0;
}

int chatllm_set_print_rewritten_query(struct chatllm_obj *obj, f_chatllm_print f_print)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    chat->f_print_query = f_print;
    return 0;
}

int chatllm_start(struct chatllm_obj *obj, f_chatllm_print f_print, f_chatllm_end f_end, void *user_data)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);

    Args args;
    auto count = parse_args(args, chat->params);

    if (count < chat->params.size())
        return (int)count;

    if (args.num_threads <= 0)
        args.num_threads = get_num_physical_cores();

    args.interactive = true;

    try
    {
        chatllm::ModelObject::extra_args pipe_args(args.max_length);

        if (args.embedding_model_path.size() < 1)
        {
            if (args.model_path.size() < 1)
                return -1;

            auto pipeline = new chatllm::Pipeline(args.model_path, pipe_args);
            auto streamer = new FFIStreamer(pipeline->tokenizer, f_print, nullptr, nullptr, f_end, user_data);
            return start_chat(chat, args, *pipeline, *streamer);
        }
        else
        {
            auto pipeline = new chatllm::RAGPipeline(args.model_path, pipe_args,
                args.vc, args.vector_store,
                args.embedding_model_path, args.reranker_model_path);
            pipeline->hide_reference = args.hide_reference;
            pipeline->retrieve_top_n = args.retrieve_top_n;
            pipeline->rerank_top_n   = args.rerank_top_n;
            pipeline->dump           = args.rag_dump;
            pipeline->rerank_score_threshold = args.rerank_score_thres;
            pipeline->rag_post_extending     = args.rag_post_extending;
            pipeline->rerank_rewrite         = args.rerank_rewrite;
            pipeline->composer.set_context_sep(args.rag_context_sep);
            pipeline->composer.set_prompt_template(args.rag_template);
            pipeline->composer.set_rewrite_template(args.retrieve_rewrite_template);
            auto streamer = new FFIStreamer(pipeline->tokenizer, f_print, chat->f_print_ref, chat->f_print_query, f_end, user_data);
            return start_chat(chat, args, *pipeline, *streamer);
        }

    }
    catch (std::exception &e)
    {
        f_print(user_data, e.what());
        return EXIT_FAILURE;
    }
}

int chatllm_user_input(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (!streamer->is_prompt) return -1;

    chat->history.push_back(utf8_str);
    std::string output = chat->pipeline->chat(chat->history, chat->gen_config, streamer);
    chat->history.emplace_back(std::move(output));
    return 0;
}

void chatllm_abort_generation(struct chatllm_obj *obj)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    if (chat->pipeline)
        chat->pipeline->abort_generation();
}

#endif