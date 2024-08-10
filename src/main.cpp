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
    std::string merge_vs = "";
    std::string system = "";
    std::string prompt = "你好";
    std::string sampling = "top_p";
    chatllm::Pipeline::ExtendingMethod extending = chatllm::Pipeline::ExtendingMethod::Restart;
    std::string test_fn = "";
    std::string rag_template = "";
    std::string rag_context_sep = "";
    std::string retrieve_rewrite_template = "";
    std::map<std::string, std::string> additional;
    std::string layer_spec;
    std::string load_session;
    std::string save_session;
    std::string n_gpu_layers;
    int max_length = -1;
    int max_context_length = 512;
    bool interactive = false;
    bool show = false;
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
    int save_session_rounds = -1;
};

#define MULTI_LINE_END_MARKER_W  L"\\."
#define MULTI_LINE_END_MARKER     "\\."

bool has_extending = false;

static chatllm::Pipeline::ExtendingMethod parse_extending_method(const std::string &s)
{
    has_extending = true;
    if (s == "shift")
        return chatllm::Pipeline::ExtendingMethod::Shift;
    else if (s == "restart")
        return chatllm::Pipeline::ExtendingMethod::Restart;
    else
        return chatllm::Pipeline::ExtendingMethod::None;
}

void usage(const std::string &prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Basic options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        model path\n"
              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
              << "      --prompt_file FN    prompt from file\n"
              << "  -s, --system SYSTEM     system prompt (instruction) (default: model specific)\n"
              << "      --sys_file FN       system prompt (instruction) from file\n"
              << "  -i, --interactive       run in interactive mode\n"
              << "  -l, --max_length N      max total length including prompt and output (default: model specific)\n"
              << "                          generally, this is used to reduce KV cache size.\n"
              << "                          for models that does not show its max context window in `config.json`,\n"
              << "                          use this to enlarge it (use with caution!).\n"
              << "  --layer_spec LAYERS     select/redesign layers.\n"
              << "                          LAYERS=S0,S1,.... where S0/S1/... are like slices of Python, `start:stop[:step]`,\n"
              << "                          negative values in `start` and `stop` can be used referencing layers in reversed order,\n"
              << "                          `step` is optional, e.g.\n"
              << "                            --layer_spec 0:3,1:4 (3 + 3 = 6 layers are selected, layer #1/2 are used twice)\n"
              << "                                                 layer structure: 0->1->2->1->2->3\n"
              << "  -c, --max_context_length N\n"
              << "                          max context length (default: 512)\n"
              << "  --extending EXT         context extending method (EXT = restart | shift | none)\n"
              << "                          (default: none if `--load_session` is specified, otherwise restart)\n"
              << "  --multi                 enabled multiple lines of input\n"
              << "                          when enabled,  `" << MULTI_LINE_END_MARKER << "` marks the end of your input.\n"
              << "  --format FMT            conversion format (model specific, FMT = chat | completion | qa) (default: chat)\n"
              << "Performance options:\n"
              << "  -n, --threads N         number of threads for inference (default: number of cores)\n"
              << "  -ngl, --n_gpu_layers N  number of model layers to offload to each GPU (default: GPU not used)\n"
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
              << "Session:\n"
              << "  --save_session N FILE   save session to FILE after N round(s) of chatting (N >= 0) and quit\n"
              << "                          when N = 0, system prompt is evaluated.\n"
              << "  --load_session FILE     load session from FILE\n"
              << "Misc:\n"
              << "  --init_vs FILE          init vector store file from input\n"
              << "  --merge_vs FILE         merge multiple vector store files into a single one\n"
              << "  --tokenize              (debug) tokenize `prompt` and exit\n"
              << "  --test FILE             test against inputs from a file and exit\n"
              << "  --hide_banner           hide banner\n"
              << "  --show                  show model info and quit\n"
              << "Additional key-value args:\n"
              << "  --kv                    start of additional args. all following options are interpreted as k-v pairs\n"
              << "  key value               a key-value pair of args\n"
              << std::endl;
}

static std::string load_txt(const std::string &fn)
{
    std::ifstream f(fn);
    std::ostringstream sstr;

    if (f.is_open())
    {
        sstr << f.rdbuf();
        f.close();
    }
    return sstr.str();
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

    try
    {
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
            else if (strcmp(arg, "--show") == 0)
            {
                args.show = true;
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
            else if (strcmp(arg, "--save_session") == 0)
            {
                c++;
                if (c + 1 < argc)
                {
                    args.save_session_rounds = std::stoi(argv[c]);
                    args.save_session        = argv[c + 1];
                    c++;
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
            handle_para0("--prompt_file",                 prompt,               load_txt)
            handle_param("--system",                "-s", system,               std::string)
            handle_para0("--sys_file",                    system,               load_txt)
            handle_param("--max_length",            "-l", max_length,           std::stoi)
            handle_param("--max_context_length",    "-c", max_context_length,   std::stoi)
            handle_para0("--extending",                   extending,            parse_extending_method)
            handle_para0("--sampling",                    sampling,             std::string)
            handle_param("--top_k",                 "-k", top_k,                std::stoi)
            handle_param("--top_p",                 "-q", top_p,                std::stof)
            handle_para0("--tfs_z",                       tfs_z,                std::stof)
            handle_param("--temp",                  "-t", temp,                 std::stof)
            handle_para0("--presence_penalty",            presence_penalty,     std::stof)
            handle_param("--threads",               "-n", num_threads,          std::stoi)
            handle_param("--n_gpu_layers",          "-ngl", n_gpu_layers,       std::string)
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
            handle_para0("--merge_vs",                    merge_vs,             std::string)
            handle_para0("--layer_spec",                  layer_spec,           std::string)
            handle_para0("--load_session",                load_session,         std::string)
            else
                break;

            c++;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return c;
    }

#undef append_param

    if (!has_extending && (args.load_session.size() > 0))
        args.extending = chatllm::Pipeline::ExtendingMethod::None;

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
        BaseStreamer(tokenizer),
        cout(std::cout),
        reference_tag("Reference:"), ref_count(0) {}
    void put_chunk(bool first, const std::string &chunk) override;
    void putln(const std::string &line, TextType type = TextType::META) override;
    void end() override;

public:
    std::ostream &cout;
    std::string reference_tag;
    int ref_count;
};

static void print_timing(char *str, const char *prefix, size_t tok_number, double duration_sec)
{
    sprintf(str, "%s = %12.2f ms / %5zd tokens ( %8.2f ms per token, %8.2f tokens per second)", prefix, duration_sec, tok_number,
            duration_sec / tok_number,
            tok_number / duration_sec * 1000);
}

static void show_stat(chatllm::Pipeline &pipeline, chatllm::BaseStreamer &streamer)
{
    streamer.putln("");

    chatllm::ModelPerfInfo *perf = &pipeline.performance;
    char str[1024];
    print_timing(str, "timings: prompt eval time", perf->timings[chatllm::ModelPerfInfo::Type::Prompt].tok_count, perf->timings[chatllm::ModelPerfInfo::Type::Prompt].duration_ms);
    streamer.putln(str);

    print_timing(str, "timings:        eval time", perf->timings[chatllm::ModelPerfInfo::Type::Generation].tok_count, perf->timings[chatllm::ModelPerfInfo::Type::Generation].duration_ms);
    streamer.putln(str);

    sprintf(str,      "timings:       total time = %12.2f ms / %5zd tokens",
        (perf->timings[chatllm::ModelPerfInfo::Type::Generation].duration_ms + perf->timings[chatllm::ModelPerfInfo::Type::Prompt].duration_ms),
        perf->timings[chatllm::ModelPerfInfo::Type::Generation].tok_count    + perf->timings[chatllm::ModelPerfInfo::Type::Prompt].tok_count);
    streamer.putln(str);
}

static void run_file(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config)
{
    chatllm::Messages history;
    std::string input;
    std::ifstream f(args.test_fn);

    if (f.is_open())
    {
        while (std::getline(f, input))
        {
            trim(input);
            streamer.cout << "You  > " << input << std::endl;
            history.push_back(input, chatllm::MsgRole::User);

            streamer.cout << "A.I. > " << std::flush;
            std::string output = pipeline.chat(history, gen_config, &streamer);
            history.push_back(output, chatllm::MsgRole::Assistant);
        }
    }

    f.close();
    streamer.cout << std::endl << pipeline.model->get_n_past() << " tokens are processed/generated. Bye" << std::endl;

    show_stat(pipeline, streamer);
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
                                         args.top_p, args.temp, args.num_threads, args.sampling, args.presence_penalty, args.tfs_z, args.n_gpu_layers)

void chat(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer)
{
    if (args.system.size() > 0)
        pipeline.set_system_prompt(args.system);

    if (pipeline.is_loaded())
    {
        pipeline.model->seed(args.seed);
        args.max_length = pipeline.model->get_max_length();

        pipeline.set_extending_method(args.extending);

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
    const std::string user_prompt = "You ";

    auto show_msg_role = [&](chatllm::MsgRole role) -> std::string
    {
        switch (role)
        {
        case chatllm::MsgRole::Assistant:
            return ai_prompt;
        case chatllm::MsgRole::User:
            return user_prompt;
        case chatllm::MsgRole::Tool:
            return "Tool";
        default:
            return "????";
        }
    };

    DEF_GenerationConfig(gen_config, args);
    chatllm::Messages history;

    show_banner(pipeline, args.interactive && args.show_banner, &streamer);

    pipeline.prepare(gen_config);

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
        history.push_back(args.prompt, chatllm::MsgRole::User);
        pipeline.chat(history, gen_config, &streamer);
        show_stat(pipeline, streamer);
        return;
    }

    if (0 == args.save_session_rounds)
    {
        std::cout << std::endl << "evaluating system prompt... ";
        pipeline.eval_sys_prompt(gen_config);
        std::cout << "saving session..." << std::endl;
        pipeline.save_session(history, args.save_session);
        return;
    }

    if (args.load_session.size() > 0)
    {
        CHATLLM_CHECK(pipeline.load_session(history, args.load_session, nullptr) == 0) << "failed to load session file";

        for (int i = 0; i < (int)history.size(); i++)
        {
            auto &m = history[i];
            streamer.cout << show_msg_role(m.role)
                          << " > "
                          << m.content << std::endl << std::flush;
        }
        if (history.size() > 0)
        {
            auto &last = history[history.size() - 1];
            if (last.role != chatllm::MsgRole::Assistant)
            {
                std::string output = pipeline.chat(history, gen_config, &streamer);
                history.push_back(output, chatllm::MsgRole::Assistant);
            }
        }
    }

    while (1)
    {
        if ((args.save_session_rounds > 0) && ((int)(history.size() / 2) == args.save_session_rounds))
        {
            std::cout << std::endl << "saving session..." << std::endl;
            pipeline.save_session(history, args.save_session);
            break;
        }

        streamer.cout << user_prompt << " > " << std::flush;
        std::string input;
        if (!get_utf8_line(input, args.multi_line))
        {
            streamer.cout << "FAILED to read line." << std::endl;
            break;
        }
        if (input.empty()) continue;

        history.push_back(input, chatllm::MsgRole::User);
        streamer.cout << ai_prompt << " > " << std::flush;
        std::string output = pipeline.chat(history, gen_config, &streamer);
        history.push_back(output, chatllm::MsgRole::Assistant);
    }
    streamer.cout << "Bye\n";
}

void TextStreamer::putln(const std::string &line, TextType type)
{
    switch (type)
    {
    case TextType::ERR:
        cout << "ERROR: " << line << std::endl << std::flush;
        break;
    case TextType::REF:
        if (ref_count == 0)
        {
            putln("");
            putln(reference_tag);
        }
        ref_count++;
        cout << ref_count << ". " << line << std::endl << std::flush;
        break;
    case TextType::REWRITTEN_QUERY:
        cout << "Searching " << line << " ..." << std::endl << std::flush;
        break;
    case TextType::HISTORY_USER:
    case TextType::HISTORY_AI:
        break;
    case  TextType::TOOL_CALLING:
        cout << " <TOOL_CALLING> Run this tool and tell AI the result: " << line << std::endl << std::flush;
        break;
    default:
        cout << line << std::endl << std::flush;
        break;
    }
}

void TextStreamer::put_chunk(bool first, const std::string &chunk)
{
    cout << chunk << std::flush;
}

void TextStreamer::end()
{
    BaseStreamer::end();
    ref_count = 0;
    cout << std::endl;
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

    pipeline.prepare(gen_config);

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

static int merge_vector_store(Args &args)
{
    CVectorStore vs(args.vc, args.vector_store);
    vs.ExportDB(args.merge_vs.c_str());
    printf("Vector store saved to: %s\n", args.merge_vs.c_str());
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

    if (args.show)
    {
        chatllm::ModelLoader loader(args.model_path);
        std::cout << chatllm::ModelFactory::load_info(loader) << std::endl;
        return 0;
    }

    if (args.vector_store_in.size() > 0)
        return init_vector_store(args);

    if (args.merge_vs.size() > 0)
        return merge_vector_store(args);

    try
    {
        chatllm::ModelObject::extra_args pipe_args(args.max_length, args.layer_spec);
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

#include "../bindings/libchatllm.h"

class Chat
{
public:
    Chat():
        streamer(nullptr), pipeline(nullptr),
        sess_n_past(-1), sess_hist_len(-1)
    {
        append_param("...");
    }

    void append_param(const char *utf8_str)
    {
        params.push_back(utf8_str);
    }

public:
    std::vector<std::string> params;
    chatllm::Messages history;
    chatllm::BaseStreamer *streamer;
    chatllm::Pipeline *pipeline;
    chatllm::GenerationConfig gen_config;
    int sess_n_past;
    int sess_hist_len;
    Args args;
    std::string tool_input;
    std::string tool_completion;    // part of the output is generated by external tools
};

class FFIStreamer : public chatllm::BaseStreamer
{
public:
    FFIStreamer(chatllm::BaseTokenizer *tokenizer,
        f_chatllm_print f_print,
        f_chatllm_end f_end, void *user_data) :
        chatllm::BaseStreamer(tokenizer),
        f_print(f_print), f_end(f_end), user_data(user_data),
        ref_count(0)
    {
    }

    void put_chunk(bool is_first, const std::string &chunk) override
    {
        f_print(user_data, PRINT_CHAT_CHUNK, chunk.c_str());
    }

    void putln(const std::string &line, TextType type = TextType::META) override
    {
        f_print(user_data, (int)type, line.c_str());
    }

    void end() override
    {
        f_end(user_data);
        ref_count = 0;
        chatllm::BaseStreamer::end();
    }

public:
    f_chatllm_print f_print;
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
    int r = 0;
    chat->pipeline = &pipeline;
    chat->streamer = &streamer;

    if (args.system.size() > 0)
        pipeline.set_system_prompt(args.system);

    if (pipeline.is_loaded())
    {
        pipeline.model->seed(args.seed);
        args.max_length = pipeline.model->get_max_length();

        pipeline.set_extending_method(args.extending);

        pipeline.tokenizer->set_chat_format(args.format);
    }

    pipeline.set_additional_args(args.additional);

    DEF_GenerationConfig(gen_config, args);
    pipeline.prepare(gen_config);

    chat->gen_config = gen_config;

    show_banner(pipeline, args.interactive && args.show_banner, &streamer);

    if (args.load_session.size() > 0)
    {
        r = chatllm_load_session(reinterpret_cast<struct chatllm_obj *>(chat), args.load_session.c_str());
        if (r) return r;
    }

    if (args.save_session_rounds == 0)
    {
        r = chatllm_save_session(reinterpret_cast<struct chatllm_obj *>(chat), args.save_session.c_str());
    }

    return r;
}

int chatllm_start(struct chatllm_obj *obj, f_chatllm_print f_print, f_chatllm_end f_end, void *user_data)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    Args &args = chat->args;

    auto count = parse_args(args, chat->params);

    if (count < chat->params.size())
        return (int)count;

    if (args.num_threads <= 0)
        args.num_threads = get_num_physical_cores();

    args.interactive = true;

    try
    {
        chatllm::ModelObject::extra_args pipe_args(args.max_length, args.layer_spec);

        if (args.embedding_model_path.size() < 1)
        {
            if (args.model_path.size() < 1)
                return -1;

            auto pipeline = new chatllm::Pipeline(args.model_path, pipe_args);
            auto streamer = new FFIStreamer(pipeline->tokenizer, f_print, f_end, user_data);
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
            auto streamer = new FFIStreamer(pipeline->tokenizer, f_print, f_end, user_data);
            return start_chat(chat, args, *pipeline, *streamer);
        }

    }
    catch (std::exception &e)
    {
        f_print(user_data, PRINTLN_ERROR, e.what());
        return EXIT_FAILURE;
    }
}

static void chatllm_continue_chat(Chat *chat)
{
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    std::string external_ai = chat->tool_completion;

    if (external_ai.size() < 1) return;

    chat->tool_completion.clear();
    if (chat->history.size() < 1) return;

    int last_id = (int)chat->history.size() - 1;

    chat->history[last_id].content = chat->history[last_id].content + external_ai;

    std::string output = chat->pipeline->chat(chat->history, chat->gen_config, streamer);

    chat->history[last_id].content = chat->history[last_id].content + output;
}

int chatllm_user_input(struct chatllm_obj *obj, const char *utf8_str)
{
    int r = 0;
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (!streamer->is_prompt) return -1;

    if (chat->pipeline->is_loaded() && (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::Chat))
        return -1;

    chat->history.push_back(utf8_str, chatllm::MsgRole::User);

generate:
    std::string output = chat->pipeline->chat(chat->history, chat->gen_config, streamer);
    chat->history.push_back(output, chatllm::MsgRole::Assistant);

    if (chat->tool_completion.size() > 0)
        chatllm_continue_chat(chat);

    if ((chat->args.save_session_rounds > 0) && (chat->history.size() / 2 == (size_t)chat->args.save_session_rounds))
    {
        streamer->putln("saving session...", chatllm::BaseStreamer::TextType::META);
        r = chatllm_save_session(obj, chat->args.save_session.c_str());
    }

    if (chat->tool_input.size() > 0)
    {
        chat->history.push_back(chat->tool_input.c_str(), chatllm::MsgRole::Tool);
        chat->tool_input.clear();
        goto generate;
    }

    return r;
}

int chatllm_tool_input(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    chat->tool_input = std::string(utf8_str);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (!streamer->is_prompt)
        chatllm_continue_chat(chat);
    return 0;
}

int chatllm_tool_completion(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (streamer->is_prompt)
        return chatllm_user_input(obj, utf8_str);

    chat->tool_input = std::string(utf8_str);
    return 0;
}

int chatllm_text_embedding(struct chatllm_obj *obj, const char *utf8_str)
{
    int r = 0;
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);

    if (!chat->pipeline->is_loaded() || (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::TextEmbedding))
        return -1;

    std::vector<float> result;
    std::string input(utf8_str);
    chat->pipeline->text_embedding(input, chat->gen_config, result);

    std::ostringstream oss;
    for (size_t i = 0; i < result.size() - 1; i++)
    {
        if ((i > 0) && ((i % 8) == 0)) oss << std::endl;
        oss << std::setw(14) << std::fixed << std::setprecision(8) << result[i] << ",";
    }
    oss << std::setw(14) << std::fixed << std::setprecision(8) << result.back();

    streamer->putln(oss.str(), chatllm::BaseStreamer::TextType::EMBEDDING);

    return r;
}

int chatllm_qa_rank(struct chatllm_obj *obj, const char *utf8_str_q, const char *utf8_str_a)
{
    int r = 0;
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);

    if (!chat->pipeline->is_loaded() || (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::Ranker))
        return -1;

    std::string q(utf8_str_q);
    std::string a(utf8_str_a);
    float result = chat->pipeline->qa_rank(q, a, chat->gen_config);

    std::ostringstream oss;
    oss << std::setw(14) << std::fixed << std::setprecision(8) << result;
    streamer->putln(oss.str(), chatllm::BaseStreamer::TextType::RANKING);

    return r;
}

void chatllm_restart(struct chatllm_obj *obj)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (!streamer->is_prompt) return;

    if (chat->sess_hist_len > 0)
    {
        if (chat->history.size() > (size_t)chat->sess_hist_len)
            chat->history.history.erase(chat->history.history.begin() + chat->sess_hist_len, chat->history.history.end());
        chat->pipeline->rewind(chat->sess_n_past);
    }
    else
    {
        chat->history.clear();
        chat->pipeline->restart();
    }
}

void chatllm_abort_generation(struct chatllm_obj *obj)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    if (chat->pipeline)
        chat->pipeline->abort_generation();
}

void chatllm_set_gen_max_tokens(struct chatllm_obj *obj, int gen_max_tokens)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    chat->pipeline->gen_max_tokens = gen_max_tokens;
}

void chatllm_show_statistics(struct chatllm_obj *obj)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    show_stat(*(chat->pipeline), *(chat->streamer));
}

int chatllm_save_session(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (!streamer->is_prompt) return -1;

    streamer->putln("saving session ...", chatllm::BaseStreamer::TextType::META);

    if (chat->history.size() < 1)
    {
        chat->pipeline->eval_sys_prompt(chat->gen_config);
    }
    return chat->pipeline->save_session(chat->history, utf8_str);
}

int  chatllm_load_session(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer);
    if (!streamer->is_prompt) return -1;

    int r = chat->pipeline->load_session(chat->history, utf8_str, streamer, &chat->sess_n_past);
    if (0 == r)
        chat->sess_hist_len = (int)chat->history.size();
    return r;
}

#endif