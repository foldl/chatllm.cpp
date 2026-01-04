#include "chat.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <thread>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstring>
#include <climits>
#include <random>
#include <thread>
#include <map>
#include <filesystem>

#include "vectorstore.h"
#include "vision_process.h"
#include "audio_process.h"
#include "models.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

static chatllm::ThoughtChunkInterceptor thought_interceptor;

struct Args
{
    std::string model_path = "";
    std::string embedding_model_path = "";
    std::string reranker_model_path = "";
    std::string vector_store_in = "";
    std::string merge_vs = "";
    std::string system = "";
    std::string prompt = "你好";
    std::string ai_prefix = "";
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
    std::string cur_vs_name = "default";
    std::string dump_dot;
    std::string emb_rank_query_sep;
    std::map<std::string, std::vector<std::string>> vector_stores;
    std::string rpc_endpoints;
    std::string serve_rpc;
    std::string ggml_dir;
    std::string cache_dtype = "f16";
    std::string thought_tags[2] = {"", ""};
    std::string multimedia_file_tags[2] = {"", ""};
    std::string tts_export;
    std::string re_quantize;
    std::map<std::string, std::string> model_n_gpu_layers;
    int max_length = -1;
    int max_context_length = 512;
    bool interactive = false;
    bool show = false;
    int top_k = 20;
    float top_p = 0.7f;
    float temp = 0.7f;
    float tfs_z = 0.95f;
    float presence_penalty = 0.0f;
    float repeat_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    int num_threads = 0;
    bool multi_line = false;
    int seed = -1;
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
    bool show_devices = false;
    bool rerank_rewrite = false;
    bool reversed_role = false;
    int save_session_rounds = -1;
    int beam_size = -1;
    int log_level = 4;
    bool moe_on_cpu = false;
    int batch_size = 4096;
    bool detect_thoughts = false;
    int penalty_window = 256;
    int max_new_tokens = -1;
    bool single_turn = false;
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

const std::vector<std::pair<std::string, std::string>> THOUGHT_TAGS = {
    {"<think>",         "</think>"},
    {"◁think▷",         "◁/think▷"},
    {"<thought>",       "</thought>"},
    {"<reasoning>",     "</reasoning>"},
};

static std::string show_default_thought_tags(void)
{
    std::string tags;
    for (int i = 0; i < (int)(sizeof(THOUGHT_TAGS) / sizeof(THOUGHT_TAGS[0])); i++)
    {
        if (i > 0)
            tags += ", ";
        tags += THOUGHT_TAGS[i].first;
        tags += " ";
        tags += THOUGHT_TAGS[i].second;
    }
    return tags;
}

void usage(const std::string &prog)
{
    Args args;
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Basic options:\n"
              << "  -h, --help              show this help message and exit                                                         [*]\n"
              << "  -m, --model PATH        model path\n"
              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
              << "      --prompt_file FN    prompt from file\n"
              << "  -s, --system SYSTEM     system prompt (instruction) (default: model specific)\n"
              << "      --sys_file FN       system prompt (instruction) from file\n"
              << "      --ai_prefix         AI prefix for generation (default: empty)\n"
              << "  -i, --interactive       run in interactive mode                                                                 [*]\n"
              << "      --reversed_role     AI becomes `user`, user becomes `AI`                                                    [#]\n"
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
              << "                          max context length (default: " << args.max_context_length << ")\n"
              << "  --extending EXT         context extending method (EXT = restart | shift | none)\n"
              << "                          (default: none if `--load_session` is specified, otherwise restart)\n"
              << "  --multi                 enabled multiple lines of input                                                         [*]\n"
              << "                          when enabled,  `" << MULTI_LINE_END_MARKER << "` marks the end of your input.\n"
              << "  --format FMT            conversion format (model specific, FMT = chat | completion | qa) (default: chat)\n"
              << "  --max_new_tokens N      max number of new tokens in a round of generation (default: -1, i.e. unlimited)\n"
              << "  +single_turn            single-turn (i.e. restart on each turn) (default: OFF)                                  [*]\n"
              << "Performance options:\n"
              << "  -n, --threads N         number of threads for inference (default: number of cores)\n"
              << "  -ngl, --n_gpu_layers N  number of the main model layers to offload to a backend device (GPU) (default: GPU not used)\n"
              << "                          N ::= one_spec;...\n"
              << "                          one_spec ::= [id:]spec, where spec ::= [n|epilog|prolog|all]\n"
              << "                          `-ngl N` is equiv. to `-mgl any N`.\n"
              << "  -mgl, --model_gpu_layers MODEL N\n"
              << "                          number of accessray model layers to offload to a backend device (GPU) (default: GPU not used)\n"
              << "                          MODEL ::= main | any | vis | tts | asr | ... \n"
              << "                          `main` and `any` are two special identifiers for the main model and wildcard to any model. \n"
              << "                          N ::= one_spec;..., see `-ngl`\n"
              << "  +moe_on_cpu             alway use CPU for sparse operations (MoE) (default: off)\n"
              << "  --rpc_endpoints EP..    RPC endpoints (i.e. servers) for distributed inference (default: empty)\n"
              << "                          EP1;EP2, where EP ::= host:port\n"
              << "  --cache_dtype T         cache data type, T ::= f32 | f16 (default: f16)\n"
              << "  --batch_size N          batch size (default: " << args.batch_size << ")\n"
              << "                          note: trade-off between prompt throughput and memory usage.\n"
              << "  --re_quantize Q         re-quantize model weights during loading (Q ::= q8_0 | q4_0 | q4_1 | q4_k | ...) (default: no re-quantization)\n"
              << "                          note: it does not make sense to re-quantize to a larger size.\n"
              << "Sampling options:\n"
              << "  --sampling ALG          sampling algorithm (ALG = greedy | top_p | tfs) (default: top_p) \n"
              << "                          where, tfs = Tail Free Sampling\n"
              << "  -t, --temp T            temperature (default: " << args.temp << ") (Note: `-t 0` also sets sampling algorithm to greedy)\n"
              << "  --top_k N               top-k sampling (default: " << args.top_k << ")\n"
              << "  --top_p N               top-p sampling (default: " << args.top_p << ")\n"
              << "  --tfs_z Z               Z param for TFS (default: " << args.tfs_z << ")\n"
              << "  --repeat_penalty N      repetition penalty (default: " << args.repeat_penalty << ", 1.0=no penalty)\n"
              << "  --presence_penalty N    penalty alpha for presence (default: " << args.presence_penalty << ", 0.0=disabled)\n"
              << "  --frequency_penalty N   penalty alpha for probability (default: " << args.frequency_penalty << ", 0.0=disabled)\n"
              << "  --penalty_window N      last N tokens to consider for penalize (default: " << args.penalty_window << ", 0=disable all)\n"
              << "  --seed N                seed for random generator (default: random)\n"
              << "  --beam_size N           beam size for generation (default: -1, disabled)\n"
              << "                          functionality of beam search limited.\n"
              << "RAG options:\n"
              << "  --set_vs_name           set vector store name.\n"
              << "                          all following vector store files are merged into this vector store. (optional. default: `default`)\n"
              << "                          Note: command line RAG chat will always the first store.\n"
              << "  --vector_store FILE     append a vector store file (when RAG enabled, at lease one is required)\n"
              << "  --embedding_model PATH  embedding model path (when set, RAG is enabled)\n"
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
              << "  --emb_rank_query_sep    separator for embedding & rerank query (default: \"\", i.e. disabled)\n"
              << "                          only used without main model\n"
              << "  --hide_reference        do not show references (default: false)                                                     [*]\n"
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
              << "CoT options:\n"
              << "   +detect_thoughts       turn on detection of thoughts in the output (default: OFF)\n"
              << "  --thought_tags O C      customize thought tags (Opening & Closing) to override built-in tags\n"
              << "                          built-in tags: " << show_default_thought_tags() << "\n"
              << "                          this options enables detection of thoughts implicitly.\n"
              << "Multi-model options:\n"
              << "  --multimedia_file_tags OPENING CLOSING\n"
              << "                          multimedia file tags. Default: Not set. Example {{ }}: {{TAG:/path/to/image.png}}\n"
              << "                          where TAG ::= image | video | audio\n"
              << "  --tts_export FILE       save generated PCM samples into a file. (default: not save)                                 [*]\n"
              << "Session:\n"
              << "  --save_session N FILE   save session to FILE after N round(s) of chatting (N >= 0) and quit                         [*]\n"
              << "                          when N = 0, system prompt is evaluated.\n"
              << "  --load_session FILE     load session from FILE                                                                      [*]\n"
              << "Misc:\n"
              << "  --init_vs FILE          init vector store file from input                                                           [*]\n"
              << "  --merge_vs FILE         merge multiple vector store files into a single one                                         [*]\n"
              << "  --tokenize              (debug) tokenize `prompt` and exit                                                          [*]\n"
              << "  --test FILE             test against inputs from a file and exit                                                    [*]\n"
              << "  --hide_banner           hide banner                                                                                 [*]\n"
              << "  --show                  show model info and quit                                                                    [*]\n"
              << "  --show_devices          show info about backends and devices, then quit                                             [*]\n"
              << "  --dump_dot FILE         dump sched splits to a DOT file, and exit with -1\n"
              << "  --log_level             log level. (default: 4 - ERROR)\n"
              << "  --serve_rpc [H:]P[@id]  as a RPC server on host:port (optional: host default to 127.0.0.1, id defaults to 0)        [#]\n"
              << "  --ggml_dir DIR          specify directory of GGML\n"
              << "  --set KEY VALUE         set a pair of additional args.\n"
              << "Additional key-value args:\n"
              << "  --kv                    start of additional args. all following options are interpreted as k-v pairs\n"
              << "  key value               a key-value pair of args\n"
              << "\n------------------------\n"
              << "*: implemented by front end (i.e. `main.cpp` or apps using bindings)\n"
              << "#: implemented by front end & backend\n"
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
    const size_t argc = argv.size();

    #define handle_para0(fmt1, field, f)        \
        else if ((utils::is_same_command_option(arg, fmt1)))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field = f(argv[c].c_str());                            \
        }

    #define handle_param(fmt1, fmt2, field, f)    \
        else if ((utils::is_same_command_option(arg, fmt1)) || (utils::is_same_command_option(arg, fmt2)))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field = f(argv[c].c_str());                            \
        }

    #define append_param(fmt1, field, f)        \
        else if ((utils::is_same_command_option(arg, fmt1)))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field.push_back(f(argv[c].c_str()));                   \
        }

    #define handle_flag(field)    \
        else if ((utils::is_same_command_option(arg, "+" #field)) || (utils::is_same_command_option(arg, "--" #field))) \
        {                                                                   \
            args.field = true;                                              \
        }

    size_t c = 1;

    try
    {
        while (c < argc)
        {
            const char *arg = argv[c].c_str();
            if ((utils::is_same_command_option(arg, "--help"))
                || (utils::is_same_command_option(arg, "-h"))
                || (utils::is_same_command_option(arg, "-?")))
            {
                args.show_help = true;
            }
            else if ((utils::is_same_command_option(arg, "--interactive")) || (strcmp(arg, "-i") == 0))
            {
                args.interactive = true;
            }
            else if (utils::is_same_command_option(arg, "--multi"))
            {
                args.multi_line = true;
            }
            else if (utils::is_same_command_option(arg, "--hide_banner"))
            {
                args.show_banner = false;
            }
            handle_flag(tokenize)
            handle_flag(hide_reference)
            handle_flag(show)
            handle_flag(show_devices)
            handle_flag(reversed_role)
            handle_flag(rag_dump)
            handle_flag(rerank_rewrite)
            handle_flag(moe_on_cpu)
            handle_flag(detect_thoughts)
            handle_flag(single_turn)
            else if (utils::is_same_command_option(arg, "--format"))
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
            else if (utils::is_same_command_option(arg, "--save_session"))
            {
                c++;
                if (c + 1 < argc)
                {
                    args.save_session_rounds = std::stoi(argv[c]);
                    args.save_session        = argv[c + 1];
                    c++;
                }
            }
            else if (utils::is_same_command_option(arg, "--kv"))
            {
                while (c + 2 < argc)
                {
                    args.additional.insert_or_assign(argv[c + 1], argv[c + 2]);
                    c += 2;
                }
            }
            else if (utils::is_same_command_option(arg, "--vector_store"))
            {
                c++;
                if (c < argc)
                {
                    if (args.vector_stores.find(args.cur_vs_name) == args.vector_stores.end())
                    {
                        args.vector_stores.insert(std::pair(args.cur_vs_name, std::vector<std::string>()));
                    }

                    args.vector_stores.at(args.cur_vs_name).push_back(argv[c]);
                }
            }
            else if (utils::is_same_command_option(arg, "--thought_tags"))
            {
                if (c + 2 < argc)
                {
                    args.thought_tags[0] = argv[c + 1];
                    args.thought_tags[1] = argv[c + 2];
                    c += 2;
                    args.detect_thoughts = true;
                }
            }
            else if (utils::is_same_command_option(arg, "--multimedia_file_tags"))
            {
                if (c + 2 < argc)
                {
                    args.multimedia_file_tags[0] = argv[c + 1];
                    args.multimedia_file_tags[1] = argv[c + 2];
                    c += 2;
                }
            }
            else if (utils::is_same_command_option(arg, "--set"))
            {
                if (c + 2 < argc)
                {
                    args.additional[argv[c + 1]] = argv[c + 2];
                    c += 2;
                }
            }
            else if (utils::is_same_command_option(arg, "-mgl") || utils::is_same_command_option(arg, "--model_gpu_layers"))
            {
                if (c + 2 < argc)
                {
                    args.model_n_gpu_layers[argv[c + 1]] = argv[c + 2];
                    c += 2;
                }
            }
            else if (utils::is_same_command_option(arg, "-ngl") || utils::is_same_command_option(arg, "--n_gpu_layers"))
            {
                c++;
                if (c < argc)
                {
                    args.model_n_gpu_layers["any"] = argv[c];
                }
            }
            handle_param("--model",                 "-m", model_path,           std::string)
            handle_param("--prompt",                "-p", prompt,               std::string)
            handle_para0("--prompt_file",                 prompt,               load_txt)
            handle_param("--system",                "-s", system,               std::string)
            handle_para0("--sys_file",                    system,               load_txt)
            handle_para0("--ai_prefix",                   ai_prefix,            std::string)
            handle_param("--max_length",            "-l", max_length,           std::stoi)
            handle_param("--max_context_length",    "-c", max_context_length,   std::stoi)
            handle_para0("--extending",                   extending,            parse_extending_method)
            handle_para0("--sampling",                    sampling,             std::string)
            handle_param("--top_k",                 "-k", top_k,                std::stoi)
            handle_param("--top_p",                 "-q", top_p,                std::stof)
            handle_para0("--tfs_z",                       tfs_z,                std::stof)
            handle_param("--temp",                  "-t", temp,                 std::stof)
            handle_para0("--presence_penalty",            presence_penalty,     std::stof)
            handle_para0("--repeat_penalty",              repeat_penalty,       std::stof)
            handle_para0("--frequency_penalty",           frequency_penalty,    std::stof)
            handle_para0("--penalty_window",              penalty_window,       std::stoi)
            handle_param("--threads",               "-n", num_threads,          std::stoi)
            handle_para0("--seed",                        seed,                 std::stoi)
            handle_para0("--test",                        test_fn,              std::string)
            handle_para0("--set_vs_name",                 cur_vs_name,          std::string)
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
            handle_para0("--emb_rank_query_sep",          emb_rank_query_sep,   std::string)
            handle_para0("--init_vs",                     vector_store_in,      std::string)
            handle_para0("--merge_vs",                    merge_vs,             std::string)
            handle_para0("--layer_spec",                  layer_spec,           std::string)
            handle_para0("--load_session",                load_session,         std::string)
            handle_para0("--dump_dot",                    dump_dot,             std::string)
            handle_para0("--beam_size",                   beam_size,            std::stoi)
            handle_para0("--log_level",                   log_level,            std::stoi)
            handle_para0("--rpc_endpoints",               rpc_endpoints,        std::string)
            handle_para0("--serve_rpc",                   serve_rpc,            std::string)
            handle_para0("--ggml_dir",                    ggml_dir,             std::string)
            handle_para0("--cache_dtype",                 cache_dtype,          std::string)
            handle_para0("--batch_size",                  batch_size,           std::stoi)
            handle_para0("--tts_export",                  tts_export,           std::string)
            handle_para0("--re_quantize",                 re_quantize,          std::string)
            handle_para0("--max_new_tokens",              max_new_tokens,       std::stoi)
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

    if (args.detect_thoughts)
    {
        if ((args.thought_tags[0].size() > 0) && (args.thought_tags[0].size() > 0))
            thought_interceptor.init({{args.thought_tags[0], args.thought_tags[1]}});
        else
            thought_interceptor.init(THOUGHT_TAGS);
    }

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
    void put_thought_chunk(bool first, const std::string &chunk) override;
    void end_thought(void) override;
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
            tok_number > 0 ? duration_sec / tok_number : 0.0,
            duration_sec > 0.0 ? tok_number / duration_sec * 1000 : 0.0);
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
    chatllm::Messages history(args.multimedia_file_tags[0], args.multimedia_file_tags[1]);
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
            oss    << "with " << total_param_num << " (" << std::fixed << std::setprecision(1) << (double)total_effective_param_num / 1000000000. << "B active) parameters." << '\n';
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

static void play_audio(const std::vector<int16_t> &data, const int sample_rate, const int channels, TextStreamer &streamer, std::string export_fn)
{
    auto fn = export_fn.size() > 0 ? export_fn : utils::tmpname();
    std::ofstream file(fn, std::ios::binary);
    file.write((const char *)data.data(), data.size() * sizeof(data[0]));
    file.close();

    char cmd[2048];
    sprintf(cmd, "ffplay -loglevel error -autoexit -f s16le -ch_layout %s -sample_rate %d \"%s\"", channels == 1 ? "mono" : "stereo", sample_rate, fn.c_str());
    int r = system(cmd);
    if (export_fn.size() < 1)
        std::remove(fn.c_str());
    if (r != 0)
        streamer.cout << "FAILED to play audio. Please check ffplay is installed." << std::endl;
}

static void run_tts(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config)
{
    std::vector<int16_t> result;
    int sample_rate = 0;
    int channels = 0;

    if (!args.interactive)
    {
        pipeline.speech_synthesis(args.prompt, gen_config, result, sample_rate, channels);
        play_audio(result, sample_rate, channels, streamer, args.tts_export);
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
        pipeline.speech_synthesis(input, gen_config, result, sample_rate, channels);
        streamer.cout << "      > ";

        play_audio(result, sample_rate, channels, streamer, args.tts_export);

    }
    streamer.cout << "Bye\n";
}

static void run_asr_ocr(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config, const std::string &tag)
{
    std::vector<int16_t> result;
    int sample_rate = 0;
    int channels = 0;

    const std::string opening = "{{";
    const std::string closing = "}}";

    chatllm::Messages history(opening, closing);

    auto make_msg = [&tag, &opening, &closing](const std::string &fn) {
        std::ostringstream oss;
        oss << opening << tag << ":" << fn << closing;
        return oss.str();
    };

    if (!args.interactive)
    {
        history.push_back(make_msg(args.prompt), chatllm::MsgRole::User);
        pipeline.chat(history, gen_config, &streamer);
        return;
    }

    while (1)
    {
        streamer.cout << "File > " << std::flush;
        std::string input;
        if (!get_utf8_line(input, args.multi_line))
        {
            streamer.cout << "FAILED to read line." << std::endl;
            break;
        }
        if (input.empty()) continue;

        history.push_back(make_msg(input), chatllm::MsgRole::User);
        pipeline.chat(history, gen_config, &streamer);

        history.clear();
        pipeline.restart();
    }
    streamer.cout << "Bye\n";
}

static void run_text_embedding(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer, const chatllm::GenerationConfig &gen_config)
{
    std::vector<float> result;
    chatllm::BaseTokenizer::EmbeddingPurpose purpose = chatllm::BaseTokenizer::EmbeddingPurpose::Document;

    if (!args.interactive)
    {
        pipeline.text_embedding(args.prompt, gen_config, result, purpose);
        print_embedding(result, streamer.cout);
        return;
    }

    while (1)
    {
        streamer.cout << "Input " << (purpose == chatllm::BaseTokenizer::EmbeddingPurpose::Document ? "Doc" : "Query") <<  " > " << std::flush;
        std::string input;
        if (!get_utf8_line(input, args.multi_line))
        {
            streamer.cout << "FAILED to read line." << std::endl;
            break;
        }
        if (input.empty()) continue;

        result.clear();
        pipeline.text_embedding(input, gen_config, result, purpose);
        streamer.cout << "      > ";

        print_embedding(result, streamer.cout);

        purpose = purpose == chatllm::BaseTokenizer::EmbeddingPurpose::Document ?
                    chatllm::BaseTokenizer::EmbeddingPurpose::Query : chatllm::BaseTokenizer::EmbeddingPurpose::Document;
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

#define DEF_GenerationConfig(gen_config, args) chatllm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.reversed_role, \
                                         args.top_k, args.top_p, args.temp, args.num_threads, args.sampling, args.presence_penalty, args.tfs_z); \
                                         gen_config.set_ai_prefix(args.ai_prefix); gen_config.dump_dot = args.dump_dot; \
                                         gen_config.emb_rank_query_sep = args.emb_rank_query_sep; \
                                         gen_config.repeat_penalty = args.repeat_penalty; \
                                         gen_config.frequency_penalty = args.frequency_penalty; \
                                         gen_config.penalty_window = args.penalty_window; \
                                         gen_config.max_new_tokens = args.max_new_tokens;

#define DEF_ExtraArgs(pipe_args, args)  \
    chatllm::ModelObject::extra_args pipe_args(args.max_length, args.layer_spec, args.moe_on_cpu, args.num_threads, args.batch_size, args.cache_dtype, args.re_quantize);\
    pipe_args.model_n_gpu_layers = args.model_n_gpu_layers; \
    pipe_args.additional = args.additional

chatllm::BaseStreamer *get_streamer_for_log(void);

void log_internal(int level, const char * text)
{
    chatllm::BaseStreamer *streamer = get_streamer_for_log();
    if (nullptr == streamer) return;

    std::ostringstream oss;
    static const char tags[] = {' ', 'D', 'I', 'W', 'E', '.'};
    if (level < streamer->log_level) return;

    if ((0 <= level) && (level < (int)sizeof(tags)))
        oss << tags[level];
    else
        oss << '?';
    std::string s(text);
    s.erase(s.find_last_not_of('\n') + 1);
    oss << s;
    streamer->putln(oss.str(),  chatllm::BaseStreamer::LOGGING);
}

static void _ggml_log_callback(enum ggml_log_level level, const char * text, void * user_data)
{
    log_internal(level, text);
}

void chat(Args &args, chatllm::Pipeline &pipeline, TextStreamer &streamer)
{
    streamer.set_tokenizer(pipeline.tokenizer);

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
    streamer.set_interceptor(&thought_interceptor);

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
    chatllm::Messages history(args.multimedia_file_tags[0], args.multimedia_file_tags[1]);

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
        case chatllm::ModelPurpose::TTS:
            run_tts(args, pipeline, streamer, gen_config);
            return;
        case chatllm::ModelPurpose::ASR:
            run_asr_ocr(args, pipeline, streamer, gen_config, "audio");
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

    if (args.reversed_role)
    {
        CHATLLM_CHECK(args.save_session_rounds < 0) << "TODO: save_session_rounds for reversed_role";

        streamer.cout << ai_prompt << " > " << args.prompt << std::endl << std::flush;
        history.push_back(args.prompt, chatllm::MsgRole::User);

        while (1)
        {
            streamer.cout << user_prompt << " > " << std::flush;
            std::string input;
            if (!get_utf8_line(input, args.multi_line))
            {
                streamer.cout << "FAILED to read line." << std::endl;
                break;
            }
            if (input.empty()) continue;

            history.push_back(input, chatllm::MsgRole::Assistant);
            streamer.cout << ai_prompt << " > " << std::flush;
            std::string output = pipeline.chat(history, gen_config, &streamer);
            history.push_back(output, chatllm::MsgRole::User);

            if (args.single_turn)
            {
                history.clear();
                pipeline.restart();
            }
        }
    }
    else
    {
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

            if (args.single_turn)
            {
                history.clear();
                pipeline.restart();
            }
        }
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

void TextStreamer::put_thought_chunk(bool first, const std::string &chunk)
{
    if (first)
        cout << "====== Thinking ======" << std::endl;
    cout << chunk << std::flush;
}

void TextStreamer::end_thought(void)
{
    cout << std::endl << "========================" << std::endl << std::flush;
}

void TextStreamer::end()
{
    BaseStreamer::end();
    ref_count = 0;
    cout << std::endl;
}

static void prepare_rpc_devices(const Args &args)
{
    if (!chatllm::ComputeManager::prepare_rpc_devices(args.rpc_endpoints))
    {
        chatllm::ggml::log(GGML_LOG_LEVEL_ERROR, "Failed to prepare RPC devices");
    }
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

static void start_rpc_server(std::string endpoint)
{
    int device = 0;

    auto pos = endpoint.find('@');
    if (pos != std::string::npos)
    {
        device = std::stoi(endpoint.substr(pos + 1));
        endpoint = endpoint.substr(0, pos);
    }

    if (!chatllm::ComputeManager::start_rpc_server(device, endpoint.c_str()))
    {
        chatllm::ggml::log(GGML_LOG_LEVEL_ERROR, "Failed to start RPC server at %s@%d", endpoint.c_str(), device);
    }
    else;
}

static chatllm::BaseStreamer *log_streamer = nullptr;
chatllm::BaseStreamer *get_streamer_for_log(void)
{
    return log_streamer;
}

static int init_vector_store(Args &args)
{
    DEF_ExtraArgs(pipe_args, args);
    chatllm::Pipeline pipeline(args.embedding_model_path, pipe_args);
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

static int merge_vector_store(Args &args)
{
    std::vector<std::string> files;
    for (auto x : args.vector_stores)
    {
        files.insert(files.end(), x.second.begin(), x.second.end());
    }
    CVectorStore vs(args.vc, files);
    vs.ExportDB(args.merge_vs.c_str());
    printf("Vector store saved to: %s\n", args.merge_vs.c_str());
    return 0;
}

static void show_devices(void)
{
    std::vector<chatllm::ComputeManager::DeviceInfo> devs;
    chatllm::ComputeManager::get_devices_info(devs);
    for (size_t i = 0; i < devs.size(); i++)
    {
        auto &dev = devs[i];
        printf("%2zd: %s - %s (%s)\n", i, dev.backend_name.c_str(), dev.name.c_str(), dev.description.c_str());
        printf("    type: %s\n", chatllm::ComputeManager::dev_type_to_str(dev.type).c_str());
        printf("    memory total: %zd B\n", dev.total_memory);
        printf("    memory free : %zd B\n", dev.free_memory);
    }
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
    //setvbuf(stdout, nullptr, _IOFBF, 1000);

#else
int main(int argc, const char **argv)
{
    std::vector<std::string> utf_args;
    for (int i = 0; i < argc; i++)
        utf_args.push_back(argv[i]);
#endif

    //audio::test();
    //return -1;

    Args args;
    auto count = parse_args(args, utf_args);
    if (args.show_help)
    {
        usage(utf_args[0]);
        return 0;
    }

    TextStreamer streamer(nullptr);
    log_streamer = &streamer;
    streamer.log_level = args.log_level;
    ggml_log_set(_ggml_log_callback, nullptr);

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

    chatllm::ComputeManager::init(args.ggml_dir);
    prepare_rpc_devices(args);

    if (args.show_devices)
    {
        show_devices();
        return 0;
    }

    if (args.vector_store_in.size() > 0)
        return init_vector_store(args);

    if (args.merge_vs.size() > 0)
        return merge_vector_store(args);

    if (args.serve_rpc.size() > 0)
    {
        start_rpc_server(args.serve_rpc);
        return 0;
    }

    try
    {
        DEF_ExtraArgs(pipe_args, args);

        if (args.embedding_model_path.size() < 1)
        {
            if (args.beam_size < 1)
            {
                chatllm::Pipeline pipeline(args.model_path, pipe_args);
                chat(args, pipeline, streamer);
            }
            else
            {
                chatllm::BeamSearchPipeline pipeline(args.model_path, pipe_args, args.beam_size);
                chat(args, pipeline, streamer);
            }
        }
        else
        {
            CHATLLM_CHECK(args.beam_size < 1) << "beam search is not supported for RAG";

            chatllm::RAGPipeline pipeline(args.model_path, pipe_args,
                args.vc, args.vector_stores,
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

static std::vector<std::string> init_params;
static Args init_args;

void chatllm_append_init_param(const char *utf8_str)
{
    if (init_params.size() == 0) init_params.push_back("");
    init_params.push_back(utf8_str);
}

static int chatllm_do_init(const std::vector<std::string> *p_obj_params = nullptr)
{
    static bool initialized = false;
    if (initialized) return 0;

    initialized = true;

    // it's ok to call this multiple times
    ggml_log_set(_ggml_log_callback, nullptr);

    if (p_obj_params && (init_params.size() == 0))
    {
        auto count = parse_args(init_args, *p_obj_params);
        if (count < p_obj_params->size())
            return (int)count;
    }
    else
    {
        auto count = parse_args(init_args, init_params);
        if (count < init_params.size())
            return (int)count;
    }

    chatllm::ComputeManager::init(init_args.ggml_dir);
    prepare_rpc_devices(init_args);

    return 0;
}

int API_CALL chatllm_init(void)
{
    return chatllm_do_init();
}

class Chat
{
public:
    Chat():
        streamer(nullptr), pipeline(nullptr),
        content_scratch(&history, ""),
        sess_n_past(-1), sess_hist_len(-1), is_rag(false),
        is_async_busy(false), async_result_int(0)
    {
        append_param("...");
    }

    void append_param(const char *utf8_str)
    {
        params.push_back(utf8_str);
    }

    std::string new_tmp_file(void)
    {
        tmp_files.push_back(utils::tmpname());
        return tmp_files.back();
    }

    ~Chat()
    {
        for (auto &s : tmp_files)
            std::filesystem::remove(s);
    }

public:
    std::vector<std::string> params;
    chatllm::Messages history;
    std::unique_ptr<chatllm::BaseStreamer> streamer;
    std::unique_ptr<chatllm::Pipeline> pipeline;
    chatllm::GenerationConfig gen_config;
    chatllm::Content content_scratch;
    std::vector<std::string> tmp_files;
    int sess_n_past;
    int sess_hist_len;
    Args args;
    std::string tool_input;
    std::string tool_completion;    // part of the output is generated by external tools
    bool is_rag;
    bool is_async_busy;
    int async_result_int;
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

    void put_thought_chunk(bool is_first, const std::string &chunk) override
    {
        f_print(user_data, PRINT_THOUGHT_CHUNK, chunk.c_str());
    }

    void end_thought() override
    {
        f_print(user_data, PRINT_EVT_THOUGHT_COMPLETED, "");
    }

    void putln(const std::string &line, TextType type = TextType::META) override
    {
        f_print(user_data, (int)type, line.c_str());
    }

    void put_event(int type)
    {
        f_print(user_data, type, "");
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

static std::vector<std::unique_ptr<Chat>> chat_objects;

#define DEF_CHAT()                         \
    Chat *chat = reinterpret_cast<Chat *>(obj);

#define DEF_CHAT_STREAMER()                         \
    DEF_CHAT();                                     \
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer.get())

chatllm::BaseStreamer *get_streamer_for_log(void)
{
    return chat_objects.size() > 0 ? chat_objects[0]->streamer.get() : nullptr;
}

struct chatllm_obj *chatllm_create(void)
{
    auto chat = new Chat();
    chat_objects.emplace_back(chat);
    if (chat_objects.size() == 1) {
        // it's ok to call this multiple times
        ggml_log_set(_ggml_log_callback, nullptr);
    }
    return (chatllm_obj *)chat;
}

int chatllm_destroy(struct chatllm_obj *obj)
{
    DEF_CHAT_STREAMER();

    if (!streamer->is_prompt || chat->is_async_busy) return -1;

    auto it = find_if(chat_objects.begin(), chat_objects.end(), [=](auto &c) { return c.get() == chat; });

    if (it != chat_objects.end())
    {
        chat_objects.erase(it);
        return 0;
    }

    return -1;
}

void chatllm_append_param(struct chatllm_obj *obj, const char *utf8_str)
{
    DEF_CHAT();

    chat->append_param(utf8_str);
}

void chatllm_multimedia_msg_prepare(struct chatllm_obj *obj)
{
    DEF_CHAT();
    chat->content_scratch.pieces.clear();
}

int chatllm_multimedia_msg_append(struct chatllm_obj *obj, const char *type, const char *utf8_str)
{
    DEF_CHAT();
    if (strcmp(type, "text") == 0)
    {
        chat->content_scratch.push_back(std::string(utf8_str), chatllm::ContentPiece::Type::Text);
        return 0;
    }

    auto t = chatllm::ContentPiece::type_parse(type);
    if (chatllm::ContentPiece::Type::Text == t) return -1;

    auto data = base64::decode(utf8_str);
    auto fn   = chat->new_tmp_file();
    auto f = std::fopen(fn.c_str(), "wb");
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);

    chat->content_scratch.push_back(fn, t);

    return 0;
}

static void emit_model_info(Chat *chat, const Args &args, chatllm::Pipeline &pipeline)
{
    auto o = json::JSON::Make(json::JSON::Class::Object);
    o["name"]               = pipeline.model->type_name();
    auto native_name        = pipeline.model->native_name();
    if (pipeline.model->native_name().size() > 0)
        o["native_name"]    = pipeline.model->native_name();
    o["purpose"]            = chatllm::to_string(pipeline.model->get_purpose());
    o["param_num"]          = pipeline.model->get_param_num(false);
    o["active_param_num"]   = pipeline.model->get_param_num(true);
    o["context_length"]     = pipeline.model->get_max_length();
    o["training_context_length"] = pipeline.get_loader()->basic_config.max_length;
    if (pipeline.model->get_text_embedding_dim() > 0)
        o["embedding_dim"]  = pipeline.model->get_text_embedding_dim();

    std::string s = chatllm::format_model_capabilities(pipeline.model->get_type());
    auto cap = json::JSON::Make(json::JSON::Class::Array);
    size_t pos = 0;
    int i = 0;
    while ((pos = s.find(", ")) != std::string::npos)
    {
        auto capability = s.substr(0, pos);
        cap[i] = capability;
        i++;
        s.erase(0, pos + 2);
    }
    if (s.size() > 0) cap[i] = s;
    o["capabilities"] =  cap;

    chat->streamer->putln(o.dump(), chatllm::BaseStreamer::TextType::MODEL_INFO);
}

static int start_chat(Chat *chat, Args &args, chatllm::Pipeline &pipeline)
{
    int r = 0;
    chat->pipeline = std::unique_ptr<chatllm::Pipeline>(&pipeline);

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
    chat->streamer->set_interceptor(&thought_interceptor);

    DEF_GenerationConfig(gen_config, args);

    chat->gen_config = gen_config;

    show_banner(pipeline, args.interactive && args.show_banner, chat->streamer.get());

    emit_model_info(chat, args, pipeline);

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

    auto r = chatllm_do_init(&chat->params);
    if (r != 0) return r;

    auto count = parse_args(args, chat->params);

    if (count < chat->params.size())
        return (int)count;

    if (args.num_threads <= 0)
        args.num_threads = get_num_physical_cores();

    args.interactive = true;
    chat->streamer = std::unique_ptr<chatllm::BaseStreamer>(new FFIStreamer(nullptr, f_print, f_end, user_data));
    chat->streamer->log_level = init_args.log_level;
    chat->history.set_mm_tags(args.multimedia_file_tags[0], args.multimedia_file_tags[1]);

    try
    {
        DEF_ExtraArgs(pipe_args, args);

        if ((args.embedding_model_path.size() < 1) || (args.vector_stores.empty()))
        {
            if (args.model_path.size() < 1)
                return -1;

            if (args.beam_size < 1)
            {
                auto pipeline = new chatllm::Pipeline(args.model_path, pipe_args);
                chat->streamer->tokenizer = pipeline->tokenizer;
                return start_chat(chat, args, *pipeline);
            }
            else
            {
                auto pipeline = new chatllm::BeamSearchPipeline(args.model_path, pipe_args, args.beam_size);
                chat->streamer->tokenizer = pipeline->tokenizer;
                return start_chat(chat, args, *pipeline);
            }
        }
        else
        {
            CHATLLM_CHECK(args.beam_size < 1) << "beam search is not supported for RAG";

            auto pipeline = new chatllm::RAGPipeline(args.model_path, pipe_args,
                args.vc, args.vector_stores,
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
            chat->is_rag = true;
            chat->streamer->tokenizer = pipeline->tokenizer;
            return start_chat(chat, args, *pipeline);
        }

    }
    catch (std::exception &e)
    {
        f_print(user_data, PRINTLN_ERROR, e.what());
        return EXIT_FAILURE;
    }
}

#define ERR_ASYNC_ONGOING       (INT_MIN)

int chatllm_get_async_result_int(struct chatllm_obj *obj)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    return chat->is_async_busy ? ERR_ASYNC_ONGOING : chat->async_result_int;
}

#define ASYNC_FUN_BODY(expr)    do {        \
    DEF_CHAT_STREAMER();                    \
    if (chat->is_async_busy) return -1;                     \
    chat->is_async_busy = true;                             \
                                                            \
    std::thread t([=]() {                                   \
        chat->async_result_int = expr;                      \
        chat->is_async_busy = false;                        \
        streamer->put_event(PRINT_EVT_ASYNC_COMPLETED);     \
    });                                                     \
    t.detach();                                             \
    return 0;                                               \
} while (false)

int chatllm_async_start(struct chatllm_obj *obj, f_chatllm_print f_print, f_chatllm_end f_end, void *user_data)
{
    ASYNC_FUN_BODY(chatllm_start(obj, f_print, f_end, user_data));
}

int chatllm_set_ai_prefix(struct chatllm_obj *obj, const char *utf8_str)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    chat->gen_config.set_ai_prefix(utf8_str);
    return 0;
}

static void chatllm_continue_chat(Chat *chat)
{
    FFIStreamer *streamer = dynamic_cast<FFIStreamer *>(chat->streamer.get());
    std::string external_ai = chat->tool_completion;

    if (external_ai.size() < 1) return;

    chat->tool_completion.clear();
    if (chat->history.size() < 1) return;

    int last_id = (int)chat->history.size() - 1;

    chat->history[last_id].content.push_back(external_ai);

    std::string output = chat->pipeline->chat(chat->history, chat->gen_config, streamer);

    chat->history[last_id].content.push_back(output);
    chat->history.save_token_cursor(chat->pipeline->get_cursor());
}

static int chatllm_generate(struct chatllm_obj *obj)
{
    int r = -0;
    DEF_CHAT_STREAMER();
    auto role_asst = chat->gen_config.reversed_role ? chatllm::MsgRole::User : chatllm::MsgRole::Assistant;

generate:
    std::string output = chat->pipeline->chat(chat->history, chat->gen_config, streamer);
    chat->history.push_back(output, role_asst);
    chat->history.save_token_cursor(chat->pipeline->get_cursor());

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

int chatllm_user_input(struct chatllm_obj *obj, const char *utf8_str)
{
    DEF_CHAT_STREAMER();
    auto role_user = chat->gen_config.reversed_role ? chatllm::MsgRole::Assistant : chatllm::MsgRole::User;

    if (!streamer->is_prompt) return -1;

    if (chat->pipeline->is_loaded() && (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::Chat))
        return -1;

    chat->history.push_back(utf8_str, role_user);

    return chatllm_generate(obj);
}

int chatllm_user_input_multimedia_msg(struct chatllm_obj *obj)
{
    int r = 0;
    DEF_CHAT_STREAMER();
    auto role_user = chat->gen_config.reversed_role ? chatllm::MsgRole::Assistant : chatllm::MsgRole::User;

    if (!streamer->is_prompt) return -1;

    if (chat->pipeline->is_loaded() && (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::Chat))
        return -1;

    chat->history.push_back(chat->content_scratch, role_user);

    return chatllm_generate(obj);
}

int chatllm_async_user_input(struct chatllm_obj *obj, const char *utf8_str)
{
    ASYNC_FUN_BODY(chatllm_user_input(obj, utf8_str));
}

int chatllm_async_user_input_multimedia_msg(struct chatllm_obj *obj)
{
    ASYNC_FUN_BODY(chatllm_user_input_multimedia_msg(obj));
}

int chatllm_ai_continue(struct chatllm_obj *obj, const char *utf8_str)
{
    int r = 0;
    DEF_CHAT_STREAMER();

    if (!streamer->is_prompt) return -1;

    if (chat->pipeline->is_loaded() && (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::Chat))
        return -1;

    if (chat->history.size() < 1) return -2;
    if (chat->history[chat->history.size() - 1].role != chatllm::MsgRole::Assistant) return -3;

    std::string more = chat->pipeline->chat_continue(chat->history, utf8_str, chat->gen_config, streamer);

    chat->history[chat->history.size() - 1].content.push_back(more);
    chat->history.save_token_cursor(chat->pipeline->get_cursor());

    return r;
}

int chatllm_async_ai_continue(struct chatllm_obj *obj, const char *utf8_str)
{
    ASYNC_FUN_BODY(chatllm_ai_continue(obj, utf8_str));
}

int chatllm_tool_input(struct chatllm_obj *obj, const char *utf8_str)
{
    DEF_CHAT_STREAMER();

    chat->tool_input = std::string(utf8_str);
    if (!streamer->is_prompt)
        chatllm_continue_chat(chat);
    return 0;
}

int chatllm_async_tool_input(struct chatllm_obj *obj, const char *utf8_str)
{
    ASYNC_FUN_BODY(chatllm_tool_input(obj, utf8_str));
}

int chatllm_tool_completion(struct chatllm_obj *obj, const char *utf8_str)
{
    DEF_CHAT_STREAMER();

    if (streamer->is_prompt)
        return chatllm_user_input(obj, utf8_str);

    chat->tool_input = std::string(utf8_str);
    return 0;
}

int chatllm_async_tool_completion(struct chatllm_obj *obj, const char *utf8_str)
{
    ASYNC_FUN_BODY(chatllm_tool_completion(obj, utf8_str));
}

int chatllm_text_embedding(struct chatllm_obj *obj, const char *utf8_str, int purpose)
{
    int r = 0;
    DEF_CHAT_STREAMER();

    if (!chat->pipeline->is_loaded() || (chat->pipeline->model->get_purpose() != chatllm::ModelPurpose::TextEmbedding))
        return -1;

    chatllm::BaseTokenizer::EmbeddingPurpose purp = (chatllm::BaseTokenizer::EmbeddingPurpose)purpose;

    std::vector<float> result;
    std::string input(utf8_str);
    chat->pipeline->text_embedding(input, chat->gen_config, result, purp);

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

int chatllm_async_text_embedding(struct chatllm_obj *obj, const char *utf8_str, int purpose)
{
    ASYNC_FUN_BODY(chatllm_text_embedding(obj, utf8_str, purpose));
}

int chatllm_text_tokenize(struct chatllm_obj *obj, const char *utf8_str)
{
    DEF_CHAT_STREAMER();

    if (!chat->pipeline->is_loaded())
        return -1;

    std::vector<int> result;
    std::string input(utf8_str);
    chat->pipeline->text_tokenize(input, chat->gen_config, result);

    std::ostringstream oss;
    for (size_t i = 0; i < result.size() - 1; i++)
    {
        if ((i > 0) && ((i % 16) == 0)) oss << std::endl;
        oss << result[i] << ",";
    }
    oss << result.back();

    streamer->putln(oss.str(), chatllm::BaseStreamer::TextType::TOKEN_IDS);

    return (int)result.size();
}

int chatllm_rag_select_store(struct chatllm_obj *obj, const char *name)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);

    if (!chat->is_rag)
        return -1;

    auto pipeline = dynamic_cast<chatllm::RAGPipeline *>(chat->pipeline.get());
    return pipeline->select_vector_store(name) ? 0 : -1;
}

int chatllm_qa_rank(struct chatllm_obj *obj, const char *utf8_str_q, const char *utf8_str_a)
{
    int r = 0;
    DEF_CHAT_STREAMER();

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

int chatllm_async_qa_rank(struct chatllm_obj *obj, const char *utf8_str_q, const char *utf8_str_a)
{
    ASYNC_FUN_BODY(chatllm_qa_rank(obj, utf8_str_q, utf8_str_a));
}

void chatllm_restart(struct chatllm_obj *obj, const char *utf8_sys_prompt)
{
    DEF_CHAT_STREAMER();

    if (!streamer->is_prompt) return;

    if ((chat->sess_hist_len > 0) && (nullptr == utf8_sys_prompt))
    {
        if (chat->history.size() > (size_t)chat->sess_hist_len)
            chat->history.history.erase(chat->history.history.begin() + chat->sess_hist_len, chat->history.history.end());
        chat->pipeline->rewind(chat->sess_n_past);
    }
    else
    {
        chat->history.clear();
        chat->pipeline->restart();
        if (utf8_sys_prompt)
            chat->pipeline->set_system_prompt(utf8_sys_prompt);
    }
}

void chatllm_history_append(struct chatllm_obj *obj, int role_type, const char *utf8_str)
{
    DEF_CHAT_STREAMER();

    if (!streamer->is_prompt) return;
    if ((role_type < chatllm::MsgRole::User) || (role_type >= chatllm::MsgRole::LAST)) return;

    chat->history.push_back(utf8_str, static_cast<chatllm::MsgRole>(role_type));
}

int chatllm_history_append_multimedia_msg(struct chatllm_obj *obj, int role_type)
{
    DEF_CHAT_STREAMER();

    if (!streamer->is_prompt) return - 1;
    if ((role_type < chatllm::MsgRole::User) || (role_type >= chatllm::MsgRole::LAST)) return -2;

    chat->history.push_back(chat->content_scratch, static_cast<chatllm::MsgRole>(role_type));
    return (int)chat->history.size();
}

int API_CALL chatllm_history_get_cursor(struct chatllm_obj *obj)
{
    DEF_CHAT_STREAMER();
    return chat->history.get_cursor();
}

int API_CALL chatllm_history_set_cursor(struct chatllm_obj *obj, int pos)
{
    DEF_CHAT_STREAMER();
    int old = chat->history.get_cursor();
    if (pos >= old) return chatllm_history_get_cursor(obj);

    if (pos <= 0)
    {
        chatllm_restart(obj, nullptr);
        return chatllm_history_get_cursor(obj);
    }

    int cnt = 0;
    while (chat->history.size() > pos)
    {
        chat->history.history.pop_back();
    }

    chat->history.move_cursor_to(0);
    for (int i = (int)chat->history.size() - 1; i >= 0; i++)
    {
        if (chat->history[i].tok_pos >= 0)
        {
            chat->history.move_cursor_to(i + 1);
            break;
        }
    }

    chat->pipeline->set_cursor(chat->history.get_token_cursor());
    return chatllm_history_get_cursor(obj);
}

int API_CALL chatllm_get_cursor(struct chatllm_obj *obj)
{
    DEF_CHAT_STREAMER();
    return chat->pipeline->get_cursor();
}

int API_CALL chatllm_set_cursor(struct chatllm_obj *obj, int pos)
{
    DEF_CHAT_STREAMER();
    return chat->pipeline->set_cursor(pos);
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
    chat->gen_config.max_new_tokens = gen_max_tokens;
}

void chatllm_show_statistics(struct chatllm_obj *obj)
{
    Chat *chat = reinterpret_cast<Chat *>(obj);
    show_stat(*(chat->pipeline), *(chat->streamer));
}

int chatllm_save_session(struct chatllm_obj *obj, const char *utf8_str)
{
    DEF_CHAT_STREAMER();

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
    DEF_CHAT_STREAMER();

    if (!streamer->is_prompt) return -1;

    int r = chat->pipeline->load_session(chat->history, utf8_str, streamer, &chat->sess_n_past);
    if (0 == r)
        chat->sess_hist_len = (int)chat->history.size();
    return r;
}

#endif
