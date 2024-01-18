#include "chat.h"
#include <iomanip>
#include <iostream>
#include <thread>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstring>
#include <random>
#include <isocline.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

struct Args
{
    std::string model_path = "chatllm-ggml.bin";
    std::string system = "";
    std::string prompt = "你好";
    std::string extending = "restart";
    int max_length = -1;
    int max_context_length = 512;
    bool interactive = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.7;
    int num_threads = 0;
    int seed;
    chatllm::ChatFormat format = chatllm::ChatFormat::CHAT;
    bool tokenize = false;
};

#define MULTI_LINE_END_MARKER_W  L"\\."
#define MULTI_LINE_END_MARKER     "\\."

void usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        model path (default: chatllm-ggml.bin)\n"
              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
              << "  -s, --system SYSTEM     system prompt (instruction) (default: model specific)\n"
              << "  -i, --interactive       run in interactive mode\n"
              << "  -l, --max_length N      max total length including prompt and output (default: model specific)\n"
              << "  -t, --threads N         number of threads for inference (default: number of cores)\n"
              << "  -c, --max_context_length N\n"
              << "                          max context length (default: 512)\n"
              << "  --extending EXT         context extending method (EXT = restart | shift) (default: restart)\n"
              << "  --top_k N               top-k sampling (default: 0)\n"
              << "  --top_p N               top-p sampling (default: 0.7)\n"
              << "  --temp N                temperature (default: 0.95)\n"
              << "  --seed N                seed for random generator (default: random)\n"
              << "  --format FMT            conversion format (model specific, FMT = chat | completion | qa) (default: chat)\n"
              << "  --tokenize              (debug)tokenize `prompt` and exit\n"
              << std::endl;
}

static Args parse_args(int argc, const char **argv)
{
    Args args;
    std::random_device rd;
    args.seed = rd();

    #define handle_para0(fmt1, field, f)    \
        else if ((strcmp(arg, fmt1) == 0))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field = f(argv[c]);                                    \
        }

    #define handle_param(fmt1, fmt2, field, f)    \
        else if ((strcmp(arg, fmt1) == 0) || (strcmp(arg, fmt2) == 0))      \
        {                                                                   \
            c++;                                                            \
            if (c < argc)                                                   \
                args.field = f(argv[c]);                                    \
        }

    int c = 1;
    while (c < argc)
    {
        const char *arg = argv[c];
        if ((strcmp(arg, "--help") == 0) || (strcmp(arg, "-h") == 0))
        {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if ((strcmp(arg, "--interactive") == 0) || (strcmp(arg, "-i") == 0))
        {
            args.interactive = true;
        }
        else if (strcmp(arg, "--tokenize") == 0)
        {
            args.tokenize = true;
        }
        else if (strcmp(arg, "--format") == 0)
        {
            c++;
            if (c < argc)
            {
                if (strcmp(argv[c], "completion") == 0)
                    args.format = chatllm::ChatFormat::COMPLETION;
                else if (strcmp(argv[c], "qa") == 0)
                    args.format = chatllm::ChatFormat::QA;
                else
                    args.format = chatllm::ChatFormat::CHAT;
            }
        }
        handle_param("--model",                 "-m", model_path,           std::string)
        handle_param("--prompt",                "-p", prompt,               std::string)
        handle_param("--system",                "-s", system,               std::string)
        handle_param("--max_length",            "-l", max_length,           std::stoi)
        handle_param("--max_context_length",    "-c", max_context_length,   std::stoi)
        handle_para0("--extending",                   extending,            std::string)
        handle_param("--top_k",                 "-k", top_k,                std::stoi)
        handle_param("--top_p",                 "-q", top_p,                std::stof)
        handle_param("--temp",                  "-t", temp,                 std::stof)
        handle_param("--threads",               "-n", num_threads,          std::stoi)
        handle_para0("--seed",                        seed,                 std::stoi)
        else
            break;

        c++;
    }

    if (c < argc)
    {
        std::cerr << "Unknown arguments:";
        for (int i = c; i < argc; i++)
        {
            std::cerr << " " << argv[i];
        }
        std::cerr << std::endl;
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    return args;
}

static void init_ic() {
    ic_style_def("kbd","gray underline");
    ic_set_prompt_marker(" > ", "| ");
    ic_printf(
        "For multiline input:\n"
        "1) Use [kbd]shift-tab[/]\n"
        "2) or [kbd]ctrl-enter[/] ([kbd]option-enter[/] on macOS), or [kbd]ctrl-j[/])\n"
        "3) or type \\ like in llama.cpp.\n\n"
    );
}

static bool get_input(const std::string &prompt, std::string &line)
{
    char* input = ic_readline(prompt.c_str());
    if (input == NULL) return false;
    line.append(input);
    free(input);
    return true;
}

static inline int get_num_physical_cores()
{
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

void chat(Args &args)
{
    chatllm::Pipeline pipeline(args.model_path);

    if (args.system.size() > 0)
        pipeline.set_system_prompt(args.system);
    pipeline.model->seed(args.seed);
    if (args.max_length < 0)
        args.max_length = pipeline.model->get_max_length();
    if (args.max_length > pipeline.model->get_max_length())
        args.max_length = pipeline.model->get_max_length();

    if (args.extending == "shift")
        pipeline.set_extending_method(chatllm::Pipeline::ExtendingMethod::Shift);
    else
        pipeline.set_extending_method(chatllm::Pipeline::ExtendingMethod::Restart);

    pipeline.tokenizer->set_chat_format(args.format);

    const std::string ai_prompt   = "A.I.";
    const std::string user_prompt = "You";
    const int prompt_len = 4;

    chatllm::TextStreamer streamer(pipeline.tokenizer.get());

    chatllm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,
                                         args.top_p, args.temp, args.num_threads);

#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_WTEXT);
    // Set console code page to UTF-8 so console known how to interpret string data
    SetConsoleOutputCP(CP_UTF8);
    // Enable buffering to prevent VS from chopping up UTF-8 byte sequences
    setvbuf(stdout, nullptr, _IOFBF, 1000);
#endif

    if (args.tokenize)
    {
        auto ids = pipeline.tokenizer->encode(args.prompt);
        std::cout << "ID: ";
        for (auto x : ids)
            std::cout << x << ", ";
        std::cout << std::endl;
        return;
    }

    if (!args.interactive)
    {
        pipeline.chat({args.prompt}, gen_config, &streamer);
        return;
    }

    #define MODEL_INFO()     "You are served by " << std::left << std::setw(28) << pipeline.model->type_name() + ","
    #define SHOW_NATIVE()    if (pipeline.model->native_name().size() > 0) { std::cout << "(" << pipeline.model->native_name() << ")"; }

    const int64_t total_param_num = pipeline.model->get_param_num(false);
    const int64_t total_effective_param_num = pipeline.model->get_param_num(true);

    std::cout   << R"(    ________          __  __    __    __  ___ )"; SHOW_NATIVE(); std::cout << '\n'
                << R"(   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____  )" << '\n'
                << R"(  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                << R"( / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                << R"( \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n';
    std::cout   << MODEL_INFO()                               << R"(/_/   /_/       )" << '\n';
    if (total_param_num == total_effective_param_num)
        std::cout   << "with " << total_param_num << " (" << std::fixed << std::setprecision(1) << total_param_num / 1000000000. << "B) parameters." << '\n';
    else
        std::cout   << "with " << total_param_num << " (effect. " << std::fixed << std::setprecision(1) << total_effective_param_num / 1000000000. << "B) parameters." << '\n';

    std::cout << std::endl;

    init_ic();
    std::vector<std::string> history;
    while (1)
    {
        std::string input;
        if (!get_input(user_prompt, input))
        {
            break;
        }
        if (input.empty()) continue;

        history.emplace_back(std::move(input));
        std::cout << std::setw(prompt_len) << std::left << ai_prompt << " > " << std::flush;
        std::string output = pipeline.chat(history, gen_config, &streamer);
        history.emplace_back(std::move(output));
    }
    std::cout << "Bye\n";
}

int main(int argc, const char **argv)
{
    Args args = parse_args(argc, argv);
    if (args.num_threads <= 0)
        args.num_threads = get_num_physical_cores();

    try
    {
        chat(args);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
