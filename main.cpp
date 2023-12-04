#include "chat.h"
#include <iomanip>
#include <iostream>
#include <thread>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstring>

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
    int max_length = 2048;
    int max_context_length = 512;
    bool interactive = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.7;
    int num_threads = 0;
};

void usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        model path (default: chatllm-ggml.bin)\n"
              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
              << "  -s, --system SYSTEM     system prompt (instruction) (default: model specific)"
              << "  -i, --interactive       run in interactive mode\n"
              << "  -l, --max_length N      max total length including prompt and output (default: 2048)\n"
              << "  -c, --max_context_length N\n"
              << "                          max context length (default: 512)\n"
              << "  --top_k N               top-k sampling (default: 0)\n"
              << "  --top_p N               top-p sampling (default: 0.7)\n"
              << "  --temp N                temperature (default: 0.95)\n"
              << "  -t, --threads N         number of threads for inference\n";
}

static Args parse_args(int argc, const char **argv)
{
    Args args;

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
        handle_param("--model",                 "-m", model_path,           std::string)
        handle_param("--prompt",                "-p", prompt,               std::string)
        handle_param("--system",                "-s", system,               std::string)
        handle_param("--max_length",            "-l", max_length,           std::stoi)
        handle_param("--max_context_length",    "-c", max_context_length,   std::stoi)
        handle_param("--top_k",                 "-k", top_k,                std::stoi)
        handle_param("--top_p",                 "-q", top_p,                std::stof)
        handle_param("--temp",                  "-t", temp,                 std::stof)
        handle_param("--threads",               "-n", num_threads,          std::stoi)
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

static bool get_utf8_line(std::string &line)
{
    std::wstring prompt;
    std::wcin >> prompt;
    for (auto wc : prompt)
        append_utf8(wc, line);
    return true;
}
#else
static bool get_utf8_line(std::string &line)
{
    return !!std::getline(std::cin, line);
}
#endif

static inline int get_num_physical_cores()
{
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

void chat(const Args &args)
{
    chatllm::Pipeline pipeline(args.model_path);
    std::string model_name = pipeline.model->type_name();
    if (args.system.size() > 0)
        pipeline.set_system_prompt(args.system);

    int prompt_len = model_name.length();
    if (prompt_len < 6) prompt_len = 6;

    chatllm::TextStreamer streamer(pipeline.tokenizer.get());

    chatllm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,
                                         args.top_p, args.temp, args.num_threads);

#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    if (!args.interactive)
    {
        pipeline.chat({args.prompt}, gen_config, &streamer);
        return;
    }

    std::cout << R"(    ________          __  __    __    __  ___                 )" << '\n'
                << R"(   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____  )" << '\n'
                << R"(  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                << R"( / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                << R"( \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n'
                << R"(                                              /_/   /_/       )" << '\n';
    std::cout << "You are served by " << model_name << std::endl << std::endl;

    std::vector<std::string> history;
    while (1)
    {
        std::cout << std::setw(prompt_len) << std::left << "Prompt"
                    << " > " << std::flush;
        std::string input;
        if (!get_utf8_line(input))
        {
            std::cout << "FAILED to read line." << std::endl;
            break;
        }
        if (input.empty())
        {
            continue;
        }
        history.emplace_back(std::move(input));
        std::cout << std::setw(prompt_len) << model_name << " > ";
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
