#pragma once

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <sstream>
#include <map>

#if defined(_WIN32)
#define strcasecmp stricmp
#endif

namespace chatllm
{
    class LogMessageFatal
    {
    public:
        LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
        ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
        std::ostringstream &stream() { return oss_; }

    private:
        std::ostringstream oss_;
    };

#define CHATLLM_THROW ::chatllm::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHATLLM_CHECK(cond) \
    if (!(cond))            \
    CHATLLM_THROW << "check failed (" #cond ") "

    template <class T> void ordering(const std::vector<T> &lst, std::vector<size_t> &order, bool descending = false)
    {
        order.resize(lst.size());
        for (size_t i = 0; i < lst.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&lst, descending](auto a, auto b)
        {
            return descending ? lst[a] > lst[b] : lst[a] < lst[b];
        });
    }

    template <class T, class V> void ordering(const std::vector<T> &lst, std::vector<size_t> &order, std::function<V(const T &)> by, bool descending = false)
    {
        order.resize(lst.size());
        for (size_t i = 0; i < lst.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&lst, descending, by](auto a, auto b)
        {
            return descending ? by(lst[a]) > by(lst[b]) : by(lst[a]) < by(lst[b]);
        });
    }
}

namespace base64
{
    std::string encode(const void *raw_data, int data_len, bool for_url);
    std::vector<uint8_t> decode(const char *encoded_string);
    void decode_to_utf8(const char *encoded_string, std::string &s);
    std::string encode_utf8(const std::string &s);
}

namespace utils
{
    std::string trim(const std::string& str);

    std::string join(const std::vector<std::string>& vec, const std::string& sep);

    std::string replace_all(const std::string& str, const std::string& from, const std::string& to);

    bool starts_with(const std::string& value, const std::string& starting);
    bool ends_with(const std::string& value, const std::string& ending);

    // for (i = start; i < end; i++) { func(i); }
    void parallel_for(int64_t start, int64_t end, std::function<void(int64_t)> func, int num_threads = 0);

    std::string load_file(const char *fn);

    std::string num2words(int value);

    std::string sec2hms(double seconds, bool hour_2digits = false, bool show_ms = false);
    std::string sec2ms(double seconds, bool show_ms = false);

    // create a unique temp file name (full path)
    std::string tmpname(void);

    bool is_same_command_option(const char *a, const char *b);
    bool is_same_command_option(const std::string &a, const std::string &b);

    int         get_opt(const std::map<std::string, std::string> &options, const char *key, int def);
    double      get_opt(const std::map<std::string, std::string> &options, const char *key, double def);
    bool        get_opt(const std::map<std::string, std::string> &options, const char *key, const bool def);
    std::string get_opt(const std::map<std::string, std::string> &options, const char *key, const char *def);
    std::string get_opt(const std::map<std::string, std::string> &options, const char *key, const std::string &def);

    std::string now(const char *fmt = "%Y-%m-%d");

    int parse_int_lists(std::vector<int> &values, const std::string &s, int num_elements);

    //#define TIME_STAMP (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())
}

#if defined(_MSC_VER)
#define popen  _popen
#define pclose _pclose
#endif