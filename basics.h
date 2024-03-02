#pragma once

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <sstream>

namespace chatllm
{
    class LogMessageFatal
    {
    public:
        LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
        [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
        std::ostringstream &stream() { return oss_; }

    private:
        std::ostringstream oss_;
    };

#define CHATLLM_THROW ::chatllm::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHATLLM_CHECK(cond) \
    if (!(cond))            \
    CHATLLM_THROW << "check failed (" #cond ") "

    template <class T> void ordering(const std::vector<T> &lst, std::vector<int> &order, bool descending = false)
    {
        order.resize(lst.size());
        for (size_t i = 0; i < lst.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&lst, descending](int a, int b)
        {
            return descending ? lst[a] > lst[b] : lst[a] < lst[b];
        });
    }
}
