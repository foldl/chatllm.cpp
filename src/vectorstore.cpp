#include "vectorstore.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <thread>
#include <regex>

#include "basics.h"

static const char VS_FILE_HEADER[] = "CHATLLMVS";

struct file_header
{
    char magic[9];
    size_t emb_len;
    size_t size;
};

static float vector_euclidean_distance(const float *a, const float *b, int len)
{
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        float t = a[i] - b[i];
        sum += t * t;
    }
    return sqrtf(sum);
}

static float vector_inner_product(const float *a, const float *b, int len)
{
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

static float vector_cos_similarity(const float *a, const float *b, int len)
{
    float inner_sum = 0.0;
    float norm_a = 0.0;
    float norm_b = 0.0;
    for (int i = 0; i < len; i++)
    {
        inner_sum += a[i] * b[i];
        norm_a    += a[i] * a[i];
        norm_b    += b[i] * b[i];
    }
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    return inner_sum / (norm_a * norm_b + 1e-6f);
}

static float vector_measure(DistanceStrategy ds, const float *a, const float *b, int len)
{
    switch (ds)
    {
    case EuclideanDistance:
        return vector_euclidean_distance(a, b, len);
    case MaxInnerProduct:
    case InnerProduct:
        return vector_inner_product(a, b, len);
    case CosineSimilarity:
        return vector_cos_similarity(a, b, len);
    default:
        CHATLLM_CHECK(false) << "not implemented: " << ds << std::endl;
        return 0.0;
    }
}

bool is_dist_strategy_max_best(DistanceStrategy ds)
{
    switch (ds)
    {
    case MaxInnerProduct:
        return true;
    default:
        return false;
    }
}

CVectorStore::CVectorStore(DistanceStrategy vec_cmp, int emb_len,
    std::function<void (const std::string &, float *)> text_emb, const char *fn)
    : vec_cmp(vec_cmp), emb_len(emb_len)
{
    FromPlainData(text_emb, fn);
    embeddings.resize(GetSize() * emb_len);
    printf("ingesting...\n");
    for (size_t i = 0; i < GetSize(); i++)
    {
        text_emb(contents[i], embeddings.data() + i * emb_len);
        printf("%8zu / %8zu\r", i, GetSize());
        fflush(stdout);
    }
    printf("\ndone\n");
}

CVectorStore::CVectorStore(DistanceStrategy vec_cmp, const char *fn)
    : vec_cmp(vec_cmp), emb_len(0)
{
    LoadDB(fn);
}

CVectorStore::CVectorStore(DistanceStrategy vec_cmp, const std::vector<std::string> &files)
    : vec_cmp(vec_cmp), emb_len(0)
{
    for (auto fn : files)
        LoadDB(fn.c_str());
}

namespace base64
{
    static unsigned int pos_of_char(const unsigned char chr) {
        if (chr >= 'A' && chr <= 'Z') return chr - 'A';
        else if (chr >= 'a' && chr <= 'z') return chr - 'a' + ('Z' - 'A') + 1;
        else if (chr >= '0' && chr <= '9') return chr - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
        else if (chr == '+' || chr == '-') return 62; // Be liberal with input and accept both url ('-') and non-url ('+') base 64 characters (
        else if (chr == '/' || chr == '_') return 63; // Ditto for '/' and '_'
        else
            return 0;
    }

    std::string encode(const void *raw_data, int data_len, bool for_url)
    {
        const uint8_t *data = (const uint8_t *)raw_data;
        int len = (data_len + 2) / 3 * 4;
        int i, j;
        const char padding = for_url ? '.' : '=';
        std::string r;
        r.resize(len);
        char *res = r.data();

        const char *base64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        for(i = 0, j = 0; i < len - 2; j += 3, i += 4)
        {
            res[i + 0] = base64_table[data[j] >> 2];
            res[i + 1] = base64_table[(data[j] & 0x3) << 4 | (data[j + 1] >> 4)];
            res[i + 2] = base64_table[(data[j + 1] & 0xf) << 2 | (data[j + 2] >> 6)];
            res[i + 3] = base64_table[data[j + 2] & 0x3f];
        }

        switch (data_len % 3)
        {
            case 1:
                res[i - 2] = padding;
                res[i - 1] = padding;
                break;
            case 2:
                res[i - 1] = padding;
                break;
        }

        return r;
    }

    std::vector<uint8_t> decode(const char *encoded_string) {
        std::vector<uint8_t> ret;
        size_t length_of_string = strlen(encoded_string);
        size_t pos = 0;

        size_t approx_length_of_decoded_string = length_of_string / 4 * 3;
        ret.reserve(approx_length_of_decoded_string);

        while (pos + 1 < length_of_string) {
            size_t pos_of_char_1 = pos_of_char(encoded_string[pos + 1]);

            ret.push_back(static_cast<std::string::value_type>(((pos_of_char(encoded_string[pos + 0])) << 2) + ((pos_of_char_1 & 0x30) >> 4)));

            if ((pos + 2 < length_of_string) &&  // Check for data that is not padded with equal signs (which is allowed by RFC 2045)
                encoded_string[pos + 2] != '=' &&
                encoded_string[pos + 2] != '.'            // accept URL-safe base 64 strings, too, so check for '.' also.
                )
            {
                unsigned int pos_of_char_2 = pos_of_char(encoded_string[pos + 2]);
                ret.push_back(static_cast<std::string::value_type>(((pos_of_char_1 & 0x0f) << 4) + ((pos_of_char_2 & 0x3c) >> 2)));

                if ((pos + 3 < length_of_string) &&
                    encoded_string[pos + 3] != '=' &&
                    encoded_string[pos + 3] != '.'
                    )
                {
                    ret.push_back(static_cast<std::string::value_type>(((pos_of_char_2 & 0x03) << 6) + pos_of_char(encoded_string[pos + 3])));
                }
            }

            pos += 4;
        }

        return ret;
    }

    void decode_to_utf8(const char *encoded_string, std::string &s)
    {
        std::vector<uint8_t> content = base64::decode(encoded_string);
        s.resize(content.size());
        memcpy(s.data(), content.data(), s.size());
    }

    std::string encode_utf8(const std::string &s)
    {
        return encode(s.c_str(), (int)s.size(), false);
    }
}

void CVectorStore::FromPlainData(std::function<void (const std::string &, float *)> text_emb, const char *fn)
{
    std::ifstream f(fn);
    std::string lines[2];
    int counter = 0;

    if (f.is_open())
    {
        while (std::getline(f, lines[counter++]))
        {
            if (counter == 2)
            {
                std::string c;
                std::string i;

                base64::decode_to_utf8(lines[0].c_str(), c);
                base64::decode_to_utf8(lines[1].c_str(), i);

                contents.push_back(c);
                metadata.push_back(i);

                counter = 0;
            }
        }
    }
    f.close();
}

static void write_string(FILE *f, const std::string s)
{
    uint32_t len = (uint32_t)s.size();
    fwrite(&len, sizeof(len), 1, f);
    fwrite(s.data(), 1, len, f);
}

static bool read_string(FILE *f, std::string &s)
{
    uint32_t len = 0;
    if (fread(&len, sizeof(len), 1, f) != 1) return false;
    s.resize(len);
    return fread(s.data(), 1, len, f) == len;
}

void CVectorStore::LoadDB(const char *fn)
{
    bool flag = false;
    size_t old_size = contents.size();
    FILE *f = fopen(fn, "rb");
    file_header header;

    CHATLLM_CHECK(f != nullptr) << "can not open db file: " << fn;

    if (fread(&header, sizeof(header), 1, f) < 1)
        goto cleanup;

    if (memcmp(header.magic, VS_FILE_HEADER, sizeof(header.magic)))
        goto cleanup;

    if (emb_len == 0)
        emb_len = (int)header.emb_len;

    if (emb_len != (int)header.emb_len)
        goto cleanup;

    contents.reserve(old_size + header.size);
    metadata.reserve(old_size + header.size);
    for (size_t i = 0; i < header.size; i++)
    {
        std::string c;
        std::string m;
        if (!read_string(f, c)) goto cleanup;
        if (!read_string(f, m)) goto cleanup;
        contents.push_back(c);
        metadata.push_back(m);
    }

    embeddings.resize((old_size + header.size) * emb_len);
    if (fread(embeddings.data() + old_size * emb_len, emb_len * sizeof(float), header.size, f) != header.size)
        goto cleanup;

    flag = true;

cleanup:
    fclose(f);

    CHATLLM_CHECK(flag) << "LoadDB failed";
}

void CVectorStore::ExportDB(const char *fn)
{
    FILE *f = fopen(fn, "wb");

    file_header header =
    {
        .magic = 0,
        .emb_len = (size_t)emb_len,
        .size = GetSize(),
    };
    memcpy(header.magic, VS_FILE_HEADER, sizeof(header.magic));
    fwrite(&header, sizeof(header), 1, f);
    for (size_t i = 0; i < GetSize(); i++)
    {
        write_string(f, contents[i]);
        write_string(f, metadata[i]);
    }

    fwrite(embeddings.data(), sizeof(float), GetSize() * emb_len, f);

    fclose(f);
}

// TODO: use GGML to accelerate.
void CVectorStore::Query(const text_vector &vec, std::vector<int64_t> &indices, int top_n)
{
    std::vector<float> scores;
    const float *emb = embeddings.data();

    CHATLLM_CHECK(vec.size() == (size_t)emb_len) << "embedding length must match: " << vec.size() << " vs " << emb_len;

    for (size_t i = 0; i < GetSize(); i++, emb += emb_len)
        scores.push_back(vector_measure(vec_cmp, vec.data(), emb, emb_len));

    std::vector<size_t> order;
    chatllm::ordering(scores, order, is_dist_strategy_max_best(vec_cmp));

    if (top_n > (int)order.size()) top_n = (int)order.size();

    for (int i = 0; i < top_n; i++)
        indices.push_back(order[i]);
}

size_t CVectorStore::GetSize(void)
{
    return contents.size();
}

bool CVectorStore::GetRecord(int64_t index, std::string &content, std::string &meta)
{
    if (index < 0) return false;
    if ((size_t)index >= GetSize()) return false;
    content = contents[index];
    meta = metadata[index];
    return true;
}

DistanceStrategy ParseDistanceStrategy(const char *s)
{
    #define match_item(item) else if (strcasecmp(s, #item) == 0) return DistanceStrategy::item

    if (false) ;
    match_item(EuclideanDistance);
    match_item(MaxInnerProduct);
    match_item(InnerProduct);
    match_item(CosineSimilarity);
    else return DistanceStrategy::EuclideanDistance;
}

namespace utils
{
    std::string trim(const std::string& str)
    {
        size_t first = str.find_first_not_of(" \t\n\r\f\v");
        if (first == std::string::npos) {
            return "";
        }
        size_t last = str.find_last_not_of(" \t\n\r\f\v");
        return str.substr(first, last - first + 1);
    }

    std::string join(const std::vector<std::string>& vec, const std::string& sep)
    {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0) result += sep;
            result += vec[i];
        }
        return result;
    }

    std::string replace_all(const std::string& str0, const std::string& from, const std::string& to)
    {
        if (from.empty()) return str0;

        std::string str(str0);
        size_t start_pos = 0;
        while((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
        return str;
    }

    bool starts_with(const std::string& value, const std::string& starting)
    {
        if (starting.size() > value.size()) return false;
        return value.substr(0, starting.size()) == starting;
    }

    bool ends_with(const std::string& value, const std::string& ending)
    {
        if (ending.size() > value.size()) return false;
        return value.substr(value.size() - ending.size()) == ending;
    }

    void parallel_for(int64_t start, int64_t end, std::function<void(int64_t)> func, int num_threads)
    {
        // Determine number of threads to use
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) {
                num_threads = 4; // fallback if hardware_concurrency() fails
            }
        }

        // Calculate chunk size for each thread
        const int64_t range = end - start;
        if (range <= 0) return;
        const int64_t chunk_size = (range + num_threads - 1) / num_threads; // round up

        // Vector to store thread objects
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            const int64_t thread_start = start + i * chunk_size;
            const int64_t thread_end = std::min(thread_start + chunk_size, end);

            // Only create thread if there's work to do
            if (thread_start < thread_end) {
                threads.emplace_back([=]() {
                    for (int64_t j = thread_start; j < thread_end; ++j) {
                        func(j);
                    }
                });
            }
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::string load_file(const char *fn)
    {
        std::ifstream file(fn);
        if (!file.is_open()) return "";

        std::stringstream buffer;
        buffer << file.rdbuf();

        return buffer.str();
    }

    std::string num2words(int value)
    {
        if (value < 0)
            return "minus " + num2words(-value);

        if (value == 0)
            return "zero";

        const static std::vector<std::string> units = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
        const static std::vector<std::string> teens = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
        const static std::vector<std::string> tens = {"", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

        std::string result;

        if (value >= 1000000000) {
            int billion = value / 1000000000;
            result += num2words(billion) + " billion";
            value %= 1000000000;
            if (value > 0) {
                result += ", ";
            }
        }

        if (value >= 1000000) {
            int million = value / 1000000;
            result += num2words(million) + " million";
            value %= 1000000;
            if (value > 0) {
                result += ", ";
            }
        }

        if (value >= 1000) {
            int thousand = value / 1000;
            result += num2words(thousand) + " thousand";
            value %= 1000;
            if (value > 0) {
                result += ", ";
            }
        }

        if (value >= 100) {
            int hundred = value / 100;
            result += units[hundred] + " hundred";
            value %= 100;
            if (value > 0) {
                result += " and ";
            }
        }

        if (value >= 20)
        {
            int ten = value / 10;
            result += tens[ten];
            value %= 10;

            if (value > 0)
            {
                result += "-" + units[value];
            }
        }
        else if (value >= 10)
        {
            result += teens[value - 10];
        }
        else if (value >= 1)
        {
            result += units[value];
        }

        return result;
    }

    std::string sec2hms(double seconds, bool hour_2digits, bool show_ms)
    {
        int sec = (int)seconds;
        int ms  = (int)((seconds - sec) * 1000000);
        int min = sec / 60;
        int hh  = min / 60;
        sec %= 60;
        min %= 60;
        char s[100];
        if (hour_2digits)
        {
            if (show_ms)
                sprintf(s, "%02d:%02d:%02d.%06d", hh, min, sec, ms);
            else
                sprintf(s, "%02d:%02d:%02d", hh, min, sec);
        }
        else
        {
            if (show_ms)
                sprintf(s, "%d:%02d:%02d.%06d", hh, min, sec, ms);
            else
                sprintf(s, "%d:%02d:%02d", hh, min, sec);
        }
        return s;
    }

    std::string sec2ms(double seconds, bool show_ms)
    {
        int sec = (int)seconds;
        int ms  = (int)((seconds - sec) * 1000000);
        int min = sec / 60;
        sec %= 60;
        char s[100];
        if (show_ms)
            sprintf(s, "%02d:%02d.%06d", min, sec, ms);
        else
            sprintf(s, "%02d:%02d", min, sec);
        return s;
    }

    std::string tmpname(void)
    {
        // Note: theoretically unsafe!
        return std::tmpnam(nullptr);
    }

    bool is_same_command_option(const char *a, const char *b)
    {
        while (*a && *b)
        {
            char ca = *a;
            char cb = *b;
            if (ca == '-') ca = '_';
            if (cb == '-') cb = '_';
            if (ca != cb)
                return false;
            a++;
            b++;
        }

        return *a == *b;
    }

    bool is_same_command_option(const std::string &a, const std::string &b)
    {
        return is_same_command_option(a.c_str(), b.c_str());
    }

    int         get_opt(const std::map<std::string, std::string> &options, const char *key, int def)
    {
        auto s = get_opt(options, key, "");
        return s == "" ? def : std::atoi(s.c_str());
    }

    double      get_opt(const std::map<std::string, std::string> &options, const char *key, double def)
    {
        auto s = get_opt(options, key, "");
        return s == "" ? def : std::atof(s.c_str());
    }

    bool        get_opt(const std::map<std::string, std::string> &options, const char *key, const bool def)
    {
        auto s = get_opt(options, key, "");
        if (s == "") return def;
        if (s == "0") return false;
        if (s == "false") return false;
        return true;
    }

    std::string get_opt(const std::map<std::string, std::string> &options, const char *key, const std::string &def)
    {
        return get_opt(options, key, def.c_str());
    }

    std::string get_opt(const std::map<std::string, std::string> &options, const char *key, const char *def)
    {
        for (auto k : options)
        {
            if (is_same_command_option(k.first, key)) return k.second;
        }
        return def;
    }

    std::string now(const char *fmt)
    {
        std::time_t now = std::time(nullptr);
        std::tm* timeinfo = std::localtime(&now);
        char buffer[200];
        std::strftime(buffer, sizeof(buffer), fmt, timeinfo);
        return std::move(std::string(buffer));
    }

    static void parse_slice(std::vector<int> &values, const std::string &s, int num_elements)
    {
        int spec[3] = {0, num_elements, 1};
        int index = 0;
        std::string t(s);
        if (t.size() > 0) index = 1;

        while ((t.size() > 0) && (index <= 3))
        {
            size_t pos = t.find_first_of(':');
            std::string part = t.substr(0, pos);
            if (part.size() > 0)
                spec[index - 1] = atoi(part.c_str());
            if (pos == std::string::npos) break;
            index++;
            t = t.substr(pos + 1);
        }

        if (index < 1) return;

        if (index == 1)
        {
            values.push_back(spec[0]);
            return;
        }

        if (spec[2] == 0) return;
        if (spec[0] < 0) spec[0] += num_elements;
        if (spec[1] < 0) spec[1] += num_elements;

        if (spec[2] > 0)
        {
            for (int i = spec[0]; i < spec[1]; i += spec[2])
            {
                values.push_back(i);
            }
        }
        else
        {
            for (int i = spec[0]; i > spec[1]; i += spec[2])
                values.push_back(i);
        }
    }

    int parse_int_lists(std::vector<int> &values, const std::string &s, int num_elements)
    {
        const static std::regex r(R""([\r\n]+)"");
        std::string t(s);
        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(',');
            parse_slice(values, t.substr(0, pos), num_elements);
            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }
        return 0;
    }
}