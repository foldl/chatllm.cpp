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

#if defined(_WIN32)
#define strcasecmp stricmp
#endif

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