#pragma once

#include <vector>
#include <string>
#include <functional>
#include <algorithm>

typedef std::vector<float> text_vector;

enum DistanceStrategy
{
    EuclideanDistance,
    MaxInnerProduct,
    InnerProduct,
    CosineSimilarity,
};

DistanceStrategy ParseDistanceStrategy(const char *s);

class CVectorStore
{
public:
    CVectorStore(DistanceStrategy vec_cmp, int emb_len,
                std::function<void (const std::string &, float *)> text_emb, const char *fn);

    CVectorStore(DistanceStrategy vec_cmp, const char *fn);
    CVectorStore(DistanceStrategy vec_cmp, const std::vector<std::string> &files);

    void ExportDB(const char *fn);

    void Query(const text_vector &vec, std::vector<int64_t> &indices, int top_n = 20);

    size_t GetSize(void);

    bool GetRecord(int64_t index, std::string &content, std::string &meta);

protected:
    void Clear();
    void FromPlainData(std::function<void (const std::string &, float *)> text_emb, const char *fn);
    void LoadDB(const char *fn);

    DistanceStrategy vec_cmp;
    int emb_len;

    std::vector<std::string> contents;
    std::vector<std::string> metadata;
    std::vector<float> embeddings;
};
