#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

namespace tokenizer
{

enum token_type
{
    UNDEFINED    = 0,
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

struct _vocab
{
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
        token_type type;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;

    std::unordered_map<id, token> special_tokens_cache;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    int byte_fallback_tok_ids[256];
    bool byte_fallback_ready;

    int find_bpe_rank(std::string token_left, std::string token_right) const
    {
        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }

    bool is_token_of_type(id id, token_type t) const
    {
        if ((id < 0) || (id >= (int)id_to_token.size())) return false;
        return t == id_to_token[id].type;
    }

    bool is_normal_token(id id) const
    {
        return is_token_of_type(id, token_type::NORMAL);
    }

    bool is_control_token(id id) const
    {
        return is_token_of_type(id, token_type::CONTROL);
    }
};

class TextPreprocessor
{
public:
    virtual ~TextPreprocessor() {}
    virtual std::string transform(const std::string &s) = 0;
};

class TextPrepTrim : public TextPreprocessor
{
public:
    std::string transform(const std::string &s) override;
};

class TextTrim : public TextPreprocessor
{
public:
    std::string transform(const std::string &s) override;
};

class TextPrepDeleteMultiSpaces : public TextPreprocessor
{
public:
    std::string transform(const std::string &s) override;
};

class TextPrepNewlineToSpaces : public TextPreprocessor
{
public:
    std::string transform(const std::string &s) override;
};

class TextPrepAddLeadingSpace : public TextPreprocessor
{
public:
    std::string transform(const std::string &s) override;
};

class DataReader
{
public:
    virtual int64_t tell() = 0;
    virtual void seek(int64_t offset, int whence) = 0;
    virtual int64_t size(void) const { return _size; }

    virtual size_t read_buffer(void *output, size_t len) = 0;

    template <typename T> T read_basic()
    {
        T obj;
        read_buffer(&obj, sizeof(obj));
        return obj;
    }
protected:
    int64_t _size;
};

class Processor
{
public:
    struct TokenId
    {
        std::string token;
        int id;
    };

    Processor() : piece_size(0), id_unk_token(-1), token_unk_id("<?>"), ret_special_token(false)
    {
        vocab_.byte_fallback_ready = false;
    }

    virtual size_t Load(DataReader *data_reader, int n_vocab) = 0;

    virtual int PieceToId(std::string_view piece) const;

    virtual const std::string IdToPiece(int id) const;

    virtual int Encode(const std::string &input,
            std::vector<std::string> *pieces) const;

    // Given a UTF8 input, encodes it into a sequence of ids.
    virtual int Encode(const std::string &input,
            std::vector<int> *ids) const;

    // Given a sequence of ids, decodes it into a detokenized output.
    virtual int Decode(const std::vector<int> &ids,
            std::string *detokenized) const;

    int GetPieceSize(void) const { return piece_size; }

    void SetIdUnkownToken(int id) { id_unk_token = id; }

    void SetTokenUnknownId(const std::string &s) { token_unk_id = s; }

    void EnableReturnSpecialToken(bool en) { ret_special_token = en; }

    void RegisterPreprocessor(TextPreprocessor* prep);

    void OverrideTokenDecoding(int id, const std::string &tok);

    void AddAddedToken(const std::string &tok, int id);

protected:
    virtual int DoEncode(const std::string &input, std::vector<int> *ids) const = 0;

protected:
    _vocab vocab_;
    int piece_size;
    int id_unk_token;
    std::string token_unk_id;
    bool ret_special_token;
    std::vector<std::unique_ptr<TextPreprocessor>> pp;
    std::map<int, std::string> token_override;
    std::vector<TokenId> added_tokens;
};

class BPEProcessor1: public Processor
{
public:
    BPEProcessor1() : Processor::Processor()  {}

    size_t Load(DataReader *data_reader, int n_vocab) override;

protected:
    int DoEncode(const std::string &input,
            std::vector<int> *ids) const override;
};

class NearestKeywordSearcher
{
public:
    void rebuild(const std::unordered_map<_vocab::id, std::string> keywords);

    std::string search(std::string &input, int &kw_id) const;

protected:

    struct Item
    {
        std::string s;
        int value;

        Item(const std::string &s, const int value) : s(s), value(value) {}
    };
    struct Node
    {
        char ch;
        int value;
        std::vector<std::unique_ptr<Node>> child;
    };

    Node *make_tree(std::vector<Item> &items, char ch, int value);

    int match(const std::string &input, int index, Node *node, int &level) const;

    std::unique_ptr<Node> root;
};

class BPEProcessor2: public Processor
{
public:
    BPEProcessor2();

    BPEProcessor2(std::vector<std::string> regex_exprs);

    size_t Load(DataReader *data_reader, int n_vocab) override;

    const std::string IdToPiece(int id) const override;

protected:
    int DoEncode(const std::string &input,
            std::vector<int> *ids) const override;

    virtual int DoEncode2(const std::string &input,
            std::vector<int> *ids) const;

    std::vector<std::string> regex_exprs;
    NearestKeywordSearcher searcher;
};

class BPEProcessor3: public BPEProcessor2
{
public:
    BPEProcessor3(std::vector<std::string> regex_exprs);

    const std::string IdToPiece(int id) const override;
protected:
    int DoEncode2(const std::string &input,
            std::vector<int> *ids) const override;
};

class UnigramProcessor: public Processor
{
public:
    UnigramProcessor(int unk_tok_id);

    size_t Load(DataReader *data_reader, int n_vocab) override;

    int unk_tok_id;

private:
    int DoEncode(const std::string &input,
            std::vector<int> *ids) const override;

private:
    size_t tok_max_len;
};

size_t get_end_of_valid_utf8(const std::string &utf8, const size_t offset);
}
