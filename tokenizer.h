#pragma once

#include <string>
#include <vector>
#include <unordered_map>

struct _vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

namespace tokenizer
{

// simplified SentencePieceProcessor
class SentencePieceProcessor
{
public:
    SentencePieceProcessor() : id_unk_token(-1), token_unk_id("<?>"), piece_size(0) {}

    size_t Load(const char *buffer, int n_vocab);

    virtual int PieceToId(std::string_view piece) const;

    virtual const std::string &IdToPiece(int id) const;

    virtual int Encode(const std::string &input,
            std::vector<std::string> *pieces) const;

    // Given a UTF8 input, encodes it into a sequence of ids.
    virtual int Encode(const std::string &input,
            std::vector<int> *ids) const;

    // Given a sequence of ids, decodes it into a detokenized output.
    virtual int Decode(const std::vector<int> &ids,
            std::string *detokenized) const;

    void SetIdUnkownToken(int id) { id_unk_token = id; }

    void SetTokenUnknownId(const std::string &s) { token_unk_id = s; }

    int GetPieceSize(void) { return piece_size; }

protected:
    _vocab vocab_;
    int id_unk_token;
    std::string token_unk_id;
    int piece_size;
};

}
