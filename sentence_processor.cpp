#include "sentence_processor.h"

#include <queue>
#include <memory>
#include <cstring>

#include "chat.h"

// copied from `llama.cpp`
struct llama_sp_symbol
{
    using index = int;
    index prev;
    index next;
    const char *text;
    size_t n;
};

struct llama_sp_bigram
{
    struct comparator
    {
        bool operator()(llama_sp_bigram &l, llama_sp_bigram &r)
        {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_tokenizer
{
    llama_tokenizer(const _vocab &vocab) : vocab_(vocab) {}

    void tokenize(const std::string &text, std::vector<_vocab::id> &output)
    {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size())
        {
            llama_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        if (symbols_.size() == 0)
            return;

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i)
        {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty())
        {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto &left_sym = symbols_[bigram.left];
            auto &right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size)
            {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            // printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0)
            {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next)
        {
            auto &symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end())
            {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int)symbol.n; ++j)
                {
                    _vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            }
            else
            {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right)
    {
        if (left == -1 || right == -1)
        {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end())
        {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size())
        {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const _vocab &vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};

class Reader
{
public:
    Reader(const char *buffer)
        : buffer(buffer), offset(0) {}

    void read_raw(void *r, size_t size)
    {
        memcpy(r, buffer + offset, size);
        offset += size;
    }

    uint32_t read_u32(void)
    {
        uint32_t r;
        read_raw(&r, sizeof(r));
        return r;
    }

    int32_t read_i32(void)
    {
        int32_t r;
        read_raw(&r, sizeof(r));
        return r;
    }

    std::string read_string(int len)
    {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    size_t get_total_size(void) { return offset; }

private:
    const char *buffer;
    size_t offset;
};

using namespace sentencepiece;

size_t SentencePieceProcessor::Load(const char *buffer, int n_vocab)
{
    Reader reader(buffer);

    vocab_.token_to_id.clear();
    vocab_.id_to_token.resize((size_t)n_vocab + 100);

    piece_size = 0;

    while (true)
    {
        int len = reader.read_i32();

        if (len < 0)
        {
            break;
        }

        CHATLLM_CHECK((size_t)piece_size < vocab_.id_to_token.size()) << "too many extra tokens";

        std::string word = reader.read_string(len);

        float score = 0.0f;
        reader.read_raw(&score, sizeof(score));

        vocab_.token_to_id[word] = piece_size;

        auto &tok_score = vocab_.id_to_token[piece_size];
        tok_score.tok = std::move(word);
        tok_score.score = score;

        piece_size++;
    }
    vocab_.id_to_token.resize(piece_size);
    return reader.get_total_size();
}

int SentencePieceProcessor::PieceToId(std::string_view piece) const
{
    auto r = vocab_.token_to_id.find(std::string(piece));
    return r != vocab_.token_to_id.end() ? r->second : id_unk_token;
}

const std::string &SentencePieceProcessor::IdToPiece(int id) const
{
    if (id < 0) return token_unk_id;
    return id < (int)vocab_.id_to_token.size() ? vocab_.id_to_token[id].tok : token_unk_id;
}

int SentencePieceProcessor::Encode(const std::string &input,
                                   std::vector<std::string> *pieces) const
{
    std::vector<int> ids;
    Encode(input, &ids);
    for (auto id : ids)
        pieces->emplace_back(IdToPiece(id));
    return 0;
}

int SentencePieceProcessor::Encode(const std::string &input,
                                   std::vector<int> *ids) const
{
    llama_tokenizer tokenizer(vocab_);
    tokenizer.tokenize(input, *ids);
    return 0;
}

int SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                           std::string *detokenized) const
{
    std::string_view view;
    for (auto id : ids)
        detokenized->append(IdToPiece(id));
    return 0;
}
