#include "tokenizer.h"

#include <queue>
#include <memory>
#include <cstring>
#include <limits>

#include "unicode.h"
#include "chat.h"

using namespace tokenizer;

// copied from `llama.cpp`
struct llama_sp_symbol
{
    using index = int;
    index prev;
    index next;
    const char *text;
    size_t n;
};

//#define debug_bigram

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

#ifdef debug_bigram
    std::string text;
#endif
};

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_tokenizer
{
    llama_sp_tokenizer(const _vocab &vocab) : vocab_(vocab) {}

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
#ifdef debug_bigram
        bigram.text = text;
#endif
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

int Processor::Encode(const std::string &input, std::vector<std::string> *pieces) const
{
    std::vector<int> ids;
    Encode(input, &ids);
    for (auto id : ids)
        pieces->emplace_back(IdToPiece(id));
    return 0;
}

int Processor::PieceToId(std::string_view piece) const
{
    auto r = vocab_.token_to_id.find(std::string(piece));
    return r != vocab_.token_to_id.end() ? r->second : id_unk_token;
}

const std::string Processor::IdToPiece(int id) const
{
    if (id < 0) return token_unk_id;
    return id < (int)vocab_.id_to_token.size() ? vocab_.id_to_token[id].tok : token_unk_id;
}

int Processor::Decode(const std::vector<int> &ids, std::string *detokenized) const
{
    std::string_view view;
    for (auto id : ids)
        detokenized->append(IdToPiece(id));
    return 0;
}

static int load_vocab_list(_vocab &vocab, Reader &reader, bool has_score, bool has_type, int start_piece_id)
{
    int count = 0;
    while (true)
    {
        int len = reader.read_i32();

        if (len < 0)
        {
            break;
        }

        int id = start_piece_id + count;
        CHATLLM_CHECK((size_t)(id) < vocab.id_to_token.size()) << "too many extra tokens";

        std::string word = reader.read_string(len);

        float score = 0.0f;
        uint8_t type = token_type::NORMAL;
        if (has_score)
            reader.read_raw(&score, sizeof(score));
        if (has_type)
            reader.read_raw(&type, sizeof(type));

        vocab.token_to_id[word] = id;

        auto &tok_score = vocab.id_to_token[id];
        tok_score.tok = std::move(word);
        tok_score.score = score;
        tok_score.type = (token_type)type;

        count++;
    }

    return count;
}

static int load_vocab_merges(_vocab &vocab, Reader &reader)
{
    int count = 0;

    while (true)
    {
        int len = reader.read_i32();

        if (len < 0)
            break;

        std::string word = reader.read_string(len);

        std::string first;
        std::string second;

        const size_t pos = word.find(' ', 1);

        if (pos != std::string::npos) {
            first  = word.substr(0, pos);
            second = word.substr(pos + 1);
        }

        vocab.bpe_ranks.emplace(std::make_pair(first, second), count);

        count++;
    }

    return count;
}

static void build_special_token_cache(_vocab &vocab)
{
    for (const auto & t : vocab.token_to_id)
    {
        const auto & token = t.first;
        const auto & id    = t.second;

        // Count all non-normal tokens in the vocab while iterating
        if (vocab.id_to_token[id].type != token_type::NORMAL)
        {
            vocab.special_tokens_cache[id] = token;
        }
    }
}

size_t BPEProcessor1::Load(const char *buffer, int n_vocab)
{
    Reader reader(buffer);

    vocab_.token_to_id.clear();
    vocab_.id_to_token.resize((size_t)n_vocab + 100);

    piece_size = load_vocab_list(vocab_, reader, true, false, 0);

    vocab_.id_to_token.resize(piece_size);
    return reader.get_total_size();
}

int BPEProcessor1::Encode(const std::string &input,
                                   std::vector<int> *ids) const
{
    llama_sp_tokenizer tokenizer(vocab_);
    tokenizer.tokenize(input, *ids);
    return 0;
}

size_t BPEProcessor2::Load(const char *buffer, int n_vocab)
{
    Reader reader(buffer);

    vocab_.token_to_id.clear();
    vocab_.id_to_token.resize((size_t)n_vocab + 100);
    piece_size = load_vocab_list(vocab_, reader, false, true, 0);

    vocab_.id_to_token.resize(piece_size);
    load_vocab_merges(vocab_, reader);
    build_special_token_cache(vocab_);

    return reader.get_total_size();
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_bpe_tokenizer {
    llm_bpe_tokenizer(const _vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<_vocab::id> & output) {
        int final_prev_index = -1;
        auto word_collection = bpe_gpt2_preprocess(text);

        symbols_final.clear();

        for (auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) ::utf8_len(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (size_t i = 1; i < symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.top();
                work_queue.pop();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the fnished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte == vocab.token_to_id.end()) {
                            throw std::runtime_error("ERROR: byte not found in vocab");
                        }
                        output.push_back((*token_multibyte).second);
                    }
                } else {
                    output.push_back((*token).second);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    std::vector<std::string> bpe_gpt2_preprocess(const std::string & text) {
        std::vector<std::string> bpe_words;
        std::vector<std::string> bpe_encoded_words;

        std::string token = "";
        // GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        bool collecting_numeric = false;
        bool collecting_letter = false;
        bool collecting_special = false;
        bool collecting_whitespace_lookahead = false;
        bool collecting = false;

        std::vector<std::string> text_utf;
        text_utf.reserve(text.size());
        bpe_words.reserve(text.size());
        bpe_encoded_words.reserve(text.size());

        auto cps = codepoints_from_utf8(text);
        for (size_t i = 0; i < cps.size(); ++i)
            text_utf.emplace_back(codepoint_to_utf8(cps[i]));

        for (int i = 0; i < (int)text_utf.size(); i++) {
            const std::string & utf_char = text_utf[i];
            bool split_condition = false;
            int bytes_remain = text_utf.size() - i;
            // forward backward lookups
            const std::string & utf_char_next = (i + 1 < (int)text_utf.size()) ? text_utf[i + 1] : "";
            const std::string & utf_char_next_next = (i + 2 < (int)text_utf.size()) ? text_utf[i + 2] : "";

            // handling contractions
            if (!split_condition && bytes_remain >= 2) {
                // 's|'t|'m|'d
                if (utf_char == "\'" && (utf_char_next == "s" || utf_char_next == "t" || utf_char_next == "m" || utf_char_next == "d")) {
                    split_condition = true;
                }
                if (split_condition) {
                    if (token.size()) {
                        bpe_words.emplace_back(token); // push previous content as token
                    }
                    token = utf_char + utf_char_next;
                    bpe_words.emplace_back(token);
                    token = "";
                    i++;
                    continue;
                }
            }
            if (!split_condition && bytes_remain >= 3) {
                // 're|'ve|'ll
                if (utf_char == "\'" && (
                    (utf_char_next == "r" && utf_char_next_next == "e") ||
                    (utf_char_next == "v" && utf_char_next_next == "e") ||
                    (utf_char_next == "l" && utf_char_next_next == "l"))
                    ) {
                    split_condition = true;
                }
                if (split_condition) {
                    // current token + next token can be defined
                    if (token.size()) {
                        bpe_words.emplace_back(token); // push previous content as token
                    }
                    token = utf_char + utf_char_next + utf_char_next_next;
                    bpe_words.emplace_back(token); // the contraction
                    token = "";
                    i += 2;
                    continue;
                }
            }

            if (!split_condition && !collecting) {
                if (codepoint_type(utf_char) == CODEPOINT_TYPE_LETTER || (!token.size() && utf_char == " " && codepoint_type(utf_char_next) == CODEPOINT_TYPE_LETTER)) {
                    collecting_letter = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_DIGIT || (!token.size() && utf_char == " " && codepoint_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    collecting_numeric = true;
                    collecting = true;
                }
                else if (
                    ((codepoint_type(utf_char) != CODEPOINT_TYPE_LETTER && codepoint_type(utf_char) != CODEPOINT_TYPE_DIGIT) && (codepoint_type(utf_char) != CODEPOINT_TYPE_WHITESPACE)) ||
                    (!token.size() && utf_char == " " && codepoint_type(utf_char_next) != CODEPOINT_TYPE_LETTER && codepoint_type(utf_char_next) != CODEPOINT_TYPE_DIGIT && codepoint_type(utf_char_next) != CODEPOINT_TYPE_WHITESPACE)
                    ) {
                    collecting_special = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE && codepoint_type(utf_char_next) == CODEPOINT_TYPE_WHITESPACE) {
                    collecting_whitespace_lookahead = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE) {
                    split_condition = true;
                }
            }
            else if (!split_condition && collecting) {
                if (collecting_letter && codepoint_type(utf_char) != CODEPOINT_TYPE_LETTER) {
                    split_condition = true;
                }
                else if (collecting_numeric && codepoint_type(utf_char) != CODEPOINT_TYPE_DIGIT) {
                    split_condition = true;
                }
                else if (collecting_special && (codepoint_type(utf_char) == CODEPOINT_TYPE_LETTER || codepoint_type(utf_char) == CODEPOINT_TYPE_DIGIT || codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE)) {
                    split_condition = true;
                }
                else if (collecting_whitespace_lookahead && (codepoint_type(utf_char_next) == CODEPOINT_TYPE_LETTER || codepoint_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    split_condition = true;
                }
            }

            if (utf_char_next == "") {
                split_condition = true; // final
                token += utf_char;
            }

            if (split_condition) {
                if (token.size()) {
                    bpe_words.emplace_back(token);
                }
                token = utf_char;
                collecting = false;
                collecting_letter = false;
                collecting_numeric = false;
                collecting_special = false;
                collecting_whitespace_lookahead = false;
            }
            else {
                token += utf_char;
            }
        }

        for (std::string & word : bpe_words) {
            std::string encoded_token = "";
            for (char & c : word) {
                encoded_token += bytes_to_unicode_bpe(c);
            }
            bpe_encoded_words.emplace_back(encoded_token);
        }

        return bpe_encoded_words;
    }

    const _vocab & vocab;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

int BPEProcessor2::DoEncode(const std::string &input,
        std::vector<int> *ids) const
{
    if (input.size() < 1) return 0;

    llm_bpe_tokenizer tokenizer(vocab_);
    tokenizer.tokenize(input, *ids);
    return 0;
}

static std::string search_first_special_token(std::string &input, const _vocab &vocab, int &sp_tok_id)
{
    sp_tok_id = -1;
    auto nearest_match = std::string::npos;
    for (auto & st: vocab.special_tokens_cache)
    {
        const auto & special_id     = st.first;
        const auto & special_token  = st.second;

        auto match = input.find(special_token, 0);

        if (match < nearest_match)
        {
            nearest_match = match;
            sp_tok_id = special_id;
        }
    }

    if (sp_tok_id >= 0)
    {
        const auto & special_token  = vocab.special_tokens_cache.at(sp_tok_id);
        std::string r = input.substr(0, nearest_match);
        input = input.substr(nearest_match + special_token.size());
        return r;
    }
    else
    {
        std::string r(input);
        input = "";
        return r;
    }
}

int BPEProcessor2::Encode(const std::string &input,
        std::vector<int> *ids) const
{
    std::string text(input);
    int sp_tok_id = -1;
    while (text.size() > 0)
    {
        auto leading = search_first_special_token(text, vocab_, sp_tok_id);
        DoEncode(leading, ids);
        if (sp_tok_id < 0) break;
        ids->push_back(sp_tok_id);
    }

    return 0;
}

static std::string _decode_text(const std::string & text) {
    std::string decoded_text;
    auto unicode_sequences = codepoints_from_utf8(text);
    for (auto& unicode_sequence : unicode_sequences) {
        decoded_text += unicode_to_bytes_bpe(codepoint_to_utf8(unicode_sequence));
    }

    return decoded_text;
}

const std::string BPEProcessor2::IdToPiece(int id) const
{
    if (vocab_.is_normal_token(id))
    {
        std::string result = vocab_.id_to_token[id].tok;
        return _decode_text(result);
    }
    else if (vocab_.is_control_token(id))
    {
        return "";
    }
    else
    {
        if (ret_special_token)
            return vocab_.id_to_token[id].tok;
        else
            return "";
    }
}

struct unigram_tokenizer
{
    using index = int;

    struct best
    {
        best(int prev, float score, int tok_id = 0): prev(prev), score(score), tok_id(tok_id) {}
        index prev;
        float score;
        index tok_id;
    };

    struct unigram
    {
        unigram(const char *text, int pos, size_t n): text(text), start(pos), end(pos + n) {}

        const char *text;
        index start, end;
    };

    unigram_tokenizer(const _vocab &vocab, int tok_max_len) : vocab_(vocab), tok_max_len(tok_max_len) {}

    void tokenize(const std::string &text, std::vector<_vocab::id> &output)
    {
        size_t offs = 0;

        symbols_.emplace_back(text.c_str(), 0, 0);
        trace.push_back(best(0, 0.0f));

        while (offs < text.size())
        {
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            symbols_.emplace_back(text.c_str() + offs, offs, char_len);
            offs += char_len;
        }

        if (symbols_.size() <= 1)
            return;

        // Viterbi algorithm
        for (size_t i = 1; i < symbols_.size(); i++)
        {
            find_best(i);
        }

        // backtrace
        int prev = (int)(trace.size()) - 1;
        std::vector<int> temp;
        while (prev != 0)
        {
            auto &b = trace[prev];
            temp.push_back(b.tok_id);
            prev = b.prev;
        }
        for (int i = (int)temp.size() - 1; i >= 0; i--)
            output.push_back(temp[i]);
    }

private:
    void find_best(int pos)
    {
        int start = pos - tok_max_len;
        if (start < 0) start = 0;

        float max_prop = std::numeric_limits<float>::lowest();
        int prev = -1;
        int tok_id = -1;

        for (int i = start; i < pos; i++)
        {
            auto &b = trace[i];
            auto &sym = symbols_[i];

            const std::string text = std::string(sym.text + sym.end - sym.start, symbols_[pos].end - sym.end);

            auto token = vocab_.token_to_id.find(text);
            if (token == vocab_.token_to_id.end())
                continue;

            auto &tok = vocab_.id_to_token[token->second];

            if (b.score + tok.score > max_prop)
            {
                prev = i;
                max_prop = b.score + tok.score;
                tok_id = token->second;
            }
        }

        trace.emplace_back(prev, max_prop, tok_id);
    }

    const _vocab &vocab_;
    std::vector<best> trace;
    std::vector<unigram> symbols_;
    int tok_max_len;
};

UnigramProcessor::UnigramProcessor() : Processor::Processor(), tok_max_len(0)
{

}

size_t UnigramProcessor::Load(const char *buffer, int n_vocab)
{
    Reader reader(buffer);

    vocab_.token_to_id.clear();
    vocab_.id_to_token.resize((size_t)n_vocab + 100);

    piece_size = load_vocab_list(vocab_, reader, true, false, 0);
    vocab_.id_to_token.resize(piece_size);

    // TODO: tok_max_len should be number of characters, not bytes
    for (auto & tok : vocab_.id_to_token)
    {
        if (tok.tok.size() > tok_max_len)
            tok_max_len = tok.tok.size();
    }

    return reader.get_total_size();
}

int UnigramProcessor::Encode(const std::string &input,
        std::vector<int> *ids) const
{
    unigram_tokenizer tokenizer(vocab_, (int)tok_max_len);
    tokenizer.tokenize(input, *ids);
    return 0;
}
