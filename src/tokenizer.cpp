#include "tokenizer.h"

#include <queue>
#include <memory>
#include <cstring>
#include <limits>
#include <regex>
#include <iostream>

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
        for (int i = 1; i < (int)symbols_.size(); ++i)
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
    Reader(DataReader *data_reader)
        : data_reader(data_reader), offset(data_reader->tell()) {}

    void read_raw(void *r, size_t size)
    {
        data_reader->read_buffer(r, size);
    }

    template <typename T> T read_basic()
    {
        return data_reader->read_basic<T>();
    }

    uint32_t read_u32(void)
    {
        return read_basic<uint32_t>();
    }

    int32_t read_i32(void)
    {
        return read_basic<int32_t>();
    }

    std::string read_string(int len)
    {
        std::vector<char> chars(len);
        data_reader->read_buffer(chars.data(), len);
        return std::string(chars.data(), len);
    }

    size_t get_total_size(void) { return data_reader->tell() - offset; }

private:
    DataReader *data_reader;
    const size_t offset;
};

int Processor::Encode(const std::string &input, std::vector<std::string> *pieces) const
{
    std::vector<int> ids;
    Encode(input, &ids);
    for (auto id : ids)
        pieces->emplace_back(IdToPiece(id));
    return 0;
}

int Processor::Encode(const std::string &input, std::vector<int> *ids) const
{
    std::string s = input;
    for (auto & p : pp)
        s = p->transform(s);

    while (true)
    {
        size_t special_pos = std::string::npos;
        int special_id = -1;
        size_t special_tok_len = 0;

        for (auto & tok : added_tokens)
        {
            size_t pos = s.find(tok.token);

            if (pos < special_pos)
            {
                special_pos = pos;
                special_id = tok.id;
                special_tok_len = tok.token.size();
            }
        }

        if (std::string::npos == special_pos)
            break;

        std::string part = s.substr(0, special_pos);
        DoEncode(part, ids);

        s = s.substr(special_pos + special_tok_len);
        ids->push_back(special_id);
    }

    DoEncode(s, ids);

    return 0;
}

int Processor::PieceToId(std::string_view piece) const
{
    auto r = vocab_.token_to_id.find(std::string(piece));
    return r != vocab_.token_to_id.end() ? r->second : id_unk_token;
}

const std::string Processor::IdToPiece(int id) const
{
    if (token_override.contains(id))
        return token_override.find(id)->second;

    if (id < 0) return token_unk_id;
    return id < (int)vocab_.id_to_token.size() ? vocab_.id_to_token[id].tok : token_unk_id;
}

void Processor::OverrideTokenDecoding(int id, const std::string &tok)
{
    token_override.emplace(id, tok);
}

void Processor::AddAddedToken(const std::string &tok, int id)
{
    OverrideTokenDecoding(id, tok);
    added_tokens.emplace_back(TokenId{tok, id});
}

int Processor::Decode(const std::vector<int> &ids, std::string *detokenized) const
{
    std::string_view view;
    for (auto id : ids)
        detokenized->append(IdToPiece(id));
    return 0;
}

void Processor::RegisterPreprocessor(TextPreprocessor *prep)
{
    pp.push_back(std::unique_ptr<TextPreprocessor>(prep));
}

static int load_vocab_list(_vocab &vocab, Reader &reader, bool has_score, bool has_type, int start_piece_id)
{
    int count = 0;
    bool byte_fallback_started = false;
    int last_byte = -1;
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
            score = reader.read_basic<float>();
        if (has_type)
            type = reader.read_basic<uint8_t>();

        bool flag = false;
        if (word.size() == 1)
        {
            int ch = (uint8_t)(word[0]);
            if (!byte_fallback_started && (ch == 0))
            {
                byte_fallback_started = true;
                last_byte = -1;
            }

            if (byte_fallback_started)
            {
                if (ch == last_byte + 1)
                {
                    last_byte = ch;
                    vocab.byte_fallback_tok_ids[last_byte] = id;
                    flag = true;
                }
                else
                {
                    byte_fallback_started = false;
                }
            }
        }

        if (!flag)
            vocab.token_to_id[word] = id;

        auto &tok_score = vocab.id_to_token[id];
        tok_score.tok = std::move(word);
        tok_score.score = score;
        tok_score.type = (token_type)type;

        count++;
    }

    for (int i = 0; i <= last_byte; i++)
    {
        std::string word(1, (char)i);
        vocab.token_to_id.insert(std::pair{word, vocab.byte_fallback_tok_ids[i]});
    }

    if (last_byte == 255)
        vocab.byte_fallback_ready = true;

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

size_t BPEProcessor1::Load(DataReader *data_reader, int n_vocab)
{
    Reader reader(data_reader);

    vocab_.token_to_id.clear();
    vocab_.id_to_token.resize((size_t)n_vocab + 100);

    piece_size = load_vocab_list(vocab_, reader, true, false, 0);

    vocab_.id_to_token.resize(piece_size);
    return reader.get_total_size();
}

int BPEProcessor1::DoEncode(const std::string &input,
                                   std::vector<int> *ids) const
{
    llama_sp_tokenizer tokenizer(vocab_);
    tokenizer.tokenize(input, *ids);
    return 0;
}

BPEProcessor2::BPEProcessor2():
    BPEProcessor2(
        // default
        {
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "\\p{N}+",
            "[0-9][0-9][0-9]",
        }
    )
{
}

BPEProcessor2::BPEProcessor2(std::vector<std::string> regex_exprs):
    Processor::Processor()
{
    this->regex_exprs = regex_exprs;
}

size_t BPEProcessor2::Load(DataReader *data_reader, int n_vocab)
{
    Reader reader(data_reader);

    vocab_.token_to_id.clear();
    vocab_.id_to_token.resize((size_t)n_vocab + 100);
    piece_size = load_vocab_list(vocab_, reader, false, true, 0);

    vocab_.id_to_token.resize(piece_size);
    load_vocab_merges(vocab_, reader);
    build_special_token_cache(vocab_);
    searcher.rebuild(vocab_.special_tokens_cache);

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

    void tokenize(std::vector<_vocab::id> & output, const std::vector<std::string> &word_collection) {
        int final_prev_index = -1;

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
            for (int i = 1; i < (int)symbols.size(); ++i) {
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
                        symbols_final[final_prev_index].next = (llm_symbol::index)symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = (int)symbols_final.size() - 1;
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

    const _vocab & vocab;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

int BPEProcessor2::DoEncode2(const std::string &input,
        std::vector<int> *ids) const
{
    if (input.size() < 1) return 0;

    std::vector<std::string> bpe_encoded_words = unicode_regex_split(input, regex_exprs);

    llm_bpe_tokenizer tokenizer(vocab_);
    tokenizer.tokenize(*ids, bpe_encoded_words);

    return 0;
}

BPEProcessor3::BPEProcessor3(std::vector<std::string> regex_exprs)
    : BPEProcessor2(regex_exprs)
{
}

int BPEProcessor3::DoEncode2(const std::string &input,
        std::vector<int> *ids) const
{
    if (input.size() < 1) return 0;

    // TODO: split by regexpr

    llama_sp_tokenizer tokenizer(vocab_);
    tokenizer.tokenize(input, *ids);
    return 0;
}

const std::string BPEProcessor3::IdToPiece(int id) const
{
    if (token_override.contains(id))
        return token_override.find(id)->second;

    if (vocab_.is_normal_token(id))
    {
        std::string result = vocab_.id_to_token[id].tok;
        return result;
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

NearestKeywordSearcher::Node *NearestKeywordSearcher::make_tree(std::vector<Item> &items, char ch, int value)
{
    Node * r = new Node();
    r->ch = ch;
    r->value = items.size() < 1 ? value : -1;

    while (true)
    {
        bool flag = false;
        char tag = 0;
        int  v   = -1;

        std::vector<Item> sub;
        for (int i = (int)items.size() - 1; i >= 0; i--)
        {
            if (items[i].s.size() < 1) continue;

            if (!flag)
            {
                flag = true;
                tag = items[i].s[0];
                v   = items[i].value;
            }
            else
            {
                if (items[i].s[0] != tag) continue;
            }

            if (items[i].s.size() > 1)
                sub.emplace_back(items[i].s.substr(1), items[i].value);

            // mark as visited
            items[i].s = "";
        }

        if (!flag) break;

        Node *child = make_tree(sub, tag, v);
        r->child.emplace_back(std::unique_ptr<Node>(child));
    }

    std::sort(r->child.begin(), r->child.end(), [](auto &p1, auto &p2) { return p1->ch <= p2->ch; });

    return r;
}

void NearestKeywordSearcher::rebuild(const std::unordered_map<_vocab::id, std::string> keywords)
{
    root.reset(nullptr);

    std::vector<Item> sub;

    for (auto & st: keywords)
    {
        sub.emplace_back(st.second, (int)st.first);
    }
    root.reset(make_tree(sub, 0, -1));
}

int NearestKeywordSearcher::match(const std::string &input, int index, Node *node, int &level) const
{
    if (node->child.size() < 1) return node->value;
    if (index >= (int)input.size()) return -1;
    const char ch = input[index];

    int low  = 0;
    int high = (int)node->child.size() - 1;
    while (high >= low)
    {
        // assuming no overflow
        int middle = (high + low) / 2;
        Node *n = node->child[middle].get();
        if (n->ch < ch)
        {
            low = middle + 1;
        }
        else if (ch < n->ch)
        {
            high = middle - 1;
        }
        else
        {
            level++;
            return match(input, index + 1, n, level);
        }
    }

    return -1;
}

std::string NearestKeywordSearcher::search(std::string &input, int &kw_id) const
{
    int index = 0;
    while (index < (int)input.size())
    {
        int len = 0;
        kw_id = match(input, index, root.get(), len);
        if (kw_id >= 0)
        {
            std::string r = input.substr(0, index);
            input = input.substr(index + len);
            return r;
        }
        index++;
    }

    std::string r(input);
    input = "";
    return r;
}

int BPEProcessor2::DoEncode(const std::string &input,
        std::vector<int> *ids) const
{
    std::string text(input);
    int sp_tok_id = -1;
    while (text.size() > 0)
    {
        auto leading = searcher.search(text, sp_tok_id);
        DoEncode2(leading, ids);
        if (sp_tok_id < 0) break;
        ids->push_back(sp_tok_id);
    }

    return 0;
}

static std::string _decode_text(const std::string & text) {
    std::string decoded_text;
    auto unicode_sequences = unicode_cpts_from_utf8(text);
    for (auto& unicode_sequence : unicode_sequences) {
        decoded_text += unicode_utf8_to_byte(unicode_cpt_to_utf8(unicode_sequence));
    }

    return decoded_text;
}

const std::string BPEProcessor2::IdToPiece(int id) const
{
    if (token_override.contains(id))
        return token_override.find(id)->second;

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
        unigram(const char *text, int pos, int n): text(text), start(pos), end(pos + n) {}

        const char *text;
        index start, end;
    };

    unigram_tokenizer(const _vocab &vocab, int tok_max_len, int unk_id) : vocab_(vocab), tok_max_len(tok_max_len), unk_id(unk_id) {}

    void tokenize(const std::string &text, std::vector<_vocab::id> &output)
    {
        size_t offs = 0;

        symbols_.emplace_back(text.c_str(), 0, 0);
        trace.push_back(best(0, 0.0f));

        while (offs < text.size())
        {
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            symbols_.emplace_back(text.c_str() + offs, (int)offs, (int)char_len);
            offs += char_len;
        }

        if (symbols_.size() <= 1)
            return;

        // Viterbi algorithm
        for (int i = 1; i < (int)symbols_.size(); i++)
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

        if (prev < 0)
        {
            auto i = pos - 1;
            auto &b = trace[i];
            auto &tok = vocab_.id_to_token[unk_id];

            prev = i;
            max_prop = b.score + tok.score;
            tok_id = unk_id;
        }

        trace.emplace_back(prev, max_prop, tok_id);
    }

    const _vocab &vocab_;
    std::vector<best> trace;
    std::vector<unigram> symbols_;
    int tok_max_len;
    int unk_id;
};

UnigramProcessor::UnigramProcessor(int unk_tok_id) : Processor::Processor(), unk_tok_id(unk_tok_id), tok_max_len(0)
{

}

size_t UnigramProcessor::Load(DataReader *data_reader, int n_vocab)
{
    Reader reader(data_reader);

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

int UnigramProcessor::DoEncode(const std::string &input,
        std::vector<int> *ids) const
{
    unigram_tokenizer tokenizer(vocab_, (int)tok_max_len, unk_tok_id);
    tokenizer.tokenize(input, *ids);
    return 0;
}

std::string TextPrepTrim::transform(const std::string &s)
{
    int size = (int)s.size();
    while ((size >= 1) && (s[size - 1] == ' '))
        size--;
    return s.substr(0, size);
}

std::string TextTrim::transform(const std::string &s0)
{
    std::string s(s0);
    int size = (int)s.size();
    while ((size >= 1) &&
            ((s[size - 1] == ' ') || (s[size - 1] == '\t') || (s[size - 1] == '\r') || (s[size - 1] == '\n'))
        )
        size--;
    s = s.substr(0, size);
    size = 0;
    while ((size < (int)s.size()) &&
            ((s[size] == ' ') || (s[size] == '\t') || (s[size] == '\r') || (s[size] == '\n'))
        )
        size++;
    return s.substr(size);
}

std::string TextPrepDeleteMultiSpaces::transform(const std::string &s)
{
    const static std::regex r(R""( {2,})"");
    return std::regex_replace(s, r, " ");
}

std::string TextPrepAddLeadingSpace::transform(const std::string &s)
{
    if (s.size() < 1) return " ";
    return s[0] == ' ' ? s : " " + s;
}

std::string TextPrepNewlineToSpaces::transform(const std::string &s)
{
    const static std::regex r(R""([\r\n]+)"");
    return std::regex_replace(s, r, " ");
}

size_t tokenizer::get_end_of_valid_utf8(const std::string &utf8, const size_t offset)
{
    size_t end = offset;
    uint32_t ch = 0;
    while (try_unicode_cpt_from_utf8(utf8, end, ch))
        ;
    return end;
}