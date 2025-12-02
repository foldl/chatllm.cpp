#include "chat.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <regex>
#include <string>
#include <functional>

#include <sys/stat.h>
#include <thread>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <stdio.h>
#endif

namespace chatllm
{

    std::string trim(std::string str, const char *spaces)
    {
        str.erase(str.find_last_not_of(spaces) + 1);
        str.erase(0, str.find_first_not_of(spaces));
        return str;
    }

    ContentPiece::Type ContentPiece::type_parse(const char *s)
    {
        if (strcmp(s, "image") == 0)
            return ContentPiece::Type::Image;
        if (strcmp(s, "audio") == 0)
            return ContentPiece::Type::Audio;
        if (strcmp(s, "video") == 0)
            return ContentPiece::Type::Video;
        return ContentPiece::Type::Text;
    }

    Content::Content(Messages *owner, const std::string &content) : owner(owner)
    {
        push_back(content);
    }

    Content::Content(Messages *owner, const Content &c) : owner(owner)
    {
        for (const auto &p : c.pieces)
        {
            pieces.push_back(p);
        }
    }

    void Content::push_back(const std::string &content)
    {
        if (owner->get_opening().empty() || owner->get_closing().empty())
        {
            push_back(content, ContentPiece::Type::Text);
            return;
        }

        push_back("", ContentPiece::Type::Text);

        size_t start = 0;
        while (start < content.size())
        {
            size_t pos = content.find(owner->get_opening(), start);
            if (pos == std::string::npos)
            {
                push_back(content.substr(start), ContentPiece::Type::Text);
                break;
            }

            size_t end = content.find(owner->get_closing(), pos + owner->get_opening().size());
            if (end == std::string::npos)
            {
                push_back(content.substr(start), ContentPiece::Type::Text);
                break;
            }

            if (pos > start)
            {
                push_back(content.substr(start, pos - start), ContentPiece::Type::Text);
            }

            std::string inner = content.substr(pos + owner->get_opening().size(), end - (pos + owner->get_opening().size()));
            start = end + owner->get_closing().size();

            pos = inner.find(':');
            if (pos != std::string::npos)
            {
                auto tag  = inner.substr(0, pos);
                auto path = inner.substr(pos + 1);
                // remove quotes
                if (path.size() >= 2)
                {
                    if ((path[0] == '"') && (path.back() == '"'))
                    {
                        path = path.substr(1, path.size() - 2);
                    }
                }

                if (path.size() == 0) continue;

                if (tag == "image")
                {
                    push_back(path, ContentPiece::Type::Image);
                }
                else if (tag == "video")
                {
                    push_back(path, ContentPiece::Type::Video);
                }
                else if (tag == "audio")
                {
                    push_back(path, ContentPiece::Type::Audio);
                }
                else
                {
                    CHATLLM_CHECK(false) << "Unknown multi-media tag: " << tag;
                }
            }
        }
    }

    void Content::push_back(const ContentPiece &piece)
    {
        pieces.push_back(piece);
    }

    void Content::push_back(const Content &content)
    {
        for (const auto &piece : content.pieces)
            pieces.push_back(piece);
    }

    void Content::push_back(const std::string &s, ContentPiece::Type type)
    {
        if (pieces.size() > 0)
        {
            auto &last = pieces.back();
            if ((last.type == type) && (type == ContentPiece::Type::Text))
            {
                last.content += s;
                return;
            }
        }
        pieces.emplace_back(ContentPiece(s, type));
    }

    Content & Content::operator =(const Content & right)
    {
        pieces.clear();
        for (const auto &piece : right.pieces)
        {
            pieces.push_back(piece);
        }
        return *this;
    }

    Content & Content::operator =(const std::string & right)
    {
        pieces.clear();
        push_back(right);
        return *this;
    }

    std::string Content::to_string(void) const
    {
        std::string result;
        for (const auto &piece : pieces)
        {
            switch (piece.type)
            {
            case ContentPiece::Type::Text:
                result += piece.content;
                break;
            case ContentPiece::Type::Image:
                result += owner->get_opening() + "image:" + piece.content + owner->get_closing();
                break;
            case ContentPiece::Type::Audio:
                result += owner->get_opening() + "audio:" + piece.content + owner->get_closing();
                break;
            case ContentPiece::Type::Video:
                result += owner->get_opening() + "video:" + piece.content + owner->get_closing();
                break;
            default:
                CHATLLM_CHECK(false) << "Unsupported content type: " << (int)piece.type;
                break;
            }
        }
        return result;
    }

    bool Content::is_simple_text(void) const
    {
        if (pieces.size() > 1)
            return false;
        if (pieces.size() == 0)
            return true;
        return pieces[0].type == ContentPiece::Type::Text;
    }

    Message & Message::operator =(const chatllm::Message &m)
    {
        this->content = m.content;
        return *this;
    }

    bool Message::save(FILE *f) const
    {
        if (f == nullptr)
            return false;
        uint64_t len = content.pieces.size();
        int role = this->role;
        if (fwrite(&round, sizeof(len), 1, f) != 1) return false;
        if (fwrite(&role, sizeof(role), 1, f) != 1) return false;
        if (fwrite(&len, sizeof(len), 1, f) != 1) return false;

        for (auto &p : content.pieces)
        {
            int type = p.type;
            if (fwrite(&type, sizeof(type), 1, f) != 1) return false;
            size_t len = p.content.size();
            if (fwrite(&len, sizeof(len), 1, f) != 1) return false;
            if (fwrite(p.content.data(), 1, len, f) != len) return false;
        }
        return true;
    }

    Message Message::load(Messages *owner, FILE *f)
    {
        if (f == nullptr)
            return Message(owner, "", MsgRole::User, 0);
        uint64_t len = 0;
        int role = 0;
        int round = 0;
        if (fread(&round, sizeof(round), 1, f) != 1) return Message(owner, "", MsgRole::User, 0);
        if (fread(&role, sizeof(role), 1, f) != 1) return Message(owner, "", MsgRole::User, 0);
        if (fread(&len, sizeof(len), 1, f) != 1) return Message(owner, "", MsgRole::User, 0);

        Message msg(owner, "", (MsgRole)role, round);
        msg.content.pieces.clear();
        for (uint64_t i = 0; i < len; i++)
        {
            int type = 0;
            if (fread(&type, sizeof(type), 1, f) != 1) return msg;
            size_t len = 0;
            if (fread(&len, sizeof(len), 1, f) != 1) return msg;
            std::string s(len, '\0');
            if (fread(s.data(), 1, len, f) != len) return msg;
            msg.content.push_back(s, (ContentPiece::Type)type);
        }
        return msg;
    }

    Messages::Messages(const std::string &opening, const std::string &closing)
        : cursor(0), sep(""), auto_aggregate(true), round(-1), mm_opening(opening), mm_closing(closing)
    {
    }

    Messages::Messages() : Messages("", "")
    {
    }

    void Messages::clear(void)
    {
        history.clear();
        cursor = 0;
        round = -1;
    }

    void Messages::set_mm_tags(const std::string &opening, const std::string &closing)
    {
        mm_opening = opening;
        mm_closing = closing;
    }

    void Messages::push_back(const Content &content, MsgRole role)
    {
        if (MsgRole::Auto == role)
        {
            if (history.size() > 0)
            {
                for (int i = (int)history.size() - 1; (i >= 0) && (MsgRole::Auto == role); i--)
                {
                    switch (history[i].role)
                    {
                    case MsgRole::User:
                        role = MsgRole::Assistant;
                        break;
                    case MsgRole::Assistant:
                        role = MsgRole::User;
                        break;
                    default:
                        break;
                    }
                }
            }

            if (MsgRole::Auto == role)
                role = MsgRole::User;
        }

        if (auto_aggregate && (history.size() > 0))
        {
            auto &last = history[history.size() - 1];
            if (role == last.role)
            {
                last.content.push_back(sep);
                last.content.push_back(content);
                return;
            }
        }

        if (role != MsgRole::Assistant)
        {
            const MsgRole last = history.size() > 0 ? history[history.size() - 1].role : MsgRole::Assistant;
            if (MsgRole::Assistant == last)
                round++;
        }

        history.emplace_back(this, content, role, round);
    }

    void Messages::push_back(const Message &m)
    {
        push_back(m.content, m.role);
    }

    void Messages::push_back(const std::string &content, MsgRole role)
    {
        push_back(Message(this, content, role, round));
    }

    void Messages::move_cursor_to_end(void) const
    {
        // FIXME: a hack
        const_cast<Messages *>(this)->move_cursor_to_end();
    }

    void Messages::shrink(int size)
    {
        while (history.size() > size)
            history.pop_back();
        move_cursor_to_end();
    }

    void Messages::move_cursor_to_end(void)
    {
        cursor = (int)history.size();
    }

    BaseStreamer::BaseStreamer(BaseTokenizer *tokenizer)
        : is_prompt(true), tokenizer(tokenizer), log_level(0), is_first(true), print_len(0),
          interceptor(nullptr)
    {
    }

    void BaseStreamer::put(const std::vector<int> &output_ids)
    {
        is_prompt = false;

        token_cache.insert(token_cache.end(), output_ids.begin(), output_ids.end());
        std::string text = tokenizer->decode(token_cache);
        if (text.empty())
        {
            return;
        }

        std::string printable_text;
        if ((text.back() == '\n') || (text.back() == '\r'))
        {
            // flush the cache after newline
            printable_text = text.substr(print_len);
            token_cache.clear();
            print_len = 0;
        }
        else
        {
            size_t end = tokenizer::get_end_of_valid_utf8(text, print_len);
            if (end > print_len)
            {
                printable_text = text.substr(print_len, end - print_len);
                print_len = end;
            }
        }

        if (printable_text.size() > 0)
        {
            call_put_chunk(is_first, printable_text);
            is_first = false;
        }
    }

    void BaseStreamer::end()
    {
        if (tokenizer)
        {
            std::string text = tokenizer->decode(token_cache);
            size_t end = tokenizer::get_end_of_valid_utf8(text, print_len);
            if (end > print_len)
            {
                call_put_chunk(is_first, text.substr(print_len));
                print_len = end;
            }
        }

        if (interceptor)
        {
            auto it = interceptor;
            interceptor = nullptr;
            it->end();
            interceptor = it;
        }

        is_first = true;
        is_prompt = true;
        token_cache.clear();
        print_len = 0;
    }

    void BaseStreamer::set_interceptor(ChunkInterceptor *interceptor)
    {
        ChunkInterceptor *t = this->interceptor;
        while (t)
        {
            if (t == interceptor)
                return;
            t = t->get_next();
        }

        interceptor->intercept(this->interceptor ? this->interceptor : this, this);
        this->interceptor = interceptor;
    }

    void BaseStreamer::call_put_chunk(bool first, const std::string &chunk)
    {
        if (interceptor)
            interceptor->put_chunk(first, chunk);
        else
            put_chunk(first, chunk);
    }

    void ThoughtChunkInterceptor::init(std::vector<std::pair<std::string, std::string>> tags)
    {
        this->tags = tags;
        active = tags.size() > 0;
        if (active)
            std::sort(this->tags.begin(), this->tags.end(), [](auto& a, auto& b) { return a.first.size() < b.first.size(); });
    }

    void ThoughtChunkInterceptor::put_chunk(bool first, const std::string &chunk)
    {
        if (!active)
        {
            next->put_chunk(first, chunk);
            return;
        }

        if (first) reset();

        if (is_thinking)
        {
            thought_chunk(false, chunk);
            return;
        }

        if (detecting)
        {
            raw_acc = raw_acc + chunk;
            cleaned_acc = utils::trim(cleaned_acc + chunk);

            size_t i = 0;
            for (; i < tags.size(); i++)
            {
                if (cleaned_acc.size() < tags[i].first.size())
                    break;
                if (cleaned_acc.substr(0, tags[i].first.size()) == tags[i].first)
                {
                    detecting = false;
                    is_thinking = true;
                    tag_opening = tags[i].first;
                    tag_closing = tags[i].second;

                    auto chunk = cleaned_acc.substr(tag_opening.size());
                    raw_acc = "";
                    thought_chunk(true, chunk);
                    return;
                }
            }

            if (i >= tags.size())
            {
                detecting = false;
                next->put_chunk(first, raw_acc);
            }
            return;
        }

        next->put_chunk(first, chunk);
    }

    void ThoughtChunkInterceptor::reset(void)
    {
        raw_acc = "";
        cleaned_acc = "";
        detecting = true;
        is_thinking = false;
        chunk_lengths.clear();
        first_thought_chunk = true;
    }

    void ThoughtChunkInterceptor::thought_chunk(bool first, const std::string &chunk)
    {
        raw_acc = raw_acc + chunk;
        chunk_lengths.push_back((int)chunk.size());
        size_t pos = raw_acc.find(tag_closing);
        if (pos != std::string::npos)
        {
            if (pos > 0)
            {
                streamer->put_thought_chunk(first_thought_chunk, raw_acc.substr(0, pos));
                first_thought_chunk = false;
            }
            is_thinking = false;
            streamer->end_thought();
            std::string text = raw_acc.substr(pos + tag_closing.size());
            next->put_chunk(first, text);
        }
        else
        {
            int sum = 0;
            int i = (int)chunk_lengths.size() - 1;
            for (; i >= 0; i--)
            {
                sum += chunk_lengths[i];
                if (sum >= (int)tag_closing.size())
                    break;
            }

            while (i > 0)
            {
                streamer->put_thought_chunk(first_thought_chunk, raw_acc.substr(0, chunk_lengths[0]));
                first_thought_chunk = false;
                raw_acc = raw_acc.substr(chunk_lengths[0]);
                chunk_lengths.erase(chunk_lengths.begin());
                i--;
            }
        }
    }

    BaseTokenizer::BaseTokenizer(const BaseConfig &config,
                        BaseHistoryEncoder *chat_encoder,
                        BaseHistoryEncoder *qa_encoder,
                        BaseHistoryEncoder *completion_encoder) :
        bos_token_id(config.bos_token_id),
        eos_token_id(config.eos_token_id),
        pad_token_id(config.pad_token_id),
        sep_token_id(config.sep_token_id),
        tp(nullptr),
        sys_prompt(""),
        max_length(config.max_length),
        format(ChatFormat::CHAT),
        chat_encoder(chat_encoder),
        completion_encoder(completion_encoder),
        qa_encoder(qa_encoder),
        auto_add_bos(true),
        vocab_size(config.vocab_size)
    {
        if (chat_encoder)
            chat_encoder->set_tokenizer(this);
        if (completion_encoder)
            completion_encoder->set_tokenizer(this);
        if (qa_encoder)
            qa_encoder->set_tokenizer(this);
    }

    void BaseTokenizer::set_chat_encoder(BaseHistoryEncoder *encoder)
    {
        chat_encoder = encoder;
        if (encoder)
            encoder->set_tokenizer(this);
    }

    int BaseTokenizer::get_image_total_emb_vectors(void)
    {
        int r = 0;
        for (auto &image : media_emb) r += image.emb_vec_number;
        return r;
    }

    bool BaseTokenizer::is_terminate_token_id(int id) const
    {
        if (id == eos_token_id) return true;

        return terminate_ids.find(id) != terminate_ids.end();
    }

    std::string BaseTokenizer::preprocess(const std::string &text) const
    {
        return text;
    }

    std::string BaseTokenizer::postprocess(const std::string &text) const
    {
        return text;
    }

    void BaseTokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        std::string input = preprocess(text);
        tp->Encode(input, &ids);
    }

    void BaseTokenizer::encode_external_text_completion(const std::string &text, std::vector<int> &ids) const
    {
        std::string input = preprocess(text);
        tp->Encode(input, &ids);
    }

    void BaseTokenizer::encode_qa(const std::string &q, const std::string &a, std::vector<int> &ids) const
    {
        std::string input = preprocess(q);
        tp->Encode(input, &ids);

        input = preprocess(a);
        tp->Encode(input, &ids);
    }

    void BaseTokenizer::encode_embedding(const std::string &text, std::vector<int> &ids, EmbeddingPurpose purpose) const
    {
        encode(text, ids);
    }

    std::vector<int> BaseTokenizer::encode(const std::string &text) const
    {
        std::vector<int> ids;
        encode(text, ids);
        return ids;
    }

    std::string BaseTokenizer::decode(const std::vector<int> &ids) const
    {
        // filter out special tokens
        std::vector<int> normal_ids(ids);
        normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(),
                                        [this](int id)
                                        { return is_special_id(id); }),
                        normal_ids.end());
        std::string text;
        tp->Decode(normal_ids, &text);
        text = postprocess(text);
        return text;
    }

    int BaseTokenizer::get_history_start(const Messages &history, int max_length) const
    {
        int start = (int)history.size() - 1;
        size_t total_id_num = encode(history[start].content.to_string()).size();
        start--;
        while (start >= 1)
        {
            total_id_num += encode(history[start].content.to_string()).size();
            total_id_num += encode(history[start - 1].content.to_string()).size();
            if ((int)total_id_num >= max_length)
                break;
            start -= 2;
        }
        return start < 0 ? 0 : start;
    }

    std::vector<int> BaseTokenizer::encode_history(BaseHistoryEncoder *encoder, const Messages &history,
        int max_length, const bool incremental, const bool ai_opening, const bool reversed_role)
    {
        std::vector<int> input_ids;

        auto encode_start = history.get_cursor();

        if (!incremental)
        {
            encode_start = get_history_start(history, max_length / 2);
        }
        else
        {
            for (; (encode_start < (int)history.size()) && (history[encode_start].role == (reversed_role ? MsgRole::User : MsgRole::Assistant)); encode_start++)
                ;
        }

        for (; encode_start < (int)history.size(); encode_start++)
        {
            encoder->append_message(history[encode_start], input_ids);
        }

        if (ai_opening)
        {
            if (reversed_role)
                encoder->append_user_opening(encode_start > 0 ? history[encode_start - 1].round : 0, input_ids);
            else
                encoder->append_ai_opening(encode_start > 0 ? history[encode_start - 1].round : 0, input_ids);
        }

        history.move_cursor_to_end();

        // DEBUG
        //return {6023, 25};

        return input_ids;
    }

    std::vector<int> BaseTokenizer::encode_sys_prompt(void)
    {
        std::vector<int> ids;
        switch (format)
        {
        case ChatFormat::COMPLETION:
            if (completion_encoder)
                completion_encoder->append_sys_prompt(ids);
            else;
            break;

        case ChatFormat::QA:
            if (qa_encoder)
                qa_encoder->append_sys_prompt(ids);
            else if (chat_encoder)
            {
                chat_encoder->append_sys_prompt(ids);
            }
            else;
            break;
        default:
            if (chat_encoder)
                chat_encoder->append_sys_prompt(ids);
            else;
        }

        if (auto_add_bos && (bos_token_id >= 0))
        {
            if ((ids.size() > 0) && (ids[0] != bos_token_id))
                ids.insert(ids.begin(), bos_token_id);
        }
        return ids;
    }

    std::vector<int> BaseTokenizer::encode_history(const Messages &history, int max_length, const bool incremental,
                                                   const bool ai_opening, const bool reversed_role)
    {
        switch (format)
        {
        case ChatFormat::COMPLETION:
            if (completion_encoder)
                return encode_history(completion_encoder, history, max_length, incremental, ai_opening, reversed_role);
            else;
            break;

        case ChatFormat::QA:
            if (qa_encoder)
                return encode_history(qa_encoder, history, max_length, incremental, ai_opening, reversed_role);
            else if (chat_encoder)
            {
                Messages copied(history.get_opening(), history.get_closing());
                copied.push_back(history[history.size() - 1]);
                return encode_history(chat_encoder, copied, max_length, incremental, ai_opening, reversed_role);
            }
            else;
            break;
        default:
            if (chat_encoder)
                return encode_history(chat_encoder, history, max_length, incremental, ai_opening, reversed_role);
            else;
        }

        std::vector<int> r = encode(history[(int)history.size() - 1].content.to_string());
        if (auto_add_bos && (bos_token_id >= 0))
        {
            if ((r.size() > 0) && (r[0] != bos_token_id))
                r.insert(r.begin(), bos_token_id);
            else
                r.push_back(bos_token_id);
        }
        return r;
    }

    void BaseTokenizer::set_skip_sys_prompt(bool skip)
    {
        if (chat_encoder)
            chat_encoder->skip_sys_prompt = skip;
        if (completion_encoder)
            completion_encoder->skip_sys_prompt = skip;
        if (qa_encoder)
            qa_encoder->skip_sys_prompt = skip;
    }

    int BaseTokenizer::load_added_tokens(const json::JSON &config, std::initializer_list<std::pair<std::string, int *>> added_tokens)
    {
        int r = -1;
        auto cfg = config["tokenizer_config.json"];
        if (!cfg.IsObject()) return r;
        auto added_tokens_decoder = cfg["added_tokens_decoder"];
        if (!added_tokens_decoder.IsObject()) return r;

        r = 0;

        for (auto &item : added_tokens_decoder.ObjectRange())
        {
            for( auto tok = added_tokens.begin(), e = added_tokens.end(); tok != e; ++tok)
            {
                if (tok->first == item.second["content"].ToString())
                {
                    *tok->second = std::stol(item.first);
                    break;
                }
            }
        }

        return r;
    }

    void BaseHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
    {
    }

    void BaseHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
    {
        tokenizer->encode(user, ids);
    }

    void BaseHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        CHATLLM_THROW << "BaseHistoryEncoder::append_ai no override.";
    }

    void BaseHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        CHATLLM_THROW << "BaseHistoryEncoder::append_ai no override.";
    }

    void BaseHistoryEncoder::append_user_opening(int round_idx, std::vector<int> &ids) const
    {
        CHATLLM_THROW << "BaseHistoryEncoder::append_user_opening no override.";
    }

    void BaseHistoryEncoder::append_user(int round_idx, const Content &user, std::vector<int> &ids) const
    {
        CHATLLM_THROW << "BaseHistoryEncoder::append_user(<Content>) no override.";
    }

    void BaseHistoryEncoder::append_tool(int round_idx, const std::string &tool, std::vector<int> &ids) const
    {
        append_user(round_idx, tool, ids);
    }

    void BaseHistoryEncoder::append_message(const Message &msg, std::vector<int> &ids) const
    {
        if ((ids.size() < 1) && (msg.round == 0))
        {
            if (!skip_sys_prompt)
                append_sys_prompt(ids);
        }

        int round_idx = msg.round;
        size_t num = ids.size();

        switch (msg.role)
        {
        case MsgRole::User:
            if (msg.content.is_simple_text())
                append_user(round_idx, msg.content.to_string(), ids);
            else
                append_user(round_idx, msg.content, ids);
            break;
        case MsgRole::Assistant:
            CHATLLM_CHECK(msg.content.is_simple_text());
            append_ai(round_idx, msg.content.to_string(), ids);
            break;
        case MsgRole::Tool:
            CHATLLM_CHECK(msg.content.is_simple_text());
            append_tool(round_idx, msg.content.to_string(), ids);
            break;
        default:
            CHATLLM_CHECK(false) << "unexpected msg.role " << msg.role;
            break;
        }
    }

    static std::string shape_to_string(ggml::tensor *tensor)
    {
        int n_dims = ggml::n_dims(tensor);
        std::ostringstream oss;
        oss << '[';
        for (int i = n_dims - 1; i >= 0; i--)
        {
            oss << tensor->ne[i] << (i > 0 ? ", " : "");
        }
        oss << ']';
        return oss.str();
    }

    static std::string strides_to_string(ggml::tensor *tensor)
    {
        int n_dims = ggml::n_dims(tensor);
        std::ostringstream oss;
        oss << '[';
        for (int i = n_dims - 1; i >= 0; i--)
        {
            oss << tensor->nb[i] << (i > 0 ? ", " : "");
        }
        oss << ']';
        return oss.str();
    }

    std::string to_string(ggml::tensor *tensor, bool with_data)
    {
        std::ostringstream oss;
        int n_dims = ggml::n_dims(tensor);
        oss << "ggml::tensor(";

        if (with_data)
        {
            if (n_dims > 3)
                oss << "[";
            for (int i3 = 0; i3 < tensor->ne[3]; i3++)
            {
                if (n_dims > 2)
                    oss << (i3 > 0 ? ",\n\n[" : "[");
                for (int i2 = 0; i2 < tensor->ne[2]; i2++)
                {
                    if (n_dims > 1)
                        oss << (i2 > 0 ? ",\n\n[" : "[");
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++)
                    {
                        oss << (i1 > 0 ? ",\n[" : "[");
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++)
                        {
                            auto ptr = (char *)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                                       i0 * tensor->nb[0];
                            float val;
                            if (tensor->type == GGML_TYPE_F32)
                            {
                                val = *(float *)ptr;
                            }
                            else if (tensor->type == GGML_TYPE_F16)
                            {
                                val = ggml_fp16_to_fp32(*(ggml_fp16_t *)ptr);
                            }
                            else
                            {
                                CHATLLM_THROW << "unimplemented";
                            }
                            oss << (i0 > 0 ? ", " : "") << std::setw(7) << std::fixed << std::setprecision(4) << val;
                        }
                        oss << "]";
                    }
                    if (n_dims > 1)
                        oss << "]";
                }
                if (n_dims > 2)
                    oss << "]";
            }
            if (n_dims > 3)
                oss << "]";
        }

        oss << ", shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
        return oss.str();
    }

#ifdef _POSIX_MAPPED_FILES
    MappedFile::MappedFile(const std::string &path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        CHATLLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct stat sb;
        CHATLLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
        _size = sb.st_size;

        data = (char *)mmap(nullptr, _size, PROT_READ, MAP_SHARED, fd, 0);
        CHATLLM_CHECK(data != MAP_FAILED) << strerror(errno);

        CHATLLM_CHECK(close(fd) == 0) << strerror(errno);
        ptr = data;
    }

    MappedFile::~MappedFile() { CHATLLM_CHECK(munmap(data, _size) == 0) << strerror(errno); }
#elif defined(_WIN32)
    MappedFile::MappedFile(const std::string &path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        CHATLLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct _stat64 sb;
        CHATLLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
        _size = sb.st_size;

        HANDLE hFile = (HANDLE)_get_osfhandle(fd);

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        CHATLLM_CHECK(hMapping != NULL) << strerror(errno);

        data = (char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMapping);

        CHATLLM_CHECK(data != NULL) << strerror(errno);
        CHATLLM_CHECK(close(fd) == 0) << strerror(errno);
        ptr = data;
    }

    MappedFile::~MappedFile()
    {
        CHATLLM_CHECK(UnmapViewOfFile(data)) << strerror(errno);
    }
#endif

    int64_t MappedFile::tell()
    {
        return ptr - data;
    }

    void MappedFile::seek(int64_t offset, int whence)
    {
        if (whence == SEEK_SET)
            ptr = data + offset;
        else if (whence == SEEK_CUR)
            ptr += offset;
        else if (whence == SEEK_END)
            ptr = data + size() + offset;
        else
            CHATLLM_THROW << "invalid seek mode " << whence;
    }

    size_t MappedFile::read_buffer(void *output, size_t len)
    {
        size_t remain = size() - tell();
        if (len > remain) len = remain;
        memcpy(output, ptr, len);
        ptr += len;
        return len;
    }

    SimpleFile::SimpleFile(const std::string &path)
    {
        f = std::fopen(path.c_str(), "rb");
        CHATLLM_CHECK(f != nullptr) << "cannot open file " << path << ": " << strerror(errno);
        seek(0, SEEK_END);
        _size = tell();
        seek(0, SEEK_SET);
    }

    SimpleFile::~SimpleFile()
    {
        std::fclose(f);
        f = nullptr;
    }

#if defined(_MSC_VER)
#define ftello64    _ftelli64
#define fseeko64    _fseeki64
#elif defined(__APPLE__)
#define ftello64    ftello
#define fseeko64    fseeko
#endif

    int64_t SimpleFile::tell()
    {
        return ftello64(f);
    }

    void SimpleFile::seek(int64_t offset, int whence)
    {
        CHATLLM_CHECK(fseeko64(f, offset, whence) == 0) << "SimpleFile::seek: fails offset = " << offset << ", whence = " << whence;
    }

    size_t SimpleFile::read_buffer(void *output, size_t len)
    {
        return fread(output, 1, len, f);
    }

    TensorInfo::TensorInfo(ggml::type type, int n_dim, const int64_t *ne, size_t _offset, const char *name)
        : _offset(_offset), data(nullptr), original_type(ggml::type::GGML_TYPE_F32)
    {
        ggml::init_tensor(&tensor, type, n_dim, ne);
        usage = ggml::n_dims(&tensor) > 1 ? BackendBufAllocator::Usage::Matrix : BackendBufAllocator::Usage::Others;
        ggml::set_name(&tensor, name);
    }

    TensorInfo::~TensorInfo()
    {
        // data is freed by allocator
        data = nullptr;
    }

    size_t TensorInfo::aligned_data_start(size_t offset)
    {
        constexpr size_t MEM_ALIGNED = 16;
        return (offset + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
    }

    size_t TensorInfo::aligned_size()
    {
        return (aligned_data_start(_offset) - _offset) + ggml::nbytes(&tensor);
    }

    size_t TensorInfo::get_nbytes(void)
    {
        return ggml::nbytes(&tensor);
    }

    bool TensorInfo::load(tokenizer::DataReader *reader, LayerBufAllocator *alloc, ggml::type target_type, size_t override_buffer_size)
    {
        if (data)
        {
            CHATLLM_CHECK(ggml::type_of(tensor) == target_type) << "type mismatch: " << ggml::type_of(tensor) << ", " << target_type;
            CHATLLM_CHECK(this->alloc == alloc) << "TODO: tensor (" << tensor.name << ") loaded, but using different allocators";
            return true;
        }

        this->original_type = ggml::type_of(tensor);
        ggml::change_type(&tensor, target_type);
        size_t alloc_size = std::max(override_buffer_size, alloc->get_alloc_size(&tensor, usage));
        data = alloc->alloc(alloc_size, usage);
        data->assign_to(&tensor);
        this->alloc = alloc;

        if (reader)
            read_tensor_data(reader, _offset, 0, ggml::nbytes(&tensor), target_type);

        return true;
    }

    size_t TensorInfo::read_raw_tensor_data(tokenizer::DataReader *reader, size_t data_size, void *p)
    {
        reader->seek(aligned_data_start(_offset), SEEK_SET);
        reader->read_buffer((uint8_t *)p, data_size);
        return data_size;
    }

    size_t TensorInfo::read_tensor_data(tokenizer::DataReader *reader, size_t read_offset, size_t write_offset, size_t data_size,
                                        ggml::type target_type)
    {
        CHATLLM_CHECK(data) << "backend buffer still not allocated!";
        CHATLLM_CHECK(target_type == ggml::type_of(tensor)) << "tensor type mismatch!";
        CHATLLM_CHECK(data->get_size() >= write_offset + data_size) << "read_tensor_data(" << ggml::get_name(&tensor) << "): write data exceeds tensor data size";

        reader->seek(aligned_data_start(read_offset), SEEK_SET);

        if (target_type != original_type)
        {
            if (  (ggml::type::GGML_TYPE_F32 == original_type)
               && (ggml::type::GGML_TYPE_F16 == target_type))
                return read_tensor_data_f32_f16(reader, read_offset, write_offset, data_size);
            else if (  (ggml::type::GGML_TYPE_F16 == original_type)
                    && (ggml::type::GGML_TYPE_F32 == target_type))
                return read_tensor_data_f16_f32(reader, read_offset, write_offset, data_size);
            else
            {
                std::vector<uint8_t> buf;
                std::vector<uint8_t> buf_q;

                ggml::tensor t;
                ggml::init_tensor(&t, ggml::type::GGML_TYPE_F32, 4, tensor.ne);
                buf.resize(ggml::nbytes(&t));

                if (ggml::type::GGML_TYPE_F32 == original_type)
                {
                    reader->read_buffer(buf.data(), buf.size());
                }
                else
                {
                    ggml::init_tensor(&t, original_type, 4, tensor.ne);
                    buf_q.resize(ggml::nbytes(&t));
                    reader->read_buffer(buf_q.data(), buf_q.size());

                    ggml::to_float(original_type, buf_q.data(), (float *)buf.data(), ggml::get_dim(&t, 0), ggml::nrows(&t));
                }

                ggml::init_tensor(&t, target_type, 4, tensor.ne);
                buf_q.resize(ggml::nbytes(&t));
                ggml::from_float(target_type, (const float *)buf.data(), (void *)buf_q.data(), ggml::get_dim(&t, 0), ggml::nrows(&t));

                CHATLLM_CHECK(buf_q.size() == data_size) << "size mismatch? " << buf_q.size() << " : " << data_size;

                alloc->get_backend()->write_tensor_data(&tensor, buf_q.data(), write_offset, buf_q.size());

                return buf_q.size();
            }
        }


        if (data->is_host())
        {
            reader->read_buffer((uint8_t *)data->get_base() + write_offset, data_size);
#if (0)
            if (std::string(tensor.name).find("embed_tokens.weight") != std::string::npos)
            {
                int8_t *p = (int8_t *)data->get_base() + write_offset;
                printf("patching emb (Q8_0)\n");

                #define DIM 1536
                for (int k = 0; k < 2; k++)
                {
                    if (tensor.type == GGML_TYPE_Q8_0)
                    {
                        #define GGML_QK8_0 32
                        memset(p, 0, (DIM / GGML_QK8_0) * (GGML_QK8_0 + 2));
                        *(ggml_fp16_t *)p = ggml_fp32_to_fp16(1.0f);
                        p[2] = 1;
                        p += (DIM / GGML_QK8_0) * (GGML_QK8_0 + 2);
                    }
                    else if (tensor.type == GGML_TYPE_F16)
                    {
                        memset(p, 0, DIM * sizeof(ggml_fp16_t));
                        *(ggml_fp16_t *)p = ggml_fp32_to_fp16(1.0f);
                        p += DIM * sizeof(ggml_fp16_t);
                    }
                    else if (tensor.type == GGML_TYPE_F32)
                    {
                        memset(p, 0, DIM * sizeof(float));
                        *(float *)p = 1.0f;
                        p += DIM * sizeof(float);
                    }
                    else
                        CHATLLM_CHECK(false) << tensor.type;
                }
            }
#endif
        }
        else
        {
            const int BUF_SIZE = 1024 * 1000;
            std::vector<uint8_t> buf;
            buf.resize(BUF_SIZE);
            size_t remain = data_size;
            size_t offset = write_offset;
            while (remain > 0)
            {
                size_t block = remain > buf.size() ? buf.size() : remain;
                reader->read_buffer(buf.data(), block);
                alloc->get_backend()->write_tensor_data(&tensor, buf.data(), offset, block);

                remain -= block;
                offset += block;
            }
        }

        return data_size;
    }

    size_t TensorInfo::read_tensor_data_f32_f16(tokenizer::DataReader *reader, size_t read_offset, size_t write_offset, size_t data_size)
    {
        std::vector<uint8_t> buf;
        buf.resize(data_size * 2);
        reader->read_buffer(buf.data(), buf.size());

        ggml_fp16_t *p16 = (ggml_fp16_t *)buf.data();
           float    *p32 = (      float *)buf.data();
        for (size_t i = 0; i < data_size / 2; i++)
        {
            p16[i] = ggml_fp32_to_fp16(p32[i]);
        }

        if (data->is_host())
            memcpy((uint8_t *)data->get_base() + write_offset, buf.data(), data_size);
        else
            alloc->get_backend()->write_tensor_data(&tensor, buf.data(), write_offset, data_size);

        return data_size;
    }

    size_t TensorInfo::read_tensor_data_f16_f32(tokenizer::DataReader *reader, size_t read_offset, size_t write_offset, size_t data_size)
    {
        std::vector<uint8_t> buf;
        buf.resize(data_size);
        reader->read_buffer(buf.data(), data_size / 2);

        ggml_fp16_t *p16 = (ggml_fp16_t *)buf.data();
           float    *p32 = (      float *)buf.data();
        for (int i = (int)data_size / sizeof(float) - 1; i >= 0; i--)
        {
            p32[i] = ggml_fp16_to_fp32(p16[i]);
        }

        if (data->is_host())
            memcpy((uint8_t *)data->get_base() + write_offset, buf.data(), data_size);
        else
            alloc->get_backend()->write_tensor_data(&tensor, buf.data(), write_offset, data_size);

        return data_size;
    }

    void TensorInfo::assign_to(ggml::tensor *tensor)
    {
        data->assign_to(tensor);
    }

    void ModelLoader::load_all_tensors(void)
    {
        if (tensor_dict.size() > 0) return;

        while (tell() < _file->size())
        {
            std::string weight_name;

            // read and check tensor name
            {
                int name_size = read_basic<int>();
                weight_name = read_string(name_size);
            }

            int64_t ne[4] = {1, 1, 1, 1};

            // read and check tensor shape
            int ndim = read_basic<int>();
            for (int i = ndim - 1; i >= 0; i--)
            {
                int dim_size = read_basic<int>();
                ne[i] = dim_size;
            }

            // read and check tensor dtype
            ggml::type dtype = (ggml::type)read_basic<int>();

            tensor_dict.emplace(weight_name, TensorInfo(dtype, ndim, ne, tell(), weight_name.c_str()));

            TensorInfo &t = tensor_dict.at(weight_name);

            seek(t.aligned_size(), SEEK_CUR);
        }
    }

    LayerAllocatorManager *ModelLoader::alloc_manager()
    {
        return alloc_managers.back();
    }

    void ModelLoader::push_allocator_manager(LayerAllocatorManager *alloc_manager)
    {
        alloc_managers.push_back(alloc_manager);
    }

    void ModelLoader::pop_allocator_manager(void)
    {
        alloc_managers.pop_back();
    }

    void ModelLoader::read_tensor(const std::string &name, ggml::tensor *tensor, bool partial)
    {
        read_tensor(name, tensor, alloc_manager()->get_allocator(tensor), partial);
    }

    void ModelLoader::read_tensor(const std::string &name,
                    const std::string &layer_prefix, int num, const std::string &suffix,
                    ggml::tensor *tensor)
    {
        read_tensor(name, layer_prefix, num, suffix, tensor, alloc_manager()->get_allocator(tensor));
    }

    bool ModelLoader::has_tensor(const std::string &name) const
    {
        auto translated = translate_tensor_name(name);
        auto search = tensor_dict.find(translated);
        return search != tensor_dict.end();
    }

    void  ModelLoader::read_scaler(const std::string &name, float *value)
    {
        auto translated = translate_tensor_name(name);
        auto search = tensor_dict.find(translated);
        CHATLLM_CHECK(search != tensor_dict.end()) << "tensor not exist: " << translated;

        TensorInfo &t = search->second;

        // read and check tensor shape
        {
            int ndim = ggml::n_dims(&t.tensor);

            if (ndim == 1)
            {
                CHATLLM_CHECK(t.tensor.ne[0] == 1)
                    << "scaler " << name << " ndim mismatch: got shape: " << shape_to_string(&t.tensor);
            }
            else
            {
                CHATLLM_CHECK(ndim == 0)
                    << "scaler " << name << " expected, but find dim = " << ndim;
            }
            CHATLLM_CHECK(t.tensor.type == ggml::type::GGML_TYPE_F32);
            t.read_raw_tensor_data(_file.get(), sizeof(float), value);
        }
    }

    void ModelLoader::read_tensor(const std::string &name, ggml::tensor *tensor, LayerBufAllocator *allocator, bool partial)
    {
        auto translated = translate_tensor_name(name);
        auto search = tensor_dict.find(translated);
        CHATLLM_CHECK(search != tensor_dict.end()) << "tensor not exist: " << translated;

        ggml::set_name(tensor, name.c_str());
        TensorInfo &t = search->second;

        // read and check tensor shape
        {
            int ndim = ggml::n_dims(&t.tensor);
            int n_dims = ggml::n_dims(tensor);

            // a quick fix
            if ((n_dims == 1) && (ndim == 2) && (tensor->ne[1] == 1))
                n_dims = 2;

            CHATLLM_CHECK(ndim == n_dims)
                << "tensor " << name << " ndim mismatch: expect " << n_dims << " but got " << ndim << ". expected shape: " << shape_to_string(tensor);

            if (partial)
            {
                for (int i = 0; i < ndim; i++)
                {
                    int64_t dim_size = t.tensor.ne[i];
                    if (dim_size < tensor->ne[i])
                    {
                        CHATLLM_CHECK(i == 1) << "partial loading only allowed on dim 1";
                        continue;
                    }
                    CHATLLM_CHECK(dim_size == tensor->ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                                            << ": expect " << tensor->ne[i] << " but got " << dim_size;
                }
            }
            else
            {
                for (int i = 0; i < ndim; i++)
                {
                    int64_t dim_size = t.tensor.ne[i];
                    CHATLLM_CHECK(dim_size == tensor->ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                                            << ": expect " << tensor->ne[i] << " but got " << dim_size;
                }
            }
        }

        size_t override_alloc_size = 0;
        if (partial)
        {
            override_alloc_size = allocator->get_alloc_size(tensor, t.usage);
        }

        CHATLLM_CHECK(t.load(_file.get(), allocator, tensor->type, override_alloc_size)) << "failed to load tensor: " << name;

        t.assign_to(tensor);
    }

    std::string ModelLoader::translate_tensor_name(const std::string &name) const
    {
        std::string translated_name = name;
        for (const auto &pair : name_translation)
        {
            auto search = translated_name.find(pair.first);
            if (search != std::string::npos)
            {
                translated_name.replace(search, pair.first.length(), pair.second);
            }
        }

        return translated_name;
    }

    void ModelLoader::read_tensor(const std::string &name,
        const std::string &layer_prefix, int num, const std::string &suffix,
        ggml::tensor *tensor, LayerBufAllocator *allocator)
    {
        std::vector<std::string> names;
        for (int j = 0; j < num; j++)
        {
            names.push_back(layer_prefix + std::to_string(j) + suffix);
        }
        read_tensor(name, names, tensor, allocator);
    }

    void ModelLoader::read_tensor(const std::string &name, const std::vector<std::string> &concat_list,
                                  ggml::tensor *tensor, LayerBufAllocator *allocator)
    {
        auto search = tensor_dict.find(name);
        if (search != tensor_dict.end())
        {
            read_tensor(name, tensor, allocator);
            return;
        }

        tensor_dict.emplace(name, TensorInfo(tensor->type, 4, tensor->ne, 0, name.c_str()));

        TensorInfo &t = tensor_dict.at(name);
        t.load(nullptr, allocator, tensor->type);

        size_t total_size = t.get_nbytes();
        size_t write_offset = 0;
        for (size_t i = 0; i < concat_list.size(); i++)
        {
            auto translated = translate_tensor_name(concat_list[i]);
            auto search = tensor_dict.find(translated);
            CHATLLM_CHECK(search != tensor_dict.end()) << "tensor " << translated << " not exists.";

            for (int j = 0; j < GGML_MAX_DIMS; j++)
            {
                if (search->second.tensor.ne[j] == 1) break;
                CHATLLM_CHECK(tensor->ne[j] == search->second.tensor.ne[j]) << translated << ": shape mismatch dim[" << j << "]: " << tensor->ne[j] << " != " << search->second.tensor.ne[j];
            }

            size_t size = search->second.get_nbytes();
            t.read_tensor_data(_file.get(), search->second._offset, write_offset, size, tensor->type);

            write_offset += size;
            total_size -= size;
        }
        CHATLLM_CHECK(total_size == 0) << "tensor " << name << " not fully loaded.";

        t.assign_to(tensor);
    }

    int BaseModel::save_session(FILE *f) const
    {
        struct state state = {.type = type_, .n_past = n_past, .n_past_offset = n_past_offset};
        if (fwrite(&state, sizeof(state), 1, f) != 1)
            return -1;
        return 0;
    }

    int BaseModel::load_session(FILE *f)
    {
        struct state state;
        if (fread(&state, sizeof(state), 1, f) != 1)
            return -1;
        if (state.type != type_) return -1;
        n_past = state.n_past;
        n_past_offset = state.n_past_offset;
        return 0;
    }

    int BaseModel::save_session(ModelSessionMemory &session) const
    {
        session.set_n_past(n_past);
        session.set_n_past_offset(n_past_offset);
        return 0;
    }

    int BaseModel::load_session(ModelSessionMemory &session)
    {
        n_past = session.get_n_past();
        n_past_offset = session.get_n_past_offset();
        return 0;
    }

    ModelObject::ModelObject(const std::string &path)
        : ModelObject(path, ModelObject::extra_args())
    {
    }

    ModelObject::ModelObject(const std::string &path, const extra_args &args)
        : loader(nullptr), loaded(path.size() > 0)
    {
        ModelFactory::Result result = {nullptr, nullptr};
        if (path.size() > 0)
        {
            loader = std::unique_ptr<ModelLoader>(new ModelLoader(path));
            if (!ModelFactory::load(*loader, result, args))
                CHATLLM_THROW << "ModelFactory::load() failed";
        }

        tokenizer = std::move(result.tokenizer);
        model = std::move(result.model);
    }

    AbstractModel *ModelObject::fork_model(const extra_args &args)
    {
        if (!loaded) return nullptr;
        return ModelFactory::load_model_again(*loader, args);
    }

    ModelSessionMemory::ModelSessionMemory() : n_past(0), n_past_offset(0)
    {
    }

    void ModelSessionMemory::prepare(int id)
    {
        while (id >= (int)buffers.size())
        {
            buffers.push_back(std::vector<uint8_t>());
        }
    }

    void *ModelSessionMemory::prepare_buffer(int id, size_t size)
    {
        if (id >= 0)
        {
            prepare(id);
            buffers[id].resize(size);
            return get_buffer(id);
        }
        return nullptr;
    }

    void *ModelSessionMemory::get_buffer(int id, size_t *size)
    {
        if (id >= 0)
        {
            prepare(id);
            if (size)
                *size = buffers[id].size();
            return buffers[id].data();
        }
        return nullptr;
    }

    void ModelSessionMemory::set_n_past(int n_past)
    {
        this->n_past = n_past;
    }

    void ModelSessionMemory::set_n_past_offset(int n_past_offset)
    {
        this->n_past_offset = n_past_offset;
    }

    int ModelSessionMemory::get_n_past(void) const
    {
        return n_past;
    }

    int ModelSessionMemory::get_n_past_offset(void) const
    {
        return n_past_offset;
    }

    void ModelSessionMemory::copy_from(const ModelSessionMemory &sess)
    {
        if (this == &sess) return;

        n_past = sess.n_past;
        n_past_offset = sess.n_past_offset;

        for (int i = 0; i < (int)buffers.size(); i++)
            memcpy(buffers[i].data(), sess.buffers[i].data(), buffers[i].size());
    }

    void ModelSessionMemory::dump(const char *fn)
    {
        FILE *f = fopen(fn, "wb");
        struct state
        {
            int n_past;
            int n_past_offset;
        } state =
        {
            .n_past = this->n_past,
            .n_past_offset = this->n_past_offset,
        };
        fwrite(&state, sizeof(state), 1, f);

        for (int i = 0; i < (int)buffers.size(); i++)
        {
            fwrite(buffers[i].data(), 1, buffers[i].size(), f);
        }
        fclose(f);
    }

    // ===== pipeline =====

    Pipeline::Pipeline(const std::string &path)
        : Pipeline(path, ModelObject::extra_args())
    {
    }

    Pipeline::Pipeline(const std::string &path, const ModelObject::extra_args &args)
        : initializing(true),
          extending(ExtendingMethod::Restart),
          modelobj(path, args)
    {
        model = modelobj.model.get();
        tokenizer = modelobj.tokenizer.get();
    }

    std::string Pipeline::chat_with_restart(const Messages &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = tokenizer->get_chat_format() == ChatFormat::CHAT;
        bool completed = false;
        std::vector<int> input_ids;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous, true, gen_config.reversed_role);
        add_ai_prefix(input_ids, gen_config, streamer);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, streamer);
        if (!completed)
        {
            if (continuous)
            {
                streamer->putln("\nRUN OUT OF CONTEXT. Let me forget something and try again ...\n");
                input_ids = tokenizer->encode_history(history, gen_config.max_context_length, false, true, gen_config.reversed_role);
                add_ai_prefix(input_ids, gen_config, streamer);
                output_ids = model->generate(input_ids, gen_config, false, completed, &performance, streamer);
            }
            else
                streamer->putln("\nRUN OUT OF CONTEXT. I have to stop now.\n");
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    std::string Pipeline::chat_without_extending(const Messages &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = tokenizer->get_chat_format() == ChatFormat::CHAT;
        bool completed = false;
        std::vector<int> input_ids;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous, true, gen_config.reversed_role);
        add_ai_prefix(input_ids, gen_config, streamer);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, streamer);
        if (!completed)
        {
            streamer->putln("\nRUN OUT OF CONTEXT. I have to stop now.\n");
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    std::string Pipeline::chat_with_shift(const Messages &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        bool continuous = true;
        bool completed = false;
        if (initializing)
        {
            continuous = false;
            initializing = false;
        }
        else;

        std::vector<int> input_ids = tokenizer->encode_history(history, gen_config.max_context_length, continuous, true, gen_config.reversed_role);
        add_ai_prefix(input_ids, gen_config, streamer);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, streamer);

        while (!completed)
        {
            streamer->putln("\nRUN OUT OF CONTEXT. Try to forget something and continue ...\n");
            model->shift_memory(gen_config.max_context_length);
            if (output_ids.size() > 0)
                input_ids = {output_ids[output_ids.size() - 1]};
            else
                input_ids.clear();

            auto ids = model->generate(input_ids, gen_config, true, completed, &performance, streamer);
            output_ids.insert(output_ids.end(), ids.begin(), ids.end());
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    void Pipeline::eval_sys_prompt(const GenerationConfig &gen_config)
    {
        bool completed = false;
        std::vector<int> input_ids = tokenizer->encode_sys_prompt();

        GenerationConfig copy(gen_config);
        copy.max_new_tokens = 1;
        model->generate(input_ids, copy, false, completed, &performance, nullptr);

        // just in case that chatting is continued
        tokenizer->set_skip_sys_prompt(true);
        initializing = false;
        return;
    }

    std::string Pipeline::chat(Messages &history, const GenerationConfig &gen_config,
                               BaseStreamer *streamer)
    {
        std::string r;

        if (modelobj.loaded && streamer)
        {
            ChunkInterceptor *interceptor = model->get_interceptor();
            if (interceptor)
            {
                streamer->set_interceptor(interceptor);
            }
        }

        before_chat(history, gen_config, streamer);

        if (modelobj.loaded)
        {
            switch (extending)
            {
            case ExtendingMethod::Shift:
                r = chat_with_shift(history, gen_config, streamer);
                break;
            case ExtendingMethod::Restart:
                r = chat_with_restart(history, gen_config, streamer);
                break;
            default:
                r = chat_without_extending(history, gen_config, streamer);
                break;
            }
        }

        post_chat(history, gen_config, streamer);

        if (streamer)
            streamer->end();

        return r;
    }

    void Pipeline::add_ai_prefix(std::vector<int> &input_ids, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        if (gen_config.ai_prefix.size() > 0)
        {
            tokenizer->encode(gen_config.ai_prefix, input_ids);
            if (streamer)
                streamer->call_put_chunk(true, gen_config.ai_prefix);
        }
    }

    std::string Pipeline::chat_with_ext_completion(Messages &history, const std::string &external, const GenerationConfig &gen_config,
                         BaseStreamer *streamer)
    {
        bool continuous = true;
        bool completed = false;
        std::vector<int> input_ids;

        input_ids = tokenizer->encode_history(history, gen_config.max_length, false, false);
        rewind((int)input_ids.size());
        input_ids.clear();

        tokenizer->encode_external_text_completion(external, input_ids);
        if (streamer)
            streamer->call_put_chunk(false, external);

        history[history.size() - 1].content.push_back(external);

        std::vector<int> output_ids = model->generate(input_ids, gen_config, continuous, completed, &performance, streamer);
        if (!completed)
        {
            if (continuous)
            {
                streamer->putln("\nRUN OUT OF CONTEXT. Let me forget something and try again ...\n");
                input_ids = tokenizer->encode_history(history, gen_config.max_context_length, false, false);
                output_ids = model->generate(input_ids, gen_config, false, completed, &performance, streamer);
            }
            else
                streamer->putln("\nRUN OUT OF CONTEXT. I have to stop now.\n");
        }

        std::string output = tokenizer->decode(output_ids);
        return output;
    }

    std::string Pipeline::chat_continue(Messages &history, const std::string &external_ai, const GenerationConfig &gen_config,
                         BaseStreamer *streamer)
    {
        std::string r;

        if (history.size() < 1) return r;
        if (history[history.size() - 1].role != MsgRole::Assistant) return r;

        if (modelobj.loaded && streamer)
        {
            ChunkInterceptor *interceptor = model->get_interceptor();
            if (interceptor)
            {
                streamer->set_interceptor(interceptor);
            }
        }

        before_chat(history, gen_config, streamer);

        r = chat_with_ext_completion(history, external_ai, gen_config, streamer);

        post_chat(history, gen_config, streamer);

        if (streamer)
            streamer->end();

        return r;
    }

    void Pipeline::abort_generation(void)
    {
        if (modelobj.loaded)
            modelobj.model->abort_generation();
    }

    void Pipeline::text_tokenize(const std::string &input, const GenerationConfig &gen_config, std::vector<int> &result)
    {
        tokenizer->encode(input, result);
    }

    void Pipeline::text_embedding(const std::string &input, const GenerationConfig &gen_config, std::vector<float> &result, BaseTokenizer::EmbeddingPurpose purpose)
    {
        if (!modelobj.loaded) return;
        std::vector<int> input_ids;
        tokenizer->encode_embedding(input, input_ids, purpose);
        model->text_embedding(gen_config, input_ids, result);
    }

    bool Pipeline::speech_synthesis(const std::string &input, const GenerationConfig &gen_config, std::vector<int16_t> &audio, int &sample_rate, int &channels)
    {
        if (!modelobj.loaded) return false;
        std::vector<int> input_ids;
        tokenizer->encode(input, input_ids);
        model->speech_synthesis(gen_config, input_ids, audio, sample_rate, channels);
        return true;
    }

    int Pipeline::get_text_embedding_dim(void)
    {
        if (!modelobj.loaded) return 0;
        return model->get_text_embedding_dim();
    }

    std::string Pipeline::get_additional_description(void) const
    {
        return "";
    }

    void Pipeline::restart(void)
    {
        initializing = true;
    }

    void Pipeline::rewind(int n_past)
    {
        if (!modelobj.loaded) return;
        if (model->get_n_past() > n_past)
            model->set_n_past(n_past);
    }

    int Pipeline::get_cursor(void)
    {
        if (!modelobj.loaded) return -1;
        return model->get_n_past();
    }

    int Pipeline::set_cursor(int pos)
    {
        if (!modelobj.loaded) return -1;
        rewind(pos);
        return get_cursor();
    }

    int Pipeline::save_session(const Messages &history, const std::string &file_name)
    {
        if (!modelobj.loaded) return -1000;

        FILE *f = fopen(file_name.c_str(), "wb");
        file_header header;
        memcpy(header.magic, head_magic, sizeof(header.magic));
        header.history_len = history.size();

        if (fwrite(&header, sizeof(header), 1, f) != 1)
            return -1;

        for (auto &s : history.history)
        {
            s.save(f);
        }

        model->save_session(f);

        fclose(f);
        return 0;
    }

    int Pipeline::load_session(Messages &history, const std::string &file_name, BaseStreamer *streamer, int *n_past)
    {
        if (!modelobj.loaded) return -1000;

        int r = -1;
        FILE *f = fopen(file_name.c_str(), "rb");
        file_header header;
        if (fread(&header, sizeof(header), 1, f) != 1)
            return -1;

        history.clear();

        if (memcmp(header.magic, head_magic, sizeof(header.magic)) == 0)
        {
            for (size_t i = 0; i < header.history_len; i++)
            {
                history.push_back(Message::load(&history, f));
                auto &last = history.back();
                if (streamer)
                {
                    if (MsgRole(last.role) == MsgRole::Assistant)
                        streamer->put_history_ai(last.content.to_string());
                    else
                        streamer->put_history_user(last.content.to_string());
                }
            }

            r = model->load_session(f);
            if (r != 0) goto exit;
        }
    exit:
        fclose(f);
        if (r == 0)
        {
            initializing = false;
            tokenizer->set_skip_sys_prompt(true);
            if (n_past != nullptr)
                *n_past = model->get_n_past();
        }
        history.move_cursor_to_end();
        return r;
    }

    float Pipeline::qa_rank(const std::string &q, const std::string &a, const GenerationConfig &gen_config)
    {
        if (!modelobj.loaded) return -1.0f;
        std::vector<int> input_ids;
        tokenizer->encode_qa(q, a, input_ids);
        return model->qa_rank(gen_config, input_ids);
    }

    void Pipeline::set_system_prompt(const std::string &prompt)
    {
        if (!modelobj.loaded) return;
        tokenizer->set_system_prompt(prompt);
    }

    void Pipeline::set_extending_method(ExtendingMethod method)
    {
        extending = method;
    }

    void Pipeline::set_additional_args(const std::map<std::string, std::string> &args)
    {
        if (!modelobj.loaded) return;
        tokenizer->set_additional_args(args);
        model->set_additional_args(args);
    }

    void Pipeline::before_chat(Messages &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
    }

    void Pipeline::post_chat(Messages &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
    }

    RAGPipeline::RAGPipeline(const std::string &path, const ModelObject::extra_args &args,
        DistanceStrategy vec_cmp, const std::map<std::string, std::vector<std::string>> &vector_stores,
        const std::string &embedding_model, const std::string &reranker_model)
        : Pipeline(path, args),
          composer(),
          hide_reference(false),
          retrieve_top_n(5), rerank_top_n(3),
          dump(false),
          rerank_score_threshold(0.5f),
          rag_post_extending(0),
          rerank_rewrite(false),
          vs(vec_cmp, vector_stores),
          embedding(embedding_model),
          reranker(reranker_model.size() > 0 ? new ModelObject(reranker_model) : nullptr),
          rewrite_model(nullptr)
    {
    }

    void RAGPipeline::rerank(const std::string &query, std::vector<int64_t> &candidates, const GenerationConfig &gen_config, int top_n)
    {
        std::vector<float> scores;
        std::vector<size_t> order;
        std::vector<int64_t> result;

        for (size_t i = 0; i < candidates.size(); i++)
        {
            std::vector<int> input_ids;
            std::string c, m;
            vs.get()->GetRecord(candidates[i], c, m);
            reranker->tokenizer->encode_qa(query, c, input_ids);
            scores.push_back(reranker->model->qa_rank(gen_config, input_ids));
        }

        utils::ordering(scores, order, true);

        if (top_n > (int)order.size()) top_n = (int)order.size();

        for (int i = 0; i < top_n; i++)
        {
            size_t index = order[i];
            if (scores[index] >= rerank_score_threshold)
                result.push_back(candidates[index]);
            else
                break;
        }

        candidates.clear();
        candidates.insert(candidates.begin(), result.begin(), result.end());
    }

    std::string RAGPipeline::rewrite_query(const std::string &prompt, const GenerationConfig &gen_config)
    {
        if (rewrite_model.get() == nullptr)
        {
            rewrite_model = std::unique_ptr<AbstractModel>(modelobj.fork_model(ModelObject::extra_args()));
            rewrite_model->set_tokenizer(tokenizer);
        }

        Messages history("", "");
        history.push_back(prompt, MsgRole::User);
        bool completed = false;

        std::vector<int> input_ids = tokenizer->encode_history(history, gen_config.max_context_length, false);
        std::vector<int> output_ids = rewrite_model->generate(input_ids, gen_config, false, completed, &performance, nullptr);

        if (!completed) return "";

        return tokenizer->decode(output_ids);
    }

    void RAGPipeline::before_chat(Messages &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        size_t index = history.size() - 1;
        std::string query(history.back().content);
        std::string rewritten_query(query);
        std::vector<float> query_emb;
        std::vector<int64_t> selected;

        CHATLLM_CHECK(!gen_config.reversed_role) << "you can't do reversed chat with RAG";

        metainfo.clear();

        if (modelobj.loaded && composer.is_rewritten_template_set())
        {
            std::string rewritten = composer.rewrite_query_for_retrieve(query);
            rewritten_query = rewrite_query(rewritten, gen_config);
            rewritten_query = composer.parse_rewritten_query_result(rewritten_query);
            if (rewritten_query.size() > 0)
                streamer->put_rewritten_query(rewritten_query);
            else
                rewritten_query = query;
        }

        if (!modelobj.loaded && (gen_config.emb_rank_query_sep.size() > 0))
        {
            auto pos = query.find(gen_config.emb_rank_query_sep);
            if (pos != std::string::npos)
            {
                rewritten_query = query.substr(0, pos);
                query.erase(0, pos + gen_config.emb_rank_query_sep.size());
            }
        }

        text_embedding(rewritten_query, gen_config, query_emb, BaseTokenizer::EmbeddingPurpose::Query);

        vs.get()->Query(query_emb, selected, retrieve_top_n);

        if (reranker != nullptr)
            rerank(rerank_rewrite ? rewritten_query : query, selected, gen_config, rerank_top_n);

        std::vector<std::string> augments;

        for (auto i : selected)
        {
            std::string c, m;
            vs.get()->GetRecord(i, c, m);
            augments.push_back(c);
            metainfo.push_back(m);

            size_t last = augments.size() - 1;

            if (rag_post_extending > 0)
            {
                for (auto j = i - 1; (j >= 0) && (j >= i - rag_post_extending); j--)
                {
                    std::string c0, m0;
                    vs.get()->GetRecord(j, c0, m0);
                    if (m0 == m)
                        augments[last] = c0 + "\n" + augments[last];
                    else
                        break;
                }

                for (auto j = i + 1; (j >= 0) && (j <= i + rag_post_extending); j++)
                {
                    std::string c0, m0;
                    vs.get()->GetRecord(j, c0, m0);
                    if (m0 == m)
                        augments[last] = augments[last] + "\n" + c0;
                    else
                        break;
                }
            }
        }

        if (augments.size() < 1) return;

        if (dump)
        {
            for (size_t i = 0; i < augments.size(); i++)
            {
                streamer->putln(metainfo[i]);
                streamer->putln(augments[i]);
                streamer->putln("");
            }
        }

        auto composed = composer.compose_augmented_query(query, augments);

        if (!modelobj.loaded)
            streamer->call_put_chunk(true, composed);

        history[index].content = composed;
    }

    bool RAGPipeline::select_vector_store(const std::string &name)
    {
        return vs.select(name);
    }

    std::string RAGPipeline::get_additional_description(void) const
    {
        std::ostringstream oss;

        int64_t total_param_num = embedding.model->get_param_num(false);

        oss << "Augmented by " << embedding.model->type_name() << " (" << std::fixed << std::setprecision(1) << (double)total_param_num / 1000000000. << "B)";

        if (reranker != nullptr)
        {
            total_param_num = reranker->model->get_param_num(false);
            oss << " and " << reranker->model->type_name() << " (" << std::fixed << std::setprecision(1) << (double)total_param_num / 1000000000. << "B)";
        }

        oss << ".";

        return oss.str();
    }

    void RAGPipeline::post_chat(Messages &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        if (hide_reference || (metainfo.size() < 1)) return;

        std::set<std::string> merged;

        for (auto &s : metainfo)
        {
            if (merged.find(s) != merged.end()) continue;

            merged.insert(s);
            streamer->put_reference(s);
        }
    }

    AugmentedQueryComposer::AugmentedQueryComposer()
        : prompt_template("Answer this question based on given information: {question}\n```\n{context}\n```"),
          context_sep("\n```\n")
    {}

    void replace_all(std::string &s, const std::string &sub, const std::string replace)
    {
        size_t pos = 0;
        while (true)
        {
            pos = s.find(sub, pos);
            if (pos == std::string::npos) {
                break;
            }

            s.erase(pos, sub.length());
            s.insert(pos, replace);

            pos += replace.length();
        }
    }

    void unescape_c_sequences(std::string &s)
    {
        replace_all(s, "\\n", "\n");
        replace_all(s, "\\t", "\t");
    }

    void GenerationConfig::set_ai_prefix(const std::string &prefix)
    {
        ai_prefix = prefix;
        unescape_c_sequences(ai_prefix);
    }

    std::string AugmentedQueryComposer::compose_augmented_query(const std::string &query, const std::vector<std::string> augments) const
    {
        const static std::regex r(R""([\r\n]+)"");

        if (augments.size() < 1) return query;

        std::ostringstream oss;
        oss << augments[0];
        for (size_t i = 1; i < augments.size(); i++)
            oss << context_sep << augments[i];

        std::string context = oss.str();

        std::string s(prompt_template);
        replace_all(s, "{context}", oss.str());
        replace_all(s, "{question}", query);
        return s;
    }

    bool AugmentedQueryComposer::is_rewritten_template_set(void) const
    {
        return query_rewritten_template.find("{question}", 0) != std::string::npos;
    }

    std::string AugmentedQueryComposer::rewrite_query_for_retrieve(const std::string &query) const
    {
        if (query_rewritten_template.size() < 1)
            return query;

        std::string s(query_rewritten_template);
        replace_all(s, "{question}", query);
        return s;
    }

    // TODO: this might not work because:
    // 1. some LLM does not support this;
    // 2. output format varies.
    std::string AugmentedQueryComposer::parse_rewritten_query_result(const std::string &query) const
    {
        std::string r(query);
        size_t pos = 0;
        tokenizer::TextTrim t;
        const char *delimiters[] = {":", "："};
        bool found = false;

        pos = r.find("\n");
        if (pos != std::string::npos)
        {
            // xxx:
            // ...
            // ...
            r = r.substr(pos + 1);
            std::ostringstream oss;

            while (r.size() > 0)
            {
                std::string part;
                pos = r.find("\n");
                if (pos != std::string::npos)
                {
                    part = r.substr(0, pos);
                    r = r.substr(pos + 1);
                }
                else
                {
                    part = r;
                    r = "";
                }

                pos = part.find(" ");
                if (pos != std::string::npos)
                    part = part.substr(pos + 1);

                part = t.transform(part);

                oss << part << " ";
            }

            r = oss.str();
            r = t.transform(r);

            goto final;
        }

        for (const char *delimiter : delimiters)
        {
            pos = r.find(delimiter);
            if (pos != std::string::npos)
            {
                r = r.substr(pos + strlen(delimiter));
                found = true;
                break;
            }
        }

        if (found)
        {
            replace_all(r, ",", " ");
            replace_all(r, "，", " ");
            replace_all(r, ".", " ");

            r = t.transform(r);

            goto final;
        }

        if ((pos = r.find(",")) != std::string::npos)
        {
            replace_all(r, ",", " ");
            replace_all(r, "，", " ");
            replace_all(r, ".", " ");

            r = t.transform(r);

            goto final;
        }

        return "";

final:
        tokenizer::TextPrepDeleteMultiSpaces del;
        return del.transform(r);
    }

    void AugmentedQueryComposer::set_prompt_template(const std::string &s)
    {
        if (s.size() < 1) return;
        prompt_template = s;
        unescape_c_sequences(prompt_template);
    }

    void AugmentedQueryComposer::set_rewrite_template(const std::string &s)
    {
        if (s.size() < 1) return;
        query_rewritten_template = s;
        unescape_c_sequences(query_rewritten_template);
    }

    void AugmentedQueryComposer::set_context_sep(const std::string &s)
    {
        if (s.size() < 1) return;
        context_sep = s;
        unescape_c_sequences(context_sep);
    }

    ModelPerfInfo::ModelPerfInfo()
    {
        memset(&timings, 0, sizeof(timings));
    }

    void ModelPerfInfo::Accumulate(Type type, size_t tok_count)
    {
        timings[type].tok_count += tok_count;
        timings[type].duration_ms += Elapsed();
    }

    void ModelPerfInfo::Reset(void)
    {
        m_beg = Clock::now();
    }

    double ModelPerfInfo::Elapsed(void)
    {
        double r = std::chrono::duration_cast<MilliSecond>(Clock::now() - m_beg).count();
        Reset();
        return r;
    }

    VectorStores::VectorStores(DistanceStrategy vec_cmp, const std::map<std::string, std::vector<std::string>> &vector_stores)
        : def_store(nullptr)
    {
        for (auto x : vector_stores)
        {
            auto p = new CVectorStore(vec_cmp, x.second);
            if (nullptr == def_store) def_store = p;

            stores.insert(std::pair(x.first, p));
        }
    }

    VectorStores::~VectorStores()
    {
        for (auto x : stores) delete x.second;
    }

    CVectorStore *VectorStores::get(const std::string &name)
    {
        auto vs = stores.find(name);
        return vs == stores.end() ? nullptr : vs->second;
    }

    bool VectorStores::select(const std::string &name)
    {
        auto p = get(name);
        if (p)
        {
            def_store = p;
            return true;
        }

        return false;
    }

    CVectorStore *VectorStores::get()
    {
        return def_store;
    }

    BeamSearchPipeline::Beam::Beam(int vocab_size, int max_length, BaseTokenizer *tokenizer)
        : score(0.0f), completed(false), max_length(max_length), tokenizer(tokenizer)
    {
        scores.resize(vocab_size, 0.0f);
    }

    void BeamSearchPipeline::Beam::clear(const std::vector<int> &init_ids)
    {
        completed = false;
        trace.clear();
        trace.insert(trace.end(), init_ids.begin(), init_ids.end());
        score = 0.0f;
    }

    void BeamSearchPipeline::Beam::add(int token_id, float score)
    {
        CHATLLM_CHECK(!completed) << "Beam::add() called on a completed beam";
        trace.push_back(token_id);
        this->score = score;
        completed = tokenizer->is_terminate_token_id(token_id) || ((max_length > 0) && ((int)trace.size() >= max_length));
    }

    int BeamSearchPipeline::Beam::get_last_token(void) const
    {
        return trace[trace.size() - 1];
    }

    void BeamSearchPipeline::Beam::set_max_length(int max_length)
    {
        this->max_length = max_length;
    }

    static void log_soft_max(std::vector<float> &vector)
    {
        float max_score = *std::max_element(vector.begin(), vector.end());
        float sum = 0.f;
        for (float &s : vector)
        {
            float s1 = std::exp(s - max_score);
            sum += s1;
        }
        float inv_sum = std::log(1.f / sum);
        for (float& s : vector)
            s = s - max_score + inv_sum;
    }

    void BeamSearchPipeline::Beam::refresh_scores(void)
    {
        log_soft_max(scores);
        for (auto& s : scores)
            s += score;
    }

    BeamSearchPipeline::BeamSearchPipeline(const std::string &path, const ModelObject::extra_args &args, int num_beams)
        : Pipeline(path, args)
    {
        for (int i = 0; i < num_beams; i++)
            beams.emplace_back(tokenizer->get_vocab_size(), 0, tokenizer);
    }

    static void topk_sampling(const std::vector<float> &logits, int top_k, std::vector<int> &selected_ids)
    {
        struct TokenIdScore
        {
            int id;
            float score;

            bool operator<(const TokenIdScore &other) const { return score < other.score; }
            bool operator>(const TokenIdScore &other) const { return score > other.score; }
        };

        const int vocab_size = (int)logits.size();
        std::vector<TokenIdScore> token_scores;

        token_scores.reserve(vocab_size);
        token_scores.resize(vocab_size);

        for (int i = 0; i < vocab_size; i++)
        {
            token_scores[i] = {.id = i, .score = logits[i]};
        }

        // top_k sampling
        std::nth_element(token_scores.begin(), token_scores.begin() + top_k, token_scores.end(),
                         std::greater<TokenIdScore>());
        token_scores.resize(top_k);

        selected_ids.clear();
        for (int i = 0; i < top_k; i++)
        {
            selected_ids.push_back(token_scores[i].id);
        }
    }

    void BeamSearchPipeline::do_chat0(Messages &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        std::vector<int> input_ids = tokenizer->encode_history(history, gen_config.max_context_length, false, true, gen_config.reversed_role);

        for (auto &beam : beams)
        {
            beam.clear({});
            beam.set_max_length(gen_config.max_length);
        }
        model->set_n_past(0);

        std::vector<float> lm_logits;
        model->generate_next_token(input_ids, gen_config, lm_logits);
        model->set_n_past((int)input_ids.size());

        log_soft_max(lm_logits);

        std::vector<int> selected_ids;
        topk_sampling(lm_logits, (int)beams.size(), selected_ids);

        for (int i = 0; i < (int)beams.size(); i++)
        {
            beams[i].add(selected_ids[i], lm_logits[selected_ids[i]]);
            model->save_session(beams[i].session);
        }
    }

    void BeamSearchPipeline::do_chat(Messages &history, const GenerationConfig &gen_config, BaseStreamer *streamer)
    {
        bool completed = false;
        std::vector<int> input_ids;

        do_chat0(history, gen_config, streamer);

        while (!completed)
        {
            completed = true;

            for (auto &beam : beams)
            {
                if (beam.completed)
                    continue;

                completed = false;
                std::vector<int> input_ids = {beam.get_last_token()};

                model->load_session(beam.session);
                model->generate_next_token(input_ids, gen_config, beam.scores);
                model->set_n_past(model->get_n_past() + (int)input_ids.size());
                model->save_session(beam.session);

                beam.refresh_scores();
            }

            if (completed)
                break;

            std::vector<int> selected_ids;
            std::vector<int> remaining_beams;
            std::vector<float> all_logits;
            for (int i = 0; i < (int)beams.size(); i++)
            {
                if (beams[i].completed)
                    continue;
                remaining_beams.push_back(i);

                all_logits.insert(all_logits.end(), beams[i].scores.begin(), beams[i].scores.end());
            }

            topk_sampling(all_logits, (int)beams.size(), selected_ids);

            std::vector<std::vector<int>> selected_traces;
            for (int i = 0; i < (int)selected_ids.size(); i++)
            {
                int beam_id = selected_ids[i] / tokenizer->get_vocab_size();
                beam_id = remaining_beams[beam_id];

                selected_traces.push_back({});

                selected_traces[i].insert(selected_traces[i].end(), beams[beam_id].trace.begin(), beams[beam_id].trace.end());
            }

            std::set<int> updated_beams;
            std::vector<int> candidates;
            for (int i = 0; i < (int)remaining_beams.size(); i++)
            {
                int beam_id = selected_ids[i] / tokenizer->get_vocab_size();
                beam_id = remaining_beams[beam_id];
                if (updated_beams.find(beam_id) != updated_beams.end())
                {
                    candidates.push_back(i);
                    continue;
                }

                updated_beams.emplace(beam_id);

                int tok_id = selected_ids[i] % tokenizer->get_vocab_size();
                beams[beam_id].add(tok_id, beams[beam_id].scores[tok_id]);
            }

            int candidate_id = 0;
            for (int i = 0; i < (int)beams.size(); i++)
            {
                if ((beams[i].completed) || (updated_beams.find(i) != updated_beams.end()))
                    continue;
                updated_beams.emplace(i);

                const int selection = candidates[candidate_id++];

                int beam_id = selected_ids[selection] / tokenizer->get_vocab_size();
                beam_id = remaining_beams[beam_id];
                int tok_id = selected_ids[selection] % tokenizer->get_vocab_size();

                beams[i].clear(selected_traces[selection]);
                beams[i].add(tok_id, beams[beam_id].scores[tok_id]);
                beams[i].session.copy_from(beams[beam_id].session);
            }

            int selection = (int)remaining_beams.size();
            for (int i = 0; i < (int)beams.size(); i++)
            {
                if (updated_beams.find(i) != updated_beams.end()) continue;

                int beam_id = selected_ids[selection] / tokenizer->get_vocab_size();
                beam_id = remaining_beams[beam_id];
                int tok_id = selected_ids[selection] % tokenizer->get_vocab_size();

                if (beams[i].score > beams[beam_id].scores[tok_id]) continue;

                beams[i].clear(selected_traces[selection]);
                beams[i].add(tok_id, beams[beam_id].scores[tok_id]);
                beams[i].session.copy_from(beams[beam_id].session);

                selection++;
            }
        }
    }

    std::string BeamSearchPipeline::chat(Messages &history, const GenerationConfig &gen_config,
                                         BaseStreamer *streamer)
    {
        std::string r;

        before_chat(history, gen_config, streamer);

        if (modelobj.loaded)
        {
            do_chat(history, gen_config, streamer);

            if (streamer)
            {
                std::vector<size_t> order;
                utils::ordering<BeamSearchPipeline::Beam, float>(beams, order, [=](const auto &a) {return a.score; }, true);
                for (auto &id: order)
                {
                    auto beam = beams[id];
                    std::ostringstream oss;
                    oss << beam.score << "," << tokenizer->decode(beam.trace);
                    streamer->putln(oss.str(), BaseStreamer::TextType::BEAM_SEARCH);
                }
            }
        }

        post_chat(history, gen_config, streamer);

        if (streamer)
            streamer->end();

        return r;
    }

} // namespace chatllm
