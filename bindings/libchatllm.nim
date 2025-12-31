import std/[os, tables, json, strformat, strutils, sequtils, httpclient, asyncdispatch, algorithm]

type
    PrintType* = enum
        PRINT_CHAT_CHUNK = 0,           ##  below items share the same value with BaseStreamer::TextType
        PRINTLN_META = 1,               ##  print a whole line: general information
        PRINTLN_ERROR = 2,              ##  print a whole line: error message
        PRINTLN_REF = 3,                ##  print a whole line: reference
        PRINTLN_REWRITTEN_QUERY = 4,    ##  print a whole line: rewritten query
        PRINTLN_HISTORY_USER = 5,       ##  print a whole line: user input history
        PRINTLN_HISTORY_AI = 6,         ##  print a whole line: AI output history
        PRINTLN_TOOL_CALLING = 7,       ##  print a whole line: tool calling (supported by only a few models)
        PRINTLN_EMBEDDING = 8,          ##  print a whole line: embedding (example: "0.1,0.3,...")
        PRINTLN_RANKING = 9,            ##  print a whole line: ranking (example: "0.8")
        PRINTLN_TOKEN_IDS = 10,         ##  print a whole line: token ids (example: "1,3,5,8,...")
        PRINTLN_LOGGING =11,            ##  print a whole line: internal logging with the first char indicating level
                                        ##  (space): None; D: Debug; I: Info; W: Warn; E: Error; .: continue
        PRINTLN_BEAM_SEARCH =12,        ##  print a whole line: a result of beam search with a prefix of probability
                                        ##  (example: "0.8,....")
        RINTLN_MODEL_INFO =13,          ##  when a model is started, print a whole line of basic model information (json format)
                                        ##  (example: {"name": "llama", "context_length": 100, "capabilities": [text, ...], ...})
        PRINT_THOUGHT_CHUNK     =14,    ## same as PRINT_CHAT_CHUNK, but this from "thoughts".
                                        ## possible leading or trailing tags (such as <think>, </think>) are removed.
                                        ## use `+detect_thoughts` to enable this.

        PRINT_EVT_ASYNC_COMPLETED       = 100   ##  last async operation completed (utf8_str is "")
        PRINT_EVT_THOUGHT_COMPLETED     = 101,  ## thought completed

type
    chatllm_obj* = object
    f_chatllm_print* = proc (user_data: pointer; print_type: cint; utf8_str: cstring) {.cdecl.}
    f_chatllm_end* = proc (user_data: pointer) {.cdecl.}


when defined(windows):
    const libName = "libchatllm.dll"
elif defined(macosx):
    const libName = "libchatllm.dylib"
else:
    const libName = "libchatllm.so"

##
## @brief create ChatLLM object
##
## @return                  the object
##
proc chatllm_create*(): ptr chatllm_obj {.stdcall, dynlib: libName, importc.}

##
##  @brief append a command line option
##
##  @param[in] obj               model object
##  @param[in] utf8_str          a command line option
##
proc chatllm_append_param*(obj: ptr chatllm_obj; utf8_str: cstring) {.stdcall, dynlib: libName, importc.}

##
##  @brief start
##
##  @param[in] obj               model object
##  @param[in] f_print           callback function for printing
##  @param[in] f_end             callback function when model generation ends
##  @param[in] user_data         user data provided to callback functions
##  @return                      0 if succeeded
##
proc chatllm_start*(obj: ptr chatllm_obj; f_print: f_chatllm_print;
                    f_end: f_chatllm_end; user_data: pointer): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief set max number of generated tokens in a new round of conversation
##
##  @param[in] obj               model object
##  @param[in] gen_max_tokens    -1 for as many as possible
##
proc chatllm_set_gen_max_tokens*(obj: ptr chatllm_obj; gen_max_tokens: cint) {.stdcall, dynlib: libName, importc.}

##
##  @brief restart (i.e. discard history)
##
##  * When a session has been loaded, the model is restarted to the point that the session is loaded;
##
##       Note: this would not work if `--extending` is not `none` or the model uses SWA.
##
##  * Otherwise, it is restarted from the very beginning.
##
##  @param[in] obj               model object
##  @param[in] utf8_sys_prompt   update to a new system prompt
##                               if NULL, then system prompt is kept unchanged.
##
proc chatllm_restart*(obj: ptr chatllm_obj; utf8_sys_prompt: cstring) {.stdcall, dynlib: libName, importc.}

##
## @brief prepare to generate a multimedia input, i.e. clear previously added pieces.
##
## Each `chatllm_obj` has a global multimedia message object, which can be used as user input,
## or chat history, etc.
##
## @param[in] obj               model object
## @return                      0 if succeeded
##
proc chatllm_multimedia_msg_prepare*(obj: ptr chatllm_obj) {.stdcall, dynlib: libName, importc.}

##
## @brief add a piece to a multimedia message
##
## Remember to clear the message by `chatllm_multimedia_msg_prepare` when starting a new message.
##
## @param[in] obj               model object
## @param[in] type              type ::= "text" | "image" | "video" | "audio" | ...
## @param[in] utf8_str          content, i.e. utf8 text content, or base64 encoded data of multimedia data.
## @return                      0 if succeeded
##
proc chatllm_multimedia_msg_append*(obj: ptr chatllm_obj; content_type: cstring; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

type
    RoleType* = enum
        ROLE_USER = 2,
        ROLE_ASSISTANT = 3,
        ROLE_TOOL = 4,

##
## @brief push back a message to the end of chat history.
##
## This can be used to restore session after `chatllm_restart`.
## This would not trigger generation. Use `chatllm_user_input`, etc  to start generation.
##
## @param[in] obj               model object
## @param[in] role_type         message type (see `RoleType`)
## @param[in] utf8_str          content
##
proc chatllm_history_append*(obj: ptr chatllm_obj; role_type: cint; utf8_str: cstring) {.stdcall, dynlib: libName, importc.}

##
## @brief push back current multimedia message to the end of chat history.
##
## see `chatllm_history_append`
##
## @param[in] obj               model object
## @param[in] role_type         message type (see `RoleType`)
## @return                      >= 0 if success else < 0
##
proc chatllm_history_append_multimedia_msg*(obj: ptr chatllm_obj; role_type: cint): cint {.stdcall, dynlib: libName, importc.}

##
## @brief brief get current position of "cursor": total number of processed/generated tokens
##
## Possible use case: token usage statistics.
##
## @param[in] obj               model object
## @return                      position of cursor
##
proc chatllm_get_cursor*(obj: ptr chatllm_obj): cint {.stdcall, dynlib: libName, importc.}

##
## @brief set current position of "cursor"
##
## Possible use case: rewind and re-generate.
##
## Note: once used, the history in save session is not reliable any more.
##
## @param[in] obj               model object
## @return                      position of cursor
##
proc chatllm_set_cursor(obj: ptr chatllm_obj, pos: cint): int {.stdcall, dynlib: libName, importc.}

##
##  @brief user input
##
##  This function is synchronized, i.e. it returns after model generation ends and `f_end` is called.
##
##  @param[in] obj               model object
##  @param[in] utf8_str          user input
##  @return                      0 if succeeded
##
proc chatllm_user_input*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
## @brief take current multimedia message as user input and run
##
## This function is synchronized, i.e. it returns after model generation ends and `f_end` is called.
##
## @param[in] obj               model object
## @return                      0 if succeeded
##
proc chatllm_user_input_multimedia_msg*(obj: ptr chatllm_obj): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief set prefix for AI generation
##
##  This prefix is used in all following rounds..
##
##  @param[in] obj               model object
##  @param[in] utf8_str          prefix
##  @return                      0 if succeeded
##
proc chatllm_set_ai_prefix*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief tool input
##
##  - If this function is called before `chatllm_user_input` returns, this is asynchronized,
##  - If this function is called after `chatllm_user_input` returns, this is equivalent to
##    `chatllm_user_input`.
##
##  @param[in] obj               model object
##  @param[in] utf8_str          user input
##  @return                      0 if succeeded
##
proc chatllm_tool_input*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief feed in text generated by external tools
##
##  This text is treated as part of AI's generation. After this is called, LLM generation
##  is continued.
##
##  Example:
##
##  ```c
##  // in `f_print` callback:
##  chatllm_abort_generation();
##  chatllm_tool_completion(...);
##  ```
##
##  @param[in] obj               model object
##  @param[in] utf8_str          text
##  @return                      0 if succeeded
##
proc chatllm_tool_completion*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief tokenize
##
##  token ids are emitted through `PRINTLN_TOKEN_IDS`.
##
##  @param[in] obj               model object
##  @param[in] utf8_str          text
##  @return                      number of ids if succeeded. otherwise -1.
##
proc chatllm_text_tokenize*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

type
    EmbeddingPurpose* = enum
        EMBEDDING_FOR_DOC   = 0,    # for document
        EMBEDDING_FOR_QUERY = 1,    # for query

##
## @brief text embedding
##
## embedding is emitted through `PRINTLN_EMBEDDING`.
##
## Note: Not all models support specifying purpose.(see _Qwen3-Embedding_).
##
## @param[in] obj               model object
## @param[in] utf8_str          text
## @param[in] purpose           purpose, see `EmbeddingPurpose`
## @return                      0 if succeeded
##
proc chatllm_text_embedding*(obj: ptr chatllm_obj; utf8_str: cstring; purpose: cint): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief question & answer ranking
##
##  embedding is emit through `PRINTLN_RANKING`.
##
##  @param[in] obj               model object
##  @param[in] utf8_str_q        question
##  @param[in] utf8_str_q        answer
##  @return                      0 if succeeded
##
proc chatllm_qa_rank*(obj: ptr chatllm_obj; utf8_str_q: cstring;
                      utf8_str_a: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief switching RAG vector store
##
##  @param[in] obj               model object
##  @param[in] name              vector store name
##  @return                      0 if succeeded
##
proc chatllm_rag_select_store*(obj: ptr chatllm_obj; name: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief abort generation
##
##  This function is asynchronized, i.e. it returns immediately.
##
##  @param[in] obj               model object
##
proc chatllm_abort_generation*(obj: ptr chatllm_obj) {.stdcall, dynlib: libName, importc.}

##
##  @brief show timing statistics
##
##  Result is sent to `f_print`.
##
##  @param[in] obj               model object
##
proc chatllm_show_statistics*(obj: ptr chatllm_obj) {.stdcall, dynlib: libName, importc.}

##
##  @brief save current session on demand
##
##  Note: Call this from the same thread of `chatllm_user_input()`.
##
##  If chat history is empty, then system prompt is evaluated and saved.
##
##  @param[in] obj               model object
##  @param[in] utf8_str          file full name
##  @return                      0 if succeeded
##
proc chatllm_save_session*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief load a session on demand
##
##  Note: Call this from the same thread of `chatllm_user_input()`.
##
##  @param[in] obj               model object
##  @param[in] utf8_str          file full name
##  @return                      0 if succeeded
##
proc chatllm_load_session*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief get integer result of last async operation
##
##  @param[in] obj               model object
##  @return                      last result (if async is still ongoing, INT_MIN)
##
proc chatllm_get_async_result_int*(obj: ptr chatllm_obj): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_start`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_start*(obj: ptr chatllm_obj; f_print: f_chatllm_print;
                          f_end: f_chatllm_end; user_data: pointer): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_user_input`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_user_input*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
## @brief async version of `chatllm_user_input_multimedia_msg`
##
## @param   ...
## @return                      0 if started else -1
##
proc chatllm_async_user_input_multimedia_msg*(obj: ptr chatllm_obj): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_tool_input`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_tool_input*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_tool_completion`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_tool_completion*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_text_embedding`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_text_embedding*(obj: ptr chatllm_obj; utf8_str: cstring; purpose: cint): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_qa_rank`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_qa_rank*(obj: ptr chatllm_obj; utf8_str_q: cstring;
                            utf8_str_a: cstring): cint {.stdcall, dynlib: libName, importc.}

func is_same_command_option(a, b: string): bool =
    if len(a) != len(b): return false
    for i in 0 ..< len(a):
        var c1 = a[i]
        var c2 = b[i]
        if c1 == '-': c1 = '_'
        if c2 == '-': c2 = '_'
        if c1 != c2: return false
    return true

func is_same_command_option(a: string, options: openArray[string]): bool =
    for s in options:
        if a.is_same_command_option(s): return true
    return false

var all_models: JsonNode = nil

proc get_model(model_id: string; storage_dir: string): string =
    if not os.dirExists(storage_dir):
        os.createDir(storage_dir)

    func calc_wer[T](ref_words, hyp_words: openArray[T]): float =
        var d = newSeq[seq[int]](len(ref_words) + 1)
        for i in 0 ..< len(d): d[i] = newSeq[int](len(hyp_words) + 1)

        for i in 0..len(ref_words):
            d[i][0] = i
        for j in 0..len(hyp_words):
            d[0][j] = j
        for i in 1..len(ref_words):
            for j in 1..len(hyp_words):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    let substitution = d[i - 1][j - 1] + 1
                    let insertion    = d[i    ][j - 1] + 1
                    let deletion     = d[i - 1][j    ] + 1
                    d[i][j] = min([substitution, insertion, deletion])
        let wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return wer

    func calc_cer(ref_str, hyp: string): float = calc_wer(ref_str.toSeq(), hyp.toSeq())

    func find_nearest_item(s: string; candidates: openArray[string]): seq[string] =
        var l = candidates.sortedByIt(calc_cer(s, it))
        return l[0 ..< min(3, len(l))]

    proc get_model_url_on_modelscope(url: seq[string]): string =
        let proj = url[0]
        let fn = url[1]
        let user = if len(url) >= 3: url[2] else: "judd2024"

        return fmt"https://modelscope.cn/api/v1/models/{user}/{proj}/repo?Revision=master&FilePath={fn}"

    proc print_progress_bar(iteration: BiggestInt, total: BiggestInt, prefix = "", suffix = "", decimals = 1, length = 60, fill = "█", printEnd = "\r", auto_nl = true) =
        let percent = formatFloat(100.0 * (iteration.float / total.float), ffDecimal, decimals)
        let filledLength = int(length.float * iteration.float / total.float)
        let bar = fill.repeat(filledLength) & '-'.repeat(length - filledLength)
        stdout.write(fmt"{printEnd}{prefix} |{bar}| {percent}% {suffix}")
        if iteration == total and auto_nl:
            echo ""

    proc parse_model_id(model_id: string): JsonNode =
        let parts = model_id.split(":")
        if all_models == nil:
            let fn = joinPath([parentDir(paramStr(0)), "../scripts/models.json"])
            const compiled_file = readFile(currentSourcePath.parentDir() & "/../scripts/models.json")
            all_models = if fileExists(fn): json.parseFile(fn) else: json.parseJson(compiled_file)

        let id = parts[0]
        if not all_models.contains(id):
            let guess = find_nearest_item(id, all_models.keys().toSeq())
            raise newException(ValueError, fmt"""`{id}` is recognized as a model id. Did you mean something like `{guess.join(", ")}`?""")

        let model = all_models[id]
        let variants = model["variants"]
        let variant = variants[if len(parts) >= 2: parts[1] else: model["default"].getStr()]
        let r = variant["quantized"][variant["default"].getStr()].copy()
        let url = r["url"].getStr().split("/")
        r["url"] = json.newJString(get_model_url_on_modelscope(url))
        r["fn"] = json.newJString(url[1])
        return r

    proc download_file(url: string, fn: string, prefix: string) =
        echo fmt"Downloading {prefix}"
        let client = newAsyncHttpClient()
        defer: client.close()

        proc onProgressChanged(total, progress, speed: BiggestInt) {.async} =
            print_progress_bar(progress, total, prefix)

        client.onProgressChanged = onProgressChanged
        client.downloadFile(url, fn).waitFor()

    let info = parse_model_id(model_id)
    assert info != nil, fmt"unknown model id {model_id}"

    let fn = joinPath([storage_dir, info["fn"].getStr()])
    if os.fileExists(fn):
        if os.getFileSize(fn) == info["size"].getBiggestInt():
            return fn
        else:
            echo(fmt"{fn} is incomplete, download again")

    download_file(info["url"].getStr(), fn, model_id)
    assert (os.fileExists(fn)) and (os.getFileSize(fn) == info["size"].getBiggestInt())
    print_progress_bar(100, 100)

    return fn

## Streamer in OOP style
type
    FrontendOptions* = object
        help*: bool = false
        interactive*: bool = false
        reversed_role*: bool = false
        use_multiple_lines*: bool = false
        prompt*: string
        sys_prompt*: string

    StreamerMessageType = enum
        Done = 0,
        Chunk = 1,
        ThoughtChunk = 2,
        ThoughtDone = 3,

    StreamerMessage = tuple[t: StreamerMessageType, chunk: string]

    ChunkType* = enum
        Chat = 0
        Thought = 1

    Streamer* = ref object of RootObj
        llm*: ptr chatllm_obj
        auto_restart: bool
        system_prompt*: string
        system_prompt_updating: bool
        acc*: string
        thought_acc*: string
        is_generating*: bool
        input_id: int
        tool_input_id: int
        references: seq[string]
        rewritten_query: string
        result_embedding*: string
        result_ranking*: string
        result_token_ids*: string
        result_beam_search: seq[string]
        model_info*: string
        fe_options*: FrontendOptions
        chan_output: Channel[StreamerMessage]

var streamer_dict = initTable[int, Streamer]()

proc get_streamer(id: pointer): Streamer =
    return streamer_dict[cast[int](id)]

method on_call_tool(streamer: Streamer, query: string) {.base.} =
    raise newException(IOError, "call_tool not implemented (overrided)!")

method on_logging(streamer: Streamer, text: string) {.base.} =
    discard

method on_error(streamer: Streamer, text: string) {.base.} =
    raise newException(IOError, "Error: " & text)

method on_thought_completed(streamer: Streamer) {.base.} =
    discard

method on_print_meta(streamer: Streamer, text: string) {.base.} =
    discard

proc streamer_on_end(user_data: pointer) {.cdecl.} =
    var streamer = get_streamer(user_data)
    streamer.is_generating = false

proc streamer_on_print(user_data: pointer, print_type: cint, utf8_str: cstring) {.cdecl.} =
    var streamer = get_streamer(user_data)
    case cast[PrintType](print_type):
        of PrintType.PRINT_CHAT_CHUNK:
            streamer.chan_output.send((t: StreamerMessageType.Chunk, chunk: $utf8_str))
        of PrintType.PRINTLN_META:
            streamer.on_print_meta $utf8_str
        of PrintType.PRINTLN_ERROR:
            on_error(streamer, $utf8_str)
        of PrintType.PRINTLN_REF:
            streamer.references.add $utf8_str
        of PrintType.PRINTLN_REWRITTEN_QUERY:
            streamer.rewritten_query = $utf8_str
        of PrintType.PRINTLN_HISTORY_USER:
            discard
        of PrintType.PRINTLN_HISTORY_AI:
            discard
        of PrintType.PRINTLN_TOOL_CALLING:
            on_call_tool(streamer, $utf8_str)
        of PrintType.PRINTLN_EMBEDDING:
            streamer.result_embedding = $utf8_str
        of PrintType.PRINTLN_RANKING:
            streamer.result_ranking = $utf8_str
        of PrintType.PRINTLN_TOKEN_IDS:
            streamer.result_token_ids = $utf8_str
        of PrintType.PRINTLN_LOGGING:
            on_logging(streamer, $utf8_str)
        of PrintType.PRINTLN_BEAM_SEARCH:
            streamer.result_beam_search.add $utf8_str
        of PrintType.RINTLN_MODEL_INFO:
            streamer.model_info = $utf8_str
        of PrintType.PRINT_THOUGHT_CHUNK:
            streamer.chan_output.send((t: StreamerMessageType.ThoughtChunk, chunk: $utf8_str))
        of PrintType.PRINT_EVT_ASYNC_COMPLETED:
            streamer.is_generating = false
            streamer.chan_output.send((t: StreamerMessageType.Done, chunk: ""))
        of PrintType.PRINT_EVT_THOUGHT_COMPLETED:
            streamer.chan_output.send((t: StreamerMessageType.ThoughtDone, chunk: ""))

proc initStreamer*(streamer: Streamer; args: openArray[string], auto_restart: bool = false): bool =
    const candidates = ["-m", "--model", "--embedding_model", "--reranker_model"]

    var storage_dir = getEnv("CHATLLM_QUANTIZED_MODEL_PATH")
    if storage_dir == "":
        storage_dir = joinPath([parentDir(paramStr(0)), "../quantized"])

    let id = streamer_dict.len + 1
    streamer_dict[id] = streamer

    streamer.chan_output.open()
    streamer.system_prompt = ""
    streamer.system_prompt_updating = false
    streamer.auto_restart = auto_restart
    streamer.is_generating = false
    streamer.input_id = 0
    streamer.tool_input_id = 0
    streamer.references = @[]
    streamer.result_embedding = ""
    streamer.result_ranking = ""
    streamer.result_token_ids = ""
    streamer.model_info = ""

    var args_pp = newSeq[string]()
    var i = 0
    while i < len(args):
        let s = args[i]
        if s.is_same_command_option(["-h", "--help"]):
            streamer.fe_options.help = true
            break
        elif s.is_same_command_option(["-i", "--interactive"]):
            streamer.fe_options.interactive = true
        elif s.is_same_command_option("--reversed_role"):
            streamer.fe_options.reversed_role = true
        elif s.is_same_command_option("--multi"):
            streamer.fe_options.use_multiple_lines = true
        elif s.is_same_command_option(["-p", "--prompt"]):
            inc i
            if i < len(args): streamer.fe_options.prompt = args[i]
        elif s.is_same_command_option(["-s", "--system"]):
            inc i
            if i < len(args): streamer.fe_options.sys_prompt = args[i]
        else:
            args_pp.add s
            if s.is_same_command_option(candidates):
                inc i
                if i >= len(args): break
                if args[i][0] == ':':
                    args_pp.add get_model(args[i][1..^1], storage_dir)
                else:
                    args_pp.add args[i]
        inc i

    if streamer.fe_options.help: return true

    streamer.llm = chatllm_create()
    for s in args_pp:
        chatllm_append_param(streamer.llm, s.cstring)

    let r = chatllm_start(streamer.llm, streamer_on_print, streamer_on_end, cast[pointer](id))
    result = r == 0

proc newStreamer*(args: openArray[string], auto_restart: bool = false): Streamer =
    var streamer: Streamer
    new(streamer)
    let r = initStreamer(streamer, args, auto_restart)
    result = if r: streamer else: nil

proc set_system_prompt*(streamer: Streamer, prompt: string) =
    if streamer.system_prompt == prompt: return
    streamer.system_prompt = prompt
    streamer.system_prompt_updating = true

proc abort*(streamer: Streamer) =
    chatllm_abort_generation(streamer.llm)

method restart*(streamer: Streamer) {.base gcsafe.} =
    if not streamer.is_generating:
        chatllm_restart(streamer.llm, if streamer.system_prompt_updating: streamer.system_prompt.cstring else: nil)

proc clear(chan: var Channel[StreamerMessage]) =
    while chan.tryRecv().dataAvailable:
        discard

proc flush*(streamer: Streamer) =
    streamer.chan_output.clear()

proc start_chat*(streamer: Streamer, user_input: string): bool =
    if streamer.is_generating:
        return false
    inc streamer.input_id
    if streamer.auto_restart or streamer.system_prompt_updating:
        streamer.restart()
    else:
        discard
    streamer.acc = ""
    streamer.thought_acc = ""
    streamer.references = @[]
    streamer.result_embedding = ""
    streamer.result_ranking = ""
    streamer.result_token_ids = ""
    streamer.result_beam_search = @[]
    streamer.chan_output.clear()
    result = chatllm_async_user_input(streamer.llm, user_input.cstring) == 0
    if result:
        streamer.is_generating = true

iterator chunks*(streamer: Streamer): tuple[t: ChunkType; chunk: string] =
    while true:
        let msg = streamer.chan_output.recv()
        case msg.t:
            of StreamerMessageType.Chunk:
                streamer.acc &= msg.chunk
                yield (t: ChunkType.Chat, chunk: msg.chunk)
            of StreamerMessageType.ThoughtChunk:
                streamer.thought_acc &= msg.chunk
                yield (t: ChunkType.Thought, chunk: msg.chunk)
            of StreamerMessageType.Done:
                break
            of StreamerMessageType.ThoughtDone:
                streamer.on_thought_completed()

proc set_max_gen_tokens*(streamer: Streamer, max_new_tokens: int) =
    chatllm_set_gen_max_tokens(streamer.llm, cint(max_new_tokens))

proc id*(streamer: Streamer): int = streamer.input_id

proc busy*(streamer: Streamer): bool = streamer.is_generating

proc get_cursor*(streamer: Streamer): int =
    result = chatllm_get_cursor(streamer.llm)

proc set_cursor*(streamer: Streamer, pos: int): int =
    result = chatllm_set_cursor(streamer.llm, cint(pos))
