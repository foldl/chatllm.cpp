import std/[asynchttpserver, asyncdispatch, asyncnet]
import std/[os, cmdline, strutils, strformat, json, tables, options, times, sequtils, parseutils]
import libchatllm

type
    RequestHandler* = proc(request: Request) {.async gcsafe.}

    Router* = object
        ## Called when the HTTP method is not registered for the route
        errorHandler*: RequestErrorHandler
        ## Called when the route request handler raises an Exception
        routes*: seq[Route]

    RequestErrorHandler* = proc(request: Request, e: ref Exception) {.async gcsafe.}

    Route = object
        httpMethod: HttpMethod
        route: string
        handler: RequestHandler

proc addRoute(router: var Router, httpMethod: HttpMethod, route: string | static string, handler: RequestHandler) =
    router.routes.add(Route(httpMethod: httpMethod, route: route, handler: handler))

proc get*(router: var Router, route: string | static string, handler: RequestHandler) =
    router.addRoute(HttpMethod.HttpGet, route, handler)

proc head*(router: var Router, route: string | static string, handler: RequestHandler) =
    router.addRoute(HttpMethod.HttpHead, route, handler)

proc post*(router: var Router, route: string | static string, handler: RequestHandler) =
    router.addRoute(HttpMethod.HttpPost, route, handler)

proc put*(router: var Router, route: string | static string, handler: RequestHandler) =
    router.addRoute(HttpMethod.HttpPut, route, handler)

proc delete*(router: var Router, route: string | static string, handler: RequestHandler) =
    router.addRoute(HttpMethod.HttpDelete, route, handler)

proc options*(router: var Router, route: string | static string, handler: RequestHandler) =
    router.addRoute(HttpMethod.HttpOptions, route, handler)

proc defaultNotFoundHandler(request: Request) {.async gcsafe.} =
    const body = "<h1>Not Found</h1>"
    echo "TODO: handler is missing for ", request.reqMethod, " ", request.url.path
    if request.reqMethod == HttpMethod.HttpHead:
        let headers = {"Content-type": "text/html", "Content-Length": $body.len}
        await request.respond(Http404, "", headers.newHttpHeaders())
    else:
        let headers = {"Content-type": "text/html"}
        await request.respond(Http404, body, headers.newHttpHeaders())

proc defaultMethodNotAllowedHandler(request: Request) {.async gcsafe.} =
    const body = "<h1>Method Not Allowed</h1>"
    echo "TODO: handler is missing for ", request.reqMethod, " ", request.url.path
    if request.reqMethod == HttpMethod.HttpHead:
        let headers = {"Content-type": "text/html", "Content-Length": $body.len}
        await request.respond(Http405, "", headers.newHttpHeaders())
    else:
        let headers = {"Content-type": "text/html"}
        await request.respond(Http405, body, headers.newHttpHeaders())

proc toHandler*(router: Router): RequestHandler =
    return proc(request: Request) {.async .} =
        echo $request.reqMethod & " " & request.url.path
        ## All requests arrive here to be routed
        template notFound() =
            await defaultNotFoundHandler(request)

        let path = request.url.path
        if path.len == 0 or path[0] != '/':
            notFound()
            return

        try:
            var matchedSomeRoute: bool = false
            for route in router.routes:

                if route.route == path:
                    matchedSomeRoute = true

                    if request.reqMethod == route.httpMethod: # We have a winner
                        await route.handler(request)
                        request.client.close()
                        return

            if matchedSomeRoute: # We matched a route but not the HTTP method
                await defaultMethodNotAllowedHandler(request)
            else:
                notFound()
        except Exception as e:
            if router.errorHandler != nil:
                await router.errorHandler(request, e)
            else:
                raise e

type
    StreamerType = enum
        Chat = "chat"
        FIM = "fim"
        Emb = "emb"

    FlatMessage = tuple[role: string; content: seq[tuple[t: string, content: string]]]

    StreamerWithHistory = ref object of Streamer
        history: seq[tuple[pos: int; messages: seq[FlatMessage]]]
        tokens_start: int

const ARG_SEP = "---"
var port = 11434
var ui: string = ""

var streamer_table = initTable[StreamerType, StreamerWithHistory]()

converter toStreamerType(s: string): StreamerType =
    for t in StreamerType:
        if s == $t: return t
    raise newException(ValueError, s)

proc get_streamer(t: StreamerType): StreamerWithHistory {.gcsafe.} =
    {.cast(gcsafe).}:
        result = streamer_table.getOrDefault(t, nil)
        if result != nil:
            if result.busy():
                result.abort()
            while result.busy():
                sleep(10)

proc newStreamerWithHistory*(args: openArray[string], auto_restart: bool = false): StreamerWithHistory =
    var streamer: StreamerWithHistory
    new(streamer)
    let r = initStreamer(streamer, args, auto_restart)
    result = if r: streamer else: nil

func flatten(messages: JsonNode, sys_prompt: var string): seq[FlatMessage] =
    result = @[]
    for m in messages.getElems():
        let role = m["role"].getStr()
        if role == "system":
            sys_prompt &= m["content"].getStr()
            continue
        if (result.len < 1) or (result[^1].role != role):
            result.add (role: role, content: @[(t: "text", content: "")])

        var t = "text"
        case m["content"].kind:
            of JString:
                if result[^1].content[^1].t == "text":
                    result[^1].content[^1].content &= m["content"].getStr()
                else:
                    result[^1].content.add (t: t, content: m["content"].getStr())
            of JObject:
                t = m["content"]["type"].getStr()
                if "url" in t:
                    result[^1].content.add (t: t, content: m["content"][t]["url"].getStr())
                else:
                    result[^1].content.add (t: t, content: m["content"]["text"].getStr())
            of JArray:
                for o in m["content"].getElems():
                    t = o["type"].getStr()
                    if "url" in t:
                        result[^1].content.add (t: t, content: o[t]["url"].getStr())
                    else:
                        result[^1].content.add (t: t, content: o["text"].getStr())
            else:
                raise newException(ValueError, fmt"""expected kind: {m["content"].kind}""")

proc start_chat(streamer: StreamerWithHistory, messages: JsonNode): bool =
    result = false

    var sys_prompt = ""
    let msg = flatten(messages, sys_prompt)

    if streamer.system_prompt != sys_prompt:
        streamer.history = @[]
        streamer.set_system_prompt(sys_prompt)

    streamer.flush()

    var k = 0
    var pos = -1
    block search:
        for i in 0..<streamer.history.len:
            let items = streamer.history[i]
            for j in 0..<items.messages.len:
                if k >= msg.len:
                    break search
                if msg[k] != items.messages[j]:
                    pos = i - 1
                    break search
                inc k

    if k >= msg.len:
        pos = streamer.history.len - 2
    elif pos < 0:
        pos = streamer.history.len - 1

    if pos >= 0:
        if pos < streamer.history.len - 1:
            if streamer.history.len >= pos + 2:
                streamer.history.delete(pos + 1, streamer.history.len - 1)
            discard streamer.set_cursor(streamer.history[pos].pos)
    else:
        streamer.restart()
        streamer.history = @[]

    k = 0
    for i in 0..pos:
        let items = streamer.history[i]
        k += items.messages.len

    func extract_base64(s: string): string =
        # data:image/png;base64,
        doAssert s.startswith("data:")
        let pos = s.find(";base64,")
        doAssert pos > 0
        return s.substr(pos + 8)

    proc add_mm_msg(parts: openArray[tuple[t: string, content: string]]) =
        streamer.llm.chatllm_multimedia_msg_prepare()
        for m in parts:
            case m.t:
                of "text":
                    discard streamer.llm.chatllm_multimedia_msg_append("text", m.content.cstring)
                of "image_url":
                    discard streamer.llm.chatllm_multimedia_msg_append("image", extract_base64(m.content).cstring)
                of "audio_url":
                    discard streamer.llm.chatllm_multimedia_msg_append("audio", extract_base64(m.content).cstring)
                of "video_url":
                    discard streamer.llm.chatllm_multimedia_msg_append("video", extract_base64(m.content).cstring)

    streamer.history.add (pos: -1, messages: @[])
    for i in k..<msg.len: streamer.history[^1].messages.add msg[i]

    while k <= msg.len - 2:
        add_mm_msg(msg[k].content)
        case msg[k].role:
            of "user":
                discard streamer.llm.chatllm_history_append_multimedia_msg(cint(RoleType.ROLE_USER))
            of "assistant":
                discard streamer.llm.chatllm_history_append_multimedia_msg(cint(RoleType.ROLE_ASSISTANT))
            else:
                discard streamer.llm.chatllm_history_append_multimedia_msg(cint(RoleType.ROLE_TOOL))
        inc(k)

    streamer.tokens_start = streamer.get_cursor()

    case msg[^1].role:
        of "user":
            add_mm_msg(msg[^1].content)
            result = streamer.llm.chatllm_async_user_input_multimedia_msg() == 0
        of "tool":
            doAssert (msg[^1].content.len == 1) and (msg[^1].content[0].t == "text")
            result = streamer.llm.chatllm_async_tool_completion(msg[^1].content[0].content.cstring) == 0
        else:
            raise newException(ValueError, fmt"unexpected {msg[^1].role}")
    if result:
        streamer.is_generating = true

proc sendCode(req: Request, code: HttpCode): Future[void] =
    var msg = "HTTP/1.1 " & $code & "\c\L"
    result = req.client.send(msg)

proc sendStr(req: Request, str: string): Future[void] =
    result = req.client.send(str)

type
    APIUsage = object
        prompt_tokens: int = 1
        completion_tokens: Option[int]
        total_tokens: int = 1

    ChatMessageTimings = object
        cache_n: int = 0
        predicted_ms: float = 0.0
        predicted_n: int = 0
        prompt_ms: float = 0.0
        prompt_n: int = 0

    ChunkChoiceDelta = object
        role: string = "assistant"
        content: Option[string] = none(string)
        reasoning_content: Option[string] = none(string)

    ChunkChoice = object
        delta: ChunkChoiceDelta
        index: int = 0
        logprobs: Option[string]
        finish_reason: Option[string]

    CompletionChunkResponse = object
        id: string
        created: int
        model: string
        choices: seq[ChunkChoice]
        `object`: string = "chat.completion.chunk"
        system_fingerprint: string = "fp_xxx"
        timings: Option[ChatMessageTimings] = none(ChatMessageTimings)

    CompletionResponse = object
        id: string
        created: int
        model: string
        choices: seq[ChunkChoice]
        `object`: string = "chat.completion.chunk"
        system_fingerprint: string = "fp_xxx"
        usage: APIUsage

    HttpResponder = ref object of RootObj
        req: Request
        timestamp: int
        model: string
        id: string
        timings: ChatMessageTimings

    ChatCompletionStreamResponder = ref object of HttpResponder

    ChatCompletionNonStreamResponder = ref object of HttpResponder

proc initHttpResponder(responder: HttpResponder, req: Request, timestamp: int, model, id: string) =
    responder.req = req
    responder.timestamp = timestamp
    responder.model = model
    responder.id = id

method recv_chunk(self: HttpResponder, chunk: string): Future[bool] {.async base gcsafe.} =
    result = true

method recv_thought(self: HttpResponder, chunk: string): Future[bool] {.async base gcsafe.} =
    result = true

method done(self: HttpResponder; all, thought: string, prompt_tokens, completion_tokens: int): Future[bool] {.async base gcsafe.} =
    result = true

method send_str(self: HttpResponder, s: string): Future[bool] {.async base gcsafe.} =
    try:
        await self.req.client.send(s, {})
        result = true
    except:
        echo "send error"
        result = false

proc newChatCompletionStreamResponder(req: Request, timestamp: int, model, id: string): ChatCompletionStreamResponder =
    new(result)
    initHttpResponder(result, req, timestamp, model, id)

method recv_chunk(self: ChatCompletionStreamResponder, chunk: string): Future[bool] {.async.} =
    var rsp = CompletionChunkResponse(id: self.id, created: self.timestamp, model: self.model)
    rsp.choices.add(ChunkChoice(delta: ChunkChoiceDelta(content: some(chunk))))
    rsp.timings = some(self.timings)
    return await self.send_str("data: " & $(%* rsp) & "\n\n")

method recv_thought(self: ChatCompletionStreamResponder, chunk: string): Future[bool] {.async.} =
    var rsp = CompletionChunkResponse(id: self.id, created: self.timestamp, model: self.model)
    rsp.choices.add(ChunkChoice(delta: ChunkChoiceDelta(reasoning_content: some(chunk))))
    rsp.timings = some(self.timings)
    return await self.send_str("data: " & $(%* rsp) & "\n\n")

method done(self: ChatCompletionStreamResponder; all, thought: string, prompt_tokens, completion_tokens: int): Future[bool] {.async.} =
    var rsp = CompletionResponse(id: self.id, created: self.timestamp, model: self.model)
    rsp.usage = APIUsage(prompt_tokens: prompt_tokens, completion_tokens: some(completion_tokens), total_tokens: prompt_tokens + completion_tokens)
    rsp.choices.add(ChunkChoice(delta: ChunkChoiceDelta(), finish_reason: some("stop")))
    return await self.send_str("data: " & $(%* rsp) & "\n\ndata: [DONE]\n")

proc newChatCompletionNoneStreamResponder(req: Request, timestamp: int, model, id: string): ChatCompletionNonStreamResponder =
    new(result)
    initHttpResponder(result, req, timestamp, model, id)

method done(self: ChatCompletionNonStreamResponder; all, thought: string, prompt_tokens, completion_tokens: int): Future[bool] {.async.} =
    var rsp = CompletionResponse(id: self.id, created: self.timestamp, model: self.model)
    rsp.usage = APIUsage(prompt_tokens: prompt_tokens, completion_tokens: some(completion_tokens), total_tokens: prompt_tokens + completion_tokens)
    rsp.choices.add(ChunkChoice(
        delta: ChunkChoiceDelta(content: some(all), reasoning_content: if thought != "": some(thought) else: none(string)),
        finish_reason: some("stop")))
    return await self.send_str($(%* rsp) & "\n\n")

proc send_headers(req: Request, headers: seq[(string, string)]) {.async gcsafe.} =
    await req.sendHeaders(headers.newHttpHeaders)

proc send_header(req: Request, name, value: string) {.async gcsafe.} =
    await req.sendHeaders([(name, value)].newHttpHeaders)

proc end_headers(req: Request, length: int = -1) {.async gcsafe.} =
    let more = [("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE, PUT"),
        ("Access-Control-Max-Age", "0"),
        ("Access-Control-Allow-Headers", "Content-Type,Access-Token,Authorization,ybg")]
    await req.sendHeaders(more.newHttpHeaders)
    if length > 0:
        let more = [("Content-Length", $ length)]
        await req.sendHeaders(more.newHttpHeaders)

    await req.client.send "\c\L"

proc end_headers_send(req: Request, response: string) {.async gcsafe.} =
    await req.end_headers(response.len)
    await req.client.send response

proc handle_completions(req: Request) {.async gcsafe.} =
    let streamer = get_streamer(StreamerType.Chat)
    if streamer == nil:
        await req.respond(Http404, "model not available")
        return
    let body = parseJson(req.body)
    let stream = body.getOrDefault("stream").getBool()
    let model = body.getOrDefault("model").getStr("chat")
    let timestamp = getTime().toUnix()
    let id = "id_" & $streamer.id()
    streamer.set_max_gen_tokens(body.getOrDefault("max_tokens").getInt(-1))

    let headers = @[("Cache-Control", "no-cache"),
                ("vary", "origin, access-control-request-method, access-control-request-headers"),
                ("access-control-allow-credentials", "true"),
                ("x-content-type-options", "nosniff")]

    let messages = body["messages"]
    if (messages == nil) or (messages.kind != JArray):
        await req.respond(Http404, "`messages` is missing")
        return

    let t_prompt_start = cpuTime()
    if not streamer.start_chat(messages):
        await req.respond(Http404, "Interval error")
        return

    var cls: HttpResponder = if stream: newChatCompletionStreamResponder(req, timestamp, model, id) else: newChatCompletionNoneStreamResponder(req, timestamp, model, id)

    await req.sendCode(Http200)
    await req.send_headers(headers)
    await req.send_header("Content-type", if stream: "text/event-stream; charset=utf-8" else: "application/json")
    await req.end_headers()

    cls.timings.cache_n = streamer.tokens_start

    var waiting_first = true

    var t_prediction_start = 0.0
    var predict_tok_n = 0

    for chunk in chunks(streamer):
        if req.client.isClosed():
            streamer.abort()
            break

        var t = cpuTime()

        if waiting_first:
            waiting_first = false
            predict_tok_n = streamer.get_cursor() - 1
            t_prediction_start = t

            cls.timings.prompt_n  = streamer.get_cursor() - cls.timings.cache_n
            cls.timings.prompt_ms = (t - t_prompt_start) * 1000

        cls.timings.predicted_n  = streamer.get_cursor() - predict_tok_n
        cls.timings.predicted_ms = (t - t_prediction_start) * 1000

        case chunk[0]:
            of ChunkType.Chat:
                if not await cls.recv_chunk(chunk[1]):
                    streamer.abort()
                    break
            of ChunkType.Thought:
                if not await cls.recv_thought(chunk[1]):
                    streamer.abort()
                    break
            else: discard

    streamer.history[^1].messages.add (role: "assistant", content: @[(t: "text", content: streamer.acc)])
    streamer.history[^1].pos = streamer.get_cursor()

    if streamer.tokens_start > streamer.history[^1].pos:
        streamer.tokens_start = streamer.history[^1].pos

    if not req.client.isClosed():
        discard await cls.done(streamer.acc, streamer.thought_acc, cls.timings.prompt_n, cls.timings.predicted_n)

    streamer.acc = ""

func parse_embedding(s: string): seq[float] =
    result = @[]
    for w in s.tokenize({',', ' ', '\n', '\r'}):
        if w[1]: continue
        result.add parseFloat(w[0])

proc handle_embeddings(req: Request) {.async gcsafe.} =
    let streamer = get_streamer(StreamerType.Emb)
    if streamer == nil:
        await req.respond(Http404, "model not available")
        return

    let body = parseJson(req.body)
    let model = body.getOrDefault("model").getStr()
    let encoding_format = body.getOrDefault("encoding_format").getStr("float")
    if encoding_format != "float":
        await req.respond(Http404, fmt"unsupported {encoding_format}")
        return

    var inputs = newSeq[string]()
    let input = body.getOrDefault("input")
    if input.kind == JString:
        inputs.add input.getStr()
    elif input.kind == JArray:
        for s in input.getElems():
            inputs.add(s.getStr())

    if inputs.len < 1:
        await req.respond(Http404, fmt"malformed input")
        return

    let headers = @[("Cache-Control", "no-cache"),
                ("vary", "origin, access-control-request-method, access-control-request-headers"),
                ("access-control-allow-credentials", "true"),
                ("x-content-type-options", "nosniff"),
                ("Content-type", "application/json")]

    type
        EmbObject = object
            `object`: string = "embedding"
            embedding: seq[float]
            index: int

        Result = object
            `object`: string = "list"
            data: seq[EmbObject]
            model: string
            usage: APIUsage

    await req.sendCode(Http200)

    var r = Result(data: @[], model: model)
    for i in 0 ..< inputs.len:
        var o = EmbObject(index: i)
        discard streamer.llm.chatllm_text_embedding(inputs[i].cstring, cint(EmbeddingPurpose.EMBEDDING_FOR_DOC))
        o.embedding = parse_embedding(streamer.result_embedding)
        r.data.add o

    await req.send_headers(headers)
    await req.end_headers_send($(%* r))

proc handle_index(req: Request) {.async gcsafe.} =
    const defaultUI {.strdefine: "defaultUI".}: string = currentSourcePath.parentDir() & "/../scripts/chat_ui.html"
    const compiled_file = readFile(defaultUI)
    var headers = @[("Content-type", "text/html; charset=utf-8")]
    {.cast(gcsafe).}:
        let fn_ui = if ui != "": ui else: defaultUI
        let content = if ui != "": readFile(ui) else: compiled_file

    if fn_ui.endswith(".gz"):
        headers.add ("Content-Encoding", "gzip")
    await req.respond(Http200, content, headers.newHttpHeaders())

proc handle_oai_models(req: Request) {.async gcsafe.} =
    type
        Meta = object
            n_params: int
            n_ctx_train: int

        Info = object
            id: string
            `object`: string = "model"
            created: int
            owned_by: string = "You"
            meta: Meta

        Infos = object
            `object`: string = "list"
            data: seq[Info]

    var infos = Infos()
    let streamer = get_streamer(StreamerType.Chat)
    if streamer != nil:
        let info = parseJson(streamer.model_info)
        var m = Info(id: info["name"].getStr(), created: now().toTime().toUnix())
        m.meta.n_params     = info["param_num"].getInt()
        m.meta.n_ctx_train  = info["training_context_length"].getInt()
        infos.data.add(m)

    let headers = {"Content-type": "application/json"}
    await req.respond(Http200, $(%* infos), headers.newHttpHeaders())

func format_model_name(info: JsonNode): string =
    let name = info.getOrDefault("name").getStr()
    let native = info.getOrDefault("native_name").getStr()
    let n = info.getOrDefault("param_num").getInt()
    result = fmt"{name}-{float(n)/1000000000.0:.1f}B"
    if native != "":
        result = result & fmt" ({native})"

proc handle_ollama_tags(req: Request) {.async gcsafe.} =
    type
        Info = object
            name: string
            model: string
        Infos = object
            models: seq[Info]

    {.cast(gcsafe).}:
        var tags = Infos(models: @[])
        for k in streamer_table.keys():
            let info = parseJson(streamer_table[k].model_info)
            var m = Info(name: info["name"].getStr(), model: format_model_name(info))
            tags.models.add(m)
    let headers = {"Content-type": "application/json"}
    await req.respond(Http200, $(%* tags), headers.newHttpHeaders())

proc handle_ollama_version(req: Request) {.async gcsafe.} =
    let rsp = """{"version": "0.13.0"}"""
    let headers = {"Content-type": "application/json"}
    await req.respond(Http200, rsp, headers.newHttpHeaders())

proc handle_ollama_show(req: Request) {.async gcsafe.} =
    let model_name = parseJson(req.body)["model"].getStr()
    var model: JsonNode = nil
    {.cast(gcsafe).}:
        for k in streamer_table.keys():
            let info = parseJson(streamer_table[k].model_info)
            if info["name"].getStr() == model_name:
                model = info
                break

    if model == nil:
        await req.respond(Http404, "NOT FOUND")
        return

    var rsp = parseJson("{}")
    var info = parseJson("{}")
    info["general.parameter_count"] = model["param_num"]
    info["llama.context_length"]    = model["context_length"]
    rsp["template"]   = %"|placeholder|"
    rsp["model_info"] = info
    rsp["capabilities"] = newJArray()

    const mapping = toTable({"Text Embedding": "embedding", "Ranker": "ranking", "Text": "completion", "Image Input": "vision", "Audio Input": "audio"})
    for s in model["capabilities"].getElems():
        let ss = s.getStr()
        if ss in mapping:
            rsp["capabilities"].add(% mapping[ss])

    let headers = {"Content-type": "application/json"}
    await req.respond(Http200, $(%* rsp), headers.newHttpHeaders())

proc handle_llama_props(req: Request) {.async gcsafe.} =
    type
        Empty = object
        Modalities = object
            vision: bool = false
            audio: bool = false
        GenerationSettings = object
            id: int = 0
            id_task: int = -1
            n_ctx: int = 0
            speculative: bool = false
            is_processing: bool
            params: Empty
        Props = object
            default_generation_settings: GenerationSettings
            total_slots: int =                      1
            model_alias: string =                   ""
            model_path: string =                    "/some/where"
            modalities: Modalities
            build_info: string =                    "Today"

    var props = Props()
    var streamer = get_streamer(StreamerType.Chat)
    if streamer != nil:
        let info = parseJson(streamer.model_info)
        let capabilities = info.getOrDefault("capabilities")
        props.default_generation_settings.n_ctx         = info["context_length"].getInt()
        props.default_generation_settings.is_processing = streamer.busy()
        props.model_alias = format_model_name(info)
        for c in capabilities.getElems():
            if c.getStr() == "Image Input":
                props.modalities.vision = true
            elif c.getStr() == "Audio Input":
                props.modalities.audio = true

    let headers = {"Content-type": "application/json"}
    await req.respond(Http200, $(%* props), headers.newHttpHeaders())

proc handle_llama_slots(req: Request) {.async gcsafe.} =
    type
        Params = object
            n_predict: int = -1
            seed: int = 1
            temperature: float = 0.8
            dynatemp_rang: float = 0.0
            dynatemp_exponent: float = 1.0
            top_k: int = 40

        Slot = object
            id: int = 0
            id_task: int = 1
            n_ctx: int = 0
            speculative: bool = false
            is_processing: bool
            params: Params

    var slot = Slot()
    var streamer = get_streamer(StreamerType.Chat)
    if streamer != nil:
        let info = parseJson(streamer.model_info)
        slot.n_ctx = info.getOrDefault("context_length").getInt()
        slot.is_processing = streamer.busy()

    let headers = {"Content-type": "application/json"}
    await req.respond(Http200, $(%* [slot]), headers.newHttpHeaders())

proc run {.async.} =
    var server = newAsyncHttpServer()
    var router = Router()

    router.get  "/",                        handle_index

    # OAI-compatible
    router.post "/v1/chat/completions",     handle_completions
    router.post "/v1/embeddings",           handle_embeddings
    router.get  "/v1/models",               handle_oai_models

    # llama-compatible
    router.get  "/props",                   handle_llama_props
    router.get  "/slots",                   handle_llama_slots

    # ollama-compatible
    router.get  "/api/tags",                handle_ollama_tags
    router.get  "/api/version",             handle_ollama_version
    router.post "/api/show",                handle_ollama_show
    router.post "/api/chat",                handle_completions

    server.listen(Port(port))
    let port = server.getPort
    echo "Serving at curl http://localhost:" & $port.uint16 & "/"
    while true:
        if server.shouldAcceptRequest():
            try:
                await server.acceptRequest(router.toHandler())
            except:
                discard
        else:
            await sleepAsync(500)

proc main(): int =
    if paramCount() < 1:
        echo fmt"usage: {paramStr(0)} [app_args] [{ARG_SEP}TYPE path/to/model [additional args]]"
        echo fmt"where app_args :: --ui /path/to/ui --port PORT"
        echo fmt"where TYPE ::= chat | fim | emb"
        return -1

    var args: seq[string] = @[]
    for i in 1..paramCount():
        args.add paramStr(i)

    var args_tab = initTable[StreamerType, seq[string]]()
    for t in StreamerType:
        args_tab[t] = @["-m"]

    while args.len > 0:
        if args[0] == "--ui":
            doAssert args.len >= 2
            ui = args[1]
            args.delete(0, 1)
        elif args[0] == "--port":
            doAssert args.len >= 2
            port = parseInt(args[1])
            args.delete(0, 1)
        elif args[0].startswith(ARG_SEP):
            break
        else:
            raise newException(ValueError, fmt"bad argument: {args[0]}")

    var current: ptr seq[string] = nil
    for i in 0 ..< args.len:
        if args[i].startswith(ARG_SEP):
            let tag = args[i][len(ARG_SEP)..^1]
            current = addr args_tab[tag]
        else:
            doAssert current != nil
            current[].add(args[i])

    for t in StreamerType:
        if args_tab[t].len > 1:
            streamer_table[t] = newStreamerWithHistory(args_tab[t])

    waitFor run()
    return 0

quit(main())