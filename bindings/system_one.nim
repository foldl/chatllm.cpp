import std/[os, times, math, strutils, strformat, json, tables, options, times, sequtils, sugar, enumerate]
import std/[sets]

import libchatllm

type
    DecisionCompiled* = ref object of RootObj
        state*: string
        fields*: seq[string]
        options*: Table[string, seq[(string, string)]]
        kinds*: Table[string, string]
        instructions*: Table[string, string]

    DecisionModel* = ref object of RootObj
        name: string
        labels: seq[string]
        streamer: Streamer

    InternDecisionModel = ref object of DecisionModel
        DECISION_TOKEN = "<decision>"
        SYSTEM_PROMPT = "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text."
        DEFAULT_TEMPERATURE: float = 2.747760550703

    NextTokenDecisionModel = ref object of DecisionModel
        SYSTEM_PROMPT = "You are a decision engine. You read a record and answer one question about it by choosing exactly one option. Reply with a single letter."
        DEFAULT_TEMPERATURE: float = 0.8
        label_ids: seq[int] # token id for each labels
        ai_prompt = "<think>\n\n</think>\n\nAnswer:"

proc decode_tokens(chat: Streamer; l: seq[string]): seq[int] =
    var n_vocab: cint = 0
    var width: cint = 0
    var p = chatllm_get_token_vocab(chat.llm, addr n_vocab, addr width)
    result = newSeq[int](l.len)
    var tab = collect:
        for (i, tok) in enumerate(l): {tok: i}
    for i in 0..<n_vocab:
        let tok = $ cast[cstring](cast[uint](p) + cast[uint](width * i))
        if not (tok in tab): continue
        let id = tab[tok]
        result[id] = i
        tab.del(tok)
        if tab.len < 1: break

    assert tab.len < 1, fmt"some tokens can't be found in vocab: {tab.keys.toSeq()}"

proc init_from_streamer*(chat: Streamer): DecisionModel =
    let info = parseJson(chat.model_info)
    let name = info["name"].getStr()
    case name
    of "Intern-Decision":
        var r = new(InternDecisionModel)
        r.labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".mapIt($ it)
        result = r
    of "H2O-Lightning":
        var r = new(NextTokenDecisionModel)
        r.labels = @["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"]
        r.label_ids = decode_tokens(chat, r.labels)
        result = r
    else:
        var r = new(NextTokenDecisionModel)
        r.labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".mapIt($ it)
        r.label_ids = decode_tokens(chat, r.labels)
        r.ai_prompt = "<think>\n\n</think>\n\nAnswer: "
        result = r

    result.name = name
    result.streamer = chat

func get_options(q: JsonNode): seq[(string, string)] =
    let criteria = q{"criteria"}
    case q["type"].getStr()
    of "choice":
        return collect:
            for k, v in criteria.getFields(): (k, v.getStr())
    of "score":
        case criteria.kind
        of JObject:
            return collect:
                for k, v in criteria.getFields(): (k, v.getStr())
        of JArray:
            return collect:
                for k, v in enumerate(criteria.getElems()): ($ k, v.getStr())
        else:
            assert false
    of "noul":
        func try_get(o: JsonNode, keys: openArray[string], default: string): string =
            if (o == nil) or (o.kind != JObject): return default
            for k in keys:
                for ak, av in o.getFields():
                    if ak.toLower() == k:
                        return av.getStr()
            return default

        let yes = criteria.try_get(["yes", "true",  "1"], "The answer is yes (affirmative, or align with the claim).")
        let no  = criteria.try_get(["no",  "false", "0"], "The answer is no (negative, or disagree with the claim).")
        return @[("yes", yes), ("no", no)]

func softmax[T](data: openArray[T]): seq[float] =
    let D = max(data)
    result = data.mapIt(it * 1.0)
    result.applyIt(exp(it - D))
    let sum = result.foldl(a + b, 0.0)
    result.applyIt(it / sum)

func scale_probabilities(probs: seq[float], temp: float): seq[float] =
    if temp <= 0.0: return probs
    let values = probs.mapIt(if it > 0: math.ln(it) else: NegInf)
    let maximum = max(values)
    let weights = values.mapIt(exp((it - maximum) / temp))
    let total = weights.foldl(a + b)
    return weights.mapIt(it / total)

func logits_to_probabilities[T](logits: seq[T], temp: float): seq[float] =
    let t = max(temp, 1e-3)
    let v = logits.mapIt(float(it / t))
    let m = max(v)
    result = v.mapIt(exp(it - m))
    let s = result.foldl(a + b, 0.0)
    result.applyIt(it / s)

func confidence(probs: seq[float]): float =
    let n = float(probs.len)
    if n < 2: return 1.0
    return (max(probs) - 1.0 / n) / (1.0 - 1.0 / n)

func confidence(max_prob: float, nn: Natural): float =
    if nn < 2: return 1.0
    let n = float(nn)
    return (max_prob - 1.0 / n) / (1.0 - 1.0 / n)

proc decode(compiled: DecisionCompiled; clogits: ptr float32, token_length: int, N_SYMBOLS: int,
    t: float): JsonNode =
    let logits = cast[ptr array[0 .. 0x7fffffff, float32]](clogits)
    var answers = parseJson("{}")
    for idx, field in enumerate(compiled.fields):
        let kind = compiled.kinds[field]
        let opts = compiled.options[field]
        let values = collect:
            for (v1, v2) in opts: v1
        let selected_logits = logits[N_SYMBOLS * idx ..< N_SYMBOLS * idx + len(opts)]
        let probabilities = scale_probabilities(softmax(selected_logits), t)
        let probs = zip(values, probabilities).toTable()
        let best = max(values, (a, b) => sgn(probs[a] - probs[b]))
        var answer = %*{"type": kind, "probabilities": probs, "confidence": confidence(probs[best], probabilities.len)}
        case kind
            of "noul":
                answer["noul"] = % probs["yes"]
            of "score":
                answer["score"] = % zip(values.mapIt(parseFloat(it)), values.mapIt(probs[it])).mapIt(it[0] * it[1]).foldl(a + b)
                answer["legend"] = % opts.toTable()
            else:
                answer["choice"] = % best
        answer["source"] = % "local"
        answer["decision"] = % best
        answers[field] = answer

    result = %* {"answers": answers, "usage": {"input_tokens": token_length, "output_tokens": 0}}
    result["usage"]["decision_count"] = % len(compiled.fields)

proc decode_one(model: NextTokenDecisionModel, compiled: DecisionCompiled; field: string; clogits: ptr float32, t: float): JsonNode =
    let logits = cast[ptr array[0 .. 0x7fffffff, float32]](clogits)
    let kind = compiled.kinds[field]
    let opts = compiled.options[field]
    let values = collect:
        for (v1, v2) in opts: v1
    let selected_logits = collect:
        for id in model.label_ids[0 ..< opts.len]: logits[id]
    let probabilities = selected_logits.logits_to_probabilities(t)
    let probs = zip(values, probabilities).toTable()
    let best = max(values, (a, b) => sgn(probs[a] - probs[b]))
    result = %*{"type": kind, "probabilities": probs, "confidence": confidence(probs[best], probabilities.len)}
    case kind
        of "noul":
            result["noul"] = % probs["yes"]
        of "score":
            result["score"] = % zip(values.mapIt(parseFloat(it)), values.mapIt(probs[it])).mapIt(it[0] * it[1]).foldl(a + b)
            result["legend"] = % opts.toTable()
        else:
            result["choice"] = % best
    result["source"] = % "local"
    result["decision"] = % best

proc compile*(validated_req: JsonNode, compiled: var DecisionCompiled) =
    var all_opts = Table[string, seq[(string, string)]]()
    var kinds = Table[string, string]()
    var instructions = Table[string, string]()
    for name, question in pairs(validated_req["questions"]):
        all_opts[name] = question.get_options()
        kinds[name] = question["type"].getStr()
        instructions[name] = question["instructions"].getStr()

    compiled.state = pretty(validated_req["state"], 2)
    compiled.fields = validated_req["questions"].keys().toSeq()
    compiled.options = all_opts
    compiled.kinds = kinds
    compiled.instructions = instructions

method messages(model: DecisionModel; compiled: DecisionCompiled): JsonNode {.base.} =
    raise newException(CatchableError, "Method without implementation override")

method messages(model: InternDecisionModel; compiled: DecisionCompiled): JsonNode =
    result = %*[{ "role": "system", "content": model.SYSTEM_PROMPT }]

    let all_opts = compiled.options
    var schema_lines: seq[string] = @[]
    for name in compiled.fields:
        schema_lines.add fmt"""{name}: {compiled.instructions[name]}"""
        let options = all_opts[name]
        for (symbol, opt) in zip(model.labels, options):
            let (value, description) = opt
            schema_lines.add fmt"""    {symbol} = {value}: {description}"""

    let user = fmt"""Return one answer for every field using the supplied answer symbols.

## State
{compiled.state}
## Decision schema
{schema_lines.join("\n")}
"""

    assert not (model.DECISION_TOKEN in user)
    result.add %*{ "role": "user", "content": user }

    var assistant = parseJson("{}")
    for k in compiled.fields: assistant[k] = % model.DECISION_TOKEN
    result.add %*{ "role": "assistant", "content": pretty(assistant,4) }

method messages*(model: NextTokenDecisionModel; compiled: DecisionCompiled): JsonNode =
    result = %*[]

    let all_opts = compiled.options
    for name in compiled.fields:
        var question = %*[{ "role": "system", "content": model.SYSTEM_PROMPT }]
        var schema_lines: seq[string] = @[]
        let options = all_opts[name]
        for (symbol, opt) in zip(model.labels, options):
            let (value, description) = opt
            schema_lines.add fmt"""{symbol}) {value}: {description}"""

        let user = &"""record: {compiled.state}
question: {compiled.instructions[name]}
options:
{schema_lines.join("\n")}"""

        question.add %*{ "role": "user", "content": user }
        question.add %*{ "role": "assistant", "content": model.ai_prompt }

        result.add question

proc build_history(messages: JsonNode; chat: Streamer) =
    for i, msg in enumerate(messages.getElems()):
        if i == 0:
            if msg["role"].getStr() == "system":
                chat.set_system_prompt(msg["content"].getStr())
            chat.restart()
            continue

        case msg["role"].getStr()
        of "user":
            chat.llm.chatllm_history_append(cast[cint](RoleType.ROLE_USER), msg["content"].getStr().cstring)
        of "assistant":
            chat.llm.chatllm_history_append(cast[cint](RoleType.ROLE_ASSISTANT), msg["content"].getStr().cstring)
        else:
            assert false, fmt"""bad role: {msg["role"]}"""

method do_decide(model: DecisionModel; compiled: DecisionCompiled): JsonNode {.base gcsafe.} =
  raise newException(CatchableError, "Method without implementation override")

method do_decide(model: InternDecisionModel; compiled: DecisionCompiled): JsonNode =
    model.messages(compiled).build_history(model.streamer)
    var input_length: cint = 0
    let logits = model.streamer.llm.chatllm_make_decisions(addr input_length)
    return decode(compiled, logits, input_length, model.labels.len(), model.DEFAULT_TEMPERATURE)

# TODO: optimize
method do_decide(model: NextTokenDecisionModel; compiled: DecisionCompiled): JsonNode =
    let msgs_list = model.messages(compiled)
    var answers = parseJson("{}")
    var token_length = 0
    for (i, msgs) in enumerate(msgs_list.getElems()):
        let field = compiled.fields[i]
        msgs.build_history(model.streamer)
        let input_tokens = model.streamer.tokenize_history(false)
        if input_tokens.len < 1: continue
        token_length += input_tokens.len
        let logits = model.streamer.predict_next_token(input_tokens)
        let answer = decode_one(model, compiled, field, logits, model.DEFAULT_TEMPERATURE)
        answers[field] = answer

    result = %* {"answers": answers, "usage": {"input_tokens": token_length, "output_tokens": len(compiled.fields)}}
    result["usage"]["decision_count"] = % len(compiled.fields)

proc decide*(model: DecisionModel; compiled: DecisionCompiled): JsonNode {.gcsafe.} =
    let t0 = cpuTime()
    result = model.do_decide(compiled)
    let inference_ms = (cpuTime() - t0) * 1000
    result["timing"] = %* {"inference_ms": round(inference_ms, 2)}
    result["model"] = % model.name
    result["backend"] = % "chatllm.cpp"

proc validate_request*(req: JsonNode, max_choices: int = 62): JsonNode =
    var r = %*{}
    r["state"] = req["state"]
    r["questions"] = parseJson("{}")
    for name, question in pairs(req["questions"]):
        let kind = question["type"].getStr()
        let criteria = question{"criteria"}
        var cleaned = %*{"type": kind, "instructions": question["instructions"].getStr()}
        case kind
        of "choice":
            assert criteria.kind == JObject
            assert criteria.getFields().len <= max_choices
            cleaned["criteria"] = criteria.copy()
        of "score":
            case criteria.kind
            of JObject:
                assert criteria.getFields().len <= max_choices
                for _, v in criteria.getFields():
                    assert v.kind == JFloat
            of JArray:
                assert criteria.getElems().len <= max_choices
                for s in criteria.getElems():
                    assert s.kind == JString
            else:
                assert false, fmt"invalid: {criteria}"
            cleaned["criteria"] = criteria.copy()
        of "noul":
            discard
        else:
            assert false, fmt"unknown kind: {kind}"

        r["questions"][name] = cleaned

    return r

when isMainModule:
    proc main(): int =
        let request = parseJson("""{
            "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
            "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                "billing": "payments, charges, refunds, invoices",
                "shipping": "delivery, tracking, lost or late parcels",
                "technical": "bugs, errors, login problems"
                }
            },
            "angry": {
                "type": "noul",
                "instructions": "Is the customer angry?"
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]
            }
            }
        }""")
        let validated = validate_request(request)
        var compiled = new(DecisionCompiled)
        validated.compile(compiled)
        echo(compiled.fields)
        echo(compiled.kinds)
        echo(compiled.options)

        var args = newSeq[string]()
        for i in 1 .. paramCount():
            args.add paramStr(i)
        var chat = newStreamer(args)
        var model = init_from_streamer(chat)
        assert model != nil

        let answers = model.decide(compiled)
        echo answers.pretty(4)

    quit(main())