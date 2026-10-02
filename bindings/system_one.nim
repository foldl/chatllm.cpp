import std/[os, times, math, strutils, strformat, json, tables, options, times, sequtils, sugar, enumerate]

import libchatllm

type
    InternDecisionCompiled* = ref object
        messages*: JsonNode
        fields*: seq[string]
        options*: Table[string, seq[(string, string)]]
        kinds*: Table[string, string]

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
    let values = probs.mapIt(if it > 0: math.ln(it) else: NegInf)
    let maximum = max(values)
    let weights = values.mapIt(exp((it - maximum) / temp))
    let total = weights.foldl(a + b)
    return weights.mapIt(it / total)

proc decode*(compiled: InternDecisionCompiled, clogits: ptr float32, token_length: int = 1, inference_ms: float = 1.0, model_name: string = "Intern-Decision",
             temp: float = -1.0): JsonNode =
    const DEFAULT_TEMPERATURE = 2.747760550703
    const N_SYMBOLS = 62
    let t = if temp > 0.0: temp else: DEFAULT_TEMPERATURE
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
        var answer = %*{"type": kind, "probabilities": probs, "confidence": probs[best]}
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

    result = %* {"answers": answers, "usage": {"input_tokens": token_length, "output_tokens": len(compiled.fields)}, "timing": {"inference_ms": round(inference_ms, 2)}}
    result["usage"]["decision_count"] = % len(compiled.fields)
    result["model"] = % model_name
    result["backend"] = % "chatllm.cpp"

proc compile*(compiled: var InternDecisionCompiled, validated_req: JsonNode) =
    const
        DECISION_TOKEN = "<decision>"
        ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        SYSTEM_PROMPT = "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text."

    var messages = %*[{ "role": "system", "content": SYSTEM_PROMPT }]

    var all_opts = Table[string, seq[(string, string)]]()
    var kinds = Table[string, string]()
    var schema_lines: seq[string] = @[]
    for name, question in pairs(validated_req["questions"]):
        schema_lines.add fmt"""{name}: {question["instructions"].getStr()}"""
        let options = question.get_options()
        for (symbol, opt) in zip(ANSWER_SYMBOLS, options):
            let (value, description) = opt
            schema_lines.add fmt"""    {symbol} = {value}: {description}"""
        all_opts[name] = options
        kinds[name] = question["type"].getStr()

    let user = fmt"""Return one answer for every field using the supplied answer symbols.

## State
{pretty(validated_req["state"], 2)}
## Decision schema
{schema_lines.join("\n")}
"""

    assert not (DECISION_TOKEN in user)
    messages.add %*{ "role": "user", "content": user }

    var assistant = parseJson("{}")
    for k in validated_req["questions"].keys(): assistant[k] = % DECISION_TOKEN
    messages.add %*{ "role": "assistant", "content": pretty(assistant,4) }

    compiled.messages = messages
    compiled.fields = validated_req["questions"].keys().toSeq()
    compiled.options = all_opts
    compiled.kinds = kinds

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
    var compiled = new(InternDecisionCompiled)
    compile(compiled, validated)
    echo(pretty(compiled.messages, 4))
    echo(compiled.fields)
    echo(compiled.kinds)
    echo(compiled.options)

    var args = newSeq[string]()
    for i in 1 .. paramCount():
        args.add paramStr(i)
    var chat = newStreamer(args)

    for i, msg in enumerate(compiled.messages.getElems()):
        case msg["role"].getStr()
        of "system":
            assert i == 0
            chat.set_system_prompt(msg["content"].getStr())
            chat.restart()
        of "user":
            chat.llm.chatllm_history_append(cast[cint](RoleType.ROLE_USER), msg["content"].getStr().cstring)
        of "assistant":
            chat.llm.chatllm_history_append(cast[cint](RoleType.ROLE_ASSISTANT), msg["content"].getStr().cstring)
        else:
            assert false, fmt"""bad role: {msg["role"]}"""

    var input_length: cint = 0
    let t0 = cpuTime()
    let logits = chat.llm.chatllm_make_decisions(addr input_length)
    let answers = decode(compiled, logits, input_length, (cpuTime() - t0) * 1000)
    echo answers.pretty(4)

quit(main())