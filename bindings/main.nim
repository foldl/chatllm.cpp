import strutils, std/[strformat, httpclient], os, json, asyncdispatch
import libchatllm
import packages/docutils/highlite, terminal

var all_models: JsonNode = nil

proc show_help() =
    echo """
Usage: ./program [OPTIONS]

A command-line interface for interacting with language models using libchatllm.

Options:
  -m, --model <model_id>          Specify the model to use (e.g., :qwen:1.5b)
  --embedding_model <model_id>    Specify the embedding model to use
  --reranker_model <model_id>     Specify the reranker model to use
  -p, --prompt <prompt>           Set the initial prompt for the model
  -i, --interactive                Enable interactive mode
  --reversed_role                  Reverse the role of user and AI in interactive mode
  --multi                          Allow multi-line input in interactive mode
  -h, --help                       Show this help message

Examples:
  ./program -m :qwen:1.5b -p "Hello, world!"
  ./program --interactive --model :qwen:1.5b
  ./program --help
"""

proc get_model_url_on_modelscope(url: seq[string]): string =
    let proj = url[0]
    let fn = url[1]
    let user = if len(url) >= 3: url[2] else: "judd2024"

    return fmt"https://modelscope.cn/api/v1/models/{user}/{proj}/repo?Revision=master&FilePath={fn}"

proc parse_model_id(model_id: string): JsonNode =
    let parts = model_id.split(":")
    if all_models == nil:
        let fn = joinPath([parentDir(paramStr(0)), "../scripts/models.json"])
        const compiled_file = readFile("../scripts/models.json")
        all_models = if fileExists(fn): json.parseFile(fn) else: json.parseJson(compiled_file)

    if not all_models.contains(parts[0]): return nil
    let model = all_models[parts[0]]
    let variants = model["variants"]
    let variant = variants[if len(parts) >= 2: parts[1] else: model["default"].getStr()]
    let r = variant["quantized"][variant["default"].getStr()].copy()
    let url = r["url"].getStr().split("/")
    r["url"] = json.newJString(get_model_url_on_modelscope(url))
    r["fn"] = json.newJString(url[1])
    return r

proc print_progress_bar(iteration: BiggestInt, total: BiggestInt, prefix = "", suffix = "", decimals = 1, length = 60, fill = "█", printEnd = "\r", auto_nl = true) =
    let percent = formatFloat(100.0 * (iteration.float / total.float), ffDecimal, decimals)
    let filledLength = int(length.float * iteration.float / total.float)
    let bar = fill.repeat(filledLength) & '-'.repeat(length - filledLength)
    stdout.write(fmt"{printEnd}{prefix} |{bar}| {percent}% {suffix}")
    if iteration == total and auto_nl:
        echo ""

proc download_file(url: string, fn: string, prefix: string) =
    echo fmt"Downloading {prefix}"
    let client = newAsyncHttpClient()
    defer: client.close()

    proc onProgressChanged(total, progress, speed: BiggestInt) {.async} =
        print_progress_bar(progress, total, prefix)

    client.onProgressChanged = onProgressChanged
    client.downloadFile(url, fn).waitFor()

proc get_model(model_id: string; storage_dir: string): string =
    if not os.dirExists(storage_dir):
        os.createDir(storage_dir)

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

type
    highlighter = object
        line_acc: string
        lang: SourceLanguage
        thought_acc: string

proc receive_thought_chunk(ht: var highlighter, chunk: string) =
    if ht.thought_acc == "":
        stdout.setForegroundColor(fgMagenta)

    ht.thought_acc &= chunk
    stdout.write(chunk)

proc thought_end(ht: var highlighter) =
     stdout.setForegroundColor(fgDefault)

proc reset(ht: var highlighter) =
    ht.line_acc = ""
    ht.thought_acc = ""
    ht.lang = langNone

proc receive_chunk(ht: var highlighter, chunk: string) =
    ht.line_acc &= chunk

    proc none_chunk(ht: var highlighter) =
        if chunk == "":
            if ht.line_acc.startsWith("```"):
                let l = ht.line_acc[3..<len(ht.line_acc)]
                ht.lang = if l != "": getSourceLanguage(l) else: langCmd
                if ht.lang == langNone: ht.lang = langC
            stdout.writeLine("")
            ht.line_acc = ""
        else:
           stdout.write(chunk)

    proc lang_chunk(ht: var highlighter) =
        if chunk == "":
            if ht.line_acc.startsWith("```"):
                terminal.eraseLine()
                stdout.writeLine(ht.line_acc)
                ht.lang = langNone
            else:
                stdout.writeLine("")
            ht.line_acc = ""
        else:
            terminal.eraseLine()
            try:
                for t in tokenize(ht.line_acc, ht.lang):
                    case t[1]
                    of gtKeyword:
                        stdout.styledWrite(fgBlue, styleBright, t[0])
                    of gtDecNumber, gtBinNumber, gtHexNumber, gtOctNumber, gtFloatNumber:
                        stdout.styledWrite(fgMagenta, styleBright, t[0])
                    of gtStringLit, gtLongStringLit, gtCharLit:
                        stdout.styledWrite(fgYellow, styleBright, t[0])
                    of gtOperator:
                        stdout.styledWrite(fgRed, styleBright, t[0])
                    of gtPunctuation:
                        stdout.styledWrite(fgCyan, styleBright, t[0])
                    of gtEscapeSequence:
                        stdout.styledWrite(fgMagenta, styleBright, t[0])
                    of gtPreprocessor, gtDirective:
                        stdout.styledWrite(fgRed, styleBright, t[0])
                    of gtComment, gtLongComment:
                        stdout.styledWrite(fgGreen, styleDim, t[0])
                    else:
                        stdout.write(t[0])
            except:
                stdout.write(ht.line_acc)

    if ht.lang == langNone:
        none_chunk(ht)
    else:
        lang_chunk(ht)

proc chatllm_print(user_data: pointer, print_type: cint, utf8_str: cstring) {.cdecl.} =
    var ht = cast[ptr highlighter](user_data)
    var s: string = $utf8_str
    case cast[PrintType](print_type)
    of PRINT_CHAT_CHUNK:
        var n = 0
        for l in s.splitLines():
            if n > 0: receive_chunk(ht[], "")
            n += 1
            if l != "":
                receive_chunk(ht[], l)
    of PRINT_THOUGHT_CHUNK:
        receive_thought_chunk(ht[], s)
    of PRINT_EVT_THOUGHT_COMPLETED:
        thought_end(ht[])
    of RINTLN_MODEL_INFO:
        discard
    else:
        echo s
    stdout.flushFile()

proc chatllm_end(user_data: pointer) {.cdecl.} =
    echo ""

proc user_input(multi_input: bool): string =
    if multi_input:
        var r: seq[string] = @[]
        while true:
            var line = readLine(stdin)
            if line == "\\.": break
            r.add line
        return r.join("\n")
    else:
        return readLine(stdin)

const candidates = ["-m", "--model", "--embedding_model", "--reranker_model"]
let storage_dir = joinPath([parentDir(paramStr(0)), "../quantized"])

var ht = highlighter(line_acc: "", lang: langNone)
let chat = chatllm_create()

# related front end parameters are ignored by `libchatllm`
var prompt: string = "hello"
var interactive: bool = false
var reversed_role = false
var use_multiple_lines = false

for i in 1 .. paramCount():
    if paramStr(i) in ["-h", "--help"]:
        show_help()
        quit(0)

    if paramStr(i) in ["-i", "--interactive"]:
        interactive = true

    if paramStr(i) in ["--reversed_role"]:
        reversed_role = true

    if paramStr(i) in ["--multi"]:
        use_multiple_lines = true

    if (i > 1) and (paramStr(i - 1) in ["-p", "--prompt"]):
        prompt = paramStr(i)

    if (i > 1) and (paramStr(i - 1) in candidates) and paramStr(i).startsWith(":"):
        var m = paramStr(i)
        m = m[1..<len(m)]
        chatllm_append_param(chat, get_model(m, storage_dir).cstring)
    else:
        chatllm_append_param(chat, paramStr(i).cstring)

let r = chatllm_start(chat, chatllm_print, chatllm_end, addr(ht))
if r != 0:
    echo ">>> chatllm_start error: ", r
    quit(r)

if interactive:
    let user_tag = "You  > "
    let   ai_tag = "A.I. > "

    enableTrueColors()

    if reversed_role:
        stdout.write(ai_tag)
        stdout.writeLine(prompt)
        chatllm_history_append(chat, ord(RoleType.ROLE_USER), prompt.cstring)

    while true:
        stdout.write(user_tag)
        let input = user_input(use_multiple_lines)
        if input.isEmptyOrWhitespace(): continue

        stdout.write(ai_tag)
        ht.reset()
        let r = chatllm_user_input(chat, input.cstring)
        if r != 0:
            echo ">>> chatllm_user_input error: ", r
            break
else:
    discard chatllm_user_input(chat, prompt.cstring)

chatllm_show_statistics(chat)