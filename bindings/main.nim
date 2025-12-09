import std/[strformat, strutils, os, json]
import libchatllm
import packages/docutils/highlite, terminal

proc show_help() =
    echo fmt"""
Usage: {paramStr(0)} [OPTIONS]

A command-line interface for interacting with language models using libchatllm.

Options (run `main -h` to get a full list of options):
  -m, --model <model_id>          Specify the model to use (e.g., :qwen:1.5b)
  --embedding_model <model_id>    Specify the embedding model to use
  --reranker_model <model_id>     Specify the reranker model to use
  -p, --prompt <prompt>           Set the initial prompt for the model
  -i, --interactive               Enable interactive mode
  --reversed_role                 Reverse the role of user and AI in interactive mode
  --multi                         Allow multi-line input in interactive mode
  -h, --help                      Show this help message

Examples:
  {paramStr(0)} -m :qwen:1.5b -p "Hello, world!"
  {paramStr(0)} --interactive --model :qwen:1.5b
  {paramStr(0)} --help
"""

type
    highlighter = object
        line_acc: string
        lang: SourceLanguage
        thought_acc: string

    DerivedStreamer = ref object of Streamer

proc newDerivedStreamer*(args: openArray[string], auto_restart: bool = false): DerivedStreamer =
    var streamer: DerivedStreamer
    new(streamer)
    let r = initStreamer(streamer, args, auto_restart)
    result = if r: streamer else: nil

proc receive_thought_chunk(ht: var highlighter, chunk: string) =
    ht.thought_acc &= chunk
    stdout.styledWrite(fgMagenta, chunk)

proc thought_end(ht: var highlighter) =
    discard

proc reset(ht: var highlighter) =
    ht.line_acc = ""
    ht.thought_acc = ""
    ht.lang = langNone

proc receive_chunk0(ht: var highlighter, chunk: string) =
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

proc receive_chunk(ht: var highlighter, s: string) =
    var n = 0
    for l in s.splitLines():
        if n > 0: receive_chunk0(ht, "")
        n += 1
        if l != "":
            receive_chunk0(ht, l)
    stdout.flushFile()

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

var ht = highlighter(line_acc: "", lang: langNone)

method on_logging(streamer: DerivedStreamer, text: string) =
    echo text

method on_print_meta(streamer: DerivedStreamer, text: string) =
    echo text

method on_thought_completed(streamer: DerivedStreamer) =
    ht.thought_end()

if paramCount() < 1:
    show_help()
    quit(0)

var args = newSeq[string]()
for i in 1 .. paramCount():
    args.add paramStr(i)
var chat = newDerivedStreamer(args)

if (chat == nil) or chat.fe_options.help:
    show_help()
    quit(0)

proc ctrl_c_handler() {.noconv.} =
    if chat.busy():
        chat.abort()
        setControlCHook(ctrl_c_handler)
    else:
        chatllm_show_statistics(chat.llm)
        quit(0)

setControlCHook(ctrl_c_handler)

chat.set_system_prompt(chat.fe_options.sys_prompt)

if chat.fe_options.interactive:
    let user_tag = "You  > "
    let   ai_tag = "A.I. > "

    enableTrueColors()

    if chat.fe_options.reversed_role:
        stdout.write(ai_tag)
        stdout.writeLine(chat.fe_options.prompt)
        chatllm_history_append(chat.llm, ord(RoleType.ROLE_USER), chat.fe_options.prompt.cstring)

    while true:
        stdout.write(user_tag)
        let input = user_input(chat.fe_options.use_multiple_lines)
        if input.isEmptyOrWhitespace(): continue

        stdout.write(ai_tag)
        ht.reset()
        doAssert chat.start_chat(input)
        for c in chat.chunks():
            case c.t:
                of ChunkType.Chat: ht.receive_chunk(c.chunk)
                of ChunkType.Thought: ht.receive_thought_chunk(c.chunk)

        echo ""
else:
    doAssert chat.start_chat(chat.fe_options.prompt)
    for c in chat.chunks():
        case c.t:
            of ChunkType.Chat: ht.receive_chunk(c.chunk)
            of ChunkType.Thought: ht.receive_thought_chunk(c.chunk)
    stdout.writeLine("")

chatllm_show_statistics(chat.llm)
