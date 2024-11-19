import strutils
import os
import libchatllm
import packages/docutils/highlite, terminal

import std/terminal
import std/[os, strutils]

type
  highlighter = object
    line_acc: string
    lang: SourceLanguage

proc receive_chunk(ht: var highlighter, chunk: string) =
  ht.line_acc &= chunk

  proc none_chunk(ht: var highlighter) =
    if chunk == "":
      if ht.line_acc.startsWith("```"):
        let l = ht.line_acc[3..<len(ht.line_acc)]
        ht.lang = getSourceLanguage(l)
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
  case cast[PrintType](print_type)
  of PRINT_CHAT_CHUNK:
    var ht = cast[ptr highlighter](user_data)
    var s: string = $utf8_str
    for l in s.splitLines():
      receive_chunk(ht[], l)
  else:
    if utf8_str != nil: echo utf8_str
  stdout.flushFile()

proc chatllm_end(user_data: pointer) {.cdecl.} =
  echo ""

var ht = highlighter(line_acc: "", lang: langNone)
let chat = chatllm_create()
for i in 1 .. paramCount():
  chatllm_append_param(chat, paramStr(i).cstring)

let r = chatllm_start(chat, chatllm_print, chatllm_end, addr(ht))
if r != 0:
  echo ">>> chatllm_start error: ", r
  quit(r)

while true:
  stdout.write("You  > ")
  let input = stdin.readLine()
  if input.isEmptyOrWhitespace(): continue

  stdout.write("A.I. > ")
  let r = chatllm_user_input(chat, input.cstring)
  if r != 0:
    echo ">>> chatllm_user_input error: ", r
    break
