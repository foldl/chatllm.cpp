import strutils
import os
import libchatllm

proc chatllm_print(user_data: pointer, print_type: cint, utf8_str: cstring) {.cdecl.} =
  case cast[PrintType](print_type)
  of PRINT_CHAT_CHUNK:
    stdout.write(utf8_str)
  else:
    if utf8_str != nil: echo utf8_str
  stdout.flushFile()

proc chatllm_end(user_data: pointer) {.cdecl.} =
  echo ""

let chat = chatllm_create()
for i in 1 .. paramCount():
  chatllm_append_param(chat, paramStr(i).cstring)

let r = chatllm_start(chat, chatllm_print, chatllm_end, nil)
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
