import std/[net]
import std/[os, osproc, cmdline, strutils, strformat, json, tables, options, times, sequtils, parseutils]
import libchatllm

# big-endian
const PACKET_LENGTH_LEN = 4

const FUN_ID_chatllm_append_init_param                  = 1
const FUN_ID_chatllm_init                               = 2
const FUN_ID_chatllm_create                             = 3
const FUN_ID_chatllm_destroy                            = 4
const FUN_ID_chatllm_append_param                       = 5
const FUN_ID_chatllm_start                              = 6
const FUN_ID_chatllm_set_gen_max_tokens                 = 7
const FUN_ID_chatllm_restart                            = 8
const FUN_ID_chatllm_multimedia_msg_prepare             = 9
const FUN_ID_chatllm_multimedia_msg_append              = 10
const FUN_ID_chatllm_history_append                     = 11
const FUN_ID_chatllm_history_append_multimedia_msg      = 12
const FUN_ID_chatllm_get_cursor                         = 13
const FUN_ID_chatllm_set_cursor                         = 14
const FUN_ID_chatllm_user_input                         = 15
const FUN_ID_chatllm_user_input_multimedia_msg          = 16
const FUN_ID_chatllm_set_ai_prefix                      = 17
const FUN_ID_chatllm_ai_continue                        = 18
const FUN_ID_chatllm_tool_input                         = 19
const FUN_ID_chatllm_tool_completion                    = 20
const FUN_ID_chatllm_text_tokenize                      = 21
const FUN_ID_chatllm_async_start                        = 22
const FUN_ID_chatllm_async_user_input                   = 23
const FUN_ID_chatllm_async_user_input_multimedia_msg    = 24
const FUN_ID_chatllm_async_ai_continue                  = 25
const FUN_ID_chatllm_async_tool_input                   = 26
const FUN_ID_chatllm_async_tool_completion              = 27

const RSP_TYPE_FUN_RET                  = 1
const RSP_TYPE_CALLBACK_chatllm_print   = 2
const RSP_TYPE_CALLBACK_chatllm_end     = 3

proc to_be_seq_32bit[T](v: T, result: var openArray[uint8], pos: Natural = 0) =
    result[pos + 0] = cast[uint8]((v shr 24) and 0xff)
    result[pos + 1] = cast[uint8]((v shr 16) and 0xff)
    result[pos + 2] = cast[uint8]((v shr  8) and 0xff)
    result[pos + 3] = cast[uint8]((v shr  0) and 0xff)

type
    Comm = ref object of RootObj

    CommFile = ref object of Comm
        input: File
        output: File

    CommSocket = ref object of Comm
        socket: Socket

method send_response(comm: Comm, rsp: seq[uint8]): bool = false
method read_command(comm: Comm, cmd: var seq[uint8]): bool = false

method send_response(comm: CommFile, rsp: seq[uint8]): bool =
    var header: array[0 .. PACKET_LENGTH_LEN - 1, uint8]
    to_be_seq_32bit(rsp.len, header)
    if comm.output.writeBytes(header, 0, header.len) != header.len: return false
    return comm.output.writeBytes(rsp, 0, rsp.len) == rsp.len

method read_command(comm: CommFile, cmd: var seq[uint8]): bool =
    var header: array[0 .. PACKET_LENGTH_LEN - 1, uint8]
    if comm.input.readBytes(header, 0, len(header)) != PACKET_LENGTH_LEN: return false
    var pkt_len: int = 0
    for i in 0 .. high(header):
        pkt_len = (pkt_len shl 8) or cast[int](header[i])
    cmd.setLenUninit(pkt_len)
    if comm.input.readBytes(cmd, 0, pkt_len) != pkt_len: return false
    return true

method send_response(comm: CommSocket, rsp: seq[uint8]): bool =
    var header: array[0 .. PACKET_LENGTH_LEN - 1, uint8]
    to_be_seq_32bit(rsp.len, header)
    if comm.socket.send(addr header[0], header.len) != header.len: return false
    return comm.socket.send(addr rsp[0], rsp.len) == rsp.len

method read_command(comm: CommSocket, cmd: var seq[uint8]): bool =
    var header: array[0 .. PACKET_LENGTH_LEN - 1, uint8]
    if comm.socket.recv(addr header, len(header)) != PACKET_LENGTH_LEN: return false
    var pkt_len: int = 0
    for i in 0 .. high(header):
        pkt_len = (pkt_len shl 8) or cast[int](header[i])
    cmd.setLenUninit(pkt_len)
    return comm.socket.recv(addr cmd[0], pkt_len) == pkt_len

var
    channel: Comm = nil
proc send_response(rsp: seq[uint8]): bool = channel.send_response(rsp)
proc read_command(cmd: var seq[uint8]): bool = channel.read_command(cmd)

proc append_data(rsp: var seq[uint8], v: int32) =
    let l = rsp.len
    rsp.setLen(l + 4)
    copyMem(addr rsp[l], addr v, 4)

proc append_data(rsp: var seq[uint8], v: pointer) =
    let l = rsp.len
    rsp.setLen(l + 8)
    copyMem(addr rsp[l], addr v, 8)

proc append_data(rsp: var seq[uint8], v: string) =
    rsp.append_data cast[int32](v.len)
    let l = rsp.len
    rsp.setLen(l + v.len)
    if v.len > 0: copyMem(addr rsp[l], addr v[0], v.len)

proc send_response_fun_ret(id: int32, ret: int32): bool =
    var rsp = @[cast[uint8](RSP_TYPE_FUN_RET)]
    rsp.append_data(id)
    rsp.append_data(ret)
    return send_response(rsp)

proc port_on_end(user_data: pointer) {.cdecl.} =
    var rsp = @[cast[uint8](RSP_TYPE_CALLBACK_chatllm_end)]
    rsp.append_data(user_data)
    discard send_response(rsp)

proc port_on_print(user_data: pointer, print_type: cint, utf8_str: cstring) {.cdecl.} =
    var rsp = @[cast[uint8](RSP_TYPE_CALLBACK_chatllm_print)]
    rsp.append_data(user_data)
    rsp.append_data(print_type)
    rsp.append_data(if utf8_str != nil: $ utf8_str else: "")
    discard send_response(rsp)

proc read_uint32(cmd: seq[uint8], start: int = 0): uint32 =
    return (cast[uint32](cmd[start + 3]) shl 24) or (cast[uint32](cmd[start + 2]) shl 16) or (cast[uint32](cmd[start + 1]) shl 8) or cmd[start + 0]

proc parse_param_S(cmd: seq[uint8], start: int = 0): string =
    let l = read_uint32(cmd, start)
    result.setLen(l)
    copyMem(addr(result[0]), addr(cmd[start + 4]), l)

proc parse_param_ptr(cmd: seq[uint8], start: int = 0): pointer =
    return cast[pointer]((cast[uint64](read_uint32(cmd, start + 4)) shl 32) or cast[uint64](read_uint32(cmd, start + 0)))

var
    obj_param_cache: Table[pointer, seq[string]]

proc prepare_params_for_obj(obj: pointer): seq[string] =
    result = @[]
    let args = obj_param_cache[obj]
    var i = 0
    while i < len(args):
        let s = args[i]
        result.add s
        if s.cmd_opt_is_model_selection():
            inc i
            if i >= len(args): break
            result.add normalize_model_name(args[i])
        inc i

proc handle_command(cmd: seq[uint8]): bool =
    if cmd.len < 4: return false
    let cmd_id = read_uint32(cmd, 0)
    let param = cmd[4..^1]
    case cmd_id:
        of FUN_ID_chatllm_append_init_param:
            chatllm_append_init_param(cstring(parse_param_S(param)))
        of FUN_ID_chatllm_init:
            block:
                let ret = chatllm_init()
                return send_response_fun_ret(FUN_ID_chatllm_init, ret)
        of FUN_ID_chatllm_create:
            block:
                let ret = chatllm_create()
                var rsp = @[cast[uint8](RSP_TYPE_FUN_RET)]
                rsp.append_data(FUN_ID_chatllm_create)
                rsp.append_data(ret)
                obj_param_cache[ret] = newSeq[string]()
                return send_response(rsp)
        of FUN_ID_chatllm_destroy:
            block:
                let ret = chatllm_destroy(cast[ptr chatllm_obj](parse_param_ptr(param)))
                return send_response_fun_ret(FUN_ID_chatllm_destroy, ret)
        of FUN_ID_chatllm_append_param:
            block:
                let obj = cast[ptr chatllm_obj](parse_param_ptr(param, 0))
                let arg = parse_param_S(param, 8)
                obj_param_cache[obj].add arg
        of FUN_ID_chatllm_start:
            block:
                let obj = cast[ptr chatllm_obj](parse_param_ptr(param, 0))
                let args = prepare_params_for_obj(obj)
                for s in args: chatllm_append_param(obj, cstring(s))
                let ret = chatllm_start(obj, port_on_print, port_on_end, obj)
                return send_response_fun_ret(FUN_ID_chatllm_start, ret)
        of FUN_ID_chatllm_async_start:
            block:
                let obj = cast[ptr chatllm_obj](parse_param_ptr(param, 0))
                let args = prepare_params_for_obj(obj)
                for s in args: chatllm_append_param(obj, cstring(s))
                let ret = chatllm_async_start(obj, port_on_print, port_on_end, obj)
                return send_response_fun_ret(FUN_ID_chatllm_async_start, ret)
        of FUN_ID_chatllm_async_user_input:
            block:
                let obj = cast[ptr chatllm_obj](parse_param_ptr(param, 0))
                let input = parse_param_s(param, 8)
                let ret = chatllm_async_user_input(obj, cstring(input))
                return send_response_fun_ret(FUN_ID_chatllm_async_user_input, ret)
        else:
            return false
    return true

proc main(): int =
    if paramCount() == 1:
        var comm: CommSocket
        new(comm)
        comm.socket = newSocket()
        echo fmt"connect to {paramStr(1)}"
        comm.socket.connect("localhost", Port(parseInt(paramStr(1))))
        channel = comm
    else:
        echo fmt"using stdio (not recommended)"
        var comm: CommFile
        new(comm)
        comm.output = stdout
        comm.input  = stdin
        channel = comm

    var cmd: seq[uint8]
    while true:
        if not read_command(cmd): break
        if not handle_command(cmd): break

quit(main())