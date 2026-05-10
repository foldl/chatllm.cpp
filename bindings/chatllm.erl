-module(chatllm).

-export([open/0, open/1, close/1, start_model/2, user_input/3]).
-export([demo/0]).

-define(FUN_ID_chatllm_append_init_param                  , 1).
-define(FUN_ID_chatllm_init                               , 2).
-define(FUN_ID_chatllm_create                             , 3).
-define(FUN_ID_chatllm_destroy                            , 4).
-define(FUN_ID_chatllm_append_param                       , 5).
-define(FUN_ID_chatllm_start                              , 6).
-define(FUN_ID_chatllm_set_gen_max_tokens                 , 7).
-define(FUN_ID_chatllm_restart                            , 8).
-define(FUN_ID_chatllm_multimedia_msg_prepare             , 9).
-define(FUN_ID_chatllm_multimedia_msg_append              , 10).
-define(FUN_ID_chatllm_history_append                     , 11).
-define(FUN_ID_chatllm_history_append_multimedia_msg      , 12).
-define(FUN_ID_chatllm_get_cursor                         , 13).
-define(FUN_ID_chatllm_set_cursor                         , 14).
-define(FUN_ID_chatllm_user_input                         , 15).
-define(FUN_ID_chatllm_user_input_multimedia_msg          , 16).
-define(FUN_ID_chatllm_set_ai_prefix                      , 17).
-define(FUN_ID_chatllm_ai_continue                        , 18).
-define(FUN_ID_chatllm_tool_input                         , 19).
-define(FUN_ID_chatllm_tool_completion                    , 20).
-define(FUN_ID_chatllm_text_tokenize                      , 21).
-define(FUN_ID_chatllm_async_start                        , 22).
-define(FUN_ID_chatllm_async_user_input                   , 23).
-define(FUN_ID_chatllm_async_user_input_multimedia_msg    , 24).
-define(FUN_ID_chatllm_async_ai_continue                  , 25).
-define(FUN_ID_chatllm_async_tool_input                   , 26).
-define(FUN_ID_chatllm_async_tool_completion              , 27).

-define(FSP_TYPE_FUN_RET                  , 1).
-define(FSP_TYPE_CALLBACK_chatllm_print   , 2).
-define(FSP_TYPE_CALLBACK_chatllm_end     , 3).

-record(state, {
          server,
          obj_pid_map = #{},
          fun_caller_map = #{}
         }).

open() ->
    ExtPrg = case os:type() of
        {win32, _} ->
            "chatllm_port.exe";
        _OSType ->
            "chatllm_port"
    end,
    open(ExtPrg).

open(ExtPrg) -> open(ExtPrg, self()).

open(ExtPrg, ServerPid) ->
    {ok, ListenSock} = gen_tcp:listen(0, [{active, false}, binary, {packet, 4}]),
    {ok, Port} = inet:port(ListenSock),
    start_ext_prg(ExtPrg, Port),
    Self = self(),
    Pid = spawn(fun () ->
        case gen_tcp:accept(ListenSock, 15000) of
            {ok, Socket} ->
                inet:setopts(Socket, [{active, true}]),
                gen_tcp:close(ListenSock),
                Self ! {Self, self()},
                loop(Socket, #state{server = ServerPid});
            X ->
                io:format("~p~n", [X]),
                gen_tcp:close(ListenSock)
        end
    end),
    receive {Self, Pid} -> ok end,
    {ok, Pid}.

start_ext_prg(ExtPrg, Port) ->
    Cmd = case os:type() of
        {win32, _} ->
            "cmd /c start cmd.exe /c \"" ++ ExtPrg ++ "\" " ++ integer_to_list(Port);
        _OSType ->
            throw(todo)
    end,
    _Port = open_port({spawn, Cmd}, []).

close(PortPid) -> PortPid ! stop.

start_model(PortPid, ModelStartArgs) ->
    {ok, Obj} = chatllm_create(PortPid),
    PortPid ! {register, Obj, self()},
    lists:foreach(fun (Param) -> ok = chatllm_append_param(PortPid, Obj, Param) end, ModelStartArgs),
    {ok, 0} = chatllm_start(PortPid, Obj),
    {ok, Obj}.

user_input(PortPid, Obj, Input) ->
    {ok, 0} = chatllm_async_user_input(PortPid, Obj, Input),
    ok.

handle_func_ret(FunId, Ret, #state{fun_caller_map = CallerMap} = State) ->
    Caller = maps:get(FunId, CallerMap, undefined),
    if is_pid(Caller) -> Caller ! {fun_ret, Caller, Ret}; true -> ok end,
    M10 = maps:remove(FunId, CallerMap),
    State#state{fun_caller_map = M10}.

loop(Socket, #state{server = _ServerPid, obj_pid_map = PidMap, fun_caller_map = CallerMap} = State) ->
    receive
        {tcp, Socket, <<?FSP_TYPE_FUN_RET, FunId:32/little, Ret/binary>>} ->
            State10 = handle_func_ret(FunId, Ret, State),
            loop(Socket, State10);
        {tcp, Socket, <<?FSP_TYPE_CALLBACK_chatllm_print, Obj:8/binary, TypeId:32/little, _Len:32/little, Chunk/binary>>} ->
            Type = parse_print_type(TypeId),
            Msg = case Type of
                async_completed -> {chatllm, Obj, async_completed};
                thought_completed -> {chatllm, Obj, thought_completed};
                X -> {chatllm, Obj, X, Chunk}
            end,
            case maps:get(Obj, PidMap, undefined) of
                undefined -> ok;
                ProcId -> ProcId ! Msg
            end,
            loop(Socket, State);
        {tcp, Socket, <<?FSP_TYPE_CALLBACK_chatllm_end, Obj:8/binary>>} ->
            case maps:get(Obj, PidMap, undefined) of
                undefined -> ok;
                ProcId -> ProcId ! {chatllm, Obj, 'end'}
            end,
            loop(Socket, State);
        {tcp_closed, Socket} ->
            ok;
        {register, Obj, ProcId} ->
            PidMap10 = PidMap#{Obj => ProcId},
            loop(Socket, State#state{obj_pid_map = PidMap10});
        {call_fun, CallerPid, FunId, Param} ->
            ok = gen_tcp:send(Socket, <<FunId:32/little, Param/binary>>),
            M10 = CallerMap#{FunId => CallerPid},
            loop(Socket, State#state{fun_caller_map = M10});
        {call_fun, FunId, Param} ->
            ok = gen_tcp:send(Socket, <<FunId:32/little, Param/binary>>),
            loop(Socket, State);
        stop ->
            gen_tcp:close(Socket);
        {'EXIT', _Pid, _Reason} ->
            exit(port_terminated);
        X ->
            io:format("unknown message: ~p~n", [X]),
            loop(Socket, State)
    end.

call_function(PortPid, FunId, Param) -> call_function(PortPid, FunId, Param, 1000).

call_function(PortPid, FunId, Param, Timeout) ->
    Self = self(),
    PortPid ! {call_fun, Self, FunId, Param},
    receive
        {fun_ret, Self, Ret} -> {ok, Ret}
    after
        Timeout -> timeout
    end.

call_void_function(PortPid, FunId, Param) ->
    PortPid ! {call_fun, FunId, Param},
    ok.

chatllm_create(PortPid) ->
    call_function(PortPid, ?FUN_ID_chatllm_create, <<>>).

chatllm_append_param(PortPid, Obj, Param) when is_binary(Param) ->
    Len = size(Param),
    call_void_function(PortPid, ?FUN_ID_chatllm_append_param, <<Obj/binary, Len:32/little, Param/binary>>);
chatllm_append_param(PortPid, Obj, Param) when is_list(Param) ->
    chatllm_append_param(PortPid, Obj, unicode:characters_to_binary(Param)).

chatllm_start(PortPid, Obj) ->
    case call_function(PortPid, ?FUN_ID_chatllm_start, <<Obj/binary>>, 1000 * 1000) of
        {ok, <<Ret:32/little>>} -> {ok, Ret};
        Other -> {error, Other}
    end.

chatllm_async_start(PortPid, Obj) ->
    case call_function(PortPid, ?FUN_ID_chatllm_async_start, <<Obj/binary>>) of
        {ok, <<Ret:32/little>>} -> {ok, Ret};
        Other -> {error, Other}
    end.

chatllm_async_user_input(PortPid, Obj, Input) when is_binary(Input) ->
    Len = size(Input),
    case call_function(PortPid, ?FUN_ID_chatllm_async_user_input, <<Obj/binary, Len:32/little, Input/binary>>) of
        {ok, <<Ret:32/little>>} -> {ok, Ret};
        Other -> {error, Other}
    end;
chatllm_async_user_input(PortPid, Obj, Input) when is_list(Input) ->
    chatllm_async_user_input(PortPid, Obj, unicode:characters_to_binary(Input)).

parse_print_type(  0) -> chat_chunk      ;
parse_print_type(  1) -> meta            ;
parse_print_type(  2) -> error           ;
parse_print_type(  3) -> ref             ;
parse_print_type(  4) -> rewritten_query ;
parse_print_type(  5) -> history_user    ;
parse_print_type(  6) -> history_ai      ;
parse_print_type(  7) -> tool_calling    ;
parse_print_type(  8) -> embedding       ;
parse_print_type(  9) -> ranking         ;
parse_print_type( 10) -> token_ids       ;
parse_print_type( 11) -> logging         ;
parse_print_type( 12) -> beam_search     ;
parse_print_type( 13) -> model_info      ;
parse_print_type( 14) -> thought_chunk   ;
parse_print_type(100) -> async_completed   ;
parse_print_type(101) -> thought_completed ;
parse_print_type(  X) -> {unknown, X}.

demo() ->
    {ok, Pid} = open(),
    {ok, Model} = start_model(Pid, ["-m", ":qwen3:0.6b", "--max-length", "1000"]),
    user_input(Pid, Model, "hello"),
    loop(Model),
    close(Pid).

loop(Model) ->
    receive
        {chatllm, Model, meta, Msg} ->
            io:format("~ts~n", [Msg]),
            loop(Model);
        {chatllm, Model, model_info, Msg} ->
            loop(Model);
        {chatllm, Model, chat_chunk, Chunk} ->
            io:format("~ts", [Chunk]),
            loop(Model);
        {chatllm, Model, thought_chunk, Chunk} ->
            io:format("~ts", [Chunk]),
            loop(Model);
        {chatllm, Model, MsgType, Msg} ->
            io:format("~p~ts~n", [MsgType, Msg]),
            loop(Model);
        {chatllm, Model, async_completed} ->
            ok;
        {chatllm, Model, 'end'} ->
            io:format("~n", []),
            loop(Model);
        {chatllm, Model, MsgType} ->
            io:format("~p~n", [MsgType]),
            loop(Model);
        Other ->
            io:format("OTHER: ~p~n", [Other]),
            loop(Model)
    after
        4000 -> ok
    end.
