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
        PRINT_EVT_ASYNC_COMPLETED = 100 ##  last async operation completed (utf8_str is "")

type
    chatllm_obj = object
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
proc chatllm_history_append*(obj: ptr chatllm_obj; role_type: int; utf8_str: cstring) {.stdcall, dynlib: libName, importc.}

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

##
##  @brief text embedding
##
##  embedding is emitted through `PRINTLN_EMBEDDING`.
##
##  @param[in] obj               model object
##  @param[in] utf8_str          text
##  @return                      0 if succeeded
##
proc chatllm_text_embedding*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

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
proc chatllm_async_text_embedding*(obj: ptr chatllm_obj; utf8_str: cstring): cint {.stdcall, dynlib: libName, importc.}

##
##  @brief async version of `chatllm_qa_rank`
##
##  @param   ...
##  @return                      0 if started else -1
##
proc chatllm_async_qa_rank*(obj: ptr chatllm_obj; utf8_str_q: cstring;
                            utf8_str_a: cstring): cint {.stdcall, dynlib: libName, importc.}