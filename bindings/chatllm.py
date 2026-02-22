from ctypes import *
from enum import IntEnum
import os, sys, signal, queue
import threading
import json, base64
from typing import Any, Iterable, List, Union

try:
    import model_downloader
except:
    this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    sys.path.append(os.path.join(this_dir, '..', 'scripts'))
    import model_downloader

class PrintType(IntEnum):
    PRINT_CHAT_CHUNK        = 0,
    PRINTLN_META            = 1,    # print a whole line: general information
    PRINTLN_ERROR           = 2,    # print a whole line: error message
    PRINTLN_REF             = 3,    # print a whole line: reference
    PRINTLN_REWRITTEN_QUERY = 4,    # print a whole line: rewritten query
    PRINTLN_HISTORY_USER    = 5,    # print a whole line: user input history
    PRINTLN_HISTORY_AI      = 6,    # print a whole line: AI output history
    PRINTLN_TOOL_CALLING    = 7,    # print a whole line: tool calling (supported by only a few models)
    PRINTLN_EMBEDDING       = 8,    # print a whole line: embedding (example: "0.1,0.3,...")
    PRINTLN_RANKING         = 9,    # print a whole line: ranking (example: "0.8")
    PRINTLN_TOKEN_IDS       =10,    # print a whole line: token ids (example: "1,3,5,8, ...")
    PRINTLN_LOGGING         =11,    # print a whole line: internal logging with the first char indicating level
                                    # (space): None; D: Debug; I: Info; W: Warn; E: Error; .: continue
    PRINTLN_BEAM_SEARCH     =12,    # print a whole line: a result of beam search with a prefix of probability
                                    # (example: "0.8,....")
    PRINTLN_MODEL_INFO      =13,    # when a model is started, print a whole line of basic model information (json format)
                                    # (example: {"name": "llama", "context_length": 100, "capabilities": [text, ...], ...})
    PRINT_THOUGHT_CHUNK     =14,    # same as PRINT_CHAT_CHUNK, but this from "thoughts".
                                    # possible leading or trailing tags (such as <think>, </think>) are removed.
                                    # use `+detect_thoughts` to enable this.

    PRINT_EVT_ASYNC_COMPLETED       = 100,   # last async operation completed (utf8_str is null)
    PRINT_EVT_THOUGHT_COMPLETED     = 101,   # thought completed

class EmbeddingPurpose(IntEnum):
    Document = 0,                   # for document
    Query    = 1,                   # for query

def dll_name(lib: str) -> str:
    if sys.platform == 'win32':
        return lib + '.dll'
    elif sys.platform == 'darwin':
        return lib + '.dylib'
    else:
        return lib + '.so'

class LibChatLLM:

    _obj2id = {}
    _id2obj = {}

    def __init__(self, lib: str = '', model_storage: str = '', init_params: list[str] = []) -> None:

        if lib == '':
            this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            lib = this_dir
            if not os.path.isfile(os.path.join(lib, dll_name('libchatllm'))):
                lib = os.path.abspath(os.path.join(this_dir, '..', 'bin'))
                assert os.path.isfile(os.path.join(lib, dll_name('libchatllm')))

        self._lib_path = lib
        self.model_storage = os.path.abspath(model_storage if model_storage != '' else os.path.join(lib, '..', 'quantized'))

        init_params = ['--ggml_dir', lib] + init_params
        lib = os.path.join(lib, dll_name('libchatllm'))
        if sys.platform == 'win32':
            import re
            for path in os.getenv('PATH').split(';'):
                if re.match(r'.+\\CUDA\\v[0-9]+\.[0-9]+\\bin', path) is not None:
                    os.add_dll_directory(path)

        if sys.platform == 'win32':
            self._lib = windll.LoadLibrary(lib)
            self._PRINTFUNC = WINFUNCTYPE(None, c_void_p, c_int, c_char_p)
            self._ENDFUNC = WINFUNCTYPE(None, c_void_p)
        else:
            self._lib = cdll.LoadLibrary(lib)
            self._PRINTFUNC = CFUNCTYPE(None, c_void_p, c_int, c_char_p)
            self._ENDFUNC = CFUNCTYPE(None, c_void_p)

        _chatllm_append_init_param = self._lib.chatllm_append_init_param
        _chatllm_append_init_param.restype = None
        _chatllm_append_init_param.argtypes = [c_char_p]
        _chatllm_init = self._lib.chatllm_init
        _chatllm_init.restype = c_int
        _chatllm_init.argtypes = []

        for s in init_params:
            _chatllm_append_init_param(c_char_p(s.encode()))
        assert _chatllm_init() == 0

        self._chatllm_create            = self._lib.chatllm_create
        self._chatllm_append_param      = self._lib.chatllm_append_param
        self._chatllm_start             = self._lib.chatllm_start
        self._chatllm_set_ai_prefix     = self._lib.chatllm_set_ai_prefix
        self._chatllm_ai_continue       = self._lib.chatllm_ai_continue
        self._chatllm_user_input        = self._lib.chatllm_user_input
        self._chatllm_tool_input        = self._lib.chatllm_tool_input
        self._chatllm_tool_completion   = self._lib.chatllm_tool_completion
        self._chatllm_text_tokenize     = self._lib.chatllm_text_tokenize
        self._chatllm_text_embedding    = self._lib.chatllm_embedding
        self._chatllm_qa_rank           = self._lib.chatllm_qa_rank
        self._chatllm_rag_select_store  = self._lib.chatllm_rag_select_store
        self._chatllm_abort_generation  = self._lib.chatllm_abort_generation
        self._chatllm_restart           = self._lib.chatllm_restart
        self._chatllm_set_gen_max_tokens= self._lib.chatllm_set_gen_max_tokens
        self._chatllm_show_statistics   = self._lib.chatllm_show_statistics
        self._chatllm_save_session      = self._lib.chatllm_save_session
        self._chatllm_load_session      = self._lib.chatllm_load_session
        self._chatllm_multimedia_msg_prepare        = self._lib.chatllm_multimedia_msg_prepare
        self._chatllm_multimedia_msg_append         = self._lib.chatllm_multimedia_msg_append
        self._chatllm_user_input_multimedia_msg     = self._lib.chatllm_user_input_multimedia_msg

        self._chatllm_async_user_input      = self._lib.chatllm_async_user_input
        self._chatllm_async_ai_continue     = self._lib.chatllm_async_ai_continue
        self._chatllm_async_tool_input      = self._lib.chatllm_async_tool_input
        self._chatllm_async_tool_completion = self._lib.chatllm_async_tool_completion
        self._chatllm_async_user_input_multimedia_msg = self._lib.chatllm_async_user_input_multimedia_msg

        self._chatllm_create.restype = c_void_p
        self._chatllm_create.argtypes = []

        self._chatllm_append_param.restype = None
        self._chatllm_append_param.argtypes = [c_void_p, c_char_p]

        self._chatllm_start.restype = c_int
        self._chatllm_start.argtypes = [c_void_p, self._PRINTFUNC, self._ENDFUNC, c_void_p]

        self._chatllm_set_ai_prefix.restype = c_int
        self._chatllm_set_ai_prefix.argtypes = [c_void_p, c_char_p]

        self._chatllm_ai_continue.restype = c_int
        self._chatllm_ai_continue.argtypes = [c_void_p, c_char_p]
        self._chatllm_async_ai_continue.restype = c_int
        self._chatllm_async_ai_continue.argtypes = [c_void_p, c_char_p]

        self._chatllm_multimedia_msg_prepare.argtypes = [c_void_p]
        self._chatllm_multimedia_msg_append.restype = c_int
        self._chatllm_multimedia_msg_append.argtypes = [c_void_p, c_char_p, c_char_p]

        self._chatllm_user_input.restype = c_int
        self._chatllm_user_input.argtypes = [c_void_p, c_char_p]
        self._chatllm_async_user_input.restype = c_int
        self._chatllm_async_user_input.argtypes = [c_void_p, c_char_p]

        self._chatllm_user_input_multimedia_msg.restype         = c_int
        self._chatllm_user_input_multimedia_msg.argtypes        = [c_void_p]
        self._chatllm_async_user_input_multimedia_msg.restype   = c_int
        self._chatllm_async_user_input_multimedia_msg.argtypes  = [c_void_p]

        self._chatllm_tool_input.restype = c_int
        self._chatllm_tool_input.argtypes = [c_void_p, c_char_p]
        self._chatllm_async_tool_input.restype = c_int
        self._chatllm_async_tool_input.argtypes = [c_void_p, c_char_p]

        self._chatllm_tool_completion.restype = c_int
        self._chatllm_tool_completion.argtypes = [c_void_p, c_char_p]
        self._chatllm_async_tool_completion.restype = c_int
        self._chatllm_async_tool_completion.argtypes = [c_void_p, c_char_p]

        self._chatllm_text_embedding.restype = c_int
        self._chatllm_text_embedding.argtypes = [c_void_p, c_char_p, c_int]

        self._chatllm_text_tokenize.restype = c_int
        self._chatllm_text_tokenize.argtypes = [c_void_p, c_char_p]

        self._chatllm_qa_rank.restype = c_int
        self._chatllm_qa_rank.argtypes = [c_void_p, c_char_p, c_char_p]

        self._chatllm_rag_select_store.restype = c_int
        self._chatllm_rag_select_store.argtypes = [c_void_p, c_char_p]

        self._chatllm_abort_generation.restype = None
        self._chatllm_abort_generation.argtypes = [c_void_p]

        self._chatllm_restart.restype = None
        self._chatllm_restart.argtypes = [c_void_p, c_char_p]

        self._chatllm_set_gen_max_tokens.restype = None
        self._chatllm_set_gen_max_tokens.argtypes = [c_void_p, c_int]

        self._chatllm_show_statistics.restype = None
        self._chatllm_show_statistics.argtypes = [c_void_p]

        self._chatllm_save_session.restype = c_int
        self._chatllm_save_session.argtypes = [c_void_p, c_char_p]
        self._chatllm_load_session.restype = c_int
        self._chatllm_load_session.argtypes = [c_void_p, c_char_p]

        self._cb_print = self._PRINTFUNC(LibChatLLM.callback_print)
        self._cb_end = self._ENDFUNC(LibChatLLM.callback_end)

    @staticmethod
    def callback_print(user_data: int, print_type: c_int, s: bytes) -> None:
        obj = LibChatLLM._id2obj[user_data]

        if print_type == PrintType.PRINT_EVT_ASYNC_COMPLETED.value:
            obj.callback_async_done()
            return

        txt = s.decode()
        if print_type == PrintType.PRINT_CHAT_CHUNK.value:
            obj.callback_print(txt)
        elif print_type == PrintType.PRINT_THOUGHT_CHUNK.value:
            obj.callback_print_thought(txt)
        elif print_type == PrintType.PRINTLN_META.value:
            obj.callback_print_meta(txt)
        elif print_type == PrintType.PRINTLN_REF.value:
            obj.callback_print_reference(txt)
        elif print_type == PrintType.PRINTLN_REWRITTEN_QUERY.value:
            obj.callback_print_rewritten_query(txt)
        elif print_type == PrintType.PRINTLN_HISTORY_USER.value:
            obj.callback_print_history_user(txt)
        elif print_type == PrintType.PRINTLN_HISTORY_AI.value:
            obj.callback_print_history_ai(txt)
        elif print_type == PrintType.PRINTLN_TOOL_CALLING.value:
            obj.call_tool(txt)
        elif print_type == PrintType.PRINTLN_EMBEDDING.value:
            obj.callback_print_embedding(txt)
        elif print_type == PrintType.PRINTLN_RANKING.value:
            obj.callback_print_ranking(txt)
        elif print_type == PrintType.PRINTLN_TOKEN_IDS.value:
            obj.callback_text_tokenize(txt)
        elif print_type == PrintType.PRINTLN_ERROR.value:
            raise Exception(txt)
        elif print_type == PrintType.PRINTLN_LOGGING.value:
            obj.callback_print_log(txt)
        elif print_type == PrintType.PRINTLN_BEAM_SEARCH.value:
            obj.callback_print_beam_search(txt)
        elif print_type == PrintType.PRINT_EVT_ASYNC_COMPLETED.value:
            obj.callback_async_done()
        elif print_type == PrintType.PRINTLN_MODEL_INFO.value:
            obj._model_info = json.loads(txt)
        else:
            raise Exception(f"unhandled print_type({print_type}): {txt}")

    @staticmethod
    def callback_end(user_data: int) -> None:
        obj = LibChatLLM._id2obj[user_data]
        obj.callback_end()

    def alloc_id_for_obj(self, obj: Any) -> int:
        if obj in LibChatLLM._obj2id:
            return LibChatLLM._obj2id[obj]
        id = len(LibChatLLM._obj2id) + 1
        LibChatLLM._obj2id[obj] = id
        LibChatLLM._id2obj[id] = obj
        return id

    def append_param(self, obj: c_void_p, param: Union[str, List[str]]) -> None:
        if isinstance(param, str):
            param = [param]
            return

        param = model_downloader.preprocess_args(param, self.model_storage)
        for s in param:
            self._chatllm_append_param(obj, c_char_p(s.encode()))

    def start(self, obj: c_void_p, callback_obj: Any) -> int:
        id = self.alloc_id_for_obj(callback_obj)
        return self._chatllm_start(obj, self._cb_print, self._cb_end, c_void_p(id))

    def set_ai_prefix(self, obj: c_void_p, prefix: str) -> int:
        return self._chatllm_set_ai_prefix(obj, c_char_p(prefix.encode()))

    def _input_multimedia_msg(self, obj: c_void_p, user_input: List[dict | str]) -> int:
        self._chatllm_multimedia_msg_prepare(obj)
        for x in user_input:
            if isinstance(x, str):
                self._chatllm_multimedia_msg_append(obj, c_char_p('text'), c_char_p(x))
            elif isinstance(x, dict):
                t = x['type']
                if t == 'text':
                    data = x['text'].encode()
                else:
                    if 'file' in x:
                        with open(x['file'], 'rb') as f:
                            data = f.read()
                    elif 'url' in x:
                        url: str = x['url']
                        if url.startswith('data:'):
                            i = url.find('base64,')
                            data = base64.decodebytes(url[i + 7 :].encode())
                        else:
                            data = model_downloader.download_file_to_bytes(url)
                    else:
                        raise Exception(f'unknown message piece: {x}')
                    data = base64.b64encode(data)
                self._chatllm_multimedia_msg_append(obj, c_char_p(t.encode()), c_char_p(data))

    def chat(self, obj: c_void_p, user_input: str | List[dict | str]) -> int:
        if isinstance(user_input, str):
            return self._chatllm_user_input(obj, c_char_p(user_input.encode()))
        elif isinstance(user_input, list):
            self._input_multimedia_msg(obj, user_input)
            return self._chatllm_user_input_multimedia_msg(obj)

    def async_chat(self, obj: c_void_p, user_input: str | List[dict | str]) -> int:
        if isinstance(user_input, str):
            return self._chatllm_async_user_input(obj, c_char_p(user_input.encode()))
        else:
            self._input_multimedia_msg(obj, user_input)
            self._chatllm_async_user_input_multimedia_msg(obj)

    def ai_continue(self, obj: c_void_p, suffix: str) -> int:
        return self._chatllm_ai_continue(obj, c_char_p(suffix.encode()))

    def tool_input(self, obj: c_void_p, user_input: str) -> int:
        return self._chatllm_tool_input(obj, c_char_p(user_input.encode()))

    def async_tool_completion(self, obj: c_void_p, user_input: str) -> int:
        return self._chatllm_async_tool_completion(obj, c_char_p(user_input.encode()))

    def async_ai_continue(self, obj: c_void_p, suffix: str) -> int:
        return self._chatllm_async_ai_continue(obj, c_char_p(suffix.encode()))

    def async_tool_input(self, obj: c_void_p, user_input: str) -> int:
        return self._chatllm_async_tool_input(obj, c_char_p(user_input.encode()))

    def tool_completion(self, obj: c_void_p, user_input: str) -> int:
        return self._chatllm_tool_completion(obj, c_char_p(user_input.encode()))

    def text_tokenize(self, obj: c_void_p, text: str) -> str:
        return self._chatllm_text_tokenize(obj, c_char_p(text.encode()))

    def embedding(self, obj: c_void_p, text: str, purpose: EmbeddingPurpose = EmbeddingPurpose.Document) -> str:
        return self._chatllm_text_embedding(obj, c_char_p(text.encode()), c_int(purpose.value))

    def qa_rank(self, obj: c_void_p, q: str, a: str) -> float:
        return self._chatllm_qa_rank(obj, c_char_p(q.encode()), c_char_p(a.encode()))

    def rag_select_store(self, obj: c_void_p, store_name: str) -> str:
        return self._chatllm_rag_select_store(obj, c_char_p(store_name.encode()))

    def abort(self, obj: c_void_p) -> None:
        self._chatllm_abort_generation(obj)

    def restart(self, obj: c_void_p, sys_prompt: str | None = None) -> None:
        self._chatllm_restart(obj, c_char_p(sys_prompt.encode()) if sys_prompt is not None else c_char_p(None))

    def set_max_gen_tokens(self, obj: c_void_p, max_gen: int) -> None:
        self._chatllm_set_gen_max_tokens(obj, max_gen)

    def show_statistics(self, obj: c_void_p) -> None:
        self._chatllm_show_statistics(obj)

    def save_session(self, obj: c_void_p, file_name: str) -> str:
        return self._chatllm_save_session(obj, c_char_p(file_name.encode()))

    def load_session(self, obj: c_void_p, file_name: str) -> str:
        return self._chatllm_load_session(obj, c_char_p(file_name.encode()))

class LLMChatDone:
    def __init__(self, id: Any) -> None:
        self.id = id

class LLMChatChunk:
    def __init__(self, id: Any, chunk: str) -> None:
        self.id = id
        self.chunk = chunk

class LLMChatThoughtChunk:
    def __init__(self, id: Any, chunk: str) -> None:
        self.id = id
        self.chunk = chunk

class LLMChatMeta:
    def __init__(self, id: Any, text: str) -> None:
        self.id = id
        self.text = text

def is_same_command_option(a: str, b: Union[str, list[str]]) -> bool:
    if isinstance(b, list):
        for s in b:
            if is_same_command_option(a, s): return True
        return False

    if len(a) != len(b): return False
    for i in range(len(a)):
        c1 = a[i]
        c2 = b[i]
        if c1 == '-': c1 = '_'
        if c2 == '-': c2 = '_'
        if c1 != c2: return False
    return True

class FrontendOptions:
    def __init__(self, param: Union[None, str, List[str]]) -> None:
        self.help = False
        self.interactive = False
        self.reversed_role = False
        self.use_multiple_lines = False
        self.single_turn = False
        self.prompt = ""
        self.sys_prompt = ""

        if param is None: return
        if isinstance(param, str):
            param = [param]

        i = 0
        while i < len(param):
            s = param[i]
            if is_same_command_option(s, ["-h", "--help"]):
                self.help = True
            elif is_same_command_option(s, ["-i", "--interactive"]):
                self.interactive = True
            elif is_same_command_option(s, "--reversed_role"):
                self.reversed_role = True
            elif is_same_command_option(s, "--multi"):
                self.use_multiple_lines = True
            elif is_same_command_option(s, "+single_turn"):
                self.single_turn = True
            elif is_same_command_option(s, ["-p", "--prompt"]):
                i += 1
                if i < len(param): self.prompt = param[i]
            elif is_same_command_option(s, ["-s", "--system"]):
                i += 1
                if i < len(param): self.sys_prompt = param[i]
            i += 1

class ChatLLM:
    def __init__(self, lib: LibChatLLM, param: Union[None, str, List[str]], auto_start: bool = True) -> None:
        self._lib = lib
        self._chat = lib._chatllm_create()
        self.is_generating = False
        self.out_queue = None
        self.input_id = None
        self.tool_input_id = None
        self.references = []
        self.beam_search_results = []
        self.rewritten_query = ''
        self._result_embedding = None
        self._result_ranking = None
        self._result_text_tokenize = None
        self._model_info = None
        self.is_first_thought_chunk = True
        self.fe_options = FrontendOptions(param)
        if param is not None:
            self.append_param(param)
            if auto_start:
                self.start()

    def append_param(self, param: Union[str, List[str]]) -> None:
        self._lib.append_param(self._chat, param)

    def set_ai_prefix(self, prefix: str) -> int:
        return self._lib.set_ai_prefix(self._chat, prefix)

    def start(self) -> None:
        r = self._lib.start(self._chat, self)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `start()` with error code {r}')

    def get_model_info(self) -> dict:
        if self._model_info is None:
            raise Exception('Model info not available')
        return self._model_info

    def chat(self, user_input: str, input_id = None) -> None:
        self.is_generating = True
        self.input_id = input_id
        self.references = []
        self.beam_search_results = []
        self.rewritten_query = ''
        self.is_first_thought_chunk = True
        if self.fe_options.single_turn:
            self.restart()
        r = self._lib.chat(self._chat, user_input)
        self.is_generating = False
        if r != 0:
            raise Exception(f'ChatLLM: failed to `chat()` with error code {r}')

    def async_chat(self, user_input: str, input_id = None) -> None:
        self.is_generating = True
        self.input_id = input_id
        self.references = []
        self.beam_search_results = []
        self.rewritten_query = ''
        self.is_first_thought_chunk = True
        r = self._lib.async_chat(self._chat, user_input)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `async_chat()` with error code {r}')

    def ai_continue(self, suffix: str) -> int:
        self.is_generating = True
        r = self._lib.ai_continue(self._chat, suffix)
        self.is_generating = False
        if r != 0:
            raise Exception(f'ChatLLM: failed to `ai_continue()` with error code {r}')

    def tool_input(self, user_input: str, input_id = None) -> None:
        self.tool_input_id = input_id
        r = self._lib.tool_input(self._chat, user_input)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `tool_input()` with error code {r}')

    def tool_completion(self, user_input: str, completion_id = None) -> None:
        self.tool_completion_id = completion_id
        self.abort()
        r = self._lib.tool_completion(self._chat, user_input)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `tool_completion()` with error code {r}')

    def async_ai_continue(self, suffix: str) -> int:
        self.is_generating = True
        r = self._lib.async_ai_continue(self._chat, suffix)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `ai_continue()` with error code {r}')

    def async_tool_input(self, user_input: str, input_id = None) -> None:
        self.tool_input_id = input_id
        r = self._lib.async_tool_input(self._chat, user_input)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `tool_input()` with error code {r}')

    def async_tool_completion(self, user_input: str, completion_id = None) -> None:
        self.tool_completion_id = completion_id
        self.abort()
        r = self._lib.async_tool_completion(self._chat, user_input)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `tool_completion()` with error code {r}')

    def text_tokenize(self, txt: str) -> list[int]:
        self._result_text_tokenize = ''
        assert self._lib.text_tokenize(self._chat, txt) >= 0, 'text_tokenize failed'
        return json.loads(f"[{self._result_text_tokenize}]")

    def embedding(self, txt: str) -> list[float]:
        self._result_embedding = ''
        assert self._lib.embedding(self._chat, txt) == 0, 'embedding failed'
        return json.loads(f"[{self._result_embedding}]")

    def qa_rank(self, q: str, a: str) -> float:
        self._result_ranking = '-1.0'
        assert self._lib.qa_rank(self._chat, q, a) == 0, 'qa_rank failed'
        return float(self._result_ranking)

    def select_vector_store(self, name: str):
        assert self._lib.rag_select_store(self._chat, name) == 0

    def abort(self) -> None:
        self._lib.abort(self._chat)

    def restart(self, sys_prompt: str | None = None) -> None:
        self._lib.restart(self._chat, sys_prompt)

    def set_max_gen_tokens(self, max_gen: int) -> None:
        self._lib.set_max_gen_tokens(self._chat, max_gen)

    def show_statistics(self) -> None:
        self._lib.show_statistics(self._chat)

    def save_session(self, file_name: str) -> str:
        return self._lib.save_session(self._chat, file_name)

    def load_session(self, file_name: str) -> str:
        return self._lib.load_session(self._chat, file_name)

    def destroy(self) -> int:
        if hasattr(self, "_chat") and self._chat:
            if self.is_generating: self.abort()
            obj_id = LibChatLLM._obj2id.get(self)
            if obj_id is not None:
                LibChatLLM._obj2id.pop(self, None)
                LibChatLLM._id2obj.pop(obj_id, None)
            self._lib.destroy(self._chat)
            self._chat = None
        return 0

    def callback_print_reference(self, s: str) -> None:
        self.references.append(s)

    def callback_print_log(self, s: str) -> None:
        print(s)

    def callback_print_beam_search(self, s: str) -> None:
        l = s.split(',', maxsplit=1)
        self.beam_search_results.append({'str': l[1], 'score': float(l[0])})

    def callback_print_rewritten_query(self, s: str) -> None:
        self.rewritten_query = s

    def callback_print_meta(self, s: str) -> None:
        if self.out_queue is None:
            print(s)
        else:
            self.out_queue.put(LLMChatMeta(self.input_id, s))

    def callback_print(self, s: str) -> None:
        if self.out_queue is None:
            print(s, end="", flush=True)
        else:
            self.out_queue.put(LLMChatChunk(self.input_id, s))

    def callback_print_thought(self, s: str) -> None:
        if self.out_queue is None:
            if self.is_first_thought_chunk:
                self.is_first_thought_chunk = False
                print('====== Thinking ======', flush=True)
            print(s, end="", flush=True)
        else:
            self.out_queue.put(LLMChatThoughtChunk(self.input_id, s))

    def callback_print_history_user(self, s: str) -> None:
        pass

    def callback_print_history_ai(self, s: str) -> None:
        pass

    def callback_print_embedding(self, s: str) -> None:
        self._result_embedding = s

    def callback_print_ranking(self, s: str) -> None:
        self._result_ranking = s

    def callback_text_tokenize(self, s: str) -> None:
        self._result_text_tokenize = s

    def call_tool(self, s: str) -> None:
        raise Exception(f'Tool calling not implemented! {s}')

    def callback_end(self) -> None:
        if self.out_queue is None:
            print('')
        else:
            self.out_queue.put(LLMChatDone(self.input_id))
        self.input_id = self.tool_input_id
        self.tool_input_id = None

    def callback_thought_done(self) -> None:
        if self.out_queue is None:
            print('\n===================', flush=True)

    def callback_async_done(self) -> None:
        self.is_generating = False

class LLMChatInput:
    def __init__(self, input: str, id: Any) -> None:
        self.input = input
        self.id = id

class ChatLLMStreamer:
    def __init__(self, llm: ChatLLM) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        llm.out_queue = self.output_queue
        self.llm = llm
        self.run = True
        self.input_counter = 0
        self.thread = threading.Thread(target=lambda: self.thread_fun())
        self.thread.start()
        self.llm.start()
        self.acc = ''
        self.thought_acc = ''
        self.auto_restart = False

    def thread_fun(self) -> None:
        while self.run:
            input: LLMChatInput = self.input_queue.get()
            if self.auto_restart:
                self.llm.restart()
            self.llm.chat(input.input, input.id)

    def flush_output(self) -> str:
        r = ''
        while not self.output_queue.empty():
            t = self.output_queue.get_nowait()
            if isinstance(t, str):
                r = r + t
        return r

    def chat(self, user_input: str) -> Iterable[str | tuple[str, str]]:
        id = self.input_counter
        self.input_counter = self.input_counter + 1
        self.input_queue.put(LLMChatInput(user_input, id))
        self.acc = ''
        self.thought_acc = ''
        while True:
            output = self.output_queue.get()
            if isinstance(output, LLMChatChunk):
                if output.id == id:
                    self.acc = self.acc + output.chunk
                    yield output.chunk
            elif isinstance(output, LLMChatThoughtChunk):
                if output.id == id:
                    self.thought_acc = self.thought_acc + output.chunk
                    yield ('thought', output.chunk)
            elif isinstance(output, LLMChatDone):
                if output.id == id:
                    break
            elif isinstance(output, LLMChatMeta):
                pass
            else:
                print(output)
                raise Exception(output)

    def abort(self) -> None:
        self.llm.abort()

    def restart(self) -> None:
        self.llm.restart()

    def set_max_gen_tokens(self, max_gen: int) -> None:
        self.llm.set_max_gen_tokens(max_gen)

    def show_statistics(self) -> None:
        self.llm.show_statistics()

    def get_acc_resp(self) -> str:
        return self.acc

    def terminate(self) -> None:
        self.run = False

llm: ChatLLM = None

def handler(signal_received, frame):
    if llm.is_generating:
        print('\naborting...')
        llm.abort()
    else:
        llm.show_statistics()
        sys.exit(0)

def demo_streamer():
    global llm
    signal.signal(signal.SIGINT, handler)
    llm = ChatLLM(LibChatLLM(), sys.argv[1:])

    streamer = ChatLLMStreamer(llm)

    while True:
        s = input('You  > ')
        print('A.I. > ', end='', flush=True)
        for s in streamer.chat(s):
            print(s, end='', flush=True)

def demo_embedding(params, cls = ChatLLM, lib_path: str =''):
    global llm
    signal.signal(signal.SIGINT, handler)
    llm = cls(LibChatLLM(lib_path), params)

    while True:
        s = input('Input    > ')
        print('Embedding > ', flush=True)
        print(llm.embedding(s))

def demo_ranking(params, cls = ChatLLM, lib_path: str =''):
    global llm
    signal.signal(signal.SIGINT, handler)
    llm = cls(LibChatLLM(lib_path), params)

    while True:
        q = input('Question > ')
        a = input('Answer   > ')
        print(f'Score    > {llm.qa_rank(q, a)}', flush=True)

def demo_simple(params, cls = ChatLLM, lib_path: str =''):
    global llm
    signal.signal(signal.SIGINT, handler)
    llm = cls(LibChatLLM(lib_path), params)

    while True:
        s = input('You  > ')
        print('A.I. > ', end='', flush=True)
        llm.chat(s)
        if len(llm.references) > 0:
            print(llm.references)

if __name__ == '__main__':
    demo_simple(sys.argv[1:])