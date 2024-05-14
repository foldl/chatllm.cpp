from ctypes import *
from enum import IntEnum
import os, sys, signal, queue
import threading
from typing import Any, Iterable, List, Union


class PrintType(IntEnum):
    PRINT_CHAT_CHUNK        = 0,
    PRINTLN_META            = 1,    # print a whole line: general information
    PRINTLN_ERROR           = 2,    # print a whole line: error message
    PRINTLN_REF             = 3,    # print a whole line: reference
    PRINTLN_REWRITTEN_QUERY = 4,    # print a whole line: rewritten query

class LibChatLLM:

    _obj2id = {}
    _id2obj = {}

    def __init__(self, lib: str = '') -> None:

        if lib == '':
            this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            lib = os.path.join(this_dir, 'libchatllm.')
            if sys.platform == 'win32':
                lib = lib + 'dll'
            elif sys.platform == 'darwin':
                lib = lib + 'dylib'
            else:
                lib = lib + 'so'

        if sys.platform == 'win32':
            self._lib = windll.LoadLibrary(lib)
            self._PRINTFUNC = WINFUNCTYPE(None, c_void_p, c_int, c_char_p)
            self._ENDFUNC = WINFUNCTYPE(None, c_void_p)
        else:
            self._lib = cdll.LoadLibrary(lib)
            self._PRINTFUNC = CFUNCTYPE(None, c_void_p, c_int, c_char_p)
            self._ENDFUNC = CFUNCTYPE(None, c_void_p)

        self._chatllm_create = self._lib.chatllm_create
        self._chatllm_append_param = self._lib.chatllm_append_param
        self._chatllm_start = self._lib.chatllm_start
        self._chatllm_user_input = self._lib.chatllm_user_input
        self._chatllm_abort_generation = self._lib.chatllm_abort_generation
        self._chatllm_restart = self._lib.chatllm_restart

        self._chatllm_create.restype = c_void_p
        self._chatllm_create.argtypes = []

        self._chatllm_append_param.restype = None
        self._chatllm_append_param.argtypes = [c_void_p, c_char_p]

        self._chatllm_start.restype = c_int
        self._chatllm_start.argtypes = [c_void_p, self._PRINTFUNC, self._ENDFUNC, c_void_p]

        self._chatllm_user_input.restype = c_int
        self._chatllm_user_input.argtypes = [c_void_p, c_char_p]

        self._chatllm_abort_generation.restype = None
        self._chatllm_abort_generation.argtypes = [c_void_p]

        self._chatllm_restart.restype = None
        self._chatllm_restart.argtypes = [c_void_p]

        self._cb_print = self._PRINTFUNC(LibChatLLM.callback_print)
        self._cb_end = self._ENDFUNC(LibChatLLM.callback_end)

    @staticmethod
    def callback_print(user_data: int, print_type: c_int, s: bytes) -> None:
        obj = LibChatLLM._id2obj[user_data]
        txt = s.decode()
        if print_type == PrintType.PRINT_CHAT_CHUNK.value:
            obj.callback_print(txt)
        elif print_type == PrintType.PRINTLN_META.value:
            obj.callback_print(txt + '\n')
        elif print_type == PrintType.PRINTLN_REF.value:
            obj.callback_print_reference(txt)
        elif print_type == PrintType.PRINTLN_REWRITTEN_QUERY.value:
            obj.callback_print_rewritten_query(txt)
        elif print_type == PrintType.PRINTLN_ERROR.value:
            raise Exception(txt)
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
        for s in param:
            self._chatllm_append_param(obj, c_char_p(s.encode()))

    def start(self, obj: c_void_p, callback_obj: Any) -> int:
        id = self.alloc_id_for_obj(callback_obj)
        return self._chatllm_start(obj, self._cb_print, self._cb_end, c_void_p(id))

    def chat(self, obj: c_void_p, user_input: str) -> int:
        return self._chatllm_user_input(obj, c_char_p(user_input.encode()))

    def abort(self, obj: c_void_p) -> None:
        self._chatllm_abort_generation(obj)

    def restart(self, obj: c_void_p) -> None:
        self._chatllm_restart(obj)

class LLMChatDone:
    def __init__(self, id: Any) -> None:
        self.id = id

class LLMChatChunk:
    def __init__(self, id: Any, chunk: str) -> None:
        self.id = id
        self.chunk = chunk

class ChatLLM:
    def __init__(self, lib: LibChatLLM, param: Union[None, str, List[str]], auto_start: bool = True) -> None:
        self._lib = lib
        self._chat = lib._chatllm_create()
        self.is_generating = False
        self.out_queue = None
        self.input_id = None
        self.references = []
        self.rewritten_query = ''
        if param is not None:
            self.append_param(param)
            if auto_start:
                self.start()

    def append_param(self, param: Union[str, List[str]]) -> None:
        self._lib.append_param(self._chat, param)

    def start(self) -> None:
        r = self._lib.start(self._chat, self)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `start()` with error code {r}')

    def chat(self, user_input: str, input_id = None) -> None:
        self.is_generating = True
        self.input_id = input_id
        self.references = []
        self.rewritten_query = ''
        r = self._lib.chat(self._chat, user_input)
        self.is_generating = False
        if r != 0:
            raise Exception(f'ChatLLM: failed to `chat()` with error code {r}')

    def abort(self) -> None:
        self._lib.abort(self._chat)

    def restart(self) -> None:
        self._lib.restart(self._chat)

    def callback_print_reference(self, s: str) -> None:
        self.references.append(s)

    def callback_print_rewritten_query(self, s: str) -> None:
        self.rewritten_query = s

    def callback_print(self, s: str) -> None:
        if self.out_queue is None:
            print(s, end="", flush=True)
        else:
            self.out_queue.put(LLMChatChunk(self.input_id, s))

    def callback_end(self) -> None:
        if self.out_queue is not None:
            self.out_queue.put(LLMChatDone(self.input_id))

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

    def chat(self, user_input: str) -> Iterable[str]:
        id = self.input_counter
        self.input_counter = self.input_counter + 1
        self.input_queue.put(LLMChatInput(user_input, id))
        self.acc = ''
        while True:
            output = self.output_queue.get()
            if isinstance(output, LLMChatChunk):
                if output.id == id:
                    self.acc = self.acc + output.chunk
                    yield output.chunk
            elif isinstance(output, LLMChatDone):
                if output.id == id:
                    break
            else:
                print(output)
                raise Exception(output)

    def abort(self) -> None:
        self.llm.abort()

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
        sys.exit(0)

def demo_streamer():
    global llm
    llm = ChatLLM(LibChatLLM(), sys.argv[1:])

    streamer = ChatLLMStreamer(llm)

    while True:
        s = input('You  > ')
        print('A.I. > ', end='', flush=True)
        for s in streamer.chat(s):
            print(s, end='', flush=True)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    llm = ChatLLM(LibChatLLM(), sys.argv[1:])

    while True:
        s = input('You  > ')
        print('A.I. > ', end='', flush=True)
        llm.chat(s)
        if len(llm.references) > 0:
            print(llm.references)