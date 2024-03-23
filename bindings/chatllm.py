from ctypes import *
import os, sys, signal
from typing import Any, List, Union

class LibChatLLM:

    _obj2id = {}
    _id2obj = {}

    def __init__(self, lib: str = '') -> None:

        if lib == '':
            this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            lib = os.path.join(this_dir, 'libchatllm.') + ('dll' if sys.platform == 'win32' else 'so')

        if sys.platform == 'win32':
            self._lib = windll.LoadLibrary(lib)
            self._PRINTFUNC = WINFUNCTYPE(None, c_void_p, c_char_p)
            self._ENDFUNC = WINFUNCTYPE(None, c_void_p)
        else:
            self._lib = cdll.LoadLibrary(lib)
            self._PRINTFUNC = CFUNCTYPE(None, c_void_p, c_char_p)
            self._ENDFUNC = CFUNCTYPE(None, c_void_p)

        self._chatllm_create = self._lib.chatllm_create
        self._chatllm_append_param = self._lib.chatllm_append_param
        self._chatllm_start = self._lib.chatllm_start
        self._chatllm_user_input = self._lib.chatllm_user_input
        self._chatllm_abort_generation = self._lib.chatllm_abort_generation

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

        self._cb_print = self._PRINTFUNC(LibChatLLM.callback_print)
        self._cb_end = self._ENDFUNC(LibChatLLM.callback_end)

    @staticmethod
    def callback_print(user_data: int, s: bytes) -> None:
        obj = LibChatLLM._id2obj[user_data]
        obj.callback_print(s)

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

class ChatLLM:
    def __init__(self, lib: LibChatLLM) -> None:
        self._lib = lib
        self._chat = lib._chatllm_create()
        self.is_generating = False

    def append_param(self, param: Union[str, List[str]]) -> None:
        self._lib.append_param(self._chat, param)

    def start(self) -> None:
        r = self._lib.start(self._chat, self)
        if r != 0:
            raise Exception(f'ChatLLM: failed to `start()` with error code {r}')

    def chat(self, user_input: str) -> None:
        self.is_generating = True
        r = self._lib.chat(self._chat, user_input)
        self.is_generating = False
        if r != 0:
            raise Exception(f'ChatLLM: failed to `chat()` with error code {r}')

    def abort(self) -> None:
        self._lib.abort(self._chat)

    def callback_print(self, s: bytes) -> None:
        print(s.decode(), end="", flush=True)

    def callback_end(self) -> None:
        pass

llm: ChatLLM = None

def handler(signal_received, frame):
    if llm.is_generating:
        print('\naborting...')
        llm.abort()
    else:
        sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    llm = ChatLLM(LibChatLLM())
    llm.append_param(sys.argv[1:])
    llm.start()

    while True:
        s = input('You  > ')
        print('A.I. > ', end='', flush=True)
        llm.chat(s)