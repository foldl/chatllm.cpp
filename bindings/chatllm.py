from ctypes import *
import os, sys

def callback_print(s: bytes) -> None:
    print(s.decode(), end="", flush=True)

def callback_end() -> None:
    pass

PRINTFUNC = CFUNCTYPE(None, c_char_p)
ENDFUNC = CFUNCTYPE(None)

class LibChatLLM:

    def __init__(self, lib: str = '') -> None:
        if lib == '':
            this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            lib = os.path.join(this_dir, 'libchatllm.') + ('dll' if sys.platform == 'win32' else 'so')

        self._chatllm = windll.LoadLibrary(lib) if sys.platform == 'win32' else cdll.LoadLibrary(lib)

        self._chatllm_create = self._chatllm.chatllm_create
        self._chatllm_append_param = self._chatllm.chatllm_append_param
        self._chatllm_start = self._chatllm.chatllm_start
        self._chatllm_user_input = self._chatllm.chatllm_user_input

        self._chatllm_create.restype = c_void_p
        self._chatllm_create.argtypes = []

        self._chatllm_append_param.restype = None
        self._chatllm_append_param.argtypes = [c_void_p, c_char_p]

        self._chatllm_start.restype = c_int
        self._chatllm_start.argtypes = [c_void_p, PRINTFUNC, ENDFUNC]

        self._chatllm_user_input.restype = c_int
        self._chatllm_user_input.argtypes = [c_void_p, c_char_p]

        self._chat = self._chatllm_create()

        self._cb_print = PRINTFUNC(callback_print)
        self._cb_end = ENDFUNC(callback_end)

    def append_param(self, param: str) -> None:
        self._chatllm_append_param(self._chat, c_char_p(param.encode()))

    def start(self) -> int:
        return self._chatllm_start(self._chat, self._cb_print, self._cb_end)

    def chat(self, user_input: str) -> int:
        return self._chatllm_user_input(self._chat, c_char_p(user_input.encode()))

if __name__ == '__main__':

    lib = LibChatLLM()
    for arg in sys.argv[1:]:
        lib.append_param(arg)

    r = lib.start()
    if r != 0:
        raise Exception(f'start = {r}')

    while True:
        s = input('You  > ')
        print('A.I. > ', end='')
        r = lib.chat(s)
        if r != 0:
            raise Exception(f'chat = {r}')