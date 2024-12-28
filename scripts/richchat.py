import sys, signal, time

from binding import PATH_BINDS

from chatllm import ChatLLM, LibChatLLM
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich import print

class RichChatLLM(ChatLLM):
    chunk_acc = ''

    def async_chat(self, user_input: str, input_id = None) -> None:
        self.chunk_acc = ''
        super().async_chat(user_input, input_id)

    def callback_print_meta(self, s: str) -> None:
        print(Panel(s, title='Information'))

    def callback_print(self, s: str) -> None:
        self.chunk_acc = self.chunk_acc + s

llm: RichChatLLM = None

def render_ai_output():
    return Panel(Markdown(llm.chunk_acc), title='A.I.')

def handler(signal_received, frame):
    if llm.is_generating:
        print('\naborting...')
        llm.abort()
    else:
        llm.show_statistics()
        sys.exit(0)

def demo_simple(params, lib_path: str, cls = RichChatLLM):
    global llm
    signal.signal(signal.SIGINT, handler)
    llm = cls(LibChatLLM(lib_path), params)

    while True:
        s = input('You  > ')
        llm.async_chat(s)
        time.sleep(0.1)
        with Live(render_ai_output(), refresh_per_second=4) as live:
            while llm.is_generating:
                time.sleep(0.2)
                live.update(render_ai_output())

if __name__ == '__main__':
    demo_simple(sys.argv[1:], lib_path=PATH_BINDS)