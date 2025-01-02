import sys, signal, time

from binding import PATH_BINDS

from chatllm import ChatLLM, LibChatLLM
try:
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.live import Live
    from rich import print
except:
    raise Exception('package `rich` is missing. install it: `pip install rich`')

class RichChatLLM(ChatLLM):
    chunk_acc = ''
    meta = []

    def async_chat(self, user_input: str, input_id = None) -> None:
        self.chunk_acc = ''
        super().async_chat(user_input, input_id)

    def callback_print_meta(self, s: str) -> None:
        self.meta.append(s)

    def show_meta(self, title: str) -> None:
        print('')
        print(Panel('\n'.join(self.meta), title=title))
        self.meta.clear()

    def render_ai(self) -> Panel:
        return Panel(Markdown(llm.chunk_acc), title='A.I.')

    def callback_print(self, s: str) -> None:
        self.chunk_acc = self.chunk_acc + s

llm: RichChatLLM = None

def handler(signal_received, frame):
    if llm.is_generating:
        print('\naborting...')
        llm.abort()
    else:
        llm.show_statistics()
        llm.show_meta('Statistics')
        sys.exit(0)

def demo_simple(params, lib_path: str, cls = RichChatLLM):
    global llm
    signal.signal(signal.SIGINT, handler)
    llm = cls(LibChatLLM(lib_path), params)

    llm.show_meta('Model')

    render_ai = lambda: llm.render_ai()

    while True:
        s = input('You  > ')
        llm.async_chat(s)
        time.sleep(0.1)
        with Live(render_ai(), refresh_per_second=4) as live:
            while llm.is_generating:
                time.sleep(0.2)
                live.update(render_ai())

if __name__ == '__main__':
    demo_simple(sys.argv[1:], lib_path=PATH_BINDS)