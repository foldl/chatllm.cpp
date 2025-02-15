import sys, signal, time

from binding import PATH_BINDS

from chatllm import ChatLLM, LibChatLLM
try:
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.status import Status
    from rich.console import Console
    from rich.text import Text
    from rich import print
except:
    raise Exception('package `rich` is missing. install it: `pip install rich`')

TAG_THINK_START = '<think>'
TAG_THINK_END = '</think>'

class RichChatLLM(ChatLLM):
    chunk_acc = ''
    thoughts_acc = ''
    acc = ''
    meta = []
    is_thinking = False
    think_time = 0
    detecting = False

    def async_chat(self, user_input: str, input_id = None) -> None:
        self.chunk_acc = ''
        self.is_thinking = True
        self.thoughts_acc = ''
        self.acc = ''
        self.detecting = True
        self.think_time = 0
        super().async_chat(user_input, input_id)

    def callback_print_meta(self, s: str) -> None:
        self.meta.append(s)

    def show_meta(self, title: str) -> None:
        print('')
        print(Panel('\n'.join(self.meta), title=title))
        self.meta.clear()

    def render_ai(self) -> Panel:
        return Panel(Markdown(llm.chunk_acc), title='A.I.')

    def render_thoughts(self) -> str:
        thoughts = self.thoughts_acc.split('\n')
        s = thoughts[-1] if len(thoughts) > 0 else ''
        return s

    def callback_print(self, s: str) -> None:
        if self.detecting:
            self.acc = (self.acc + s).lstrip(" \n\r")
            if len(self.acc) < len(TAG_THINK_START):
                return
            self.detecting = False
            if self.acc.startswith(TAG_THINK_START):
                self.thoughts_acc = self.acc[len(TAG_THINK_START):]
                self.think_time = time.perf_counter()
            else:
                self.is_thinking = False
                self.chunk_acc = self.acc
            return

        if self.is_thinking:
            self.thoughts_acc = self.thoughts_acc + s
            pos = self.thoughts_acc.find(TAG_THINK_END)
            if pos > 0:
                self.is_thinking = False
                self.think_time = time.perf_counter() - self.think_time
                self.chunk_acc = self.thoughts_acc[pos + len(TAG_THINK_END)]
            return

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
    render_thoughts = lambda: llm.render_thoughts()

    console = Console()

    while True:
        s = input('You  > ')
        if s == '': continue

        llm.async_chat(s)
        time.sleep(0.1)

        with Status("thinking...", spinner="bouncingBall", console=console) as status:
            while llm.is_thinking:
                status.update('[bright_black]' + render_thoughts())
                time.sleep(0.2)
        if llm.think_time > 0:
            text = Text()
            text.append(f"<<Thought for {llm.think_time:.2f}s>>", style="bright_black")
            console.print(text)

        with Live(render_ai(), refresh_per_second=4) as live:
            while llm.is_generating:
                time.sleep(0.2)
                live.update(render_ai())

if __name__ == '__main__':
    demo_simple(sys.argv[1:], lib_path=PATH_BINDS)