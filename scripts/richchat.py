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

class RichChatLLM(ChatLLM):
    chunk_acc = ''
    thoughts_acc = ''
    meta = []
    is_thinking = False
    think_time = 0
    think_start = 0

    def async_chat(self, user_input: str, input_id = None) -> None:
        self.chunk_acc = ''
        self.is_thinking = True
        self.thoughts_acc = ''
        self.think_time = 0
        self.think_start = 0
        self.is_generating = True
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

    def callback_thought_done(self) -> None:
        self.think_time = time.perf_counter() - self.think_start
        self.is_thinking = False

    def callback_print_thought(self, s: str) -> None:
        if self.thoughts_acc == '':
            self.is_thinking = True
            self.think_start = time.perf_counter()
        self.thoughts_acc = self.thoughts_acc + s

    def callback_print(self, s: str) -> None:
        self.is_thinking = False
        if self.think_start > 0:
            self.think_time = time.perf_counter() - self.think_start
        self.chunk_acc = self.chunk_acc + s

    def callback_async_done(self) -> None:
        super().callback_async_done()
        if self.is_thinking:
            self.is_thinking = False
            self.think_time = time.perf_counter() - self.think_start

llm: RichChatLLM = None
MAX_THOUGHT_TIME = 60 * 3
multiple_lines_input = False

def params_preprocess(params: list[str]) -> list[str]:
    global multiple_lines_input
    multiple_lines_input = '--multi' in params
    for i, s in enumerate(params):
        if (s == '--max-thought-time') and (i + 1 < len(params)):
            global MAX_THOUGHT_TIME
            MAX_THOUGHT_TIME = float(params[i + 1])
            return params[:i] + params[i + 2:]
    params.append('+detect_thoughts')
    return params

def handler(signal_received, frame):
    if llm.is_generating:
        print('\naborting...')
        llm.abort()
    else:
        llm.show_statistics()
        llm.show_meta('Statistics')
        sys.exit(0)

def user_input(prompt: str) -> str:
    global multiple_lines_input
    if multiple_lines_input:
        print(prompt, end='', flush=True)
        return sys.stdin.read()
    else:
        return input(prompt)

def demo_simple(params, lib_path: str, cls = RichChatLLM):
    global llm
    global multiple_lines_input
    signal.signal(signal.SIGINT, handler)
    llm = cls(LibChatLLM(lib_path), params)

    llm.show_meta('Model')
    if multiple_lines_input:
        print('Press Ctrl+D / Ctrl+Z (Windows) to finish input')

    render_ai = lambda: llm.render_ai()
    render_thoughts = lambda: llm.render_thoughts()

    console = Console()

    while True:
        s = user_input('You  > ')
        if s == '': continue

        if s.startswith('/start'):
            llm.restart()
            continue

        llm.async_chat(s)
        time.sleep(0.1)

        with Status("thinking...", spinner="bouncingBall", console=console) as status:
            while llm.is_thinking and llm.is_generating:
                if llm.think_time > MAX_THOUGHT_TIME:
                    llm.abort()
                status.update('[bright_black]' + render_thoughts())
                time.sleep(0.2)

        if llm.think_time > 0:
            text = Text()
            text.append(f"« Thought for {llm.think_time:.2f}s »", style="bright_black")
            console.print(text)

        with Live(render_ai(), refresh_per_second=4) as live:
            while llm.is_generating:
                time.sleep(0.2)
                live.update(render_ai())

if __name__ == '__main__':
    params = params_preprocess(sys.argv[1:])
    demo_simple(params, lib_path=PATH_BINDS)