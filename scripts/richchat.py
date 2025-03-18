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

# sorted by length!
THINK_TAGS = [
    ('<think>',         '</think>'),
    ('<thought>',       '</thought>'),
    ('<reasoning>',     '</reasoning>'),
]

TAG_THINK_START = ''
TAG_THINK_END = ''

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
        self.think_start = 0
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
        global TAG_THINK_START, TAG_THINK_END
        if self.detecting:
            self.acc = (self.acc + s).lstrip(" \n\r")
            padding = False
            for tags in THINK_TAGS:
                if len(self.acc) < len(tags[0]):
                    padding = True
                    break
                if self.acc.startswith(tags[0]):
                    TAG_THINK_START = tags[0]
                    TAG_THINK_END = tags[1]

                    self.thoughts_acc = self.acc[len(TAG_THINK_START):]
                    self.think_start = time.perf_counter()
                    self.think_time = 0
                    self.detecting = False

                    break

            if (not padding) and self.detecting:
                self.is_thinking = False
                self.detecting = False
                self.chunk_acc = self.acc
            return

        if self.is_thinking:
            self.thoughts_acc = self.thoughts_acc + s
            pos = self.thoughts_acc.find(TAG_THINK_END)
            self.think_time = time.perf_counter() - self.think_start
            if pos > 0:
                self.is_thinking = False
                self.chunk_acc = self.thoughts_acc[pos + len(TAG_THINK_END):]
            return

        self.chunk_acc = self.chunk_acc + s

    def callback_async_done(self) -> None:
        super().callback_async_done()
        if self.is_thinking:
            self.is_thinking = False
            self.think_time = time.perf_counter() - self.think_start
            self.async_ai_continue(TAG_THINK_END)

llm: RichChatLLM = None
MAX_THOUGHT_TIME = 60 * 3

def params_preprocess(params: list[str]) -> list[str]:
    for i, s in enumerate(params):
        if (s == '--max-thought-time') and (i + 1 < len(params)):
            global MAX_THOUGHT_TIME
            MAX_THOUGHT_TIME = float(params[i + 1])
            return params[:i] + params[i + 2:]
    return params

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