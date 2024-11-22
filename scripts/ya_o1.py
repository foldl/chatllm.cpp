import sys, signal, os

this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)
from chatllm import LibChatLLM, ChatLLM

LIB = LibChatLLM(PATH_BINDS)

def print_green(skk): print("\033[92m{}\033[00m".format(skk))
def print_yellow(skk): print("\033[93m{}\033[00m".format(skk))

class CallableLLM(ChatLLM):

    echo = True

    def chat(self, user_input: str) -> str:
        self.chunk_acc = ''
        super().chat(user_input)
        return self.chunk_acc

    def callback_print(self, s: str) -> None:
        if self.echo:
            print(s, end="", flush=True)
        self.chunk_acc = self.chunk_acc + s

    def callback_end(self) -> None:
        if self.echo:
            print('')

class ThoughtLLM2:
    thought_procedures = [
        ["", "Let's break it down and think step by step.\n"],
        ["think again", "This question looks complex. Let me think again.\n"],
        ["check your answer", "Let me check if my answer is correct or not.\n"]
    ]

    def __init__(self, llm: CallableLLM) -> None:
        self.llm = llm

    def think(self, prompt: str) -> str:
        self.llm.restart()
        self.thought_procedures[0][0] = prompt
        r = []
        for i in range(len(self.thought_procedures)):
            print_green(f"think #{i + 1}")
            step = self.thought_procedures[i]
            self.llm.set_ai_prefix(step[1])
            r.append(self.llm.chat(step[0]))

        return '\n'.join(r)

class ThoughtLLM1:
    thought_procedures = [
        "Let's break it down and think step by step.\n",
        "\nThis question looks complex. Let me think again.\n",
        "\nLet me check if my answer is correct or not.\n"
    ]

    def __init__(self, llm: CallableLLM) -> None:
        self.llm = llm

    def think(self, prompt: str) -> str:
        self.llm.restart()
        self.llm.restart()
        self.llm.set_ai_prefix(self.thought_procedures[0])
        print_green(f"think #1")
        self.llm.chat(prompt)
        for i in range(1, len(self.thought_procedures)):
            print_green(f"think #{i + 1}")
            self.llm.ai_continue(self.thought_procedures[i])

        return self.llm.chunk_acc

def run(think_model: CallableLLM, chat_model: ChatLLM):

    def handler(signal_received, frame):
        nonlocal think_model, chat_model
        think_model.abort()
        chat_model.abort()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    PROMPT = """Answer this question by summarizing the given analysis and thoughts:

# Question:

{prompt}

# Analysis and thought:

{references}
"""

    thoughts_gen = ThoughtLLM1(think_model)

    while True:
        s = input('You  > ')
        thoughts = thoughts_gen.think(s)
        print_yellow('-------')
        #print(thoughts)
        #print_yellow('-------')
        chat_model.chat(PROMPT.format(prompt=s, references=thoughts))

if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) < 2:
        print(f"usage: python ya_o1.py path/to/think/model /path/to/chat/model ")
        exit(-1)

    SYSTEM_PROMPT = """You are an AI assistant that answers questions by checking and summarizing the thought given by user.
ALWAYS use the thought given by user. DO NOT answer it yourself."""

    run(CallableLLM(LIB, ['-m', args[0], '--temp', '0.7']),
        ChatLLM(LIB, ['-m', args[1], '--temp', '0', '--system', SYSTEM_PROMPT]))