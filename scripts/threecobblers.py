import sys, signal

from binding import PATH_BINDS

import chatllm
from chatllm import ChatLLM
import tool_numinamath


def build_sys_prompt():
    return ""

LIB = chatllm.LibChatLLM(PATH_BINDS)

class Expert:
    def __init__(self, domain: str, id: str, name: str, cls = ChatLLM) -> None:
        self.domain = domain
        self.id = id
        self.name = name
        self.cls = cls

    def load(self, additional_params) -> None:
        self.model = self.cls(LIB, ['-m', self.id] + additional_params)

class CallableLLM(ChatLLM):

    def chat(self, user_input: str, input_id = None) -> str:
        self.chunk_acc = ''
        super().chat(user_input, input_id)
        return self.chunk_acc

    def callback_print(self, s: str) -> None:
        self.chunk_acc = self.chunk_acc + s

def classify(model: CallableLLM, prompt: str, classes: list[str]) -> int:
    class_str = '\n'.join([f"{i + 1}. {c}" for i, c in enumerate(classes)])
    input = f"Please classify the question to one of these classes:\n{class_str}\n\nQuestion: {prompt}"
    model.restart()
    selected = model.chat(input)
    for i, c in enumerate(classes):
        if c in selected: return i
    return len(classes) - 1

def run(additional_params, MODEL_CLASSIFIER: str = ':llama3.1'):
    EXPERTS = [
        Expert('Programming', ':deepseek-coder-v2:light', 'DeepSeek-Coder'),
        Expert('Maths', ':numinamath', 'NuminaMath', tool_numinamath.ToolChatLLM),
        Expert('Others', ':yi-1.5:6b', 'Yi 1.5'),
    ]

    classifier = CallableLLM(LIB, ['-m', MODEL_CLASSIFIER, '--temp', '0'] + additional_params)

    for e in EXPERTS:
        e.load(additional_params)

    def handler(signal_received, frame):
        classifier.abort()
        for e in EXPERTS:
            e.model.abort()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    classes = [e.domain for i, e in enumerate(EXPERTS)]

    selected = None
    print('Tip: use `/restart` to restart.')

    while True:
        s = input('You  > ')
        if s.startswith('/restart'):
            selected = None
            continue

        if selected is None:
            selected = EXPERTS[classify(classifier, s, classes=classes)]
            selected.model.restart()

        print(f'{selected.name} > ', end='', flush=True)
        selected.model.chat(s)

if __name__ == '__main__':
    run(sys.argv[1:])
