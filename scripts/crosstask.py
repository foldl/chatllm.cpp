# Let two LLMs talk to each other!

import sys, os, signal

this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)

from chatllm import ChatLLM, LibChatLLM

class ForwardChatLLM(ChatLLM):
    chunk_acc = ''

    def callback_print(self, s: str) -> None:
        self.chunk_acc = self.chunk_acc + s
        super().callback_print(s)

    def get_output_acc(self) -> str:
        s = self.chunk_acc
        self.chunk_acc = ''
        return s

model_a: ForwardChatLLM = None
model_b: ForwardChatLLM = None
aborted = False

def handler(signal_received, frame):
    global aborted
    aborted = True
    if model_a.is_generating:
        model_a.abort()
        return
    if model_b.is_generating:
        model_b.abort()
        return

    if model_a.is_generating or model_b.is_generating:
        return

    model_a.show_statistics()
    model_b.show_statistics()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    args = sys.argv[1:]
    if len(args) < 3:
        print(f"usage: python crosstalk.py path/to/model/A path/to/model/B initial_input")
        exit(-1)

    input = args[2]
    model_a = ForwardChatLLM(LibChatLLM(PATH_BINDS), ['-m', args[0], '-i', '-p', input] + args[3:], True)
    model_b = ForwardChatLLM(LibChatLLM(PATH_BINDS), ['-m', args[1], '-i'], True)

    print('A > ' + input)
    while not aborted:
        print('B > ', end='')
        model_b.chat(input)
        input = model_b.get_output_acc()

        if aborted: break

        print('A > ', end='')
        model_a.chat(input)
        input = model_a.get_output_acc()