# Let two LLMs talk to each other!

from chatllm import ChatLLM, LibChatLLM
import signal, sys

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
    if len(args) != 3:
        print(f"usage: python crosstalk.py path/to/model/A path/to/model/B initial_input")
        exit(-1)

    model_a = ForwardChatLLM(LibChatLLM(), ['-m', args[0], '-i'], True)
    model_b = ForwardChatLLM(LibChatLLM(), ['-m', args[1], '-i'], True)

    input = args[2]
    print('A > ' + input)
    while not aborted:
        print('B > ', end=None)
        model_b.chat(input)
        input = model_b.get_output_acc()

        if aborted: break

        print('A > ', end=None)
        model_a.chat(input)
        input = model_a.get_output_acc()