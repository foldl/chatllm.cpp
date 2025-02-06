import sys, signal, os

this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)
from chatllm import LibChatLLM, ChatLLM

class ThoughtLLM(ChatLLM):
    thought_procedures = []
    final_answer = 'Final answer:'

    def init_thought_procedures(self) -> None:
        self.chunk_acc = ''
        self.final_answer = 'Final answer:'
        self.thought_procedures = [
            "Okay, let me think step by step. The question is",
            "Hmm,",
            "But wait,",
            "Wait,",
        ]

    def chat(self, user_input: str) -> str:
        self.init_thought_procedures()
        self.set_ai_prefix(self.thought_procedures.pop(0))
        super().chat(user_input)
        while True:
            if len(self.thought_procedures) > 0:
                self.ai_continue(self.thought_procedures.pop(0))
            elif len(self.final_answer) > 0:
                self.ai_continue(self.final_answer)
                self.final_answer = ''
            else:
                break

    def callback_print(self, s: str) -> None:
        print(s, end="", flush=True)
        self.chunk_acc = self.chunk_acc + s

if __name__ == '__main__':
    import chatllm
    chatllm.demo_simple(sys.argv[1:], cls=ThoughtLLM, lib_path=PATH_BINDS)