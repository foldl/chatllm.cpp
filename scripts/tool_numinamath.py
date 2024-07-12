import sys, re, io

from binding import PATH_BINDS

import chatllm
from chatllm import ChatLLM, LLMChatChunk

def exec_code(python_code: str) -> str:
    output = io.StringIO()

    def p(*args, **kwargs):
        print(*args, file=output, **kwargs)

    try:
        exec(python_code, {'print': p})
    except:
        pass

    return output.getvalue()

class ToolChatLLM(ChatLLM):
    chunk_acc = ''
    disp_acc = ''

    OUTPUT_TAG = '```output'

    def callback_print(self, s: str) -> None:
        self.chunk_acc = self.chunk_acc + s
        self.disp_acc = self.disp_acc + s
        if not self.OUTPUT_TAG.startswith(self.disp_acc):
            super().callback_print(self.disp_acc)
            self.disp_acc = ''

        if self.chunk_acc.endswith(self.OUTPUT_TAG):
            self.abort()

    def callback_end(self) -> None:
        all_output = self.chunk_acc
        self.disp_acc = ''
        self.chunk_acc = ''
        super().callback_end()
        if not all_output.endswith(self.OUTPUT_TAG): return

        python_code = re.findall(r"```python(.*?)```", all_output, re.DOTALL)
        if len(python_code) < 1: return
        python_code = python_code[0]
        if len(python_code) < 1: return

        print('[EXEC CODE] ', end='')
        result = exec_code(python_code).strip(' \r\n')
        print(result)

        self.tool_completion(result)

if __name__ == '__main__':
    print("💣💣 DANGEROUS! NO SAND-BOXING. DEMO ONLY. 💣💣")
    chatllm.demo_simple(sys.argv[1:], ToolChatLLM, lib_path=PATH_BINDS)