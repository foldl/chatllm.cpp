import copy
import json, sys, re

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

def get_tools() -> dict:

    def convert(tool: dict):
        r = copy.deepcopy(tool)

        r['params'] = r['parameters']
        del r['parameters']
        return r

    return [convert(t) for t in tool_definition._TOOL_DESCRIPTIONS]

def build_sys_prompt():
    return "Answer the following questions as best as you can. You have access to the following tools: \n\n" + \
            json.dumps(get_tools(), indent=4, ensure_ascii=False)

import chatllm
from chatllm import ChatLLM

def call_function(s) -> str:

    def extract_code(text: str) -> str:
        pattern = r'```([^\n]*)\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1][1]

    def tool_call(*args, **kwargs) -> dict:
        return kwargs

    try:
        tool_name, *call_args_text = s.strip().split('\n')
        code = extract_code('\n'.join(call_args_text))
        args = eval(code, {'tool_call': tool_call}, {})
        observation = dispatch_tool(tool_name, args)
        return observation.text
    except Exception as e:
        print(f"error occurs: {e}")
        return "failed to call the function"

class ToolChatLLM(ChatLLM):
    def call_tool(self, s: str) -> None:
        print(f"[Use Tool]: {s.strip().split()[0]}")
        rsp = call_function(s)
        self.tool_input(rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_sys_prompt()], ToolChatLLM, lib_path=PATH_BINDS)
