import json
import re, sys
from datetime import datetime

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

# https://huggingface.co/watt-ai/watt-tool-8B

FUNCTION_CALL_START = "["
FUNCTION_CALL_CLOSE = "]"

def convert_tool_def(func: dict) -> dict:
    params = {}
    required = set()
    for p in func['parameters']:
        params[p['name']] = {
            "type": p['type'],
            "description": p['description'],
        }
        if p['required']: required.add(p['name'])

    desc = {
        "name": func['name'],
        "description": func['description'],
        "arguments": {
            "type": "dict",
            "properties": params,
            "required": list(required)
        }
    }
    return desc

SYS_PROMPT_TEMPLATE = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""

def build_system_prompt(functions: list[dict]):
    s = SYS_PROMPT_TEMPLATE.format(functions=[convert_tool_def(f) for f in functions])
    return s

import chatllm
from chatllm import ChatLLM

def call_internal_tool(s: str) -> str:
    print(f"[Use Builtin Tool]{s}")
    return "Error: not implemented"

def call_functions(s: str) -> str:
    try:
        r = []
        for tool_name, code in parse_function_calls(s):
            print(f'[Use Tool] {tool_name}')
            observation = dispatch_tool(tool_name, code)
            r.append(observation.text)
        return '\n\n'.join(r)
    except Exception as e:
        print(f"error occurs: {e}")
        return "failed to call the function"


def parse_function_calls(s: str) -> list[tuple[str, dict]] | None:
    try:
        matches = re.findall(r'(\w+)\((.*?)\)', s)
        if matches is None: return None

        def parse_args(s: str) -> str:
            r = []
            for pair in s.split(', '):
                k, v = pair.split('=')
                r.append(f'"{k}": {v}')
            return f"{{{','.join(r)}}}"

        return [(match[0], json.loads(parse_args(match[1]))) for match in matches]
    except:
        return None

class ToolChatLLM(ChatLLM):

    chunk_acc = ''

    def callback_print(self, s: str) -> None:

        if self.chunk_acc == '':
            if FUNCTION_CALL_START.startswith(s):
                self.chunk_acc = s
            else:
                super().callback_print(s)

            return

        self.chunk_acc = (self.chunk_acc + s).strip()

        if len(self.chunk_acc) < len(FUNCTION_CALL_START):
            return

        if not self.chunk_acc.startswith(FUNCTION_CALL_START):
            super().callback_print(self.chunk_acc)
            self.chunk_acc = ''
            return

    def callback_end(self) -> None:
        s = self.chunk_acc
        self.chunk_acc = ''
        super().callback_end()

        s = s.strip()
        if len(s) < 1: return

        if parse_function_calls(s) is not None:
            rsp = call_functions(s)
            self.tool_input(rsp)
        else:
            super().callback_print(s)

    def call_tool(self, s: str) -> None:
        rsp = call_internal_tool(s.strip())
        self.tool_input(rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(tool_definition._TOOL_DESCRIPTIONS)], ToolChatLLM, lib_path=PATH_BINDS)