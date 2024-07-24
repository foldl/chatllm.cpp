import json
import re, sys

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

# https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#user-defined-custom-tool-calling

FUNCTION_CALL_START = "<function="
FUNCTION_CALL_CLOSE = "</function>"

def gen_prompt_for_tool(func: dict) -> str:
    params = {}
    for p in func['parameters']:
        params[p['name']] = {
            "param_type": p['type'],
            "description": p['description'],
            "required": p['required']
        }
    desc = {
        "name": func['name'],
        "description": func['description'],
        "parameters": params
    }
    return f"Use the function '{func['name']}' to: {func['description']}" + '\n' + json.dumps(desc, ensure_ascii=False, indent=2)

SYS_PROMPT_TEMPLATE = """
You have access to the following functions:

{tools_def}

If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line"
- Always add your sources when using search results to answer the user query

You are a helpful Assistant."""

def build_system_prompt(functions: list[dict]):

    prompt = '\n\n'.join([gen_prompt_for_tool(f) for f in functions])

    return SYS_PROMPT_TEMPLATE.replace('{tools_def}', prompt)

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
        matches = re.findall(r'<function=([^>]*)>(.*)' + FUNCTION_CALL_CLOSE, s)
        if matches is None: return None

        return [(match[0], json.loads(match[1])) for match in matches]
    except:
        return None

class ToolChatLLM(ChatLLM):

    chunk_acc = ''

    def callback_print(self, s: str) -> None:

        if self.chunk_acc == '':
            if FUNCTION_CALL_START.startswith(s):
                self.chunk_acc = s
            if FUNCTION_CALL_START.startswith(s):
                self.chunk_acc = s
            else:
                super().callback_print(s)

            return

        self.chunk_acc = self.chunk_acc + s

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