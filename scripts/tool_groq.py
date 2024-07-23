import copy
import json
import sys

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

def gen_prompt_for_tool(func: dict) -> str:
    tool_params = {}
    required_params = []
    for p in func['parameters']:
        if p['required']: required_params.append(p['name'])

        tool_params[p['name']] = { "description": p['description'], "type": p['type'] }
    params = { "type": "object", "properties": tool_params, "required": required_params }
    return {"name": func['name'], "description": func['description'], "parameters": params}

SYS_PROMPT_TEMPLATE = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

Here are the available tools:
<tools> {tools_def} </tools>"""

def build_system_prompt(functions: list[dict]):

    prompt = '\n'.join([json.dumps(gen_prompt_for_tool(f), ensure_ascii=False, indent=4) for f in functions])

    return SYS_PROMPT_TEMPLATE.replace('{tools_def}', prompt)

import chatllm
from chatllm import ChatLLM

def call_function(s: str) -> str:
    try:
        c= json.loads(s)
        print(f"[Use Tool]: {c['name']}")
        observation = dispatch_tool(c['name'], c['arguments'], c['id'])
        rsp = {'id': c['id'], 'result': observation.text}
        return json.dumps(rsp, ensure_ascii=False)
    except Exception as e:
        print(f"error occurs: {e}")
        return "failed to call the function"

def split_input(user_input: str):
    keywords = []
    if user_input.startswith(':'):
        parts = user_input.split(':', maxsplit=2)
        if len(parts) == 3:
            keywords = parts[1].split()
            user_input = parts[-1]
    return user_input.strip(), keywords

class ToolChatLLM(ChatLLM):
    def call_tool(self, s: str) -> None:
        s = s.strip()
        rsp = call_function(s)
        self.tool_input(rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(tool_definition._TOOL_DESCRIPTIONS)], ToolChatLLM, lib_path=PATH_BINDS)