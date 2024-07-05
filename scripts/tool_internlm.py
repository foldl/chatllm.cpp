import copy
import json
import sys

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

def get_tools() -> list[dict]:

    def convert(tool: dict):
        r = copy.deepcopy(tool)
        r['parameters'] = [p for p in r['parameters'] if p['required']]
        for p in r['parameters']:
            del p['required']
        return r

    return [convert(t) for t in tool_definition._TOOL_DESCRIPTIONS]

def build_system_prompt(functions: list[dict]):
    META_CN = ('当开启工具以及代码时，根据需求选择合适的工具进行调用\n'
           '你可以使用如下工具：'
           '\n{prompt}\n'
           '如果你已经获得足够信息，请直接给出答案. 避免不必要的工具调用! '
           '同时注意你可以使用的工具，不要随意捏造！')

    prompt = json.dumps(functions, ensure_ascii=False, indent=4)

    return META_CN.format(prompt=prompt)

import chatllm, sys, re
from chatllm import ChatLLM

class ToolChatLLM(ChatLLM):
    def call_tool(self, s: str) -> None:
        s = s.strip()
        TOK_PLUGIN = '<|plugin|>'
        if s.startswith(TOK_PLUGIN):
            plugin = s[len(TOK_PLUGIN):].strip()
            try:
                c = json.loads(plugin)
                print(f"[Use Tool]: {c['name']}")
                observations = dispatch_tool(c['name'], c['parameters'], c['id'] if 'id' in c else None)
                r = observations.text
                rsp = json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else str(r)
            except Exception as e:
                print(f"error occurs: {e}")
                rsp = "failed to call the function"

            self.tool_input(rsp)
        else:
            self.tool_input('error: unknown tool')

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(get_tools())], ToolChatLLM, lib_path=PATH_BINDS)