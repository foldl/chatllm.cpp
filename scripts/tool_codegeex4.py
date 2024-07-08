import copy
import json, sys, re

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

def gen_prompt_for_tool(index: int, func: dict) -> str:
    tool_params = {}
    required_params = []
    for p in func['parameters']:
        if p['required']: required_params.append(p['name'])

        tool_params[p['name']] = { "description": p['description'], "type": p['type'] }
    params = { "type": "object", "properties": tool_params, "required": required_params }

    prompt  = f"\n## Function {index}\n"
    prompt += f"\n### Name\n{func['name']}\n"
    prompt += f"\n### Description\n{func['description']}\n"
    prompt += f"\n### Parameters\n```json\n{json.dumps(params, ensure_ascii=False)}\n```\n"
    return prompt

def gen_prompt_for_tools() -> str:
    funcs = tool_definition._TOOL_DESCRIPTIONS
    return ''.join([gen_prompt_for_tool(index + 1, func) for index, func in enumerate(funcs)])

def build_sys_prompt():
    return """
你将接收到一个用户提出的问题，并请撰写清晰、简洁且准确的答案。

# Note
- 我将给你提供一些函数工具的接口信息，包括函数的定义、用途、名字、参数名和参数类型。
- 请根据这些信息，为用户的指令，从中选择最合适的函数，并给出调用时需要使用的参数。
- **返回类型为一个json格式的字符串，包含函数名和参数字典。**
    - name: 函数名
    - arguments: 参数字典，其中key为参数名，value为参数类型。
- **只需要生成答案即可，无需在你的回答之前或之后做出解释，也不要直接回答用户的问题。**
- 只用当提供的函数工具不足以完成任务时，请你用正常的语气告知用户并解释原因。

# Functions
以下是可使用的函数工具的接口信息。
""" + gen_prompt_for_tools()

import chatllm
from chatllm import ChatLLM

def call_function(s: str) -> str:
    tool_name = 'unknown'
    try:
        func = json.loads(s.strip())
        tool_name = func['name']
        print(f"[Use Tool] {tool_name}")
        args = func['arguments']
        observation = dispatch_tool(tool_name, args)
        return observation.text
    except Exception as e:
        print(f"error occurs: {e}")
        return f"error occurs when using `{tool_name}`"

def extract_call(text: str) -> list[str]:
    """
    each call is wrapped with ```json```.
    """
    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [s.strip() for s in matches]

class ToolChatLLM(ChatLLM):
    chunk_acc = ''
    non_empty_acc = ''
    not_tool = False
    JSON_START = '```json'
    JSON_END = '```'
    tool_calls = []

    def callback_print(self, s: str) -> None:
        if self.not_tool:
            super().callback_print(s)
            return

        self.chunk_acc = self.chunk_acc + s
        self.non_empty_acc = self.non_empty_acc + s.strip('\t\n ')

        if len(self.non_empty_acc) < len(self.JSON_START): return

        if not self.non_empty_acc.startswith(self.JSON_START):
            self.not_tool = True
            super().callback_print(self.chunk_acc)
            self.chunk_acc = ''
            return

        extracted = extract_call(self.non_empty_acc)
        if len(extracted) > 0:
            self.tool_calls = self.tool_calls + extracted
            self.non_empty_acc = ''
            self.chunk_acc = ''

    def callback_end(self) -> None:
        super().callback_end()

        rsp = '\n'.join([call_function(s) for s in self.tool_calls])
        if rsp != '':
            self.tool_input(rsp)

        self.chunk_acc = ''
        self.non_empty_acc = ''
        self.not_tool = False
        self.tool_calls = []

if __name__ == '__main__':
    #print(build_sys_prompt())
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_sys_prompt()], ToolChatLLM, lib_path=PATH_BINDS)
