import json
from typing import Literal
import sys

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

def get_tools() -> list[dict]:
    def convert(tool: dict):
        tool_params = {}
        required_params = []
        for p in tool['parameters']:
            if p['required']: required_params.append(p['name'])

            tool_params[p['name']] = { "description": p['description'], "type": p['type'] }

        r = {
            "name": tool['name'],
            "description": tool['description'],
            "parameters": { "type": "object", "properties": tool_params, "required": required_params }
        }

        return r

    return [convert(t) for t in tool_definition._TOOL_DESCRIPTIONS]

FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'
FN_STOP_WORDS = [FN_RESULT, f'{FN_RESULT}:', f'{FN_RESULT}:\n']

FN_CALL_TEMPLATE_ZH = """

# 工具

## 你拥有如下工具：

{tool_descs}

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

%s: 工具名称，必须是[{tool_names}]之一。
%s: 工具输入
%s: 工具结果，需将图片用![](url)渲染出来。
%s: 根据工具结果进行回复""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_EN = """

# Tools

## You have access to the following tools:

{tool_descs}

## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

%s: The tool to use, should be one of [{tool_names}]
%s: The input of the tool
%s: The result returned by the tool. The image needs to be rendered as ![](url)
%s: Reply based on tool result""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE = {
    'zh': FN_CALL_TEMPLATE_ZH,
    'en': FN_CALL_TEMPLATE_EN,
}

def get_function_description(function: dict, lang: Literal['en', 'zh']) -> str:
    """
    Text description of function
    """
    tool_desc_template = {
        'zh': '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
        'en': '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
    }
    tool_desc = tool_desc_template[lang]
    name = function.get('name', None)
    name_for_human = function.get('name_for_human', name)
    name_for_model = function.get('name_for_model', name)
    assert name_for_human and name_for_model
    args_format = function.get('args_format', '')
    return tool_desc.format(name_for_human=name_for_human,
                            name_for_model=name_for_model,
                            description_for_model=function['description'],
                            parameters=json.dumps(function['parameters'], ensure_ascii=False),
                            args_format=args_format).rstrip()

def build_system_prompt(functions: list[dict], lang: Literal['en', 'zh']):
    tool_desc_template = FN_CALL_TEMPLATE[lang]
    tool_descs = '\n\n'.join(get_function_description(function, lang=lang) for function in functions)
    tool_names = ','.join(function.get('name', function.get('name_for_model', '')) for function in functions)
    tool_system = tool_desc_template.format(tool_descs=tool_descs, tool_names=tool_names)
    return tool_system

import chatllm, sys, re
from chatllm import ChatLLM, LLMChatChunk

def parse_function_call(s: str) -> tuple[str, dict] | None:
    try:
        tool_name, content = s.split("\n", maxsplit=1)
        tool_name = tool_name.strip()
        content   = content.strip()

        if not tool_name.startswith(FN_NAME) or not content.startswith(FN_ARGS):
            return None

        tool_name = tool_name[len(FN_NAME + ':') :].strip()
        content = content[len(FN_ARGS + ':') :].strip()

        code = json.loads(content)

        return tool_name, code
    except:
        return None

def call_function(s: str, session_id: str = '') -> str:

    try:
        tool_name, code = parse_function_call(s)
        observation = dispatch_tool(tool_name, code, session_id)
        return observation.text
    except Exception as e:
        print(f"error occurs: {e}")
        return "failed to call the function"

class ToolChatLLM(ChatLLM):
    chunk_acc = ''

    def callback_print(self, s: str) -> None:
        if self.chunk_acc is None:
            self.chunk_acc = ''

        if self.chunk_acc == '':
            if FN_NAME.startswith(s):
                self.chunk_acc = s
            if FN_EXIT.startswith(s):
                self.chunk_acc = s
            else:
                super().callback_print(s)

            return

        self.chunk_acc = self.chunk_acc + s

        # Note: len(FN_EXIT) < len(FN_NAME)
        if self.chunk_acc == FN_EXIT + ':':
            self.chunk_acc = ''
            return

        if len(self.chunk_acc) < len(FN_NAME):
            return

        if len(self.chunk_acc) == len(FN_NAME):
            if self.chunk_acc != FN_NAME:
                super().callback_print(self.chunk_acc)
                self.chunk_acc = ''
                return

        if parse_function_call(self.chunk_acc) is not None:
            self.abort()
            self.call_tool(self.chunk_acc)
            self.chunk_acc = ''

    def callback_end(self) -> None:
        if self.chunk_acc != '':
            if self.chunk_acc.find(FN_ARGS) > 0:
                self.call_tool(self.chunk_acc)
            else:
                super().callback_print(self.chunk_acc)

        self.chunk_acc = ''
        super().callback_end()

    def call_tool(self, s: str) -> None:
        s = s.strip()
        print(f"[Use Tool]: {s.split()[1]}")
        rsp = call_function(s)
        self.tool_input(FN_RESULT + ': ' + rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(get_tools(), 'en')], ToolChatLLM, lib_path=PATH_BINDS)
