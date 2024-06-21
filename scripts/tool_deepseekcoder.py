from collections.abc import Callable
import copy
import inspect
import json
import traceback
from types import GenericAlias
from typing import Any, get_origin, Annotated
from dataclasses import dataclass

import binding
from binding import PATH_BINDS
from tool_glm4 import build_system_prompt, get_tools, dispatch_tool

import chatllm, sys, re
from chatllm import ChatLLM

SELFCOG_PROMPT = "你是一个名为 DeepSeek Coder 的人工智能助手。你的任务是针对用户的问题和要求提供适当的答复和支持。"
DATE_PROMPT = "当前日期: %Y-%m-%d"

def build_system_prompt(functions: list[dict]):
    from datetime import datetime
    value = SELFCOG_PROMPT
    value += "\n\n" + datetime.now().strftime(DATE_PROMPT)
    value += "\n\n# 可用工具"
    contents = []
    for function in functions:
        content = f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
        content += "\n在调用上述函数时，直接输出函数名，并请使用 Json 格式表示调用的参数。"
        contents.append(content)
    value += "".join(contents)
    return value

def parse_function_call(s: str) -> tuple[str, dict] | None:

    def extract_code(text: str) -> str:
        pattern = r'```([^\n]*)\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1][1]

    try:
        tool_name, content = s.split("\n", maxsplit=1)
        tool_name = tool_name.strip()
        content   = extract_code(content.strip())

        code = json.loads(content)

        return tool_name, code
    except:
        return None

class ToolChatLLM(ChatLLM):
    chunk_acc = ''
    found = False
    first_chunk = True

    def callback_print(self, s: str) -> None:
        if self.first_chunk:
            self.first_chunk = False
            if s == 'function':
                self.found = True
                return

        if not self.found:
            super().callback_print(s)
            return

        self.chunk_acc = self.chunk_acc + s

        if parse_function_call(self.chunk_acc) is not None:
            self.abort()
            self.call_tool(self.chunk_acc)
            self.chunk_acc = ''

    def callback_end(self) -> None:
        self.first_chunk = True
        self.found = False
        self.chunk_acc = ''
        super().callback_end()

    def call_tool(self, s: str) -> None:
        tool_name, tool_param = parse_function_call(s)
        print(f"[Use Tool]: {tool_name}")
        rsp = dispatch_tool(tool_name, tool_param, '')
        self.tool_input('<｜tool▁output▁begin｜>' + rsp[0].text + '<｜tool▁output▁end｜>')

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(get_tools())], ToolChatLLM, lib_path=PATH_BINDS)

