import json
from typing import Literal
import sys
from datetime import datetime

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

from tool_mistral import get_tools

FN_CALL_TEMPLATE = """system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: {date_string}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

def build_system_prompt(functions: list[dict]):
    tool_desc_template = FN_CALL_TEMPLATE
    tools_json = '\n\n'.join([json.dumps(f, ensure_ascii=False) for f in functions])
    tool_system = tool_desc_template.format(date_string=datetime.now().strftime('%Y-%m-%d'), tools_json=tools_json)
    return tool_system

import chatllm, sys, re
from chatllm import ChatLLM, LLMChatChunk

def call_function(c: dict) -> str:
    try:
        observations = dispatch_tool(c['name'], c['arguments'], c['id'] if 'id' in c else None)
        return observations.text
    except Exception as e:
        print(f"error occurs: {e}")
        return "failed to call the function"

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

TOOL_RESULT_START = "<tool_response>"
TOOL_RESULT_CLOSE = "</tool_response>"

class ToolChatLLM(ChatLLM):
    chunk_acc = ''
    tool_calls = []

    def callback_print(self, s: str) -> None:
        if self.chunk_acc is None:
            self.chunk_acc = ''

        if self.chunk_acc == '':
            if TOOL_CALL_START.startswith(s):
                self.chunk_acc = s
            else:
                super().callback_print(s)

            return

        self.chunk_acc = self.chunk_acc + s

        if len(self.chunk_acc) <= len(TOOL_CALL_START): return

        if not self.chunk_acc.startswith(TOOL_CALL_START):
            super().callback_print(self.chunk_acc)
            self.chunk_acc = ''

        close = self.chunk_acc.find(TOOL_CALL_CLOSE)
        if close > 0:
            self.tool_calls.append(self.chunk_acc[len(TOOL_CALL_START):close])
            s = self.chunk_acc[close + len(TOOL_CALL_CLOSE):]
            if len(s) > 0: super().callback_print(s)
            self.chunk_acc = ''

    def callback_end(self) -> None:
        for t in self.tool_calls:
            self.call_tool(t)

        self.chunk_acc = ''
        super().callback_end()
        self.tool_calls = []

    def call_tool(self, s: str) -> None:
        s = s.strip()
        tc = tool_definition.json_decode_ignore_extra(s)
        if not isinstance(tc, dict): return
        if not 'name' in tc: return

        print(f"[Use Tool]: {tc['name']}")
        rsp = call_function(tc)
        self.tool_input(TOOL_RESULT_START + rsp + TOOL_RESULT_CLOSE)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(get_tools())], ToolChatLLM, lib_path=PATH_BINDS)
