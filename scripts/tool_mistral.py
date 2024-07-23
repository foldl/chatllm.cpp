from collections.abc import Callable
import copy
import inspect
import json
import traceback
from types import GenericAlias
from typing import Any, get_origin, Annotated
from dataclasses import dataclass
import sys

from binding import PATH_BINDS

import tool_definition
from tool_definition import dispatch_tool

_TOOL_DESCRIPTIONS = None

def get_tools() -> list[dict]:
    def convert(tool: dict):
        tool_params = {}
        required_params = []
        for p in tool['parameters']:
            if p['required']: required_params.append(p['name'])

            tool_params[p['name']] = { "description": p['description'], "type": p['type'] }

        r = {
            "type": "function",
            "function": {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": { "type": "object", "properties": tool_params, "required": required_params }
            }
        }

        return r

    return [convert(t) for t in tool_definition._TOOL_DESCRIPTIONS]

def build_tool_prompt(functions: list[dict], keywords: list[str]):
    def filter_func(f: dict) -> bool:
        desc = f['function']['name']
        for k in keywords:
            if desc.find(k) >= 0:
                return True
        return False

    selected = [f for f in functions if filter_func(f)]
    if len(selected) < 1:
        return ""

    value = f"{json.dumps(selected, ensure_ascii=False)}"
    return "[AVAILABLE_TOOLS]" + value + "[/AVAILABLE_TOOLS]"

import chatllm
from chatllm import ChatLLM

def call_function(s: str) -> str:
    try:
        calls = tool_definition.json_decode_ignore_extra(s)
        observations = [dispatch_tool(c['name'], c['arguments'], c['id'] if 'id' in c else None) for c in calls]
        rsp = [{ 'content': o.text, 'id': o.id} if o.id is not None else { 'content': o.text} for o in observations]
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
        print(f"[Use Tools]: {s}")
        rsp = call_function(s)
        self.tool_input(rsp)

    def chat(self, user_input: str, input_id = None) -> None:
        user_input, keywords = split_input(user_input)
        user_input = build_tool_prompt(_TOOL_DESCRIPTIONS, keywords) + user_input
        super().chat(user_input, input_id = input_id)

if __name__ == '__main__':
    _TOOL_DESCRIPTIONS = get_tools()
    chatllm.demo_simple(sys.argv[1:], ToolChatLLM, lib_path=PATH_BINDS)
