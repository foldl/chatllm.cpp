from collections.abc import Callable
import copy
import inspect
import json
import traceback
from types import GenericAlias
from typing import Any, get_origin, Annotated
from dataclasses import dataclass

import sys, os
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)

@dataclass
class ToolObservation:
    content_type: str
    text: str | dict
    image_url: str | None = None
    role_metadata: str | None = None
    metadata: Any = None
    id: str | None = None

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = []


def register_tool(func: Callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = {}
    required_params = []

    tpye_mapping = {
        "str": "string",
        "int": "integer",
    }

    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        if required:
            required_params.append(name)

        if typ in tpye_mapping:
            typ = tpye_mapping[typ]

        tool_params[name] = {
                "description": description,
                "type": typ,
            }

    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": tool_params,
                "required": required_params
            }
        }
    }
    # print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS.append(tool_def)

    return func


def dispatch_tool(tool_name: str, tool_params: dict, session_id: str | None) -> ToolObservation:
    if tool_name not in _TOOL_HOOKS:
        err = f"Tool `{tool_name}` not found. Please use a provided tool."
        return ToolObservation("system_error", err, id=session_id)

    tool_hook = _TOOL_HOOKS[tool_name]
    try:
        ret =tool_hook(**tool_params)
        return ToolObservation(tool_name, ret, id=session_id)
    except:
        err = traceback.format_exc()
        return ToolObservation("system_error", err, id=session_id)

def get_tools() -> list[dict]:
    return copy.deepcopy(_TOOL_DESCRIPTIONS)

# Tool Definitions

@register_tool
def get_weather(
        city_name: Annotated[str, "The name of the city to be queried", True]
) -> dict:
    """
    Get the current weather for `city_name`
    """

    import tool_glm4
    return tool_glm4.get_weather(city_name)

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
        calls = json.loads(s)
        observations = [dispatch_tool(c['name'], c['arguments'], c['id'] if 'id' in c else None) for c in calls]
        rsp = [{ 'content': o.text, 'id': o.id} if o.id is not None else { 'content': o.text} for o in observations]
        return json.dumps(rsp, ensure_ascii=False)
    except:
        print("error occurs")
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
        self.tool_input('[TOOL_RESULTS]' + rsp + '[/TOOL_RESULTS]')

    def chat(self, user_input: str, input_id = None) -> None:
        user_input, keywords = split_input(user_input)
        user_input = build_tool_prompt(get_tools(), keywords) + user_input
        super().chat(user_input, input_id = input_id)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:], ToolChatLLM, lib_path=PATH_BINDS)
