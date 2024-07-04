from collections.abc import Callable
import copy
import inspect
import json
import traceback
from types import GenericAlias
from typing import Any, get_origin, Annotated
from dataclasses import dataclass
import sys

import binding
from binding import PATH_BINDS

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = []

def register_tool(func: Callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
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

        if not required:
            continue

        if typ in tpye_mapping:
            typ = tpye_mapping[typ]

        tool_params.append({
                "name": name,
                "description": description,
                "type": typ,
            })

    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "parameters": tool_params,
    }
    # print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS.append(tool_def)

    return func

@register_tool
def get_weather(
        city_name: Annotated[str, "The name of the city to be queried", True],
) -> str:
    """
    Get the current weather for `city_name`
    """

    import tool_glm4
    return tool_glm4.get_weather(city_name)

def build_system_prompt(
    functions: list[dict],
):
    META_CN = ('当开启工具以及代码时，根据需求选择合适的工具进行调用\n'
           '你可以使用如下工具：'
           '\n{prompt}\n'
           '如果你已经获得足够信息，请直接给出答案. 避免不必要的工具调用! '
           '同时注意你可以使用的工具，不要随意捏造！')

    prompt = json.dumps(functions, ensure_ascii=False, indent=4)

    return META_CN.format(prompt=prompt)

@dataclass
class ToolObservation:
    content_type: str
    text: str | dict
    image_url: str | None = None
    role_metadata: str | None = None
    metadata: Any = None
    id: str | None = None

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
                print(f"[Use Tools]: {c['name']}")
                observations = dispatch_tool(c['name'], c['parameters'], c['id'] if 'id' in c else None)
                r = observations.text
                rsp = json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else str(r)
            except:
                print("error occurs")
                rsp = "failed to call the function"

            self.tool_input(rsp)
        else:
            self.tool_input('error: unknown tool')

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(_TOOL_DESCRIPTIONS)], ToolChatLLM, lib_path=PATH_BINDS)