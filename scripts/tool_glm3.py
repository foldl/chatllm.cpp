"""
Copied from: https://github.com/THUDM/ChatGLM3/blob/main/composite_demo/tool_registry.py

This code is the tool registration part. By registering the tool, the model can call the tool.
This code provides extended functionality to the model, enabling it to call and interact with a variety of utilities
through defined interfaces.
"""

import copy
import inspect
from pprint import pformat
import traceback
from types import GenericAlias
from typing import get_origin, Annotated
import json, sys, re, os

this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}


def register_tool(func: callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
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

        tool_params.append({
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params
    }

    # print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS[tool_name] = tool_def

    return func


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_call(**tool_params)
    except:
        ret = traceback.format_exc()
    return str(ret)


def get_tools() -> dict:
    return copy.deepcopy(_TOOL_DESCRIPTIONS)


# Tool Definitions

@register_tool
def get_weather(
        city_name: Annotated[str, 'The name of the city to be queried', True],
) -> str:
    """
    Get the current weather for `city_name`
    """

    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc()

    return str(ret)

def build_sys_prompt():
    return "Answer the following questions as best as you can. You have access to the following tools: \n\n" + \
            json.dumps(get_tools(), indent=4, ensure_ascii=False)

import chatllm
from chatllm import ChatLLM

def call_function(s) -> str:

    def extract_code(text: str) -> str:
        pattern = r'```([^\n]*)\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1][1]

    def tool_call(*args, **kwargs) -> dict:
        return kwargs

    try:
        tool_name, *call_args_text = s.strip().split('\n')
        code = extract_code('\n'.join(call_args_text))
        args = eval(code, {'tool_call': tool_call}, {})
        observation = dispatch_tool(tool_name, args)
        return observation
    except:
        print("error occurs")
        return "failed to call the function"

class ToolChatLLM(ChatLLM):
    def call_tool(self, s: str) -> None:
        print(f"[Use Tool]: {s.strip().split()[0]}")
        rsp = call_function(s)
        self.tool_input(rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_sys_prompt()], ToolChatLLM, lib_path=PATH_BINDS)
