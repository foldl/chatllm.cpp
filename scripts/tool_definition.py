from collections.abc import Callable
import inspect
import traceback
from types import GenericAlias
from typing import Any, get_origin, Annotated
from dataclasses import dataclass
import json

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = []

def json_try_decode(s: str) -> dict | None:
    try:
        return json.loads(s)
    except:
        return None

def json_decode_ignore_extra(s: str) -> dict | None:
    for i in range(len(s)):
        d = json_try_decode(s[:i + 1])
        if d is not None: return d
    return None

def register_tool(func: Callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []

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

        if typ in tpye_mapping:
            typ = tpye_mapping[typ]

        tool_params.append({
                "name": name,
                "description": description,
                "type": typ,
                "required": required,
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

    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": [
            "temp_C",
            "FeelsLikeC",
            "humidity",
            "weatherDesc",
            "observation_time",
        ],
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

@dataclass
class ToolObservation:
    content_type: str
    text: str
    image_url: str | None = None
    role_metadata: str | None = None
    metadata: Any = None
    id: str | None = None

def dispatch_tool(tool_name: str, tool_params: dict, session_id: str | None = None) -> ToolObservation:
    if tool_name not in _TOOL_HOOKS:
        err = f"Tool `{tool_name}` not found. Please use a provided tool."
        return ToolObservation("system_error", err, id=session_id)

    tool_hook = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_hook(**tool_params)
        if isinstance(ret, dict):
            ret = json.dumps(ret, ensure_ascii=False)
        else:
            ret = str(ret)
        return ToolObservation(tool_name, ret, id=session_id)
    except:
        err = traceback.format_exc()
        return ToolObservation("system_error", err, id=session_id)

if __name__ == '__main__':
    print(_TOOL_DESCRIPTIONS)