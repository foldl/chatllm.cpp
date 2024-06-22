"""
Copied from: https://github.com/THUDM/GLM-4/blob/main/composite_demo/src/tools/tool_registry.py

This code is the tool registration part. By registering the tool, the model can call the tool.
This code provides extended functionality to the model, enabling it to call and interact with a variety of utilities
through defined interfaces.
"""

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

@dataclass
class ToolObservation:
    content_type: str
    text: str
    image_url: str | None = None
    role_metadata: str | None = None
    metadata: Any = None

def tool_not_implemented(code: str, session_id: str) -> list[ToolObservation]:
    return [ToolObservation("system_error", 'not implemented yet')]

ALL_TOOLS = {
    #"simple_browser": tool_not_implemented,
    "python": tool_not_implemented,
    #"cogview": tool_not_implemented,
}

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = []


def register_tool(func: Callable):
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

        tool_params.append(
            {
                "name": name,
                "description": description,
                "type": typ,
                "required": required,
            }
        )
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params,
    }
    # print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS.append(tool_def)

    return func

def dispatch_tool(tool_name: str, tool_params: dict, session_id: str) -> list[ToolObservation]:
    # Dispatch predefined tools
    if tool_name in ALL_TOOLS:
        return ALL_TOOLS[tool_name](json.dumps(tool_params, ensure_ascii=False), session_id)

    if tool_name not in _TOOL_HOOKS:
        err = f"Tool `{tool_name}` not found. Please use a provided tool."
        return [ToolObservation("system_error", err)]

    tool_hook = _TOOL_HOOKS[tool_name]
    try:
        ret: str = tool_hook(**tool_params)
        return [ToolObservation(tool_name, str(ret))]
    except:
        err = traceback.format_exc()
        return [ToolObservation("system_error", err)]

def get_tools() -> list[dict]:
    return copy.deepcopy(_TOOL_DESCRIPTIONS)


# Tool Definitions


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

        ret = (
                "Error encountered while fetching weather data!\n" + traceback.format_exc()
        )

    return str(ret)

SELFCOG_PROMPT = "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
DATE_PROMPT = "当前日期: %Y-%m-%d"
TOOL_SYSTEM_PROMPTS = {
    "python": "当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。",
    "simple_browser": "你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。",
    "cogview": "如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。",
}

def build_system_prompt(
    enabled_tools: list[str],
    functions: list[dict],
):
    from datetime import datetime
    value = SELFCOG_PROMPT
    value += "\n\n" + datetime.now().strftime(DATE_PROMPT)
    value += "\n\n# 可用工具"
    contents = []
    for tool in enabled_tools:
        contents.append(f"\n\n## {tool}\n\n{TOOL_SYSTEM_PROMPTS[tool]}")
    for function in functions:
        content = f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
        content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
        contents.append(content)
    value += "".join(contents)
    return value

import chatllm, sys, re
from chatllm import ChatLLM

def call_function(s: str, session_id: str = '') -> str:

    try:
        tool_name, content = s.split("\n", maxsplit=1)
        code = eval(content)
        observations = dispatch_tool(tool_name, code, session_id)
        rsp = '\n'.join([o.text for o in observations])
        return rsp
    except:
        print("error occurs")
        return "failed to call the function"

class ToolChatLLM(ChatLLM):
    def call_tool(self, s: str) -> None:
        s = s.strip()
        print(f"[Use Tool]: {s.split()[0]}")
        rsp = call_function(s)
        self.tool_input(rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(ALL_TOOLS, get_tools())], ToolChatLLM, lib_path=PATH_BINDS)
