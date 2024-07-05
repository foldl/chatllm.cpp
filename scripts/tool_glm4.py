import json
import sys

from binding import PATH_BINDS

import tool_definition
from tool_definition import ToolObservation

def tool_not_implemented(code: str, session_id: str) -> list[ToolObservation]:
    return [ToolObservation("system_error", 'not implemented yet')]

ALL_TOOLS = {
    "simple_browser": tool_not_implemented,
    "python": tool_not_implemented,
    "cogview": tool_not_implemented,
}

def dispatch_tool(tool_name: str, tool_params: dict, session_id: str) -> list[ToolObservation]:
    if tool_name in ALL_TOOLS:
        return ALL_TOOLS[tool_name](json.dumps(tool_params, ensure_ascii=False), session_id)

    return tool_definition.dispatch_tool(tool_name, tool_params, session_id)

def get_tools() -> list[dict]:
    import tool_glm3
    return tool_glm3.get_tools()


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
        observation = dispatch_tool(tool_name, code, session_id)
        return observation.text
    except Exception as e:
        print(f"error occurs: {e}")
        return "failed to call the function"

class ToolChatLLM(ChatLLM):
    def call_tool(self, s: str) -> None:
        s = s.strip()
        print(f"[Use Tool]: {s.split()[0]}")
        rsp = call_function(s)
        self.tool_input(rsp)

if __name__ == '__main__':
    chatllm.demo_simple(sys.argv[1:] + ['-s', build_system_prompt(ALL_TOOLS, get_tools())], ToolChatLLM, lib_path=PATH_BINDS)
