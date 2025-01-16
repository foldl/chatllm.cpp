import re, sys

from binding import PATH_BINDS
from chatllm import ChatLLM, LibChatLLM

class CallableLLM(ChatLLM):

    def chat(self, user_input: str, input_id = None) -> str:
        self.chunk_acc = ''
        super().chat(user_input, input_id)
        return self.chunk_acc

    def callback_print(self, s: str) -> None:
        self.chunk_acc = self.chunk_acc + s

# Patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )

def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(
        SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html

def create_prompt(text: str, instruction: str = None, schema: str = None) -> str:
    """
    Create a prompt for the model with optional instruction and JSON schema.
    """
    if not instruction:
        instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    if schema:
        instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
        prompt = f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
    else:
        prompt = f"{instruction}\n```html\n{text}\n```"

    return prompt

def html2json(html: str):
    schema = """
    {
    "type": "object",
    "properties": {
        "title": {
        "type": "string"
        },
        "author": {
        "type": "string"
        },
        "date": {
        "type": "string"
        },
        "content": {
        "type": "string"
        }
    },
    "required": ["title", "author", "date", "content"]
    }
    """

    return create_prompt(html, schema=schema)

if __name__ == '__main__':

    model = ':reader-lm-v2'

    args = sys.argv[1:]
    if len(args) < 1:
        print(f"usage: python html2md.py path/to/html/file [{model}]")
        exit(-1)

    if len(args) > 1: model = args[1]

    llm = CallableLLM(LibChatLLM(PATH_BINDS), ['-m', model])

    with open(args[0], 'r', encoding='utf-8') as f:
        html = f.read()

    html = clean_html(html)
    input_prompt = create_prompt(html)
    print(llm.chat(input_prompt))