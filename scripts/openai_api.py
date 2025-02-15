import sys, signal, time, os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import itertools
from dataclasses import dataclass, field, asdict

this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)
from chatllm import LibChatLLM, ChatLLM, ChatLLMStreamer

@dataclass
class APIUsage:
    prompt_tokens: int = 1
    completion_tokens: int = 1
    total_tokens: int = 1

@dataclass
class ChunkChoiceDelta:
    role: str = "assistant"
    content: str = ""

@dataclass
class ChunkChoice:
    delta: ChunkChoiceDelta
    index: int = 0
    logprobs: str | None = None
    finish_reason: str | None = None

@dataclass
class CompletionChunkResponse:
    id: str
    created: int
    model: str
    choices: list[ChunkChoice] = field(default_factory=list)
    object: str = "chat.completion.chunk"
    system_fingerprint: str = "fp_xxx"

@dataclass
class CompletionResponse:
    id: str
    created: int
    model: str
    choices: list[ChunkChoice] = field(default_factory=list)
    object: str = "chat.completion.chunk"
    system_fingerprint: str = "fp_xxx"
    usage = APIUsage()

@dataclass
class LegacyChoice:
    index: int = 0
    text: str = ""
    logprobs = None
    finish_reason = None

@dataclass
class LegacyCompletionChunkResponse:
    id: str
    created: int
    model: str
    choices: list[LegacyChoice] = field(default_factory=list)
    object: str = "chat.completion.chunk"
    system_fingerprint: str = "fp_xxx"

@dataclass
class LegacyCompletionResponse:
    id: str
    created: int
    model: str
    choices: list[LegacyChoice] = field(default_factory=list)
    object: str = "text_completion"
    system_fingerprint: str = "fp_xxx"
    usage = APIUsage()

@dataclass
class Embedding:
    index: int
    embedding: list[float] = field(default_factory=list)
    object: str = "embedding"

@dataclass
class TokenUsage:
    prompt_tokens: int = 8
    total_tokens: int = 8

@dataclass
class DataListResponse:
    model: str
    data: list[str] = field(default_factory=list)
    object: str = "list"
    usage: TokenUsage = field(default_factory=TokenUsage)

class HttpResponder:
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        self.req = req
        self.timestamp = timestamp
        self.model = model
        self.id = id

    def recv_chunk(self, chunk: str) -> bool:
        return True

    def done(self) -> None:
        pass

    def send_str(self, s: str) -> bool:
        self.req.wfile.write(s.encode('utf-8'))
        self.req.wfile.flush()
        return True

class ChatCompletionNonStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)
        self.acc = ''

    def recv_chunk(self, content: str) -> bool:
        self.acc = self.acc + content
        return True

    def done(self) -> None:
        rsp = CompletionResponse(id=self.id, created=self.timestamp, model=self.model)
        rsp.choices.append(ChunkChoice(delta=ChunkChoiceDelta(content=self.acc), finish_reason="stop"))
        self.send_str(json.dumps(asdict(rsp)) + '\n\n')

class ChatCompletionStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)

    def recv_chunk(self, content: str) -> bool:
        rsp = CompletionChunkResponse(id=self.id, created=self.timestamp, model=self.model)
        rsp.choices.append(ChunkChoice(delta=ChunkChoiceDelta(content=content)))
        self.send_str('data: ' + json.dumps(asdict(rsp)) + '\n\n')
        return True

    def done(self) -> None:
        rsp = CompletionResponse(id=self.id, created=self.timestamp, model=self.model)
        rsp.choices.append(ChunkChoice(delta=ChunkChoiceDelta(), finish_reason="stop"))
        self.send_str('data: ' + json.dumps(asdict(rsp)) + '\n\n')
        self.send_str('data: [DONE]\n')

class LegacyCompletionNonStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)
        self.acc = ''

    def recv_chunk(self, content: str) -> bool:
        self.acc = self.acc + content
        return True

    def done(self) -> None:
        rsp = LegacyCompletionResponse(id=self.id, created=self.timestamp, model=self.model)
        rsp.choices.append(LegacyChoice(text=self.acc, finish_reason="length"))
        self.send_str(json.dumps(asdict(rsp)) + '\n')
        self.send_str('data: [DONE]\n')

class LegacyCompletionStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)
        self.first_chunk = True

    def recv_chunk(self, content: str) -> bool:
        rsp = LegacyCompletionChunkResponse(id=self.id, created=self.timestamp, model=self.model)
        rsp.choices.append(LegacyChoice(text=content))
        self.send_str('data: ' + json.dumps(asdict(rsp)) + '\n\n')
        return True

    def done(self) -> None:
        self.send_str('data: [DONE]\n')

class SessionManager:
    def __init__(self) -> None:
        self._id = 1

    def make_id(self) -> str:
        self._id += 1
        return '_chatllm_' + str(self._id)

session_man: SessionManager = SessionManager()
chat_streamer: ChatLLMStreamer = None
fim_streamer: ChatLLMStreamer = None
emb_model_obj: ChatLLM = None
http_server: HTTPServer = None

def get_streamer(model: str) -> ChatLLMStreamer | None:
    if model.endswith('fim') or model.startswith('fim'):
        return fim_streamer
    else:
        return chat_streamer

def handler(signal_received, frame):
    print("Ctrl+C pressed, shutting down server...")
    os._exit(0)

class HttpHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(404, 'POST')

    def handl_EMBEDDING(self, obj: dict):
        if emb_model_obj is None:
            self.send_response(404, 'NOT SUPPORTED')
            return

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        def mk_emb(i: int, emb) -> dict:
            return { "object": "embedding", "embedding": emb, "index": i }

        input = obj['input']
        if isinstance(input, str):
            input = [input]

        rsp = DataListResponse(model=obj['model'])
        for i, s in enumerate(input):
            rsp.data.append(Embedding(index=i, embedding=emb_model_obj.text_embedding(s)))

        self.wfile.write(json.dumps(asdict(rsp)).encode('utf-8'))

    def do_POST(self):
        print(self.path)
        args = self.rfile.read(int(self.headers['content-length'])).decode('utf-8')
        try:
            obj = json.loads(args)
            print(obj)
        except:
            self.send_error(404, 'BAD REQ')
            return

        model = obj['model'] if 'model' in obj else 'chat'

        max_tokens = obj['max_tokens'] if 'max_tokens' in obj else -1

        if self.path.endswith('/completions'):
            pass
        elif self.path.endswith('/generate'):
            model = 'fim'
        elif self.path.endswith('/embeddings'):
            self.handl_EMBEDDING(obj)
            return
        else:
            self.send_error(404, 'NOT FOUND')
            return

        self.send_response(200)
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('vary', 'origin, access-control-request-method, access-control-request-headers')
        self.send_header('access-control-allow-credentials', 'true')
        self.send_header('x-content-type-options', 'nosniff')

        id = session_man.make_id()
        timestamp = int(time.time())
        prompt = ''
        stream = False
        restart = False
        if 'stream' in obj:
            stream = obj['stream']
        if 'messages' in obj:
            counter = 0
            flag = True
            # aggregate all user messages
            for i in range(len(obj['messages']) - 1, -1, -1):
                x = obj['messages'][i]
                if x['role'] == 'user':
                    counter = counter + 1
                else:
                    flag = False

                if flag:
                    if isinstance(x['content'], list):
                        prompt = '\n'.join([o['text'] for o in x['content'] if o['type'] == 'text']) + '\n' + prompt
                    else:
                        prompt = str(x['content']) + '\n' + prompt

            restart = counter < 2

            responder_cls = ChatCompletionStreamResponder if stream else ChatCompletionNonStreamResponder
        else:
            prompt = obj['prompt']
            responder_cls = LegacyCompletionStreamResponder if stream else LegacyCompletionNonStreamResponder

        if stream:
            self.send_header('Content-type', 'text/event-stream; charset=utf-8')
        else:
            self.send_header('Content-type', 'application/json')

        self.end_headers()

        responder = responder_cls(self, id, timestamp, model)
        streamer = get_streamer(model)
        if streamer is not None:
            streamer.set_max_gen_tokens(max_tokens)

            try:
                if restart: streamer.restart()

                for x in streamer.chat(prompt):
                    responder.recv_chunk(x)
            except:
                streamer.abort()
        else:
            responder.recv_chunk('CHAT model not loaded!')

        responder.done()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    ARG_SEP = '---'

    args = sys.argv[1:]
    if len(args) < 1:
        print(f"usage: python openai_api.py [{ARG_SEP}TYPE path/to/model [additional args]]")
        print('where TYPE ::= chat | fim | emb')
        exit(-1)

    chat_args = ['-m']
    fim_args = ['-m']
    emb_args = ['-m']
    current = []
    for a in args:
        if a.startswith(ARG_SEP):
            t = a[len(ARG_SEP):]
            if t == 'chat': current = chat_args
            elif t == 'fim': current = fim_args
            elif t == 'emb': current = emb_args
            else:
                current = []
        else:
            current.append(a)

    if len(chat_args) > 1:
        chat_streamer = ChatLLMStreamer(ChatLLM(LibChatLLM(PATH_BINDS), chat_args, False))

    if len(fim_args) > 1:
        fim_streamer = ChatLLMStreamer(ChatLLM(LibChatLLM(PATH_BINDS), fim_args + ['--format', 'completion'], False))
        fim_streamer.auto_restart = True

    if len(emb_args) > 1:
        emb_model_obj = ChatLLM(LibChatLLM(PATH_BINDS), emb_args)

    http_server = HTTPServer(('0.0.0.0', 3000), HttpHandler)
    http_server.serve_forever()