import sys, signal, time, os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
PATH_BINDS = os.path.join(this_dir, '..', 'bindings')
sys.path.append(PATH_BINDS)
from chatllm import LibChatLLM, ChatLLM, ChatLLMStreamer

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
        return True

class ChatCompletionNonStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)
        self.acc = ''

    def recv_chunk(self, content: str) -> bool:
        self.acc = self.acc + content
        return True

    def done(self) -> None:
        rsp = {
                "id": self.id,
                "object": "chat.completion",
                "created": self.timestamp,
                "model": self.model,
                "system_fingerprint": "fp_xxx",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": self.acc,
                    },
                    "logprobs": None,
                    "finish_reason" : "stop"
                }],
                'usage': {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 1
                }
            }
        self.send_str(json.dumps(rsp) + '\n\n')

class ChatCompletionStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)

    def recv_chunk(self, content: str) -> bool:
        rsp = {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.timestamp,
                "model": self.model,
                "system_fingerprint": "fp_xxx",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content,
                    },
                    "logprobs": None,
                    "finish_reason" : None
                }]
            }
        self.send_str(json.dumps(rsp) + '\n\n')
        return True

    def done(self) -> None:
        rsp = {
                "id": self.id,
                "object": "chat.completion",
                "created": self.timestamp,
                "model": self.model,
                "system_fingerprint": "fp_xxx",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": '',
                    },
                    "logprobs": None,
                    "finish_reason" : "stop"
                }],
                'usage': {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 1
                }
            }
        self.send_str(json.dumps(rsp) + '\n\n')

class LegacyCompletionNonStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)
        self.acc = ''

    def recv_chunk(self, content: str) -> bool:
        self.acc = self.acc + content
        return True

    def done(self) -> None:
        rsp = {
            "id": self.id,
            "object": "text_completion",
            "created": self.timestamp,
            "model": self.model,
            "system_fingerprint": "fp_xxx",
            "choices": [
                {
                "text": self.acc,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 1
            }
        }
        self.send_str(json.dumps(rsp) + '\n')

class LegacyCompletionStreamResponder(HttpResponder):
    def __init__(self, req: BaseHTTPRequestHandler, id: str, timestamp: int, model: str) -> None:
        super().__init__(req, id, timestamp, model)
        self.first_chunk = True

    def recv_chunk(self, content: str) -> bool:
        rsp = {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.timestamp,
                "choices": [
                {
                    "text": content,
                    "index": 0,
                    "logprobs": None,
                    #"finish_reason": None
                }
                ],
                "model": self.model,
                "system_fingerprint": "fp_xxx"
            }

        self.send_str('data: ' + json.dumps(rsp) + '\n\n')
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
http_server: HTTPServer = None

def get_streamer(model: str) -> ChatLLMStreamer | None:
    if model.endswith('fim') or model.startswith('fim'):
        return fim_streamer
    else:
        return chat_streamer

def handler(signal_received, frame):
    http_server.shutdown()
    sys.exit(0)

class HttpHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(404, 'POST')

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
        else:
            self.send_error(404, 'NOT FOUND')
            return

        self.send_response(200)

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
                    prompt = x['content'] + '\n' + prompt

            restart = counter < 2

            responder_cls = ChatCompletionStreamResponder if stream else ChatCompletionNonStreamResponder

            self.send_header('Content-type', 'application/json')
        else:
            prompt = obj['prompt']
            responder_cls = LegacyCompletionStreamResponder if stream else LegacyCompletionNonStreamResponder

            if stream:
                self.send_header('Content-type', 'text/event-stream')
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
            responder.recv_chunk('FIM model not loaded!')

        responder.done()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    ARG_SEP = '---'

    args = sys.argv[1:]
    if len(args) < 1:
        print(f"usage: python openai_api.py path/to/chat/model path/to/fim/model [more args for chat model {ARG_SEP} more args for fim model]")
        exit(-1)

    chat_args = args[2:]
    fim_args = []
    if ARG_SEP in chat_args:
        i = chat_args.index(ARG_SEP)
        fim_args = chat_args[i + 1:]
        chat_args = chat_args[:i]

    basic_args = ['-i', '-m']
    chat_streamer = ChatLLMStreamer(ChatLLM(LibChatLLM(PATH_BINDS), basic_args + [args[0]] + chat_args, False))

    if len(args) >= 2:
        fim_streamer = ChatLLMStreamer(ChatLLM(LibChatLLM(PATH_BINDS), basic_args + [args[1], '--format', 'completion'] + fim_args, False))
        fim_streamer.auto_restart = True

    http_server = HTTPServer(('0.0.0.0', 3000), HttpHandler)
    http_server.serve_forever()