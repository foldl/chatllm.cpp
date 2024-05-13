import threading
from typing import Any, Iterable, List, Union
import os, sys, signal, queue, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from chatllm import LibChatLLM, ChatLLM, LLMChatInput, LLMChatDone, LLMChatChunk, ChatLLMStreamer
import json

MODEL = 'chatllm-variant'

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
llm_streamer: ChatLLMStreamer = None

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

        if (self.path != '/v1/completions') and (self.path != '/v1/chat/completions') and \
           (self.path != '/completions') and (self.path != '/chat/completions'):
            self.send_error(404, 'NOT FOUND')
            return

        self.send_response(200)

        id = session_man.make_id()
        timestamp = int(time.time())
        prompt = ''
        stream = False
        if 'stream' in obj:
            stream = obj['stream']
        if 'messages' in obj:
            for x in obj['messages']:
                if x['role'] == 'user':
                    prompt = x['content']

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

        responder = responder_cls(self, id, timestamp, MODEL)
        for x in llm_streamer.chat(prompt):
            #print(x)
            try:
                responder.recv_chunk(x)
            except:
                llm_streamer.abort()

        responder.done()

def handler(signal_received, frame):
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    args = sys.argv[1:]
    llm = ChatLLM(LibChatLLM(), args, False)
    llm_streamer = ChatLLMStreamer(llm)
    httpd = HTTPServer(('localhost', 3000), HttpHandler)
    httpd.serve_forever()