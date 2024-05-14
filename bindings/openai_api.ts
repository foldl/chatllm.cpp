var worker = new Worker('chatllm.ts');

worker.postMessage({id : '#start', argv: Bun.argv.slice(2)});

const model = 'chatllm-variant';

function make_chat_completion_chunk_obj(id: string, content: string, timestamp: number, model: string) {
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "system_fingerprint": "fp_xxx",
        "choices": [{
          "index": 0,
          "delta": {
            "role": "assistant",
            "content": content,
          },
          "logprobs": null,
          "finish_reason" : null
        }]
      };
}

function make_chat_completion_obj(id: string, timestamp: number, model: string, content: string = '') {
    let obj = make_chat_completion_chunk_obj(id, content, timestamp, model);
    obj.object = "chat.completion";
    obj.choices[0].finish_reason = "stop";
    obj.usage = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 1
    }
    return obj;
}

class ChatCompletionSession {
    controller: ReadableStreamDefaultController;
    model: string;
    created: number;
    streaming: boolean;
    acc: string;

    constructor(controller: ReadableStreamDefaultController, model: string, streaming: boolean) {
        this.controller = controller;
        this.model = model;
        this.created = Date.now();
        this.streaming = streaming;
        this.acc = '';
    }

    receive_chunk(id: string, chunk: string) {
        if (this.streaming) {
            this.controller.enqueue('data: ' + JSON.stringify(make_chat_completion_chunk_obj(id, chunk, this.created, this.model)) + '\n\n');
        } else {
            this.acc = this.acc + chunk;
        }
    }

    complete(id: string) {
        if (this.streaming) {
            this.controller.enqueue('data: ' + JSON.stringify(make_chat_completion_obj(id, this.created, this.model)) + '\n\n');
        } else {
            this.controller.enqueue('data: ' + JSON.stringify(make_chat_completion_obj(id, this.created, this.model, this.acc)) + '\n\n');
        }
    }
}

function make_legacy_chat_complete_chunk_obj(id: string, timestamp: number, model: string, content: string) {
    return {
        "id": id,
        "object": "text_completion",
        "created": timestamp,
        "choices": [
          {
            "text": content,
            "index": 0,
            "logprobs": null,
            "finish_reason": null
          }
        ],
        "model": model,
        "system_fingerprint": "fp_xxx"
      };
}

function make_legacy_chat_complete_obj(id: string, timestamp: number, model: string, content: string) {
    return {
        "id": id,
        "object": "text_completion",
        "created": timestamp,
        "model": model,
        "system_fingerprint": "fp_xxx",
        "choices": [
            {
            "text": content,
            "index": 0,
            "logprobs": null,
            "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 1
        }
    };
}

class LegacyChatCompletionSession {
    controller: ReadableStreamDefaultController;
    model: string;
    created: number;
    streaming: boolean;
    acc: string;
    first_chunk: boolean;

    constructor(controller: ReadableStreamDefaultController, model: string, streaming: boolean) {
        this.controller = controller;
        this.model = model;
        this.created = Date.now();
        this.streaming = streaming;
        this.acc = '';
        this.first_chunk = true;
    }

    receive_chunk(id: string, chunk: string) {
        if (this.streaming) {
            if (this.first_chunk) {
                this.first_chunk = false;
                this.controller.enqueue(JSON.stringify(make_legacy_chat_complete_chunk_obj(id, this.created, this.model, chunk)) + '\n');
            }
            this.controller.enqueue('data: ' + chunk + '\n');
        } else {
            this.acc = this.acc + chunk;
        }
    }

    complete(id: string) {
        if (this.streaming) {
            this.controller.enqueue('data: [DONE]\n');
        } else {
            this.controller.enqueue('data: ' + JSON.stringify(make_legacy_chat_complete_obj(id, this.created, this.model, this.acc)) + '\n\n');
        }
    }
}

var map_id2session: {[key: string]: ChatCompletionSession | LegacyChatCompletionSession | null} = {};
var id_counter = 1;

function create_new_id() {
    let id = 'chatllm-' + id_counter;
    id_counter += 1;
    return id;
}

function create_new_chat_complete(id: string, controller: ReadableStreamDefaultController, streaming: boolean) {
    map_id2session[id] = new ChatCompletionSession(controller, model, streaming);
}

function create_new_legacy_chat_complete(id: string, controller: ReadableStreamDefaultController, streaming: boolean) {
    map_id2session[id] = new LegacyChatCompletionSession(controller, model, streaming);
}

function cancel_chat_session(id: string) {
    delete map_id2session[id];
}

worker.onmessage = function(e) {
    let sess = map_id2session[e.data.id];

    if ((sess instanceof ChatCompletionSession) || (sess instanceof LegacyChatCompletionSession)) {
        if (e.data.type == 'chunk') {
            sess.receive_chunk(e.data.id, e.data.content);
        } else if (e.data.type == 'end') {
            sess.complete(e.data.id);
            sess.controller.close();
            delete map_id2session[e.data.id];
        }

        return;
    }

    console.log("unexpected session", sess);
}

async function chat_completion(req: Request, obj: any) {
    let prompt = '';
    for (let msg of obj.messages) {
        if (msg.role == 'user') prompt = msg.content;
    }
    if (prompt == '') return;

    let id = create_new_id();
    let streaming = obj.stream == true;

    let streamer = new ReadableStream({
        start(controller) {
            create_new_chat_complete(id, controller, streaming);
            worker.postMessage({id: id, user: prompt});
        },
        pull(controller) {
        },
        cancel() {
            cancel_chat_session(id);
        },
    });

    return new Response(streamer);
}

async function legacy_chat_completion(req: Request, obj: any) {
    let prompt = obj.prompt;
    let id = create_new_id();
    let streaming = obj.stream == true;

    let streamer = new ReadableStream({
        start(controller) {
            create_new_legacy_chat_complete(id, controller, streaming);
            worker.postMessage({id: id, user: prompt});
        },
        pull(controller) {
        },
        cancel() {
            cancel_chat_session(id);
        },
    });

    return new Response(streamer);
}

Bun.serve({
    port: 3000,
    async fetch(req: Request) {
        if (req.method != "POST") return new Response("Please use POST");
        if (req.url.endsWith("v1/chat/completions") || req.url.endsWith("v1/completions")) {
            let body = await (await req.blob()).text();
            let obj = JSON.parse(body);
            if (typeof(obj.prompt) == 'string') return await legacy_chat_completion(req, obj);
            return await chat_completion(req, obj);
        }
        return new Response("404");
    },
    error(error) {
        return new Response(`<pre>${error}\n${error.stack}</pre>`, {
            headers: {
                "Content-Type": "text/html",
            },
        });
    },
});