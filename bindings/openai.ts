var worker = new Worker('chatllm.ts');
worker.postMessage({id : '#start', argv: Bun.argv.slice(2)});

const model = 'chatllm-variant';

class ChatCompletionSession {
    controller: ReadableStreamDefaultController;
    model: string;
    created: number;

    constructor(controller: ReadableStreamDefaultController, model: string) {
        this.controller = controller;
        this.model = model;
        this.created = Date.now();
    }
}

var map_id2session: {[key: string]: ChatCompletionSession} = {};
var id_counter = 1;

function create_new_chat_complete(controller: ReadableStreamDefaultController) {
    let id = 'chatllm-' + id_counter;
    id_counter += 1;
    map_id2session[id] = new ChatCompletionSession(controller, model);
    return id;
}

function make_chat_completion_obj(id: string, timestamp: number, model: string) {
    return {
        "id": id,
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "system_fingerprint": "fp_xxx",
        "choices": [{
          "index": 0,
          "message": {
            "role": "assistant",
            "content": '',
          },
          "logprobs": null,
          "finish_reason": "stop"
        }],
        "usage": {
          "prompt_tokens": 1,
          "completion_tokens": 1,
          "total_tokens": 1
        }
      };
}

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
          "finish_reason": null
        }]
      };
}

worker.onmessage = function(e) {
    let sess = map_id2session[e.data.id];
    if (!(sess instanceof ChatCompletionSession)) return;

    if (e.data.type == 'chunk') {
        sess.controller.enqueue(JSON.stringify(make_chat_completion_chunk_obj(e.data.id, e.data.content, sess.created, sess.model)) + '\n');
    }
    else if (e.data.type == 'end') {
        sess.controller.enqueue(JSON.stringify(make_chat_completion_obj(e.data.id, sess.created, sess.model)));
        sess.controller.close();
        delete map_id2session[e.data.id];
    }
}

async function chat_completion(req: Request) {
    let body = await (await req.blob()).text();
    let obj = JSON.parse(body);
    let prompt = '';
    for (let msg of obj.messages) {
        if (msg.role == 'user') prompt = msg.content;
    }
    if (prompt == '') return

    let streamer = new ReadableStream({
        start(controller) {
            let id = create_new_chat_complete(controller);
            worker.postMessage({id: id, user: prompt});
        },
        pull(controller) {
        },
        cancel() {

        },
    });

    return new Response(streamer);
}

Bun.serve({
    port: 3000,
    async fetch(req: Request) {
        console.log(`${req.method} ${req.url}`);
        if (req.method != "POST") return new Response("404");
        if (!req.url.endsWith("v1/chat/completions")) return new Response("404");
        return await chat_completion(req);
    },
    error(error) {
        return new Response(`<pre>${error}\n${error.stack}</pre>`, {
            headers: {
                "Content-Type": "text/html",
            },
        });
    },
});