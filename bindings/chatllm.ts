import { dlopen, FFIType, suffix, JSCallback, ptr, CString } from "bun:ffi";

const path = `libchatllm.${suffix}`;

const {
    symbols: {
        chatllm_create,
        chatllm_append_param,
        chatllm_set_print_reference,
        chatllm_set_print_rewritten_query,
        chatllm_start,
        chatllm_user_input,
        chatllm_abort_generation,
    },
} = dlopen(
    path,
    {
        chatllm_create: {
            args: [],
            returns: FFIType.ptr,
        },
        chatllm_append_param: {
            args: [FFIType.ptr, FFIType.cstring],
        },
        chatllm_set_print_reference: {
            args: [FFIType.ptr, FFIType.function],
            returns: FFIType.i32
        },
        chatllm_set_print_rewritten_query: {
            args: [FFIType.ptr, FFIType.function],
            returns: FFIType.i32
        },
        chatllm_start: {
            args: [FFIType.ptr, FFIType.function, FFIType.function, FFIType.ptr],
            returns: FFIType.i32
        },
        chatllm_user_input: {
            args: [FFIType.ptr, FFIType.cstring],
            returns: FFIType.i32
        },
        chatllm_abort_generation: {
            args: [FFIType.ptr]
        },
    },
);

class ChatLLMHandler {
    references: string[]

    constructor() {
        this.references = []
    }

    print(s: string) {
    }

    print_reference(s: string) {
        this.references.push(s);
    }

    print_rewritten_query(s: string) { }

    start () {
        this.references = [];
    }

    end() { }
};

class ChatLLM {
    obj: any;
    callback_print: JSCallback;
    callback_print_reference: JSCallback;
    callback_print_rewritten_query: JSCallback;
    callback_end: JSCallback;
    handler: ChatLLMHandler;

    constructor(params: string[], handler: ChatLLMHandler) {
        this.handler = handler;
        this.obj = chatllm_create();
        this.callback_print = new JSCallback(
            (p_obj, ptr) => this.handler.print(new CString(ptr)),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_print_reference = new JSCallback(
            (p_obj, ptr) => this.handler.print_reference(new CString(ptr)),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_print_rewritten_query = new JSCallback(
            (p_obj, ptr) => this.handler.print_rewritten_query(new CString(ptr)),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_end = new JSCallback(
            (p_obj) => this.handler.end(),
            {
                args: ["ptr"],
            },
        );
        if (params.length > 0) {
            for (let param of params)
                this.append_param(param);
            this.start();
        }
    }

    append_param(s) {
        let str = Buffer.from(s + '\0', "utf8");
        chatllm_append_param(this.obj, ptr(str));
    }

    start() {
        chatllm_set_print_reference(this.obj, this.callback_print_reference);
        chatllm_set_print_rewritten_query(this.obj, this.callback_print_rewritten_query);
        let r = chatllm_start(this.obj, this.callback_print, this.callback_end, 0);
        if (r != 0) {
            throw `ChatLLM::start error code ${r}`;
        }
    }

    chat(s: string) {
        let str = Buffer.from(s + '\0', "utf8");
        let r = chatllm_user_input(this.obj, ptr(str));
        if (r != 0) {
            throw `ChatLLM::chat error code ${r}`;
        }
    }

    abort() {
        chatllm_abort_generation(this.obj);
    }
};

class WorkerHandler extends ChatLLMHandler {
    id: string;

    constructor(id: string) {
        super();
        this.id = id;
    }

    print(s: string) {
        if (this.id == '') return;
        postMessage({type: 'chunk', id: this.id, content: s});
    }

    print_rewritten_query(s: string) {
        if (this.id == '') return;
        postMessage({type: 'rewritten_query', id: this.id, content: s});
    }

    end() {
        if (this.id == '') return;

        if (this.references.length > 0)
            postMessage({type: 'references', id: this.id, content: this.references});

        postMessage({type: 'end', id: this.id});
    }
};

class StdIOHandler extends ChatLLMHandler {

    print(s: string) {
        process.stdout.write(s);
    }

    print_rewritten_query(s: string) {
        console.log(`Searching ${s} ...`)
    }

    end() {
        if (this.references.length < 1) return;
        console.log("\nReferences:")
        for (let s of this.references)
            console.log(s)
    }

};

if (Bun.argv.slice(2).length > 0)
{
    let llm = new ChatLLM(Bun.argv.slice(2), new StdIOHandler());

    const prompt = 'You  > ';
    const AI     = 'A.I. > ';

    process.stdout.write(prompt);
    for await (const line of console) {
        process.stdout.write(AI);
        llm.chat(line);
        process.stdout.write(prompt);
    }
}

let llm: ChatLLM | null = null;

onmessage = function(msg) {
    console.log('Worker: ', msg.data);

    if (msg.data.id == '#start') {
        llm = new ChatLLM(msg.data.argv, new WorkerHandler(''));
        return;
    }

    if (llm == null) return;

    llm.handler = new WorkerHandler(msg.data.id);
    llm.chat(msg.data.user);
}