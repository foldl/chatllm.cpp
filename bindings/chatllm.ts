import { dlopen, FFIType, suffix, JSCallback, ptr, CString } from "bun:ffi";

const path = `libchatllm.${suffix}`;

enum PrintType {
    PRINT_CHAT_CHUNK        = 0,
    PRINTLN_META            = 1,    // print a whole line: general information
    PRINTLN_ERROR           = 2,    // print a whole line: error message
    PRINTLN_REF             = 3,    // print a whole line: reference
    PRINTLN_REWRITTEN_QUERY = 4,    // print a whole line: rewritten query
}

const {
    symbols: {
        chatllm_create,
        chatllm_append_param,
        chatllm_start,
        chatllm_user_input,
        chatllm_abort_generation,
        chatllm_restart,
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
        chatllm_restart: {
            args: [FFIType.ptr, FFIType.cstring]
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

    print_error(s: string) {
        throw s;
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
            (p_obj, print_type, ptr) => {
                let txt = new CString(ptr);
                switch (print_type) {
                    case PrintType.PRINT_CHAT_CHUNK:
                        this.handler.print(txt);
                        break;
                    case PrintType.PRINTLN_META:
                        this.handler.print(txt + '\n');
                        break;
                    case PrintType.PRINTLN_REF:
                        this.handler.print_reference(txt);
                        break;
                    case PrintType.PRINTLN_REWRITTEN_QUERY:
                        this.handler.print_rewritten_query(txt);
                        break;
                    case PrintType.PRINTLN_ERROR:
                        this.handler.print_error(txt);
                        break;
                    default:
                        throw print_type;
                }
            },
            {
                args: ["ptr", "i32", "ptr"],
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

    restart(sys_prompt: string | null) {
        // TODO: NULL pointer
        chatllm_restart(this.obj, sys_prompt != null ? ptr(sys_prompt) : ptr(0));
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

// FIXME: this does not work
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