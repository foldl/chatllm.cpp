import { dlopen, FFIType, suffix, JSCallback, ptr, CString } from "bun:ffi";

const path = `libchatllm.${suffix}`;

const {
    symbols: {
        chatllm_create,
        chatllm_append_param,
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

class ChatLLM {
    constructor(params) {
        this.obj = chatllm_create();
        this.callback_print = new JSCallback(
            (p_obj, ptr) => process.stdout.write(new CString(ptr)),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_end = new JSCallback(
            (p_obj) => 0,
            {
                args: ["ptr"],
            },
        );
        if (params.constructor === Array) {
            for (let param of params)
                this.append_param(param);
            this.start();
        }
    }

    append_param(s) {
        let str = Buffer.from(s + '\0', "utf8")
        chatllm_append_param(this.obj, ptr(str));
    }

    start() {
        let r = chatllm_start(this.obj, this.callback_print, this.callback_end);
        if (r != 0) {
            throw `ChatLLM::start error code ${r}`;
        }
    }

    chat(s) {
        let str = Buffer.from(s + '\0', "utf8")
        let r = chatllm_user_input(this.obj, ptr(str));
        if (r != 0) {
            throw `ChatLLM::chat error code ${r}`;
        }
    }
}

let llm = new ChatLLM(Bun.argv.slice(2));

const prompt = 'You  > ';
const AI     = 'A.I. > ';

process.stdout.write(prompt);
for await (const line of console) {
    process.stdout.write(AI);
    llm.chat(line);
    process.stdout.write(prompt);
}