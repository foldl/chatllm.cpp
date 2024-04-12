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

class ChatLLM {
    references: string[];
    obj: any;
    callback_print: JSCallback
    callback_print_reference: JSCallback
    callback_print_rewritten_query: JSCallback
    callback_end: JSCallback

    constructor(params: string[]) {
        this.obj = chatllm_create();
        this.callback_print = new JSCallback(
            (p_obj, ptr) => process.stdout.write(new CString(ptr)),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_print_reference = new JSCallback(
            (p_obj, ptr) => this.references.push(new CString(ptr)),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_print_rewritten_query = new JSCallback(
            (p_obj, ptr) => console.log(`Searching ${new CString(ptr)} ...`),
            {
                args: ["ptr", "ptr"],
            },
        );
        this.callback_end = new JSCallback(
            (p_obj) => {
                if (this.references.length < 1) return;
                console.log('References:');
                for (let x of this.references) console.log(x);
            },
            {
                args: ["ptr"],
            },
        );
        if (params.length > 0) {
            console.log(params);
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
        chatllm_set_print_reference(this.obj, this.callback_print_reference)
        chatllm_set_print_rewritten_query(this.obj, this.callback_print_rewritten_query)
        let r = chatllm_start(this.obj, this.callback_print, this.callback_end, 0);
        if (r != 0) {
            throw `ChatLLM::start error code ${r}`;
        }
    }

    chat(s: string) {
        this.references = []
        let str = Buffer.from(s + '\0', "utf8")
        let r = chatllm_user_input(this.obj, ptr(str));
        if (r != 0) {
            throw `ChatLLM::chat error code ${r}`;
        }
    }

    abort() {
        chatllm_abort_generation(this.obj);
    }
};

let llm = new ChatLLM(Bun.argv.slice(2));

const prompt = 'You  > ';
const AI     = 'A.I. > ';

process.stdout.write(prompt);
for await (const line of console) {
    process.stdout.write(AI);
    llm.chat(line);
    process.stdout.write(prompt);
}