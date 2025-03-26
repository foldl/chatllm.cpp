# GGMM File Format

GGMM is an extension of the original GGML format, by prepending meta information to the `Config` structure.
The meta field is just a JSON string allowing non ASCII characters (`json.dumps(..., , ensure_ascii=False)`).

Ideally, with this meta information, `Config` structure can be eliminated. But at present, meta information is
not used yet. A use case will be customizing model name shown in banner:

```json
{
    "model_name": "...",
    "model_native_name": "...",
}
```

For example, `python chatllm.py -m :aqualichat2:7b`.

## Why the name?

`M` is `Succ('L')` and the first letter of `meta`.

## Why not GGUF?

1. I don't like the idea of defining a kv data structure *again* with JSON already in-hand.

1. The ["Universal"](https://github.com/ggml-org/ggml/issues/220) concept does not make sense to me.

    GGMM is tightly couple with this inference: whenever this is a GGMM file, it will just works.