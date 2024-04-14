# Bindings

## Precondition

Build target `libchatllm`:

### Windows:

Assume MSVC is used.

1. Build target `libchatllm`:

    ```sh
    cmake --build build --config Release --target libchatllm
    ```

1. Copy `libchatllm.dll`, `libchatllm.lib` and `ggml.dll` to `bindings`;

### Linux:

1. Build target `libchatllm`:

    ```sh
    cmake --build build --target libchatllm
    ```

## Python

### Command line

Run [chatllm.py](../bindings/chatllm.py) with exactly the same command line options.

For example,

* Linux: `python3 chatllm.py -i -m path/to/model`

* Windows: `python chatllm.py -i -m path/to/model`

### Web demo

There is also a [Chatbot](../bindings/chatllm_st.py) powered by [Streamlit](https://streamlit.io/):

![](chatbot_st.png)

To start it:

```sh
streamlit run chatllm_st.py -- -i -m path/to/model
```

Note: "STOP" function is not implemented yet.

## JavaScript/TypeScript

Run [chatllm.ts](../bindings/chatllm.ts) with exactly the same command line options using [Bun](https://bun.sh/):

```shell
bun run chatllm.ts -i -m path/to/model
```

WARNING: Bun [looks buggy on Linux](https://github.com/oven-sh/bun/issues/10242).

## Other Languages

`libchatllm` can be utilized by all languages that can call into dynamic libraries. Take C as an example:

* Linux

    1. Build `bindings\main.c`:

        ```sh
        export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
        gcc main.c libchatllm.so
        ```

    1. Test `a.out` with exactly the same command line options.

* Windows:

    1. Build `bindings\main.c`:

        ```shell
        cl main.c libchatllm.lib
        ```

    1. Test `main.exe` with exactly the same command line options.