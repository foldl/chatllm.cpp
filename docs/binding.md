# Bindings

## Precondition

Build target `libchatllm`:

### Windows:

Assume MSVC is used.

1. Build target `libchatllm`:

    ```shell
    cmake --build build --config Release --target libchatllm
    ```

1. Copy `libchatllm.dll`, `libchatllm.lib` and `ggml.dll` to `bindings`;

### Linux:

1. Build target `libchatllm`:

    ```shell
    cmake --build build --target libchatllm
    ```

## C

### Linux:

1. Build `bindings\main.c`:

    ```shell
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
    gcc main.c libchatllm.so
    ```

1. Test `a.out` with exactly the same command line options.

### Windows:

1. Build `bindings\main.c`:

    ```shell
    cl main.c libchatllm.lib
    ```

1. Test `main.exe` with exactly the same command line options.

## Python

Run [chatllm.py](../bindings/chatllm.py) with exactly the same command line options.

For example,

* Linux: `python3 chatllm.py -i -m path/to/model`

* Windows: `python chatllm.py -i -m path/to/model`