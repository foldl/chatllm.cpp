## Quick Start

For Windows users, the easies way is to download a release, extract it, and start chatting:

```
main_nim -i -m :qwen2:0.5b
Downloading qwen2:0.5b
 |████████████████████████████████████████████████████████████| 100.0%
    ________          __  __    __    __  ___ (通义千问)
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by QWen2,                      /_/   /_/
with 494032768 (0.5B) parameters.

You  > hi
A.I. > Hello! How can I assist you today?
You  >
```

For Linux/MacOS (and Windows) users, build [binding](binding.md) and start chatting.

Note: `main_nim` built from [`main.nim`](../bindings/main.nim) supports model id and model downloading, while the _default_ `main`
built from [`main.cpp`](../src/main.cpp) does not.

### Download Quantized Models

A [script](../scripts/model_downloader.py) is provided, which can download some quantized models on demand.
When a model name starting with `:` is given to `-m` option (as shown in above example), this script will
treat it as a model ID and try to download it if the file does not exist.

Use `python model_downloader.py` to check all quantized models.