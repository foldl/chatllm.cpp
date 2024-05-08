# Have fun with ChatLLM!

## Layer Shuffling

Llama-3 70B has been self-merged into a [120B](https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct) model,
which sounds quite interesting.

Now, using `--layer_spec`, we can do the same thing on-the-fly. For example, here we create a Llama-3 13.8B from 8B model and chat with it:

```sh
./bin/main -i -m path/to/llama3-8b.bin --layer_spec 0:8,4:12,8:16,12:20,16:24,20:28,24:32

    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by LlaMa3,                     /_/   /_/
with 13264949248 (13.3B) parameters.
```

The layers of new model are 0-1-2-3-4-5-6-7, 4-5-6-7-8-9-10-11, ... layers from the original model.

Before shuffling a model's layers, use `--show` to view basic information about it.