
# About Grok-1

Disclaimer: I am not sure if the implementation is correct or not, because I don't have enough compute resource to run the full model.

## Convert the model

To convert the base model, `jax` is needed:

```sh
pip install jax[cpu]

```

Download the [base model](https://huggingface.co/xai-org/grok-1) and [repository](https://github.com/xai-org/grok-1).

Use `convert.py` to convert it, for example quantized to Q4_0:

```sh
python convert.py -i /path/to/model/ckpt-0 --vocab_dir /path/to/repository -o grok.bin -a Grok-1-Base -t q4_0
```

**Bonus**: Use `--experts` to export a subset of experts, such as  `--experts 0,1,2,3` for the first 4 experts.
The converted model will have less parameters but performance will degrade significantly.
At least 2 experts are required. Remember that `NUM_EXPERTS` in `grok.cpp` should be the actual number of experts.

## Test

Below is a test run with the first 4 experts:

```sh
./bin/main -m ../grok-1-4_q4_0.bin -i --temp 0 --max_length 1024

    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by Grok-1,                     /_/   /_/
with 161064425472 (83.8B effect.) parameters.

You  > what is your name?
A.I. >

what is your age?

what is your weight?

...
```