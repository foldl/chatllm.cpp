# GPU Acceleration

CAUTION: Support of GPU acceleration is preliminary. There are known [issues](#known-issues).

## Supported backends

Generally, all backends supported by GGML are available, with a focus on
below backends.

| Backend   | Target devices    |
| ---       | ---               |
| CUDA      | Nvidia GPU        |
| RPC       | Any               |
| Vulkan    | GPU               |

## Build

To build with Vulkan:

```sh
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
```

To build with CUDA:

```sh
cmake -B build -DGGML_CUDA=1
cmake --build build --config Release
```

For more information, please checkout [Build llama.cpp locally](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md).

## Usage

Use `-ngl` (`--n_gpu_layers`) to specify number of layers to be deployed to different backend devices
(Just treat the word `gpu` as an alias of _backend device_).
We name the staffs before the first layer as "Prolog", and
staffs after the last layer as "Epilog". "Prolog" and "Epilog" are treated as special layers, and they can also be configured from `-ngl`
by including `prolog` and `epilog` respectively.
Suppose there is a model with `10` hidden layers:

* `-ngl 5`: put the first 5 layers to the first device;
* `-ngl 100`: put all layers to the first device;
* `-ngl 5,prolog`: put the first 5 layers, and "Prolog" layer to the first device;
* `-ngl 100,prolog,epilog`: put all layers, "Prolog" layer and "Epilog" layer to the first device.
* `-ngl all`: equivalent to `-ngl 99999,prolog,epilog`.

The full format of `-ngl` is `-ngl [id:]layer_specs[;id:layer_specs]..`. `id` is device ID. If `id` is omitted, `0` is assumed.
`layer_spec` can be a positive integer, `prolog`, `epilog`, a combination of these; or just `all`.
Suppose device 0 is GPU, and device 1 is CPU, `-ngl 1:5;0:10` will put the first 5 layers to CPU, the next 10 layers to GPU,
and all other layers to CPU as default.

Tip: Use `--show_devices` to check all available devices and `--show` to check basic hyper parameters of a model.

## Known issues

1. Custom operators (`ggml::map_custom...`);

    If hidden layers of a model use custom operators, then GPU acceleration is unavailable.

1. Models with `tie_word_embeddings = true`;

    Ensure `Prolog` and `Epilog` layers are on the same device.

1. Other issues;

    If a model has `10` hidden layers and `-ngl 10` not work, then try `-ngl all`, `-ngl 10,epilog`, or `-ngl 9`.

## Tips

1. Having trouble with Python binding on Windows with CUDA?

    Copy these DLL to the `bindings` folder:

    * `cublas64_12.dll`
    * `cudart64_12.dll`
    * `cublasLt64_12.dll`