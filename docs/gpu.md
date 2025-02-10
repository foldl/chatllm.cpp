# GPU Acceleration

CAUTION: Support of GPU acceleration is preliminary. There are known [issues](#known-issues).

## Supported backends

Generally, all backends supported by GGML are available, with a focus on
below backends.

| Backend   | Target devices    |
| ---       | ---               |
| CUDA      | Nvidia GPU        |
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

Use `-ngl` (`--n_gpu_layers`) to specify number of layers to be deployed to GPU. We name the staffs before the first layer as "Prolog", and
staffs after the last layer as "Epilog". At present, "Prolog" is always put to CPU, while others are configurable with `-ngl`.
Suppose a model has `10` hidden layers, then `-ngl 10` will put all layers to GPU, and `-ngl 11` will also put the "Epilog" to GPU.

## Known issues

1. Custom operators (`ggml::map_custom...`);

    If hidden layers of a model use custom operators, then GPU acceleration is unavailable.

1. Use of `tie_word_embeddings`;

    "Epilog" may work on GPU (iGPU). If not, do not place "Epilog" on GPU.

1. `ggml` scheduler can't handle some operators properly.

    If a model has `10` hidden layers and `-ngl 10` not work, then try `-ngl 11`. If `-ngl 11` not work, then try `-ngl 9`.
