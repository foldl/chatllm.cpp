# Distributed Inference

Distributed inference is easy with the help RPC backend.
Basically, start one or more RPC server(s), then they can be used just like a backend device.
Each RPC server corresponds to **a single** backend device. For example, if there are two GPUs in a box,
then two RPC servers needs to be started, one for each GPU.

## Start a Server

Use `--serve_rpc SPEC` to start a RPC server. Examples of `SPEC`:

* `8080`: start a server on `0.0.0.0:8080` with device #0.

* `127.0.0.1:9000`: start a server on `127.0.0.1:9000` with device #0.

* `8080@1`: start a server on `0.0.0.0:8080` with device #1.

* `127.0.0.1:9000@1`: start a server on `127.0.0.1:9000` with device #1.

Don't forget to use `--show_devices` to check device IDs, and `--log_level 2` to view more logs. Example:

```sh
main --serve_rpc 80 --log_level 2
Itrying to start RPC server at 0.0.0.0:80, using
IVulkan - Vulkan0 (NVIDIA GeForce ...)
I    type: GPU
I    memory total: .. B
I    memory free : .. B
```

## Use RPC Servers

**After** RPC servers are started, they can be used. Use `--rcp_endpoints EPS` to register RPC servers (each is called an endpoint.).
Each endpoint is specified by `HOST:PORT`, and when `HOST` is omitted, `127.0.0.1` is assumed. Multiple endpoints are joined by `;`.

Use `--show_devices` to check if everything is Okay:

```sh
main --rpc_endpoints 80 --show_devices
 0: .....
 1: RPC - RPC[127.0.0.1:80] (NVIDIA GeForce ...)
    type: GPU
    memory total: .. B
    memory free : .. B
 1: CPU - CPU
    ...
```

Now, let go with distributed inference:

```sh
main --rpc_endpoints 80 -ngl 1:all -m ...
```
