# OpenVINO MegaKernel on Artemis

Run these helpers on the same machine as your Artemis runner so Discovery can reuse that local cache.

## 1. One-time setup

From the repository root:

```bash
./MEGAKERNEL_POC/artemis/setup-cache.sh
```

This clones/configures a persistent OpenVINO tree, runs the initial `make install`, creates a Python venv, and converts the Qwen3 0.6B model.

Default cache location (shared by `setup-cache.sh`, `compile.sh`, and `benchmark.sh` via `common.sh`):

```text
~/.cache/artemis/openvino-megakernel/
├── source/    cached OpenVINO source
├── build/     persistent CMake/Make build tree
├── install/   installed runtime + setupvars.sh
├── venv/      benchmark Python environment
└── models/    converted Qwen3 IR
```

Override with `ARTEMIS_CACHE_ROOT=/other/path` if needed. Use the same value for setup, compile, and benchmark.

## 2. Compile (cache + rsync)

Artemis command:

```text
./MEGAKERNEL_POC/artemis/compile.sh
```

Each Discovery version is downloaded in a temporary directory by the runner. `compile.sh` syncs MegaKernel sources from the downloaded version into the cache using:

```text
rsync -a --checksum --no-times
```

Only content changes are copied; destination mtimes are preserved so Make stays incremental. Then it runs `make install` in the cached build directory.

## 3. Benchmark + metrics

Artemis command (GPU example):

```text
ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh
```

`ARTEMIS_DEVICE` is a wrapper flag (default `CPU`). It is forwarded to `e2e_performance_measurement.py --device …`. Use an OpenVINO device name such as `CPU`, `GPU`, or `GPU.1`.

The Python benchmark writes a flat numeric `artemis_results.json` (SUMMARY table values). `benchmark.sh` copies that file to the Artemis task root so the platform can pick it up. Primary ranking signals:

- `ms_per_tok_*_megakernel` — lower is better
- `decode_x_*` — higher is better

## 4. Example Artemis commands

```text
compile:   ./MEGAKERNEL_POC/artemis/compile.sh
test:      true
benchmark: ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh
```

If you set a custom cache root:

```text
compile:   ARTEMIS_CACHE_ROOT=/path/to/cache ./MEGAKERNEL_POC/artemis/compile.sh
benchmark: ARTEMIS_CACHE_ROOT=/path/to/cache ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh
```
