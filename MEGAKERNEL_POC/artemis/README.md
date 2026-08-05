# OpenVINO MegaKernel on Artemis

This directory makes Przemek's MegaKernel proof of concept straightforward to
evaluate with Artemis:

- the runner keeps one persistent OpenVINO build cache;
- each Artemis candidate is checksum-synced into that cache;
- `make install` recompiles only changed code;
- the existing benchmark writes numeric `artemis_results.json` metrics where
  Artemis expects them.

The scripts are repository-owned and contain no user-specific absolute paths.

## 1. Prerequisites

Use a Linux runner with:

- Git, CMake, GNU Make, GCC/G++, `rsync`, and Python 3;
- the OpenVINO build dependencies;
- an Intel GPU driver when benchmarking the GPU;
- enough disk for an OpenVINO checkout, build, install, Python environment, and
  Qwen3 0.6B model.

Clone this branch, then install any missing OpenVINO build dependencies as
appropriate for the runner.

## 2. Initialize the persistent cache once

From the repository root:

```bash
./MEGAKERNEL_POC/artemis/setup-cache.sh
```

By default, persistent state is stored under:

```text
~/.cache/artemis/openvino-megakernel/
├── source/       cached OpenVINO source
├── build/        persistent CMake/Make build tree
├── install/      installed OpenVINO runtime and setupvars.sh
├── venv/         benchmark Python environment
└── models/       converted Qwen3 0.6B OpenVINO IR
```

The initial build and model conversion are intentionally one-time operations.
They can take tens of minutes and require network access. Later candidate
builds reuse these assets.

To use another location or tune the job count:

```bash
export ARTEMIS_CACHE_ROOT=/path/on/the/runner/openvino-megakernel
export ARTEMIS_BUILD_JOBS=12
./MEGAKERNEL_POC/artemis/setup-cache.sh
```

See `artemis.env.example` for all optional overrides. Scripts can load such a
file through `ARTEMIS_ENV_FILE=/path/to/artemis.env`.

## 3. Smoke-test locally

Run these commands from the repository root:

```bash
./MEGAKERNEL_POC/artemis/compile.sh
./MEGAKERNEL_POC/artemis/benchmark.sh
```

The first compile after setup should be almost a no-op. If a supported
MegaKernel source changes, `compile.sh` copies only content changes into the
cache and runs Przemek's `make -j12 install` incrementally.

The benchmark defaults to CPU so its plumbing can be tested on any compatible
machine. It runs the native baseline and MegaKernel paths and creates:

```text
./artemis_results.json
```

The file is a flat JSON object of numeric SUMMARY-table metrics. The primary
optimization signals are the `ms_per_tok_*_megakernel` values (lower is
better); `decode_x_*` reports baseline/MegaKernel speedup (higher is better).

## 4. Select the Panther Lake GPU

List the OpenVINO devices visible through the cached runtime:

```bash
source "${ARTEMIS_CACHE_ROOT:-$HOME/.cache/artemis/openvino-megakernel}/install/setupvars.sh"
"${ARTEMIS_CACHE_ROOT:-$HOME/.cache/artemis/openvino-megakernel}/venv/bin/python" \
  -c 'import openvino as ov; print(ov.Core().available_devices)'
```

Use the exact returned device name, commonly `GPU` for the integrated GPU:

```bash
ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh
```

Do not compare the CPU smoke-test timings with Panther Lake GPU results.

## 5. Artemis commands

Install and register an Artemis runner on the same host. The cache is
runner-owned, so every Discovery version must execute on that runner.

Use these commands from the fresh checkout root:

```text
compile:   ./MEGAKERNEL_POC/artemis/compile.sh
test:      true
benchmark: ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh
```

`true` is currently a placeholder test phase for this proof of concept. The
benchmark still reports baseline/MegaKernel output agreement, but a dedicated
fast correctness test should be added before production optimization work.

Validate the exact three commands on the runner before Discovery:

```bash
artemis --output-format json changeset create --project "<project-uuid>"

artemis --output-format json changeset validate "<changeset-id>" \
  --project "<project-uuid>" \
  --version original \
  --command './MEGAKERNEL_POC/artemis/compile.sh' \
  --command 'true' \
  --command 'ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh' \
  --runner "<runner-name>" \
  --wait
```

Then launch Discovery with the same commands and an explicit catalogue model:

```bash
artemis --output-format json discovery create \
  --project "<project-uuid>" \
  --runner "<runner-name>" \
  --model "<model-uuid>" \
  --task "Improve OpenVINO Intel GPU MegaKernel decode latency. Minimize native MegaKernel milliseconds per token while preserving correctness." \
  --compile-cmd './MEGAKERNEL_POC/artemis/compile.sh' \
  --test-cmd 'true' \
  --benchmark-cmd 'ARTEMIS_DEVICE=GPU ./MEGAKERNEL_POC/artemis/benchmark.sh' \
  --versions 5 \
  --mode automatic \
  --target-files src/plugins/intel_gpu/src/graph/impls/ocl_v2/megakernel/megakernel.cpp \
  --target-files src/plugins/intel_gpu/src/graph/impls/ocl_v2/megakernel/megakernel.hpp
```

If the cache is outside the default location, prefix both compile and benchmark
commands with the same `ARTEMIS_CACHE_ROOT=/path` value.

## How incremental compilation works

Artemis evaluates every version in a fresh checkout. CMake cannot directly
reuse one build tree across different source paths, so `compile.sh` bridges the
candidate checkout and a stable runner-owned source tree:

1. Locate the current candidate from the script's repository path.
2. Checksum-compare supported MegaKernel files against the cached source.
3. Copy files only when their contents differ, without copying checkout mtimes.
4. Run `make install` in the persistent configured build tree.

The default sync surface matches the MegaKernel target files. Set
`ARTEMIS_SYNC_FULL_SRC=1` only if Discovery is allowed to edit other OpenVINO
source paths.

## Troubleshooting

- **Candidate changes have no effect:** compile through `compile.sh`; running
  `make` directly in a fixed cache never applies the fresh Artemis checkout.
- **Most of OpenVINO recompiles:** preserve destination mtimes for unchanged
  content. These scripts use `rsync --checksum --no-times`.
- **Cache loses required files:** never use a broad `rsync --delete`; Artemis
  checkouts may be thinner than the submodule-initialized cache.
- **Build appears successful after a compiler error:** do not pipe Make directly
  into `tail`. `compile.sh` records and returns Make's real exit code.
- **Artemis finds no metrics:** the file must be named exactly
  `artemis_results.json` in the task root. `benchmark.sh` validates and copies it
  there after every successful run.
- **Shell rejects `pipefail`:** Artemis starts commands through a
  non-interactive shell; the executable helpers select Bash through their
  shebangs.
- **Discovery fails after a successful baseline:** pass an explicit `--model`
  catalogue UUID.
- **A stale metric survives a failure:** `benchmark.sh` deletes both the local
  and task-root results before running and only copies validated fresh output.
