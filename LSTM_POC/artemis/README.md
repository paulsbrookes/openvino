# OpenVINO LSTM on Artemis

These helpers maintain one commit-pinned OpenVINO source and Debug build on the
runner. Each Artemis candidate is synchronized into that source tree and built
incrementally before the focused Intel GPU LSTM tests run.

## One-time setup

Run setup on the runner using the repository and exact commit imported by the
Artemis project:

```bash
./LSTM_POC/artemis/setup-workspace.sh \
  --repo-url https://github.com/michal-miotk/openvino.git \
  --commit efafc0ea790bfb7ba1620ad96db80f7f77e00b88
```

The default workspace is:

```text
~/.cache/artemis/openvino-lstm/
├── source/            commit-pinned persistent source
├── build/             persistent Debug CMake build
├── .seed-commit       source revision used during setup
└── workspace.lock     shared-workspace lock
```

Use `--cache-root PATH` or set `ARTEMIS_CACHE_ROOT` to choose another location.
Use `--jobs N` or `ARTEMIS_BUILD_JOBS` to adjust build parallelism.

A workspace is tied to one baseline. When the Artemis project moves to another
commit or repository, initialize a new cache root rather than reusing the old
build.

## Artemis commands

Configure the following commands from the project root:

```text
compile: ./LSTM_POC/artemis/compile.sh
test:    ./LSTM_POC/artemis/test.sh
benchmark: ./LSTM_POC/artemis/benchmark.sh
```

For a non-default workspace, pass the same cache root to all helpers:

```text
compile: ARTEMIS_CACHE_ROOT=/path/to/cache ./LSTM_POC/artemis/compile.sh
test:    ARTEMIS_CACHE_ROOT=/path/to/cache ./LSTM_POC/artemis/test.sh
benchmark: ARTEMIS_CACHE_ROOT=/path/to/cache ./LSTM_POC/artemis/benchmark.sh
```

The compile helper uses `rsync --checksum --no-times` to copy content changes
from the temporary Artemis checkout into the persistent `src/` tree without
changing timestamps for identical files. It then builds only
`ov_gpu_func_tests`.

The test helper runs the Discovery's LSTMSequence GPU filter and excludes the
uninstantiated suite plus the two known CM bidirectional f16 failures. Override
the filter with `ARTEMIS_LSTM_GTEST_FILTER` when narrowing a diagnostic run.
For example, this checks the native bidirectional cases relevant to the task:

```bash
ARTEMIS_LSTM_GTEST_FILTER='*LSTMSequenceGPUTest.Inference/mode=PURE_SEQ*activations=(sigmoid.tanh.tanh)_direction=bidirectional*-LSTMSequenceCM/*' \
  ./LSTM_POC/artemis/test.sh
```

The default filter is the calibrated Artemis correctness gate for the target
GPU. It covers eligible bidirectional outputs, unsupported-case fallback,
shared weights, and export/import cache parity.

The benchmark invokes a disabled-by-default `ov_gpu_func_tests` case explicitly.
For each of the f16 profiles `batch=1, sequence=2, hidden=128, input=64` and
`batch=10, sequence=20, hidden=10, input=10`, it compiles a native
bidirectional LSTM and independent forward/reverse branches consuming the
identical problem (the decomposed branches read direction slices of the native
state tensors), performs 5 warmups, and records the median of 30 synchronous
inferences. Compilation is outside the timed region. Both workloads disable the
Intel GPU CM backend and request oneDNN. The script removes stale output and
atomically writes numeric metrics to `artemis_results.json`, including the
decomposed/native latency ratio, both median latencies, both compile times, the
native LSTM primitive count, `native_single_primitive` (1 iff exactly one
LSTM_Seq primitive — the topology goal), and `native_matches_decomposed`
(1 iff the native outputs numerically match the decomposed reference — a
candidate must never score speed on wrong results).

Implementation-name strings are logged (`[ARTEMIS_DIAG] ... impl=...`) but never
asserted or turned into metrics: oneDNN's GPU RNN reports `ocl:simple:any`, so
substring checks like `find("onednn")` are false negatives by construction.
This exact bug cost discovery run b35519db most of its version budget.

`metrics-schema.json` documents the intended fitness weights: the two
`native_single_primitive` goal indicators carry 40% and the two speedups 60%.
It is documentation only — `artemis discovery create` derives the metric schema
from the benchmark output and the task text, so the weights and the
worker-metrics-only rule must be stated in the task prompt (see
`TASK_PROMPT.md`) and verified on the created run before the baseline proceeds.

## Pre-flight calibration

Before creating a discovery, run the calibration on the runner:

```bash
ARTEMIS_CACHE_ROOT=... ./LSTM_POC/artemis/preflight.sh              # stages 1-2, 4
ARTEMIS_CACHE_ROOT=... ./LSTM_POC/artemis/preflight.sh --with-probe # + goal-state observability
```

It verifies the gates pass on the pristine baseline, the benchmark emits every
expected metric with sane baseline values, and (with `--with-probe`) that
forcing native preservation via `preflight-probe.patch` flips
`native_single_primitive` to 1 — i.e. the harness can observe the goal state
it is supposed to reward. Do not run it while a discovery is executing.

## Shared-workspace constraint

The helpers lock each compile or test command, but the persistent binary can
still change between phases if multiple Artemis workers share the same cache.
Run one worker against a cache, or give each worker a separate
`ARTEMIS_CACHE_ROOT`.
