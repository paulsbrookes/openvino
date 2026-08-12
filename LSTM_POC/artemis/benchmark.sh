#!/usr/bin/env bash

# Benchmark native and explicitly decomposed bidirectional GPU LSTMSequence.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command flock
require_command python3
require_cache

test -x "$ARTEMIS_TEST_BINARY" || {
    echo "GPU functional test binary is missing: $ARTEMIS_TEST_BINARY" >&2
    exit 1
}

result_file="$PWD/artemis_results.json"
rm -f -- "$result_file"

mkdir -p "$ARTEMIS_CACHE_ROOT"
exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

benchmark_log="$(mktemp "${TMPDIR:-/tmp}/artemis-lstm-benchmark.XXXXXX")"
result_tmp="$(mktemp "$PWD/.artemis_results.json.XXXXXX")"
cleanup() {
    rm -f -- "$benchmark_log" "$result_tmp"
}
trap cleanup EXIT

echo "Artemis bidirectional LSTM benchmark"
echo "  binary: $ARTEMIS_TEST_BINARY"
echo "  device: GPU"

# Backend preferences are controlled by debug environment settings, not public
# compile_model properties. Set them before the GPU plugin is loaded.
export OV_GPU_USE_CM=OFF
export OV_GPU_USE_ONEDNN=ON

"$ARTEMIS_TEST_BINARY" \
    --gtest_also_run_disabled_tests \
    --gtest_filter='ArtemisLSTMBenchmark.DISABLED_NativeVsExplicitBidirectional' \
    | tee "$benchmark_log"

python3 - "$benchmark_log" "$result_tmp" <<'PY'
import json
import math
import os
import sys

log_path, output_path = sys.argv[1:]
prefix = "ARTEMIS_RESULTS_JSON="
payloads = []
with open(log_path, encoding="utf-8") as log:
    for line in log:
        if line.startswith(prefix):
            payloads.append(line[len(prefix):].strip())

if len(payloads) != 1:
    raise SystemExit(f"expected exactly one {prefix} line, found {len(payloads)}")

metrics = json.loads(payloads[0])
expected = {
    "bidir_b10_speedup_x",
    "bidir_b1_speedup_x",
    "bidir_b10_native_median_us",
    "bidir_b10_decomposed_median_us",
    "bidir_b1_native_median_us",
    "bidir_b1_decomposed_median_us",
    "bidir_b10_native_compile_us",
    "bidir_b10_decomposed_compile_us",
    "bidir_b1_native_compile_us",
    "bidir_b1_decomposed_compile_us",
    "bidir_b10_native_primitive_count",
    "bidir_b1_native_primitive_count",
    "bidir_b10_native_single_primitive",
    "bidir_b1_native_single_primitive",
    "bidir_b10_native_matches_decomposed",
    "bidir_b1_native_matches_decomposed",
}
if set(metrics) != expected:
    missing = sorted(expected - set(metrics))
    extra = sorted(set(metrics) - expected)
    raise SystemExit(f"unexpected metrics; missing={missing}, extra={extra}")

for name, value in metrics.items():
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"{name} is not a finite number")
selection_metrics = (
    "bidir_b10_native_single_primitive",
    "bidir_b1_native_single_primitive",
    "bidir_b10_native_matches_decomposed",
    "bidir_b1_native_matches_decomposed",
)
for name in selection_metrics:
    if metrics[name] not in (0, 1):
        raise SystemExit(f"{name} must be 0 or 1")

with open(output_path, "w", encoding="utf-8") as output:
    json.dump(metrics, output, sort_keys=True, separators=(",", ":"))
    output.write("\n")
    output.flush()
    os.fsync(output.fileno())
PY

mv -f -- "$result_tmp" "$result_file"
echo "wrote $result_file"
