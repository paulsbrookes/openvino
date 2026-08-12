#!/usr/bin/env bash

# Pre-flight calibration for the LSTM discovery harness.
#
# Run this on the runner BEFORE creating a discovery. It proves the gates and
# benchmark metrics can actually observe the goal state — the class of failure
# that cost run b35519db ~15 versions (a gate assertion that could never pass
# on real oneDNN implementation names).
#
# Stages:
#   1. Gates on the pristine baseline must pass (incl. the diagnostic test that
#      prints device identity and exec-graph implementation names).
#   2. Benchmark on the baseline must emit every expected metric, with
#      native_single_primitive == 0 (decomposition active) and
#      native_matches_decomposed == 1 (the two legs agree numerically).
#   3. With --with-probe: apply preflight-probe.patch (force native
#      preservation), rebuild, and require the benchmark to flip
#      native_single_primitive to 1 — i.e. the metrics can see the goal state.
#      Gate failures in this stage are reported, not fatal: they describe the
#      remaining implementation work, which is the discovery's job.
#      The patch is reverted afterwards.
#   4. An ONEDNN_VERBOSE header capture, so the log records the oneDNN version
#      and device the calibration ran against.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command flock
require_command git
require_command python3
require_cache

WITH_PROBE=0
if [ "${1:-}" = "--with-probe" ]; then
    WITH_PROBE=1
fi

test -x "$ARTEMIS_TEST_BINARY" || {
    echo "GPU functional test binary is missing: $ARTEMIS_TEST_BINARY" >&2
    echo "Run compile.sh first." >&2
    exit 1
}

# No workspace lock here: test.sh and benchmark invocations below take
# ARTEMIS_LOCK_FILE themselves, and holding it around their invocation would
# deadlock against them. Preflight is an operator tool — do not run it while a
# discovery is executing on this runner.

metric() {
    # metric <json-file> <name>
    python3 - "$1" "$2" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
}

run_benchmark_json() {
    # run_benchmark_json <output-json>: run the benchmark gtest and extract
    # ARTEMIS_RESULTS_JSON into the given file.
    local out_json="$1"
    local log
    log="$(mktemp "${TMPDIR:-/tmp}/artemis-preflight-bench.XXXXXX")"
    OV_GPU_USE_CM=OFF OV_GPU_USE_ONEDNN=ON "$ARTEMIS_TEST_BINARY" \
        --gtest_also_run_disabled_tests \
        --gtest_filter='ArtemisLSTMBenchmark.DISABLED_NativeVsExplicitBidirectional' \
        | tee "$log" | grep -E "ARTEMIS_DIAG|OK \]|FAILED" || true
    grep -m1 '^ARTEMIS_RESULTS_JSON=' "$log" | sed 's/^ARTEMIS_RESULTS_JSON=//' > "$out_json"
    rm -f "$log"
    test -s "$out_json" || { echo "preflight: benchmark emitted no ARTEMIS_RESULTS_JSON" >&2; exit 1; }
}

echo "== preflight stage 1: gates on the pristine baseline =="
"$HELPER_DIR/test.sh"

echo "== preflight stage 2: benchmark metric calibration =="
baseline_json="$(mktemp "${TMPDIR:-/tmp}/artemis-preflight-baseline.XXXXXX")"
run_benchmark_json "$baseline_json"
for profile in b1 b10; do
    single="$(metric "$baseline_json" "bidir_${profile}_native_single_primitive")"
    matches="$(metric "$baseline_json" "bidir_${profile}_native_matches_decomposed")"
    test "$single" = "0" || { echo "preflight: baseline ${profile} unexpectedly reports a native single primitive" >&2; exit 1; }
    test "$matches" = "1" || { echo "preflight: baseline ${profile} legs disagree numerically — benchmark reference is broken" >&2; exit 1; }
done
echo "baseline metrics OK (single_primitive=0, matches_decomposed=1)"
rm -f "$baseline_json"

if [ "$WITH_PROBE" = "1" ]; then
    echo "== preflight stage 3: goal-state observability probe =="
    probe_patch="$HELPER_DIR/preflight-probe.patch"
    test -f "$probe_patch" || { echo "preflight: missing $probe_patch" >&2; exit 1; }
    git -C "$ARTEMIS_CACHE_SOURCE" apply "$probe_patch"
    restore_probe() {
        git -C "$ARTEMIS_CACHE_SOURCE" apply -R "$probe_patch" || true
    }
    trap restore_probe EXIT

    cmake --build "$ARTEMIS_BUILD_DIR" --target ov_gpu_func_tests --parallel "${ARTEMIS_BUILD_JOBS:-$(nproc)}"

    probe_json="$(mktemp "${TMPDIR:-/tmp}/artemis-preflight-probe.XXXXXX")"
    run_benchmark_json "$probe_json"
    for profile in b1 b10; do
        single="$(metric "$probe_json" "bidir_${profile}_native_single_primitive")"
        test "$single" = "1" || { echo "preflight: probe did not flip ${profile} native_single_primitive to 1 — metrics cannot observe the goal state" >&2; exit 1; }
        echo "probe ${profile}: single_primitive=1, matches_decomposed=$(metric "$probe_json" "bidir_${profile}_native_matches_decomposed") (mismatch here = remaining implementation work, not a harness fault)"
    done
    rm -f "$probe_json"

    echo "probe gate outcomes (informational):"
    "$HELPER_DIR/test.sh" || true

    restore_probe
    trap - EXIT
    cmake --build "$ARTEMIS_BUILD_DIR" --target ov_gpu_func_tests --parallel "${ARTEMIS_BUILD_JOBS:-$(nproc)}"
fi

echo "== preflight stage 4: oneDNN environment header =="
ONEDNN_VERBOSE=1 OV_GPU_USE_CM=OFF OV_GPU_USE_ONEDNN=ON "$ARTEMIS_TEST_BINARY" \
    --gtest_filter='ArtemisLSTMDiagnostics.PrintEnvironment' 2>&1 \
    | grep -E "onednn_verbose,v?[0-9]*,?info|ARTEMIS_DIAG" || true

echo "preflight complete."
