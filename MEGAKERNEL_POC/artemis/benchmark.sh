#!/usr/bin/env bash

# Run the MegaKernel benchmark and place metrics in the Artemis task directory.

set -euo pipefail

invocation_dir="$PWD"
HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_cache

python_bin="$ARTEMIS_VENV_DIR/bin/python"
setupvars="$ARTEMIS_INSTALL_DIR/setupvars.sh"
python_dir="$ARTEMIS_REPO_ROOT/MEGAKERNEL_POC/python"
results_file="$python_dir/artemis_results.json"
task_root="${ARTEMIS_TASK_ROOT:-$invocation_dir}"
task_results="$task_root/artemis_results.json"

test -x "$python_bin" || {
  echo "cache Python environment is missing: $python_bin" >&2
  exit 1
}
test -f "$setupvars" || {
  echo "OpenVINO setupvars.sh is missing: $setupvars" >&2
  exit 1
}
test -d "$ARTEMIS_MODEL_DIR" || {
  echo "converted model is missing: $ARTEMIS_MODEL_DIR" >&2
  exit 1
}
test -d "$task_root" || {
  echo "Artemis task directory is missing: $task_root" >&2
  exit 1
}

rm -f \
  "$results_file" \
  "$python_dir/artemis_results.json.tmp" \
  "$python_dir/artemis_results.csv" \
  "$task_results"

# shellcheck source=/dev/null
set +u
source "$setupvars" >/dev/null
set -u

echo "Artemis benchmark"
echo "  device:  $ARTEMIS_DEVICE"
echo "  model:   $ARTEMIS_MODEL_DIR"
echo "  results: $task_results"

(
  cd "$python_dir"
  "$python_bin" e2e_performance_measurement.py \
    --model-dir "$ARTEMIS_MODEL_DIR" \
    --device "$ARTEMIS_DEVICE" \
    --only-framework native \
    --gen-warmup "${ARTEMIS_GEN_WARMUP:-1}" \
    --gen-iters "${ARTEMIS_GEN_ITERS:-3}" \
    --tokens "${ARTEMIS_TOKENS:-300}" \
    "$@"
)

test -f "$results_file" || {
  echo "benchmark completed without creating $results_file" >&2
  exit 1
}

"$python_bin" - "$results_file" "$task_results" <<'PY'
import json
import math
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
metrics = json.loads(source.read_text(encoding="utf-8"))

if not isinstance(metrics, dict) or not metrics:
    raise SystemExit("artemis_results.json must be a non-empty JSON object")
invalid = {
    key: value
    for key, value in metrics.items()
    if isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not math.isfinite(value)
}
if invalid:
    raise SystemExit(f"metrics must be finite numbers; invalid values: {invalid}")

destination.parent.mkdir(parents=True, exist_ok=True)
if source.resolve() != destination.resolve():
    shutil.copyfile(source, destination)
print(destination.read_text(encoding="utf-8"), end="")
PY

echo "benchmark ok"
