#!/usr/bin/env bash

# Sync an Artemis candidate into the persistent workspace and build it.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command cmake
require_command flock
require_command git
require_command rsync
require_command tail
require_cache

test -d "$ARTEMIS_REPO_ROOT/src" || {
    echo "candidate OpenVINO source is missing: $ARTEMIS_REPO_ROOT/src" >&2
    exit 1
}

seed_commit="$(<"$ARTEMIS_CACHE_MARKER")"
if git -C "$ARTEMIS_REPO_ROOT" cat-file -e "$seed_commit^{commit}" 2>/dev/null; then
    git -C "$ARTEMIS_REPO_ROOT" merge-base --is-ancestor "$seed_commit" HEAD || {
        echo "candidate is not based on the persistent workspace seed" >&2
        echo "  seed:      $seed_commit" >&2
        echo "  candidate: $(git -C "$ARTEMIS_REPO_ROOT" rev-parse HEAD)" >&2
        exit 1
    }
else
    echo "warning: candidate checkout does not contain seed commit $seed_commit" >&2
fi

mkdir -p "$ARTEMIS_CACHE_ROOT"
exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

echo "Artemis LSTM compile"
echo "  candidate: $ARTEMIS_REPO_ROOT"
echo "  seed:      $seed_commit"
echo "  cache:     $ARTEMIS_CACHE_SOURCE"
echo "  build:     $ARTEMIS_BUILD_DIR"

# Checksum comparison avoids copying unchanged files. Preserving destination
# mtimes keeps CMake's persistent build incremental across candidate checkouts.
rsync -a --checksum --no-times --out-format='sync %n' \
    "$ARTEMIS_REPO_ROOT/src/" "$ARTEMIS_CACHE_SOURCE/src/"

log_file="$ARTEMIS_CACHE_ROOT/last-compile.log"
set +e
cmake --build "$ARTEMIS_BUILD_DIR" \
    --target ov_gpu_func_tests \
    --parallel "$ARTEMIS_BUILD_JOBS" >"$log_file" 2>&1
status=$?
set -e

tail -n 20 "$log_file"
if test "$status" -ne 0; then
    echo "OpenVINO LSTM build failed (exit $status); last 120 lines:" >&2
    tail -n 120 "$log_file" >&2
    exit "$status"
fi

echo "compile ok"
