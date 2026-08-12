#!/usr/bin/env bash

# Shared defaults for the OpenVINO LSTM Artemis helpers.

ARTEMIS_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTEMIS_REPO_ROOT="$(cd -- "$ARTEMIS_HELPER_DIR/../.." && pwd)"

ARTEMIS_CACHE_ROOT="${ARTEMIS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/artemis/openvino-lstm}"
ARTEMIS_CACHE_SOURCE="${ARTEMIS_CACHE_SOURCE:-$ARTEMIS_CACHE_ROOT/source}"
ARTEMIS_BUILD_DIR="${ARTEMIS_BUILD_DIR:-$ARTEMIS_CACHE_ROOT/build}"
ARTEMIS_BUILD_JOBS="${ARTEMIS_BUILD_JOBS:-16}"
ARTEMIS_CACHE_MARKER="${ARTEMIS_CACHE_MARKER:-$ARTEMIS_CACHE_ROOT/.seed-commit}"
ARTEMIS_LOCK_FILE="${ARTEMIS_LOCK_FILE:-$ARTEMIS_CACHE_ROOT/workspace.lock}"
ARTEMIS_TEST_BINARY="${ARTEMIS_TEST_BINARY:-$ARTEMIS_CACHE_SOURCE/bin/intel64/Debug/ov_gpu_func_tests}"

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "required command not found: $1" >&2
        exit 1
    }
}

require_cache() {
    test -f "$ARTEMIS_CACHE_MARKER" || {
        echo "OpenVINO LSTM workspace is not initialized: $ARTEMIS_CACHE_ROOT" >&2
        echo "Run $ARTEMIS_HELPER_DIR/setup-workspace.sh first." >&2
        exit 1
    }
    test -d "$ARTEMIS_CACHE_SOURCE/.git" || {
        echo "cached source is not a Git checkout: $ARTEMIS_CACHE_SOURCE" >&2
        exit 1
    }
    test -d "$ARTEMIS_BUILD_DIR" || {
        echo "configured build directory is missing: $ARTEMIS_BUILD_DIR" >&2
        exit 1
    }

    local seed_commit source_commit
    seed_commit="$(<"$ARTEMIS_CACHE_MARKER")"
    source_commit="$(git -C "$ARTEMIS_CACHE_SOURCE" rev-parse HEAD)"
    test "$source_commit" = "$seed_commit" || {
        echo "cached source HEAD does not match its seed marker" >&2
        echo "  marker: $seed_commit" >&2
        echo "  source: $source_commit" >&2
        exit 1
    }
}
