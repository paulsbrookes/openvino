#!/usr/bin/env bash

# Create a commit-pinned persistent source and Debug build for Artemis.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  setup-workspace.sh --repo-url URL --commit REV [--cache-root PATH] [--jobs N]

The repository URL and revision identify the exact source baseline that the
Artemis candidates must use. Choose a new cache root when changing baselines.
EOF
}

repo_url=""
revision=""
ARTEMIS_BUILD_JOBS="${ARTEMIS_BUILD_JOBS:-16}"
while test "$#" -gt 0; do
    case "$1" in
        --repo-url)
            test "$#" -ge 2 || { usage >&2; exit 2; }
            repo_url="$2"
            shift 2
            ;;
        --commit)
            test "$#" -ge 2 || { usage >&2; exit 2; }
            revision="$2"
            shift 2
            ;;
        --cache-root)
            test "$#" -ge 2 || { usage >&2; exit 2; }
            ARTEMIS_CACHE_ROOT="$2"
            export ARTEMIS_CACHE_ROOT
            shift 2
            ;;
        --jobs)
            test "$#" -ge 2 || { usage >&2; exit 2; }
            ARTEMIS_BUILD_JOBS="$2"
            export ARTEMIS_BUILD_JOBS
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

test -n "$repo_url" || { echo "--repo-url is required" >&2; exit 2; }
test -n "$revision" || { echo "--commit is required" >&2; exit 2; }
case "$ARTEMIS_BUILD_JOBS" in
    ''|*[!0-9]*) echo "--jobs must be a positive integer" >&2; exit 2 ;;
    0) echo "--jobs must be greater than zero" >&2; exit 2 ;;
esac

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command cmake
require_command flock
require_command git

mkdir -p "$ARTEMIS_CACHE_ROOT"
exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

if test -e "$ARTEMIS_CACHE_SOURCE"; then
    test -d "$ARTEMIS_CACHE_SOURCE/.git" || {
        echo "cache source exists but is not a Git checkout: $ARTEMIS_CACHE_SOURCE" >&2
        exit 1
    }

    requested_commit="$(git -C "$ARTEMIS_CACHE_SOURCE" rev-parse "$revision^{commit}" 2>/dev/null || true)"
    source_commit="$(git -C "$ARTEMIS_CACHE_SOURCE" rev-parse HEAD)"
    test -n "$requested_commit" && test "$requested_commit" = "$source_commit" || {
        echo "workspace source is already pinned to a different commit: $source_commit" >&2
        echo "Use a new --cache-root for revision $revision." >&2
        exit 1
    }
    if test -f "$ARTEMIS_CACHE_MARKER"; then
        seed_commit="$(<"$ARTEMIS_CACHE_MARKER")"
        test "$seed_commit" = "$source_commit" || {
            echo "workspace marker does not match source HEAD: $seed_commit" >&2
            exit 1
        }
    else
        seed_commit="$source_commit"
    fi
    test -z "$(git -C "$ARTEMIS_CACHE_SOURCE" status --porcelain)" || {
        echo "cached source contains candidate changes; use a new --cache-root for setup" >&2
        exit 1
    }
else
    echo "Cloning $repo_url"
    git clone --filter=blob:none --no-checkout "$repo_url" "$ARTEMIS_CACHE_SOURCE"
    git -C "$ARTEMIS_CACHE_SOURCE" fetch --no-tags origin "$revision"
    git -C "$ARTEMIS_CACHE_SOURCE" checkout --detach FETCH_HEAD
    git -C "$ARTEMIS_CACHE_SOURCE" submodule update --init --recursive
    seed_commit="$(git -C "$ARTEMIS_CACHE_SOURCE" rev-parse HEAD)"
fi

echo "Configuring OpenVINO LSTM workspace"
echo "  repository: $repo_url"
echo "  seed:       $seed_commit"
echo "  source:     $ARTEMIS_CACHE_SOURCE"
echo "  build:      $ARTEMIS_BUILD_DIR"
echo "  jobs:       $ARTEMIS_BUILD_JOBS"

cmake -S "$ARTEMIS_CACHE_SOURCE" -B "$ARTEMIS_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_INTEL_GPU=ON \
    -DENABLE_TESTS=ON \
    -DENABLE_PYTHON=OFF \
    -DENABLE_SAMPLES=OFF

cmake --build "$ARTEMIS_BUILD_DIR" \
    --target ov_gpu_func_tests \
    --parallel "$ARTEMIS_BUILD_JOBS"

printf '%s\n' "$seed_commit" >"$ARTEMIS_CACHE_MARKER"

echo "LSTM workspace ready."
echo "Run:"
echo "  $HELPER_DIR/compile.sh"
echo "  $HELPER_DIR/test.sh"
echo "  $HELPER_DIR/benchmark.sh"
