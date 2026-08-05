#!/usr/bin/env bash

# Sync an Artemis candidate into the persistent cache, then build incrementally.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command make
require_command rsync
require_command tail
require_cache

test -d "$ARTEMIS_BUILD_DIR" || {
  echo "configured build directory is missing: $ARTEMIS_BUILD_DIR" >&2
  exit 1
}
test -d "$ARTEMIS_CACHE_SOURCE/src" || {
  echo "cached OpenVINO source is missing: $ARTEMIS_CACHE_SOURCE" >&2
  exit 1
}

echo "Artemis compile: checksum-sync candidate, then make -j$ARTEMIS_BUILD_JOBS install"
echo "  candidate: $ARTEMIS_REPO_ROOT"
echo "  cache:     $ARTEMIS_CACHE_SOURCE"
echo "  build:     $ARTEMIS_BUILD_DIR"

# --checksum decides transfers by content. --no-times is equally important:
# unchanged candidate checkout mtimes must not dirty the persistent Make cache.
sync_tree() {
  local source_dir="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  rsync -a --checksum --no-times --out-format='sync %n' \
    "$source_dir/" "$target_dir/"
}

sync_file() {
  local relative_path="$1"
  test -f "$ARTEMIS_REPO_ROOT/$relative_path" || return 0
  mkdir -p "$(dirname -- "$ARTEMIS_CACHE_SOURCE/$relative_path")"
  rsync -a --checksum --no-times --out-format='sync %n' \
    "$ARTEMIS_REPO_ROOT/$relative_path" \
    "$ARTEMIS_CACHE_SOURCE/$relative_path"
}

if test "${ARTEMIS_SYNC_FULL_SRC:-0}" = "1"; then
  echo "  sync mode: full src/"
  sync_tree "$ARTEMIS_REPO_ROOT/src" "$ARTEMIS_CACHE_SOURCE/src"
else
  echo "  sync mode: megakernel paths"
  megakernel_dir="src/plugins/intel_gpu/src/graph/impls/ocl_v2/megakernel"
  test -d "$ARTEMIS_REPO_ROOT/$megakernel_dir" || {
    echo "candidate megakernel directory is missing: $megakernel_dir" >&2
    exit 1
  }
  sync_tree \
    "$ARTEMIS_REPO_ROOT/$megakernel_dir" \
    "$ARTEMIS_CACHE_SOURCE/$megakernel_dir"

  related_paths=(
    "src/plugins/intel_gpu/include/intel_gpu/primitives/megakernel.hpp"
    "src/plugins/intel_gpu/src/graph/include/megakernel_inst.h"
    "src/plugins/intel_gpu/src/graph/megakernel_inst.cpp"
    "src/plugins/intel_gpu/src/graph/registry/megakernel_impls.cpp"
    "src/plugins/intel_gpu/src/plugin/transformations/insert_megakernel.cpp"
    "src/plugins/intel_gpu/src/plugin/transformations/op/megakernel.cpp"
    "src/plugins/intel_gpu/src/plugin/ops/megakernel.cpp"
  )
  for relative_path in "${related_paths[@]}"; do
    sync_file "$relative_path"
  done
fi

log_file="$ARTEMIS_CACHE_ROOT/last-compile.log"
set +e
make -C "$ARTEMIS_BUILD_DIR" -j"$ARTEMIS_BUILD_JOBS" install >"$log_file" 2>&1
status=$?
set -e

tail -n 10 "$log_file"
if test "$status" -ne 0; then
  echo "OpenVINO build failed (exit $status); last 80 lines:" >&2
  tail -n 80 "$log_file" >&2
  exit "$status"
fi

echo "compile ok"
