#!/usr/bin/env bash

# One-time runner setup for persistent OpenVINO builds and benchmark assets.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command git
require_command cmake
require_command make
require_command python3
require_command rsync

seed_commit="$(git -C "$ARTEMIS_REPO_ROOT" rev-parse HEAD)"

echo "Initializing OpenVINO MegaKernel cache"
echo "  seed:    $seed_commit"
echo "  cache:   $ARTEMIS_CACHE_ROOT"
echo "  source:  $ARTEMIS_CACHE_SOURCE"
echo "  build:   $ARTEMIS_BUILD_DIR"
echo "  install: $ARTEMIS_INSTALL_DIR"

mkdir -p "$ARTEMIS_CACHE_ROOT"

if test ! -d "$ARTEMIS_CACHE_SOURCE/.git"; then
  test ! -e "$ARTEMIS_CACHE_SOURCE" || {
    echo "cache source exists but is not a Git checkout: $ARTEMIS_CACHE_SOURCE" >&2
    exit 1
  }
  git clone --no-hardlinks "$ARTEMIS_REPO_ROOT" "$ARTEMIS_CACHE_SOURCE"
  git -C "$ARTEMIS_CACHE_SOURCE" checkout --detach "$seed_commit"
fi

git -C "$ARTEMIS_CACHE_SOURCE" submodule update --init --recursive

cmake -S "$ARTEMIS_CACHE_SOURCE" -B "$ARTEMIS_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ARTEMIS_INSTALL_DIR" \
  -DENABLE_INTEL_GPU=ON \
  -DENABLE_PYTHON=ON \
  -DENABLE_SAMPLES=OFF \
  -DENABLE_TESTS=OFF \
  -DENABLE_DATA=OFF

make -C "$ARTEMIS_BUILD_DIR" -j"$ARTEMIS_BUILD_JOBS" install

if test ! -x "$ARTEMIS_VENV_DIR/bin/python"; then
  python3 -m venv "$ARTEMIS_VENV_DIR"
fi

"$ARTEMIS_VENV_DIR/bin/python" -m pip install --upgrade pip wheel
"$ARTEMIS_VENV_DIR/bin/python" -m pip install \
  numpy transformers accelerate "optimum-intel[openvino]"

if ! compgen -G "$ARTEMIS_MODEL_DIR/*.xml" >/dev/null; then
  mkdir -p "$(dirname -- "$ARTEMIS_MODEL_DIR")"
  # shellcheck source=/dev/null
  set +u
  source "$ARTEMIS_INSTALL_DIR/setupvars.sh"
  set -u
  PATH="$ARTEMIS_VENV_DIR/bin:$PATH" "$ARTEMIS_VENV_DIR/bin/python" \
    "$ARTEMIS_REPO_ROOT/MEGAKERNEL_POC/python/convert_to_openvino_ir.py" \
    --model-id Qwen/Qwen3-0.6B \
    --output-dir "$ARTEMIS_MODEL_DIR" \
    --weight-format fp16
fi

printf '%s\n' "$seed_commit" >"$ARTEMIS_CACHE_MARKER"

echo
echo "Cache ready. Run:"
echo "  $HELPER_DIR/compile.sh"
echo "  $HELPER_DIR/benchmark.sh"
