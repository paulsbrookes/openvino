#!/usr/bin/env bash

# Shared defaults for the OpenVINO MegaKernel Artemis helpers.

ARTEMIS_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTEMIS_REPO_ROOT="$(cd -- "$ARTEMIS_HELPER_DIR/../.." && pwd)"

if test -n "${ARTEMIS_ENV_FILE:-}"; then
  test -f "$ARTEMIS_ENV_FILE" || {
    echo "ARTEMIS_ENV_FILE does not exist: $ARTEMIS_ENV_FILE" >&2
    exit 1
  }
  # shellcheck source=/dev/null
  source "$ARTEMIS_ENV_FILE"
fi

ARTEMIS_CACHE_ROOT="${ARTEMIS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/artemis/openvino-megakernel}"
ARTEMIS_CACHE_SOURCE="${ARTEMIS_CACHE_SOURCE:-$ARTEMIS_CACHE_ROOT/source}"
ARTEMIS_BUILD_DIR="${ARTEMIS_BUILD_DIR:-$ARTEMIS_CACHE_ROOT/build}"
ARTEMIS_INSTALL_DIR="${ARTEMIS_INSTALL_DIR:-$ARTEMIS_CACHE_ROOT/install}"
ARTEMIS_VENV_DIR="${ARTEMIS_VENV_DIR:-$ARTEMIS_CACHE_ROOT/venv}"
ARTEMIS_MODEL_DIR="${ARTEMIS_MODEL_DIR:-$ARTEMIS_CACHE_ROOT/models/qwen3-0.6b-openvino-ir}"
ARTEMIS_BUILD_JOBS="${ARTEMIS_BUILD_JOBS:-12}"
ARTEMIS_DEVICE="${ARTEMIS_DEVICE:-CPU}"
ARTEMIS_CACHE_MARKER="${ARTEMIS_CACHE_MARKER:-$ARTEMIS_CACHE_ROOT/.artemis-openvino-megakernel-cache}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

require_cache() {
  test -f "$ARTEMIS_CACHE_MARKER" || {
    echo "OpenVINO cache is not initialized: $ARTEMIS_CACHE_ROOT" >&2
    echo "Run $ARTEMIS_HELPER_DIR/setup-cache.sh once on this runner." >&2
    exit 1
  }
}
