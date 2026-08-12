#!/usr/bin/env bash

# Run the focused Intel GPU LSTMSequence correctness gate.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command flock
require_command git
require_cache

test -x "$ARTEMIS_TEST_BINARY" || {
    echo "GPU functional test binary is missing: $ARTEMIS_TEST_BINARY" >&2
    exit 1
}

mkdir -p "$ARTEMIS_CACHE_ROOT"
exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

default_filter='ArtemisLSTMDiagnostics.PrintEnvironment:smoke_LSTMSequenceBidirectionalOneDNNCorrectness/*:smoke_LSTMSequenceOneDNNFallback/*:smoke_MultipleBidirectionalLSTMSequenceSharedWeightsOneDNN/*:LSTMSequenceTest.smoke_BidirectionalExportImportCacheInferenceParity'
gtest_filter="${ARTEMIS_LSTM_GTEST_FILTER:-$default_filter}"

echo "Artemis LSTM correctness test"
echo "  binary: $ARTEMIS_TEST_BINARY"
echo "  device: GPU"

"$ARTEMIS_TEST_BINARY" --gtest_filter="$gtest_filter"
