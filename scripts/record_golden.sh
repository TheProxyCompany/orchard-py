#!/usr/bin/env bash
# Re-record the golden files of ONE architecture with ONLY that model loaded.
#
# The normal test session preloads the whole chat matrix (tests/models.py), far
# more memory than a recording needs. This narrows the session to one row with
# ORCHARD_TEST_MODELS, turns on GOLDEN_RECORD so recorded turns are replaced
# instead of asserted, and selects only that model's golden test. Files are
# written only if every case of the model passes; review the git diff after.
#
# Usage:
#   ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> \
#     scripts/record_golden.sh <template_type>      # e.g. gpt_oss
#   ... scripts/record_golden.sh pipeline           # last; see tests/golden/README.md
#
# Both variables are required: PIE_LOCAL_BUILD so the recording comes from the
# engine build you mean (not whatever is installed), ORCHARD_CACHE_ROOT so the
# session gets its own engine namespace and cannot stop an engine already
# running on this machine.
set -euo pipefail

TARGET="${1:?usage: scripts/record_golden.sh <template_type>|pipeline}"
: "${PIE_LOCAL_BUILD:?set PIE_LOCAL_BUILD to the engine release directory to record against}"
: "${ORCHARD_CACHE_ROOT:?set ORCHARD_CACHE_ROOT to a fresh directory for this recording}"

cd "$(dirname "$0")/.."
mkdir -p "$ORCHARD_CACHE_ROOT"
export ORCHARD_TEST_LOG_DIR="${ORCHARD_TEST_LOG_DIR:-$ORCHARD_CACHE_ROOT/logs_test}"
export GOLDEN_RECORD=1

if [ "$TARGET" = "pipeline" ]; then
  # The image cases chain gemma4 -> image models -> moondream3; the test itself
  # hydrates the image and audio tool models one at a time.
  export ORCHARD_TEST_MODELS="gemma4,moondream3"
  TEST="tests/golden/test_matrix.py::test_golden_pipeline_cases"
else
  export ORCHARD_TEST_MODELS="$TARGET"
  TEST="tests/golden/test_matrix.py::test_golden_correctness_for_model"
fi

# Fail on an unknown name here, before pytest starts anything.
python -c "import tests.models"

python -m pytest "$TEST" -q -s -p no:cacheprovider
git status --short tests/golden/data
