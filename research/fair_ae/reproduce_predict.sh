#!/usr/bin/env bash
# Installed as reproduce/predict.sh in the standalone delivery.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: FAIR_PYTHON=/path/to/python bash predict.sh PUBLIC_DATA_DIR [OUTPUT_DIR]\n' >&2
  exit 2
fi
FAIR_BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FAIR_RAW="$(cd -- "$1" && pwd)"
FAIR_OUTPUT="${2:-$FAIR_BUNDLE_ROOT/rebuilt}"
FAIR_PY="${FAIR_PYTHON:-python}"
mkdir -p -- "$FAIR_OUTPUT"
FAIR_OUTPUT="$(cd -- "$FAIR_OUTPUT" && pwd)"
FAIR_TEST="$FAIR_RAW/WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy"
test -f "$FAIR_TEST"
cd -- "$FAIR_BUNDLE_ROOT"
for spec in 'v6 physics-distance 01-physical-cluster-distance' 'v2 physics 02-physical-likelihood' 'v4 summary-joint component-summary-joint' 'v5 spatial-rec component-spatial-rec'; do
  read -r FAIR_VERSION FAIR_MODEL FAIR_NAME <<< "$spec"
  "$FAIR_PY" -u export_and_record.py --code "runtimes/$FAIR_VERSION" --run "models/$FAIR_MODEL" --raw "$FAIR_RAW" --test "$FAIR_TEST" --out "$FAIR_OUTPUT/$FAIR_NAME.zip"
done
"$FAIR_PY" ensemble.py combine --manifest final-ensemble.json --zips "$FAIR_OUTPUT/02-physical-likelihood.zip" "$FAIR_OUTPUT/component-summary-joint.zip" "$FAIR_OUTPUT/component-spatial-rec.zip" --out "$FAIR_OUTPUT/03-complementary-ensemble.zip"
