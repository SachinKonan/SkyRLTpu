#!/usr/bin/env bash
# Durable completion probe for the GPT-OSS 120B v6e-32 acceptance task.
set -euo pipefail

: "${SMOKE_RESULT_GCS:?SMOKE_RESULT_GCS is required}"
gcloud storage objects describe "$SMOKE_RESULT_GCS" >/dev/null 2>&1
