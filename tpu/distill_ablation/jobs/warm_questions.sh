#!/bin/bash
# Pre-warm the cached problem statements for the frontier problems (env lives in another
# discover root, so it is fetched via cdc_prompt.get_question in a subprocess).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB/portfolio"
"$PY" -u - <<'PY'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')
from sweep import task_text
for p in ("fc46","fc302"):
    try:
        q = task_text(p)
        print(f"{p}: {len(q)} chars cached")
        print("   " + q[:220].replace("\n"," ") + " ...")
    except Exception as e:
        print(f"{p}: FAILED {type(e).__name__}: {e}")
PY
