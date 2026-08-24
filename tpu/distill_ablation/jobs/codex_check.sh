#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB"
"$PY" - <<'PYEOF'
import sys, uuid
from pathlib import Path
sys.path.insert(0,".")
from run_executor import codex_execute
base=Path("/n/fs/vision-mix/sk7524/SkyRLTpu/runs/distill_ablation/_exec/_check")/uuid.uuid4().hex[:6]
prompt=("Write ONLY a file named solution.py in the current directory defining "
        "run(seed=42, budget_s=1000, **kwargs) that returns (a python list of 10 floats each equal to 0.5, 0.5, 10). "
        "Keep it minimal; do not run anything.")
for model in ["gpt-5.6-luna","gpt-5.6-terra","gpt-5.4-mini"]:
    code,err=codex_execute(prompt, model, base/model.replace(".","_").replace("-","_"), 240)
    print(f"{model}: got_code={code is not None} len={len(code or '')} err={(err or '')[:180]}")
PYEOF
