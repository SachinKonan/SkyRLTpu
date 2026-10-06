#!/bin/bash
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32
python - <<'CHECK'
import json
from pathlib import Path
q=Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32/tpu/science/results/placement/abuplace-xplace-starts-20260921')
for variant in ('off','rudy','rudy_hv'):
 d=json.loads((q/'tasks'/f'xplace-{variant}-ibm01'/'done.json').read_text())
 r=json.loads(Path(d['report']).read_text())
 assert r.get('valid') is True and r.get('variant') == variant, (variant,r)
print('All three pilot variants independently verified',flush=True)
CHECK
if python - <<'CHECK'
import json,sys
from pathlib import Path
q=Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32/tpu/science/results/placement/abuplace-xplace-starts-20260921')
p=json.loads((q/'plan.json').read_text())
sys.exit(0 if all((q/'tasks'/(t['method']+'-'+t['case'])/'done.json').exists() for t in p['jobs']) else 1)
CHECK
then
 echo 'Queue already complete'
 exit 0
fi
exec bash /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32/tpu/science/results/placement/abuplace-xplace-starts-20260921/run.sh
