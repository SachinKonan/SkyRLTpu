"""Queue the upstream-Xplace start tasks: every IBM case x every config in xplace_start.CONFIGS."""
import json
from pathlib import Path
import sys

F = Path('/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval')
W = Path(__file__).resolve().parents[3]
PROBE = '/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32/.science/challenge-probe'
sys.path.insert(0, str(Path(__file__).parent))
from xplace_start import CONFIGS  # noqa: E402

queue = Path(sys.argv[1]) if len(sys.argv) > 1 else F / 'queue'
(queue / 'pending').mkdir(parents=True, exist_ok=True)
cases = [f'ibm{i:02d}' for i in range(1, 19) if i != 5]
for case in cases:
    for cfg in CONFIGS:
        tid = f'start-{case}-{cfg}'
        if any((queue / d / f'{tid}.json').exists() for d in ('pending', 'running', 'done', 'failed')):
            continue
        spec = dict(id=tid, kind='xplace_start', case=case, config=cfg,
                    cmd=[str(F / 'env/cuda/bin/python'), '-u', str(W / 'tpu/science/fair_eval/xplace_start.py'),
                         '--case', case, '--config', cfg, '--out', str(F / 'runs/starts' / case / cfg),
                         '--xplace', str(F / 'src/Xplace'), '--probe', PROBE, '--threads', '12'],
                    cwd=str(W), est_seconds=420, timeout_seconds=1500, gpus=1, cpus=12, memory_gb=150, priority=10)
        (queue / 'pending' / f'{tid}.json').write_text(json.dumps(spec, indent=1))
print(len(list((queue / 'pending').glob('*.json'))), 'pending')
