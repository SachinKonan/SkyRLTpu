"""Exercise tpu/fair/ray_task.py without Ray: stage() from the bucket, then grade() in fresh child processes."""
import json
import sys
import time
import types
from pathlib import Path

ray = types.ModuleType('ray')
ray.remote = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
sys.modules['ray'] = ray
sys.path.insert(0, str(Path(__file__).parent / 'tpu' / 'fair'))
import ray_task as RT  # noqa: E402

chip, stack, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
names = sys.argv[4:]
RT._chip = lambda: chip
ex = Path(__file__).parent / 'tpu' / 'fair' / 'tpu_grader' / 'examples'
t = time.time()
rec = RT.stage()
print(json.dumps(dict(chip=chip, stage=rec, stage_wall=round(time.time() - t, 1))), flush=True)
for n in names:
    t = time.time()
    r = RT.grade(dict(source=(ex / f'{n}.py').read_text(), mode=mode, stack=stack))
    print(json.dumps(dict(chip=chip, cand=n, stack=stack, mode=mode, ok=r.get('ok'), reward=r.get('reward'),
                          hidden_in_public='hidden' in r, hidden_ok=r.get('hidden_ok'),
                          fail=(r.get('contract') or {}).get('fail') or r.get('error'),
                          wall=r['metrics']['wall_s'], inner=(r.get('runtime') or {}).get('total_s'),
                          extract=(r.get('runtime') or {}).get('extract_s'), fit=(r.get('runtime') or {}).get('fit_s'))),
          flush=True)
