"""Add the Qwen circle-packing hedge without repackaging submitted runs."""
import copy
import hashlib
import json
from pathlib import Path

import yaml
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PROFILES = ROOT / 'tpu/swarm/ray_train/profiles'
RUN = 'math-v6e-qwen-cp26-grpo-lr15e4-s1-20260920'
output = ROOT / '.science/packages/math-v6e-hedge-20260920/qwen/cp26'
if (output / 'submission.json').exists():
    raise RuntimeError('Do not repackage a submitted run')
c = json.loads((PROFILES / 'fresh-v4-qwen-cp26-grpo-lr15e4-s1-20260919.json').read_text())
native = json.loads((PROFILES / 'fresh-v6e-qwen-rglru-grpo-lr15e4-s1-20260919-fix1.json').read_text())
bucket = 'gs://sk7524-tinker-tpu-us-east5'
c.update(run_id=RUN, root=f'~/.cache/{RUN}', accelerator='tpu-v6e-32',
         zone='us-east5-b', bucket=bucket, trainer=copy.deepcopy(native['trainer']),
         inference=copy.deepcopy(native['inference']))
c['client_env']['TTD_SICK_MARKER'] = f'/home/gcpuser/.cache/{RUN}/runs/{RUN}/ENGINE-SICK'
for field in ('hf', 'orbax', 'trainer_compile_seed'):
    c['cache'][field] = native['cache'][field]
c['cache']['inference_compile_seed'] = native['cache']['inference_compile']
for phase in ('trainer', 'inference'):
    c['cache'][phase + '_compile'] = bucket + f'/{RUN}-{phase}_compile-v1'
Config.from_dict(c).validate()
profile = PROFILES / (RUN + '.json')
profile.write_text(json.dumps(c, indent=2) + '\n')
archive, _, task = build(profile, output)
doc = yaml.safe_load(task.read_text())
audit = (ROOT / 'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
doc['run'] = "set -euo pipefail\npython3 - <<'FRESH_CLEAN_HOST'\n" + audit + '\nFRESH_CLEAN_HOST\n' + doc['run']
doc['resources']['priority'] = 100
task.write_text(yaml.safe_dump(doc, sort_keys=False))
row = dict(hardware='v6e', task='cp26', model='qwen', run_id=RUN, parallel_v4_job=1230,
           profile=str(profile.relative_to(ROOT)), pool='tpuswarm-v6e32-east5b-qwen35',
           zone='us-east5-b', priority=100, package_dir=str(output.relative_to(ROOT)),
           archive=str(archive.relative_to(ROOT)), task_yaml=str(task.relative_to(ROOT)),
           code_uri=doc['envs']['RAY_TRAIN_CODE'],
           archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
           yaml_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
(output / 'submission-manifest.json').write_text(json.dumps(row, indent=2) + '\n')
manifest = json.loads((HERE / 'jobs.json').read_text())
manifest['jobs'] = [r for r in manifest['jobs'] if r['run_id'] != RUN] + [row]
(HERE / 'jobs.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(row), flush=True)
