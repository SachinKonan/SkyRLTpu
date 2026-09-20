"""Check the exact launch archive outside the checkout import path."""
import ast, hashlib, json, os
from pathlib import Path
import subprocess, sys, tarfile, tempfile
import yaml
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN='science-circuit-v6e-qwen-borrow-supervised-20260920'
archive=HERE/'bundle/science-training.tar.gz'
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
doc=yaml.safe_load((HERE/'bundle'/f'{RUN}.yaml').read_text())
assert doc['envs']['RAY_TRAIN_CODE_SHA256']==digest
assert doc['resources']['zone']=='us-central1-b'
source=ast.parse((ROOT/'tests/tpu_swarm/test_science_package_startup.py').read_text())
script=next(n.value.value for n in ast.walk(source) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
script+='''
from tpu.swarm.ray_train import borrowing_supervisor
from tpu.swarm.ray_train.config import Config
cfg=Config.load(root/'tpu/swarm/ray_train/profiles/science-circuit-v6e-qwen-borrow-supervised-20260920.json')
assert cfg.inference.external_pool_updates and not cfg.inference.external_pool_urls
assert cfg.inference.external_pool_health_grace_seconds==90
assert cfg.inference.restart_limit==0 and cfg.max_restarts_on_errors==0
from tpu.science.placement_warm_start import verified_inputs
cases=verified_inputs(root,folder=root/'.science/placement-inputs')
assert len(cases)==17
from tpu.science.fast_proxy.build import build
build()
print(json.dumps(dict(verified_cases=sorted(cases),c_helper_compiled=True,dynamic_discovery=True,local_failure_fatal=True)))
'''
with tempfile.TemporaryDirectory(prefix='borrowing-artifact-') as directory:
 root=Path(directory)
 with tarfile.open(archive) as bundle: bundle.extractall(root,filter='data')
 p=subprocess.run([sys.executable,'-I','-c',script,str(root)],cwd=directory,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
 if p.returncode: raise RuntimeError(p.stdout+p.stderr)
 record=dict(passed=True,archive_sha256=digest,checks=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')])
 (HERE/'artifact-check.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))
