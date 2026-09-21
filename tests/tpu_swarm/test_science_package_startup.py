"""Exercise exported artifacts without the checkout masking missing modules."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import yaml

from tpu.swarm.ray_train.build import build


def test_generic_builder_produces_complete_science_launch(tmp_path):
    profile = Path('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json')
    archive, uri, task = build(profile, tmp_path / 'build')
    assert archive.name == 'science-training.tar.gz'
    doc = yaml.safe_load(task.read_text())
    assert doc['envs']['RAY_TRAIN_CODE'] == uri
    assert doc['envs']['RAY_TRAIN_CODE_SHA256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert 'export SCIENCE_WORKER_ROOT="$code"' in doc['run']
    assert 'prepare_cpu_host.sh' in doc['run']
    assert 'SCIENCE_CPU_BUNDLE' in doc['envs']
    extracted = tmp_path / 'extracted'
    with tarfile.open(archive) as bundle:
        bundle.extractall(extracted, filter='data')
    # -I excludes cwd/PYTHONPATH; add only the newly exported artifact. This
    # catches missing controller AND subsequent Ray grader imports.
    script = '''
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
names = ['tpu.swarm.ray_train.controller', 'tpu.swarm.ray_train.borrowing',
         'tpu.swarm.ray_train.borrowing_phase', 'tpu.science.training_setup',
         'tpu.science.placement_ray', 'tpu.science.placement_task',
         'tpu.science.ray_cpu', 'tpu.science.seed_pool',
         'tpu.science.placement_suite_guard']
for name in names:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), name
from tpu.science.training_setup import split_roles
assert split_roles('placement', [0,1,2,3], [4,5,6,7], placement_backend='cpu', accelerator='tpu-v6e-32') == ([0,1,2,3], [4,5,6,7], None)
from tpu.science.challenge_contract import CASES
assert len(CASES) == 17
for prompt in ['placement-jax-cpu.txt', 'placement-fast-proxy-cpu-ibm17-v2.txt']:
    assert (root / 'tpu/science/prompts' / prompt).is_file()
print(json.dumps({'imports': names, 'cases': len(CASES)}))
'''
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, '-I', '-c', script, str(extracted)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])['cases'] == 17
