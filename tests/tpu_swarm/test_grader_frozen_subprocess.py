"""Exercise the production subprocess from the frozen source, without repo fallback."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tpu.swarm.ray_train.overlay import ARENA_FILES


def test_frozen_source_runs_real_pregate_child(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("recurrentgemma")
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / 'source'
    for name in ARENA_FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
    request, output = tmp_path / 'request.json', tmp_path / 'result.json'
    request.write_text(json.dumps(dict(mode='pregate', payload=dict(code='def kernel(x,a,reset): return x'))))
    env = dict(os.environ, PYTHONPATH=f'{source / "tpu"}:{source}', JAX_PLATFORMS='cpu',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', ARENA_RLIMIT_GB='16')
    command = [sys.executable, '-m', 'tpu.swarm.ray_train.grader_child', str(request), str(output)]
    # First reproduce the deployed failure by removing the missing module.
    module = source / 'tpu/pallas_arena/judge/ray_pool.py'
    module.rename(module.with_suffix('.disabled'))
    failed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=150)
    assert failed.returncode != 0
    assert "No module named 'pallas_arena.judge.ray_pool'" in failed.stderr
    module.with_suffix('.disabled').rename(module)
    fixed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=150)
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    result = json.loads(output.read_text())
    assert not result['passed']
    assert any('pallas_call' in v for v in result['violations']), result
