"""The shipped sliding-window KV patch turns the base bundle's three files into the hardware-tested ones."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train.swa_kv import install as swa_kv

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'third_party/tpu-inference'
FILES = ('tpu_inference/models/vllm/muse_glimmer.py', 'tpu_inference/runner/kv_cache_manager.py',
         'tpu_inference/worker/tpu_worker.py')


def show(commit, name):
    return subprocess.run(['git', '-C', str(CORE), 'show', f'{commit}:{name}'], check=True,
                          capture_output=True, text=True).stdout


@pytest.mark.skipif(not (CORE / '.git').exists(), reason='tpu-inference submodule not checked out')
def test_patch_applies_without_fuzz_and_yields_the_tested_files(tmp_path):
    for name in FILES:
        path = tmp_path / 'third_party/tpu-inference' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(show('acfe1f056', name))
    swa_kv.install(tmp_path)
    for name in FILES:
        assert (tmp_path / 'third_party/tpu-inference' / name).read_text() == show('a4fb045bf', name)
    with pytest.raises(subprocess.CalledProcessError):  # never applied twice
        swa_kv.install(tmp_path)


def test_enabled_only_for_the_explicit_flag():
    config = lambda env: SimpleNamespace(inference=SimpleNamespace(engine_env=env))
    assert swa_kv.enabled(config({'SKYRL_SWA_KV': '1'}))
    assert not swa_kv.enabled(config({'SKYRL_SWA_KV': '0'})) and not swa_kv.enabled(config({}))
