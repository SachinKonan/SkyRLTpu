"""Farm and circuit-run placement preparation follows the declared per-case CPUs."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from tpu.science import placement_resources as r
from tpu.science.package_training import science_preparation
from tpu.swarm.ray_train.config import Config

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / 'tpu/swarm/ray_train/profiles'
FARMS = sorted(PROFILES.glob('inference-farm-rq4-*-east5*.json'))
RUNS = sorted(PROFILES.glob('rq4-notrain-local-*circuit*.json'))
HOST_CPUS = {'tpu-v5p-32': 208, 'tpu-v6e-8': 180}


def prepare(tmp_path, monkeypatch, env, cpus):
    """Run tpu.science.prepare_placement_host with env on a fake host of `cpus` CPUs."""
    (tmp_path / 'tpu').symlink_to(ROOT / 'tpu')
    for name, lock in ((env.get('SCIENCE_PLACEMENT_VENV', 'venv'), 'requirements-challenge-pilot.lock'),
                       ('candidate-venv', 'requirements-placement-candidate.lock')):
        venv = tmp_path / '.science' / name
        venv.mkdir(parents=True, exist_ok=True)
        (venv / '.placement-lock').write_text(hashlib.sha256((ROOT / 'tpu/science' / lock).read_bytes()).hexdigest())
    monkeypatch.chdir(tmp_path)
    for key in [k for k in os.environ if k.startswith('SCIENCE_')]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('SKYPILOT_NODE_RANK', '0')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(cpus)))
    read = Path.read_text
    meminfo = 'MemTotal: 470000000 kB\nMemAvailable: 460000000 kB\n'
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **k: meminfo if str(self) == '/proc/meminfo' else read(self, *a, **k))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: None)
    import tpu.science.fast_proxy_deployment as deployment
    monkeypatch.setattr(deployment, 'verified_mounts', lambda root: (None, {}))
    runpy.run_module('tpu.science.prepare_placement_host', run_name='__main__')
    return json.loads((tmp_path / '.science/ready.json').read_text())


@pytest.mark.parametrize('profile', FARMS, ids=lambda p: p.stem)
def test_farm_prepares_placement_from_its_family(profile, tmp_path, monkeypatch):
    config = Config.load(profile)
    config.validate()
    family = config.grading_families['placement']
    env, run = science_preparation(config)
    assert 'prepare_placement_host' in run
    assert (env['SCIENCE_PLACEMENT_SLOTS_PER_HOST'], env['SCIENCE_PLACEMENT_CPUS_PER_CASE'],
            env['SCIENCE_PLACEMENT_MEMORY_GIB']) == tuple(str(family[k]) for k in ('slots_per_host', 'cpus', 'memory_gib'))
    cpus = HOST_CPUS[config.accelerator]
    ready = prepare(tmp_path, monkeypatch, env, cpus)
    assert ready['profile'] == 'placement_jax_cpu_v1'
    grading, service = r.partition(family['slots_per_host'], cpus_per_case=family['cpus'])
    assert len(grading) == family['slots_per_host'] * family['cpus'] and len(service) >= 32
    # The readiness probe and every graded case use the family's contract.
    r.validate(r.contract(family['slots_per_host'], cpus=family['cpus'], memory_gib=family['memory_gib']))
    # Services stay on the farm's core-pool service block.
    from tpu.science.core_pool import layout
    pool, services, _ = layout(config.grading_service_cpus, affinity=set(range(cpus)), topology_root=str(tmp_path / 'none'))
    assert len(pool) + len(services) == cpus


def test_v5p_farm_two_cpu_family_at_80_slots(tmp_path, monkeypatch):
    env = dict(SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_RUNTIME=r.VERSION,
               SCIENCE_PLACEMENT_HELPER='fast_proxy_v1', SCIENCE_PLACEMENT_SLOTS_PER_HOST='80',
               SCIENCE_PLACEMENT_CPUS_PER_CASE='2', SCIENCE_PLACEMENT_MEMORY_GIB='2',
               PLACEMENT_TPU_RANKS='0,1,2,3', SCIENCE_ACCELERATOR='tpu-v5p-32')
    assert prepare(tmp_path, monkeypatch, env, 208)['rank'] == 0


def test_four_cpu_contract_still_bounded(tmp_path, monkeypatch):
    env = dict(SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_RUNTIME=r.VERSION,
               SCIENCE_PLACEMENT_SLOTS_PER_HOST='80', PLACEMENT_TPU_RANKS='0,1,2,3')
    with pytest.raises(ValueError, match='exceed 192'):
        prepare(tmp_path, monkeypatch, env, 208)


def test_legacy_four_cpu_profile_prepares(tmp_path, monkeypatch):
    env = dict(SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_RUNTIME=r.VERSION,
               SCIENCE_PLACEMENT_SLOTS_PER_HOST='40', PLACEMENT_TPU_RANKS='0,1,2,3')
    assert prepare(tmp_path, monkeypatch, env, 208)['rank'] == 0


@pytest.mark.parametrize('profile', RUNS, ids=lambda p: p.stem)
def test_circuit_run_env_prepares_and_sends_its_contract(profile, tmp_path, monkeypatch):
    from tpu.swarm.ray_train.commands import client_environment
    config = Config.load(profile)
    config.validate()
    client = client_environment(config, tmp_path, '127.0.0.1')
    expected = r.contract(config.science_placement_slots_per_host, cpus=config.science_placement_cpus_per_case,
                          memory_gib=config.science_placement_memory_gib)
    env, _ = science_preparation(config)
    assert {k: client[k] for k in env if k.startswith('SCIENCE_PLACEMENT_') and k != 'SCIENCE_PLACEMENT_VENV'} == \
        {k: v for k, v in env.items() if k.startswith('SCIENCE_PLACEMENT_') and k != 'SCIENCE_PLACEMENT_VENV'}
    prepare(tmp_path, monkeypatch, env, 208)
    for key in ('SCIENCE_PLACEMENT_RUNTIME', 'SCIENCE_PLACEMENT_SLOTS_PER_HOST', 'SCIENCE_PLACEMENT_CPUS_PER_CASE',
                'SCIENCE_PLACEMENT_MEMORY_GIB', 'SCIENCE_PLACEMENT_HELPER'):
        monkeypatch.setenv(key, client[key])
    from tpu.science.training_env import transport_requests
    requests = transport_requests('placement', 'source', 7200)
    assert len(requests) == 17
    for request in requests:
        spec = request.spec
        assert spec['resource_contract'] == expected
        r.validate(spec['resource_contract'])
        assert spec['resource_contract']['slots'] == spec['slots_per_host']
