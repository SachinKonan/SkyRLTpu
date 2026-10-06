"""Grading capacity declared on farms and trainers: config, Ray tokens, CPU partition."""
import json
import os
from pathlib import Path

import pytest

from tpu.science import farm_resources
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.config import Config

FARM = 'tpu/swarm/ray_train/profiles/farm10-qwen-1-20260921-10step-deployment.json'
PILOT = 'tpu/swarm/ray_train/profiles/remote-only-v432-qwen-ac2-pilot-20260922.json'
AC2 = {'math': {'slots_per_host': 16, 'cpus': 2, 'memory_gib': 4, 'memory_max_gib': 8}}


def farm_config(tmp_path, families=AC2, reserve_gib=192, **extra):
    raw = Config.load(FARM).to_dict()
    raw['root'] = str(tmp_path)
    raw['cache']['reserve_gib'] = reserve_gib
    raw['grading'] = dict({'families': families}, **extra)
    return Config.from_dict(raw)


def test_farm_profile_declares_slots_ray_cpus_and_tokens(tmp_path):
    config = farm_config(tmp_path)
    assert config.grading_families == AC2
    assert config.ray_cpus_per_host == 17 + 16 * 2
    assert workload_resources(config, 0) == {'TPU': 4, 'grading_math': 16}
    assert config.grading_science_task is None
    assert not config.grading_farm_transport
    # Defaults fill unspecified family keys.
    partial = farm_config(tmp_path, families={'math': {'slots_per_host': 8}}, reserve_gib=128)
    # Math reserves 4 GiB and kills at 8 GiB (legacy grading had no memory limit).
    assert partial.grading_families == {'math': {'slots_per_host': 8, 'cpus': 2, 'memory_gib': 4, 'memory_max_gib': 8}}
    assert partial.ray_cpus_per_host == 33


def test_farm_science_families_use_their_contracts(tmp_path):
    routing = farm_config(tmp_path, families={'routing': {'slots_per_host': 4}}, reserve_gib=256)
    assert routing.grading_families['routing'] == {'slots_per_host': 4, 'cpus': 4, 'memory_gib': 8, 'memory_max_gib': 8}
    assert routing.grading_science_task == 'routing'
    assert workload_resources(routing, 1) == {'TPU': 4, 'grading_routing': 4}
    placement = farm_config(tmp_path, families={'placement': {'slots_per_host': 8, 'memory_gib': 4}}, reserve_gib=128)
    assert placement.grading_science_task == 'placement'
    assert workload_resources(placement, 2) == {'TPU': 4, 'grading_placement': 8, 'placement_cpu_host': 8}
    # One core pool serves every family, so a farm may grade all of them.
    both = farm_config(tmp_path, families={'routing': {'slots_per_host': 4}, 'placement': {'slots_per_host': 8},
                                           'ac2': {}}, reserve_gib=128)
    # 'ac2' is the legacy alias of the math family.
    assert set(both.grading_families) == {'math', 'routing', 'placement'} and both.grading_pool
    assert workload_resources(both, 2) == {'TPU': 4, 'grading_routing': 4, 'grading_placement': 8,
                                           'placement_cpu_host': 8, 'grading_math': 16}


@pytest.mark.parametrize('change', [
    dict(families={'other': {}}),
    dict(families={'math': {'slots_per_host': 0}}),
    dict(families={'math': {'slots_per_host': 65}}),
    dict(families={'math': {'unknown': 1}}),
    dict(families=AC2, farm_transport=True),
    dict(families=AC2, max_infra_retries=0),
    dict(families=AC2, poll_seconds=0),
])
def test_reject_invalid_grading_settings(tmp_path, change):
    with pytest.raises(ValueError):
        farm_config(tmp_path, **change)


def test_farm_grading_requires_lease_systemd_and_reserve(tmp_path):
    with pytest.raises(ValueError, match='reserve_gib'):
        farm_config(tmp_path, families={'placement': {'slots_per_host': 4, 'memory_gib': 100}}, reserve_gib=128)
    raw = Config.load(FARM).to_dict()
    raw['root'] = str(tmp_path)
    raw['cache']['reserve_gib'] = 192
    raw['grading'] = {'families': AC2}
    raw['systemd_runtime'] = False
    with pytest.raises(ValueError):
        Config.from_dict(raw)
    raw['systemd_runtime'] = True
    raw['inference']['require_lease'] = False
    raw['inference']['external_pool_attestation'] = False
    with pytest.raises(ValueError, match='lease-fenced'):
        Config.from_dict(raw)


def test_pilot_trainer_declares_local_ac2_slots_and_farm_transport(tmp_path):
    raw = Config.load(PILOT).to_dict()
    raw['root'] = str(tmp_path)
    config = Config.from_dict(raw)
    assert config.grading_farm_transport and config.grading.local_systemd
    assert config.grading_families == AC2
    assert config.ray_cpus_per_host == 32  # Trainer formula unchanged: 16 two-CPU tasks fit.
    assert workload_resources(config, 3) == {'TPU': 4, 'grading_math': 16}
    legacy = Config.load('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json')
    assert legacy.grading_families == {} and workload_resources(legacy, 5) == {'TPU': 4}
    assert Config.from_dict(config.to_dict()) == config


def test_partition_stacks_science_and_ac2_blocks_below_each_other():
    affinity = set(range(240))
    blocks, service = farm_resources.partition(AC2, affinity)
    assert blocks == {'math': list(range(208, 240))} and service == list(range(208))
    families = dict(AC2, placement={'slots_per_host': 8, 'cpus': 4, 'memory_gib': 4})
    blocks, service = farm_resources.partition(families, affinity)
    assert blocks['placement'] == list(range(208, 240)) and blocks['math'] == list(range(176, 208))
    assert service == list(range(176))
    assert farm_resources.math_slot_cpus(AC2, 0, affinity) == [208, 209]
    assert farm_resources.math_slot_cpus(AC2, 15, affinity) == [238, 239]
    with pytest.raises(RuntimeError):
        farm_resources.partition(AC2, set(range(48)))
    assert farm_resources.partition({}, affinity) == ({}, list(range(240)))


def test_partition_with_routing_uses_the_numa_contract(tmp_path):
    for node, cpus in enumerate(('0-119', '120-239')):
        folder = tmp_path / f'node{node}'
        folder.mkdir()
        (folder / 'cpulist').write_text(cpus + '\n')
    families = dict(AC2, routing={'slots_per_host': 10, 'cpus': 10, 'memory_gib': 20})
    blocks, service = farm_resources.partition(families, set(range(240)), topology_root=str(tmp_path))
    assert len(blocks['routing']) == 100 and len(blocks['math']) == 32 and len(service) == 108
    assert not set(blocks['routing']) & set(blocks['math']) and not set(service) & set(blocks['math'])


AC2_FARM = 'tpu/swarm/ray_train/profiles/farm-v432-qwen-grading-ac2-20260922.json'
ROUTING_FARM = 'tpu/swarm/ray_train/profiles/farm-v432-qwen-grading-routing-example-20260922.json'
REPO = Path(__file__).resolve().parents[2]


def test_grading_farm_profiles_are_valid_relaunch_targets():
    for name in ('farm-v432-qwen-grading-ac2-20260922', 'farm-v5p32-qwen-grading-ac2-20260922',
                 'farm-v6e32-qwen-grading-ac2-20260923'):
        config = Config.load(f'tpu/swarm/ray_train/profiles/{name}.json')
        assert config.inference_only and config.inference.require_lease and config.systemd_runtime
        assert config.grading_families == AC2 and config.inference.farm_drain_timeout == 600
        assert config.inference.farm_cancel_grace_seconds == 5 and config.ray_cpus_per_host == 49
        assert config.run_id == name and config.root.endswith(name)
    v6e = Config.load('tpu/swarm/ray_train/profiles/farm-v6e32-qwen-grading-ac2-20260923.json')
    assert v6e.hosts == 8 and v6e.inference_only_ranks == list(range(8)) and v6e.inference_hosts == 8
    assert v6e.inference.prefix_caching and v6e.inference.tp == 4 and v6e.zone == 'us-central1-b'
    assert workload_resources(v6e, 7) == {'TPU': 4, 'grading_math': 16}
    routing = Config.load(ROUTING_FARM)
    assert routing.grading_science_task == 'routing' and routing.inference_only


def test_overlay_ships_grading_and_science_files_for_farms_and_trainers(tmp_path):
    from tpu.swarm.ray_train.overlay import GRADING_FILES, SCIENCE_FILES, install, manifest
    ac2 = manifest(REPO, Config.load(AC2_FARM))
    present = {name for name in GRADING_FILES if (REPO / name).is_file()}
    assert present <= ac2.keys() and not SCIENCE_FILES & ac2.keys() - present
    assert 'tpu/vllm_tpu_server.py' in ac2
    routing = manifest(REPO, Config.load(ROUTING_FARM))
    assert SCIENCE_FILES <= routing.keys() and present <= routing.keys()
    pilot = manifest(REPO, Config.load(PILOT))
    assert present <= pilot.keys() and 'tpu/swarm/ray_train/borrowing_phase.py' in pilot
    # The installer accepts the grading file set combined with the other layers.
    for records in (ac2, pilot):
        overlay = tmp_path / 'overlay'
        overlay.mkdir(exist_ok=True)
        for name in records:
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
        (overlay / 'manifest.json').write_text(json.dumps(records, sort_keys=True))
        install(overlay, tmp_path / 'installed')


def test_generic_build_bundles_grading_modules_and_worker_root(tmp_path):
    import tarfile
    import yaml
    from tpu.swarm.ray_train.build import build
    archive, uri, task = build(Path(AC2_FARM), tmp_path / 'build')
    assert archive.name == 'ray-training.tar.gz'
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
    for name in ('tpu/science/math_grade.py', 'tpu/science/math_runner.py', 'tpu/science/farm_resources.py',
                 'tpu/science/worker.py', 'tpu/science/cgroup_limits.py', 'tpu/swarm/select_v4_32_topology.py',
                 'tpu/swarm/ray_train/grading_service.py', 'tpu/swarm/ray_train/multi_borrowing.py'):
        assert name in names, name
    doc = yaml.safe_load(task.read_text())
    assert 'export SCIENCE_WORKER_ROOT="$code"' in doc['run']
    assert doc['run'].index('SCIENCE_WORKER_ROOT') < doc['run'].index('exec python3 -m tpu.swarm.ray_train.bootstrap')
    legacy = build(Path('tpu/swarm/ray_train/profiles/farm10-qwen-1-20260921-10step-deployment.json'), tmp_path / 'legacy')
    assert 'SCIENCE_WORKER_ROOT' not in yaml.safe_load(legacy[2].read_text())['run']


def test_science_package_accepts_a_farm_with_a_science_family(tmp_path):
    import yaml
    from tpu.science.package_training import package
    task = package(Path(ROUTING_FARM), tmp_path / 'pkg')
    doc = yaml.safe_load(task.read_text())
    assert 'prepare_cpu_host.sh' in doc['run'] and 'export SCIENCE_WORKER_ROOT="$code"' in doc['run']
    assert 'SCIENCE_CPU_BUNDLE' in doc['envs']
    assert json.loads((tmp_path / 'pkg' / 'manifest.json').read_text())['task'] == 'routing'
    with pytest.raises(ValueError):
        package(Path('tpu/swarm/ray_train/profiles/farm10-qwen-1-20260921-10step-deployment.json'), tmp_path / 'no')


def test_client_environment_exposes_grading_transport_settings(tmp_path):
    from tpu.swarm.ray_train.commands import client_environment
    pilot = Config.load(PILOT)
    env = client_environment(pilot, Path(tmp_path), '10.0.0.1')
    assert env['SKYRL_GRADING_URL'] == 'http://10.0.0.1:%d' % pilot.ports.inference
    assert json.loads(env['SKYRL_GRADING_FAMILIES']) == AC2 and env['SKYRL_GRADING_LOCAL_SLOTS'] == '64'
    assert env['SKYRL_GRADING_MAX_INFRA_RETRIES'] == '3' and env['SKYRL_GRADING_LOCAL_SYSTEMD'] == '1'
    assert env['TTD_EVAL_BACKEND'] == 'hybrid' and env['TTD_SAFE_GRADE_MAX_WORKERS'] == '256'
    assert env['SKYRL_GRADING_EVENTS'].endswith('inference-events.jsonl')
    legacy = Config.load('tpu/swarm/ray_train/profiles/fresh-v4-qwen-ac2-grpo-lr15e4-s1-20260919.json')
    legacy_env = client_environment(legacy, Path(tmp_path), '10.0.0.1')
    # Sandbox trainers now grade through the pooled transport by default (no farms).
    assert 'SKYRL_GRADING_URL' not in legacy_env and legacy_env['TTD_EVAL_BACKEND'] == 'hybrid'
    assert legacy_env['SKYRL_GRADING_LOCAL_SLOTS'] == str(16 * legacy.hosts)  # the legacy local math capacity
    opted_out = Config.from_dict(dict(legacy.to_dict(), client_env=dict(legacy.client_env, TTD_EVAL_BACKEND='ray')))
    opted_env = client_environment(opted_out, Path(tmp_path), '10.0.0.1')
    assert opted_env['TTD_EVAL_BACKEND'] == 'ray' and 'SKYRL_GRADING_FAMILIES' not in opted_env


def test_ac2_admission_is_bounded_by_slots_and_pins_the_cpu_map(tmp_path):
    root = tmp_path / 'locks'
    families = {'math': {'slots_per_host': 2, 'cpus': 2, 'memory_gib': 4}}
    affinity = set(range(40))
    first = farm_resources.acquire_math(families, .5, root=root, affinity=affinity)
    second = farm_resources.acquire_math(families, .5, root=root, affinity=affinity)
    assert {first[0], second[0]} == {0, 1} and first[1] == [36, 37] and second[1] == [38, 39]
    with pytest.raises(TimeoutError):
        farm_resources.acquire_math(families, .3, root=root, affinity=affinity)
    assert json.loads((root / 'farm-ac2-v1-map.json').read_text()) == [36, 37, 38, 39]
    with pytest.raises(RuntimeError, match='different CPU map'):
        farm_resources.acquire_math(families, .3, root=root, affinity=set(range(44)))
    first[2].close()
    slot, cpus, lease = farm_resources.acquire_math(families, .5, root=root, affinity=affinity)
    assert slot == 0 and cpus == [36, 37]
    lease.close()
    second[2].close()
    assert oct(root.stat().st_mode & 0o777) == '0o700'
    os.chmod(root, 0o755)
    with pytest.raises(RuntimeError):
        farm_resources.acquire_math(families, .1, root=root, affinity=affinity)
