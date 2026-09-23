"""Grading capacity declared on farms and trainers: config, Ray tokens, CPU partition."""
from dataclasses import replace
import json
import os

import pytest

from tpu.science import farm_resources
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.config import Config

FARM = 'tpu/swarm/ray_train/profiles/farm10-qwen-1-20260921-10step-deployment.json'
PILOT = 'tpu/swarm/ray_train/profiles/remote-only-v432-qwen-ac2-pilot-20260922.json'
AC2 = {'ac2': {'slots_per_host': 16, 'cpus': 2, 'memory_gib': 4}}


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
    assert workload_resources(config, 0) == {'TPU': 4, 'grading_ac2': 16}
    assert config.grading_science_task is None
    assert not config.grading_farm_transport
    # Defaults fill unspecified family keys.
    partial = farm_config(tmp_path, families={'ac2': {'slots_per_host': 8}}, reserve_gib=128)
    assert partial.grading_families == {'ac2': {'slots_per_host': 8, 'cpus': 2, 'memory_gib': 4}}
    assert partial.ray_cpus_per_host == 33


def test_farm_science_families_use_their_contracts(tmp_path):
    routing = farm_config(tmp_path, families={'routing': {'slots_per_host': 4}}, reserve_gib=256)
    assert routing.grading_families['routing'] == {'slots_per_host': 4, 'cpus': 4, 'memory_gib': 8}
    assert routing.grading_science_task == 'routing'
    assert workload_resources(routing, 1) == {'TPU': 4, 'grading_routing': 4}
    placement = farm_config(tmp_path, families={'placement': {'slots_per_host': 8, 'memory_gib': 4}}, reserve_gib=128)
    assert placement.grading_science_task == 'placement'
    assert workload_resources(placement, 2) == {'TPU': 4, 'grading_placement': 8, 'placement_cpu_host': 8}
    with pytest.raises(ValueError, match='cannot share'):
        farm_config(tmp_path, families={'routing': {}, 'placement': {}}, reserve_gib=512)


@pytest.mark.parametrize('change', [
    dict(families={'other': {}}),
    dict(families={'ac2': {'slots_per_host': 0}}),
    dict(families={'ac2': {'slots_per_host': 65}}),
    dict(families={'ac2': {'unknown': 1}}),
    dict(families=AC2, farm_transport=True),
    dict(families=AC2, max_infra_retries=0),
    dict(families=AC2, poll_seconds=0),
])
def test_reject_invalid_grading_settings(tmp_path, change):
    with pytest.raises(ValueError):
        farm_config(tmp_path, **change)


def test_farm_grading_requires_lease_systemd_and_reserve(tmp_path):
    with pytest.raises(ValueError, match='reserve_gib'):
        farm_config(tmp_path, reserve_gib=100)
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
    assert workload_resources(config, 3) == {'TPU': 4, 'grading_ac2': 16}
    legacy = Config.load('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json')
    assert legacy.grading_families == {} and workload_resources(legacy, 5) == {'TPU': 4}
    assert Config.from_dict(config.to_dict()) == config


def test_partition_stacks_science_and_ac2_blocks_below_each_other():
    affinity = set(range(240))
    blocks, service = farm_resources.partition(AC2, affinity)
    assert blocks == {'ac2': list(range(208, 240))} and service == list(range(208))
    families = dict(AC2, placement={'slots_per_host': 8, 'cpus': 4, 'memory_gib': 4})
    blocks, service = farm_resources.partition(families, affinity)
    assert blocks['placement'] == list(range(208, 240)) and blocks['ac2'] == list(range(176, 208))
    assert service == list(range(176))
    assert farm_resources.ac2_slot_cpus(AC2, 0, affinity) == [208, 209]
    assert farm_resources.ac2_slot_cpus(AC2, 15, affinity) == [238, 239]
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
    assert len(blocks['routing']) == 100 and len(blocks['ac2']) == 32 and len(service) == 108
    assert not set(blocks['routing']) & set(blocks['ac2']) and not set(service) & set(blocks['ac2'])


def test_ac2_admission_is_bounded_by_slots_and_pins_the_cpu_map(tmp_path):
    root = tmp_path / 'locks'
    families = {'ac2': {'slots_per_host': 2, 'cpus': 2, 'memory_gib': 4}}
    affinity = set(range(40))
    first = farm_resources.acquire_ac2(families, .5, root=root, affinity=affinity)
    second = farm_resources.acquire_ac2(families, .5, root=root, affinity=affinity)
    assert {first[0], second[0]} == {0, 1} and first[1] == [36, 37] and second[1] == [38, 39]
    with pytest.raises(TimeoutError):
        farm_resources.acquire_ac2(families, .3, root=root, affinity=affinity)
    assert json.loads((root / 'farm-ac2-v1-map.json').read_text()) == [36, 37, 38, 39]
    with pytest.raises(RuntimeError, match='different CPU map'):
        farm_resources.acquire_ac2(families, .3, root=root, affinity=set(range(44)))
    first[2].close()
    slot, cpus, lease = farm_resources.acquire_ac2(families, .5, root=root, affinity=affinity)
    assert slot == 0 and cpus == [36, 37]
    lease.close()
    second[2].close()
    assert oct(root.stat().st_mode & 0o777) == '0o700'
    os.chmod(root, 0o755)
    with pytest.raises(RuntimeError):
        farm_resources.acquire_ac2(families, .1, root=root, affinity=affinity)
