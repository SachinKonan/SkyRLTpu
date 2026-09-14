"""Placement admission, device exposure, and compatibility with existing roles."""
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys

import pytest

from tpu.science.placement_slots import (
    chip_cpus, chip_lock, chips_from_env, device_paths, grading_nodes,
    host_resources, task_resources, tpu_environment, assigned_chip, grading_bundles,
)
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap import workload_resources

PROFILE = 'tpu/swarm/ray_train/profiles/science-placement-v5p-muse-pilot-001.json'


def test_v5p_roles_and_existing_v4_pilot():
    config = Config.load(PROFILE)
    assert config.inference_hosts == 2 and config.engines_per_host == 2
    assert workload_resources(config, 0) == {'TPU': 0}
    assert workload_resources(config, 1) == workload_resources(config, 2) == {'TPU': 4}
    resources = workload_resources(config, 3)
    assert resources == host_resources(range(4))
    nodes = grading_nodes([dict(Alive=True, NodeID=str(r), Resources=workload_resources(config, r))
                            for r in range(4)])
    assert [n['NodeID'] for n in nodes] == ['3']
    assert nodes[0]['Resources']['TPU'] == 4
    old = Config.load('tpu/swarm/ray_train/profiles/science-placement-v4-muse-pilot-002.json')
    assert workload_resources(old, 5) == host_resources([0])
    assert old.inference_hosts == 4


def test_training_roles_exclude_grading_host():
    config = Config.load('tpu/swarm/ray_train/profiles/native-v5p-muse-tp2-ac2-canary-002.json')
    changed = replace(config, client_env={**config.client_env,
                      'PLACEMENT_TPU_RANKS': '3', 'PLACEMENT_TPU_CHIPS': '0,1,2,3'})
    changed.validate()
    assert changed.inference_hosts == 2
    assert workload_resources(changed, 0) == {'TPU': 4}
    assert workload_resources(changed, 3) == host_resources(range(4))
    with pytest.raises(ValueError, match='disjoint'):
        replace(changed, client_env={**changed.client_env, 'PLACEMENT_TPU_RANKS': '0'}).validate()
    with pytest.raises(ValueError, match='at least one inference'):
        replace(changed, client_env={**changed.client_env, 'PLACEMENT_TPU_RANKS': '1,2,3'}).validate()


@pytest.mark.parametrize('value', ['0,0', '-1', '4', '', '0,1.5'])
def test_invalid_chip_assignments_fail_closed(value):
    with pytest.raises(ValueError):
        chips_from_env({'PLACEMENT_TPU_CHIPS': value})


def test_cpu_sets_and_single_chip_environment():
    assert len({cpu for chip in range(4) for cpu in chip_cpus(chip)}) == 16
    for chip in range(4):
        env = tpu_environment(chip, 'tpu-v5p-32')
        assert env['TPU_VISIBLE_CHIPS'] == str(chip)
        assert env['TPU_ACCELERATOR_TYPE'] == 'v5p-32'
        assert env['TPU_PROCESS_PORT'] == str(8476 + chip)
        assert env['TPU_CHIPS_PER_PROCESS_BOUNDS'] == '1,1,1'


def test_only_one_vfio_group_is_exposed(tmp_path):
    # Symlinks to a real character device model node discovery without mknod.
    (tmp_path / 'vfio').mkdir()
    for name in ['0', '1', '2', '3', 'vfio']:
        (tmp_path / 'vfio' / name).symlink_to('/dev/null')
    assert device_paths(2, 'tpu-v5p-32', tmp_path) == [tmp_path/'vfio/2', tmp_path/'vfio/vfio']
    (tmp_path/'vfio/2').unlink()
    with pytest.raises(FileNotFoundError):
        device_paths(2, 'tpu-v5p-32', tmp_path)
    with pytest.raises(ValueError, match='unsupported'):
        device_paths(0, 'tpu-unknown', tmp_path)


def test_locks_exclude_same_chip_across_processes_but_allow_other_chips(tmp_path):
    locks = tmp_path / 'locks'
    command = [sys.executable, '-c',
               'from tpu.science.placement_slots import chip_lock; import sys; '
               'c=chip_lock(int(sys.argv[1]),sys.argv[2]); c.__enter__(); c.__exit__(None,None,None)']
    with chip_lock(0, locks):
        same = subprocess.run(command + ['0', str(locks)], capture_output=True)
        other = subprocess.run(command + ['1', str(locks)], capture_output=True)
        assert same.returncode != 0 and b'BlockingIOError' in same.stderr
        assert other.returncode == 0, other.stderr
    assert subprocess.run(command + ['0', str(locks)], capture_output=True).returncode == 0


def test_lock_directory_symlink_is_rejected(tmp_path):
    target = tmp_path / 'target'
    target.mkdir(mode=0o700)
    link = tmp_path / 'alias'
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match='private and user-owned'):
        with chip_lock(0, link):
            pytest.fail('symlink lock directory accepted')


def test_placement_group_reserves_four_bundles_on_the_grading_node():
    node = dict(NodeManagerAddress='10.0.0.3', Resources={**host_resources(range(4)), 'node:10.0.0.3': 1})
    bundles = grading_bundles(node)
    assert len(bundles) == 4
    assert all(b == {'CPU': 4, 'memory': 16*1024**3, 'TPU': 1,
                     'placement_tpu_host': 1, 'node:10.0.0.3': .001} for b in bundles)


@pytest.mark.parametrize('ids', [{}, {'TPU': []}, {'TPU': ['0', '1']},
                                {'TPU': ['bad']}, {'TPU': [True]}, {'TPU': ['4']}])
def test_missing_or_ambiguous_ray_assignment_is_rejected(ids):
    with pytest.raises((RuntimeError, ValueError)):
        assigned_chip(ids)


def test_ray_assignment_overrides_parent_environment(monkeypatch):
    monkeypatch.setenv('TPU_VISIBLE_CHIPS', '0,1,2,3')
    assert assigned_chip({'TPU': ['3']}) == 3
    assert tpu_environment(assigned_chip({'TPU': ['3']}), 'tpu-v5p-32')['TPU_VISIBLE_CHIPS'] == '3'


def test_ray_id_reaches_subprocess_device_and_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tpu.science import placement_task
    source = tmp_path/'candidate.py'
    source.write_text('def place(*args, **kwargs): pass')
    captured = {}
    monkeypatch.setenv('TPU_VISIBLE_CHIPS', '0')  # Must not override Ray chip 3.
    monkeypatch.setattr(placement_task, 'python_mounts', lambda _: [])
    monkeypatch.setattr(placement_task, 'device_paths',
                        lambda chip, accelerator: [Path(f'/dev/vfio/{chip}'), Path('/dev/vfio/vfio')])
    monkeypatch.setattr(placement_task.subprocess, 'run',
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout=b'', stderr=b''))
    class StopBeforeExecution:
        def __init__(self, command, **kwargs): captured['command'] = command
        def wait(self, **kwargs): return 1
        def poll(self): return 1
    monkeypatch.setattr(placement_task.subprocess, 'Popen', StopBeforeExecution)
    request = dict(root=str(tmp_path), work=str(tmp_path/'work'), source=str(source),
                   case='ibm01', backend='tpu', accelerator='tpu-v5p-32', tpu_ids=['3'])
    with pytest.raises(RuntimeError, match='candidate exited 1'):
        placement_task.evaluate(request)
    command = captured['command']
    binds = [(command[i+1], command[i+2]) for i,x in enumerate(command) if x == '--dev-bind']
    assert binds == [('/dev/vfio/3','/dev/vfio/3'), ('/dev/vfio/vfio','/dev/vfio/vfio')]
    env = {command[i+1]:command[i+2] for i,x in enumerate(command) if x == '--setenv'}
    assert env['TPU_VISIBLE_CHIPS'] == '3'
    assert env['TPU_PROCESS_PORT'] == '8479'
