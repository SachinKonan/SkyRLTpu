import fcntl
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tpu.swarm.ray_train import cache_admission as admission


def make_root(base, name, model='Qwen/test', role='inference'):
    root = base / name
    mount = root / 'ram'
    (mount / 'hf/hub' / ('models--' + model.replace('/', '--'))).mkdir(parents=True)
    (mount / 'role.json').write_text(json.dumps({'role': role}))
    (root / 'owner.lock').touch()
    run = root / 'runs' / name
    run.mkdir(parents=True)
    (run / admission.LEASE).touch()
    (run / admission.MANIFEST).write_text(json.dumps(dict(
        version=1, root=str(root), run_id=name, task_id='sky-managed-test_100-0',
        checkpoint_gcs=f'gs://test/ray-training/{name}/checkpoints')))
    (run / 'checkpoint.tar').write_bytes(b'durable state')
    return root


@pytest.fixture
def local_mounts(tmp_path, monkeypatch):
    mounted = set()
    calls = []
    monkeypatch.setattr(admission, 'tmpfs_mounts', lambda: mounted.copy())
    monkeypatch.setattr(admission, 'run_is_active', lambda *args: False)
    monkeypatch.setattr(admission, 'unused_mount', lambda *args: True)
    def run(args, **kwargs):
        calls.append(args)
        if args[2] == 'umount':
            mounted.remove(Path(args[3]))
        elif args[2:4] == ['mount', '--bind']:
            old, new = map(Path, args[4:6])
            mounted.add(new)
            for child in old.iterdir():
                if child.is_dir(): shutil.copytree(child, new / child.name)
                else: shutil.copyfile(child, new / child.name)
        else:
            raise AssertionError(args)
    monkeypatch.setattr(admission.subprocess, 'run', run)
    def reserve(target, cap, reserve):
        calls.append(['reserve', str(target)])
        return target
    monkeypatch.setattr(admission, 'mount_cache', reserve)
    cfg = SimpleNamespace(model='Qwen/test', trainer=SimpleNamespace(maxtext_model='qwen'),
                          cache=SimpleNamespace(trainer_gib=128, inference_gib=128, reserve_gib=128))
    return mounted, calls, cfg


def test_reuses_compatible_cache_and_evicts_other_model_before_reservation(tmp_path, local_mounts):
    mounted, calls, cfg = local_mounts
    old = make_root(tmp_path, 'old-qwen')
    other = make_root(tmp_path, 'old-gemma', model='Gemma/test')
    mounted.update([old / 'ram', other / 'ram'])
    new = tmp_path / 'new'; new.mkdir()
    assert admission.admit_cache(cfg, new, 'inference', tmp_path / 'events') == new / 'ram'
    assert mounted == {new / 'ram'}
    assert calls[0][2:4] == ['mount', '--bind']
    assert calls[1][2] == 'umount'
    assert calls[-1][0] == 'reserve'
    for root in [old, other]:
        assert (root / 'runs' / root.name / 'checkpoint.tar').read_bytes() == b'durable state'


@pytest.mark.parametrize('protection', ['root_lock', 'run_lock', 'process', 'open_mount',
                                       'unknown_files', 'missing_owner', 'bad_owner', 'symlink'])
def test_protected_caches_are_never_moved_or_unmounted(tmp_path, local_mounts, monkeypatch, protection):
    mounted, calls, cfg = local_mounts
    old = make_root(tmp_path, 'old')
    mounted.add(old / 'ram')
    fd = None
    run = old / 'runs/old'
    if protection in ('root_lock', 'run_lock'):
        lock = old / 'owner.lock' if protection == 'root_lock' else run / admission.LEASE
        fd = os.open(lock, os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif protection == 'process':
        monkeypatch.setattr(admission, 'run_is_active', lambda *args: True)
    elif protection == 'open_mount':
        monkeypatch.setattr(admission, 'unused_mount', lambda *args: False)
    elif protection == 'unknown_files':
        (old / 'ram/checkpoint').write_text('not disposable')
    elif protection == 'missing_owner':
        (run / admission.MANIFEST).unlink()
    elif protection == 'bad_owner':
        (run / admission.MANIFEST).write_text('{}')
    elif protection == 'symlink':
        (old / 'owner.lock').unlink(); (old / 'owner.lock').symlink_to(run / admission.LEASE)
    new = tmp_path / 'new'; new.mkdir()
    try:
        admission.admit_cache(cfg, new, 'inference', tmp_path / 'events')
    finally:
        if fd is not None: os.close(fd)
    assert mounted == {old / 'ram'}
    assert all(args[0] == 'reserve' for args in calls)


def test_busy_mount_is_not_force_unmounted(tmp_path, local_mounts, monkeypatch):
    mounted, calls, cfg = local_mounts
    old = make_root(tmp_path, 'old', model='Gemma/test'); mounted.add(old / 'ram')
    new = tmp_path / 'new'; new.mkdir()
    def busy(args, **kwargs):
        calls.append(args)
        raise admission.subprocess.CalledProcessError(32, args)
    monkeypatch.setattr(admission.subprocess, 'run', busy)
    admission.admit_cache(cfg, new, 'inference', tmp_path / 'events')
    assert mounted == {old / 'ram'}
    assert calls[0] == ['sudo', '-n', 'umount', str(old / 'ram')]


def test_mount_user_inspection_errors_are_not_idle():
    for code, stdout, stderr, expected in [(1, '', '', True), (0, '123', '', False),
                                         (1, '', 'permission denied', False)]:
        with patch.object(admission.subprocess, 'run', return_value=SimpleNamespace(
                returncode=code, stdout=stdout, stderr=stderr)):
            assert admission.unused_mount('/unused') is expected


@pytest.mark.parametrize('code,output,active', [(0, '{"active":false}', False),
                                              (0, '{"active":true}', True),
                                              (1, '', True), (0, 'broken', True)])
def test_privileged_process_audit_fails_closed(monkeypatch, code, output, active):
    monkeypatch.setattr(admission, 'unprivileged_run_is_active', lambda *args: True)
    monkeypatch.setattr(admission.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=code, stdout=output))
    assert admission.run_is_active({'task_id': 'task', 'run_id': 'run'}, Path('/root/run')) is active


def test_grader_reclaims_without_reserving_model_cache(tmp_path, local_mounts):
    mounted, calls, cfg = local_mounts
    old = make_root(tmp_path, 'old'); mounted.add(old / 'ram')
    new = tmp_path / 'new'; new.mkdir()
    assert admission.admit_cache(cfg, new, None, tmp_path / 'events') is None
    assert mounted == set()
    assert len(calls) == 1 and calls[0][2] == 'umount'


def test_failed_reuse_rolls_back_new_bind(tmp_path, local_mounts, monkeypatch):
    mounted, calls, cfg = local_mounts
    old = make_root(tmp_path, 'old'); mounted.add(old / 'ram')
    new = tmp_path / 'new'; new.mkdir()
    def failed_detach(args, **kwargs):
        calls.append(args)
        if args[2:4] == ['mount', '--bind']:
            mounted.add(new / 'ram')
        elif args == ['sudo', '-n', 'umount', str(old / 'ram')]:
            raise admission.subprocess.CalledProcessError(32, args)
        elif args == ['sudo', '-n', 'umount', str(new / 'ram')]:
            mounted.remove(new / 'ram')
        else: raise AssertionError(args)
    monkeypatch.setattr(admission.subprocess, 'run', failed_detach)
    admission.admit_cache(cfg, new, 'inference', tmp_path / 'events')
    assert mounted == {old / 'ram'}
    assert calls[2] == ['sudo', '-n', 'umount', str(new / 'ram')]


def test_interrupted_empty_reservation_does_not_prevent_reuse(tmp_path, local_mounts):
    mounted, calls, cfg = local_mounts
    old = make_root(tmp_path, 'old'); mounted.add(old / 'ram')
    new = tmp_path / 'new'; (new / 'ram').mkdir(parents=True); mounted.add(new / 'ram')
    admission.admit_cache(cfg, new, 'inference', tmp_path / 'events')
    assert mounted == {new / 'ram'}
    assert calls[0] == ['sudo', '-n', 'umount', str(new / 'ram')]
    assert calls[1][2:4] == ['mount', '--bind']
