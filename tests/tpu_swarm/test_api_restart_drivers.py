"""Restart drivers never leave the API or controllers suspended on failure."""
import os
import signal
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tpu', 'swarm',
                                'api_restart'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tpu', 'swarm'))

import api_plain_restart  # noqa: E402
import api_reattach_restart  # noqa: E402
import api_vacuum_restart as base  # noqa: E402
import skypilot_queue_drain_worker  # noqa: E402


class FakeProc:

    def __init__(self, pid):
        self.pid = pid
        self.signals = []
        self.killed = False

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):
        self.killed = True

    def create_time(self):
        return 1000.0 + self.pid

    def ppid(self):
        return 1

    def is_running(self):
        return True

    def children(self, recursive=False):
        del recursive
        return []


@pytest.fixture
def frozen_world(monkeypatch, tmp_path):
    """An API tree (pid 100 + child 101), 5 pool and 2 jobs controllers."""
    root, child = FakeProc(100), FakeProc(101)
    root.children = lambda recursive=False: [child]
    pools = [FakeProc(200 + i) for i in range(5)]
    jobs = [FakeProc(300), FakeProc(301)]
    group_signals = []
    monkeypatch.setattr(base, 'api_root', lambda: root)
    monkeypatch.setattr(base, 'pool_controllers', lambda: pools)
    monkeypatch.setattr(base, 'jobs_controllers', lambda: jobs)
    monkeypatch.setattr(base, 'mine', lambda: pools + jobs)
    monkeypatch.setattr(base, '_pgid', lambda p: p.pid)
    monkeypatch.setattr(os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(os, 'killpg', lambda g, sig: group_signals.append((g, sig)))
    monkeypatch.setattr(base, 'load_state', lambda: {'prep_done': True, 'lock_pid': 1})
    monkeypatch.setattr(base, 'save_state', lambda **kw: None)
    monkeypatch.setattr(base, 'audit', lambda: tmp_path)
    monkeypatch.setattr(api_reattach_restart, 'load_state', base.load_state)
    monkeypatch.setattr(api_reattach_restart, 'save_state', base.save_state)
    monkeypatch.setattr(api_reattach_restart, 'audit', base.audit)
    monkeypatch.setattr(api_plain_restart, 'load_state', base.load_state)
    monkeypatch.setattr(api_plain_restart, 'save_state', base.save_state)
    monkeypatch.setattr(base.psutil, 'pid_exists', lambda pid: True)
    monkeypatch.setattr(base.time, 'sleep', lambda s: None)
    return root, child, pools + jobs, group_signals


def _assert_everything_thawed(root, child, controllers, group_signals):
    for proc in (root, child):
        assert proc.signals[-1] == signal.SIGCONT, proc.signals
        assert not proc.killed
    thawed = {g for g, sig in group_signals if sig == signal.SIGCONT}
    assert thawed == {p.pid for p in controllers}


def test_reattach_copy_failure_thaws_everything(frozen_world, monkeypatch):
    root, child, controllers, group_signals = frozen_world
    monkeypatch.setattr(api_reattach_restart, '_deleted_fds', lambda pid: ('6', '7'))

    def full_disk(*args, **kwargs):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(api_reattach_restart.shutil, 'copyfile', full_disk)
    with pytest.raises(SystemExit, match='ABORTED, everything thawed'):
        api_reattach_restart.stop()
    _assert_everything_thawed(root, child, controllers, group_signals)


def test_reattach_existing_recovery_dir_thaws(frozen_world, monkeypatch, tmp_path):
    root, child, controllers, group_signals = frozen_world
    monkeypatch.setattr(api_reattach_restart, '_deleted_fds', lambda pid: ('6', '7'))
    (tmp_path / 'recovered').mkdir()
    with pytest.raises(SystemExit, match='FileExistsError'):
        api_reattach_restart.stop()
    _assert_everything_thawed(root, child, controllers, group_signals)


def test_plain_db_read_failure_thaws_everything(frozen_world, monkeypatch):
    root, child, controllers, group_signals = frozen_world

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(api_plain_restart.sqlite3, 'connect', locked)
    with pytest.raises(SystemExit, match='ABORTED, thawed'):
        api_plain_restart.stop()
    _assert_everything_thawed(root, child, controllers, group_signals)


@pytest.mark.parametrize('cmdline,ok', [
    (['python', '-m', 'sky.server.server', '--deploy', '--port=46580'], True),
    (['python', 'tpu/swarm/skypilot_preserving_restart.py', '--deploy',
      '--port=46580'], True),
    (['python', '-m', 'sky.jobs.controller', 'abc'], False),
    (['python', 'some_other_server.py', '--deploy'], False),
])
def test_drain_worker_accepts_both_api_entrypoints(cmdline, ok):
    assert skypilot_queue_drain_worker.is_skypilot_api_cmdline(cmdline) is ok
