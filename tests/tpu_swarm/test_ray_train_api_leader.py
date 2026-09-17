"""The API database follows physical trainer rank zero, independently of Ray head."""
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import host as module


@pytest.mark.parametrize('rank', [0, 3, 7])
def test_only_api_leader_backs_up_database_and_logs_do_not_collide(tmp_path, monkeypatch, rank):
    calls = []
    class GCS:
        def __init__(self, *args):
            pass
        def transfer(self, command, label, **kwargs):
            if label == 'database-writeback':
                with sqlite3.connect(command[1]) as db:
                    assert db.execute('select value from proof').fetchone() == ('current-api',)
            calls.append((label, command))
    monkeypatch.setattr(module, 'GCS', GCS)
    with sqlite3.connect(tmp_path / 'tinker.db') as db:
        db.execute('create table proof(value text)')
        db.execute('insert into proof values (?)', ('current-api',))
    (tmp_path / 'client').mkdir()
    (tmp_path / 'client/cohort.json').write_text('{}')
    (tmp_path / 'trainer.log').write_text('trainer')
    host = SimpleNamespace(rank=rank, trainer_leader=7, run=tmp_path,
                           config=SimpleNamespace(run_gcs='gs://test/run', cache=None))
    host.run_sync_lock = threading.Lock()
    host._sync_run = lambda: module.Host._sync_run(host)
    module.Host.sync_run(host)
    assert any(label == 'database-writeback' for label, _ in calls) == (rank == 7)
    assert any(label == 'client-writeback' for label, _ in calls) == (rank == 0)
    logs = next(command for label, command in calls if label == 'logs-writeback')
    assert logs[-1] == ('gs://test/run/logs/' if rank == 0 else f'gs://test/run/logs/host-{rank}/')


@pytest.mark.parametrize('rank', [0, 7])
def test_restore_routes_client_to_head_and_database_to_api_leader(tmp_path, rank):
    calls = []
    def transfer(command, label, *args, **kwargs):
        calls.append(label)
        if label == 'restore-database':
            with sqlite3.connect(command[-1]) as db:
                db.execute('create table proof(value text)')
    gcs = SimpleNamespace(list=lambda *a, **kw: ['client'], metadata=lambda *a, **kw: json.dumps(['db']),
                          transfer=transfer)
    host = SimpleNamespace(rank=rank, trainer_leader=7, run=tmp_path, gcs=gcs,
                           config=SimpleNamespace(run_gcs='gs://test/run', seed_pool_sha256=''))
    module.Host.restore_run(host)
    assert calls == (['restore-run'] if rank == 0 else ['restore-database'])


def test_readiness_checks_leader_log_instead_of_head_rpc_worker(tmp_path):
    log = tmp_path / 'trainer.log'
    log.write_text('Initialized TinkerEngine with\nbackend=DistributedTunixBackend')
    process = SimpleNamespace(poll=lambda: None, log_offset=0)
    leader = SimpleNamespace(rank=7, trainer_leader=7, run=tmp_path, processes={'trainer': process})
    assert module.Host.trainer_ready(leader)
    process.log_offset = log.stat().st_size
    assert not module.Host.trainer_ready(leader)
