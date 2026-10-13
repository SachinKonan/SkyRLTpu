import json
import sqlite3

import pytest

from tpu.swarm.ray_train.database_snapshot import reconcile_checkpoint_manifest


def fixture(tmp_path):
    path = tmp_path / 'db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE checkpoints(model_id TEXT,checkpoint_id TEXT,checkpoint_type TEXT,status TEXT,completed_at TEXT,error_message TEXT)')
        for step, status in [('000009', 'COMPLETED'), ('000010', 'PENDING'), ('000011', 'PENDING')]:
            for kind in ['TRAINING', 'SAMPLER']:
                db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,NULL,NULL)', ('model_a', step, kind, status))
    logs = tmp_path / 'logs'
    member = logs / 'member_qwen'
    member.mkdir(parents=True)
    record = dict(batch=10, state_path='tinker://model_a/weights/000010', sampler_path='tinker://model_a/000010')
    (member / 'checkpoints.jsonl').write_text(json.dumps(record) + '\n')
    (logs / 'puct_sampler_step_000010.json').write_text('{}')
    return path, logs


def states(path):
    with sqlite3.connect(path) as db:
        return db.execute('SELECT checkpoint_id,status FROM checkpoints ORDER BY rowid').fetchall()


def test_repairs_only_acknowledged_checkpoint_and_is_idempotent(tmp_path):
    path, logs = fixture(tmp_path)
    checked = []
    assert reconcile_checkpoint_manifest(path, logs, lambda archive: checked.append(archive) or True) == 2
    assert checked == ['checkpoints/model_a/000010.tar.gz', 'checkpoints/model_a/sampler_weights/000010.tar.gz']
    assert states(path) == [('000009', 'COMPLETED')] * 2 + [('000010', 'COMPLETED')] * 2 + [('000011', 'PENDING')] * 2
    assert reconcile_checkpoint_manifest(path, logs, lambda _: pytest.fail('already completed')) == 0


def test_missing_archive_rolls_back_all_rows(tmp_path):
    path, logs = fixture(tmp_path)
    before = states(path)
    with pytest.raises(RuntimeError, match='archive is missing'):
        reconcile_checkpoint_manifest(path, logs, lambda archive: 'sampler_weights' not in archive)
    assert states(path) == before


def test_requires_search_snapshot(tmp_path):
    path, logs = fixture(tmp_path)
    (logs / 'puct_sampler_step_000010.json').unlink()
    with pytest.raises(RuntimeError, match='matching search snapshot'):
        reconcile_checkpoint_manifest(path, logs, lambda _: True)
    assert ('000010', 'COMPLETED') not in states(path)


def test_no_manifest_does_not_guess_from_archive(tmp_path):
    path, logs = fixture(tmp_path)
    (logs / 'member_qwen/checkpoints.jsonl').unlink()
    assert reconcile_checkpoint_manifest(path, logs, lambda _: pytest.fail('no manifest')) == 0
    assert ('000010', 'COMPLETED') not in states(path)


@pytest.mark.parametrize('value', ['tinker://../weights/000010', 'tinker://model_a/weights/../../bad'])
def test_rejects_invalid_paths(tmp_path, value):
    path, logs = fixture(tmp_path)
    manifest = logs / 'member_qwen/checkpoints.jsonl'
    record = json.loads(manifest.read_text())
    record['state_path'] = value
    manifest.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='Invalid checkpoint manifest'):
        reconcile_checkpoint_manifest(path, logs, lambda _: True)


def test_does_not_revive_failed_checkpoint(tmp_path):
    path, logs = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE checkpoints SET status='FAILED' WHERE checkpoint_id='000010'")
    with pytest.raises(RuntimeError, match='conflicts with database'):
        reconcile_checkpoint_manifest(path, logs, lambda _: True)
    assert ('000010', 'COMPLETED') not in states(path)


def missing_registration(tmp_path):
    path, logs = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('ALTER TABLE checkpoints ADD COLUMN created_at TEXT')
        db.execute('CREATE TABLE models(model_id TEXT PRIMARY KEY, lora_config TEXT)')
        db.execute('INSERT INTO models VALUES (?,?)', ('model_a', '{"seed": 42}'))
        db.execute("DELETE FROM checkpoints WHERE checkpoint_id='000010'")
    return path, logs


def test_registers_acknowledged_missing_checkpoint_without_changing_model(tmp_path):
    path, logs = missing_registration(tmp_path)
    assert reconcile_checkpoint_manifest(path, logs, lambda _: True) == 2
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM models').fetchall() == [('model_a', '{"seed": 42}')]
        rows = db.execute("SELECT status,created_at,completed_at FROM checkpoints WHERE checkpoint_id='000010'").fetchall()
        assert len(rows) == 2 and all(s == 'COMPLETED' and c and t for s, c, t in rows)
    assert reconcile_checkpoint_manifest(path, logs, lambda _: pytest.fail('idempotent')) == 0


def test_missing_registration_requires_both_archives_atomically(tmp_path):
    path, logs = missing_registration(tmp_path)
    before = states(path)
    with pytest.raises(RuntimeError, match='archive is missing'):
        reconcile_checkpoint_manifest(path, logs, lambda a: 'sampler_weights' not in a)
    assert states(path) == before


def test_missing_registration_never_invents_model(tmp_path):
    path, logs = missing_registration(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM models')
    before = states(path)
    with pytest.raises(RuntimeError, match='model is not registered'):
        reconcile_checkpoint_manifest(path, logs, lambda _: True)
    assert states(path) == before


def test_host_reconciles_before_starting_api(tmp_path):
    from types import SimpleNamespace
    from tpu.swarm.ray_train.host import Host
    path, logs = fixture(tmp_path)
    database = tmp_path / 'tinker.db'
    path.rename(database)
    client = tmp_path / 'client'
    (client / 'tinker_log').mkdir(parents=True)
    logs.rename(client / 'tinker_log/run')
    checked = []
    def metadata(*args, **kwargs):
        uri = args[-1]
        checked.append(uri)
        return json.dumps([{'type': 'cloud_object', 'metadata': {'bucket': 'test', 'name': uri.removeprefix('gs://test/'), 'size': '10'}}])
    cfg = SimpleNamespace(run_gcs='gs://test/run', run_id='run', checkpoint_resume=True,
                          seed_pool_sha256='', resume_min_checkpoint_step=0, science_task='routing')
    host = SimpleNamespace(rank=0, trainer_leader=0, run=tmp_path, config=cfg,
                           log=tmp_path/'host.jsonl', gcs=SimpleNamespace(metadata=metadata))
    Host.restore_run(host)
    assert len(checked) == 2
    assert states(database)[2:4] == [('000010', 'COMPLETED')] * 2
    assert states(database)[4:] == [('000011', 'PENDING')] * 2
