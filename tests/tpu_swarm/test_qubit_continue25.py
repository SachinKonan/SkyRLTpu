"""Continuation follows the checkpoint index across engine incarnations."""
import pytest

from tpu.science.ops.qubit_continue25 import resume_object_keys, source_ready, require_checkpoint_registry


def test_recovered_final_does_not_require_numbered_archive_in_new_model():
    checkpoint = dict(batch=15, state_path='tinker://model_new/weights/final')
    published = {
        'tinker-backup.db',
        'checkpoints/model_old/000015.tar.gz',
        'checkpoints/model_new/final.tar.gz',
    }
    assert set(resume_object_keys(checkpoint)) <= published
    assert 'checkpoints/model_new/final.tar.gz' in resume_object_keys(checkpoint)


def test_numbered_resume_still_requires_its_state_archive():
    assert resume_object_keys(dict(batch=15, state_path='tinker://model_a/weights/000015')) == [
        'tinker-backup.db', 'checkpoints/model_a/000015.tar.gz']


@pytest.mark.parametrize('path', [
    'tinker://model_a/final', 'tinker://model_a/weights/../final',
    'https://model_a/weights/final', '',
])
def test_invalid_state_path_is_rejected(path):
    with pytest.raises(ValueError):
        resume_object_keys(dict(state_path=path))


@pytest.mark.parametrize('status', ['PENDING', 'RECOVERING', 'RUNNING', 'STARTING', 'CANCELLING'])
def test_shutdown_recovery_never_overlaps_old_writer(status):
    assert not source_ready(dict(status=status), dict(shutdown_recovery={'present': True}))


def test_failed_job_without_completion_evidence_is_rejected():
    with pytest.raises(RuntimeError):
        source_ready(dict(status='CANCELLED'), {})


def test_audited_shutdown_can_resume_and_evidence_cannot_change(tmp_path):
    import hashlib
    evidence = tmp_path / 'completion.json'
    evidence.write_text('{"client_exit_code":0,"completed_step":15}')
    source = dict(status='CANCELLED', job_id=1600, name='muse')
    record = dict(source_job=1600, run_id='muse', minimum_resume_step=15,
                  shutdown_recovery=dict(source_job=1600, run_id='muse', completed_step=15,
                                         client_exit_code=0, failure_phase='final_run_writeback',
                                         evidence=str(evidence),
                                         evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest()))
    assert source_ready(source, record)
    with pytest.raises(AssertionError):
        source_ready(dict(source, job_id=1601), record)
    evidence.write_text('{}')
    with pytest.raises(AssertionError):
        source_ready(source, record)


@pytest.mark.parametrize('registered,status,has_model,accepted', [
    ('000014', 'COMPLETED', True, False),
    ('000015', 'PENDING', True, False),
    ('000015', 'FAILED', True, False),
    ('000015', 'COMPLETED', False, False),
    ('000015', 'COMPLETED', True, True),
])
def test_resume_requires_matching_completed_database_registration(tmp_path, registered, status, has_model, accepted):
    import sqlite3
    path = tmp_path / 'backup.db'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE models(model_id TEXT, base_model TEXT);'
                         'CREATE TABLE checkpoints(model_id TEXT, checkpoint_id TEXT, checkpoint_type TEXT, status TEXT);')
        if has_model:
            db.execute('INSERT INTO models VALUES (?,?)', ('model_a', 'muse'))
        db.execute('INSERT INTO checkpoints VALUES (?,?,?,?)', ('model_a', registered, 'TRAINING', status))
    checkpoint = dict(state_path='tinker://model_a/weights/000015')
    if accepted:
        assert require_checkpoint_registry(path, checkpoint)['checkpoint_id'] == '000015'
    else:
        with pytest.raises(RuntimeError, match='cannot resume'):
            require_checkpoint_registry(path, checkpoint)


@pytest.mark.parametrize('status,dispatch', [('STARTING', True), ('RECOVERING', True), ('RUNNING', False)])
def test_pending_execution_repair_handles_recovery_without_resubmission(tmp_path, monkeypatch, status, dispatch):
    import json
    import sys
    from enum import Enum
    from types import SimpleNamespace, ModuleType
    import yaml
    from tpu.science.ops import qubit_continue25 as module

    states = Enum('States', ['PENDING', 'SUCCEEDED'])
    job = dict(job_id=1625, name='muse', pool=module.POOL, cluster='worker747', status=status)
    record = dict(source_job=1600, run_id='muse', code_sha256='a'*64)
    folder = tmp_path / '1600'
    folder.mkdir()
    (folder/'submitted.json').write_text(json.dumps({'job': job}))
    task = dict(name='muse', envs=dict(SKYPILOT_MANAGED_JOB_ID='1625', RAY_TRAIN_CODE_SHA256='a'*64))
    request = SimpleNamespace(name='sky.exec', cluster_name='worker747', status=states.PENDING,
                              request_body=SimpleNamespace(task=yaml.safe_dump(task)))
    executed = []
    def execute(rid, *args, **kwargs):
        executed.append(rid)
        request.status = states.SUCCEEDED
    fake = ModuleType('sky.server.requests')
    fake.requests = SimpleNamespace(get_request=lambda *args, **kwargs: request, RequestStatus=states)
    fake.executor = SimpleNamespace(executor_initializer=lambda *args: None,
                                    _request_execution_wrapper=execute)
    monkeypatch.setitem(sys.modules, 'sky.server.requests', fake)
    monkeypatch.setattr(module, 'base', tmp_path, raising=False)
    monkeypatch.setattr(module, 'r', SimpleNamespace(job=lambda jid: job), raising=False)
    module.dispatch_request(record, 'existing-request')
    assert executed == (['existing-request'] if dispatch else [])
