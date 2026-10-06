import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tpu.science.seed_handoff import storage_command
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.build import build


def test_storage_timeout_retries_without_resubmitting():
    with patch('tpu.science.seed_handoff.subprocess.run', side_effect=[
        subprocess.TimeoutExpired('gcloud', 300), SimpleNamespace(returncode=0, stdout=b'{}')
    ]) as run, patch('tpu.science.seed_handoff.time.sleep'):
        assert storage_command('gcloud', ['storage', 'cat', 'gs://test/object'], {}) == b'{}'
        assert run.call_count == 2
        assert run.call_args_list[0] == run.call_args_list[1]


def test_serve_shutdown_has_bounded_wait(tmp_path):
    from tpu.swarm.ray_train import controller
    released = threading.Event()
    events = []
    instance = SimpleNamespace(report=lambda event, **kw: events.append(event), shutdown_errors=[])
    with patch.object(controller.serve, 'shutdown', lambda: released.wait(2)):
        try:
            controller.Controller.close_serve(instance, timeout=.01)
            assert events == ['serve_cleanup_timeout']
            assert instance.shutdown_errors == [{'phase': 'serve_cleanup', 'error': 'timeout'}]
        finally:
            released.set()


@pytest.mark.parametrize('value', [-1, 4, True])
def test_error_restart_bound(value):
    raw = Config.load('tpu/swarm/ray_train/profiles/qwen_v5p_32.json').to_dict()
    raw['max_restarts_on_errors'] = value
    with pytest.raises(ValueError, match='max_restarts_on_errors'):
        Config.from_dict(raw)


def test_failed_worker_cleans_builds_before_propagating(tmp_path):
    from tpu.science import ray_cpu
    from tpu.science.worker import process_identity
    jobs=tmp_path/'jobs';jobs.mkdir()
    with patch.object(ray_cpu.subprocess, 'Popen', side_effect=OSError('launch failed')), \
         patch.object(ray_cpu.subprocess, 'run'), \
         patch.object(ray_cpu.os, 'sched_getaffinity', return_value=set(range(128))), \
         patch.object(ray_cpu,'cleanup_builds') as cleanup:
        with pytest.raises(OSError, match='launch failed'):
            ray_cpu._grade_admitted('routing', 'code', tmp_path, jobs, 0, 16)
        cleanup.assert_called_once()


def test_search_recovery_never_advances_optimizer_checkpoint():
    from ttt_discover.rl.resume import pool_step_for_resume
    assert pool_step_for_resume(1, 2) == 1  # unchanged default
    assert pool_step_for_resume(1, 2, preserve_ahead=True) == 2
    assert pool_step_for_resume(3, 2, preserve_ahead=True) == 3
    assert pool_step_for_resume(3, None, preserve_ahead=True) == 3


def test_routing_recovery_package_has_retries_and_sixteen_slots(tmp_path):
    import yaml
    from tpu.swarm.ray_train.overlay import manifest
    profile=Path('tpu/swarm/ray_train/profiles/science-routing-v4-qwen-bootstrap-l2-001-cpu16-resume.json')
    cfg=Config.load(profile)
    _, _, task=build(profile,tmp_path)
    doc=yaml.safe_load(task.read_text())
    assert doc['resources']['job_recovery']['max_restarts_on_errors'] == 3
    assert cfg.science_routing_slots_per_host == 16
    assert cfg.client_env['TTD_RESUME_STRICT'] == '1'
    assert cfg.client_env['TTD_RESUME_SEARCH_AHEAD'] == '1'
    assert 'third_party/discover/ttt_discover/rl/resume.py' in manifest(Path.cwd(),cfg)


def test_checkpoint_restart_retires_only_pending_requests(tmp_path):
    import sqlite3
    from tpu.swarm.ray_train.database_snapshot import abandon_pending_for_checkpoint_resume
    path=tmp_path/'db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE futures (status TEXT, result_data TEXT, completed_at TEXT)')
        db.executemany('INSERT INTO futures VALUES (?, ?, NULL)', [('PENDING',None),('COMPLETED','saved'),('FAILED','previous')])
    assert abandon_pending_for_checkpoint_resume(path) == 1
    assert abandon_pending_for_checkpoint_resume(path) == 0
    with sqlite3.connect(path) as db:
        rows=db.execute('SELECT status,result_data FROM futures').fetchall()
    assert rows[1:] == [('COMPLETED','saved'),('FAILED','previous')]
    assert rows[0][0] == 'FAILED'


def test_required_resume_never_silently_starts_fresh(tmp_path):
    import json
    from tpu.swarm.ray_train.database_snapshot import require_checkpoint_client
    logs=tmp_path/'tinker_log/run';member=logs/'member_qwen';member.mkdir(parents=True)
    (member/'checkpoints.jsonl').write_text(json.dumps(dict(batch=3,state_path='tinker://model/weights/000003'))+'\n')
    with pytest.raises(RuntimeError,match='matching search snapshot'):
        require_checkpoint_client(tmp_path,'run','qwen',3)
    (logs/'puct_sampler_step_000003.json').write_text('{}')
    assert require_checkpoint_client(tmp_path,'run','qwen',3) == 3
    with pytest.raises(RuntimeError,match='required checkpoint'):
        require_checkpoint_client(tmp_path,'run','qwen',4)


def test_interrupted_client_restore_is_not_published(tmp_path):
    from tpu.swarm.ray_train.host import Host
    cfg=SimpleNamespace(run_gcs='gs://test/run', checkpoint_resume=True, seed_pool_sha256='',resume_min_checkpoint_step=0)
    def transfer(command,*args,**kwargs):
        stage=Path(command[-1])/'client';stage.mkdir(exist_ok=True)
        (stage/'partial').write_text('partial')
        raise RuntimeError('transfer interrupted')
    host=SimpleNamespace(rank=0,trainer_leader=7,run=tmp_path,config=cfg,gcs=SimpleNamespace(list=lambda *a,**kw:['client'],transfer=transfer))
    with pytest.raises(RuntimeError,match='transfer interrupted'):
        Host.restore_run(host)
    assert not (tmp_path/'client').exists()
    def complete(command,*args,**kwargs):
        (Path(command[-1])/'client'/'complete').write_text('complete')
    host.gcs.transfer=complete
    Host.restore_run(host)
    assert (tmp_path/'client'/'complete').read_text() == 'complete'
    assert not (tmp_path/'client-restore').exists()
