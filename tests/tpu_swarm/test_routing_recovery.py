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
    instance = SimpleNamespace(report=lambda event, **kw: events.append(event))
    with patch.object(controller.serve, 'shutdown', lambda: released.wait(2)):
        try:
            controller.Controller.close_serve(instance, timeout=.01)
            assert events == ['serve_cleanup_timeout']
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
         patch.object(ray_cpu.subprocess, 'run'), patch.object(ray_cpu,'cleanup_builds') as cleanup:
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
