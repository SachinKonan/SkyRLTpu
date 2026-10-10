import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx

from tpu.swarm.ray_train import local_notrain


class FakeStatus:
    def __init__(self, leases):
        self.leases = leases

    def json(self):
        return {'leases': self.leases}


def watched_run(tmp_path, monkeypatch, leases, events=b''):
    run = local_notrain.Run.__new__(local_notrain.Run)
    run.args = SimpleNamespace(stall_seconds=0.3)
    run.run = tmp_path
    run.log = tmp_path / 'local.jsonl'
    run.url = 'http://127.0.0.1:1'
    run.stopping = threading.Event()
    run.config = SimpleNamespace(run_id='test-run')
    run.child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    (tmp_path / 'inference-events.jsonl').write_bytes(events)
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: FakeStatus(leases))
    return run


def watch(run, seconds=1.5):
    thread = threading.Thread(target=run.watch_progress, kwargs=dict(interval=0.05), daemon=True)
    thread.start()
    time.sleep(seconds)
    run.stopping.set()
    thread.join(5)


def reported(run):
    if not run.log.exists():
        return []
    return [json.loads(line)['event'] for line in run.log.read_text().splitlines()]


def test_client_holding_a_ready_farm_without_work_is_stopped(tmp_path, monkeypatch):
    run = watched_run(tmp_path, monkeypatch, [{'url': 'farm', 'ready': True, 'lost': False}])
    watch(run)
    assert run.child.poll() is not None
    assert 'client_stalled' in reported(run)


def test_client_waiting_without_a_ready_farm_is_left_alone(tmp_path, monkeypatch):
    run = watched_run(tmp_path, monkeypatch, [{'url': 'farm', 'ready': False, 'lost': True}])
    try:
        watch(run)
        assert run.child.poll() is None
        assert 'client_stalled' not in reported(run)
    finally:
        run.child.kill()
        run.child.wait()


def test_client_doing_work_is_left_alone(tmp_path, monkeypatch):
    run = watched_run(tmp_path, monkeypatch, [{'url': 'farm', 'ready': True, 'lost': False}])
    events = tmp_path / 'inference-events.jsonl'
    stop = threading.Event()

    def work():
        while not stop.wait(0.1):
            with open(events, 'ab') as stream:
                stream.write(b'{"event": "grading_completed"}\n')
    writer = threading.Thread(target=work, daemon=True)
    writer.start()
    try:
        watch(run)
        assert run.child.poll() is None
    finally:
        stop.set()
        run.child.kill()
        run.child.wait()
