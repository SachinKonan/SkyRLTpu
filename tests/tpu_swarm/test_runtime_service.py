import os
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from tpu.swarm.ray_train.runtime_service import SliceState, process_start, start_server, exchange, wait_preflight


def test_barrier_never_releases_before_every_host_passes():
    now = [0]
    state = SliceState(4, 'attempt', 3, now=lambda: now[0])
    for rank in range(3):
        assert not state.update(dict(token='attempt', rank=rank, phase='ready'))['release']
    assert state.update(dict(token='other', rank=3, phase='ready')) == {'error': 'wrong attempt or rank'}
    assert state.update(dict(token='attempt', rank=3, phase='ready'))['release']
    now[0] = 4
    result = state.update(dict(token='attempt', rank=0, phase='running'))
    assert result['failure'] and not result['release']
    for rank in range(4):
        assert not state.update(dict(token='attempt', rank=rank, phase='ready'))['release']


def test_preflight_failure_aborts_peers():
    s = SliceState(2, 'x', 30)
    s.update(dict(token='x', rank=0, phase='ready'))
    assert s.update(dict(token='x', rank=1, phase='failed'))['failure']
    assert not s.update(dict(token='x', rank=1, phase='ready'))['release']


def test_coordinator_completion_waits_for_peer_ack():
    state = SliceState(2, 'x', 30)
    server = start_server(('127.0.0.1', 0), state)
    try:
        exchange(server.server_address, dict(token='x', rank=0, phase='done'))
        result = exchange(server.server_address, dict(token='x', rank=1, phase='done'))
        assert result['done'] and not result['finished']
        result = exchange(server.server_address, dict(token='x', rank=1, phase='finished'))
        assert result['done'] and result['finished']
    finally:
        server.shutdown(); server.server_close()


def test_dead_or_reused_launcher_cannot_match_identity():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    original = process_start(child.pid)
    assert original and process_start(child.pid) == original
    child.kill(); child.wait()
    assert process_start(child.pid) is None


def test_preflight_is_opt_in_and_cancellable(tmp_path, monkeypatch):
    cfg = SimpleNamespace(systemd_runtime=False, setup_timeout=1)
    wait_preflight(cfg, lambda: False)
    cfg.systemd_runtime = True
    monkeypatch.setenv('SKYRL_SYSTEMD_OWNED', '1')
    monkeypatch.setenv('SKYRL_PREFLIGHT_READY', str(tmp_path/'ready'))
    monkeypatch.setenv('SKYRL_PREFLIGHT_RELEASE', str(tmp_path/'release'))
    import pytest
    with pytest.raises(RuntimeError, match='aborted'):
        wait_preflight(cfg, lambda: True)
    assert (tmp_path/'ready').exists()
    assert not (tmp_path/'release').exists()


def test_systemd_config_is_opt_in_and_reserves_distinct_port():
    import pytest
    from tpu.swarm.ray_train.config import Config
    path = 'tpu/swarm/ray_train/profiles/qwen_v5p_32.json'
    cfg = Config.load(path)
    assert cfg.systemd_runtime is False
    raw = cfg.to_dict() if hasattr(cfg, 'to_dict') else __import__('dataclasses').asdict(cfg)
    for value in (True, 1.5, 1023, 65536):
        with pytest.raises(ValueError, match='coordinator port'):
            Config.from_dict(raw | {'systemd_port': value})
    for port in (cfg.ports.ray, cfg.ports.worker_min):
        with pytest.raises(ValueError, match='overlaps'):
            Config.from_dict(raw | {'systemd_runtime': True, 'systemd_port': port})
    assert Config.from_dict(raw | {'systemd_runtime': True}).systemd_runtime


def test_separate_grader_units_follow_exact_local_runtime(monkeypatch):
    from tpu.science.cgroup_limits import runtime_owner_properties
    unit = 'skyrl-runtime-' + 'a' * 20 + '.service'
    monkeypatch.setattr(Path, 'read_text', lambda self: '0::/system.slice/'+unit+'\n')
    assert runtime_owner_properties() == [f'--property={p}={unit}' for p in ('BindsTo','After','PartOf')]
    monkeypatch.setattr(Path, 'read_text', lambda self: '0::/user.slice/other.service\n')
    assert runtime_owner_properties() == []


def test_runtime_preserves_limits_instead_of_systemd_defaults():
    import resource
    from tpu.swarm.ray_train.runtime_service import inherited_limit_properties
    properties = inherited_limit_properties()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    expected = ':'.join('infinity' if v == resource.RLIM_INFINITY else str(v) for v in (soft, hard))
    assert '--property=LimitNOFILE=' + expected in properties
    assert '--property=TasksMax=infinity' in properties


def test_preflight_checks_coordinator_range_without_rebinding_listener(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import bootstrap
    cfg = SimpleNamespace(retired_task_ids=[], retired_processes={}, systemd_runtime=True,
                          systemd_port=24900, ports=SimpleNamespace(worker_min=22000, worker_max=22999))
    calls = []
    monkeypatch.setattr(bootstrap, 'workload_ports', lambda _: [24679])
    monkeypatch.setattr(bootstrap, 'check_port_isolation', lambda ports: calls.append(('ranges',ports)))
    monkeypatch.setattr(bootstrap, 'check_ports_available', lambda ports: calls.append(('bind',ports)))
    bootstrap.retire_before_port_check(cfg, '127.0.0.1', tmp_path/'events.jsonl')
    assert calls == [('ranges',[24679,24900]),('bind',[24679])]
