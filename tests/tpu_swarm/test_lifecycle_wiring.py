"""One start path (prepare), one end path (finish), exit hooks, and the della user-unit launcher."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import lifecycle

TASK = 'sky-managed-2026_10_14-new_2600-0'
OLD = 'sky-managed-2026_10_13-old_2500-0'


def make_proc(tmp_path, procs):
    for pid, (args, cgroup) in procs.items():
        d = tmp_path / str(pid)
        d.mkdir(parents=True)
        (d / 'cmdline').write_bytes('\0'.join(args).encode() + b'\0')
        (d / 'cgroup').write_text('0::' + cgroup + '\n')
    return tmp_path


def systemctl(units):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ['systemctl', 'list-units']:
            return SimpleNamespace(returncode=0, stderr='',
                                   stdout=''.join('%s loaded active running %s\n' % u for u in units))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    return run, calls


@pytest.fixture(autouse=True)
def fresh_hooks():
    lifecycle._EXIT_HOOKS.clear()
    lifecycle._EXIT_STATE.update(ran=False, log=None)
    yield
    lifecycle._EXIT_HOOKS.clear()
    lifecycle._EXIT_STATE.update(ran=False, log=None)


def test_prepare_stops_leftovers_on_a_pool_worker_and_logs_them(tmp_path):
    proc = make_proc(tmp_path / 'proc', {
        11: (['python3', '-m', 'tpu.thinking_budget.server'], '/system.slice/skyrl-runtime-old.service'),
        12: (['/usr/bin/raylet', '--session-dir=/home/u/.rtr-abc/session'], '/user.slice/session-9.scope')})
    run, calls = systemctl([('skyrl-runtime-old.service', OLD)])
    killed = []
    log = tmp_path / 'lifecycle.jsonl'
    report = lifecycle.prepare(TASK, log=log, tpu=False, run=run, proc=proc,
                               kill=lambda pid, sig: killed.append(pid))
    assert report['units'] == ['skyrl-runtime-old.service'] and killed == [12]
    assert ['sudo', '-n', 'systemctl', 'stop', 'skyrl-runtime-old.service'] in calls
    event = json.loads(log.read_text())
    assert event['event'] == 'lifecycle_prepared' and event['processes'][0]['pid'] == 12


def test_prepare_on_a_shared_host_refuses_and_stops_nothing(tmp_path):
    proc = make_proc(tmp_path / 'proc', {21: (['python3', '-m', 'skyrl.tinker.api'], '/user.slice/x.scope')})
    run, calls = systemctl([])
    killed = []
    with pytest.raises(RuntimeError, match='refusing to start'):
        lifecycle.prepare(TASK, shared=True, tpu=False, run=run, proc=proc, kill=lambda pid, sig: killed.append(pid))
    assert not killed and not any('stop' in c for c in calls)


def test_finish_runs_hooks_stops_task_units_and_kills_escapees(tmp_path):
    proc = make_proc(tmp_path / 'proc', {
        31: (['python3', '-m', 'tpu.science.worker'], '/system.slice/placement-grade-1.service'),
        32: (['/opt/vllm_tpu_server.py'], '/user.slice/session-1.scope')})
    run, calls = systemctl([('placement-grade-1.service', TASK)])
    released = []
    lifecycle.on_exit('release-lease', lambda reason: released.append(reason))
    killed = []
    log = tmp_path / 'f.jsonl'
    report = lifecycle.finish(TASK, 'cancel', log=log, run=run, proc=proc, kill=lambda pid, sig: killed.append(pid))
    assert released == ['cancel'] and report['units'] == ['placement-grade-1.service']
    # After the stop, the list still shows the unit in this fake, so only the escaped server is killed.
    assert killed == [32]
    assert json.loads(log.read_text().splitlines()[-1])['event'] == 'lifecycle_finished'


def test_exit_hooks_run_once_and_a_failing_hook_does_not_stop_others(tmp_path):
    seen = []
    lifecycle.on_exit('bad', lambda r: 1 / 0)
    lifecycle.on_exit('good', lambda r: seen.append(r))
    lifecycle._EXIT_STATE['log'] = tmp_path / 'h.jsonl'
    results = lifecycle.run_exit_hooks('error')
    assert [r['ok'] for r in results] == [False, True] and seen == ['error']
    assert lifecycle.run_exit_hooks('exit') == [] and seen == ['error']
    assert [json.loads(l)['hook'] for l in (tmp_path / 'h.jsonl').read_text().splitlines()] == ['bad', 'good']


@pytest.mark.parametrize('how', ['sigterm', 'exit', 'error'])
def test_installed_handlers_release_on_sigterm_normal_exit_and_error(tmp_path, how):
    marker = tmp_path / 'released'
    code = textwrap.dedent(f'''
        import os, signal, sys, time
        sys.path.insert(0, {os.getcwd()!r})
        from tpu.swarm.ray_train import lifecycle
        lifecycle.on_exit('lease', lambda reason: open({str(marker)!r}, 'w').write(reason))
        lifecycle.install_exit_handlers()
        print('ready', flush=True)
        if {how!r} == 'sigterm':
            time.sleep(30)
        elif {how!r} == 'error':
            raise RuntimeError('boom')
    ''')
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert child.stdout.readline().strip() == b'ready'
    if how == 'sigterm':
        child.send_signal(signal.SIGTERM)  # what systemctl stop and a SkyPilot cancel deliver
    child.wait(timeout=20)
    assert marker.read_text() == {'sigterm': 'signal', 'exit': 'exit', 'error': 'error'}[how]
    if how == 'sigterm':
        assert child.returncode == -signal.SIGTERM


def test_user_unit_argv_properties_replace_type():
    argv = lifecycle.user_unit_argv('rq4-discovery', ['python3', '-m', 'x'], '/w', '/w/l.log', restart='always',
                                    properties=('Type=notify', 'WatchdogSec=90'))
    assert '--property=Type=exec' not in argv and '--property=Type=notify' in argv
    assert '--property=WatchdogSec=90' in argv and '--property=Restart=always' in argv
    assert '--property=Type=exec' in lifecycle.user_unit_argv('n', ['x'], '/w', '/l')


def test_start_user_unit_passes_run_environment_and_returns_main_pid():
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        if argv[:3] == ['systemctl', '--user', 'show']:
            return SimpleNamespace(returncode=0, stdout='4242\n', stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    env = dict(TTD_EAGER_STEP_FRACTION='0.9', SKYRL_OPEN_GRADING='1', PYTHONPATH='/p', SECRET_TOKEN='x', HOME='/h')
    pid = lifecycle.start_user_unit('rq4-run-x', ['python', '-m', 'r'], '/w', '/w/runner.log', environ=env, run=run)
    start = next(c for c in calls if c[0] == 'systemd-run')
    assert pid == 4242 and '--unit=rq4-run-x' in start and start[-3:] == ['python', '-m', 'r']
    assert {'--setenv=TTD_EAGER_STEP_FRACTION=0.9', '--setenv=SKYRL_OPEN_GRADING=1', '--setenv=PYTHONPATH=/p',
            '--setenv=HOME=/h'} <= set(start)
    assert not any('SECRET_TOKEN' in a for a in start)


def test_audit_cli_reports_without_stopping(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, 'reap_leftovers', lambda task, stop: dict(units=['u'], processes=[], stop=stop))
    assert lifecycle.main(['audit', '--task-id', TASK]) == 0
    assert json.loads(capsys.readouterr().out) == dict(units=['u'], processes=[], stop=False)


def test_bootstrap_prepares_before_the_runtime_unit_and_passes_shared(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import bootstrap, runtime_service
    order = []
    monkeypatch.setattr(lifecycle, 'prepare', lambda task, shared, log, **kw: order.append(('prepare', task, shared)))
    monkeypatch.setattr(runtime_service, 'launch', lambda *a, **kw: order.append(('launch', kw['shared'])) or 0)
    cfg = SimpleNamespace(hosts=1, assigned_host_ips=[], root=str(tmp_path), run_id='r', systemd_runtime=True,
                          systemd_port=24900, systemd_peer_timeout=30, systemd_shutdown_grace=90, setup_timeout=60,
                          grading_host_sized=False, chips_per_host=4, job_ports=False)
    monkeypatch.setattr(bootstrap.Config, 'load', staticmethod(lambda path: cfg))
    monkeypatch.setenv('SKYPILOT_NODE_IPS', '10.0.0.1')
    monkeypatch.setenv('SKYPILOT_NODE_RANK', '0')
    monkeypatch.setenv('SKYPILOT_TASK_ID', TASK)
    monkeypatch.delenv('SKYRL_SYSTEMD_OWNED', raising=False)
    monkeypatch.setattr(sys, 'argv', ['bootstrap', 'profile.json'])
    assert bootstrap.main() == 0
    assert order == [('prepare', TASK, False), ('launch', False)]


def test_runtime_unit_is_tagged_with_the_task_and_finish_runs_on_cancel(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import runtime_service
    seen = {}
    class Popen:
        def __init__(self, args):
            seen['args'] = args
            self.returncode = 0
            self.polls = 0
        def poll(self):
            self.polls += 1
            if self.polls == 2:
                os.kill(os.getpid(), signal.SIGTERM)  # SkyPilot cancel reaches the launcher
            return None if self.polls < 4 else 0
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(runtime_service.subprocess, 'Popen', Popen)
    monkeypatch.setattr(runtime_service.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(runtime_service.subprocess, 'check_output', lambda *a, **kw: 'ActiveState=inactive\n')
    monkeypatch.setattr(runtime_service.time, 'sleep', lambda s: None)
    finished = []
    monkeypatch.setattr(lifecycle, 'finish', lambda task, reason, shared, log: finished.append((task, reason, shared)))
    assert runtime_service.launch(['true'], tmp_path, ['10.0.0.1'], 0, TASK) == 0
    assert '--description=' + TASK in seen['args']
    assert finished == [(TASK, 'cancel', False)]


def test_finish_failure_never_masks_the_job_result(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import runtime_service
    class Popen:
        def __init__(self, args):
            self.returncode = 3
        def poll(self):
            return 3
    monkeypatch.setattr(runtime_service.subprocess, 'Popen', Popen)
    monkeypatch.setattr(runtime_service.subprocess, 'check_output', lambda *a, **kw: 'ActiveState=failed\n')
    def boom(*a, **kw):
        raise OSError('systemctl gone')
    monkeypatch.setattr(lifecycle, 'finish', boom)
    assert runtime_service.launch(['true'], tmp_path, ['10.0.0.1'], 0, TASK) == 3
    assert list(tmp_path.glob('systemd-*/lifecycle-error.json'))


def fake_dev(tmp_path):
    dev = tmp_path / 'dev'
    (dev / 'vfio').mkdir(parents=True)
    (dev / 'accel0').touch()
    return dev


def fuser(owners):
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        if argv[:3] == ['sudo', '-n', 'fuser'] and '-k' not in argv:
            return SimpleNamespace(returncode=0 if owners else 1, stdout=' '.join(owners), stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    return run, calls


def test_tpu_gates_free_devices_and_a_foreign_lock_on_a_pool_worker(tmp_path, monkeypatch):
    dev, tmp = fake_dev(tmp_path), tmp_path / 'tmp'
    tmp.mkdir()
    lock = tmp / 'libtpu_lockfile'
    lock.touch()
    monkeypatch.setattr(os, 'access', lambda path, mode: Path(path) != lock)  # another user's lock
    import shutil
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: SimpleNamespace(free=10**15))
    run, calls = fuser(['4242'])
    report = lifecycle.tpu_gates(shared=False, run=run, home=tmp_path, tmp=tmp, dev=dev)
    assert report == dict(device_owners=['4242'], foreign_tpu_lock=True)
    assert any(c[:4] == ['sudo', '-n', 'fuser', '-k'] for c in calls)
    assert ['sudo', '-n', 'rm', '-f', str(lock)] in calls


def test_tpu_gates_on_a_shared_host_only_report_and_prepare_refuses(tmp_path, monkeypatch):
    dev = fake_dev(tmp_path)
    import shutil
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: SimpleNamespace(free=10**15))
    run, calls = fuser(['77'])
    report = lifecycle.tpu_gates(shared=True, run=run, home=tmp_path, tmp=tmp_path, dev=dev)
    assert report['device_owners'] == ['77'] and not any('-k' in c for c in calls)
    monkeypatch.setattr(lifecycle, 'tpu_gates', lambda **kw: report)
    proc_dir = tmp_path / 'proc'
    proc_dir.mkdir()
    units, _ = systemctl([])
    with pytest.raises(RuntimeError, match='refusing'):
        lifecycle.prepare(TASK, shared=True, run=units, proc=proc_dir)


def test_tpu_gates_refuse_when_space_cannot_be_reclaimed(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='insufficient storage'):
        lifecycle.tpu_gates(shared=False, run=fuser([])[0], home=tmp_path, tmp=tmp_path, dev=fake_dev(tmp_path))


def test_science_and_placement_packages_take_the_lifecycle_path_not_the_audit():
    import inspect
    from tpu.science import package_placement, package_training
    assert 'clean_host_audit' not in inspect.getsource(package_training)
    source = inspect.getsource(package_placement)
    assert 'clean_host_audit' not in source and 'tpu.swarm.ray_train.lifecycle prepare' in source
