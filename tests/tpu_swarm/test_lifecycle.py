from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import lifecycle


def make_proc(tmp_path, procs):
    for pid, (args, cgroup) in procs.items():
        d = tmp_path / str(pid)
        d.mkdir()
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


TASK = 'sky-managed-2026_10_14-new_2600-0'
OLD = 'sky-managed-2026_10_13-old_2500-0'


def test_job_unit_is_stable_and_task_scoped():
    assert lifecycle.job_unit(TASK) == lifecycle.job_unit(TASK) != lifecycle.job_unit(OLD)
    argv = lifecycle.job_unit_argv(TASK, ['bash', 'run.sh'], '/tmp/x.log', '/home/u')
    assert '--unit=' + lifecycle.job_unit(TASK) in argv and '--description=' + TASK in argv
    assert '--property=KillMode=control-group' in argv and argv[-2:] == ['bash', 'run.sh']
    with pytest.raises(ValueError):
        lifecycle.job_unit('')


def test_leftovers_of_other_tasks_are_stopped_and_current_job_is_kept(tmp_path):
    current = lifecycle.job_unit(TASK)
    old_unit = lifecycle.job_unit(OLD)
    proc = make_proc(tmp_path, {
        11: (['python3', '-m', 'skyrl.tinker.api'], '/system.slice/' + current),        # ours
        12: (['python3', '/x/vllm_tpu_server.py'], '/system.slice/' + old_unit),        # dies with unit stop
        13: (['/r/raylet', '--session-dir=/home/u/.rtr-abc/session'], '/user.slice/u'), # escaped executor Ray
        14: (['python3', '-m', 'http.server'], '/user.slice/u'),                         # not a workload
        15: (['/r/raylet', '--session-dir=/tmp/ray_skypilot'], '/user.slice/u'),          # SkyPilot's own Ray
    })
    run, calls = systemctl([(current, TASK), (old_unit, OLD), ('placement-grade-ab12.service', '')])
    killed = []
    report = lifecycle.reap_leftovers(TASK, run=run, proc=proc, kill=lambda pid, sig: killed.append(pid))
    assert report['units'] == [old_unit, 'placement-grade-ab12.service']
    assert [p['pid'] for p in report['processes']] == [13]
    assert killed == [13]
    stops = [c[-1] for c in calls if c[:4] == ['sudo', '-n', 'systemctl', 'stop']]
    assert stops == [old_unit, 'placement-grade-ab12.service']


def test_shared_host_mode_only_reports(tmp_path):
    proc = make_proc(tmp_path, {21: (['python3', '/x/vllm_tpu_server.py'], '/user.slice/other')})
    run, calls = systemctl([(lifecycle.job_unit(OLD), OLD)])
    report = lifecycle.reap_leftovers(TASK, stop=False, run=run, proc=proc, kill=lambda *a: pytest.fail('killed'))
    assert report['units'] and report['processes'][0]['pid'] == 21
    assert not any(c[:3] == ['sudo', '-n', 'systemctl'] for c in calls)


def test_unit_audit_failure_is_not_an_empty_host(tmp_path):
    run = lambda argv, **kw: SimpleNamespace(returncode=1, stdout='', stderr='bus error')
    with pytest.raises(RuntimeError, match='unit audit failed'):
        lifecycle.reap_leftovers(TASK, run=run, proc=make_proc(tmp_path, {}))


def test_user_unit_argv():
    argv = lifecycle.user_unit_argv('rq4-discovery', ['python3', '-m', 'x'], '/w', '/w/x.log', env={'A': '1'})
    assert argv[:2] == ['systemd-run', '--user'] and '--unit=rq4-discovery' in argv
    assert '--setenv=A=1' in argv and argv[-3:] == ['python3', '-m', 'x']
    with pytest.raises(ValueError):
        lifecycle.user_unit_argv('bad name;rm', ['x'], '/w', '/w/x.log')
