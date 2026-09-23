"""AC2 executor: systemd command shape, runner semantics, infra vs candidate failures."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tpu.science import ac2_grade, ac2_runner

FAMILIES = {'ac2': {'slots_per_host': 2, 'cpus': 2, 'memory_gib': 4}}
REPO = Path(__file__).resolve().parents[2]


def request(tmp_path, code, timeout=20, cpus=(0, 1)):
    work = tmp_path / 'job'
    work.mkdir(exist_ok=True)
    (work / 'request.json').write_text(json.dumps(dict(
        program_code=code, function_name='construct_function', cpus=list(cpus),
        eval_timeout_seconds=timeout, stdout_limit_bytes=64, memory_gib=4, systemd=False)))
    return work


def run_runner(work):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.run([sys.executable, '-m', 'tpu.science.ac2_runner', '--request', str(work / 'request.json'),
                          '--result', str(work / 'result.json')], env=env, cwd=str(REPO),
                         capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (work / 'started.json').is_file()
    return json.loads((work / 'result.json').read_text())


def test_runner_returns_json_safe_result_and_bounded_stdout(tmp_path):
    code = ('import numpy as np\n'
            'def construct_function():\n'
            '    print("x" * 500)\n'
            '    return np.array([1.5, 2.0, 3.25])\n')
    result = run_runner(request(tmp_path, code, cpus=sorted(os.sched_getaffinity(0))[:2]))
    assert result['result'] == [1.5, 2.0, 3.25] and result['error'] is None
    assert len(result['stdout']) == 64 and result['stdout'] == 'x' * 63 + '\n'
    assert result['metrics']['cpus'] == sorted(os.sched_getaffinity(0))[:2]
    assert result['metrics']['worker_seconds'] > 0 and result['metrics']['candidate_seconds'] > 0


def test_runner_timeout_is_a_candidate_failure(tmp_path):
    code = 'import time\ndef construct_function():\n    time.sleep(30)\n    return [1.0]\n'
    result = run_runner(request(tmp_path, code, timeout=2, cpus=sorted(os.sched_getaffinity(0))[:1]))
    assert result['result'] is None and 'timed out after 2 seconds' in result['error']


@pytest.mark.parametrize('code, message', [
    ('def construct_function():\n    return [float("nan"), 1.0]\n', 'not finite numeric JSON'),
    ('def construct_function():\n    return {"a": 1}\n', 'not finite numeric JSON'),
    ('def construct_function():\n    raise ValueError("boom")\n', 'Program execution failed: ValueError: boom'),
    ('import os\ndef construct_function():\n    os._exit(3)\n', 'exited with code 3'),
])
def test_runner_candidate_failures_are_encoded_not_raised(tmp_path, code, message):
    result = run_runner(request(tmp_path, code, cpus=sorted(os.sched_getaffinity(0))[:1]))
    assert result['result'] is None and message in result['error']


def test_json_safe_accepts_numeric_sequences_only():
    assert ac2_runner.json_safe([1, 2.5]) == ([1.0, 2.5], None)
    assert ac2_runner.json_safe(3) == (3.0, None)
    assert ac2_runner.json_safe([True])[1] and ac2_runner.json_safe('x')[1]
    assert ac2_runner.json_safe([float('inf')])[1]


def fake_unit(tmp_path, monkeypatch, *, started=True, result=True, code=0):
    """Replace systemd-run with a Popen that mimics the unit's side effects."""
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            request_path = Path(command[command.index('--request') + 1])
            result_path = Path(command[command.index('--result') + 1])
            self.returncode = code
            if started:
                (result_path.parent / 'started.json').write_text('{}')
            if result:
                payload = json.loads(request_path.read_text())
                result_path.write_text(json.dumps(dict(result=[1.0], error=None, stdout='ok',
                                                       metrics=dict(cpus=payload['cpus']))))

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

    stops = []
    monkeypatch.setattr(ac2_grade.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(ac2_grade.subprocess, 'run', lambda cmd, **kw: stops.append(cmd))
    monkeypatch.setattr(ac2_grade, 'acquire_ac2', lambda families, deadline_seconds: (1, [7, 8], ExitStack()))
    monkeypatch.setattr('tpu.science.worker.process_identity', lambda pid: '12345')
    monkeypatch.setattr('tpu.science.cgroup_limits.runtime_owner_properties',
                        lambda: ['--property=BindsTo=skyrl-runtime-test.service'])
    return commands, stops


SPEC = dict(program_code='def construct_function():\n    return [1.0]\n', function_name='construct_function',
            eval_timeout_seconds=1105, admission_timeout_s=1100, stdout_limit_bytes=16384)


def test_systemd_command_shape_and_limits(tmp_path, monkeypatch):
    commands, stops = fake_unit(tmp_path, monkeypatch)
    result = ac2_grade.grade_ac2_admitted(SPEC, FAMILIES, root=tmp_path)
    command = commands[0]
    assert command[:3] == ['sudo', '-n', 'systemd-run'] and command[3].startswith('--unit=ac2-grade-')
    for prop in ('--property=MemoryMax=4G', '--property=MemorySwapMax=0', '--property=CPUQuota=200%',
                 '--property=AllowedCPUs=7,8', '--property=TasksMax=256', '--property=RuntimeMaxSec=1120',
                 '--property=KillMode=control-group', '--property=OOMPolicy=stop', '--property=PrivateTmp=yes',
                 '--property=BindsTo=skyrl-runtime-test.service', f'--working-directory={tmp_path}'):
        assert prop in command
    assert command[command.index('-m') + 1] == 'tpu.science.ac2_runner'
    assert command[command.index('--owner-start') + 1] == '12345'
    assert stops and stops[0][:4] == ['sudo', '-n', 'systemctl', 'stop']
    assert result['result'] == [1.0] and result['error'] is None
    metrics = result['metrics']
    assert metrics['hard_cpus'] == [7, 8] and metrics['slot'] == 1 and metrics['hard_memory_gib'] == 4
    assert metrics['grading_slots_per_host'] == 2 and metrics['admission_wait_seconds'] >= 0
    assert metrics['unit'].startswith('ac2-grade-') and metrics['ray_executor']
    job = tmp_path / '.science/ac2-jobs' / metrics['job_id']
    assert json.loads((job / 'verdict.json').read_text())['result'] == [1.0]
    assert json.loads((job / 'request.json').read_text())['systemd'] is True


def test_missing_started_marker_is_infrastructure(tmp_path, monkeypatch):
    fake_unit(tmp_path, monkeypatch, started=False, result=False, code=1)
    with pytest.raises(ac2_grade.GradingInfrastructureFailure, match='before the candidate started'):
        ac2_grade.grade_ac2_admitted(SPEC, FAMILIES, root=tmp_path)


def test_unit_killed_after_start_is_a_candidate_failure(tmp_path, monkeypatch):
    fake_unit(tmp_path, monkeypatch, started=True, result=False, code=137)
    result = ac2_grade.grade_ac2_admitted(SPEC, FAMILIES, root=tmp_path)
    assert result['result'] is None and 'exited 137' in result['error']


def test_admission_timeout_is_infrastructure(tmp_path, monkeypatch):
    def timed_out(families, deadline_seconds):
        raise TimeoutError('ac2 grading admission timed out before candidate execution')
    monkeypatch.setattr(ac2_grade, 'acquire_ac2', timed_out)
    with pytest.raises(ac2_grade.GradingInfrastructureFailure, match='admission timed out'):
        ac2_grade.grade_ac2_admitted(SPEC, FAMILIES, root=tmp_path)


def test_non_systemd_fallback_runs_the_runner_directly(tmp_path, monkeypatch):
    commands, stops = fake_unit(tmp_path, monkeypatch)
    result = ac2_grade.grade_ac2_admitted(dict(SPEC, systemd=False), FAMILIES, root=tmp_path)
    assert commands[0][0] == sys.executable and 'systemd-run' not in commands[0] and not stops
    assert result['metrics']['unit'] is None and result['result'] == [1.0]


def test_grade_ac2_is_a_bounded_ray_task():
    assert ac2_grade.grade_ac2._function is ac2_grade.grade_ac2_admitted
    options = ac2_grade.grade_ac2._default_options
    assert options['num_cpus'] == 2 and options['memory'] == 4 * 1024 ** 3 and options['max_retries'] == 0
