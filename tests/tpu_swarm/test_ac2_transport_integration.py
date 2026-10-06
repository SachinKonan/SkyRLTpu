"""AC2 through the grading transport: verifier injection, re-verification, error classes, dedup."""
import asyncio
from pathlib import Path

import pytest

from tpu.science.grading_transport import GradingInfrastructureError

pytest.importorskip('examples.ac_inequalities.env')
from examples.ac_inequalities.env import AutoCorrInequalityEnv, ACInequalitiesRewardEvaluator  # noqa: E402
from ttt_discover.tinker_utils.dataset_builder import Environment, VerifyResult  # noqa: E402


class FakeTransport:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error
        self.requests = []

    def grade_sync(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.payload


def evaluator(tmp_path, transport):
    task = ACInequalitiesRewardEvaluator(problem_type='ac2', log_dir=str(tmp_path), num_cpus_per_task=2,
                                         eval_timeout=1100, eval_backend='hybrid')
    task.transport = transport
    assert task.exec_fn is None  # No cpu_scheduler actor, no Ray remote function.
    return task


CODE = '```python\ndef construct_function():\n    return list(height_sequence_1)\n```'


def test_hybrid_backend_keeps_verifier_injection_and_driver_reverification(tmp_path):
    state = AutoCorrInequalityEnv.create_initial_state('ac2')
    transport = FakeTransport(payload={'result': list(state.construction), 'error': None, 'stdout': 'hello'})
    task = evaluator(tmp_path, transport)
    out = task.get_reward(CODE, state)
    request = transport.requests[0]
    assert request.task == 'ac2' and request.spec['function_name'] == 'construct_function'
    assert request.spec['eval_timeout_seconds'] == 1105 and request.admission_timeout_s == 1100
    assert 'def evaluate_sequence' in request.spec['program_code'] and 'height_sequence_1' in request.spec['program_code']
    assert 'return list(height_sequence_1)' in request.spec['program_code']
    assert out['correctness'] == 1.0 and out['reward'] == out['raw_score'] > 0 and out['stdout'] == 'hello'
    assert out['result_construction'] == list(state.construction)
    # Driver-side verification still gates reward: a non-numeric construction is invalid.
    transport.payload = {'result': ['not', 'numbers'], 'error': None, 'stdout': ''}
    bad = task.get_reward(CODE, state)
    assert bad['correctness'] == 0.0 and bad['reward'] == 0.0 and bad['msg'] == 'Invalid solution.'


def test_candidate_failures_collapse_but_infrastructure_errors_abort(tmp_path):
    state = AutoCorrInequalityEnv.create_initial_state('ac2')
    transport = FakeTransport(payload={'result': None, 'error': 'Process timed out after 1105 seconds', 'stdout': 'x'})
    task = evaluator(tmp_path, transport)
    out = task.get_reward(CODE, state)
    assert out['reward'] == 0.0 and 'Program execution failed: Process timed out' in out['msg']
    task.transport = FakeTransport(error=GradingInfrastructureError('no pool'))
    with pytest.raises(GradingInfrastructureError):
        task.get_reward(CODE, state)


def test_dataset_builder_safe_grade_reraises_abort_errors_only():
    env = object.__new__(AutoCorrInequalityEnv)
    env.timeout = 5
    env.problem_type, env.log_path, env.state = 'ac2', '/tmp/x', None
    env._run_verification = lambda *a: (_ for _ in ()).throw(GradingInfrastructureError('lost'))
    with pytest.raises(GradingInfrastructureError):
        asyncio.run(Environment._safe_grade(env, 'answer', 1))
    env._run_verification = lambda *a: (_ for _ in ()).throw(ValueError('candidate broke the grader'))
    result = asyncio.run(Environment._safe_grade(env, 'answer', 1))
    assert result.reward == 0.0 and 'Error grading' in result.msg


def test_ac2_env_dedups_exact_source_within_a_step(monkeypatch):
    monkeypatch.setenv('TTD_EVAL_BACKEND', 'hybrid')
    env = object.__new__(AutoCorrInequalityEnv)
    env.problem_type, env.eval_timeout, env.eval_backend = 'ac2', 1100, 'hybrid'
    calls = []

    async def base(self, given_answer, step):
        calls.append((given_answer, step))
        await asyncio.sleep(.01)
        return VerifyResult(reward=1.0, msg='ok', correctness=1.0, raw_score=1.0, result_construction=[1.0], stdout='')
    monkeypatch.setattr(Environment, '_safe_grade', base)

    async def run():
        first, second, other = await asyncio.gather(env._safe_grade('same source', 3), env._safe_grade('same source', 3),
                                                    env._safe_grade('other source', 3))
        assert calls == [('same source', 3), ('other source', 3)]
        assert first.metrics == {} and second.metrics == {'grading_dedup_hit': True}
        assert first is not second and other.metrics == {}
        later = await env._safe_grade('same source', 4)
        assert later.metrics == {} and len(calls) == 3
    asyncio.run(run())
    monkeypatch.setenv('TTD_EVAL_BACKEND', 'ray')
    asyncio.run(env._safe_grade('same source', 3))
    asyncio.run(env._safe_grade('same source', 3))
    assert len(calls) == 5  # No dedup outside the transport backends.


class RunnerTransport:
    """Runs each request through the real sandbox runner (no systemd)."""
    def __init__(self, tmp_path):
        self.tmp_path, self.requests = tmp_path, []

    def grade_sync(self, request, timeout=None):
        import json, os, subprocess, sys
        from tpu.science.worker import process_identity
        self.requests.append(request)
        folder = self.tmp_path / f'job-{len(self.requests)}'
        folder.mkdir()
        spec = dict(request.spec, cpus=sorted(os.sched_getaffinity(0))[:2], stdout_limit_bytes=16384,
                    memory_gib=4, systemd=False)
        (folder / 'request.json').write_text(json.dumps(spec))
        subprocess.run([sys.executable, '-m', 'tpu.science.ac2_runner', '--request', str(folder / 'request.json'),
                        '--result', str(folder / 'result.json'), '--owner-pid', str(os.getpid()),
                        '--owner-start', process_identity(os.getpid())], check=True, timeout=120,
                       cwd=str(Path(__file__).resolve().parents[2]))
        return json.loads((folder / 'result.json').read_text())


ERDOS = '''```python
import numpy as np

def run():
    n = 50
    h = np.full(n, 0.5)
    c5 = verify_c5_solution(h, 0.0, n) if False else float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / n)))
    return h, c5, n
```'''


def test_erdos_tuple_round_trips_through_the_flat_sandbox(tmp_path):
    erdos = pytest.importorskip('examples.erdos_min_overlap.env')
    state = erdos.ErdosMinOverlapEnv.create_initial_state('')
    task = erdos.ErdosMinOverlapRewardEvaluator(problem_type='', log_dir=str(tmp_path / 'log'),
                                                num_cpus_per_task=2, eval_timeout=120, eval_backend='hybrid')
    task.transport = RunnerTransport(tmp_path)
    out = task.get_reward(ERDOS, state)
    request = task.transport.requests[0]
    assert request.spec['function_name'] == '_skyrl_flat_run' and 'def run()' in request.spec['program_code']
    assert out['correctness'] == 1.0 and out['raw_score'] == pytest.approx(0.5, abs=1e-9)
    assert len(out['result_construction']) == 50
    # A run() that does not return the triple is a candidate failure, not a crash.
    bad = task.get_reward('```python\ndef run():\n    return 1.0\n```', state)
    assert bad['correctness'] == 0.0 and 'Program execution failed' in bad['msg']
