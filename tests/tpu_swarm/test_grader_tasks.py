"""Real Ray scheduling and process cleanup, with only TPU computation stubbed."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

import pytest
import ray

from tpu.swarm.ray_train import grader_tasks as module
from pallas_arena.rl.task import public_contract, translate_verdict, ArenaInfrastructureError


@pytest.fixture
def cluster(tmp_path):
    ray.init(num_cpus=12, resources={'TPU': 4, 'arena_grader': 4, 'arena_pregate': 2},
             include_dashboard=False, _node_ip_address='127.0.0.1',
             object_store_memory=100 * 1024**2, namespace='grader-tasks-test')
    # Production task, production guardian and production resource assignment;
    # this small executable substitutes for the expensive TPU judge only.
    shim = tmp_path / 'envs/arena/bin/python'
    shim.parent.mkdir(parents=True)
    shim.write_text('#!' + sys.executable + '\n' + r'''
import json, os, pathlib, signal, sys, time
request, output = map(pathlib.Path, sys.argv[-2:])
r = json.loads(request.read_text())
payload = r['payload']
if r['mode'] == 'pregate':
    result = {'passed': payload['code'] != 'invalid', 'violations': ['non-Pallas']}
else:
    assert os.environ['TPU_VISIBLE_CHIPS'] == r['chip']
    assert 'RAY_ADDRESS' not in os.environ
    root = pathlib.Path(payload.get('test_root', request.parents[4]))
    lease = root / ('chip-' + r['chip'])
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    def stop(*args):
        lease.unlink(missing_ok=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, stop)
    try:
        time.sleep(.1)
        case = r['case']
        marker = request.parent / (case + '.attempt')
        n = int(marker.read_text()) + 1 if marker.exists() else 1
        marker.write_text(str(n))
        noisy = payload['code'] in ('transient', 'persistent') and case == payload['cases'][0]
        floor = 14.92 if noisy and (n == 1 or payload['code'] == 'persistent') else .05
        result = dict(passed=True, score=2, grad_ok=True, grad_scores={case:2}, task_noise_floor=floor)
    finally:
        lease.unlink(missing_ok=True)
output.write_text(json.dumps(result))
''')
    shim.chmod(0o755)
    yield tmp_path
    ray.shutdown()


def payload(root, code='valid'):
    return dict(problem='rg_lru', cases=[n for n, _ in public_contract()[1]],
                code=code, enforce_pallas=True, test_root=str(root))


def test_real_tasks_share_four_chips_and_retry_only_noisy_case(cluster):
    def grade(code):
        return module.grade_candidate(cluster, 'test', payload(cluster, code), timeout_s=120)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(grade, ['valid', 'transient', 'valid']))
    assert all(translate_verdict(r)['correctness'] == 1 for r in results)
    assert all(r['dispatch'] == 'ray_tasks' for r in results)
    noisy = payload(cluster)['cases'][0]
    assert results[1]['case_attempts'][noisy] == 2
    assert all(n == 1 for c, n in results[1]['case_attempts'].items() if c != noisy)
    assert not list(cluster.glob('chip-*'))
    rejected = grade('invalid')
    assert rejected['gate'] == 'pregate' and translate_verdict(rejected)['reward'] == 0
    persistent = grade('persistent')
    assert persistent['case_attempts'][noisy] == 3
    with pytest.raises(ArenaInfrastructureError):
        translate_verdict(persistent)


def test_frozen_client_submits_32_completions_to_tasks(cluster):
    from tpu.swarm.ray_train.overlay import ARENA_FILES
    repo = Path(__file__).resolve().parents[2]
    source = cluster / 'frozen'
    for name in ARENA_FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
    code = r'''
import ast, json, os, sys, uuid, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
source = Path(os.environ['GRADER_SOURCE'])
sys.path[:0] = [str(source / 'tpu'), str(source)]
from pallas_arena.rl.task import public_contract, translate_verdict, ArenaInfrastructureError
tree = ast.parse((source / 'tpu/pallas_arena/rl/env.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RecurrentGemmaRewardEvaluator')
ns = dict(globals(), BaseRewardEvaluator=object, State=object)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'evaluator', 'exec'), ns)
e = ns['RecurrentGemmaRewardEvaluator']('rg_lru', source / 'evidence', eval_timeout=120)
try:
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: e.get_reward('invalid', None), range(32)))
    assert all(r['correctness'] == 0 and r['reward'] == 0 for r in results)
    print('TASK_TRANSPORT_PASS', flush=True)
finally:
    import ray
    ray.shutdown()
'''
    env = dict(os.environ, GRADER_SOURCE=str(source), ARENA_RAY_TASKS='1',
               ARENA_RAY_ACTOR='', ARENA_RAY_ROOT=str(cluster), ARENA_WAIT_TIMEOUT='110',
               RAY_ADDRESS=ray.get_runtime_context().gcs_address, RAY_NAMESPACE='grader-tasks-test')
    result = subprocess.run([sys.executable, '-c', code], cwd=source, env=env,
                            capture_output=True, text=True, timeout=140)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TASK_TRANSPORT_PASS' in result.stdout
    evidence = list((source / 'evidence/arena').glob('*.json'))
    assert len(evidence) == 32
    assert all(json.loads(p.read_text())['result']['dispatch'] == 'ray_tasks' for p in evidence)


def test_guardian_holds_chip_until_child_cleanup_after_owner_is_killed(tmp_path):
    guard = Path(module.__file__).with_name('process.py')
    child = tmp_path / 'child.py'
    child.write_text('''import pathlib, signal, sys, time
p = pathlib.Path(sys.argv[1])
p.write_text("active")
def stop(*args):
    time.sleep(.3)
    p.unlink()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
time.sleep(60)
''')
    owner = tmp_path / 'owner.py'
    owner.write_text('''import json, os, subprocess, sys, time
subprocess.Popen([sys.executable, sys.argv[1], 'guard', str(os.getpid()),
                  json.dumps([sys.executable, sys.argv[2], sys.argv[3]]), '1', sys.argv[4]],
                 start_new_session=True)
time.sleep(60)
''')
    active, second = tmp_path / 'active', tmp_path / 'second'
    lock = tmp_path / 'chip.lock'
    first = subprocess.Popen([sys.executable, str(owner), str(guard), str(child), str(active), str(lock)])
    replacement = None
    try:
        deadline = time.monotonic() + 5
        while not active.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert active.exists()
        # SIGKILL bypasses any finally block in the task/owner process.
        first.kill()
        first.wait(timeout=5)
        command = [sys.executable, '-c',
                   'import pathlib,sys; assert not pathlib.Path(sys.argv[1]).exists(); pathlib.Path(sys.argv[2]).touch()',
                   str(active), str(second)]
        replacement = subprocess.Popen([sys.executable, str(guard), 'guard', str(os.getpid()),
                                        json.dumps(command), '1', str(lock)])
        assert replacement.wait(timeout=6) == 0
        assert second.exists() and not active.exists()
    finally:
        if first.poll() is None:
            first.kill()
        first.wait(timeout=5)
        if replacement is not None and replacement.poll() is None:
            replacement.terminate()
            replacement.wait(timeout=5)
