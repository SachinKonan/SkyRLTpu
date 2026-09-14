"""Per-case RG-LRU Ray tasks; Ray owns the queue and TPU allocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import ray

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tpu"))
from pallas_arena.judge import collect
from pallas_arena.judge.observation import attach_observation
from pallas_arena.rl.task import ArenaInfrastructureError, public_contract, translate_verdict


def run_child(root, run_id, mode, payload, tag, case=None):
    root = Path(root)
    folder = root / 'runs' / run_id / 'grader' / tag
    folder.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    request, output = folder / (token + '.request.json'), folder / (token + '.result.json')
    chip = None
    if mode == 'case':
        ids = ray.get_runtime_context().get_accelerator_ids().get('TPU', [])
        if len(ids) != 1:
            raise RuntimeError(f'Grading task requires exactly one TPU, got {ids}')
        chip = str(int(ids[0]))
    request.write_text(json.dumps(dict(mode=mode, payload=payload, case=case, chip=chip,
        cache=str(root / 'arena-cache/tpu-v5p-32/jax-0.10.2'))))
    code = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=f'{code / "tpu"}:{code}',
               JAX_PLATFORMS='cpu' if mode == 'pregate' else 'tpu',
               ARENA_CHILD_JAX_PLATFORMS='cpu' if mode == 'pregate' else 'tpu',
               ARENA_BASELINE='all', PALLAS_INTERPRET='0', ARENA_RLIMIT_GB='64',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for key in ('RAY_ADDRESS', 'RAY_NAMESPACE', 'RAY_TMPDIR', 'TPU_VISIBLE_CHIPS',
                'TPU_PROCESS_ADDRESSES', 'TPU_PROCESS_PORT', 'TPU_PROCESS_BOUNDS',
                'TPU_CHIPS_PER_PROCESS_BOUNDS', 'CLOUD_TPU_TASK_ID', 'JAX_COORDINATOR_ADDRESS'):
        env.pop(key, None)
    if chip is not None:
        env.update(TPU_VISIBLE_CHIPS=chip, TPU_PROCESS_BOUNDS='1,1,1', TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1')
    child = [str(root / 'envs/arena/bin/python'), '-m',
             'tpu.swarm.ray_train.grader_child', str(request), str(output)]
    command = [sys.executable, str(Path(__file__).with_name('process.py')), 'guard',
               str(os.getpid()), json.dumps(child), '15']
    if chip is not None:
        locks = root / 'grader-chip-locks'
        locks.mkdir(exist_ok=True)
        command.append(str(locks / f'{chip}.lock'))
    proc = None
    log_path = folder / (token + '.log')
    try:
        with log_path.open('wb') as log:
            proc = subprocess.Popen(command, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            proc.wait(timeout=180 if mode == 'pregate' else 1830)
            if proc.returncode:
                raise RuntimeError(f'Grader child exited {proc.returncode}: '
                                   + log_path.read_bytes()[-3000:].decode(errors='replace'))
            return json.loads(output.read_text())
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
            # Do not forcibly kill the guardian: it holds the chip lock until
            # its whole child process group has been terminated and reaped.
            proc.wait(timeout=35)
        request.unlink(missing_ok=True)


@ray.remote(num_cpus=2, resources={'arena_pregate': 1}, max_calls=1, max_retries=0)
def pregate_task(root, run_id, payload, tag):
    return run_child(root, run_id, 'pregate', payload, tag)


@ray.remote(num_cpus=2, resources={'TPU': 1, 'arena_grader': 1}, max_calls=1, max_retries=0)
def case_task(root, run_id, payload, tag, case):
    try:
        return {'result': run_child(root, run_id, 'case', payload, tag, case)}
    except Exception as exc:
        why = f'{type(exc).__name__}: {exc}'
        return ({'fatal': ('runtime_halt', why)} if collect.classify_task_error(why) == 'fatal'
                else {'judge_fault': why})


def case_outcome(case, entry):
    check = collect.merge_case_results('rg_lru', {case: entry},
        general_mode=True, has_bwd=True, default_floor=.05)
    fatal = not check.get('passed') and check.get('gate') != 'judge_fault'
    complete = (check.get('passed') and not check['excluded_cases']
                and not check['skipped_cases'] and check.get('n_bwd_factors') == 1
                and set(check['grad_scores']) == {case})
    return fatal, complete, str(check.get('excluded_cases') or check.get('violations')
                                or 'incomplete backward timing')


def grade_candidate(root, run_id, payload, timeout_s=14400):
    """Client-side orchestration; no long-lived actor reserves grader TPUs."""
    cases = [n for n, _ in public_contract()[1]]
    if payload.get('problem') != 'rg_lru' or payload.get('cases') != cases:
        raise ValueError('Grading request must match the full RG-LRU public suite')
    started = time.monotonic()
    tag = uuid.uuid4().hex
    deadline = started + timeout_s
    pending = {}
    attempts = dict.fromkeys(cases, 0)
    history = {c: [] for c in cases}
    entries = {}
    def submit(case):
        attempts[case] += 1
        ref = case_task.remote(str(root), run_id, payload, tag, case)
        pending[ref] = case
    try:
        pre_ref = pregate_task.remote(str(root), run_id, payload, tag)
        pending[pre_ref] = '__pregate__'
        pre = ray.get(pre_ref, timeout=max(0, deadline - time.monotonic()))
        pending.pop(pre_ref)
        if pre.get('passed') is False:
            entries['__pregate__'] = {'fatal': ('pregate', str((pre.get('violations') or ['pregate failed'])[0]))}
        elif pre.get('passed') is not True:
            raise ArenaInfrastructureError('Grader pregate returned no verdict')
        else:
            for case in cases:
                submit(case)
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1,
                                    timeout=max(0, deadline - time.monotonic()))
                if not ready:
                    raise ArenaInfrastructureError(f'Grading timed out after {timeout_s}s')
                ref = ready[0]
                case = pending.pop(ref)
                try:
                    entry = ray.get(ref)
                except Exception as exc:
                    why = f'{type(exc).__name__}: {exc}'
                    entry = ({'fatal': ('runtime_halt', why)} if collect.classify_task_error(why) == 'fatal'
                             else {'judge_fault': why})
                fatal, complete, why = case_outcome(case, entry)
                if not fatal and not complete:
                    history[case].append(why)
                    if attempts[case] < 3:
                        submit(case)
                        continue
                    entry = {'judge_fault': f'Grader case failed after 3 attempts: {why}'}
                entries[case] = entry
                if fatal:
                    break
        merged = collect.merge_case_results('rg_lru', entries,
            general_mode=True, has_bwd=True, default_floor=.05)
        merged.update(item_wall_s=round(time.monotonic() - started, 1), baseline_mode='all',
                      tag=payload.get('tag'), case_attempts=attempts,
                      retry_history={c: h for c, h in history.items() if h}, dispatch='ray_tasks')
        try:
            translate_verdict(merged)
        except ArenaInfrastructureError as exc:
            merged.update(ok=False, passed=False, judge_fault=True, gate='judge_fault',
                          violations=[str(exc)], reward=0.0, reward_with_bwd=0.0, score=0.0)
        attach_observation(merged)
        return merged
    finally:
        for ref in pending:
            try:
                ray.cancel(ref, force=False, recursive=True)
            except Exception:
                pass


@ray.remote(num_cpus=0, max_retries=0)
def self_test(root, run_id):
    code = (Path(__file__).resolve().parents[3] / 'tpu/pallas_arena/rl/seed_rglru.py').read_text()
    payload = dict(problem='rg_lru', cases=[n for n, _ in public_contract()[1]],
                   code=code, enforce_pallas=True, tag='grader-self-test-valid')
    valid = grade_candidate(root, run_id, payload, timeout_s=6900)
    reward = translate_verdict(valid)
    if not valid.get('passed'):
        raise RuntimeError(f'Grader seed self-test failed: {valid}')
    invalid = grade_candidate(root, run_id, dict(payload, code='def kernel(x,a,reset): return x'), timeout_s=180)
    if invalid.get('passed') or invalid.get('gate') != 'pregate':
        raise RuntimeError(f'Grader failed to reject non-Pallas candidate: {invalid}')
    return dict(ok=True, valid=valid, invalid=invalid, reward=reward)
