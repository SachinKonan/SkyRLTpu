"""One workload-Ray actor owns a grader host; candidates run in fresh children."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import time
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tpu"))

from pallas_arena.judge import collect
from pallas_arena.judge.observation import attach_observation
from pallas_arena.rl.task import ArenaInfrastructureError, public_contract, translate_verdict


class Grader:
    def __init__(self, root, run_id):
        import ray
        self.root = Path(root)
        self.run = self.root / 'runs' / run_id / 'grader'
        self.run.mkdir(parents=True, exist_ok=True)
        ids = ray.get_runtime_context().get_accelerator_ids().get('TPU', [])
        self.chips = [str(int(i)) for i in ids]
        if len(self.chips) != 4 or len(set(self.chips)) != 4:
            raise RuntimeError(f'Grader must exclusively own four chips, got {self.chips}')
        self.available = asyncio.Queue()
        for chip in self.chips:
            self.available.put_nowait(chip)
        self.pregate_slots = asyncio.Semaphore(2)
        self.cases = [n for n, _ in public_contract()[1]]
        if any(collect.case_width(n) != 1 for n in self.cases):
            raise ValueError('Dedicated grader currently supports the single-chip public RG-LRU suite')
        self.processes = set()
        self.requests = set()
        self.closed = False
        self.submitted = self.completed = 0

    async def status(self):
        return dict(ok=not self.closed, chips=self.chips, cases=self.cases,
                    submitted=self.submitted, completed=self.completed,
                    active=len(self.requests), children=len(self.processes),
                    free_chips=self.available.qsize())

    async def self_test(self):
        """Require a valid full-suite seed and a rejected invalid candidate."""
        code = (Path(__file__).resolve().parents[3] /
                'tpu/pallas_arena/rl/seed_rglru.py').read_text()
        payload = dict(problem='rg_lru', cases=self.cases, code=code,
                       enforce_pallas=True, tag='grader-self-test-valid')
        valid = await self.grade(payload)
        from pallas_arena.rl.task import translate_verdict
        reward = translate_verdict(valid)
        if not valid.get('passed'):
            raise RuntimeError(f'Grader seed self-test failed: {valid}')
        invalid = await self.grade(dict(payload, code='def kernel(x, a, reset): return x',
                                        tag='grader-self-test-invalid'))
        if invalid.get('passed') or invalid.get('gate') != 'pregate':
            raise RuntimeError(f'Grader failed to reject a non-Pallas candidate: {invalid}')
        result = dict(ok=True, valid=valid, invalid=invalid, reward=reward)
        (self.run / 'self-test.json').write_text(json.dumps(result))
        return result

    async def _child(self, mode, payload, tag, chip=None, case=None):
        token = uuid.uuid4().hex
        folder = self.run / tag
        folder.mkdir(exist_ok=True)
        request, output = folder / (token + '.request.json'), folder / (token + '.result.json')
        request.write_text(json.dumps(dict(mode=mode, payload=payload, case=case, chip=chip,
            cache=str(self.root / 'arena-cache/tpu-v5p-32/jax-0.10.2'))))
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
            env.update(TPU_VISIBLE_CHIPS=chip, TPU_PROCESS_BOUNDS='1,1,1',
                       TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1')
        proc = None
        with (folder / (token + '.log')).open('wb') as log:
            try:
                proc = await asyncio.create_subprocess_exec(
                    str(self.root / 'envs/arena/bin/python'), '-m',
                    'tpu.swarm.ray_train.grader_child', str(request), str(output),
                    env=env, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                self.processes.add(proc)
                await asyncio.wait_for(proc.wait(), timeout=150 if mode == 'pregate' else 1800)
                if proc.returncode:
                    tail = (folder / (token + '.log')).read_bytes()[-3000:].decode(errors='replace')
                    raise RuntimeError(f'Grader child exited {proc.returncode}: {tail}')
                return json.loads(output.read_text())
            finally:
                if proc is not None:
                    # Kill descendants too, and wait before releasing the chip.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                    self.processes.discard(proc)
                request.unlink(missing_ok=True)

    async def _case(self, case, payload, tag):
        chip = await self.available.get()
        try:
            history = []
            for attempt in range(3):
                try:
                    result = await self._child('case', payload, tag, chip, case)
                    entry = {'result': result}
                except Exception as exc:
                    why = f'{type(exc).__name__}: {exc}'
                    if collect.classify_task_error(why) == 'fatal':
                        return case, {'fatal': ('runtime_halt', why)}
                    entry = {'judge_fault': why}
                check = collect.merge_case_results('rg_lru', {case: entry},
                    general_mode=True, has_bwd=True, default_floor=.05)
                # A candidate error is final. Only missing/untrustworthy
                # measurements get another fresh child and calibration.
                candidate_failed = not check.get('passed') and check.get('gate') != 'judge_fault'
                complete = (check.get('passed') and not check['excluded_cases']
                            and not check['skipped_cases'] and check.get('n_bwd_factors') == 1
                            and set(check['grad_scores']) == {case})
                if candidate_failed or complete:
                    return case, dict(entry, attempts=attempt + 1, retry_history=history)
                why = str(check.get('excluded_cases') or check.get('violations')
                          or 'incomplete backward timing')
                history.append(why)
            return case, dict(judge_fault=f'Grader case failed after 3 attempts: {why}',
                              attempts=3, retry_history=history)
        finally:
            self.available.put_nowait(chip)

    async def grade(self, payload, timeout_s=14400):
        if self.closed:
            raise RuntimeError('Grader is shutting down')
        tag = uuid.uuid4().hex
        task = asyncio.current_task()
        self.requests.add(task)
        self.submitted += 1
        started = time.monotonic()
        pending = []
        try:
            async with asyncio.timeout(timeout_s):
                if payload.get('problem') != 'rg_lru' or payload.get('cases') != self.cases:
                    raise ValueError('Grading request must match the full RG-LRU public suite')
                async with self.pregate_slots:
                    pre = await self._child('pregate', payload, tag)
                entries = {}
                if pre.get('passed') is False:
                    entries['__pregate__'] = {'fatal': ('pregate', str((pre.get('violations') or ['pregate failed'])[0]))}
                else:
                    pending = [asyncio.create_task(self._case(c, payload, tag)) for c in self.cases]
                    for done in asyncio.as_completed(pending):
                        case, entry = await done
                        entries[case] = entry
                        r = entry.get('result', {})
                        fatal = 'fatal' in entry or (r and not r.get('passed')
                            and r.get('gate') not in collect.JUDGE_FAULT_GATES
                            and case not in (r.get('skipped_tp') or {}))
                        if fatal:
                            break
                merged = collect.merge_case_results('rg_lru', entries, general_mode=True,
                                                     has_bwd=True, default_floor=.05)
                merged.update(item_wall_s=round(time.monotonic()-started, 1),
                              tag=payload.get('tag'), baseline_mode='all',
                              case_attempts={c: e.get('attempts', 1) for c, e in entries.items()},
                              retry_history={c: e['retry_history'] for c, e in entries.items()
                                             if e.get('retry_history')})
                # The generic collector allows partial timing suites. RG-LRU
                # training requires every public forward/backward factor.
                try:
                    translate_verdict(merged)
                except ArenaInfrastructureError as exc:
                    merged.update(ok=False, passed=False, judge_fault=True,
                                  gate='judge_fault', violations=[str(exc)],
                                  reward=0.0, reward_with_bwd=0.0, score=0.0)
                attach_observation(merged)
                (self.run / tag / 'verdict.json').write_text(json.dumps(merged))
                self.completed += 1
                return merged
        finally:
            for child in pending:
                if not child.done():
                    child.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self.requests.discard(task)

    async def close(self):
        self.closed = True
        tasks = list(self.requests)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return await self.status()
