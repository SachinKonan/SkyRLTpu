"""RecurrentGemma kernel generation using the existing remote Arena queue."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import uuid
import threading

from ttt_discover import BaseRewardEvaluator, Environment, State
from pallas_arena.judge.client import ArenaQueueClient
from pallas_arena.rl.task import (
    ArenaInfrastructureError, build_prompt, public_contract, translate_verdict,
    validate_queue_url,
)


class RecurrentGemmaRewardEvaluator(BaseRewardEvaluator):
    # get_reward runs concurrently in asyncio.to_thread across completions.
    # Ray initialization is process-wide, not evaluator-instance-wide.
    _ray_init_lock = threading.Lock()

    def __init__(self, problem_type, log_dir, eval_timeout=3600, **kwargs):
        if problem_type not in ("", "rg_lru"):
            raise ValueError("RecurrentGemma currently supports only problem_type=rg_lru")
        self.actor_name = os.environ.get("ARENA_RAY_ACTOR", "")
        self.task_mode = os.environ.get("ARENA_RAY_TASKS", "") == "1"
        self.url = None if self.actor_name or self.task_mode else validate_queue_url(os.environ.get("ARENA_QUEUE_URL", ""))
        self.timeout = float(os.environ.get("ARENA_WAIT_TIMEOUT", str(eval_timeout)))
        if not 0 < self.timeout <= eval_timeout:
            raise ValueError("ARENA_WAIT_TIMEOUT must be positive and <= EVAL_TIMEOUT")
        self.log_dir = Path(log_dir) / "arena"

    def get_reward(self, code: str, state: State) -> dict:
        # This evaluator transports source only. Generated code is never
        # imported or executed on the training/inference hosts.
        tag = uuid.uuid4().hex
        cases = [name for name, _ in public_contract()[1]]
        if self.actor_name or self.task_mode:
            import ray
            ref = None
            try:
                with self._ray_init_lock:
                    if not ray.is_initialized():
                        ray.init(address=os.environ["RAY_ADDRESS"], namespace=os.environ["RAY_NAMESPACE"],
                                 ignore_reinit_error=True)
                work_id = tag
                payload = dict(problem="rg_lru", code=code, cases=cases, enforce_pallas=True, tag=tag)
                if self.task_mode:
                    from tpu.swarm.ray_train.grader_tasks import grade_candidate
                    result = grade_candidate(os.environ["ARENA_RAY_ROOT"], os.environ["RAY_NAMESPACE"],
                                             payload, timeout_s=self.timeout)
                else:
                    actor = ray.get_actor(self.actor_name, namespace=os.environ["RAY_NAMESPACE"])
                    ref = actor.grade.remote(payload, timeout_s=self.timeout)
                    result = ray.get(ref, timeout=self.timeout + 30)
            except Exception as exc:
                if ref is not None:
                    try:
                        ray.cancel(ref)
                    except Exception:
                        pass  # Preserve the original transport error.
                raise ArenaInfrastructureError(f"RG grader {'tasks' if self.task_mode else 'actor'} failed: {exc}") from exc
        else:
            client = ArenaQueueClient(self.url, timeout_s=min(30.0, self.timeout))
            work_id = client.submit("rg_lru", code, mode="full", smoke=False,
                                    cases=cases, enforce_pallas=True, tag=tag)
            result = client.wait([work_id], timeout_s=self.timeout).get(work_id)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / f"{tag}.json").write_text(json.dumps(dict(
            work_id=work_id, tag=tag, parent_id=getattr(state, "id", None),
            problem="rg_lru", cases=cases, code=code, result=result), indent=2))
        if result is None:
            raise ArenaInfrastructureError(f"Arena queue did not return a verdict within {self.timeout}s; work_id={work_id}")
        return translate_verdict(result)


class RecurrentGemmaEnv(Environment):
    reward_function = RecurrentGemmaRewardEvaluator
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type not in ("", "rg_lru"):
            raise ValueError("RecurrentGemma currently supports only rg_lru")
        return State(timestep=-1, construction=None, value=0.0,
                     code=Path(__file__).with_name("seed_rglru.py").read_text(),
                     observation="Starting seed; not graded in this run. No speedup measurement yet.")

    def _should_keep_code_separators(self) -> bool:
        return False

    def get_question(self) -> str:
        state = self.initial_state
        progress = state.observation[-480:]
        if state.timestep >= 0:
            progress = f"Measured raw speedup: {state.value:.6g}.\n" + progress
        return build_prompt() + f"\nCurrent candidate:\n```python\n{state.code}\n```\nJudge feedback:\n{progress}\nImprove this candidate."

    async def _safe_grade(self, given_answer: str, step: int):
        # The generic wrapper converts ALL exceptions/timeouts to zero rewards.
        # Queue and judge failures must instead abort the affected rollout.
        return await asyncio.to_thread(self._run_verification, given_answer,
                                       self.problem_type, self.log_path, self.state)


PallasRgLruEnv = RecurrentGemmaEnv
