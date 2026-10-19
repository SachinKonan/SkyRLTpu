"""SimpleTES evaluator that grades through the RL client's own Environment.

Drop-in for ``simpletes.evaluator.EvaluatorWorker``: SimpleTES awaits
``evaluate(code, shared_construction_path=...)`` and expects an
``EvaluationOutcome(metrics, captured_construction_payload)``. Here every
candidate goes through ``Environment._safe_grade`` -- the same sandbox, Ray
transport, timeout and scoring the RL rollouts used -- so a SimpleTES score and an
RL score are the same function of the program.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .tasks import Task, candidate_source

MSG_CHARS = 2000
# A grade that fails for infrastructure reasons (farm lost, transport gave up) is retried here,
# never charged to the candidate: the RQ4 client aborts and redoes the step in that case.
INFRA_RETRIES, INFRA_RETRY_SECONDS = 20, 120.0


class RLEnvironmentEvaluator:
    def __init__(self, task: Task, root_state, *, log_dir: Path, eval_backend: str | None = None,
                 num_cpus_per_task: int | None = None, eval_timeout: int | None = None,
                 infra_retries: int = INFRA_RETRIES, infra_retry_seconds: float = INFRA_RETRY_SECONDS):
        from ttt_discover.tinker_utils.dataset_builder import DatasetConfig
        from .compat import import_simpletes
        import_simpletes()   # vendored package first: `tpu/` on sys.path would shadow `simpletes`
        from simpletes.evaluator import EvaluationOutcome
        from simpletes.construction import decode_construction, encode_construction, read_payload

        self.task = task
        self.env_type = task.env_type()
        self.root_state = root_state
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = eval_timeout or task.timeout()
        self.infra_retries, self.infra_retry_seconds = infra_retries, infra_retry_seconds
        self._outcome = EvaluationOutcome
        self._decode, self._encode, self._read = decode_construction, encode_construction, read_payload
        self._counter = itertools.count()
        self.settings = DatasetConfig(
            problem_type=task.problem_type, env_type=self.env_type, batch_size=1,
            model_name_for_tokenizer="", renderer_name="", group_size=1,
            num_cpus_per_task=num_cpus_per_task or task.num_cpus_per_task,
            eval_backend=eval_backend or task.eval_backend, eval_timeout=self.timeout,
            # Same outer asyncio guard the RL client uses (max(8000, timeout + 60)).
            timeout=max(8000, self.timeout + 60), log_path=str(self.log_dir))
        # combined_score is higher-is-better: the RL reward, or -metric for minimize tasks (Erdős).
        self.keep_fences = self._environment(root_state)._should_keep_code_separators()

    def _environment(self, state):
        return self.env_type(renderer=None, initial_state=state, sampler=object(), config=self.settings)

    def state_for(self, shared_construction_path: str | None):
        """Root state, carrying the chain's best construction when SimpleTES has one."""
        if not shared_construction_path or not self.task.construction_global:
            return self.root_state
        value = self._decode(self._read(shared_construction_path))
        construction = [float(x) for x in np.asarray(value, dtype=float).ravel()]
        return self.env_type.state_type(timestep=-1, code="", value=self.root_state.value,
                                        construction=construction)

    async def evaluate(self, code: str, *, shared_construction_path: str | None = None):
        source = candidate_source(self.task, code)
        if self.keep_fences:
            source = f"```python\n{source}\n```"
        env = self._environment(self.state_for(shared_construction_path))
        step = next(self._counter)
        for attempt in range(self.infra_retries + 1):
            try:
                verdict = await env._safe_grade(source, step)
                break
            except Exception as exc:
                if not getattr(exc, "abort_training_step", False) or attempt == self.infra_retries:
                    raise
                with (self.log_dir / "infra_retries.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(time=time.time(), step=step, attempt=attempt + 1,
                                                 error=f"{type(exc).__name__}: {exc}"[:500])) + "\n")
                await asyncio.sleep(self.infra_retry_seconds)
        return self._outcome(metrics=self.metrics(verdict),
                             captured_construction_payload=self.payload(verdict))

    def metrics(self, verdict) -> dict[str, Any]:
        valid = verdict.correctness == 1 and np.isfinite(verdict.reward) and np.isfinite(verdict.raw_score)
        out: dict[str, Any] = {
            # combined_score is the RL reward: 1/(1e-8+C5) on Erdős (SimpleTES's own
            # Erdős score), C2 on AC2 -- higher is better in both frameworks.
            # Minimize tasks (Erdős): -metric, our State.value, so scores read as in our arm.
            "combined_score": ((-float(verdict.raw_score) if self.task.minimize else float(verdict.reward))
                               if valid else 0.0),
            "validity": 1.0 if valid else 0.0,
        }
        if valid:
            out[self.task.metric] = float(verdict.raw_score)
        else:
            out["error"] = (verdict.msg or "invalid")[:MSG_CHARS]
        for key, value in (verdict.metrics or {}).items():
            if isinstance(value, (int, float, str, bool)) and key not in out:
                out[key] = value
        return out

    def payload(self, verdict):
        if verdict.correctness != 1 or verdict.result_construction is None or not self.task.construction_global:
            return None
        return self._encode(np.asarray(verdict.result_construction, dtype=float))
