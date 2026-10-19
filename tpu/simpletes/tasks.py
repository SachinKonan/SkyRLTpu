"""Problem registry: how a SimpleTES run maps onto one of our RL environments.

Every task grades through the discover ``Environment`` the RL client uses, so the
sandbox, timeout, CPU count and score are the RL ones by construction. SimpleTES
needs three extra things per problem, all derived from that environment:

* instruction -- ``env.get_question()`` for the task's root state (the RL prompt);
* init program -- the root state's code, or ``default_seed`` when the RL root has
  none (Erdős starts from a random construction and no program), wrapped whole in
  EVOLVE-BLOCK markers;
* construction carry -- SimpleTES's per-chain ``GLOBAL_BEST_CONSTRUCTION`` becomes
  the ``State.construction`` our grader injects as ``construction_global``; the
  candidate gets a one-line alias so either name works.
"""
from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass

EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"

_ERDOS_SEED = '''import numpy as np


def run(seed=42, budget_s=1000, **kwargs):
    """Seed: return the injected starting construction unchanged."""
    h = np.asarray(initial_h_values, dtype=float)
    n = len(h)
    c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n)))
    return h, c5, n
'''


@dataclass(frozen=True)
class Task:
    name: str
    env_module: str
    env_class: str
    problem_type: str
    metric: str                    # name of raw_score in SimpleTES metrics
    construction_global: str | None
    eval_timeout: int
    num_cpus_per_task: int
    default_seed: str | None = None
    science: bool = False          # RQ4 science env (tpu.science.training_env), graded on the farms
    minimize: bool = False         # combined_score = -metric, i.e. our State.value (higher is better)
    eval_backend: str = "ray"

    def env_type(self):
        return getattr(importlib.import_module(self.env_module), self.env_class)

    def timeout(self) -> int:
        """Science tasks use the run's EVAL_TIMEOUT, as the RQ4 client does (72000 qubit, 172800 circuit)."""
        return int(os.environ.get("EVAL_TIMEOUT", self.eval_timeout)) if self.science else self.eval_timeout


TASKS = {
    "erdos": Task("erdos", "examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "",
                  "c5_bound", "initial_h_values", 1100, 1, _ERDOS_SEED, minimize=True),
    "ac1": Task("ac1", "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac1",
                "c1", "height_sequence_1", 1100, 2),
    "ac2": Task("ac2", "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac2",
                "c2", "height_sequence_1", 1100, 2),
    # RQ4 qubit and circuit: four-CPU tasks inside the local outer evaluator (ScienceTrainingReward).
    "qubit": Task("qubit", "tpu.science.training_env", "RoutingTrainingEnv", "routing",
                  "score", None, 72000, 4, science=True, eval_backend="local"),
    "circuit": Task("circuit", "tpu.science.training_env", "PlacementTrainingEnv", "placement",
                    "score", None, 172800, 4, science=True, eval_backend="local"),
}

# Where the RQ4 prompt's starter program begins (training_env.task_prompt's starter boundary).
STARTER_MARKERS = {"routing": "Valid initial policy block:",
                   "placement": "Starting implementation (replace with your improved algorithm):"}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown SimpleTES task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]


def strip_fences(code: str) -> str:
    match = re.fullmatch(r"\s*```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*", code, re.DOTALL)
    return match.group(1) if match else code


def science_instruction(task: Task) -> str:
    """The RQ4 prompt without its starter program (SimpleTES shows programs separately)."""
    from tpu.science.training_env import task_prompt
    return task_prompt(task.problem_type, include_starter=False)


def science_starter(task: Task) -> str:
    """The RQ4 starter as a candidate program, in the answer format the RQ4 prompt asks for.

    Circuit answers are a Python module, so the starter is used as is. Qubit answers are a
    Python module holding one literal ``RUST_CODE = r'''...'''`` with the policy block, so the
    Rust starter is wrapped that way.
    """
    from tpu.science.training_env import task_prompt
    prompt = task_prompt(task.problem_type, include_starter=True)
    _, separator, rest = prompt.partition(STARTER_MARKERS[task.problem_type])
    match = re.search(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)\n?```", rest, re.DOTALL) if separator else None
    if match is None:
        raise ValueError(f"{task.name}: RQ4 prompt has no starter block")
    language, code = match.group(1), match.group(2).strip("\n")
    if task.problem_type == "routing":
        if language != "rust" or "'''" in code:
            raise ValueError("qubit starter must be a Rust block that fits a raw triple-quoted string")
        return f"RUST_CODE = r'''\n{code}\n'''"
    return code


def init_program(task: Task, root_code: str) -> str:
    body = strip_fences(root_code or "").strip("\n") or (task.default_seed or "").strip("\n")
    if not body:
        raise ValueError(f"task {task.name} has no root program and no default seed")
    return f"{EVOLVE_START}\n{body}\n{EVOLVE_END}\n"


def strip_markers(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if line.strip() not in (EVOLVE_START, EVOLVE_END)).strip("\n") + "\n"


def candidate_source(task: Task, code: str) -> str:
    """Program handed to the RL grader: alias SimpleTES's global onto ours.

    Science tasks grade the code without SimpleTES's marker lines, i.e. the same text the
    RQ4 client extracts with ``answer_only_code``.
    """
    if task.science:
        return strip_markers(code)
    if not task.construction_global:
        return code
    alias = f"GLOBAL_BEST_CONSTRUCTION = globals().get({task.construction_global!r})"
    return f"{alias}\n{code}"
