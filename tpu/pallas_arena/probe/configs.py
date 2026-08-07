"""The 12 probe configurations: 2 models x 3 prompt variants x 2 tasks.

Nothing here runs anything. It is the single place that says what a
configuration IS, so the driver, the pre-gate, the judge launch flags and the
report all agree on the same 12 cells.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------- tasks
# The one-chip PROBE case sets (judge/problems/*.py, `probe=True`). These are
# the shapes the prompts declare, the shapes the pre-gate exports, and the
# shapes the judge grades -- one list, used three times, so a candidate is
# never graded against a contract it was not shown.
TASK_CASES = {
    "splash_attention": ["probe-h8-s4096", "probe-h4-s2048", "probe-holdout-h4-s2049"],
    "flce": ["probe-4096x2880x151936", "probe-2048x2880x151936", "probe-holdout-3000x2880x151936"],
}
TASKS = tuple(TASK_CASES)
VARIANTS = ("minimal", "reference", "tailored")

# --------------------------------------------------------------------- models
# Exact HF ids as used elsewhere in this repo (note gemma's capital B).


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    renderer: str
    worker: int  # v5p-16 host index that serves it
    stop: str  # end-of-turn marker, for /v1/completions
    max_model_len: int  # the served context window
    # Longest generation that still fits behind the LONGEST prompt. Measured
    # the hard way: gemma is served at 16384, its tailored splash prompt is
    # ~4.4k tokens, and 4.4k + 12000 > 16384 -- so vLLM returned
    # `HTTPError 400: Bad Request` for all 16 of that cell's requests and the
    # cell recorded 16 empty generations that LOOK like model failures.
    # Cap per model, never globally.
    max_new_tokens: int


MODELS = {
    "qwen35-27b": ModelSpec("qwen35-27b", "Qwen/Qwen3.5-27B", "qwen3", 0, "<|im_end|>", 22528, 16000),
    "gemma4-31b": ModelSpec("gemma4-31b", "google/gemma-4-31B-it", "gemma4", 1, "<turn|>", 16384, 10000),
}


@dataclass(frozen=True)
class ProbeConfig:
    model: str
    variant: str
    task: str

    @property
    def name(self) -> str:
        return f"{self.model}|{self.variant}|{self.task}"


def all_configs(models=None, variants=None, tasks=None) -> list[ProbeConfig]:
    models = list(models or MODELS)
    variants = list(variants or VARIANTS)
    tasks = list(tasks or TASKS)
    return [ProbeConfig(m, v, t) for m in models for v in variants for t in tasks]


# ---------------------------------------------------------------- generation
# Kernel programs are short next to an erdos rollout. The tailored prompt is
# the longest input; the longest sane output is a fully commented Pallas
# kernel with a custom_vjp, which is ~600 lines of python at the very worst.
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
TOP_P = 1.0
GROUP_SIZE = 16  # ttt-discover computes advantage WITHIN a group of this size
