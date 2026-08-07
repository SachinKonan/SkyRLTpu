"""The probe configurations: models x prompt variants x tasks.

Nothing here runs anything. It is the single place that says what a
configuration IS, so the driver, the pre-gate, the judge launch flags and the
report all agree on the same cells.

The seam run is 2 models x 2 variants x 5 tasks = 20 cells. `reference` is
carried forward unchanged as the CONTROL arm -- it was the previous probe's
only trainable cell (gemma x FLCE, within-group spread 0.1978) -- and `seam` is
the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------- tasks
# The one-chip PROBE case sets (judge/problems/*.py, `probe=True`). These are
# the shapes the prompts declare, the shapes the pre-gate exports, and the
# shapes the judge grades -- one list, used three times, so a candidate is
# never graded against a contract it was not shown.
#
# Every set keeps the axis the task exists for at its production width (FLCE's
# 151936 vocab, splash's 128 head_dim, RPA's 8 kv heads / 32 q heads / 64 page
# size, GMM's k=4096 and n=14336, RG-LRU's d=2560) and shrinks only the token /
# sequence / batch axis, so one 32 GB judge can hold the fp32 REFERENCE. Every
# holdout is deliberately non-block-divisible (n=3000, seq=2049, batch=17,
# m=3000, t=1500) so the explicit-padding lesson survives the shrink.
TASK_CASES = {
    "splash_attention": ["probe-h8-s4096", "probe-h4-s2048", "probe-holdout-h4-s2049"],
    "flce": ["probe-4096x2880x151936", "probe-2048x2880x151936", "probe-holdout-3000x2880x151936"],
    "ragged_paged_attention": ["probe-b16-len1024", "probe-b8-len512", "probe-holdout-b17-len512"],
    "megablox_gmm": ["probe-m4096-e4-uniform", "probe-m2048-e4-zipf", "probe-holdout-m3000-e4-zipf"],
    "rg_lru": ["probe-4x2048x2560", "probe-2x1024x2560", "probe-holdout-2x1500x2560"],
}
TASKS = tuple(TASK_CASES)
VARIANTS = ("reference", "seam")
# variants whose answer is a FILL, assembled with a harness scaffold rather
# than submitted as-is
SEAM_VARIANTS = ("seam",)

# The --problem argument the judge worker wants: `name:c1,c2;name2:c3,c4`.
# No spaces anywhere -- the worker's parser strips only the outer whitespace of
# a spec, so `a : b` becomes the problem name `'a '` and a KeyError that kills
# the whole worker before any problem boots.
def problem_arg(tasks=None) -> str:
    return ";".join(f"{t}:{','.join(TASK_CASES[t])}" for t in (tasks or TASKS))


# --------------------------------------------------------------------- models
@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    renderer: str
    worker: int  # v5p-16 host index that serves it
    stop: str  # end-of-turn marker, for /v1/completions
    max_model_len: int  # the served context window
    # Longest generation that still fits behind the LONGEST prompt. Measured
    # the hard way: gemma is served at 16384, its tailored splash prompt was
    # ~4.4k tokens, and 4.4k + 12000 > 16384 -- so vLLM returned
    # `HTTPError 400: Bad Request` for all 16 of that cell's requests and the
    # cell recorded 16 empty generations that LOOK like model failures. The
    # driver now additionally asks the server's own /tokenize for the exact
    # prompt length and clamps per request, so this is a ceiling, not the
    # whole defence.
    max_new_tokens: int
    # Head-room left below max_model_len after the prompt, so a long prompt
    # shortens the completion instead of 400-ing the request.
    reserve_tokens: int = 256


MODELS = {
    # thinking DISABLED: at 12000 tokens the thinking renderer finished 3 of 80
    # generations on these tasks and every Qwen cell was consequently 0. The
    # repo already pins `qwen3_5_disable_thinking` for the same reason.
    "qwen35-27b": ModelSpec("qwen35-27b", "Qwen/Qwen3.5-27B", "qwen3-nothink", 0, "<|im_end|>", 22528, 16000),
    "gemma4-31b": ModelSpec("gemma4-31b", "google/gemma-4-31B-it", "gemma4", 1, "<turn|>", 16384, 12000),
}


@dataclass(frozen=True)
class ProbeConfig:
    model: str
    variant: str
    task: str

    @property
    def name(self) -> str:
        return f"{self.model}|{self.variant}|{self.task}"

    @property
    def is_seam(self) -> bool:
        return self.variant in SEAM_VARIANTS


def all_configs(models=None, variants=None, tasks=None) -> list[ProbeConfig]:
    models = list(models or MODELS)
    variants = list(variants or VARIANTS)
    tasks = list(tasks or TASKS)
    return [ProbeConfig(m, v, t) for m in models for v in variants for t in tasks]


# ---------------------------------------------------------------- generation
MAX_NEW_TOKENS = 12000
TEMPERATURE = 1.0
TOP_P = 1.0
GROUP_SIZE = 16  # ttt-discover computes advantage WITHIN a group of this size
