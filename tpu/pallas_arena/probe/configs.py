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
# GENERAL-mode case sets: a SWEEP, not two shapes plus an ignored holdout.
# Every case here is scored (see Problem.general_mode -> timing.final_reward),
# so a kernel cannot hardcode the shapes it was shown and still collect reward.
# flce stays OURS_SPECIFIC (LoRA dx-only contract) and keeps its 2+1 set.
TASK_CASES = {
    "splash_attention": [
        "probe-h8-s4096", "probe-h4-s2048", "probe-h16-s1024",
        "probe-h8-s4096-d64", "probe-holdout-h4-s2049",
    ],
    "flce": ["probe-4096x2880x151936", "probe-2048x2880x151936", "probe-holdout-3000x2880x151936"],
    "ragged_paged_attention": [
        "probe-b16-len1024", "probe-b8-len512", "probe-b64-len1024",
        "probe-b32-len2048", "probe-b128-len512", "probe-holdout-b17-len512",
    ],
    "megablox_gmm": [
        "probe-m4096-e4-uniform", "probe-m2048-e4-zipf", "probe-m8192-e8-uniform",
        "probe-m8192-e8-8x7b", "probe-m4096-e16-zipf", "probe-holdout-m3000-e4-zipf",
    ],
    "rg_lru": [
        "probe-4x2048x2560", "probe-2x1024x2560", "probe-8x512x2560",
        "probe-2x4096x2560", "probe-4x2048x1024", "probe-holdout-2x1500x2560",
    ],
}
TASKS = tuple(TASK_CASES)
# The PROMPT LADDER rungs (prompt_ladder.py). Each is a strict superset of the
# one below it and each answer is a WHOLE program, not a fill.
LADDER_VARIANTS = ("p1", "p3", "p4")
# The SEAM + DIALECT arms (prompt_seam_dialect.py). The ladder measured that
# splash / RPA / GMM fail at the `pallas_call` PLUMBING, which no whole-program
# rung supplies, and that the seam is the only variant that has ever exported
# splash. These two hand over the plumbing AND carry the P1 dialect list.
SEAM_DIALECT_VARIANTS = ("sd1", "sd2")
VARIANTS = ("reference", "seam") + LADDER_VARIANTS + SEAM_DIALECT_VARIANTS
# variants whose answer is a FILL, assembled with a harness scaffold rather
# than submitted as-is
SEAM_VARIANTS = ("seam",) + SEAM_DIALECT_VARIANTS
# variants whose answer is a whole program that the harness PREPENDS a tested
# primitives prelude to (ladder.PRELUDES)
PRELUDE_VARIANTS = ("p4",)

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

    @property
    def is_prelude(self) -> bool:
        return self.variant in PRELUDE_VARIANTS


def all_configs(models=None, variants=None, tasks=None) -> list[ProbeConfig]:
    models = list(models or MODELS)
    variants = list(variants or VARIANTS)
    tasks = list(tasks or TASKS)
    return [ProbeConfig(m, v, t) for m in models for v in variants for t in tasks]


def parse_cells(spec: str) -> list[ProbeConfig]:
    """`model|variant|task,model|variant|task,...` -> configurations.

    The cross product is the wrong shape for a run that needs ONE control cell
    on a different variant from the arm under test (e.g. three kernels x two
    seam arms, plus `rg_lru|p1` to prove the harness is sound). Expressing that
    as `--variants` x `--tasks` would silently add five cells nobody asked for
    and spend their chip time.
    """
    out: list[ProbeConfig] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("|")
        if len(parts) != 3:
            raise ValueError(f"cell must be model|variant|task, got {raw!r}")
        model, variant, task = (p.strip() for p in parts)
        if model not in MODELS:
            raise KeyError(f"unknown model {model!r}")
        if task not in TASK_CASES:
            raise KeyError(f"unknown task {task!r}")
        if variant not in VARIANTS:
            raise KeyError(f"unknown variant {variant!r}")
        out.append(ProbeConfig(model, variant, task))
    if not out:
        raise ValueError("--cells named no configurations")
    return out


# ---------------------------------------------------------------- generation
MAX_NEW_TOKENS = 12000
TEMPERATURE = 1.0
TOP_P = 1.0
GROUP_SIZE = 16  # ttt-discover computes advantage WITHIN a group of this size
