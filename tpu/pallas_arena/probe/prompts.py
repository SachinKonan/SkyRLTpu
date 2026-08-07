"""Probe prompts, as DATA.

Variants, per task:

  minimal   -- task spec, signature, shapes, judge contract. Nothing else.
                (splash_attention and flce only; the cold-start probe's arm.)
  reference -- minimal + the public fp32 oracle inline (labelled as the
               correctness oracle and as an INVALID answer) + a Pallas/Mosaic
               gotcha list drawn from what THIS repo has actually hit. This is
               the CONTROL arm of the seam run: it was the previous probe's
               only trainable cell.
  tailored  -- reference + a scaffold in which every structural decision is
               GIVEN and exactly one function is blank. Kept for the record;
               it is the variant that passed 16/16 with a reward spread BELOW
               the judge's noise floor, i.e. with no usable gradient.
  seam      -- the harness owns the interface, the model owns the strategy:
               the prompt shows the exact scaffold the answer is wrapped in
               and asks only for named fill-in functions. See seam.py.

Kept out of the driver on purpose: the prompts ARE the experiment, and an
experiment's independent variable should be diffable in one file.
"""

from __future__ import annotations

from pallas_arena.probe.prompt_flce import PROMPTS as _FLCE
from pallas_arena.probe.prompt_ref_extra import REFERENCE_PROMPTS as _REF_EXTRA
from pallas_arena.probe.prompt_seam import SEAM_PROMPTS as _SEAM
from pallas_arena.probe.prompt_splash import PROMPTS as _SPLASH

PROMPTS: dict[str, dict[str, str]] = {
    "splash_attention": dict(_SPLASH),
    "flce": dict(_FLCE),
}
for _task, _p in _REF_EXTRA.items():
    PROMPTS.setdefault(_task, {})["reference"] = _p
for _task, _p in _SEAM.items():
    PROMPTS.setdefault(_task, {})["seam"] = _p


def get_prompt(task: str, variant: str) -> str:
    try:
        return PROMPTS[task][variant]
    except KeyError as e:
        raise KeyError(f"no prompt for task={task!r} variant={variant!r}") from e


def prompt_table() -> list[dict]:
    return [
        {"task": t, "variant": v, "chars": len(p), "approx_tokens": len(p) // 4}
        for t, vs in PROMPTS.items()
        for v, p in vs.items()
    ]
