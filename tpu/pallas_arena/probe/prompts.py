"""Probe prompts, as DATA.

Three variants per task, strictly nested so any arm-to-arm difference is
attributable to the added block alone:

  minimal   -- task spec, signature, shapes, judge contract. Nothing else.
  reference -- minimal + the public fp32 reference implementation inline
               + a Pallas/Mosaic gotcha list drawn from what THIS repo has
               actually hit (32MB scoped VMEM, block/tile divisibility,
               explicit padding, custom_vjp, one kernel at every declared
               shape, the 90 s compile budget, the jax 0.10.2 pin).
  tailored  -- reference + a working scaffold in which every structural
               decision (pallas_call, grid, BlockSpecs, padding, VMEM
               budgeting, custom_vjp plumbing) is GIVEN and exactly one
               function -- the inner kernel body -- is left to fill in.

Kept out of the driver on purpose: the prompts ARE the experiment, and an
experiment's independent variable should be diffable in one file.
"""

from __future__ import annotations

from pallas_arena.probe.prompt_flce import PROMPTS as _FLCE
from pallas_arena.probe.prompt_splash import PROMPTS as _SPLASH

PROMPTS: dict[str, dict[str, str]] = {
    "splash_attention": _SPLASH,
    "flce": _FLCE,
}


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
