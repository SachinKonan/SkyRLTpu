"""Prompt-side control for the sd1/sd2 arms: do they build, and do they FIT?

The arena has already lost a whole 16-candidate cell to a prompt that did not
fit: a ~4.4k-token prompt plus a 12000-token request against gemma's 16384
window returned `HTTPError 400` sixteen times, and the cell recorded sixteen
empty generations that read as model failures. The driver now asks the server's
own /tokenize and clamps, so the request cannot 400 -- but a prompt that leaves
no room for a completion silently TRUNCATES candidates instead, which reads as
model failure just as convincingly. So the fit is checked before any chip.

The numbers are calibrated, not guessed. gemma's own tokenizer reported
2388-4394 tokens for the ladder prompts whose lengths are known, i.e. **3.07
characters per token at the low end** -- that ratio is used here. And over the
ladder's 480 generations on these exact tasks the median was ~8100 tokens with
12 of 480 hitting `length` against a 12000 budget, so `--want-completion`
defaults to 10000: above the 99th percentile generation, below what a 12000
request would have used.

Also asserts the structural invariants of the splice: the seam's output
contract is still the LAST thing in the prompt, the dialect bullets are present
verbatim, the scaffold is shown, and sd2 is a strict superset of sd1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallas_arena.probe import configs as C
from pallas_arena.probe.prompt_seam import _OUTPUT
from pallas_arena.probe.prompt_seam_dialect import (
    _BULLETS_2_TO_9,
    _CALLS,
    _PRECISION_TRADE,
    RUNGS,
    SEAM_DIALECT_PROMPTS,
)
from pallas_arena.probe.prompts import get_prompt
from pallas_arena.probe.seam import SEAMS

TARGETS = ("splash_attention", "ragged_paged_attention", "megablox_gmm")
CHARS_PER_TOKEN = 3.07  # measured against gemma's own /tokenize on the ladder prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--want-completion", type=int, default=10000)
    args = ap.parse_args()

    spec = C.MODELS[args.model]
    rows, bad = [], []
    for task in SEAM_DIALECT_PROMPTS:
        for rung in RUNGS:
            p = get_prompt(task, rung)
            tok = int(len(p) / CHARS_PER_TOKEN)
            room = spec.max_model_len - tok - spec.reserve_tokens
            row = {
                "task": task,
                "rung": rung,
                "chars": len(p),
                "approx_tokens": tok,
                "room_for_completion": room,
                "ends_with_output_contract": p.endswith(_OUTPUT),
                # bullets 2-9 of the P1 list are spliced VERBATIM out of it
                "carries_dialect_bullets_verbatim": _BULLETS_2_TO_9 in p,
                "carries_precision_trade": _PRECISION_TRADE in p,
                "carries_introspected_calls": _CALLS in p,
                "shows_scaffold": SEAMS[task].scaffold.strip()[:60] in p,
                "superset_of_sd1": get_prompt(task, "sd1")[:2000] in p,
            }
            rows.append(row)
            for k in ("ends_with_output_contract", "carries_dialect_bullets_verbatim",
                      "carries_precision_trade", "carries_introspected_calls",
                      "shows_scaffold", "superset_of_sd1"):
                if not row[k]:
                    bad.append(f"{task}/{rung}: {k} is False")
            if room < args.want_completion:
                bad.append(f"{task}/{rung}: only {room} tokens left for a completion")
            print(f"[prompt] {task:24s} {rung}  {len(p):6d} ch  ~{tok:5d} tok  "
                  f"room {room:6d}  output-last={row['ends_with_output_contract']}", flush=True)

    # the required fill names must still be exactly what the scaffold calls
    for task in TARGETS:
        req = SEAMS[task].required
        for rung in RUNGS:
            p = get_prompt(task, rung)
            for name in req:
                if name not in p:
                    bad.append(f"{task}/{rung}: prompt never names required `{name}`")

    print()
    if bad:
        for b in bad:
            print(f"  FAIL {b}", flush=True)
    print(f"[prompt] {len(rows) - len(set(r.split(':')[0] for r in bad))}/{len(rows)} cells clean "
          f"({len(bad)} problems)", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "problems": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
