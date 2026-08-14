"""Multi-turn repair driver: generate -> pre-gate/judge -> SHOW THE ERROR -> fix.

NO RL. This measures whether one or two feedback turns convert the measured
failure classes, per the taxonomy mined from sd-results-3687904:

  * aot_export (the largest class, 15-28/32 in weak cells): the model sees the
    verbatim export traceback, from the CPU pre-gate, in seconds;
  * precision near-misses (29): "max err 1.1x tol" plus the accumulator rule;
  * the masked-row bug (~6, err ~= 0.05 on attention): one `where` line;
  * NaN (2): mask-value arithmetic.

Two modes, one harness:
  * cold:    turn 1 is the unchanged sd cold-start prompt (so turn-1 numbers
             stay comparable to the 224-sample baseline); repair turns then
             edit the COMPOSED program.
  * improve: turn 1 is already a repair turn, seeded with a known-good winner
             and its measured reward against the REAL denominator. The model
             edits the full program -- scaffold included -- so the seam's
             decomposition stops being a ceiling exactly where that matters.

A candidate is a CHAIN of up to --turns attempts; its reward is the best
graded result in the chain. Every turn is logged as its own row (stage:
turn1 / repair1 / repair2), so conversion-per-turn is measurable per failure
class afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tpu/

from pallas_arena.judge.queue import ArenaQueueClient  # noqa: E402
from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe import render as R  # noqa: E402
from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402
from pallas_arena.probe.prompts import get_prompt  # noqa: E402
from pallas_arena.probe.sampler import VllmSampler  # noqa: E402
from pallas_arena.probe.seam import SEAMS, compose, extract_fill  # noqa: E402


def sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


# The semantic layer of the dialect list: bug SIGNATURES, mined from the same
# run the API dialect came from. Shown only in repair turns.
SIGNATURES = """\
## Error signatures measured on previous attempts at these exact tasks

* Attention, max err ~0.05: your fully-masked rows are producing UNIFORM
  attention. With a finite mask value (-1e30) and a max-shifted softmax, a row
  whose every key is masked gets exp(0)=1 everywhere -- the mean of V, not 0.
  The contract requires exactly 0: keep a `row_live` mask and end with
  `out = jnp.where(row_live[..., None], out, 0.0)`.
* max err within ~1.5x of tol: precision, not logic. Keep every accumulator
  float32 -- `preferred_element_type=jnp.float32` on every matmul, float32
  running max/sum/output; bfloat16 only as matmul INPUTS.
* non-finite output (NaN/inf): mask-value arithmetic in the online-softmax
  rescale (inf - inf, or exp of a huge positive). Initialize the running max
  to a large FINITE negative and mask before exponentiating.
* O(1) error (max err > 1.0): a term is missing outright -- a dropped
  recurrence carry, a wrong head-group mapping. Re-derive the update rule
  against the task statement before touching anything else.
"""

REPAIR_TEMPLATE = """{base_prompt}

--------------------------------------------------------------------------
## Your current program

The complete program below was submitted to the judge. You may change ANY part
of it -- including the harness scaffolding, grid, and BlockSpecs -- as long as
it still defines `kernel(*inputs)` with the contract above.

```python
{program}
```

## Judge verdict

{feedback}

{signatures}
## What to do

Fix the program. Think briefly about WHICH failure signature above matches the
verdict, change what needs to change, and keep what already works. Output the
COMPLETE corrected program in ONE ```python fenced block. No prose after it.
"""

IMPROVE_FEEDBACK = """\
PASSED all correctness gates. Reward {reward:.3f} -- your kernel runs at
{reward:.0%} of the speed of the production kernel on the same inputs
(reward = production_time / your_time; 1.0 means parity, above 1.0 beats it).
The remaining gap is structural, not a constant to tweak: consider the grid
and block sizes, VMEM working-set, online vs two-pass softmax, and whether
work can be skipped entirely (masked blocks, empty pages)."""


def extract_program(text: str) -> str:
    """Repair turns return one fenced block containing the whole program."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return ""
    return max(blocks, key=len).strip()


class RepairProbe:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.t0 = time.time()
        servers = dict(kv.split("=", 1) for kv in args.servers.split(","))
        self.samplers = {m: VllmSampler(url, C.MODELS[m].hf_id) for m, url in servers.items()}
        self.queue = ArenaQueueClient(args.queue) if args.queue else None
        self.pool = ProcessPoolExecutor(max_workers=args.pregate_workers)
        self.sigs = {}

    def log(self, rec: dict):
        with self.lock:
            with open(self.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def time_left(self) -> float:
        return self.args.wall_s - (time.time() - self.t0)

    # ---------------------------------------------------------------- grading
    def grade(self, task: str, program: str, tag: str) -> dict:
        """CPU pre-gate, then the judge. Returns a verdict dict either way."""
        if task not in self.sigs:
            self.sigs[task] = probe_signatures(task, C.TASK_CASES[task])[0]
        try:
            v = self.pool.submit(pregate_one, task, program, self.sigs[task],
                                 timeout_s=self.args.pregate_timeout_s).result(
                timeout=self.args.pregate_timeout_s + 120)
        except Exception as e:  # noqa: BLE001
            v = {"passed": False, "gate": "harness", "reward": 0.0,
                 "observation": f"pre-gate harness error: {e!r}"}
        if not v.get("passed"):
            v["stage_gate"] = "pregate"
            return v
        if self.queue is None:
            return {"passed": True, "gate": "pregate-only", "reward": None,
                    "observation": "pre-gate passed (no judge attached)", "stage_gate": "pregate"}
        wid = self.queue.submit(task, program, tag=tag)
        deadline = time.time() + self.args.judge_timeout_s
        while time.time() < deadline:
            got = self.queue.poll_bulk([wid])
            if wid in got:
                res = got[wid].get("result") or {}
                res["stage_gate"] = "judge"
                res["attempts"] = got[wid].get("attempts")
                return res
            time.sleep(4)
        return {"passed": False, "gate": "unjudged", "reward": None,
                "observation": f"judge did not return within {self.args.judge_timeout_s:.0f}s",
                "stage_gate": "judge-timeout"}

    # ------------------------------------------------------------- turn logic
    def run_chain(self, model: str, task: str, variant: str, idx: int,
                  seed_program: str | None, seed_feedback: str | None) -> None:
        spec = C.MODELS[model]
        base_prompt = get_prompt(task, variant)
        chain = f"{task}|{variant}|{'improve' if seed_program else 'cold'}#c{idx}"
        program, feedback = seed_program, seed_feedback
        best_reward = None

        for turn in range(self.args.turns):
            if self.time_left() < 300:
                return
            stage = f"turn{turn + 1}"
            if turn == 0 and program is None:
                # cold start: byte-identical to the sd probe prompt
                prompt_text = R.render(spec.renderer, base_prompt)
            else:
                prompt_text = R.render(spec.renderer, REPAIR_TEMPLATE.format(
                    base_prompt=base_prompt, program=program,
                    feedback=feedback or "(no verdict recorded)", signatures=SIGNATURES))

            n_tok = self.samplers[model].count_tokens(prompt_text)
            budget = max(2048, min(C.MAX_NEW_TOKENS, (self.args.ctx - 64) - (n_tok or len(prompt_text) // 3)))
            gens = self.samplers[model].sample_group(
                prompt_text, 1, max_tokens=budget,
                temperature=self.args.temperature, stop=R.STOPS[spec.renderer])
            g = gens[0]

            if turn == 0 and program is None and task in SEAMS:
                fill, how, missing = extract_fill(g["text"], SEAMS[task].required)
                cand = compose(task, fill) if fill.strip() else ""
            else:
                cand = extract_program(g["text"])
                how, missing = "fenced-program", []

            rec = {
                "chain": chain, "model": model, "task": task, "variant": variant,
                "mode": "improve" if seed_program is not None and turn == 0 or seed_feedback else
                        ("improve" if self.args.mode == "improve" else "cold"),
                "cand": idx, "turn": turn + 1, "stage": stage,
                "code_sha": sha(cand), "code_chars": len(cand),
                "gen_chars": len(g["text"]), "finish_reason": g["finish_reason"],
                "extraction": how, "prompt_tokens": n_tok, "max_new_tokens": budget,
                "text": g["text"], "code": cand,
                "fed_back": feedback,
            }
            if not cand.strip():
                rec.update(gate="no_code", reward=0.0, passed=False,
                           observation="no extractable program in the reply")
                self.log(rec)
                feedback = "Your reply contained no extractable ```python block. Output the complete program in one fenced block."
                # keep the previous program as the base for the next turn
                continue

            v = self.grade(task, cand, tag=f"{chain}#t{turn + 1}")
            rec.update(gate=v.get("gate"), reward=v.get("reward"),
                       passed=bool(v.get("passed")), observation=v.get("observation"),
                       stage_gate=v.get("stage_gate"))
            self.log(rec)
            print(f"[repair] {chain} t{turn + 1}: gate={v.get('gate')} reward={v.get('reward')} "
                  f"({'PASS' if v.get('passed') else 'fail'})", flush=True)

            r = v.get("reward")
            if r is not None and (best_reward is None or r > best_reward):
                best_reward = r
            program = cand  # the next turn edits what was just graded
            if v.get("passed"):
                if self.args.mode == "cold" and not self.args.continue_after_pass:
                    return
                feedback = IMPROVE_FEEDBACK.format(reward=r or 0.0)
            else:
                feedback = v.get("observation") or f"gate={v.get('gate')}"

    # ------------------------------------------------------------------- main
    def main(self) -> int:
        cells = []
        seeds = {}
        if self.args.mode == "improve":
            seeds = json.load(open(self.args.seeds))
        for cell in self.args.cells.split(";"):
            task, variant = cell.split(":")
            cells.append((task.strip(), variant.strip()))

        model = self.args.model
        if not self.samplers[model].wait_ready(self.args.ready_deadline_s):
            print(f"[repair] FATAL: {model} engine never became ready", flush=True)
            return 1

        jobs = []
        for task, variant in cells:
            for i in range(self.args.candidates):
                if self.args.mode == "improve":
                    s = seeds[task]
                    fb = IMPROVE_FEEDBACK.format(reward=s["reward"])
                    jobs.append((model, task, variant, i, s["code"], fb))
                else:
                    jobs.append((model, task, variant, i, None, None))

        # Chains are independent; run a few concurrently so judge latency and
        # generation overlap. vLLM batches concurrent requests server-side.
        with ThreadPoolExecutor(max_workers=self.args.parallel_chains) as ex:
            futs = [ex.submit(self.run_chain, *j) for j in jobs]
            for f in futs:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[repair] chain error: {type(e).__name__}: {e}", flush=True)
        print("[repair] all chains done", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True, help="model=url[,model=url]")
    ap.add_argument("--model", required=True)
    ap.add_argument("--queue", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["cold", "improve"], required=True)
    ap.add_argument("--cells", required=True, help="task:variant;task:variant")
    ap.add_argument("--seeds", default=None, help="json {task: {code, reward}} (improve mode)")
    ap.add_argument("--candidates", type=int, default=8, help="chains per cell")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--continue-after-pass", action="store_true",
                    help="cold mode: keep improving after the first pass")
    ap.add_argument("--parallel-chains", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--pregate-workers", type=int, default=8)
    ap.add_argument("--pregate-timeout-s", type=float, default=180.0)
    ap.add_argument("--judge-timeout-s", type=float, default=900.0)
    ap.add_argument("--ready-deadline-s", type=float, default=3600.0)
    ap.add_argument("--wall-s", type=float, default=9000.0)
    args = ap.parse_args()
    if args.mode == "improve" and not args.seeds:
        ap.error("--mode improve requires --seeds")
    return RepairProbe(args).main()


if __name__ == "__main__":
    sys.exit(main())
