# Phase 5 report — grading-server FLEET demonstration (no RL)

Scope, as the user set it on 2026-08-06: prove the **grading server** works at
fleet scale. Five spot v6e-1 judges in us-east5-b, all polling **one** queue,
a fixed corpus in and verdicts out. No RL, no training, no policy, no v5p, no
prompt work.

Two prerequisites had to land before any chips were spent (PLAN.md phase 5 /
PHASE4-REPORT.md's one residual): a **compile-warm deadline**, because the
per-grade wall budget is checked *between* phases and therefore cannot bound
the single un-cancellable XLA compile that caused the phase-4 stall, and a
**compile-bomb** fixture to prove the new gate fires.

---

## Step 1 — the compile budget (the fleet-wide poison pill)

Phase 4 measured `timer-tamperer` spending **569.5 s inside one
`candidate_compile`** while using 3.1 s of chip time. Under lease-requeue that
is not one bad grade: the lease expires, the item requeues, and the same
candidate wedges the next judge, and the next.

### The budget: 90 s, and why

`DEFAULT_COMPILE_BUDGET_S = 90.0` (`judge/worker.py`). The phase-4 honest
distribution of `candidate_compile_s` on a v6e-1: median **2.98 s**, typical
band 2.2–3.7 s, worst honest candidate **46.2 s** (`nondeterministic`; 45.7 s
in attempt 1). So 90 s is:

* **~2× the worst honest compile ever measured** and ~30× the median — no
  honest kernel is at risk, and the 2× margin absorbs chip-to-chip and
  boot-to-boot compiler variance. A candidate needing >90 s of compile has
  already lost on score;
* **low enough to be worth having** — it caps a poison candidate's lane at
  ~18× the median `item_wall_s` (5.01 s) instead of 114×, so a fully
  adversarial corpus costs a 5-judge fleet bounded compile time instead of an
  unbounded stall;
* **strictly under the 240 s lease timeout**, which is the property that makes
  it work at all: the rejection is POSTed *before* the queue would requeue, so
  the poison pill is never handed to a second judge.

### Two layers, because one is not enough

1. **Between compile units** (`_check_compile_deadline`): each exported
   signature and the gradient functional is its own compile, and the deadline
   is checked before and after each. Clean — costs "one compile unit late",
   the same idiom as the wall budget.
2. **A watchdog thread** for an overrun *inside* one unit, which is the exact
   shape of the 569.5 s case. XLA releases the GIL while compiling, so the
   thread runs; there is no way to cancel the compile, so the only bound left
   is process death — but the watchdog **posts the terminal verdict first**
   (`gate=compile_budget`, `reward=0.0`, `terminal=True`, and it is written to
   the shared cache), and only then hard-exits. The queue marks the item done
   and **never requeues it**; every future judge gets the verdict for free
   from the cache. Bounded cost on one judge, once, versus an unbounded stall
   on all five.

`judge/queue.py` carries `terminal` on the item: a terminal item is excluded
from the lease-expiry requeue sweep and counted in `terminal_rejections`.

### Same budget in the CPU AOT pre-gate

`judge/aot_gate.py` already had `DEFAULT_COMPILE_BUDGET_S`; the two are now
reconciled — the pre-gate imports the *same* constant from the worker and
reports the failure at gate `compile_budget` rather than the generic
`timeout`, enforced by the parent's process-group kill on a killable CPU
child.

**A limit worth stating rather than hiding** (`test_measure_cpu_pregate_
cannot_see_a_backend_compile_bomb`): at production shapes with a 90 s budget,
XLA:CPU lowers *and* compiles the 220-deep bomb in **3.7 s** (honest control:
1.9 s) and passes it. These bombs' cost lives in the backend's
fusion/live-range/scheduling passes, which CPU and TPU do not share. So a
CPU-only pre-gate is **not** a substitute for the judge-side deadline; on a
real client (a v5p training host) the pre-gate child compiles on TPU under the
same budget and does catch it. The graph-*size* sub-class (depth 4000) is
caught on CPU at zero chip cost.

### `timer-tamperer` had become the bomb it was never meant to be

Sized down three times, always by measurement: phase 2 cut 200 → 24 sins when
`jax.grad`'s residual chain OOM'd a v6e-1; phase 4 cut 24 → 6 when the chain
held a lane 19+ minutes; **phase 5 cuts 6 → 2**, because that 6-deep chain is
the one that measured 569.5 s — 6× the new budget. Left at 6 it would have
been rejected at `compile_budget` instead of demonstrating what it exists to
demonstrate (that timer tampering buys nothing and a slow kernel scores far
below 1). `corpus.verdict_ok` additionally *tolerates* a `compile_budget`
rejection for it — that is also correct judge behaviour — while still failing
the run if it ever passes with a competitive score (`max_score` 0.95) or dies
at any other gate.

### Step-1 acceptance

**Full CPU battery green under sbatch (job 3648630, NOT the login node):
138/138 passed in 302.5 s** — the phase-4 126 plus 12 new compile-budget
tests, covering: rejection is terminal; the honest kernel survives the
production budget; the verdict is cached so the fleet pays once; the watchdog
fires inside a single un-cancellable unit; the pre-gate bounds its wall *by
the budget* (2.1 s for a 2 s budget); terminal results are never requeued.

## Step 2 — CPU fleet simulation

`phase5/simulate_fleet.py` under sbatch (job 3648631): one real queue, five
mock-timed worker processes, the real corpus builder, the real fleet driver.
**12/12 invariants green, 0 failures**, 399 items + 40 duplicates:

* `compile-bomb-rejected` 4/4 at `compile_budget`; `compile-bomb-never-
  requeued` — `attempts=[1]`, `flagged_terminal=4`;
* mid-lease SIGKILL of `sim-judge-2`: `killed_mid_lease=true`, `attempts=2`,
  completed on **`sim-judge-4`** (a survivor) 15.0 s later;
* `cache-hit-identical` 40/40 byte-identical through one shared cache dir;
* `no-duplicate-results`: 439 submitted / 439 completed / **0 duplicates**,
  1 requeue and 1 expired lease (both the deliberate kill), 4 terminal
  rejections.

The mock now honours `--cache` through the same hash→reward path as the real
worker; without that the duplicate check would have been testing the mock
rather than the cache.

---

## Step 3 — the real run

*(filled in below once the fleet run completes)*
