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

---

## Step 3 — the real run

Two attempts. Attempt 1 never graded a candidate; it is reported in full
because it found a bug on two of the always-delete paths.

### Attempt 1 — sbatch 3648750 — wedged before grading, zero results

Created its five QRs at 19:39:34–19:39:44 and then did nothing for four
minutes. Cause: a bare `wait` after the parallel creates. Bash's `wait` with
no arguments waits for **every** background child, and the queue server —
started with `&` and required to run for the whole demo — is one of them. The
script could not leave the create step until the queue exited, which it never
will.

The same bare `wait` sat in two more places, and both are worse than the one
that fired: inside `cleanup()` (after issuing the five deletes) and in the
degraded-fleet "delete the QRs that never landed" branch. Those are **teardown
paths**, so the bug's blast radius was "five QRs stay alive and nobody is
watching" — precisely what the always-delete rule exists to prevent. All three
now wait on explicitly collected PID arrays (commit `fbb68d67`).

Teardown itself behaved correctly once signalled: `scancel --batch
--signal=TERM` → `[trap] caught TERM/INT/HUP` → cleanup → five `delete issued`
→ `verified empty of arena QRs` **and** `verified empty of arena nodes` in
**both** us-east5-b and us-east5-a. (Worth recording: `scancel --signal=TERM`
*without* `--batch` does not reach the batch shell. Only `--batch` does.)

Cost: the QRs reached ACTIVE at ~19:46:30 while the deletes issued at 19:43:33
were still in flight against PROVISIONING resources, so the deletes had to be
reissued; all five confirmed gone at 19:54:31. **~8 min × 5 chips ≈ 0.67
chip-hours, zero candidates graded.** Never more than 5 QRs existed.

### Attempt 2 — sbatch 3648773 — **ALL GREEN, 15/15 invariants, 0 hard failures**

Artifacts: `runs/pallas_arena/phase5-3648773.log`,
`phase5-results-3648773.json`, `phase5-queue-3648773.log`,
`phase5-judge-<name>-3648773.log` ×5, `phase5-progress-<name>-3648773.log` ×5,
`phase5-boot-<name>-3648773.json` ×5, `phase5-keeper-3648773.{json,log}`.

**Bring-up**: QRs created 19:56:23 → all five ACTIVE, provisioned and polling
at **20:03:23 = 7.0 min**, in parallel (phase 4 measured 6.4 min for one, so
five cost essentially the same wall — capacity was never the constraint).
Worker boot **56.3 / 56.4 / 58.3 / 58.3 / 56.3 s**, of which
`calibration_warm_s` 9.65–10.27 s. Driver 20:03:23 → 20:35:57, teardown
complete and both zones verified empty by 20:38:57.

Every judge proved the **shared GCS reward cache read-write at provision
time** — `arena-cache: RW OK` from all five hosts — before any grading. That
check exists because a silently-unreachable cache degrades to a permanent miss
that looks exactly like everything working.

**Corpus**: 1599 salted candidates across 20 variants + 100 deliberate
duplicates = **1699 queue items**. Per-variant: 119 each of the seven honest
variants, 71 split-personality, 95 each of wrong-eps / constant-output /
seed-reader / wrong-grad, 71 aliased-reference, 60 each of cached-output /
memoizer / obfuscated-import, 48 no-kernel, and the capped pathological
fixtures (8 nondeterministic, 4 timer-tamperer, 4 compile-bomb).

#### Acceptance scorecard

| # | Item | Verdict | Measured |
|---|---|---|---|
| 1 | 5 judges on one queue | **GREEN** | 5/5 landed, all 5 polling one queue; `judges-on-tpu` = `["tpu"]` |
| 2 | every cheater at its correct gate | **GREEN** | **1599/1599 verdicts correct**, zero exceptions — gate histogram below |
| 2b | compile-bomb rejected at its gate | **GREEN** | 4/4 at `compile_budget`; `candidate_compile_s` **90.02–90.05 s** against a 90.0 s budget |
| 2c | compile-bomb never requeued | **GREEN** | `attempts=[1]` for all 4, `flagged_terminal=4`, `judge_restarted=[true×4]` |
| 3 | honest regrade within ±3% | **GREEN** | worst **max** dev **1.60%** (`pallas-br256`, n=119); worst p95 1.18%. Every variant's *max* is inside the band, so the gate's p95 relaxation was never needed |
| 4 | ref-vs-ref 1.00±2% on EVERY judge | **GREEN** | 15 measurements (5 judges × 3 shape cases), **median dev 0.306%, worst 1.462%** (judge-3 @ 8192×4096); `violations: {}` |
| 5 | cross-judge score agreement | **GREEN** | worst **0.794%** (`pallas-br16`), best 0.038% (`pallas-br512`), band 3%. Phase 4's cross-*boot* number was 0.55% |
| 6 | cache hits byte-identical | **GREEN** | **100/100 identical**, 100/100 flagged `cache_hit`, median **65.9 ms** vs 5.59 s for a real grade (**85× faster**) |
| 7 | chaos: kill a judge mid-lease | **GREEN** | see below — requeued and completed on a survivor, `attempts=2` |
| 8 | no lost or duplicated results | **GREEN** | **1699 submitted / 1699 completed / 0 duplicates**; 2 requeues, 2 expired leases, 9 terminal rejections; queue drained to depth 0, leased 0 |
| 9 | peak HBM per grade | **GREEN** | 15.63–19.48 GB |
| 10 | throughput | **52.4 candidates/min** | full corpus incl. the poison items; **64.9/min** steady-state (see below) |
| 11 | zero resources left behind | **GREEN** | verified empty of arena QRs **and** nodes in us-east5-b **and** us-east5-a |

#### Gate histogram — every copy of every variant, one gate each

| variant | n | gate | | variant | n | gate |
|---|---|---|---|---|---|---|
| honest-xla | 119 | `all` (pass) | | wrong-eps | 95 | `correctness` |
| honest-xla-b | 119 | `all` (pass) | | constant-output | 95 | `correctness` |
| honest-xla-f32out | 119 | `all` (pass) | | seed-reader | 95 | `correctness` |
| unjitted-robust | 119 | `all` (pass) | | wrong-grad | 95 | `gradient` |
| pallas-br16 | 119 | `all` (pass) | | aliased-reference | 71 | `ast` |
| pallas-br256 | 119 | `all` (pass) | | cached-output | 60 | `aot_export` |
| pallas-br512 | 119 | `all` (pass) | | memoizer | 60 | `aot_export` |
| split-personality | 71 | `all` (pass) | | obfuscated-import | 60 | `poison_stub` |
| nondeterministic | 8 | `all` (pass) | | no-kernel | 48 | `exec` |
| **compile-bomb** | **4** | **`compile_budget`** | | **timer-tamperer** | **4** | **`compile_budget`** |

Not one variant split across two gates: every column is a single bucket.

#### The compile budget did the job it was added for

All four compile bombs died at **90.02–90.05 s** against the 90.0 s budget —
i.e. the watchdog fired inside a single un-cancellable XLA compile, within
50 ms of the deadline, four times independently. Each was recorded
`terminal`, each has `attempts=1`, and each cost its lane **91.9–92.2 s** of
`item_wall_s` and one judge restart (~60 s of boot). **Bounded, once, on one
judge** — versus the unbounded fleet-wide stall the phase-4 residual
predicted.

**Finding: `timer-tamperer` also lands at `compile_budget`, even at 2 sins.**
All four copies. Phase 4 measured the 6-deep version at 569.5 s of compile;
the 2-deep version still exceeds 90 s, because the cost is in the *gradient
probe* through the chain at 32768×8192, not in the chain's depth per se. This
is the case `corpus.verdict_ok` was written to tolerate, and the tolerance is
narrow: a `compile_budget` rejection is accepted for this fixture, while
passing with a score ≥0.95 or dying at any other gate would still fail the
run. It is correct judge behaviour — but it means this fixture no longer
demonstrates "a slow kernel scores well below 1" on TPU (phase 4's 0.0619
does), and a future revision should shrink its *shape*, not its depth.

**Bonus, unplanned**: `terminal_rejections` is **9**, not 8. The ninth is the
one `timer-tamperer` copy that the stride-sampled duplicate set happened to
pick: resubmitted, it was served its **cached terminal verdict**, byte-
identical, in milliseconds. That is the "cached for every future judge"
half of the terminal design, demonstrated by accident on live silicon.

#### Chaos — SIGKILL a judge mid-lease, at fleet scale

Attempts 0 and 1 raced: the kill landed but the graded item had already
completed (`midlease=False`), which the retry loop exists for — without that
proof a race would score as a pass having tested nothing. Attempt 2 landed:
judge-4 holding `w000565-c5a078b9` (`pallas-br256#s00032`), `midlease=True`
verified by re-reading the item immediately after the kill.

The lease expired, the queue requeued, and the item **completed on
sk7524-pallas-judge-2 — a different judge** — with `attempts=2`,
`passed=true`, gate `all`, **248.2 s** from kill to completion (240 s of that
is the lease timeout, so the requeue-and-regrade itself cost ~8 s). The other
four judges kept grading throughout. Final accounting: **0 lost, 0
duplicated**.

This is the property phase 4 could not test. Phase 4 showed one judge coming
back; here the work migrated to a *different chip* and produced a verdict that
agrees with the fleet population to within 1.6%.

#### Throughput

| window | rate |
|---|---|
| whole corpus (1599 items, 1831 s) | **52.4 candidates/min** |
| steady state (602–1729 s) | **64.9/min** |
| post-chaos (1069–1729 s) | **60.8/min** |
| first 362 s (poison items + restarts) | 14.9/min |

Phase 4 predicted ~50–60/min at N=5 from `item_wall_s` 5.01 s. Measured
**52.4/min over everything**, **64.9/min in steady state** — the prediction
was right, and the aggregate is dragged down almost entirely by the eight
pathological fixtures: they are 0.5% of the corpus but cost ~12 judge-minutes
(8 × 92 s of lane plus 8 × ~60 s of restart boot), which is the whole gap
between 52.4 and 64.9.

Per-judge load was even — 306 / 337 / 309 / 336 / 311 items — and per-judge
median `item_wall_s` 5.26–5.77 s, i.e. no judge was systematically slower.

Per-candidate cost, medians across 1599 real grades (phase-4 attempt-2 medians
in brackets):

| stage | median | p95 | max |
|---|---|---|---|
| `export_s` (sandbox child, no device) | **0.858 s** [0.82] | 1.014 s | 14.69 s |
| `load_s` | **0.0035 s** [0.0035] | 0.0041 s | 0.63 s |
| `candidate_compile_s` | **2.833 s** [2.98] | 3.270 s | **90.05 s** (the budget) |
| `warm_chip_s` | **1.976 s** [1.944] | 2.101 s | 6.37 s |
| `item_wall_s` | **5.589 s** [5.01] | 6.355 s | 92.18 s |

Every number reproduces phase 4 on a five-chip fleet — and `candidate_
compile_s` now has a **hard ceiling** where it previously had none.

#### Resources

| | attempt 1 | attempt 2 | total |
|---|---|---|---|
| QRs created | 5 | 5 | never >5 alive |
| QR lifetime (create → confirmed deleted) | 15.0 min | 42.6 min | |
| chips × ACTIVE time | 5 × ~8 min ≈ **0.67** | 5 × ~37 min ≈ **3.10** | **≈ 3.8 chip-hours** |
| candidates graded | 0 | 1699 | 1699 |

Plus one neuronic compute node (16 cpu, 32 G) for ~2.4 h across both attempts
and the two sbatch CPU jobs.

**Hard rules, all honored**: never more than 5 arena QRs alive; distinctly
named `sk7524-pallas-judge-1..5`; deleted on every exit path taken (attempt 1
via the TERM trap, attempt 2 via the normal `rc=0` EXIT trap); the 3 h hard
cap (19:56:23 → 22:56:23) was never approached — the run finished 2 h 18 min
inside it; all 5 landed inside the 45 min deadline so the degraded-fleet path
was not exercised on silicon (it is exercised by the simulation); the keeper
never made a replacement (`phase5-keeper-3648773.json`).

**Zero arena QRs and zero arena TPU nodes remain**, verified independently of
the script's own cleanup after the job exited, in **both us-east5-b and
us-east5-a**:

```
=== us-east5-b QRs ===   (none)
=== us-east5-b TPU nodes === (none)
=== us-east5-a QRs ===   (none)
=== us-east5-a TPU nodes === (none)
```

---

## What phase 5 establishes, and what it does not

Established: the grading server is correct and reproducible **across chips**,
not just across restarts on one chip (cross-judge 0.794%); a stateless
preemptible fleet is safe for RL because a kernel's reward does not depend on
which judge draws it (100/100 byte-identical cache hits, 0.79% cross-judge);
exactly-once delivery survives a mid-lease kill (0 lost, 0 duplicated over
1699 items); and the fleet's throughput is **~52–65 candidates/min at N=5**,
which is the planning input for the RL run.

Not established, and deliberately out of scope: nothing about RL, sampling,
policies or v5p; nothing about problems other than RMSNorm; and the
degraded-fleet (<5 landed) and keeper-replacement paths were not exercised on
silicon because the fleet never degraded.

Residuals for the next phase:
* `timer-tamperer` needs a smaller *shape*, not a smaller depth, if it is to
  keep demonstrating a low score rather than a compile-budget rejection;
* the chaos retry killed two judges before landing a genuine mid-lease kill.
  Each wasted kill costs a ~60 s boot. Leasing a deliberately long-running
  item and killing *that* would make the test cheaper and deterministic;
* `export_s` has a 14.69 s tail against a 0.858 s median. It is off the chip
  clock and did not matter here, but at RL scale it is 17× the median lane
  cost in the client's sandbox and is worth a look.
