# Phase 4 report — real revalidation of the phase-3 fixes on spot TPU

Goal (PLAN.md phase 4): one persistent judge polling the queue **over the
wire**, running the RMSNorm acceptance battery, and turning **every phase-2
red green on silicon** — warm per-candidate cost, ref-vs-ref within 1.00±2%,
the recalibrated goldens, N>=5 bitwise determinism, peak HBM per grade, and
the whole cheater battery still rejected at the right gates. Always-delete on
every exit path.

Two attempts, **both on spot v6e-1 in us-east5-b**. Attempt 1 is the
diagnosis; attempt 2 revalidates on the same chip after the four fixes it
exposed. Staying on one accelerator generation is deliberate: the headline
result is a *change* in warm chip time, and a cross-generation move would
confound exactly the number being claimed.

---

## Attempt 1 — sbatch 3646330, spot v6e-1, us-east5-b

Artifacts: `runs/pallas_arena/phase4-3646330.log`,
`phase4-results-3646330.json`, `phase4-queue-3646330.log`,
`phase4-worker-3646330.log`.

Bring-up: QR created 13:59:52 → worker polling the queue over the reverse
ssh tunnel at 14:14:41 (**14.8 min**, including apt/uv provision and the
`jax[tpu]==0.10.2` install). Worker boot — jax/TPU init, reference compile
across all four shape cases, counterbalanced noise floors — **46.6 s**.
Driver 14:15:01 → 15:15:37. QR deleted by the trap; **zero pallas QRs and
zero pallas TPU nodes remained** in either us-east5-a or -b afterwards.

### Acceptance scorecard

| # | Item | Verdict | Measured |
|---|---|---|---|
| 1 | warm per-candidate cost ≈ 4 s | **RED** | median `warm_chip_s` **261.3 s** (range 254.8–269.6 over 8 passing grades); hard cap 12 s |
| 2 | ref-vs-ref 1.00±2%, counterbalanced | **GREEN** | **1.0114 / 1.0014 / 0.9968** (max dev **1.14%**) vs phase-2 1.053 / 1.021 / 1.019 |
| 3 | recalibrated goldens all pass | **PARTIAL** | br16 0.480, **br256 0.827**, unjitted-robust 0.826, f32out 0.794 all PASS; **br512 RED** (VMEM OOM) |
| 4a | determinism N>=5 bitwise | **GREEN** | enforced inside all 8 passing grades, zero violations |
| 4b | peak HBM per grade | **GREEN** | **15.6–19.5 GB**, max 19.48 GB |
| 5 | cheater battery at correct gates | **9/10** | one gate-label mismatch (`cached-output`), still rejected |
| 6 | queue/lease over the wire | **GREEN** | 18 submitted / 17 completed, 0 requeues, 0 duplicates, 0 expired leases |
| 6b | mid-lease worker kill (bonus) | **NOT RUN** | driver supported `--chaos-qr`; the sbatch never passed it |

Also green: regrade stability **0.9878** (±3% band), honest bf16-out ties to
reward exactly 1.0 (score 0.9936), `children-on-tpu` all `tpu`.

The run was **cut short**: `timer-tamperer` never returned (19+ min on the
chip), so the driver exhausted its 3600 s grade wait with 17/18 graded and
`all-graded` / `queue-drained` / `verdict:timer-tamperer` went red with it.

### Finding 1 — the phase-3 tolerance fix paid for itself in chip time

The cost is **not** where phase 2 predicted. The persistent worker did
exactly what it was designed to do — per-candidate fork/init is gone:

| stage | median | note |
|---|---|---|
| `export_s` (sandbox child, no device) | **0.8–1.4 s** | 14.0 s worst case |
| `load_s` (artifact deserialize) | **~0.0 s** | |
| `candidate_compile_s` (compile-warm, off the clock) | **2.2–3.7 s** | 45.7 s worst case |
| `warm_chip_s` (the ~4 s target metric) | **261.3 s** | **98% of the grade** |

So export + compile total ~4 s — the DESIGN estimate was right about
compile. The regression is entirely inside the timed chip section, and its
cause is `error_stats` in `judge/problems/base.py`: it upcast every output
leaf to **float64 on the host**, so at 32768×8192 (268 M elements) each call
moved ~10 GB of temporaries off-chip. The worker makes **~115 such calls per
candidate**, because phase 3's `calibrated_tolerance` evaluates
`reference_bf16` **plus three honest variants** per fixture — 4–5× the
phase-1 call count.

That is the causal story worth keeping: **phase-3 fix #3 (tolerance
recalibration across honest implementations), which turned the golden reds
green, is what made the cost red.** Phase 2 measured 112 s/candidate *with*
per-fork jax init and *without* multi-variant calibration; attempt 1 measured
261 s *without* the fork cost and *with* it. The two numbers are not the same
experiment, and quoting 112 → 261 as a regression in the persistent worker
would be wrong.

Fix (written during attempt 1, never deployed to it — the job had already
scp'd its code): one fused on-device pass per leaf, exact `max`/`mean`/
finiteness, q99 estimated from a deterministic strided subsample capped at
4,194,304 elements. Only scalars and that sample cross to the host.
Deterministic, so regrades stay bit-stable; `max` stays exact over every
element; and it is unexploitable — a candidate never learns the hidden seed,
the reference output, or which elements are compared.

### Finding 2 — the br512 golden's VMEM formula undercounted by ~3×

`pallas-br512` failed at `candidate_compile` on the **holdout** case
16384×6144: `CompileTimeScopedVmemOom`, **36.25 M requested vs 32.00 M
limit**. The judge was right; the golden was wrong. Phase 3's budgeter
counted only the bf16 in/out blocks double-buffered (8 B/element) and ignored
that the kernel body upcasts to f32, so `x32`, `x*x` and the f32 product are
all live inside the block. The measured truth is **23.05 B/element**
(36.25 M ÷ 256 × 6144). Budget constant corrected to 24 B/element.

### Finding 3 — a cheater that never reached the gate meant to catch it

`cached-output` was rejected (reward 0.0) but at `aot_export`, not
`correctness`: its hard-coded `(32, 64)` constant cannot broadcast against a
declared signature, so `jax.export` refuses to serialize it —
`TypeError: mul got incompatible shapes for broadcasting: (32, 64),
(8192, 4096)`, raised from the gradient functional. Rejection one gate
*earlier* is structurally stronger, never weaker.

But it exposed a real blind spot: **the CPU battery was passing for an
accidental reason.** Its smoke case `tiny` is exactly (32, 64) — the shape
the cheater hard-codes — so on CPU the constant broadcast fine and died at
`correctness`, and the "ignores its inputs, returns a plausible answer" cheat
was never actually exercised against the gate that is supposed to catch it at
production shapes. Added `constant-output`: broadcasts to whatever shape it
is handed, exports cleanly at every signature, must die at `correctness`.

### Finding 4 — unbounded grading is a fleet-wide poison pill

`timer-tamperer` (a 24-deep unjitted `sin` chain) held the judge for 19+
minutes and never returned. The damage does not stop at one candidate: the
lease expires, the queue requeues the item, and the **next** judge wedges
identically until the whole fleet is consumed. Two fixes:

- a per-grade wall budget (default 900 s) checked between phases — so the
  bound costs "one XLA op late" rather than needing an unsafe hard kill.
  A candidate this slow has already lost on score, so failing it at gate
  `budget` costs nothing legitimate;
- the fixture itself sized down 24 → 6 sins. It exists to prove timer
  tampering is neutralized and that a slow kernel scores well below 1;
  neither needs a deep chain. (Phase 2 already cut it 200 → 24 for the same
  class of reason.) 6 sins is still ~7× the baseline's traffic on a
  memory-bound task.

### Deviations

- **Chaos test not run in attempt 1** — `phase4/driver.py` implemented
  `--chaos-qr`, but `run_phase4.sbatch` never passed it. Wired for attempt 2,
  and hardened: it now uses the costliest *reliably-completing* variant so
  the ssh round-trip lands mid-grade, **proves** the kill was mid-lease by
  re-reading the item immediately after, and retries (up to 3 attempts) if
  the grade finished first — without that proof a race would score as a pass
  having tested nothing.
- **Always-delete had a hole** attempt 1 never hit: `trap cleanup EXIT` does
  not run when bash is killed by an untrapped SIGTERM, which is exactly what
  a slurm time limit, `scancel` or a node drain sends. That path would have
  stranded a spot QR with nobody watching. TERM/INT/HUP now run the same
  idempotent cleanup.
- **Worker stdout was invisible for the whole run.** gcloud's ssh pipe
  block-buffers, so the per-item lines never reached
  `phase4-worker-3646330.log` (it froze after the boot line at 14:15) and the
  run had to be diagnosed from queue access logs. The worker now tees to a
  host-side file that is scp'd back.

---

## Attempt 2 — sbatch 3646679, spot v6e-1, us-east5-b — **ALL GREEN**

Same chip, same battery, the four fixes in. Driver exit `rc=0`, **0 hard
failures**, 19/19 graded plus the chaos item.

Artifacts: `runs/pallas_arena/phase4-3646679.log`,
`phase4-results-3646679.json`, `phase4-queue-3646679.log`,
`phase4-worker-3646679.log`, `phase4-worker-progress-3646679.log`,
`phase4-boot-report-3646679.json`.

Bring-up: QR created 15:44:09 → ACTIVE **15:50:31 (6.4 min)** → provisioned
and driver running 15:51:50. Worker boot **55.8 s** (46.6 s in attempt 1;
the +9.2 s is exactly the new `calibration_warm_s` **9.63 s**, moved on
purpose from candidate #1's clock to boot). Driver 15:51:50 → 16:12:04.
QR deleted by the trap at 16:12:10; **zero pallas QRs and zero pallas TPU
nodes remain** in us-east5-b.

### Acceptance scorecard

| # | Item | Verdict | Measured (attempt 1 → attempt 2) |
|---|---|---|---|
| 1 | warm per-candidate cost ≈ 4 s | **GREEN** | median `warm_chip_s` **261.3 s → 1.944 s** (n=10, range 1.84–6.01). **134× faster**, 2.1× inside the 4 s target, 6.2× inside the 12 s cap |
| 2 | ref-vs-ref 1.00±2%, counterbalanced | **GREEN** | **0.9994 / 0.99996 / 0.9968** — max dev **0.32%** (attempt 1 1.14%, phase 2 5.3%) |
| 3 | recalibrated goldens all pass | **GREEN** | br16 0.4737, br256 0.8263, **br512 0.8258 (was VMEM OOM)**, unjitted-robust 0.8284, f32out 0.7834 — `failed: []` |
| 4a | determinism N≥5 bitwise | **GREEN** | enforced inside all 10 passing grades, zero violations |
| 4b | peak HBM per grade | **GREEN** | **16.78–20.92 GB**, max 20.92 GB |
| 5 | cheater battery at correct gates | **GREEN 11/11** | 9 gate-checked all at the expected gate; 2 structurally neutralized (see below) |
| 6 | queue/lease over the wire | **GREEN** | 20 submitted / 20 completed / 0 duplicates; 1 requeue + 1 expired lease, both the deliberate chaos kill |
| 6b | **mid-lease worker kill** | **GREEN** (was NOT RUN) | kill proven mid-lease (`state=leased`, `done=false` immediately after), `attempts=2`, regrade passed |

Also green: regrade stability **1.0126** (±3% band), honest bf16-out ties to
reward exactly 1.0 (score 1.0022), `children-on-tpu` all `tpu`.

### The headline: where the 261 s went

Fix #1 (device-side `error_stats`) did what it was designed to do. The full
per-candidate breakdown, now measured end to end:

| stage | attempt 1 median | attempt 2 median | max |
|---|---|---|---|
| `export_s` (sandbox child, no device) | 0.8–1.4 s | **0.82 s** | 13.8 s |
| `load_s` (artifact deserialize) | ~0.0 s | **0.0035 s** | 0.60 s |
| `candidate_compile_s` (off the clock) | 2.2–3.7 s | **2.98 s** | 569.5 s |
| **`warm_chip_s`** (the target metric) | **261.3 s** | **1.944 s** | 6.01 s |
| `item_wall_s` (whole lease, new) | — | **5.01 s** (all 19) / 5.93 s (passing) | 573.5 s |

`item_wall_s` is the number the fleet throughput table should be built
from, not `warm_chip_s`: **~5 s per candidate end to end**, and 0.74 s for
an AST-rejected cheater. At 5 judges that is **~60 candidates/min**, ahead
of the report's earlier ~30/min projection and short of DESIGN.md's
~75/min.

The one grade that still costs 6.01 s of chip time is `honest-xla` — the
first item leased. Every later grade sits at 1.84–1.99 s, so that is
residual first-touch compilation the boot warm-up does not reach, not a
property of the candidate.

### Cheater battery — 11/11, each at its correct gate

| cheater | gate | | cheater | gate |
|---|---|---|---|---|
| wrong-eps | `correctness` | | memoizer | `aot_export` |
| cached-output | `aot_export` | | wrong-grad | `gradient` |
| **constant-output** | **`correctness`** | | no-kernel | `exec` |
| aliased-reference | `ast` | | seed-reader | `correctness` |
| obfuscated-import | `poison_stub` | | | |

`constant-output` — the fixture added in attempt 2 to close the shape-
coincidence blind spot — died at `correctness` exactly as intended, so the
"ignores its inputs, returns a plausible answer" cheat is now genuinely
exercised against the gate meant to catch it at production shapes.

`split-personality` (0.7782) and `nondeterministic` (0.5681) pass, as
designed: the export path bakes python-level call counting and host-RNG
noise away at trace time, so those cheats cannot exist on the judge and the
artifacts grade as their honest selves.

`timer-tamperer` now **returns** (it never did in attempt 1) and scores
**0.0619** — a 16× slowdown, comfortably under the 0.95 bar.

### Chaos — SIGKILL a judge mid-lease, over the wire

First attempt landed it. The driver submitted the costliest reliably-
completing variant (`nondeterministic`: 13.7 s export + 46.2 s compile),
waited for the lease, then `pkill -9`'d the worker through gcloud ssh.
Re-reading the item immediately after returned `state=leased`,
`done=false` — **the kill is proven mid-grade, not a race that finished
first**. The lease then expired (`expired_leases: 1`), the queue requeued
(`requeues: 1`), the ssh restart loop brought the worker back, and it
regraded to `passed=true`, `attempts=2`.

Cross-restart agreement is a bonus result: **0.5650 on the second judge
process vs 0.5681 on the first — 0.55% apart**, inside the ±3% band, from
a completely fresh boot with independently measured noise floors. Final
accounting: 20 submitted, 20 completed, **0 duplicates, 0 lost work**.

### Residual — the wall budget does not bound XLA compilation

The one number worth not glossing over: `timer-tamperer` spent **569.5 s in
`candidate_compile_s`** and only 3.1 s on the clock. Its 573.5 s lease was
63% of the 900 s budget.

That refines finding 4. Attempt 1's 19-minute wedge was read as chip time;
it was mostly **XLA compiling the gradient probe through the sin chain**.
The budget is checked *between* phases, so it bounds everything except the
single unbounded operation that actually caused the wedge — a candidate
whose compile alone exceeds the budget still holds the lane until the
compiler returns.

It is not load-bearing for phase 4 (every item completed, nothing requeued
except by design) and the deeper 24-sin fixture that provoked it is gone,
but it should be closed before a fleet grades an adversarial corpus:
give the worker-side compile-warm its own deadline (the export child
already has `export_timeout_s=240`; the worker's compile does not), so a
pathological compile costs one lane one timeout instead of one lane 10+
minutes. Logged as the phase-5 wiring item.

### Deviations from the plan

- **None on scope.** One spot v6e-1 in us-east5-b, one QR at a time, always-
  delete honored on the single exit path taken (`rc=0` → trap → deleted;
  verified empty afterwards). No fleet demo launched (user-gated), no
  v5p/RL work.
- The `--chaos-qr` retry loop was widened 2 → 3 attempts before the run;
  it succeeded on attempt 0 and never used them.

---

## Fleet demonstration — spec (no RL)

Demonstrate the grading server at fleet scale: **5× spot v6e-1 in us-east5-b
all polling one queue**, a fixed corpus, correct verdicts, a judge killed
mid-lease, measured throughput, nothing left behind. No training, no
sampling, no RL loop.

### Topology

Queue on a neuronic compute node under sbatch (`judge/queue.py`, FIFO +
leases + heartbeat). Five spot v6e-1 judges, each reached by its own reverse
ssh tunnel from the orchestrator — the judge sees the queue at
`127.0.0.1:8770`. Judges are stateless pollers; the keeper's only job is
fleet size. This is exactly the phase-4 topology with N=5.

### Corpus

The 19-variant RMSNorm battery is too small to measure throughput, so:

- **~1600 graded candidates** = ~85 salted copies × 19 variants. Salt is a
  comment, so every copy is a distinct hash and a real grade (the
  `HONEST_XLA_B` trick, already used for the regrade invariant).
- **~100 deliberate duplicates** resubmitted after first completion, to
  demonstrate hash→reward cache hits (phase 2 measured **2.1 ms** with
  byte-identical reward).
- The battery already spans the needed cost/outcome range: instant AST
  reject → export-time reject → full grade, and scores from 0.48 (naive
  Pallas) through 1.0 (honest bf16) to a deliberately slow kernel.

### Acceptance

1. **Every candidate graded**, verdict matching its label — cheaters
   rejected at their expected gate, honest variants passed.
2. **Cross-judge reproducibility**: the same kernel graded on different
   chips agrees within ±3%. This is the genuinely new fleet property; the
   single-judge regrade test cannot show it.
3. **Chaos**: SIGKILL one judge mid-lease → its in-flight items requeue on
   lease expiry and complete on the survivors, `attempts >= 2`, final result
   set complete, no lost work.
4. **Throughput**: aggregate candidates/min and per-judge lane time.
5. **Zero resources left**: all 5 QRs deleted on every exit path, verified by
   an empty list.

### Built vs still to write

**Already built and exercised** — `judge/queue.py` (leases, heartbeat,
lazy expiry, idempotent duplicates; unit-tested + chaos-tested in phase-3
simulation + proven over the wire in phase 4); `judge/worker.py` (persistent
stateless judge); `judge/grader.py` + `child_runner.py` (AST, poison stubs,
RLIMIT, timeout, `aot_export` sandbox); `judge/problems/rmsnorm.py` (cases,
adversarial vectors, honest variants, calibration); `phase2/variants.py`
(the corpus and its expectations); `phase2/provision_judge.sh`
(chip-agnostic); `judge/cache.py` (GCS hash→reward, 2.1 ms hit measured);
`phase4/run_phase4.sbatch` (single-QR orchestrator with the always-delete
trap the fleet version generalizes).

**Still to write** (~1 day, CPU-simulable before any TPU spend — the phase-3
sim harness already runs N mock workers against a real queue):

| file | ~LOC | what's new |
|---|---|---|
| `phase5/fleet.sbatch` | 120 | create/await/provision N QRs in parallel, N tunnels, delete **all N** on every exit path |
| `phase5/corpus.py` | 60 | salted corpus + duplicate set |
| `phase5/fleet_driver.py` | 200 | generalizes `phase4/driver.py`; new: cross-judge agreement, throughput accounting |
| `phase5/keeper.py` | 80 | reconcile to 5 — delete SUSPENDED/FAILED, recreate, provision newly-ACTIVE (so one preemption doesn't end the demo) |

Plus small wiring: pass `--cache gs://sk7524-pallas-arena-us-east5/
reward-cache/` to the workers so the cache-hit item is real; and give the
worker-side compile-warm its own deadline (see "the wall budget does not
bound XLA compilation" above) — the one attempt-2 residual, and the only
one with fleet-wide blast radius.

### Cost and wall time

Per-candidate lane cost is no longer a projection — attempt 2 measured it
end to end as `item_wall_s`: **median 5.01 s** across all 19 variants,
**5.93 s** for a full passing grade, **0.74 s** for an AST-rejected cheater.
(Not `warm_chip_s`: 1.94 s is the chip slice only and would understate a
lane by ~3×.)

- Grading: 1600 candidates ÷ 5 judges × ~5.9 s ≈ **~31 min**
- Bring-up **6.4 min** measured in attempt 2 (5 QRs in parallel is the same
  wall if capacity allows) + boot 56 s + chaos ~5 min (240 s lease expiry +
  regrade) + teardown ~2 min
- **End to end ≈ 45–50 min**; **≈ 4 chip-hours** of spot v6e-1, plus one
  neuronic compute node for ~1.5 h.

Throughput at 5.9 s/lane: **~50 candidates/min** aggregate, or ~60/min on
the all-variants median. That lands between this report's earlier ~30/min
projection and DESIGN.md's ~75/min (which assumed a 4 s lane and did not
count the export child or the artifact compile). Treat ~50/min as the
honest planning input; the demo confirms it at N=5.

**Caveat on that number**: it is a *fleet* rate only if judges do not
serialize behind pathological compiles — the 573 s `timer-tamperer` lane
above is one candidate consuming what ~97 ordinary ones would. The
compile-warm deadline in the wiring list is what keeps the throughput
figure meaningful on an adversarial corpus.

**Risk**: spot v6e-1 capacity for 5 concurrent QRs in us-east5-b is
unproven — phase 0 saw a v6e-8 fail with "no more capacity" while v6e-1
landed on all three attempts so far. Mitigation: the keeper tolerates a
degraded fleet; the demo reports the actual N that landed rather than
failing, and every acceptance item except raw throughput is N-independent.
