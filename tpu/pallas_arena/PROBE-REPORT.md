# Cold-start probe — can a base model write a Pallas kernel at all?

*Status: RUNNING. This file is written incrementally so an interruption still
leaves a complete record of what was measured. Results sections are filled in
from `runs/pallas_arena/probe-results-<job>.json`.*

The question, precisely: **before any RL is built, do base models produce
trainable attempts at Pallas kernels?** Generation and grading only — no
policy, no gradients, no training.

"Trainable" is not the mean score. ttt-discover normalizes advantage *within
a group* of rollouts, so a configuration in which every candidate in a group
scores 0 contributes exactly zero gradient no matter how good its mean looks.
The headline number is therefore the **fraction of 16-candidate groups with
non-uniform reward**. A configuration with a 0.3 mean and uniform groups is
worth less than one with a 0.02 mean and mixed groups.

---

## What had to be built first

### A. Curated observations (`judge/observation.py`)

A verdict was `reward=0.0` plus a gate label and a `violations` list that is
frequently a raw traceback or a 900-character Mosaic diagnostic. The RL side
keeps only the **last 500 characters** of an observation
(`ttt_discover/tinker_utils/state.py:115`), so for a scoped-VMEM OOM the one
line carrying the number is scrolled off the top and the model is fed
`operand_layout_constraints={bf16[16384,6144]{1,0}, f32[1,6144]{1,0}}` instead.

The judge now curates the observation itself: gate, the specific violation,
and the measured numbers, capped at 480 characters so a tail-truncating
consumer keeps all of it. When one diagnostic is longer than the budget the
**middle** is elided — never the head, which names the gate, and never the
tail, which carries the numbers.

Worked examples, from the live battery:

```
GATE aot_export | splash_attention
pallas_call block does not fit in scoped VMEM
CompileTimeScopedVmemOom: 36.25M requested vs 32.00M limit (over by 4.25M); offending block bf16[16384,6144]
need: scoped VMEM ceiling is 32MB per pallas_call block; budget rows*cols*bytes_per_element under it
```

```
GATE correctness | splash_attention | probe-holdout-h4-s2049#seed0
output shape mismatch (4,2048,128) vs required (4,2049,128)
```

```
GATE exec | flce
ZeroDivisionError: integer division or modulo by zero [candidate line 31 in compute_block]
need: module-level code must import and define the entrypoint without executing the kernel
```

It is a pure function of the verdict dict: no hidden seed, no reference
output, no fixture data can leak through it. **37 unit tests**, one per gate
the judge can emit, each asserting the string survives a 500-character tail
cut whole (`tests/test_observation.py`). No change was needed in
`third_party/discover`.

### The judge could not have graded either task at all

Discovered while wiring the probe, and worth stating plainly because it was a
latent phase-6 blocker, not a probe artifact:

* `splash_attention.baseline()` was an unbound placeholder — a `jax.vmap`
  with `in_axes=(None,)` over `make_splash_mha`, marked "bound in ph2" and
  never bound. Worker boot would have died on it. It now calls the production
  kernel properly, with a query-blocked XLA fallback (recorded, not hidden)
  for shapes splash refuses.
* Neither task's fp32 **reference** fits on a judge. Splash's closed form
  materializes `[heads, seq, seq]` — 10.9 GB at `h8-s18432`, 43.5 GB at
  `h32-s18432`; FLCE's materializes `[n, v]` fp32 — 44.8 GB at
  `73728x151936`. A v6e-1 has 32 GB.
* The adversarial vector library was hard-wired to the `tiny` CPU-battery
  case, and its shapes are exported alongside the scored ones — so any judge
  running a non-default case set silently required candidates to trace at
  shapes no prompt declares.

Fixes: `ShapeCase.probe`, a one-chip case set per task at **unchanged vocab
and head_dim** (only the token/sequence axis shrinks, so the difficulty the
task exists for is intact), excluded from the default scored/holdout sets and
selected by name; a settable/skippable adversarial base case; and `--problem
'a:cases;b:cases'` so one chip serves both tasks.

| task | scored | scored | holdout (traced, not scored) |
|---|---|---|---|
| splash_attention | `heads=8, seq=4096, d=128` | `heads=4, seq=2048, d=128` | `heads=4, seq=2049, d=128` |
| flce | `n=4096, h=2880, v=151936` | `n=2048, h=2880, v=151936` | `n=3000, h=2880, v=151936` |

Each holdout is deliberately **not** block-divisible, so the explicit-padding
lesson survives the shrink.

### B. Three prompt variants per task (`probe/prompt_*.py`)

Strictly nested, so any arm-to-arm difference is attributable to the added
block alone (asserted in the dry run).

| task | minimal | reference | tailored |
|---|---|---|---|
| splash_attention | 2 676 ch (~669 tok) | 6 285 ch (~1 571 tok) | 15 373 ch (~3 843 tok) |
| flce | 3 359 ch (~839 tok) | 5 746 ch (~1 436 tok) | 9 305 ch (~2 326 tok) |

* **minimal** — task spec, signature, declared shapes, judge contract. Nothing
  else.
* **reference** — plus the public fp32 oracle inline (labelled as the
  correctness oracle and as an *invalid answer*, since it materializes what
  the task exists to avoid), plus a ten-item Pallas/Mosaic gotcha list drawn
  from failures this repo has actually hit: the ~32 MB scoped-VMEM ceiling
  with the literal error string and the measured 23–24 B/element, block sizes
  must tile, non-divisible shapes need explicit `jnp.pad`, the 128-lane /
  8-sublane native tiling, a raw `pallas_call` is not differentiable so
  fwd+bwd needs `jax.custom_vjp`, one kernel must trace at every declared
  shape with Python-int block sizes, the 90 s compile budget (and that a
  Python `for` over tiles compiles N copies where `lax.scan` compiles one),
  the jax 0.10.2 pin, and max-shifted/online LSE.
* **tailored** — plus a working scaffold in which every structural decision is
  **given** — `pallas_call`, grid, all `BlockSpec`s and index maps, the
  pad/un-pad math, the VMEM budget with its arithmetic shown, the flash
  accumulator scratch (splash) and the whole `custom_vjp` plumbing (FLCE) —
  and exactly **one** function is left blank: the inner kernel body.

Both scaffolds were validated by filling the blank and running them, not just
by `compile()`:

* splash scaffold, `interpret=True` on CPU at `(h2,s256,d128)`,
  `(h1,s257,d128)` (ragged) and `(h2,s128,d64)` (head-dim padding): max
  per-element error **3.06e-7**, padding rows exactly 0.0.
* FLCE scaffold on CPU at `n=64`, `n=45` (ragged) and `n=48`: forward within
  **1.4e-7** of the fp32 oracle, `custom_vjp` gradient within **5.05e-3**
  (plain bf16-matmul error).

One clause was added identically to all three splash variants: the arena
reference applies **no** `1/sqrt(head_dim)` scale (`make_inputs` pre-scales
`q`). Unstated, `minimal` would have been measuring "did you guess the
scaling convention" while the other two reveal it in the oracle — a confound
in the independent variable.

### C. The harness (`probe/`)

Reuses the phase-5 fleet code rather than replacing it: the same queue
(`judge/queue.py`), the same persistent worker, the same always-delete sbatch
shape, the same provisioning path.

* **Round-robin by group.** Each round issues one grouped `/v1/completions`
  request (n=16, one shared prefill) per configuration. At whatever moment
  the wall clock stops the run, every configuration has the *same* number of
  complete groups, so the comparison stays fair.
* **CPU pre-gate.** One v6e-1 at the phase-5 lane cost (5.6 s median) cannot
  grade 2 400 candidates inside a 3 h cap, and most first attempts never
  reach a chip anyway. So step 1 of `PersistentWorker.grade_code` — the same
  sandbox export child, AST gate + poison stubs + timeout + RLIMIT, no
  device, exporting *for* the TPU platform from a CPU host — runs 14-way
  parallel on the neuronic compute node first. `build_signatures` is now
  shared, so the pre-gate demands the *identical* exported contract; a
  candidate that fails there would have failed on the judge at the same gate.
  What the pre-gate cannot see is everything after StableHLO: the Mosaic/XLA
  backend compile (where `CompileTimeScopedVmemOom` lives), correctness,
  determinism, gradients, timing. Those are the judge's.
* **Rendering.** Qwen3 and Gemma4 turns are rendered by us and posted as text
  with `add_special_tokens: false`, because `Gemma4Renderer` emits a `<bos>`
  the naive template path does not, and the in-repo measurement for dropping
  it is **1/32 vs 7/8 compiling programs**. `verify_render.py` proves the
  rendering server-side through vLLM's `/tokenize` before a sample is spent.
* Sampling: temperature 1.0, top-p 1.0, `max_tokens` 6144, groups of 16.

CPU dry run of every moving part except the chips: **ALL GREEN** (sbatch
3650239) — prompts nested and declaring every probe shape, signatures exactly
matching the judge's contract (splash 3 fwd / no grad; flce 3 fwd + 1 grad),
8 hand-written candidates each at their correct gate with 136–255-character
observations, and the headline metric returning the right answer on a
synthetic set. The first dry run (3650236) caught a real one: **`flatbuffers`
missing from the pre-gate venv** made *every* candidate die at `aot_export`
with an `ImportError`, which would have read as "no model can write a kernel".

### D. Infrastructure

| QR | accel | zone | role |
|---|---|---|---|
| `sk7524-probe-serve` | v5p-16 (2 hosts × 4 chips) | us-east5-a | worker 0 `Qwen/Qwen3.5-27B`, worker 1 `google/gemma-4-31B-it`, each vLLM TP=4 |
| `sk7524-probe-judge` | v6e-1 | us-east5-b | arena judge, both problems on one chip |

Sampling only: no tinker server, no training stack. `probe/serve_vllm.sh` is
deliberately *not* `tpu/start_vllm_tpu.sh` — that script installs a
`tpu-inference` fork purely for LoRA forwarders and asserts they exist; the
probe pushes no adapters, so stock `vllm-tpu==0.23.0` is both sufficient and
one git-clone faster. Warm caches: `gs://sk7524-tinker-tpu-us-east5/hf-cache`
+ `vllm-xla-cache-22k` for Qwen at `--max-model-len 22528`, and
`hf-cache-gemma4` + `vllm-xla-cache-gemma4-31b-16k` for gemma at 16384 (the
prefixes whose bucket sizes match those context lengths). `SKIP_JAX_PRECOMPILE=1`:
a cold all-buckets precompile has been measured in-repo at ~55 min, a third of
the entire wall budget.

Hard rules, in the script rather than in a comment: at most these two QRs
alive; deleted on **every** exit path including untrapped SIGTERM (phase 4
found `trap cleanup EXIT` does not fire on it, which is exactly what a slurm
timeout sends); a 3 h cap from the first create; a 45 min landing rule with
**no** fallback to a bigger slice; and both zones verified empty of both names
at the end.

Regression: the full arena battery is **175 passed, 0 failed** with all of the
above in place (sbatch 3650246) — phase 5's 138 plus the 37 new observation
tests. Nothing about the production path changed.

---

## Bring-up (job 3650240)

| event | time | elapsed |
|---|---|---|
| both QRs created | 09:34:56 | — |
| `sk7524-probe-serve` (v5p-16) ACTIVE | 09:40:09 | **5 min 13 s** |
| `sk7524-probe-judge` (v6e-1) ACTIVE | 09:42:20 | 7 min 24 s |
| judge provisioned, shared reward cache RW OK | 09:43:48 | 8 min 52 s |
| judge booted **both** problems | 09:44:4x | ~9 min 50 s |
| vLLM installed on both hosts (stock `vllm-tpu==0.23.0`) | ~09:41:40 | ~6 min 45 s |

The v5p-16 landed comfortably inside the 45 min rule, so the no-fallback
branch was not exercised.

### The judge serving two tasks from one chip — boot report

| | splash_attention | flce |
|---|---|---|
| boot | 20.23 s | 24.91 s |
| calibration warm | 10.42 s | 11.78 s |
| ref-vs-ref, case 1 | 1.000119 | 0.998555 |
| ref-vs-ref, case 2 | 0.991150 | 0.999165 |
| noise floor, case 1 | 0.00339 | 0.00694 |
| noise floor, case 2 | 0.02843 | 0.01578 |
| baseline actually used | **`xla-fallback`** | production `custom_vjp` |

Both ref-vs-ref measurements are inside the ±2% band the arena requires, on
a chip that had never run either task.

**Caveat that must travel with every splash number below**: `baseline_impl`
came back `xla-fallback`, i.e. Google's production Pallas splash kernel
refused the probe shapes and the score denominator is the query-blocked XLA
attention instead. Splash scores are therefore "speed versus a competent XLA
implementation", not "speed versus splash". This does not affect any gate —
compile, correctness, determinism and the group-uniformity headline are all
independent of the denominator — but it does mean a splash score near 1.0
should not be read as parity with the production kernel.

### The serving slice needed a fix that is worth recording

Both engines initially hung with

```
jax/_src/xla_bridge.py: UserWarning: TPU backend initialization is taking
more than 60.0 seconds. Did you run your code on all TPU hosts?
```

and never reached the serving loop. On a v5p-16 (2 hosts × 4 chips) libtpu
defaults to forming the **full two-host mesh**, so a single-host vLLM blocks
waiting for a peer that is itself running an independent server. Unsetting
`TPU_VISIBLE_CHIPS` does not make a host standalone.
`TPU_PROCESS_BOUNDS=1,1,1` with `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` pins each
runtime to its own 4 chips — what TP=4 wants, and what
`start_colocated_vllm_tinker.sh` already does for its single-worker vLLM. It
also has to be threaded into the `tmux` environment, which does not inherit
the caller's exports. Fixed in `26c840d9` and relaunched out-of-band; neither
QR nor the judge was disturbed.

### Chat rendering, verified before a single sample was spent

| model | our rendering == server's own chat template | special tokens all single | sample non-empty |
|---|---|---|---|
| `Qwen/Qwen3.5-27B` | **True** | True | True |
| `google/gemma-4-31B-it` | **True** | True | True |

```
qwen35-27b: '<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n'
gemma4-31b: '<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n…<turn|>\n<|turn>model\n'
```

Token-for-token agreement through vLLM's `/tokenize`, including gemma's
explicit `<bos>` and its injected thinking-mode system turn. The probe is
measuring the models, not a rendering bug.

**Engines live at 10:00:59** — 26 min from QR create. Generation started
**10:05:38** with an 8058 s wall budget, judge grading enabled.

## Results

*(filled in at the end of the run)*

## Verdict

*(filled in at the end of the run)*

## Resources

*(filled in at the end of the run)*
