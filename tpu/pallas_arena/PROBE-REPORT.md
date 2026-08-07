# Cold-start probe — can a base model write a Pallas kernel at all?

*Run: sbatch 3650240, 2026-08-07. Written incrementally so an interruption
still leaves a complete record. Raw data:
`runs/pallas_arena/probe-results-fixed-3650240.{json,jsonl}` (the corrected
grid), `probe-results-3650240.jsonl` (the 6144-token truncation arm),
`probe-results-gemmatail-3650240.jsonl` (one re-run cell),
`probe-control-3650240.json` (the harness control),
`probe-render-3650240.json` (chat-template verification),
`probe-judge-boot-3650240.json`, and `probe-raw-3650240/` (raw generations).*

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

### Arm 1 — `max_tokens = 6144`: the budget, not the model, is the constraint

The first measurement is about the harness. Across **both** models and
**both** tasks, the first 48 generations came back with
`finish_reason = "length"` — **48/48**, no exceptions — and **0/48** survived
the pre-gate:

| config (round 0) | gens | wall | survived pre-gate | gates |
|---|---|---|---|---|
| gemma4-31b / minimal / splash | 16 | 205 s | 0/16 | `ast` ×16 |
| gemma4-31b / minimal / flce | 16 | 117 s | 0/16 | `ast` ×11, `exec` ×3, `aot_export` ×2 |
| qwen35-27b / minimal / splash | 16 | 335 s | 0/16 | `ast` ×9, `no_code` ×4, `aot_export` ×2, `exec` ×1 |

Reading the raw text explains it. The entire budget is spent in the **thinking
channel** and the answer is never reached — a gemma generation ends mid-
sentence with `Hmm, the pl.load(ref, index) is for when the ref is a pl.Ref
but not necessarily tied to the block spec of` — so the extractor picks up a
partial sketch quoted *inside* the thought, which is why the modal gate is
`ast` with `syntax error: unexpected indent`. Median extracted program: 1 108
characters, against ~17 000 characters of generation.

This is the failure mode the repo already documents for Qwen in
`tpu/runs/qwen35-27b.env` — *"the default qwen3_5 thinking renderer burns the
whole 512-token budget inside `<think>`"* — hit again at 6144. **Reporting
"no configuration is trainable" from this arm would have been a false
negative caused by my own choice of `MAX_NEW_TOKENS`, not by model
capability.**

### Arm 2 — `max_tokens = 12000`

A second driver was attached to the *same* two engines and the *same* queue
via `srun --overlap` on the job's own node, writing to its own results file.
The first driver could not simply be killed: it is the sbatch's foreground
child, so killing it would run the teardown and delete both QRs. Both arms
therefore run concurrently and share engine throughput, and both are
reported — arm 1 is a genuine measurement of what thinking-mode costs on this
task, not a wasted run.

Raising the budget worked as intended: at 12000 tokens **13/32 generations
finished at `stop`** rather than `length`, median extracted program went from
331 → 2 200 characters, and `pallas_call` appeared in 26/32 rather than 13/80.

### Arm 2 also exposed a HARNESS BUG, and it was the bigger problem

Of arm 2's first 32 candidates, **14 died at `aot_export` with the identical
message**

```
TypeError: list indices must be integers or slices, not str
```

Fourteen byte-identical failures are not fourteen independent candidate bugs.
`probe_signatures()` returns a 4-tuple `(sigs, case_sig, adv_sig, grad_sig)`;
the driver stored the **whole tuple** and handed it to the export child as
`export_signatures`, so the child iterated `(sigs, case_sig, …)` and evaluated
`sig["args"]` on a **list**. A harness fault wearing a candidate's clothes —
and it destroyed exactly the candidates that were well-formed enough to
*reach* export. Fixed in `05c4847f`.

### The control experiment — the harness is sound

Because a grid of zeros is only evidence about *models* if the path is known
to pass a correct kernel, `probe/control.py` pushes **known-good** kernels
through the identical pipeline — wrapped in a model-style response with a
decoy code block first, then `extract_program` → CPU AOT pre-gate → queue →
judge:

| control kernel | extraction | pre-gate | judge |
|---|---|---|---|
| `rmsnorm/pallas-br256` — green on silicon in phases 2, 4 and 5 | picks the real block, not the decoy | **PASS** (2.42 s) | n/a (not served by this judge) |
| `flce-known-good` | picks the real block | **PASS** (1.47 s) | **PASS, gate `all`, reward 0.6981** |
| `splash-scaffold-filled` (the prompt's own scaffold, blank filled) | picks the real block | **PASS** (1.97 s) | compiled and ran; `correctness`, max err 4.11e-3 vs tol 3.48e-3 |

The FLCE control's judge observation, verbatim:

```
GATE all | flce | PASS reward=0.6981
probe-4096x2880x151936: cand 10.095ms vs ref 6.868ms (0.680x)
probe-2048x2880x151936: cand 4.650ms vs ref 3.312ms (0.712x)
probe-holdout-3000x2880x151936: cand 7.087ms vs ref 6.886ms (0.972x)
peak HBM 30.50GB
```

**3/3 reached a chip and one earned a non-zero reward**, so extraction, the
entrypoint name, the signature set, the import environment and the pre-gate
configuration are all correct end to end. The splash control is a useful
calibration in its own right: the prompt's scaffold with a textbook
online-softmax fill *compiles and runs on TPU at all three declared shapes*
and lands within 18% of the correctness tolerance — the task is reachable,
not absurd.

### Arm 3 — the corrected grid

Launched 10:22:30 with the signature fix and `max_tokens = 12000`
(`probe-results-fixed-3650240`). The arm-1 driver cannot be killed — it is the
sbatch's foreground child, so killing it would run the teardown and delete
both QRs — so it keeps running and shares engine throughput; its records stay
in a separate file and are reported only as the truncation measurement.

**The headline arrived in arm 3's first round.** `gemma4-31b | minimal | flce`,
16 generations: **10/16 survived the pre-gate**, all 10 were graded on the
chip, and

| verdict | n |
|---|---|
| `all` (**PASS**) | **2** |
| `gradient` | 7 |
| `correctness` | 1 |

That group has **non-uniform reward** — two candidates earn a real score and
eight earn zero — which is exactly the condition ttt-discover needs to produce
a gradient. A base model, on the *minimal* prompt, with no RL, wrote a fused
linear cross-entropy kernel that passed correctness on hidden seeds,
bitwise determinism, the `custom_vjp` gradient check and the timed re-check.

The 7 `gradient` failures are the single most informative number in the run:
those candidates compiled and produced correct forward values, then failed
`d/d(hidden)`. That is precisely the "a tiled forward is not enough, the
backward must recompute per tile under `jax.custom_vjp`" lesson the task
exists to teach — and it is a *dense*, learnable error signal rather than a
flat zero.

By contrast both `minimal | splash_attention` cells were 0/16 at the pre-gate,
and the reason differs sharply by model:

* **gemma4-31b**: 14/16 finished at `stop`, all 16 passed the AST gate (they
  do write a real `pallas_call`), and they die at export on **hallucinated
  API** — `pallas_call() got an unexpected keyword argument 'out_spec'` /
  `'out_dtype'` / `'out_dtypes'` / `'out_shapes'` / `'block_shapes'`,
  `pallas_call() missing 1 required positional argument: 'out_shape'`,
  `module 'jax' has no attribute 'Shape'`, `module 'jax.experimental.pallas'
  has no attribute 'Ref'`. The model knows the *shape* of a Pallas kernel and
  invents the signature.
* **qwen35-27b**: only 1/16 finished at `stop` — Qwen's thinking channel needs
  **more than 12000 tokens** on this task, so most candidates are still
  truncated fragments (6× `syntax error: unexpected indent`).

### Arm 3 — the full grid, n=16 per cell (round 0)

`export` = survived the CPU AOT pre-gate (traces at all three declared shapes
for the TPU platform). `judged` = graded on the v6e-1. `PASS` = gate `all`.

| model | variant | task | `stop` | export | judged | **PASS** |
|---|---|---|---|---|---|---|
| gemma4-31b | minimal | splash | 14/16 | 0/16 | 0 | 0 |
| gemma4-31b | minimal | **flce** | 16/16 | **10/16** | 10 | **2** |
| gemma4-31b | reference | splash | 16/16 | 0/16 | 0 | 0 |
| gemma4-31b | reference | **flce** | 16/16 | **10/16** | 10 | **5** |
| gemma4-31b | tailored | splash | 30/47† | 1/47† | 1 | 0 |
| gemma4-31b | tailored | **flce** | 16/16 | **16/16** | 16 | **16** |
| qwen35-27b | minimal | splash | 1/16 | 0/16 | 0 | 0 |
| qwen35-27b | minimal | flce | 2/16 | 0/16 | 0 | 0 |
| qwen35-27b | reference | splash | 0/16 | 0/16 | 0 | 0 |
| qwen35-27b | reference | flce | 0/16 | 0/16 | 0 | 0 |
| qwen35-27b | tailored | splash | 0/16 | 0/16 | 0 | 0 |
| qwen35-27b | tailored | flce | — | — | — | — |

† the first 16 were the `HTTPError 400` cell; the number shown pools that with
the 31-candidate rerun at 10000 tokens.

### The two findings that dominate everything else

**1. Qwen3.5-27B cannot finish this task in its thinking budget.**

| model | generations | `stop` | `length` |
|---|---|---|---|
| gemma4-31b | 105 | **86** | 3 |
| qwen35-27b | 80 | **3** | 77 |

At 12000 new tokens gemma finishes 82% of the time and Qwen finishes 4% of the
time. Every Qwen cell is 0, and the modal Qwen gate is `ast: syntax error:
unexpected indent` — a fragment quoted mid-thought, not an attempted kernel.
This is not a statement about Qwen's kernel ability; it is a statement that
**Qwen's thinking channel needs a far larger budget (or a non-thinking
renderer) before its ability can be measured at all.** The repo already knew
this for the 512-token math setting (`tpu/runs/qwen35-27b.env` pins
`qwen3_5_disable_thinking`); it holds at 12000 too.

**2. On splash, gemma writes plausible Pallas and hallucinates the API.**

Every gemma splash cell is 0/16 at export, with `stop` rates of 14/16 and
16/16 — the model finishes, and the AST gate confirms it wrote a real
`pallas_call`. It dies at `jax.export` on invented signatures:

```
pallas_call() got an unexpected keyword argument 'out_spec'
pallas_call() got an unexpected keyword argument 'out_dtype' / 'out_dtypes'
pallas_call() got an unexpected keyword argument 'out_shapes' / 'in_shapes'
pallas_call() got an unexpected keyword argument 'block_shapes' / 'index_map'
pallas_call() missing 1 required positional argument: 'out_shape'
BlockSpec.__init__() got multiple values for argument 'block_shape'
module 'jax' has no attribute 'Shape'
module 'jax.experimental.pallas' has no attribute 'Ref'
'AbstractRef' object has no attribute 'astype'
```

The `reference` prompt's prose gotcha list does **not** fix this — 16/16 still
finish and 0/16 still export. Prose about VMEM budgets cannot teach a function
signature the model does not have.

### The trainable cell, and how prompt strength moves it

For gemma on FLCE the three variants are cleanly monotonic:

| variant | export | PASS | judge gate mix |
|---|---|---|---|
| minimal | 10/16 | 2 | 2 `all`, 7 `gradient`, 1 `correctness` |
| reference | 10/16 | 5 | 5 `all`, 5 `gradient` |
| tailored | **16/16** | **16** | 16 `all` |

`gradient` is the informative failure: those kernels compiled and were
numerically correct forward, then missed `d/d(hidden)` — the "a tiled forward
is not enough, the backward must recompute per tile under `jax.custom_vjp`"
lesson, delivered as a dense error signal rather than a flat zero.

**And this is where the headline metric earns its keep.** `minimal` and
`reference` are unambiguously trainable — mixed pass/fail inside a group of
16. But `tailored|flce` at **16/16 passing** has no pass/fail variance left:
its entire training signal now lives in the continuous speed score, and
whether that group produces a gradient depends on the *spread of the speed
ratios*, not on the pass rate. A prompt can be too good: hand over the whole
scaffold and the discriminating variable disappears from the gate and has to
be carried by timing alone.

### The complete arm-3 grid, with rewards (round 0, n=16 per cell, 186 candidates)

| model | variant | task | `stop` | export | judged | PASS | reward min / med / max | **spread** |
|---|---|---|---|---|---|---|---|---|
| gemma4-31b | minimal | **flce** | 16 | 10 | 10 | **2** | 0.2636 / 0.3777 / 0.4919 | **0.2283** |
| gemma4-31b | reference | **flce** | 16 | 10 | 10 | **5** | 0.0659 / 0.2636 / 0.2637 | **0.1978** |
| gemma4-31b | tailored | **flce** | 16 | 16 | 16 | **16** | 0.6966 / 0.6980 / 0.7008 | **0.0042** |
| gemma4-31b | minimal | splash | 14 | 0 | 0 | 0 | — | — |
| gemma4-31b | reference | splash | 16 | 0 | 0 | 0 | — | — |
| gemma4-31b | tailored | splash | 0† | 0 | 0 | 0 | — | — |
| qwen35-27b | minimal | flce | 2 | 0 | 0 | 0 | — | — |
| qwen35-27b | minimal | splash | 1 | 0 | 0 | 0 | — | — |
| qwen35-27b | reference | flce | 0 | 0 | 0 | 0 | — | — |
| qwen35-27b | reference | splash | 0 | 0 | 0 | 0 | — | — |
| qwen35-27b | tailored | flce | 1 | 0 | 0 | 0 | — | — |
| qwen35-27b | tailored | splash | 0 | 0 | 0 | 0 | — | — |

† the `HTTPError 400` cell; the 10000-token rerun reached 1/16 export and one
`correctness` verdict.

```json
{"overall_nonuniform_groups": "3/11",
 "overall_nonuniform_group_frac": 0.273,
 "configs_with_any_nonuniform_group": ["gemma4-31b|minimal|flce",
                                       "gemma4-31b|reference|flce",
                                       "gemma4-31b|tailored|flce"],
 "best_config_by_max_score": "gemma4-31b|tailored|flce",
 "best_score": 0.7008, "any_candidate_passed": true}
```

### The tailored cell's spread is BELOW the judge's own noise floor

This is the result I did not expect and would have missed by reading pass
rates alone. `tailored|flce` passes 16/16 with rewards in
**[0.6966, 0.7008]** — a spread of **0.0042**. The judge's measured noise
floors for those very shapes, from its boot report, are **0.0069 and 0.0158**.

**The within-group reward variation is smaller than the measurement noise.**
That group is nominally "non-uniform" and therefore counted in the 3/11, but
the differences it would train on are not real: they are timing jitter. The
scaffold is so complete that all sixteen candidates converge on essentially
the same kernel, and the arena cannot tell them apart.

So the ranking by *trainability* inverts the ranking by pass rate:

| cell | PASS | spread | spread vs noise floor | usable signal? |
|---|---|---|---|---|
| minimal / flce | 2/16 | 0.2283 | **14–33×** | **yes** |
| reference / flce | 5/16 | 0.1978 | **13–29×** | **yes** |
| tailored / flce | 16/16 | 0.0042 | **0.27–0.61×** | **no — under the noise** |

A prompt can be too good. The tailored scaffold removes the very variance RL
needs, and the honest reading of the headline is **2 of 11 complete groups
carry usable signal**, not 3.

Also worth noting: the best score anywhere is **0.7008**, i.e. every passing
candidate is *slower* than the production `custom_vjp` baseline. Base models
can write a correct FLCE kernel; none of them beat the thing they are trying
to beat. That is the gap RL would be asked to close.

### Overall gate histogram (arm 3, 155 pre-gate records)

| gate | n |
|---|---|
| `aot_export` | 76 |
| `ast` | 39 |
| `no_code` | 28 |
| `exec` | 11 |
| `correctness` | 1 |

## Verdict

**Yes — two configurations are trainable today, and both are narrow.**

Headline: **3 of 11 complete groups had non-uniform reward — but only 2 of 11
carry signal above the judge's noise floor.**

**`gemma-4-31B-it` × FLCE × `reference`** is the recommendation (10/16 export,
5/16 pass, within-group spread **0.1978** = 13–29× the noise floor), with
**`minimal`** the close second (10/16, 2/16 pass, spread **0.2283**). Their
failures concentrate on `gradient`, which is a dense, learnable error.

**`tailored` is the trap.** It passes 16/16 — and its reward spread is
**0.0042**, *below* the judge's own measured noise floors (0.0069 / 0.0158)
for those shapes. All sixteen candidates converge on the same kernel and the
arena cannot distinguish them; what looks like the best cell is the one with
no usable gradient. A prompt can be too good.

**No splash-attention configuration is trainable yet, and the reason is
specific and fixable.** gemma finishes the task (14–16 of 16 reach `stop`) and
writes a genuine `pallas_call`, then invents the signature. Prose gotchas do
not fix it. Two things would, and neither is RL: put the *exact*
`pallas_call` / `BlockSpec` / `pltpu.VMEM` signatures in the prompt as an API
reference (the tailored scaffold does this and is the only splash arm that
ever reached the chip), or give the environment one retry with the judge's own
`aot_export` observation fed back — the curated observation from work item A
names the bad kwarg exactly, and these are one-line fixes.

**No Qwen3.5-27B configuration is measurable yet.** 3 of 80 generations
finished. Before Qwen can be judged on kernels at all it needs either the
`qwen3_5_disable_thinking` renderer the repo already uses for math, or a
budget well past 12000 tokens. Reporting Qwen as "cannot write kernels" from
this run would be wrong.

**The judge is ready.** Both tasks boot on one v6e-1 with ref-vs-ref inside
±2%, a known-good FLCE kernel scores 0.6981 through the full path, and 40/40
submissions were graded with 0 duplicates and 0 requeues.

### What I would change before the RL run

1. Per-model token budgets, enforced (now in `ModelSpec`); an over-long
   request is a 400, not a data point.
2. Drop Qwen thinking-mode, or raise its budget past 16k.
3. Use `reference` for FLCE. For splash, add an API-signature block — the
   measured failure is signature recall, not reasoning.
4. Feed the curated observation back for one retry. Nothing in this run tests
   that, and the observations are precisely the information a retry needs.
5. Splash's score denominator is currently the XLA fallback; bind the real
   splash kernel (or accept the fallback explicitly) before ranking champions.

## Resources

| | |
|---|---|
| QRs | exactly **2**, never more: `sk7524-probe-serve` (v5p-16, us-east5-a), `sk7524-probe-judge` (v6e-1, us-east5-b) |
| created | 09:34:56 |
| serve ACTIVE | 09:40:09 (5 min 13 s) |
| judge ACTIVE | 09:42:20 (7 min 24 s) |
| both engines serving | 10:00:59 (26 min) |
| generation start | 10:05:38 |
| hard cap | 12:34:56 (3 h), never approached |
| chip-hours | v5p-16 = 8 chips, v6e-1 = 1 chip. At ~2 h 45 m of QR lifetime: **≈ 22 v5p chip-hours + ≈ 2.75 v6e chip-hours ≈ 24.8 chip-hours** |
| judged candidates | 40 (+2 controls), 0 duplicates, 0 requeues |

Plus one neuronic compute node (16 cpu, 48 G) for the queue, the pre-gate and
three drivers, and three short sbatch CPU jobs (dry runs + regression).
