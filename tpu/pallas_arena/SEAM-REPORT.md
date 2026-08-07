# The seam — a prompt that gives away the plumbing and keeps the decisions

*Run: 2026-08-07. CPU jobs 3650754 / 3650776 / 3650796 / 3650817 / 3650852.
Raw data: `runs/pallas_arena/seam-control-3650852.json`,
`runs/pallas_arena/seam-c{1,2,3}-*.log`, `runs/pallas_arena/seam-dry-3650817.log`.*

**Read this first: the TPU measurement did not happen.** `gcloud` credentials
on the login node have expired (`Reauthentication failed. cannot prompt during
non-interactive execution`), which is a report-and-stop condition. **Zero
queued resources were created, so zero were leaked** — but the honest
consequence is that every model-facing number this report was supposed to
carry (compile rate, pass rate, gate histogram, reward distribution, best
score, and the noise-floor-relative spread verdict) is **UNMEASURED**. §6 says
exactly what is and is not known, and §7 is the single command that produces
the missing half once auth is restored.

What *is* measured, on CPU, is the whole harness: the seam composes, the
composed programs are numerically correct against each problem's own fp32
reference, they export at every declared shape including the non-divisible
holdouts, and three of the five tasks — which **could never have been graded
at all** — now can be.

---

## 1. Why a new variant, in one paragraph

The cold-start probe measured that 47% of candidates die at the static gate
and 14% emit no code at all: models flail on *scaffolding*, not algorithms.
Every gemma splash cell was **0/16 at export** while 14–16 of 16 reached
`stop` — they write real `pallas_call` structure and then invent the
signature (`out_spec`, `out_dtype`, `jax.Shape`, `pl.Ref`). The `reference`
prompt's ten-item prose gotcha list moved that number by **zero candidates**.

And the opposite failure is just as bad. `tailored`, the fill-in-the-blank
variant, passed **16/16** — with a within-group reward spread of **0.0042**,
*below* the judge's own measured noise floors (0.0069 / 0.0158). All sixteen
candidates converged on the same kernel. Nominally "non-uniform", actually
timing jitter. **A prompt can be too good**, and pass rate is the wrong
headline.

The seam cuts between the two: **the harness owns the interface, the model
owns the strategy.**

## 2. The design

Per task the prompt shows, verbatim, the scaffold the answer will be wrapped
in, and asks only for named fill-in functions. After extraction the harness
composes `fill + scaffold` into one module and submits that.

```
model text --extract_fill--> fill source --compose--> program defining `kernel`
```

**The judge contract is unchanged.** The submission is still one
self-contained module with a module-level `kernel`; the judge cannot tell it
was assembled. The scaffold goes *last*, so a model that pastes a whole
program back is harmless — the harness's `kernel` is the one that survives.

`prompt_seam.py` interpolates `seam.SCAFFOLDS[task]` **itself**, not a retyped
copy, so what the model reads is byte-identical to what its code is appended
to. `seam_dryrun.py` asserts that for all five.

| task | harness owns | model writes | the real decisions left open |
|---|---|---|---|
| **flce** | token-axis tiling driver, pad/un-pad, `custom_vjp` registration, residual contract | `TILE`, `tile_forward(h,w,t)->(lp,carry)`, `tile_backward(h,w,t,g,carry)` | tile size, vocab chunking, one-pass online LSE vs two-pass, accum dtype, **recompute-vs-save** (that is what `carry` is) |
| **splash_attention** | the ENTIRE `pallas_call` — grid, every `BlockSpec`/`index_map`, `out_shape`, padding of *both* axes, un-pad slice | `choose_blocks(seq,d)->(bq,bk)`, `attn_block(q,k,v,sq,sk,o,*,bq,bk,kv_len)` | block shape, key chunking, skipping blocks above the diagonal, online vs two-pass softmax, accum dtype + matmul precision |
| **ragged_paged_attention** | `pallas_call`, grid `(seq, page)`, BlockSpecs, and the page-id→page-slice helper (scalar-prefetched `page_tables` in the k/v index maps) | `decode_block(...)` | mask vs `pl.when` loop-bound for the ragged tail, online vs two-pass, how to sweep the 8 kv heads, accum dtype |
| **megablox_gmm** | exclusive scan to offsets, **group-aligned row permutation** (zero-size groups and boundary tiles are the harness's trap, not the model's), `pallas_call` + scalar-prefetched per-tile expert ids, un-permute | `choose_tiles(m,k,n,g)->(bm,bk,bn)`, `gmm_tile(l,r,o,*,bm,bk,bn,k)` | tile shape (= the VMEM budget), k-loop chunking and order, accum dtype |
| **rg_lru** | outer entrypoint, time chunking + padding, carry-state and reset contract | `CHUNK`, `scan_chunk(x,a,reset,h_prev)->(h_chunk,h_last)` | chunk length (the whole sequential/parallel trade), `lax.scan` vs `lax.associative_scan`, fp32 state handling, layout |

Measured fill sizes for the known-good controls: **110–423 tokens**, against a
300–1200 target and a 2000 ceiling. Prompts are 2 849–3 684 tokens (seam) and
1 436–1 874 (reference), so a 12 000-token completion fits behind either inside
gemma's 16 384 window — and the driver now asks the server's own `/tokenize`
for the exact prompt length and clamps per request, because the previous run
lost an entire 16-candidate cell to `HTTPError 400` from a char/4 estimate.

### The API block is introspected, not remembered

Prose demonstrably did not fix signature hallucination, so every seam prompt
carries the real signatures at the pin (jax **0.10.2**) — `pallas_call`,
`BlockSpec` (positional `(block_shape, index_map)`, third positional is
`pipeline_mode`), `GridSpec`, `PrefetchScalarGridSpec`, `CompilerParams`,
`ShapeDtypeStruct`, `program_id`/`num_programs`/`when`/`ds` — **and the names
that do not exist**: `pl.load`, `pl.store`, `pltpu.ANY`,
`pltpu.TPUCompilerParams`, `jax.Shape`, and every invented `pallas_call`
kwarg. `seam_dryrun.py` re-derives all of it with `inspect.signature` and
fails if a line has drifted (33 assertions, all green).

Two facts in that block came out of introspection and would have been wrong
from memory: **`pl.load`/`pl.store` do not exist in 0.10.2 at all** (a Ref is
read and written by indexing), and `ANY` is `pl.ANY`, not `pltpu.ANY`.

A third came out of the *previous* control run and is a genuine numerics
lesson: on TPU, `jax.lax.dot_general` on **float32** operands runs **one
bfloat16 pass** at default precision. The previous splash control missed the
judge's calibrated tolerance by 18% (max err 4.11e-3 vs a 3.48e-3 budget) for
that reason alone. The block now states it as the speed/accuracy trade it is,
and the run ships **two** splash controls — `HIGHEST` and default — so the
judge *measures* which side of the tolerance the fast one lands on instead of
anyone guessing.

## 3. Three of the five tasks could never have been graded

This is the scope-change prerequisite, and it was a real blocker, not a
formality. The persistent worker calls `jax.jit(problem.baseline)` at boot and
treats **any** exception — including `BaselineUnavailable` — as *"problem not
served"* (`worker.py:201`, `:970`). It does not fall back.

* `ragged_paged_attention.baseline()` raised `BaselineUnavailable` on **both**
  branches — unconditionally, even on TPU.
* `rg_lru.baseline()` raised on TPU whether or not `recurrentgemma` imported.
* `megablox_gmm` was the only one that returned, and it was unguarded.

Fixed, with the same trade splash already makes — *a judge that refuses to
boot has graded nothing at all, which is strictly worse than scoring against a
slower-but-real denominator* — and every fallback **labelled**:

| task | baseline actually used | `baseline_impl` | honest reading of a score |
|---|---|---|---|
| flce | our production `custom_vjp` tiled kernel | n/a | vs the thing we ship |
| splash_attention | production splash MHA, else query-blocked XLA | `pallas-splash-mha` / `xla-fallback` | last run it was the **fallback** |
| ragged_paged_attention | batch-blocked XLA paged decode | `xla-paged-decode-fallback` | **not** vs vLLM's Pallas v3 kernel |
| megablox_gmm | tuned Pallas megablox `gmm`, else `ragged_dot` | `pallas-megablox-gmm` / `lax-ragged-dot-fallback` | — |
| rg_lru | `lax.associative_scan` | `lax-associative-scan` | **not** vs recurrentgemma's scan |

`baseline_impl` lands in the boot report, so the run states its own
denominator. **Any score against a fallback must be reported as "versus that
fallback", never as beating the production kernel.**

Also added: one-chip **probe shape sets** for the three new tasks, each keeping
the axis the task exists for at production width and each with a
deliberately non-block-divisible holdout —

| task | scored | scored | holdout (traced + correctness, unscored) |
|---|---|---|---|
| ragged_paged_attention | `b=16, len=1024` | `b=8, len=512` | **`b=17`**, len=512 (prime batch) |
| megablox_gmm | `m=4096, g=4, k=4096, n=14336` | `m=2048, g=4` zipf | **`m=3000`**, g=4 zipf |
| rg_lru | `b=4, t=2048, d=2560` | `b=2, t=1024` | **`t=1500`** |

`k=4096`/`n=14336`, `d=2560`, and page_size 64 / 8 kv heads / 32 q heads / 128
head_dim are all unchanged. `g` drops 8→4 for GMM for one measured reason: the
worker holds `#scored × correctness_seeds` full input tuples live at once, and
`rhs` is 940 MB per fixture at g=8.

One more latent trap closed: all three problems' `adversarial_cases()`
hardcoded `case_by_name("tiny")` instead of the settable
`adversarial_case_name`, which silently forces candidates to trace at
CPU-battery shapes no prompt declares.

## 4. Control first — and it earned its keep three times

Mandatory before chips. Every seam's known-good fill goes through the
identical path a generation takes, wrapped in a model-style response **with a
decoy block first**:

```
fill -> extract_fill -> compose -> CPU AOT export (TPU platform, CPU host)
     -> [queue -> judge]
```

plus `--interpret`, which runs the composed program on CPU (`interpret=True`
for the Pallas ones) against each problem's own fp32 reference.

**Final state (job 3650852): interpret 8/8 OK, pre-gate 8/8 PASS.**

| seam | control fill | fill size | CPU numerics (max rel. err) | AOT export at all 3 declared shapes |
|---|---|---|---|---|
| flce | `flce-recompute` (carry=None) | 590 ch / ~147 tok | fwd **0.0**, grad **0.0** | PASS (1.4 s) |
| flce | `flce-saved-lse` (carry = LSE, closed-form bwd) | 690 ch / ~172 tok | fwd **0.0**, grad **0.0** | PASS (1.2 s) |
| splash_attention | `splash-highest-precision` | 1 604 ch / ~401 tok | **1.69e-07** | PASS (1.4 s) |
| splash_attention | `splash-default-precision` | 1 486 ch / ~371 tok | **1.69e-07** | PASS (1.4 s) |
| rg_lru | `rglru-sequential` | 443 ch / ~110 tok | **9.87e-07** | PASS (1.2 s) |
| rg_lru | `rglru-associative` | 549 ch / ~137 tok | **9.87e-07** | PASS (1.2 s) |
| megablox_gmm | `gmm-ktiled` | 469 ch / ~117 tok | **9.41e-08** | PASS (1.4 s) |
| ragged_paged_attention | `rpa-online-softmax` | 1 693 ch / ~423 tok | **1.13e-07** | PASS (1.5 s) |

Two fills per seam for FLCE, splash and rg_lru on purpose: they are *genuinely
different strategies* through the same seam (recompute vs saved-LSE;
`HIGHEST` vs default precision; sequential vs associative scan). The seam
admits more than one answer — which is the entire hypothesis.

### The three bugs the control caught before any chip

1. **The extractor carried a decoy.** Models routinely show a rejected sketch
   first. `extract_fill` concatenated every block that bound a required name,
   so a decoy `def tile_forward: raise NotImplementedError` rode along with the
   real answer. Fixed: prefer the **last single block that binds every**
   required name; concatenate only when no single block is complete. (Kept the
   multi-block path — models do answer one function per block, and
   `extract_program` would silently drop all but the last.)
2. **An RPA `BlockSpec` violated Mosaic's tiling rule.** `k_pages` is
   `[num_pages, page_size, kv_heads, head_dim]`, so a per-head block of
   `[1, ps, 1, d]` has a second-minor dim of 1 against an array dim of 8:
   `block shape are divisible by 8 and 128 respectively, or be equal to the
   respective dimensions of the overall array`. The kv-head axis came out of
   the grid; one step now holds all 8 heads, which is also 8× fewer grid steps.
   **This is exactly the class of failure the seam exists to absorb** — had it
   been in the prompt instead of the harness, every RPA candidate would have
   died at export and it would have read as "models cannot write paged
   attention".
3. A `SyntaxError` in the control's own `interpret=True` injection (a keyword
   inserted before the positional kernel body).

### CPU battery

`seam_dryrun.py`: **ALL GREEN** — prompts (both variants × five tasks declare
every probe shape, embed the scaffold verbatim, name every required fill, carry
the API block), extraction (six model-answer shapes incl. decoy and
unterminated fence), composition (harness `kernel` wins in all five), case sets
(`--problem` parses, every name resolves, none holdout-only), baselines resolve
on CPU for four of five (splash is TPU-only *by design*), the 33 API
assertions, and the metric.

The metric test encodes the exact mistake the last run made: a group that is
**non-uniform but entirely below the noise floor** (16/16 passing, spread
0.0042, floor 0.0158) must count as **non-uniform** and **not** as signal.
`overall_signal_groups == "1/3"` while `overall_nonuniform_groups == "2/3"`.

The full arena regression battery is **175 passed, 0 failed** (job 3650953,
5 min) — identical to the count before this work, with three baselines
rewritten, fifteen probe shape cases added and three `adversarial_cases()`
changed. Nothing about the production path moved.

## 5. The headline metric, restated in code

`metrics.group_uniformity` now takes the judge's own per-task noise floors —
the p95 of `|ref/ref − 1|` over counterbalanced pairs, straight from the boot
report — and reports:

```
signal_group_frac = (complete groups whose within-group reward spread > that task's noise floor)
                    / (complete groups)
```

`nonuniform_group_frac` is still emitted, next to it, so the two can be
compared directly and the gap is visible rather than assumed. `report.py` leads
with a **seam-vs-reference table per task**, carrying `max spread`, the floor,
`spread/floor`, and signal groups.

One property of the judge worth stating because it interacts with this metric:
`reward = 1.0` when `|score − 1| <= noise_floor`, else `score`. Candidates near
parity are snapped to exactly 1.0, which *suppresses* spread near the baseline.
A seam that lands many candidates near parity can therefore look flat for a
reason that is the reward shaping, not the prompt — worth separating when the
numbers land.

## 6. What is and is not known

**Known (CPU, this session):**

* The seam composes and is numerically correct for all five tasks, 8/8.
* Composed programs export at every declared shape including all five
  non-divisible holdouts, 8/8.
* The API block matches the installed jax on all 33 assertions.
* The extractor survives six realistic model-answer shapes.
* The headline metric returns the right answer on the case the previous run
  got wrong.
* Three tasks that could not have been served now boot-resolve their baselines.
* The arena regression battery is unchanged at 175 passed, 0 failed.

**Not known — needs the TPU run:** compile rate, pass rate, gate histogram,
reward distribution, best score, and **the headline** — per configuration and
per task, seam against reference. Specifically unanswered:

1. Does the seam preserve gradient signal, or collapse it like `tailored`?
   The design argues it should (block/tile choice drives VMEM pressure and
   speed, which are continuous and hard), but *argues* is not *measures*.
2. Does FLCE's seam still allow beating the baseline? Nothing anywhere has
   beaten our `custom_vjp` yet — best score to date **0.7008**, i.e. every
   passing candidate so far is *slower* than the thing it is trying to beat.
3. Which of the five is the best RL target.
4. Whether Qwen is measurable at all with thinking disabled. Last run it
   finished 3 of 80 generations, which is **unmeasured, not incapable**; the
   `<think>\n\n</think>\n\n` disable-thinking renderer is now wired and
   verified against the server's own template before samples are spent. If it
   still does not finish, it must be reported as unmeasured again.
5. Whether the splash-default-precision control lands inside the calibrated
   tolerance, and whether the RPA/GMM/rg_lru baselines behave on silicon.

**Resources: zero chip-hours. Zero queued resources created, in either zone,
at any point.** The launch script was never invoked; `gcloud` failed at the
read-only list step, before anything could be created. Because no QR was ever
created there is nothing to tear down — and I must be equally plain that with
credentials expired I could **not** independently list the zones to confirm
they are empty of `sk7524-seam-*`. The names have never been used. The running
RL sweep (`sk7524-tunix-qwen35-v5p32-dbtest-{d,e}`, `sk7524-league-*`,
`forever_sweep`) was not touched, read or otherwise.

Cost: six short CPU jobs on one neuronic node (16 cpu, 48–64 G), ~17 minutes
of wall time total.

## 7. How to finish it

After `gcloud auth login`, one command:

```bash
sbatch tpu/pallas_arena/probe/seam_probe.sbatch
```

Two spot QRs, distinctly named (`sk7524-seam-serve` v5p-16 us-east5-a,
`sk7524-seam-judge` v6e-1 us-east5-b), never more than two alive, deleted on
every exit path including untrapped SIGTERM, a 3 h cap from the first create, a
45-min landing rule with no fallback to a bigger slice, and both zones verified
empty of both names at the end. `TPU_PROCESS_BOUNDS=1,1,1` /
`TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` are set **inside the tmux env** on both
serving hosts, without which the two engines deadlock forming a two-host mesh.

The grid is 2 models × {`reference`, `seam`} × 5 tasks = **20 cells**, groups of
16, round-robin by round so every cell has the same number of complete groups
whenever the clock stops. `ROUNDS=3` with a 1 800 s draining tail, chosen
because the headline counts only **complete** groups and one v6e-1 grades every
survivor: finishing 2–3 rounds across all 20 cells beats starting 5 and
completing none.

One JSONL row per candidate — full generation text, extracted fill, composed
program, gate, observation, reward — at
`runs/pallas_arena/seam-results-<jobid>.jsonl`.

## 8. Example fill-ins

**Best-in-kind — the FLCE saved-LSE strategy** (172 tokens; the seam's `carry`
channel is what makes this expressible at all, and it skips recomputing the
logsumexp in the backward):

```python
TILE = 1024

def tile_forward(h_tile, w, t_tile):
    logits = _logits(h_tile, w)
    lse = jax.nn.logsumexp(logits, axis=-1)
    tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
    return tl - lse, lse                      # <- carry

def tile_backward(h_tile, w, t_tile, g_tile, carry):
    lse = carry
    logits = _logits(h_tile, w)
    probs = jnp.exp(logits - lse[:, None])
    onehot = (jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1) == t_tile[:, None])
    dlogits = (onehot.astype(jnp.float32) - probs) * g_tile[:, None]
    return dlogits @ w.astype(jnp.float32).T
```

**A representative failure — the RPA BlockSpec, before the fix.** Not a model's
failure: the harness's, caught by the control, and the reason the control is
mandatory:

```
GATE aot_export | ragged_paged_attention | ValueError
block shape are divisible by 8 and 128 respectively, or be equal to the
respective dimensions of the overall array
need: the SAME entrypoint must trace at EVERY declared shape (jax.export, no
concrete data)
```

## 9. Verdict

The seam is **built, controlled and ready**, and the judge can now grade all
five tasks instead of two — which was a genuine latent blocker, not
bookkeeping. Whether it does the one thing it was designed to do — keep
within-group reward spread above the judge's noise floor while lifting export
rate — is **not yet measured**, and this report does not claim it. The
apparatus is one authenticated command away from saying.
