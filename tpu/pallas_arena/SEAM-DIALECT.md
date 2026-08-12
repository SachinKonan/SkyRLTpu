# Seam + dialect: getting working code out of the three kernels that never produced any

*gemma-4-31B-it only. Generation + grading, no RL. Run: 2026-08-12.
Sibling of `PROMPT-ITERATION.md`, which established the brief for this run.*

**PLACEHOLDER — results section is filled in when job 3687904 lands.**

---

## 0. The brief, in one paragraph

The prompt ladder (480 candidates, `PROMPT-ITERATION.md`) left three kernels at
**0 PASS out of 96 candidates each**, at every rung: `splash_attention`,
`ragged_paged_attention`, `megablox_gmm`. Every one of those 288 candidates
died at `aot_export`, and the error histogram said why — they fail at the
`pallas_call` **plumbing** (grid, `BlockSpec`s, index maps), which no
whole-program rung supplies by design. Two things in the arena's history bear
on that:

* the **seam** (harness owns the `pallas_call`, model writes one strategy
  function) is the only variant that has ever exported splash at all
  (0/16 `reference` → 2/16 `seam`, job 3651278);
* the **P1 DIALECT list** is the only prompt addition ever measured to remove a
  failure class cleanly (`MemoryRef` item assignment 7→0, `` Invalid shape for
  `swap` `` 7→0, `def_fwd` 1→0, `no_code` 0/480) — and two of those are
  precisely the *seam's* own modal failures.

So this run is **seam + dialect**, not more prose.

## 1. The contaminated measurement, fixed first

17 of megablox_gmm's 96 ladder failures were

```
ValueError: Unsupported TPU device kind: cpu
```

which is not a statement about a kernel. Root cause, found by reading the 17
candidates: **every one of them wrote a rank-1 `BlockSpec`** for the per-row
expert-id array —

```python
pl.BlockSpec((bm,), lambda i, j: (i,))        # row_to_group / experts / rtg
```

— and Pallas's rank-1 block-shape check asks the *current device* for its
geometry:

```python
# jax/_src/pallas/mosaic/lowering.py
sublane_count = tpu_info.get_tpu_info().num_sublanes
lane_count    = tpu_info.get_tpu_info().num_lanes
```

`get_tpu_info()` resolves `get_device_kind()`, which under `JAX_PLATFORMS=cpu`
— the sandbox export child, on both the pre-gate host and the judge — is the
string `"cpu"`, and it raises before the check can run. GMM is the only one of
the five arena tasks whose natural formulation has a rank-1 operand, which is
exactly why only GMM hit it (verified: 17/17 device-kind failures in job
3687041 are `megablox_gmm`, 0 in the other four tasks).

**The fix** (`judge/child_runner.py:_tpu_export_device_shim`) pins the device
kind to the judge's own chip while exporting *for* tpu from a non-tpu child,
and does nothing at all on a TPU host. **The effect, measured on the exact 17
contaminated candidates** (`verify/verify_export_devkind.py`, job 3687873):

| | before | after |
|---|---|---|
| `Unsupported TPU device kind: cpu` | 17/17 | **0/17** |
| clear the export gate outright | 0/17 | **11/17** |
| fail the *real* rank-1 block-shape rule | — | 6/17 |

So GMM's ladder result was contaminated in the direction that mattered:
**11 candidates that should have reached the judge never did.** The remaining 6
now fail on a legible model error (`the first (and only) dimension of the block
shape …`) rather than on the harness's hostname.

## 2. What was built

| arm | what it is |
|---|---|
| `sd1` | the seam prompt + the P1 dialect list |
| `sd2` | `sd1` + a typed skeleton **of the fill functions, at the task's own rank** |

`sd2` deliberately ships **no worked example**. The ladder measured that one
backfires: P3's RMSNorm example removed splash's invented `pallas_call` kwargs
(10→0) and replaced them with **60 candidates writing a 2-D `BlockSpec` against
a 3-D operand**, because the example was over `[rows, cols]` and models copied
its *rank*. A skeleton of the task's own function cannot transfer a wrong rank.

What `sd2` still does not decide — because that is the `tailored` trap (16/16
PASS, spread 0.0042 against floors 0.0069/0.0158, zero gradient): block and
tile sizes, key/k chunking, which blocks or pages to skip, online vs two-pass
softmax, accumulation dtype, and the matmul precision trade.

**Token budget, and why the seam's API block was replaced rather than appended
to.** gemma is served at 16384. Over the ladder's 480 generations on these
exact tasks the median generation was ~8100 tokens and 12/480 hit `length`
against a 12000 budget, so a prompt that leaves under ~11000 tokens starts
destroying candidates for reasons that have nothing to do with kernels. The
seam prompt alone is already ~4750 tokens. So `API_BLOCK` (~1400 tokens) is
replaced by the one thing in it the dialect list does not already say — the
introspected list of the calls you make *inside* a kernel body, sliced verbatim
out of `API_BLOCK` itself — and the dialect's two bullets that cannot apply to
a fill (invented `pallas_call` kwargs; `custom_vjp.defvjp`) are condensed to a
line each. Bullets 2–9 and the precision-trade closer are spliced **byte-
identically**, and the splices are asserted at import.

Measured prompt sizes (gemma's own tokenizer reports 3.07 chars/token on this
family of prompts):

| task | `sd1` | room | `sd2` | room |
|---|---|---|---|---|
| splash_attention | 4890 | 11238 | 5444 | 10684 |
| ragged_paged_attention | 4950 | 11178 | 5591 | 10537 |
| megablox_gmm | 4523 | 11605 | 4885 | 11243 |
| *(control)* rg_lru `p1` | 2598 | — | — | — |

## 3. Controls, before any chip

Three earlier runs in this arena published confident all-zero grids caused by
the harness rather than the model, so nothing went to silicon until a known
answer had been through the identical path.

**CPU control, job 3687873** (`probe/seam_dialect_control.sbatch`):

* **prompts 10/10 clean** — every cell ends with the output contract, carries
  the dialect bullets verbatim, shows its scaffold, and leaves ≥10537 tokens
  for a completion;
* **device-kind fix 17/17 rescued**, §1;
* **known-good fills 6/6 interpret OK and 6/6 pre-gate PASS** through
  `model-style response (with a decoy block) → extract_fill → compose → CPU AOT
  export at all three declared shapes`. Max forward error: splash 1.69e-07,
  RPA 1.13e-07, GMM 9.41e-08, rg_lru 9.87e-07. Every extraction was
  `fenced-single` and every decoy was dropped.

**In-grid control cell**: `rg_lru | p1`, the ladder's best cell (25/32 PASS,
reward 1.0000, PASS-only spread 10.8× the floor). If it does not reproduce, the
harness is broken and no negative result in this run means anything.

**One aborted attempt with no chip cost.** Job 3687901 was cancelled 33 seconds
in, during the pre-warm, to resubmit under a shorter slurm time limit for
backfill. The pre-warm runs *before* any `queued-resources create`, so the
cancellation cost one minute and zero chips — which is exactly the property it
was added for after job 3686851 lost a 31-minute bring-up to a transient
GitHub fetch. Both zones were verified empty of `sk7524-sd-*` afterwards.

## 4. Results

*(to be filled in)*

## 5. Verdict

*(to be filled in)*

## 6. Resources and teardown

*(to be filled in)*
