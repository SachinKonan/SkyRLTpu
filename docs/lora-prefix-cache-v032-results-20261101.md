# LoRA with prefix caching on the v0.32 runtime: changes, tests, results

Date: 2026-11-01. Build tested: **bundle v56**.

## Summary

With prefix caching on, adapters work for Qwen3.5, Muse and Gemma on v5p-32, v6e-8 and the on-demand v4-64.
The strongest test ran on v5p and v6e. v4 passed the earlier test, whose n=8 check was weak.

| Fresh-farm result on v56 | v5p-32 | v6e-8 | OD v4-64 |
|---|---|---|---|
| Qwen3.5 | PASS, 78/78 | PASS, 78/78 | PASS, weak n=8 check |
| Muse | PASS, 78/78 | 77/78, known exception | PASS, weak n=8 check |
| Gemma | 76/78, test-bar failures | 76/78, test-bar failures | PASS, weak n=8 check |

Each failure is explained in [Exceptions](#exceptions-and-limits). None is a wrong token: in every cell, each n=8
sample was complete and either matched the cold answer or split from it at an exact tie.

**Not covered:** Qwen3.8 (the user ruled it out of this validation on 2026-11-01, Qwen3.5 stands for Qwen), spot v4-64,
and any production profile. Live `inference-farm-*` profiles are not on v0.32. See [Using this in inference](#using-this-in-inference).

## Build

| Item | Value |
|---|---|
| Bundle | `gs://sk7524-tinker-tpu-us-east5/code-bundles/tpuswarm-skyrl-rq4-base-v56.tar.gz` |
| Bundle sha256 | `41a1ba225b81ace5138708c33653750ff5cc8d5994361d6c3b131d4e15c12710` |
| Inference fork | `SachinKonan/tpu-inference`, branch `skyrl/v0.32.0-muse-gemma`, commit `41854da10`, on GitHub |
| Fork port record | `third_party/tpu-inference-v032/SKYRL_PORT_v0.32.md` |
| Scaffold | branch `agent/scaffold-exp` in `SkyRL-Main-infer-scaffold`; not on GitHub as of this document |
| Test profiles | `tpu/swarm/ray_train/profiles/agentfarm-*-v032-001.json` |

## Citations

Unless noted, paths below are relative to `/scratch/gpfs/ZHUANGL/sk7524/lcc-adapters/`:

- `results/`: LoRA, cache and speed runs on v56.
- `suite-archive/`: every earlier suite run, moved there from the session's temp folder.
- `tools/`: the test scripts, including `lora_cache_check.py`, `run_steps.sh`, `n8gap.py` and `resume_check.py`.

## Changes

### Inference fork

| Commit | Change | Why | Evidence |
|---|---|---|---|
| `7f7ffda42`, `d9be7fbcd` | LoRA on replicated K/V: tile adapter weights like the base weights; refuse untileable adapters before activation | Muse adapters at TP > 2 | `results/v6e-muse-n8fix/7_lcc.log` (TP8) |
| `4ecb7654b` | Native thinking-token budget (opt-in) | Budget enforced in the engine | `results/v6e-qwen-v56/3_tb.log` |
| `498722a9f` | Budget: trace the sampler body inside `sample_with_budget` | Every capped request crashed on v6e (nested jit options) | `results/v6e-qwen-v56/6_tb_mixed.log` |
| `1fdf87d69` | Clamp NaN token logprobs at source | Port from v0.23 | Port record |
| `34c55c6da`, `d409c606a` | Sliding-window KV for mixed head sizes, on by default where it applies | Port from v0.23 for Muse and Gemma | `results/v5p-muse-n8fix/7_lcc.log` |
| `966498714` | Conservative v4 RPA block sizes | Port of v0.23 fix | `results/v4-qwen-v56/suite.log` |
| `f75c0eb7f` | Restore LoRA metadata after a failed step trace | `UnexpectedTracerError` in warmup | `results/v4-qwen-v56/4_lora.log` |
| `0b45efb76`, `6a1ac456e` | v4 GDN: 32-token mixed tile, decode tile 1, 0.9 VMEM limit on chips with 16 MiB or less | Chip faults on v4 under load (`RuntimeUnexpectedCoreHalt`) | `suite-archive/v4-tune2-full/suite.log`, `suite-archive/v4-final/suite.log` |
| `15009c325`, `435b36d32`, `abbe6ff64`, `b2fec41fc` | Async scheduling: substitute only a one-token step at or after the in-flight position | Gemma assert under KV pressure | `suite-archive/v6e-gemma-burst/`, `suite-archive/v4-gemma-preempt/suite.log` |
| `41854da10` | Keep generated tokens of a request resumed from preemption | v0.32 resends it as new with history in prefill; the runner dropped that history, so recompute ran on wrong tokens | `results/v4-gemma-v56-a32/resume.log` |

Tried and reverted: a pure-JAX GDN kernel on v4 (`e55a5f4f0`, reverted by `b0c534f13`). It worked but ran 11-14% slower,
and the user preferred the smallest change from checked-in code.

### Scaffold

| Commit | Change |
|---|---|
| `0d7f928a` | Farms: selectable `serving_runtime`, `v0.23` or `v0.32` |
| `b43fc2d8` | Farm head: `x-session-id` keeps a session on one engine, for cache locality |
| `f8fddd8a`, `5be68f8a`, `154df898` | Pin the fork as submodule `third_party/tpu-inference-v032` |
| `457b388b` | Config refuses batched RPA with prefix caching on v0.32 (see below) |
| `e5a077ed`, `b21c19bf` | LoRA + cache test farms and the Gemma v0.23 speed baseline |

**Batched RPA:** with `USE_BATCHED_RPA_KERNEL=1` on v4, sample 2 of an n>1 request read the wrong context, at 3/3 on
full-prefill rows and 9/24 overall. With it off: 0/24 and `suite-archive/v4-muse-nobatched/suite.log` passes. The v0.23
fixes for this were not ported, so `tpu/swarm/ray_train/config.py:1351` refuses the combination.

## Test method

Tool: `tools/lora_cache_check.py`, run as step 7 of `tools/run_steps.sh` on a fresh farm for each model and machine.
Differences are the largest logprob change between two runs, counted for tokens with logprob >= -5, limit 0.25.

| Check | Question |
|---|---|
| `cold`, `floor` | Are cold requests truly cold, and do repeats match? This sets the noise floor |
| `base`, `base_hit` | With an adapter loaded, is the base model unchanged, cached or not? |
| `hit`, `extend`, `partial` | Does a full, extended or partial cache hit match cold with the adapter on? |
| `isolate`, `swap`, `swap_base`, `swap_iso`, `swap_new` | Do cached entries stay separate across adapters and swaps? |
| `mixed`, `solo`, `batch_hit` | Do mixed batches match solo requests, and did they hit the cache by the engine counters? |
| `n8`, `n8_hit` | Do all 8 samples return 32 tokens matching cold, and did they hit the cache by the counters? |
| `effect`, `swap_eff` | Does each adapter visibly change output? Bar: 1.0 |

The n=8 check was rewritten on 2026-11-01. The earlier version requested 1 token and could not detect an empty sample,
so it proved little about adapter n>1. `n8` uses temperature 1e-5 because n>1 is refused at 0.

## Results

### Strong test: v5p-32 and v6e-8

| Run | Checks ok | n=8 | Samples matching cold, of 16 | Split at an exact tie | Failures |
|---|---|---|---|---|---|
| `results/v5p-qwen-n8fix/7_lcc.log` | 78/78 | 18/18 | 10 | 6 | None |
| `results/v5p-muse-n8fix/7_lcc.log` | 78/78 | 18/18 | 16 | 0 | None |
| `results/v5p-gemma-n8fix/7_lcc.log` | 76/78 | 18/18 | 9 | 7 | `swap_eff` 0.985, 0.733 |
| `results/v6e-qwen-n8fix/7_lcc.log` | 78/78 | 18/18 | 7 | 9 | None |
| `results/v6e-muse-n8fix/7_lcc.log` | 77/78 | 18/18 | 16 | 0 | `hit` 0.364 |
| `results/v6e-gemma-n8fix/7_lcc.log` | 76/78 | 18/18 | 12 | 4 | `swap_eff` 0.985, 0.732 |

- Saved samples are in each run's `7_lcc.json`. The split counts come from `tools/n8gap.py`: at every split, the top two
  cold tokens have a gap of 0.000, and the sampled token is in the cold top 5.
- `n8_hit` passes 2/2 in all six runs. Example: Qwen v5p hit 29,056 of 29,160 queried tokens on the long prompt.
- Gemma ran with `gemma-a128.tar` as adapter A and `gemma-a64.tar` as adapter B.

### Earlier test: all nine cells

Same checks, without `cold`, `batch_hit`, `n8_hit` and `swap_eff`, and with the 1-token `n8`.

| Model | v5p-32 | v6e-8 | OD v4-64 |
|---|---|---|---|
| Qwen3.5 | `results/v5p-qwen-v56-lcc/7_lcc.log` PASS | `results/v6e-qwen-v56-lcc/7_lcc.log` PASS | `results/v4-qwen-v56-lcc/7_lcc.log` PASS |
| Muse | `results/v5p-muse-v56-lcc/7_lcc.log` PASS | `results/v6e-muse-v56-lcc/7_lcc.log` 1 failure | `results/v4-muse-v56-lcc3/7_lcc.log` PASS |
| Gemma | `results/v5p-gemma-v56-lcc/7_lcc.log` PASS | `results/v6e-gemma-v56-lcc/7_lcc.log` PASS | `results/v4-gemma-v56-lcc3/7_lcc.log` PASS |

### Superseded v4 runs

| Run | Result | Reason |
|---|---|---|
| `results/v4-muse-v56-lcc/7_lcc.log` | FAIL, 18 checks | Differences of 0.31-1.11 under the first all-token limit, the largest on tokens near logprob -20. The limit then moved to logprob >= -5 (`tools/lora_cache_check.py`, header note). Rerun on a fresh farm, same alpha, passed |
| `results/v4-muse-v56-lcc2/7_lcc.log` | INCONCLUSIVE | The farm already had an adapter, so `base` proves nothing |
| `results/v4-gemma-v56-lcc/7_lcc.log` | FAIL, 3 checks | Alpha-64 adapter missed `effect`; `short_prime` got 0 cache hits, not explained. Alpha 128 on a fresh farm passed |
| `results/v4-gemma-v56-lcc2/7_lcc.log` | INCONCLUSIVE | Same as Muse |

### General runtime suites

| Run | Result |
|---|---|
| `results/v4-qwen-v56/suite.log` | PASS: cache, n=8, thinking budget, LoRA, speed, mixed caps |
| `results/v5p-qwen-v56/suite.log` | PASS: steps 1, 2, 3, 4, 6 |
| `results/v6e-qwen-v56/suite.log` | PASS: steps 1, 2, 3, 4, 6 |
| `results/v4-muse-v56/suite.log`, `results/v4-gemma-v56/suite.log`, `results/v5p-gemma-v56/suite.log`, `results/v6e-muse-v56/suite.log`, `results/v6e-gemma-v56/suite.log` | PASS: steps 2, 6 |
| `results/v5p-muse-v56/suite.log` | FAIL on step 2's first-token check. `results/v5p-muse-v56/n8_tie.log` shows 3/12 requests flip, all at near-ties, largest top-2 gap 0.125 |

Qwen on v0.23 is the reason for this migration: cached samples were corrupted, while v0.32 matched cold. The raw v0.23
measurements were not preserved. The record is the session note `qwen-hybrid-prefix-cache-corruption.md`.

### Speed

Tokens per second.

| Setup | c16 | c64 | c256 | Burst | Source |
|---|---|---|---|---|---|
| Qwen3.5 v4, v0.32 | 237 | 744 | 1275 | 2769 | `results/v4-qwen-v56/suite.log` |
| Qwen3.5 v6e, v0.32, base | 468 | 1015 | 1070 | | `results/v6e-qwen-v56/ab.log` |
| Qwen3.5 v6e, v0.32, adapter | 468 | 1014 | 1070 | | `results/v6e-qwen-v56/ab.log` |
| Gemma v4, v0.23 | 271 | 819 | 838 | 1870 | `results/v4-gemma-v023-bench/suite.log` |
| Gemma v4, v0.32, base | 258 | 748 | 768 | 1614 | `results/v4-gemma-v56-lcc3/ab.log` |
| Gemma v4, v0.32, adapter | 255 | 753 | 771 | 1614 | `results/v4-gemma-v56-lcc3/ab.log` |

Adapters cost about 1% or less on Gemma v4 and Qwen v6e. The Qwen v6e burst is left out because the farm was preempted.
There is no Muse v0.23 baseline: `results/v4-muse-v023-bench/suite.log` shows only errors.

### Preemption and resume

Gemma on v4, 24 prompts, `results/v4-gemma-v56-a32/resume.log`:

| Run | Preemptions | Identical to a solo run |
|---|---|---|
| Control | 0 | 15/24 |
| Preempting | 216 | 13/24 |

The result is close to the control, not exact, so resumed output is not guaranteed identical. Three preemption bursts
finished with 0 errors at 1408-1441 tok/s (`suite-archive/v4-gemma-preempt/suite.log`).

## Exceptions and limits

| Item | Status | Evidence |
|---|---|---|
| Muse v6e `hit` 0.364 | Accepted by the user as a known exception. Same value on 3 fresh farms; the token has about 3% probability; each adapter part alone gives 0.000 | `results/v6e-muse-v56-lcc/7_lcc.log`, `results/v6e-muse-v56-lcc2/isolate.log`, `results/v6e-muse-n8fix/7_lcc.log` |
| Gemma `swap_eff` | Still marked failed. Alpha-64 adapter B moved output 0.73-0.99 against a 1.0 bar. The bar looks too high, but it was not changed after the result | `results/v5p-gemma-n8fix/7_lcc.log`, `results/v6e-gemma-n8fix/7_lcc.log` |
| v4 strong n=8 | Not run. No v4 worker was idle, last checked 20:06Z on 2026-11-01 | Weak check only: `results/v4-qwen-v56-lcc/7_lcc.log` |
| Spot v4-64 | Never tested. The user accepted OD v4 as the v4 validation | Profiles exist; no results |
| Gemma v4 speed | 8-14% slower than v0.23, cause unknown | Speed table |
| Cache reach | Prompts hit only when they repeat or extend a whole earlier prompt. A branch earlier gets 0 hits on all three models; this affects speed only, cause unproven | `results/kv/v6e_hit_repeat.log`, `tools/branch_probe.py` |
| Adapters | Test adapters are not all production adapters. Gemma needed alpha 128 for a clear signal | `tools/retar.py`, `results/v5p-gemma-n8fix/7_lcc.log` |

### Open code-review findings

These come from code reading on 2026-11-01. None is fixed in v56, and not all are confirmed. Fixing the fork ones needs a
new bundle and a retest on each machine type.

- **Fork:** silent-skip assert removed; a bad adapter or budget value can crash the engine step; uncompiled sampler stall;
  possible lost token under load.
- **Scaffold:** circuit metrics may be wrong (may affect s1 and s2); trainer can hang when leases are released; healthy
  farms can be marked dead.
- **Test tools:** token-promotion gap in the comparison; `resume_check.py` verdict; `bench_ab.py` lost-request counting.

## Using this in inference

No new bundle is needed: v56 already contains every code change above. Production profiles need these edits:

| Field | Value |
|---|---|
| `base_bundle`, `base_bundle_sha256` | v56 values above |
| `inference.serving_runtime` | `v0.32` |
| `inference.transformers_version` | `5.19.0` |
| `inference.ragged_conv1d` | `false` |
| `inference.batched_rpa_kernel` | `false`; the Muse preset defaults to true, and the build refuses it with caching |
| `cache.inference_compile` | A new path, so old compile caches are not reused |

Copy a production profile for that model and machine, then compare it with `agentfarm-*-v032-001`. Use a v0.32 layout
that was tested here: TP2 on v5p, TP4 on v4, and on v6e TP8 for Muse and Gemma or TP4 for Qwen.

## Reproduce

```bash
python -m tpu.swarm.ray_train.build tpu/swarm/ray_train/profiles/<profile>.json --output .rq4/builds/<run> --upload
sky jobs launch .rq4/builds/<run>/<run>.yaml --pool <pool>
STEPS=7 bash run_steps.sh <job> <farm> <bucket> <label>   # env: WORKER, MODEL, TAG, ADAPTER, ADAPTER_B
```

Run `sky` from della-vis2. Keep scripted output on scratch, because home is on a 50 GiB quota.
