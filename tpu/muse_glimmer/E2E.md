# Muse-Glimmer-30B text-only: end-to-end on TPU

Status of the *real weight* half of the port. The tiny-random-weight math proof
lives in `SPEC.md` / `parity_check.py`; the MaxText training half lives in
`MAXTEXT.md`. This file covers: where the real weights are, whether our JAX
serving model reproduces HF logits with those weights, and whether the model
actually serves and decodes on a TPU through vLLM + tpu-inference.

Read `SPEC.md` first for the architecture contract. Where this file and the HF
reference disagree, the reference wins.

§1-§4 are the original run and all passed. §5 is a follow-up investigation of
the hybrid KV-cache-group change, which was **reverted**; §6 records a
follow-up slice that never landed, so the on-slice half of §5 and the long
context / temperature / LoRA-training work are written but **unrun**. Anything
marked unrun is not evidence.

---

## 1. Weight staging (done, do not re-download)

`meta-models/Muse-Glimmer-30B`, snapshot `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.

| Location | Contents | Size |
|---|---|---|
| `gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da5.../` | 13 objects, both safetensors shards + tokenizer + config | **55.49 GiB** (59,581,829,216 B) |
| `/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b/` (neuronic, shared FS) | same snapshot, used by the CPU harnesses | 75 GB incl. `.cache` |

Shards: `model-00001-of-00002.safetensors` 49,950,112,952 B and
`model-00002-of-00002.safetensors` 9,603,322,320 B. Both present in GCS and on
the shared filesystem; **staging is complete, nothing needs re-pulling.**
`stage_weights.sbatch` is the (already-run) job that produced them.

Config as parsed (`config.json` → `text_config`): 52 layers, hidden 6656,
32 query heads / 2 KV heads, head_dim 128, intermediate 19968, vocab 202048,
`sliding_window` 2048, `tie_word_embeddings: false`,
`layer_types` = **39 `sliding_attention` + 13 `full_attention`**. Matches SPEC.

Sizing: 27.855 B params ≈ 51.9 GiB in bf16 → ~13 GiB/chip at TP=4, against
95 GiB/chip on v5p. Ample; v5p-8 (4 chips) is the right shape.

---

## 2. Real-weight teacher-forced parity — **PASSED**

`real_weight_check.sbatch` → job 3686671 (neu302, 32 CPU / 250 GB, ~22 min).
Two sequential processes so only one 30 B copy is resident at a time:

1. **HF reference** — `transformers 5.16.0.dev0`, `torch 2.13.0+cpu`, float32,
   `attn_implementation="eager"`. Dumps per-position logits, per-layer hidden
   states, top-8, and greedy continuations. Confirms `lm_head` is untied
   (`max|lm_head − embed| = 3.09`).
2. **Ours** — `tpu_inference/models/jax/muse_glimmer_core.py`, float32, fed
   straight from the same safetensors.

Prompt ids are shared between the two sides, so no tokenizer/BOS ambiguity can
masquerade as a model mismatch.

| prompt | tokens | logits max-abs | max-rel (>1% peak) | teacher-forced argmax | top-8 set |
|---|---|---|---|---|---|
| p1_tiny | 6 | 1.4305e-05 | 4.6748e-05 | **6/6** | 6/6 |
| p2_odd | 35 | 2.3596e-05 | 8.5949e-05 | **35/35** | 35/35 |
| p3_mid | 267 | 4.1485e-05 | 9.5221e-05 | **267/267** | 267/267 |
| p4_long_sliding | **3609** | 1.1892e-03 | 1.1782e-03 | **3609/3609** | 3609/3609 |
| p5_code | 32 | 2.9325e-05 | 1.2046e-04 | **32/32** | 32/32 |

Worst single layer anywhere: `max_abs` 3.815e-04 at layer 22 (p5) against a
per-layer peak activation of ~1.2e3 — i.e. ~3e-7 relative. Embedding max-abs
3.815e-06. `failures: []`.

**3949 of 3949 positions agree on argmax.** p4 at 3609 tokens is 1.76× the
2048 sliding window, so the 39 sliding layers and the 13 full-attention layers
are both genuinely exercised, including the sliding/full divergence that only
appears past the window. A mis-mapped or mis-transposed tensor would show up
here as a large error at one layer; nothing does.

Artefacts: `runs/muse_glimmer/hf_ref.npz` (129 MB), `hf_ref.json`,
`jax_vs_hf.json`, log `runs/muse_glimmer/real-3686671.log`.

---

## 3. CPU dry-run of the vLLM serving path

`vllm_dryrun.sbatch` — everything vLLM does *before* it touches a TPU, run on a
CPU node so integration bugs cost nothing. This exists because three earlier
runs in this repo burned their whole slice budget on failures that were fully
diagnosable without hardware.

Four environment traps it caught, all of which would have killed a landed slice:

1. **`uv` silently downgrades transformers.** The repo's `pyproject.toml` carries
   `[tool.uv] override-dependencies = [... "transformers>=5.6.1,<=5.8.0" ...]`
   for the live RL sweep. Run with cwd = repo root, `uv` applies that override
   even to an explicit `transformers @ git+…@main` **with `--no-deps
   --force-reinstall`**, yielding 5.8.0 — which cannot parse `model_type:
   muse_glimmer`. Fix: `UV_NO_CONFIG=1` (now set in both the dry-run and
   `e2e_tpu.sh`). Any resolving install *after* the git one re-triggers it, so
   transformers goes last.
2. **`torchax` / `qwix` / `jaxtyping` missing** → `tpu_inference.layers.vllm`
   fails to import → vLLM **swallows the plugin exception** → the architecture
   never registers → an unhelpful "architectures are not supported" much later.
   These ship inside the `vllm-tpu` wheel the TPU host uses, but not in the
   plain `vllm` wheel, so the CPU venv installs them explicitly.
3. **PP group not initialised** (`NameError: name '_PP' is not defined`).
4. **torch dtype reaching a JAX layer** (`Cannot interpret 'torch.bfloat16'`).

(3) and (4) are **harness-only** gaps, *not* model bugs — verified against the
real code path: `TPUWorker` calls `init_pp_distributed_environment(...)`
unconditionally (`need_pp=False` when PP=1), and `model_loader.get_model()`
rewrites `model_config.dtype` through `to_jax_dtype()` before constructing the
module and restores it after. Every JAX model in the tree (qwen2,
qwen3_moe, deepseek_v3) reads `get_pp_group()` in `__init__` exactly as
muse_glimmer does, so muse_glimmer follows the working convention. The dry-run
now reproduces both steps itself.

| stage | result |
|---|---|
| 1 transformers parses `model_type: muse_glimmer` | PASS |
| 2 `vllm.general_plugins` registers the arch with vLLM's **own** `ModelRegistry` | PASS |
| 3 `ModelConfig` constructs (where vLLM raises "not supported") | PASS |
| 4 derived shapes off `text_config` | PASS — hidden 6656, head 128, vocab 202048, 52 layers, 2 KV heads, 32 heads, window 2048 |
| 5 tpu-inference resolves the arch to the JAX class | PASS |
| 6 nnx module builds + real checkpoint loads through the **serving** loader | see below |

Stage 2 matters most: the checkpoint advertises
`MuseGlimmerForConditionalGeneration`, vLLM resolves architectures against its
own registry in `ModelConfig.__post_init__` before any tpu-inference code runs,
and `oot_registration.py` registers a text-generation shim there so the JAX
implementation can take over under `MODEL_IMPL_TYPE=flax_nnx`.

### Stage 6 — the serving loader against the real checkpoint

Job 3686864, TP=4 on 4 forced CPU devices, layers truncated 52 → 4 (keeping the
`[S,S,S,F]` pattern so both attention flavours are built) but every tensor at
**full width**. 809 vision tensors skipped, 51 text tensors kept. Every
parameter the module consumed was compared against the source tensor:

```
embed_tokens          max_abs_err=0.000e+00  (verbatim [V,D])
lm_head               max_abs_err=0.000e+00  (transposed [D,V])
model.norm            max_abs_err=0.000e+00  (verbatim [D])
L*.{input,post_attention,pre_feedforward,post_feedforward}_layernorm
                      max_abs_err=0.000e+00  (verbatim [D])
L*.q_proj             max_abs_err=0.000e+00  ([NH*H,D] -> [D,NH,H])
L*.o_proj             max_abs_err=0.000e+00  ([D,NH*H] -> [NH,H,D])
L*.k_proj / v_proj    max_abs_err=0.000e+00  ([KV*H,D] -> [D,KV,H] kv_repeat=2)
L*.attn_gate_proj     max_abs_err=0.000e+00  (self_attn.gate_proj, transposed)
L*.mlp.{gate,up,down}_proj
                      max_abs_err=0.000e+00  (transposed)
```

**Bit-exact on every tensor.** This is the check the tiny-random-weight parity
could not do: it exercises the reshapes and transposes at real head counts, and
in particular `kv_repeat=2` — Muse-Glimmer has `num_key_value_heads=2`, so at
TP=4 the module pads the KV axis to 4 and `_load_kv_proj_replicated` replicates
element-wise (`h0 h0 h1 h1`, `repeat_interleave`, **not** `tile`) so device *i*'s
KV head matches the query heads sharded onto it. Getting `tile` vs
`repeat_interleave` backwards would be invisible on square tiny weights and
catastrophic here.

Two notes on running it: stage 6 needs `--mem` well above the checkpoint size
unless tensors are read lazily (`load_file` on the 46.5 GiB shard got the job
OOM-killed at 96 GB — it now uses `safe_open` + `get_tensor`), and it must force
a platform because the CPU-built vLLM wheel activates none.

---

## 4. End-to-end on a spot v5p-8 — **SERVED AND DECODED**

`e2e_tpu.sbatch` → slurm 3686812 → `e2e_tpu.sh`. One spot v5p-8 QR
(`sk7524-museglimmer-e2e`, us-east5-a), served through vLLM 0.23.0 +
tpu-inference under `MODEL_IMPL_TYPE=flax_nnx`, `TPU_BACKEND_TYPE=jax`, TP=4,
`max-model-len` 4096, `max-num-seqs` 16.

Timeline from QR creation (`19:58:55Z`):

| t+ | event |
|---|---|
| 15m22s | QR **ACTIVE** (922 s waiting for spot capacity) |
| 15m40s | ssh up; host provisioned (uv 0.12.3, gcsfuse mounted) |
| 17m51s | 55.5 GiB of weights on local SSD — `gcloud storage rsync`, **1m20s** |
| 24m13s | venv built; **ARCH-REGISTERED-OK**, transformers 5.16.0.dev0, vllm 0.23.0 |
| 27m43s | **ENGINE LIVE** — answered a real completion, not just `/health` |
| 29m00s | all phases done, teardown |

The slice reports 4 chips (`/dev/vfio/{0,1,2,3}`; note `ls /dev/accel*` returns
0 on this runtime, so that probe is not a liveness signal).

### Memory

| quantity | value |
|---|---|
| HBM per chip, before load | 0.00 / 95.74 GiB |
| **HBM per chip, model resident** | **13.06 / 95.74 GiB** (×4 chips) |
| total HBM used (weights) | 52.22 GiB — vs 51.9 GiB predicted for 27.855 B bf16 |
| total HBM limit / cap | 382.97 / 352.33 GiB |
| KV cache | **3,025,664 tokens**, 11 819 blocks × 52 layers |
| max concurrency @4096 tokens | **738.69×** |

KV layout came out as `num_kv_cache_groups=1` — vLLM allocated all 52 layers
uniformly rather than giving the 39 sliding layers a smaller group. Correct, but
it leaves memory on the table; a hybrid grouping would buy more KV for the same
HBM. **§5 investigates exactly that and concludes it cannot be switched on
without a change to `muse_glimmer.py` first.**

### Greedy decode vs the HF reference, token for token

`mg_client.py` submits prompts as explicit `list[int]` and reads generated ids
back via `return_token_ids`, so nothing round-trips through text and no
tokenizer/BOS discrepancy can be mistaken for a model mismatch. `ignore_eos` is
set and the HF reference hit its token budget on every prompt, so the two sides
compare like for like.

| prompt | prompt tok | gen | result | wall |
|---|---|---|---|---|
| p1_tiny | 6 | 32 | **EXACT** | 0.4 s |
| p2_odd | 35 | 48 | **EXACT** | 12.9 s |
| p3_mid | 267 | 48 | diverges at 13 — **exact tie**, see below | 15.1 s |
| p4_long_sliding | **3609** | 32 | **EXACT** — beyond the 2048 window | 14.2 s |
| p5_code | 32 | 32 | **EXACT** | 12.8 s |

**4/5 token-exact; the fifth is a genuine argmax tie.** At step 13 of `p3_mid`:

```
HF token   8535   logprob -1.4882723093032837
TPU token  6050   logprob -1.4882723093032837   <- identical
tpu_top1_top2_gap = 0.0
hf_token_rank_on_tpu = 1        (i.e. tied for first, ordered second)
```

The two candidates have **bit-identical logprobs**; the top-1/top-2 gap is
exactly 0.0. Both implementations are computing the same distribution and simply
break the tie differently (a stable-sort/index-order artefact). This is not a
numerical disagreement — there is no gap to disagree about. Greedy decoding is
chaotic after a tie, so the remainder of that continuation is unrelated to the
reference and the comparison says nothing further about it.

Prompt lengths 6, 35, 267 and 3609 are all non-block-divisible (p5's 32 is the
divisible control), and p4 at 3609 tokens is 1.76× the sliding window, so both
the 39 sliding and the 13 full-attention layers are exercised in the served
path, not just in the CPU core.

### Paged KV with concurrent requests, and throughput

| phase | result |
|---|---|
| 5 prompts fired **concurrently** | **5/5 token-identical to the sequential run**; 192 tokens in 13.6 s |
| throughput, 8 concurrent × 64 tokens | 512 tokens in **1.0 s = 513.5 tok/s** |
| same-prompt concurrent determinism | all 8 outputs identical, and identical to the batch-1 run |
| `prompt_logprobs` (known EngineCore killer) | survived, 6 positions |

The concurrency result is the one that matters for the paged KV cache: five
live sequences of very different lengths (6 … 3609 tokens) sharing the block
pool produce byte-identical ids to the one-at-a-time run, so blocks are not
being cross-contaminated between sequences. The 8-way identical-prompt run adds
that same-input sequences stay bit-identical when batched.

513.5 tok/s is a small fixed batch at `max-num-seqs 16` with a 6-token prompt,
not a benchmark — treat it as a floor.

### Cost

QR created `19:58:55Z`, verified gone `20:29:59Z` — **31m04s** alive on 4 chips
= **2.07 chip-hours**, one spot v5p-8, single attempt, no re-runs. Of that, 15m22s
was waiting for spot capacity and ~12m was one-time setup (weights, venv, model
load); the measurements themselves took ~80 s.

Teardown: `CLEANUP: QR sk7524-museglimmer-e2e is GONE (verified)`. The delete
needed two re-issues — the QR still described as `ACTIVE` for ~2 min after the
first delete request, which is exactly why the cleanup loop re-issues and
re-verifies rather than trusting one call.


---

## 5. The hybrid KV-cache-group change — **REVERTED, and here is why**

`tpu_inference/runner/kv_cache_manager.py::get_kv_cache_spec` hardcodes
`sliding_window = None` on the JAX path, so all 52 layers get a
`FullAttentionSpec` and every layer is sized for the full context (the
`num_kv_cache_groups=1` in §4). A candidate edit resolved the global and
sliding KV geometries and emitted a `SlidingWindowSpec` for sliding layers
whenever the two agree — which for Muse-Glimmer they do
(`num_global_key_value_heads` and `global_head_dim` are both absent, so both
sides fall back to `num_key_value_heads=2` / `head_dim=128`).

**The edit has been reverted (`git checkout` in the submodule); the tree is
back to the 1-group behaviour §4 measured.** It is not a bug in the edit's own
logic — the gate fires precisely where it was meant to. It is that the served
model cannot consume the layout the edit produces, and that was established
before any slice time was spent on it.

### 5.1 "Re-enable" in the TODO is a misnomer — verified

The `TODO(kwang3939): Re-enable sliding_window ...` reads as if this path once
worked. It did not. `git show 9a5dfeca2 -- tpu_inference/runner/kv_cache_manager.py`
(*[JAX] Enable using different dims across layers in kv_cache_manager (#1860)*)
shows the JAX branch **before** that commit calling

```
kv_cache_spec[f"layer.{i}"] = self._create_attention_spec(
-       block_size, num_kv_heads, head_size)
```

with no `sliding_window` argument at all. `sliding_window = None` and the TODO
arrived together, in the same commit that introduced per-layer dims. So
`SlidingWindowSpec` has **never** been exercised on the JAX path, and this is an
untested path rather than a restored one.

### 5.2 What the change actually does — CPU A/B, no slice needed

`kv_spec_probe.py` calls the real `KVCacheManager.get_kv_cache_spec` against a
duck-typed runner (everything it touches is config, a mesh and a dtype) and
hands the result to vLLM 0.23's own `get_kv_cache_groups` +
`get_kv_cache_config_from_groups` — the functions that decide
`num_kv_cache_groups`. Two copies of the tpu-inference tree, patched and
`git show HEAD:` baseline, each first on `PYTHONPATH`. TP=4, block_size 256,
300 GiB assumed available. `kvspec_probe.sbatch` job 3687246, `g4spec` job 3687309.

| config | build | layer specs | groups | blocks | **KV tokens** | **`len(kv_cache_tensors)`** |
|---|---|---|---|---|---|---|
| Muse-Glimmer-30B (real `config.json`) | baseline | 52 × Full | **1** | 23 630 | 6 049 280 | **52** |
| Muse-Glimmer-30B (real `config.json`) | **patched** | **39 × Sliding(2048) + 13 × Full** | **4** | 94 523 | **24 197 888** | **13** |
| gemma-4-31b-it (real `config.json`) | baseline | 60 × Full | 1 | 1 396 | 357 376 | 60 |
| gemma-4-31b-it (real `config.json`) | **patched** | 60 × Full | 1 | 1 396 | 357 376 | 60 |
| gemma-4 E2B-shaped (global dims `None`) | baseline | 30 × Full | 1 | 10 240 | 2 621 440 | 30 |
| gemma-4 E2B-shaped (global dims `None`) | **patched** | 25 × Sliding(512) + 5 × Full | **6** | 61 440 | 15 728 640 | **5** |

The synthetic and the real-`config.json` runs for Muse-Glimmer agree exactly.
**The prize is real: 4.00× the KV token capacity for the same memory.**

### 5.3 Why it cannot be switched on yet

vLLM does not produce *two* groups (sliding + full). Its
`_get_kv_cache_groups_uniform_page_size` groups by the repeating layer pattern —
its own docstring: *"A model with 10 full attention layers and 20 sliding window
attention layers. There are 3 layers in the pattern (1 \* full, 2 \* sw), so there
are 3 kv_cache_groups"*. Muse-Glimmer's pattern is `[S,S,S,F] × 13`, so it gets
**4 groups of 13 layers**, and one `KVCacheTensor` per group-slot: **13 arrays,
not 52**.

`tpu_inference/models/jax/muse_glimmer.py:601` is written against the 1-group
layout:

```python
kv_caches[i], x = layer(kv_caches[i], x, attention_metadata)
```

It indexes `kv_caches` by **absolute layer index over all 52 layers** and passes
a single `attention_metadata` object down. Under the patched spec it would be
handed 13 arrays (`IndexError` at `i = 13`) and a per-layer-name *dict* of
metadata. `gemma4.py` — the structural template — already handles both
(`gemma4.py:1007` `attention_metadata[layer_name]`, `:1012-1016`
`cache_idx = layer_name_to_kv_cache[layer_name]`), which is why gemma-4 E2B/E4B
would survive the same spec change and Muse-Glimmer would not.

So the failure mode is a hard one at model-construction/forward time, not the
subtle silent-corruption-past-position-2048 that the two-independent-windowing-
mechanisms argument predicts. That argument still stands as the *second* hazard
(the ragged-paged-attention kernel takes its window from the model config at
`tpu_runner.py:1001`, while block retention would now come from the KV spec) —
it simply is not the one that bites first.

### 5.4 gemma-4 regression verdict — **no regression on the production path**

`tpu/runs/gemma4-31b.env` runs `google/gemma-4-31B-it`, whose real config has

```
head_dim 256   global_head_dim 512
num_key_value_heads 16   num_global_key_value_heads 4
```

Both global attributes are **set and different**, so `uniform_kv_dims` is false
and the gate never fires. The probe confirms it end to end: the patched and
baseline JSON outputs for the real gemma-4-31b config are **byte-identical**
(`GEMMA4-31B-SPEC-IDENTICAL`, 60 × `FullAttentionSpec`, 1 group, 60 tensors).
That is a stronger statement than a serving smoke could make — a smoke samples
behaviour, spec-equality is exhaustive — and it is the reason no gemma-4
serving smoke was run: with an identical KV cache spec there is nothing for a
smoke to distinguish. (It is also not cheap to run: no gemma-4 weights are
staged in this project's GCS cache; the only local copy is the 78 GB
`gemma-4-31b-it` snapshot on the shared filesystem, which does not fit on a
v5p-8 boot disk next to the 55.5 GiB Muse-Glimmer checkpoint.)

Gemma-4 **E2B/E4B** *is* affected — `num_global_key_value_heads` is `None`
there, which is the exact case the existing regression test
`test_get_kv_cache_spec_without_compilation_cfg_none_text_config_attrs` was
written for. Those variants go from 1 group to 6. They would probably survive it
(gemma4.py is group-aware), but nothing here demonstrates that, and no E2B/E4B
weights are staged.

The repo's own `tests/runner/test_kv_cache_manager.py` gives **no** differential
signal: all 34 tests error at *setup* in **both** builds with
`RuntimeError: Failed to infer device type` — vLLM cannot pick a platform on a
CPU-only node, which is exactly the artefact `vllm_dryrun.py::force_cpu_platform`
exists to work around and which the probe therefore does not hit.

### 5.5 What landing this properly would take

1. Make `muse_glimmer.py` group-aware, following `gemma4.py` verbatim: accept
   `_layer_name_to_kv_cache`, index `kv_caches[cache_idx]`, and unwrap
   `attention_metadata[layer_name]` when it is a dict. ~15 lines.
2. Only then re-apply the `uniform_kv_dims` gate — and consider narrowing it to
   models known to be group-aware rather than widening it by config shape alone,
   since the gate as written silently changes the KV layout of every JAX model
   whose global head dims happen to be absent.
3. Re-run the on-slice verification below, which is written and unrun.

### 5.6 What was NOT verified — this is not a pass

**Nothing in §5 was measured on a TPU.** The slice never landed (§6). Missing,
and required before any claim about correctness:

- the runner's own `Hybrid KV cache layout: num_kv_cache_groups=%d` line
  (`kv_cache_manager.py` ~line 913) from a live engine;
- KV token capacity and per-chip HBM against the §4 baseline of **3 025 664
  tokens** and **13.06 / 95.74 GiB**;
- the window-boundary sweep, and the shortest prefix length at which any token
  diverges.

The harness for all of it is written and ready: `mg_client2.py --phases
boundary` fires a one-token greedy request from prefixes of the 3609-token
`p4_long_sliding` prompt at **27 lengths** — 16, 64, 512, 1024, 1536, 2016,
2032, 2040, 2044, 2046, 2047, **2048**, 2049, 2050, 2052, 2056, 2064, 2080,
2100, 2176, 2304, 2560, 3072, 3456, 3584, 3608, 3609 — and compares each to
HF's teacher-forced argmax at position `L-1`. Because attention is causal, that
argmax *is* the greedy next token for the prefix of length `L`, so the reference
is exact and costs no new HF compute; `make_ext_ref.py` extracted it from the
existing `hf_ref.npz` (`mg_ext_ref.json`, 27 probes, **zero near-ties** — every
probed position has a top1/top2 gap above 1e-3, so any disagreement would be a
real one and not a tie artefact like `p3_mid` step 13). The sweep is dense
across 2048 precisely so "first diverges at 2049" (indexing off-by-one) is
distinguishable from "first diverges at 3600" (eviction).

---

## 6. Follow-up run (items 2-4): **no slice, 0 chip-hours**

`followup_tpu.sbatch` → slurm 3687240 → `followup_tpu.sh`. One spot v5p-8 QR,
`sk7524-museglimmer-followup`, us-east5-a.

| t+ | event |
|---|---|
| 0 | QR created `21:19:48Z` |
| 0-45m | `WAITING_FOR_RESOURCES`, continuously, never ACTIVE |
| 45m03s | `NOT LANDED after 2703s (cap 2700s) -- giving up` |
| 45m29s | `CLEANUP: QR ... is GONE (verified)` |

No chips were ever allocated, so the run cost **0 chip-hours**. For contrast the
§4 run landed in 15m22s; spot v5p-8 capacity in us-east5-a was simply not
available in this window. Teardown was checked three ways: the script's own
re-issue-and-verify loop, the sbatch wrapper's backstop, and an independent
listing at `22:06:10Z` — **zero muse QRs and zero muse TPU VMs in us-east5-a,
us-east5-b and us-east5-c.**

Items 2 (context past 4096), 3 (temperature > 0) and the on-slice half of item 4
(LoRA training smoke) are therefore **not run**. What exists is the harness and
the offline prerequisites:

- **`followup_tpu.sh`** — one QR, deleted on every exit path (the §4 lifecycle
  verbatim). Boots vLLM patched and reverted at `max-model-len` 4096 for the
  item-1 A/B, then walks 8192 → 16384 → 32768 reporting where it stops, then
  frees the boot disk and runs the LoRA smoke. Every stage is independently
  guarded, and the CPU-only training prep runs in the background during the
  serving stages.
- **`mg_client2.py`** — boundary sweep, decode-crossing, long context (with a
  reference-free needle-recall fallback), and the temperature phases:
  fixed-seed reproducibility including *concurrent* same-seed requests,
  seed sensitivity, degeneration statistics, and the load-bearing one — the
  returned logprobs against HF's own distribution. `mg_logit_rows.npz` holds
  the full float32 final-position logit row for all five §4 prompts, so the
  comparison can be made at any temperature against both `log_softmax(logits)`
  and `log_softmax(logits / T)`. That is the check greedy structurally cannot
  do: argmax is invariant to the `output_multiplier` (0.196) and to the
  `T*tanh(logits/T)` softcap, and both reshape the sampled distribution.
- **`hf_ext_ref.py`** — HF references for decode *across* the window and for 8k
  / 16k teacher-forced argmax. Submitted (job 3687244) and cancelled: reading
  the 30B off the shared filesystem in float32 was at 29% after 31 minutes, far
  past the point where it could serve a slice that had already failed to land.
- **Real-weight MaxText/orbax checkpoint — DONE.** `convert_real.sbatch` job
  3687234 ran the exact converter argv the tunix backend shells out to, on the
  real 55.5 GiB checkpoint, and pushed the result to
  `gs://sk7524-tinker-tpu-us-east5/muse-glimmer/maxtext-orbax/0/items`:
  **43 577 641 331 B (40.58 GiB)**, uploaded in 2m18s at 824.5 MiB/s. This
  closes `MAXTEXT.md` §5 item 2 (real-weight conversion, previously only proven
  on tiny weights) and removes the largest obstacle to the LoRA smoke: the
  v5p-8 boot disk is 97 GB and the HF checkpoint already occupies 55.5 GiB of
  it, so converting on the slice was never going to fit. The slice now only has
  to download 40.58 GiB.
- **`lora_smoke.py`** — asserts a **non-zero** LoRA adapter count and prints the
  module paths the adapters landed on (a name mismatch yields zero adapters
  silently; MaxText's attention attribute is `self_attention` for exactly this
  reason), asserts the base params are really FSDP-sharded across the 4 chips
  rather than replicated, and takes 3 optimizer steps on a math-RL-shaped batch
  checking that grad_norm is finite and non-zero and that the loss moves.

## 7. What is still unproven

Deliberately out of scope, or simply not yet exercised. None of it is blocked by
anything above; it is just not evidence we have.

- **Vision.** This is a text-only port by construction. The loader drops vision
  weights and `get_multimodal_embeddings` rejects multimodal input with a clear
  error. `vision_config`, `image_token_id`, `video_token_id` and the projector
  in `config.json` are untouched. The checkpoint's `...ForConditionalGeneration`
  arch string is honoured only as a name.
- **Context beyond `max_model_len=4096`.** The model advertises 131072. The
  longest thing tested end-to-end is 3609 prompt tokens. Everything past the
  2048 window is exercised, so the sliding/full split is covered, but RoPE at
  tens of thousands of positions is not. Harness written (`followup_tpu.sh`
  walks 8192 → 16384 → 32768; `mg_client2.py --phases longctx,longfallback`
  plants a needle in the first ~30 tokens and asks for it at the end, so the 13
  NoPE full-attention layers carry real long-range load), **unrun — no slice.**
- **Sampling.** Only greedy (`temperature=0`) is compared. Nothing here says the
  sampled distribution matches HF, only that the argmax path does. Harness
  written (`mg_client2.py --phases temp`, including the logprob-vs-HF check at
  matched temperature), **unrun — no slice.**
- **The hybrid KV-cache-group change.** Investigated on CPU and reverted; see
  §5. The spec-level A/B is solid, the on-slice confirmation is missing, and
  §5.3 says why the served model cannot consume the new layout as it stands.
- **Quantization.** Served bf16 unquantized; no qwix/AWQ/FP8 path tried.
- **Topology.** One host, TP=4, PP=1, DP=1. KV-head replication
  (`_load_kv_proj_replicated`) is specifically a TP-vs-`num_key_value_heads=2`
  concern and is only proven at TP=4; TP=8 and multi-host are untested.
- **Training on real weights.** `MAXTEXT.md` covers the converter round-trip and
  HF parity. The real 30B checkpoint has now been converted to orbax and staged
  in GCS (§6), but **no real-weight optimizer step has been run** and no LoRA
  adapter has been injected into the real model — that is the half of item 4
  that needs a slice.
- **`prompt_logprobs`.** Probed once, as a known EngineCore killer on this stack
  (see the tinker/qwen35 notes). It is not part of the pass criteria.
- **Sustained serving.** No long-run stability, preemption/eviction-under-
  pressure, or chat-template exercise; the throughput number below is a small
  fixed batch, not a benchmark.

---

## Reproducing

```bash
# 1. real-weight teacher-forced parity (CPU, ~25 min)
sbatch tpu/muse_glimmer/real_weight_check.sbatch

# 2. small ids-only reference for the TPU client (a few KB, cheap to ship)
/n/fs/vision-mix/sk7524/caches/muse-parity/venv/bin/python \
    tpu/muse_glimmer/make_hf_greedy.py \
    --npz runs/muse_glimmer/hf_ref.npz --out runs/muse_glimmer/hf_greedy.json

# 3. CPU dry-run of the vLLM path (no TPU spend)
sbatch tpu/muse_glimmer/vllm_dryrun.sbatch

# 4. end-to-end on ONE spot v5p-8
sbatch tpu/muse_glimmer/e2e_tpu.sbatch

# 5. KV-cache-group A/B on CPU (patched vs `git show HEAD:` baseline), no slice
sbatch tpu/muse_glimmer/kv_spec_probe.sbatch      # muse-glimmer + gemma-4 shapes
sbatch tpu/muse_glimmer/gemma4_kvspec.sbatch      # the REAL gemma-4-31b config

# 6. real-weight HF -> MaxText/orbax, pushed to GCS (~25 min CPU + 2 min upload)
sbatch tpu/muse_glimmer/convert_real.sbatch

# 7. extra HF references: decode across the 2048 window, 8k/16k teacher forcing
sbatch tpu/muse_glimmer/hf_ext_ref.sbatch

# 8. the follow-up slice: KV A/B, long context, temperature, LoRA smoke
sbatch tpu/muse_glimmer/followup_tpu.sbatch
```

`hf_ext_ref.sbatch` runs `make_ext_ref.py` first — that part needs no model and
finishes in seconds, producing `mg_ext_ref.json` (the 27 window-boundary probes)
and `mg_logit_rows.npz` (the HF distributions for the temperature check), so the
slice is not blocked on the slow float32 forward behind it.

The HF reference needs the unreleased transformers and therefore an isolated
venv (`/n/fs/vision-mix/sk7524/caches/muse-parity/venv`, built by
`build_parity_venvs.sbatch`); never install it into the project venv, which is
pinned for the live RL sweep.

### QR hygiene

`followup_tpu.sh` uses a distinct name, `sk7524-museglimmer-followup`, and the
same lifecycle; its wrapper additionally audits **all three** zones
(us-east5-a/b/c) for both queued resources and TPU VMs at exit.

`e2e_tpu.sh` creates **exactly one** QR, `sk7524-museglimmer-e2e`, and deletes
it on every exit path. `trap ... EXIT` is known not to fire on an untrapped
SIGTERM here, and a `scancel` has previously killed a cleanup mid-delete and
left a QR alive, so: TERM/INT/HUP are trapped explicitly, the delete is issued
`setsid nohup` so it outlives the shell, and the script re-issues and verifies
until the QR is gone. `e2e_tpu.sbatch` adds `--signal=TERM@300` (deliberately
without `B:`, so the signal reaches the script and not just the wrapper) plus a
backstop delete of its own, and prints the remaining muse QRs in both zones
into the job log so the result is auditable without re-running anything.
