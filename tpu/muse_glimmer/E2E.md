# Muse-Glimmer-30B text-only: end-to-end on TPU

Status of the *real weight* half of the port. The tiny-random-weight math proof
lives in `SPEC.md` / `parity_check.py`; the MaxText training half lives in
`MAXTEXT.md`. This file covers: where the real weights are, whether our JAX
serving model reproduces HF logits with those weights, and whether the model
actually serves and decodes on a TPU through vLLM + tpu-inference.

Read `SPEC.md` first for the architecture contract. Where this file and the HF
reference disagree, the reference wins.

§1-§4 are the original run and all passed. §5 is the hybrid KV-cache-group
change: **measured on hardware and reverted** — it applies correctly, quadruples
the block count, is byte-exact at batch 1, and then corrupts one sequence out of
five under concurrency. §6 is the follow-up slice that carried §5's A/B plus
long context (**32768 serves; needle recall at 16 320 tokens**), temperature,
and the LoRA smoke (**56 adapter arrays, loss moves, FSDP applied**). §7 is the
TP head-to-head (**capacity 1.66× to 2×TP=2, single-stream latency 1.73× to
TP=4**) and carries the recovered temperature measurement. Anything marked
unrun is not evidence.

Total cost of the two slices in §6 and §7: **4.57 chip-hours** on spot v5p-8
(2485 s + 1625 s active, 4 chips each), plus one earlier attempt that never
landed and cost nothing.

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
it looks like it leaves memory on the table. **§5 builds the hybrid grouping,
measures it on hardware, and reverts it**: at 4096 it is a small capacity
regression, and it corrupts one sequence in five under concurrency.

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
logic — the gate fires precisely where it was meant to, and with a group-aware
`muse_glimmer.py` (§5.3) the engine boots, allocates 4 groups, and is
**byte-exact against HF at batch 1 across all 27 window-boundary probes**. It
was reverted because at batch 5 it silently changes one sequence's output
(§5.6). The rest of §5 is what that cost and what it bought.

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

### 5.3 The second half of the change: making the model group-aware

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

So `muse_glimmer.py` was given the same treatment as `gemma4.py`, verbatim: the
forward loop accepts `layer_name_to_kv_cache`, resolves `cache_idx` through it,
unwraps `attention_metadata[layer_name]` when the metadata arrives as a dict,
and falls back to the flat 1-group form when either is absent. The caller
already had a `_layer_name_to_kv_cache` parameter in its signature and was
silently dropping it; it now forwards it. ~25 lines.

**That part works.** With both edits in place the engine boots, allocates the
4 groups, and serves — no `IndexError`, no metadata type error. The hard
construction-time failure predicted here was real and was fixed. What was left
was the *second* hazard named above — two independent windowing mechanisms (the
ragged-paged-attention kernel takes its window from the model config at
`tpu_runner.py:1001`, while block retention now comes from the KV spec) — and
that is the one that actually bit, in exactly the silent form predicted, just
not where it was expected to show up. See §5.6.

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

### 5.5 On-slice A/B — the layout applies, and it is *not* a win at 4096

Both builds booted back to back on the **same slice, same session, same
prompts** (job 3689007, `sk7524-museglimmer-followup`, v5p-8/us-east5-a),
swapping only `kv_cache_manager.py`. The baseline arm reproduces §4 to the
token, which is what makes the patched arm's numbers trustworthy.

| quantity | baseline (reverted) | **patched (hybrid)** |
|---|---|---|
| `num_kv_cache_groups` | **1** | **4** |
| `num_kv_cache_tensors` | 52 | **13** |
| `kv_cache_config.num_blocks` | 11 819 | **47 279** (4.00×) |
| layer specs | 52 × Full | 39 × Sliding(2048) + 13 × Full |
| **GPU KV cache size** | **3 025 664 tokens** | **2 890 369 tokens** |
| max concurrency @ 4096 | **738.69×** | **705.66×** |
| HBM/chip, model resident | 13.06 / 95.74 GiB | 13.06 / 95.74 GiB |
| HBM/chip after KV alloc | 88.08 / 95.74 GiB | 88.08 / 95.74 GiB |
| total HBM used / avail | 52.22 / 300.11 GiB | 52.22 / 300.11 GiB |
| engine boot to first answer | 182 s | 202 s |

The CPU A/B in §5.2 predicted the group count, the tensor count and the block
count **exactly**. What it did not predict is the last two rows of the memory
story: **the hybrid layout is a 4.5 % capacity *regression* at
`max_model_len=4096`, not a 4× win.**

That is not a contradiction, it is the ratio — and it is worth being precise
about what is measured and what is inferred. **Measured:** the same 300.11 GiB
of KV memory yields 3 025 664 usable tokens flat, and 2 890 369 hybrid, at
`max_model_len=4096`. **Argued:** a sliding layer can never save more than
`1 − sliding_window/max_model_len` of its own footprint, so the *ceiling* on
this change is set by `max_model_len / sliding_window`; at 4096 against a 2048
window that ratio is only **2**, and whatever vLLM spends on splitting one pool
into four equally-sized groups — each of which must still be large enough for
the full-attention group, since a live request occupies all four at once —
evidently exceeds it. The exact block accounting behind 2 890 369 was not
reverse-engineered and is not claimed here.

§5.2's 4.00× came from a CPU probe that assumed a much longer context. **The
prize is real but it is a long-context prize**, and it was never measured at a
long context because the layout was reverted for correctness first (§5.6). At
4096 this model already supports 738× concurrency — far past any `max-num-seqs`
anyone would set — so the change is not load-bearing at the one length where it
has now been shown to be a small regression.

### 5.6 Why it was reverted: batch-1 exact, batch-5 wrong

Correctness, patched build, all against the same references:

| check | result |
|---|---|
| window-boundary sweep, 27 probes 16 … 3609 | **27/27 agree with HF** |
| shortest diverging prefix length | **none — no divergence, boundary probed** |
| decode *crossing* the window (prefix 2020/2040/2048, +24 tok) | **3/3 EXACT** |
| 5 greedy prompts vs recorded ids | p1/p2/p4/p5 **EXACT**, p3_mid the known §4 tie |
| 5 greedy prompts, **patched ids == baseline ids** | **5/5 identical** |
| 8 concurrent × 64 tok, same prompt | all identical, consistent with batch 1 |
| **5 different prompts fired concurrently** | **4/5 — `p2_odd` diverges** |

The boundary sweep is the check this whole exercise was built around and the
patched build passes it outright: all 27 probes agree, including the dense
run 2046, 2047, **2048**, 2049, 2050, 2052, 2056 that would expose a block-index
off-by-one, and 3584/3608/3609 that would expose eviction. There is no shortest
diverging length to report.

The failure is somewhere else entirely. Fire the same five prompts
*concurrently* instead of one at a time and `p2_odd` — a 35-token prompt — comes
back different from its own sequential result, **first diverging at generated
token 45**, i.e. at a total sequence length of 80 tokens, a factor of 25 *inside*
the 2048 window. So it is not a windowing bug at all; it is block-pool
bookkeeping across the 4 groups being disturbed by what else shares the pool.
The same run on the reverted build is **5/5 identical**, so this is a
patched-only regression and not the pre-existing `p3_mid` argmax tie (`p3_mid`
was identical in both builds).

This is precisely the failure mode a batch-1 test cannot see, and it is the
reason the concurrency phase exists. A change that is byte-exact on 27
adversarially-chosen prefixes and then silently alters one sequence in five
under load is worse than no change at all, so **both edits were reverted with
`git checkout`** and every later stage in §6 ran on the reverted build.

Re-landing this would need the group-aware block bookkeeping debugged under
concurrency, and the `uniform_kv_dims` gate narrowed to models known to be
group-aware rather than widened by config shape alone — as written it silently
changes the KV layout of every JAX model whose global head dims happen to be
absent. It should also be gated on `max_model_len / sliding_window` being large
enough to pay for itself, which at 4096 it is not.

The reverted diff is kept verbatim at
`runs/muse_glimmer/reverted-hybrid-kv.patch` (112 lines) so none of this has to
be reconstructed.

---

## 6. Follow-up run — **LANDED, 2.76 chip-hours**

`followup_tpu.sbatch` → slurm **3689007** → `followup_tpu.sh`. One spot v5p-8
QR, `sk7524-museglimmer-followup`, us-east5-a. (A first attempt, job 3687240,
never landed and cost 0 chip-hours; spot capacity in us-east5-a was simply
absent in that window. This one landed in **5m53s**.)

| t+ | event |
|---|---|
| 0 | QR created `03:00:32Z` |
| 2m44s | `PROVISIONING` |
| 5m53s | **ACTIVE** |
| 7m10s | weights restored from GCS to local SSD (56 GB, ~70 s) |
| 15m45s | venv built, `ARCH-REGISTERED-OK` |
| 20m48s | **patched engine answered** (202 s boot) |
| 25m19s | baseline engine answered (182 s), item-1 A/B complete |
| 30m + | 8192 → 16384 → 32768, then the LoRA smoke |
| 47m18s | all stages done, teardown |

**`SLICE-ACTIVE-SECONDS 2485`** on 4 chips = **2.76 chip-hours**, one spot
v5p-8, single attempt. Teardown verified three ways (script loop, wrapper
backstop, wrapper audit): **zero muse QRs and zero muse TPU VMs in us-east5-a,
us-east5-b and us-east5-c.**

Item 1 is §5. The rest:

### 6.1 Context past 4096 — **32768 serves; the ladder ran out, not the model**

Every rung served, on the reverted build. Prompts genuinely exceed each window,
so the 13 full-attention (NoPE) layers carry real long-range load.

| `max-model-len` | boot | KV tokens | blocks × block_size | max concurrency |
|---|---|---|---|---|
| 4096 | 182 s | 3 025 664 | 11 819 × 256 | 738.69× |
| 8192 | 182 s | 3 025 664 | 11 819 × 256 | 369.34× |
| 16384 | 222 s | 3 025 856 | 189 116 × 16 | 184.68× |
| **32768** | 278 s | 3 025 856 | 189 116 × 16 | **92.34×** |

Two things worth keeping. First, **KV token capacity is independent of
`max_model_len`** — it is `available HBM / per-token KV bytes`, and both are
fixed — so concurrency simply halves each time the window doubles, exactly
proportionally. The HBM tradeoff per rung is therefore *nil*: 13.06/95.74 GiB
per chip with the model resident and 88.08/95.74 GiB after KV allocation, at
every length. What you trade is concurrency, not memory. Second, vLLM silently
switched `block_size` 256 → 16 at 16384 and above, which is why the block count
jumps 16× while the token count does not move.

**32768 is where the ladder stopped, not where the model stopped**; nothing
failed, `followup_tpu.sh` just had no larger rung. The model advertises 131072
and there is ~92× concurrency of headroom left at 32768.

Retrieval, not just allocation: `mg_client2.py --phases longfallback` plants a
needle (`TANGERINE-7741`) and a second fact (custodian `Rasmussen`) in the first
~30 tokens and asks for both at the very end.

| context | needle | custodian | repeat-determinism |
|---|---|---|---|
| 4096 | ✅ | ✅ | ❌ |
| 8128 | ✅ | ✅ | ✅ |
| 8192 | ✅ | ✅ | ✅ |
| **16320** | ✅ | ✅ | ✅ |

Both facts are recovered at **16 320 tokens** of context — 8× the sliding
window — so the full-attention layers really are carrying information the
sliding layers cannot. (The one `deterministic=False` at 4096 is a repeat of the
same greedy request returning different ids; it did not recur at any longer
length and is not explained here.)

### 6.2 Temperature > 0 — failed here, **recovered in §7's slice**

On this slice the phase died as `PHASE FAILED: <HTTPError 400: 'Bad Request'>`
on its *first* request, measuring nothing. The cause was harness fragility, not
the model, and all three contributing bugs are now fixed in `mg_client2.py`:

1. `post()` raised a bare `HTTPError 400: Bad Request` and **threw away the
   response body**, which is where vLLM puts the actual reason. It now reads the
   body into the exception. That single missing line is why a whole phase's
   worth of slice time produced no diagnosis.
2. `phase_temp` assigned `res["temperature"]` only at the *end*, so one failed
   request discarded the logprob, determinism and degeneration blocks together.
   The result dict is now published immediately and each block is guarded
   independently.
3. The phase hardcoded `logprobs=16, seed=1234`. It now **probes** for a
   sampling-parameter combination the server accepts — `(16,1234) → (5,1234) →
   (16,None) → (5,None) → (None,1234) → (None,None)` — records which one was
   accepted and which were rejected, and reports "seed rejected, reproducibility
   untested" rather than dying.

Re-run on §7's slice at `max_model_len=16384`, it worked first time, and fix (1)
produced the answer immediately:

```
sampling-param probe: logprobs=16 seed=None
  (last rejection: HTTP 400 ... {"message":"JAX does not support per-request seed."})
```

**(a)/(b) seed reproducibility and seed sensitivity: not testable on this
stack.** The JAX/TPU backend rejects `seed` outright — this is a platform
limitation, not a result. Any RL loop that relies on per-request seeds for
reproducible sampling cannot get them here.

**(c) degeneration: none.** 256 tokens at each temperature, no periodic run at
either:

| T | distinct-1 | distinct-2 | longest periodic run | degenerate |
|---|---|---|---|---|
| 0.7 | 0.320 | 0.580 | 0 | **False** |
| 1.0 | 0.582 | 0.898 | 0 | **False** |

**(d) logprobs vs HF at matched temperature — the load-bearing one: PASSES.**
Greedy is invariant to the monotone `output_multiplier` (0.196) and the
`T*tanh(logits/T)` softcap, so §2/§4 could not have caught a sampling-side
rescaling bug. This can, and does not find one.

| T | prompt | top-1 tpu / hf | top-16 overlap | err vs **raw** | err vs **T-scaled** |
|---|---|---|---|---|---|
| 0.7 | p1_tiny | 13796 / 13796 | 15/16 | **0.0533** | 1.2350 |
| 0.7 | p2_odd | 721 / 721 | 16/16 | **0.0391** | 0.4450 |
| 0.7 | p3_mid | 589 / 589 | 16/16 | **0.0937** | 1.4865 |
| 0.7 | p4_long_sliding | 589 / 589 | 16/16 | **0.1517** | 1.0113 |
| 0.7 | p5_code | 277 / 277 | 16/16 | **0.0897** | 6.1563 |
| 1.0 | all five | identical | 15-16/16 | 0.053-0.145 | (same by definition) |

Three things fall out. **The top-1 token matches HF on every prompt at every
temperature**, and the top-16 sets match 16/16 on four of five (15/16 on the
6-token p1_tiny), so the ordering of the sampled distribution is right. **The
returned logprobs track the raw distribution, not the temperature-scaled one** —
at T=0.7 the error against `log_softmax(logits)` is 10-70× smaller than against
`log_softmax(logits/T)`, and at T=1.0 the two columns coincide exactly, as they
must. So vLLM reports pre-temperature logprobs and temperature affects sampling
only; that is a reporting convention, not a bug, but it is worth knowing before
anyone treats these logprobs as the sampling distribution in an importance
ratio. **The residual 0.04-0.15 nats is bf16-vs-float32**, not a scaling error:
a misapplied `output_multiplier` or softcap would move the error by orders of
magnitude, which is exactly what the T-scaled column shows a real mismatch
looks like.

### 6.3 LoRA smoke on the real weights — **PASS**

First real-weight optimizer step this port has ever taken. `lora_smoke.py`,
tunix + MaxText (`maxtext @ git+…@4f65ba509`), orbax pulled from GCS
(`ORBAX-FROM-GCS`), `JAX_PLATFORMS=tpu`, 4 devices, rank 32 / alpha 64.

| assertion | result |
|---|---|
| **LoRA adapter count non-zero** | **56 arrays, 191 692 800 params** |
| adapters on the right modules | `self_attention/{query,key,value,out}` and `mlp/{wi_0,wi_1,wo}` |
| base params really FSDP-sharded | **`fsdp_applied=True`**, 34/117 arrays sharded, mesh `fsdp=4` |
| base weight footprint | 51.88 GiB total = **12.97 GiB/device** |
| loss finite and moving | −0.805662 → −0.904752 → −0.925850 (final −0.929787) |
| grad_norm finite and non-zero | 10.375 → 5.0625 → 1.0703 |
| verdict | **PASS** (`loss_moved`, `loss_decreased`, `all_finite`, `grad_norm_nonzero`) |

The non-zero adapter count is the assertion that matters: qwix matches adapters
by module path, and a name mismatch injects **zero** adapters and trains
nothing while still producing a plausible-looking loss curve. The paths above
confirm MaxText's attribute really is `self_attention`, which is what the tunix
regex `layers_[0-9]+/self_attention/(query|key|value|out)` needs.

Step 2 taking 0.73 s against step 0's 12.3 s is compilation, not divergence.

## 7. TP head-to-head: one wide engine vs two narrow ones

`tp_compare_tpu.sbatch` → slurm **3689227** → `tp_compare_tpu.sh`, one spot
v5p-8, QR `sk7524-museglimmer-tp`, on the reverted (baseline) build. Landed
after **31m24s** of hunting; **`SLICE-ACTIVE-SECONDS 1625`** on 4 chips =
**1.81 chip-hours**. Teardown verified in all three zones.

**Operational lesson, paid for in slice time.** The zone-rotation loop inherited
from `followup_tpu.sh` breaks out of a zone when `ZONE_TRY_SEC` expires unless
the QR is `ACTIVE` — and this run reached `PROVISIONING` at 853 s, 47 s before
its 900 s slot expired, so the loop **deleted a slice that had already been
granted** and rotated away. It cost ~9 minutes and got lucky on the retry.
`PROVISIONING` means capacity is committed and must be treated as terminal for
the purposes of the slot deadline:

```bash
# wrong: abandons a slice that is already being built
if [ "$el" -ge "$zone_deadline" ]; then break; fi
# right: only a still-queued QR may be abandoned
if [ "$el" -ge "$zone_deadline" ] && [ "$st" != PROVISIONING ]; then break; fi
```

### 7.1 Why this is the measurement worth a slice

Muse-Glimmer has **`num_key_value_heads = 2`**, and
`tpu_inference/utils.py:230::get_padded_num_heads` pads the KV head count *up*
to the shard count whenever heads < shards:

```python
def get_padded_num_heads(num_heads, sharding_size):
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size      # <- 2 KV heads become `sharding_size`
    return num_heads
```

So per-token KV bytes scale with **TP**, not with the model's real KV width:
`get_padded_num_heads(2,2)=2`, `(2,4)=4`, `(2,8)=8`. §4's measured 106 502 B per
token at TP=4 is exactly `2 (K,V) × 52 layers × 4 padded heads × 128 × 2 B =
106 496 B`, confirming the padding is real and paid for in HBM.

Pulling the other way: *N* independent engines hold *N* copies of the 51.9 GiB
of weights, so a split configuration starts one whole model down on KV before it
wins anything back. Which effect dominates is an empirical question.

**Shape of the experiment.** A v5p-8 is **4 chips** (8 TensorCores), so TP=8 does
not exist on one host here — a v5p-16 is two hosts, and multi-host TP is a
bring-up this harness does not do. 4 chips gives the identical mechanism one
octave down:

| arm | engines | TP each | padded KV heads | weight copies |
|---|---|---|---|---|
| **A** | 1 | 4 | **4** (2× waste) | 1 |
| **B** | 2 | 2 | **2** (no waste) | 2 |

which is exactly the TP=4-vs-TP=8 relationship on an 8-chip host, shifted down
one power of two. Arm A is booted with no `TPU_*` pinning at all — i.e. the §4
configuration verbatim, so the control is a configuration already known to
serve. Arm B pins `TPU_VISIBLE_CHIPS=0,1` and `2,3` with
`TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1` and distinct `TPU_PROCESS_PORT`s, mirroring
what `tpu_worker.py::_setup_dp_chip_isolation` does for multi-process DP.
`tpu/start_colocated_vllm_tinker.sh` was **not** modified; everything is env.

Measured at `max_model_len=16384`, where the capacity question actually bites
(at 4096 this model already serves 738× concurrency).

### 7.2 Results — both engines coexisted; capacity 1.66× to the split, single-stream latency 1.73× to the wide one

Chip partitioning works: two TP=2 engines booted on chips `{0,1}` and `{2,3}`
of one v5p-8 and both answered real completions (216 s and 184 s to first
answer). Nothing in `start_colocated_vllm_tinker.sh` was touched.

| | **A — 1 × TP=4** | **B — 2 × TP=2** |
|---|---|---|
| padded KV heads | **4** (2 real, padded up) | **2** (no padding) |
| per-token KV bytes / engine | **106 496** | **53 248** |
| — measured from the pool | 106 496 | 53 248 |
| weights in HBM / engine | 52.22 GiB | 51.89 GiB (**×2 copies**) |
| HBM per chip, model resident | **13.06** / 95.74 GiB | **25.94** / 95.74 GiB |
| KV pool | 300.11 GiB | 124.28 GiB / engine |
| **KV tokens** | **3 025 856** | **5 012 160** (2 506 080 × 2) |
| **max concurrent seqs @16384** | **184.68** | **305.92** (152.96 × 2) |
| **decode tok/s, batch 1** | **69.76** | **40.28** |
| decode tok/s, saturated (64 concurrent) | 217.2 | 1452.31 |
| p50 latency at saturation | 37.69 s | 5.57 s |

**Capacity: 1.656× to the split configuration.** The per-token KV bytes measured
off the live pool are *exactly* the analytic
`2 × 52 layers × padded_heads × 128 × 2 B` — 106 496 at TP=4, 53 248 at TP=2 —
so `get_padded_num_heads` really is doubling the KV cost of every token at TP=4,
and halving TP halves it back. The split does **not** get the full 2× because it
pays for a second copy of the weights: 124.28 GiB of KV per engine instead of
150.06 GiB. A closed-form prediction from the measured constants
(`(352.33 GiB cap ÷ 2 chips − 51.9 GiB) ÷ 53 248 B`) gives 2 506 400 tokens per
engine against **2 506 080 measured — 0.013 % out**, so this trade is now fully
characterised rather than merely observed.

**Single-stream latency: 1.732× to the wide engine**, in the predicted direction
and close to the predicted size. One sequence decoding alone is bound by reading
the weights, and arm A reads 13.06 GiB per chip per step against arm B's 25.94 —
a 1.99× ratio that shows up as a 1.73× speed ratio.

**Saturated throughput: 6.687× to the split — reported, not explained.** This is
far outside what the bandwidth model predicts. At batch 64 with ~600-token
contexts the weights still dominate the per-step read (≈14.0 GB/chip of weights
against ≈1.0 GB/chip of KV for arm A), so that model predicts arm A should be
roughly **2× faster**, not 6.7× slower. Arm A's per-step time grows ~20× from
batch 1 to batch 64 while arm B's grows only ~1.8× from batch 1 to batch 32.
Two caveats before anyone plans capacity on this row:

1. Both engines ran with `SKIP_JAX_PRECOMPILE=1`, so a batch shape first seen
   inside the timed window pays XLA compilation inside the measurement. The
   bench warms the identical shape first, and the batch-1 arm shows exactly this
   artefact and survives it (`samples: [4.06, 69.76, 69.81]` — the median
   discards the compile), but **saturation was measured once, with no median to
   protect it.**
2. It is a single concurrency point (64) at a single length (16384).

The capacity and batch-1 rows are solid — analytic, reproduced, and
direction-correct. The saturation row is a real observation that wants a
repeat with precompilation on and a concurrency sweep before it is used for
anything.

### 7.3 What this says about TP=8, which was not measured

A v5p-8 is 4 chips, so **TP=8 was not run** — it needs a v5p-16, which is two
hosts, and multi-host TP is a bring-up this harness does not do. What transfers
is the mechanism, which is now measured rather than assumed:
`get_padded_num_heads(2, TP) = TP`, and per-token KV bytes are linear in it. On
an 8-chip host that gives 212 992 B/token at TP=8 against 106 496 at TP=4, so
2 × TP=4 should hold roughly **2× the tokens minus one extra copy of the
weights** — the same shape of trade measured here at 1.656×, and closer to 2×
than this 4-chip experiment because a 4-chip engine amortises the duplicated
51.9 GiB over twice as much HBM. The batch-1 direction should also hold: TP=8
reads 6.49 GiB/chip/step against 12.97, so it should decode a lone sequence
faster.

## 8. What is still unproven

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
  tens of thousands of positions is not. **Now proven to 32768** with needle
  recall at 16 320 tokens — see §6.1. Above 32768 is still untested.
- **Sampling.** Now measured (§6.2): top-1 and top-16 match HF at T=0.7 and
  T=1.0, the reported logprobs track the raw (un-temperature-scaled)
  distribution to within bf16 noise, and there is no degeneration at either
  temperature. **Not measurable on this stack:** fixed-seed reproducibility and
  seed sensitivity — the JAX backend rejects per-request `seed` with a 400.
  Still untested: top-p / top-k paths, and whether sampled *sequences* (rather
  than the next-token distribution) match HF in distribution.
- **The hybrid KV-cache-group change.** Measured on hardware and reverted; see
  §5. Byte-exact at batch 1 on all 27 boundary probes, wrong on one sequence in
  five under concurrency, and a 4.5 % capacity *regression* at 4096 rather than
  the 4× win the CPU probe suggested.
- **Quantization.** Served bf16 unquantized; no qwix/AWQ/FP8 path tried.
- **Topology.** PP=1, DP=1 throughout. TP=4 and TP=2 are both exercised (§7);
  **TP=8 and multi-host are untested** — a v5p-8 is 4 chips, so TP=8 needs a
  v5p-16, which is two hosts, and multi-host TP is a bring-up this harness does
  not do. KV-head replication (`_load_kv_proj_replicated`) is a
  TP-vs-`num_key_value_heads=2` concern and is now proven at TP=4 and TP=2.
- **Training on real weights.** **Done** — §6.3 injects 56 LoRA adapter arrays
  into the real 30B and takes 3 optimizer steps with finite, decreasing loss and
  FSDP genuinely applied. Not done: a full training run, or any evaluation that
  the adapted model is *better* at anything.
- **`prompt_logprobs`.** Probed once, as a known EngineCore killer on this stack
  (see the tinker/qwen35 notes). It is not part of the pass criteria.
- **Sustained serving.** No long-run stability, preemption/eviction-under-
  pressure, or chat-template exercise; §4's throughput number is a small fixed
  batch, not a benchmark.

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

# 9. the TP head-to-head: one TP=4 engine vs two TP=2 engines on the same 4
#    chips, plus the recovered temperature phase.  KV_VARIANT stays `baseline`
#    unless the §5 change is ever re-landed.
sbatch --export=ALL,KV_VARIANT=baseline,BENCH_LENS="16384 8192" \
    tpu/muse_glimmer/tp_compare_tpu.sbatch
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
