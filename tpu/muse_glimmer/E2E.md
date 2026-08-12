# Muse-Glimmer-30B text-only: end-to-end on TPU

Status of the *real weight* half of the port. The tiny-random-weight math proof
lives in `SPEC.md` / `parity_check.py`; the MaxText training half lives in
`MAXTEXT.md`. This file covers: where the real weights are, whether our JAX
serving model reproduces HF logits with those weights, and whether the model
actually serves and decodes on a TPU through vLLM + tpu-inference.

Read `SPEC.md` first for the architecture contract. Where this file and the HF
reference disagree, the reference wins.

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
HBM.

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

## 5. What is still unproven

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
  tens of thousands of positions is not.
- **Sampling.** Only greedy (`temperature=0`) is compared. Nothing here says the
  sampled distribution matches HF, only that the argmax path does.
- **Quantization.** Served bf16 unquantized; no qwix/AWQ/FP8 path tried.
- **Topology.** One host, TP=4, PP=1, DP=1. KV-head replication
  (`_load_kv_proj_replicated`) is specifically a TP-vs-`num_key_value_heads=2`
  concern and is only proven at TP=4; TP=8 and multi-host are untested.
- **Training on real weights.** `MAXTEXT.md` covers the converter round-trip and
  HF parity; no real-weight optimizer step has been run.
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
```

The HF reference needs the unreleased transformers and therefore an isolated
venv (`/n/fs/vision-mix/sk7524/caches/muse-parity/venv`, built by
`build_parity_venvs.sbatch`); never install it into the project venv, which is
pinned for the live RL sweep.

### QR hygiene

`e2e_tpu.sh` creates **exactly one** QR, `sk7524-museglimmer-e2e`, and deletes
it on every exit path. `trap ... EXIT` is known not to fire on an untrapped
SIGTERM here, and a `scancel` has previously killed a cleanup mid-delete and
left a QR alive, so: TERM/INT/HUP are trapped explicitly, the delete is issued
`setsid nohup` so it outlives the shell, and the script re-issues and verifies
until the QR is gone. `e2e_tpu.sbatch` adds `--signal=TERM@300` (deliberately
without `B:`, so the signal reaches the script and not just the wrapper) plus a
backstop delete of its own, and prints the remaining muse QRs in both zones
into the job log so the result is auditable without re-running anything.
