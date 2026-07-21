---
title: "Tunix Backend: Implementation Report"
description: Everything it took to train Qwen3-8B, Qwen3.5-27B, gemma4-31b, and gpt-oss-20b with LoRA RL on TPU — the fixes, the forks, and how each bug was found
---

# Tunix backend implementation report

This is the engineering record of bringing up **LoRA math-RL on TPU v5p-16 spot
machines** for four model families behind the Tinker API, with training powered
by [tunix](https://github.com/google/tunix) (MaxText model implementations,
qwix LoRA) and sampling served by a **colocated vLLM TPU server with inflight
LoRA hot-swapping**. It documents every change that was load-bearing, with
code links and snippets for the small-but-critical ones.

## Results

| Model | Arch challenge | Held-out math eval | Verdict |
|---|---|---|---|
| Qwen3-8B | none (baseline path) | 1.2% → **77.4%** (180 steps) | clean; survived spot preemption mid-run |
| Qwen3.5-27B dense | hybrid GDN + attention, unsupported upstream | 37.2% → **87.0%** peak @140, 86.4% final | cleanest run: step-0 KL 1.7e-4, no collapse |
| gemma4-31b | multimodal composite, grouped-scan decoder | 62.4% → **81.0%** peak @160, 80.4% final | survived 4 preemptions; near-monotonic |
| gpt-oss-20b | fused MoE weights (no per-expert LoRA path in vLLM) | 62.6% → **76.8%** peak @60, then decline | merge-on-load worked; bf16 serving numerics are the model's real enemy (see §6) |

## 1. Architecture

```
┌────────────────────── v5p-16 (2 hosts × 4 chips) ──────────────────────┐
│ worker 0 (trainer)                    worker 1 (sampler)               │
│ ┌──────────────────────────┐          ┌────────────────────────────┐   │
│ │ Tinker API (FastAPI)     │          │ vLLM TPU (torchax)         │   │
│ │  └ engine + SQLite       │  HTTP    │  --enable-lora             │   │
│ │  └ TunixBackend          │ ───────► │  runtime LoRA endpoints    │   │
│ │     MaxText model (qwix  │  adapter │  (from our tpu-inference   │   │
│ │     LoRA), 4-chip mesh   │  push    │   fork)                    │   │
│ └──────────────────────────┘          └────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
        ▲ SSH tunnel                                 ▲
        │ tinker SDK (math-RL cookbook client, streaming minibatches)
```

- **`skyrl/backends/tunix_backend.py`** implements the Tinker
  `AbstractBackend` contract on tunix models: jitted
  `nnx.value_and_grad` over LoRA params only, per-model gradient
  accumulation, `optax.inject_hyperparams(adamw)`, orbax/npz
  checkpointing, and HF-PEFT adapter export.
- **External sampling path**: sample requests bypass the single-threaded
  engine loop (`EXTERNAL_SAMPLING=1`) so rollouts and `forward_backward`
  run concurrently on their respective hosts. With the cookbook's
  stream-minibatch mode this pipelines a fully **on-policy** step.
- **Adapter sync**: every `save_weights_for_sampler` exports an HF-PEFT
  directory and hot-swaps it into vLLM under a versioned name
  (unload-previous + retried load) — the sampler always serves the latest
  optimizer state without restarts.

### Where each step's time goes (measured averages)

| Stage (s/step, steady state) | Qwen3.5-27B | gemma4-31b | gpt-oss full-MoE | gpt-oss attn-only |
|---|---|---|---|---|
| **step wall total** | **105** | **135** | **131** | **72** |
| sampling span (slowest of 32 groups) | 56 | 45 | 20 | 15 |
| fwd/bwd waves (un-hidden drain) | 38 | 76 | 54 | 54 |
| optim_step | 0.1 | 0.2 | 0.1 | 0.01 |
| adapter export + push | 10 | 13 | **56** | 2.5 |

The rows tile the wall clock but hide overlap: early fwd/bwd waves execute
*inside* the sampling span (their consume-waits read 0.0 s — results already
computed), and each sampling span compresses ~20 min of sequential-equivalent
rollout compute into &lt;1 min of batched decode. gpt-oss's 56 s sync row is
the MoE merge-on-load (§6); the attention-only variant collapses it to 2.5 s.

## 2. Serving stack: the tpu-inference fork

vLLM's TPU backend (`tpu-inference`) ships LoRA support that is
half-wired: the TPU worker never forwards the runtime-LoRA RPCs. We
maintain a fork, installed from source as a submodule (no deploy-time
patching):

**Fork: [`SachinKonan/tpu-inference@skyrl/v0.23.0-lora`](https://github.com/SachinKonan/tpu-inference/tree/skyrl/v0.23.0-lora)** (based on `releases/v0.23.0`)

| Commit | Change |
|---|---|
| `a77620929` | dynamic LoRA adapter management: TPUWorker `add/remove/list/pin_lora` forwarders, env-var allowlist |
| `f8a15f67f` | `apply_moe_lora_deltas` worker RPC — MoE LoRA merge-on-load for fused experts (§6) |
| `34dd47337` | skip `FusedMoEWithLoRA` wrapping on TPU (fused-MoE LoRA uses merge-on-load instead) |
| `182bc4426` | extract pure merge core + parametrized unit tests (incremental ≡ direct, §6) |
| `8b20a6749` | host-side deltas, donated-buffer adds, real w13 layout, path-based RPC (§6) |
| `0ae6bf7aa` | clamp NaN logprobs at source (§6, NaN chain layer 1) |

The NaN clamp is four lines and saved every gpt-oss run
([commit](https://github.com/SachinKonan/tpu-inference/commit/0ae6bf7aa)):

```python
# tpu_inference/layers/jax/sample/sampling.py
def compute_logprobs(logits: jax.Array) -> jax.Array:
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    # bf16 gpt-oss emits occasional NaN logprobs; clamp at source for RL stability.
    return jnp.nan_to_num(logprobs, nan=-1.0e4, posinf=0.0, neginf=-1.0e4)
```

Without it, a single NaN token logprob 500s the OpenAI-compatible API
("Out of range float values are not JSON compliant") and kills the whole
rollout group.

## 3. Qwen3.5-27B dense: three numerics bugs in a fork

**The problem:** nobody supports Qwen3.5-27B *dense* for training. MaxText
ships only the MoE SKUs (35B-A3B / 397B-A17B); Qwen3.5 is a hybrid of gated
DeltaNet (GDN) layers and periodic full-attention layers with partial RoPE.

**Fork: [`SachinKonan/maxtext@skyrl/qwen35-dense`](https://github.com/SachinKonan/maxtext/tree/skyrl/qwen35-dense)**

### 3.1 Dense architecture support ([`021d337`](https://github.com/SachinKonan/maxtext/commit/021d3370f))

Upstream's `qwen3_5` decoder hardcodes the MoE MLP. Added: a dense-MLP
branch, the `qwen3.5-27b.yml` model config, and HF→orbax converter
mappings for the dense weights.

### 3.2 mRoPE must be off for text-only ([`f71eec8`](https://github.com/SachinKonan/maxtext/commit/f71eec8dd))

MaxText's mrope implementation rotates the **entire** head dim, ignoring
`partial_rotary_factor: 0.25` (Qwen3.5 rotates only 64 of 256 dims per
head). With equal (t,t,t) position streams — i.e. text-only — HF's
interleaved mrope is mathematically identical to plain partial RoPE, which
MaxText's `is_qwen3_hybrid` path implements correctly. One yml line:

```yaml
# use_mrope MUST stay false for text-only use: the mrope branch rotates the
# full head dim and ignores partial_rotary_factor, diverging from HF/vLLM.
use_mrope: false
```

### 3.3 The big one: zero-centered final RMSNorm ([`0fd4099`](https://github.com/SachinKonan/maxtext/commit/0fd409939))

Qwen3-Next/Qwen3.5 use **zero-centered** RMSNorm everywhere — HF stores
`w` as a delta around zero and applies `(1 + w) · x̂`. MaxText's legacy
linen decoder already had this (`Qwen3NextRMSNormLinen`,
`scale_offset=1.0`), but the **pure-NNX path** (which tunix uses) built the
final `decoder_norm` as a plain `w · x̂` RMSNorm — silently rescaling the
LM-head input per channel:

```python
# src/maxtext/layers/nnx_decoders.py — get_norm_layer, pure-NNX branch
elif self.config.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5):
  return functools.partial(
      normalizations.RMSNorm,
      num_features=num_features,
      shard_mode=self.config.shard_mode,
      scale_offset=1.0,   # <-- the fix: (1 + w) * x_hat, not w * x_hat
      rngs=rngs,
  )
```

**Impact:** a flat ~2 nats/token trainer-vs-sampler logprob divergence.
Nothing crashes; outputs stay plausible; RL would silently optimize a
distorted objective through wrong importance ratios.

**How it was found** (the method matters more than the bug):

1. **Three-way logprob parity** — score the *same sampled tokens* under
   vLLM, HF transformers (CPU reference), and the MaxText trainer.
   vLLM↔HF agreed to 0.007 mean-abs ⇒ sampler exonerated, trainer indicted.
2. **Per-layer CPU bisection** vs HF `output_hidden_states` — expecting to
   find the first divergent layer. **All 64 layers matched within 1%**,
   yet logits differed by 2 nats. Paradox.
3. **The "artifact" was the answer**: HF's `hidden_states[-1]` is
   *post-final-norm*. The one comparison mismatch we'd dismissed as an
   indexing artifact was the final norm itself — the only module between
   the last matching activation and the diverging logits that the
   bisection never tested.
4. Read the code, find plain-vs-zero-centered, one-line fix. Trainer-vs-HF
   fell to 0.0066 (bf16 noise); step-0 KL fell from ~2 nats to **0.000169**
   and stayed in [1e-4, 5.3e-4] for all 180 steps.

> Lesson: per-layer bisection only clears the modules it tests, and
> reference introspection outputs have framing conventions (pre/post-norm)
> that can silently hide the culprit module from the comparison.

### 3.4 Supporting changes (this repo)

- **Mesh-aware qwix trace batch** (`skyrl/backends/tunix_backend.py`):
  GDN's `shard_map` kernel shards batch over `fsdp`, so the fixed batch=2
  dummy input used to trace LoRA fails on a 4-chip mesh:

  ```python
  # Batch must be divisible by every mesh axis the decoder shards it over.
  batch, seq_len = max(2, jax.device_count()), 4
  ```

- **LoRA scope**: full-attention q/k/v/o + dense MLP only; GDN internals
  excluded (vLLM's LoRA wraps only standard linear layer types).
- **Renderer**: `qwen3_5_disable_thinking` is required — the default
  thinking renderer burns the entire 512-token budget inside `<think>`
  (0.8% format rate).
- **Launcher**: `TUNIX_MAXTEXT_PIP_SPEC` env so bring-up installs the fork
  instead of stock PyPI maxtext (`tpu/start_colocated_vllm_tinker.sh`).

## 4. gemma4-31b: an exercise in weight-tree topology

MaxText's gemma4 numerics were already correct — every problem was
naming, layout, or serving configuration.

### 4.1 Composite-model PEFT names

HF gemma4 is a multimodal composite; the decoder lives under
`model.language_model.layers.N`. Adapter tensors exported with the plain
`model.layers.N` prefix load "successfully" and match **nothing** — vLLM
applies a no-op adapter. The export arm targets the composite names
(`skyrl/backends/tunix_backend.py`, `_peft_target_prefix`).

### 4.2 Grouped-scan layer mapping

MaxText implements gemma4 as a *scanned block of 6 layers* (5
sliding-window + 1 global attention sharing KV): the weight tree is
indexed `(stack_idx j, group g)` while HF numbers layers flat. The export
de-interleaves:

```python
# global HF layer index = stack_idx * n_groups + g
gl = j * n_groups + group
```

The layer types are inhomogeneous — global-attention layers carry no
`v_proj` adapter — so a correct export is **820** tensors
(60×7×2 − 10×2), not the naive 840. Getting this wrong is silent: the
adapter loads and does nothing for the mismapped layers.

### 4.3 vLLM serving pins

- `--max-num-batched-tokens 4096`: gemma4's multimodal default is 2496 —
  **not a power of two**, which breaks vLLM's LoRA punica bucket sizing on
  TPU. Plus `--disable-chunked-mm-input`.
- A custom `gemma4` chat renderer in the
  [tinker-cookbook fork](https://github.com/SachinKonan/tinker-cookbook/tree/skyrl/rl-stream-tolerance)
  (the cookbook had none).

## 5. Preemption safety (spot machines are the deployment reality)

Everything runs on v5p-16 **spot** slices; the gemma4 run alone survived
four preemptions. Two composable layers:

1. **Per-machine autostart** (`runs/math_rl/autostart_math_rl_qwen35_9b_v5p32_r3.sh`):
   installed on the TPU VM, relaunched by jobman on every (re)creation;
   rebuilds the environment from a pinned bundle; blocking rolling
   checkpoints; a simulated-preemption endpoint
   (`SKYRL_ENABLE_PREEMPT_ENDPOINT`) for testing the resume path.
2. **Per-run fire-and-forget supervisor** (`tpu/autoresume_run.sh` + env
   files under `tpu/runs/`): runs off-TPU, owns queued-resource recreation
   → READY wait → colocated bring-up → client launch → health loop
   (relaunch on node loss, client death, or stalled metrics; clean exit at
   `MAX_STEPS`). One command per run:

   ```bash
   ./tpu/autoresume_run.sh tpu/runs/qwen35-27b.env
   ```

Supporting fixes that resume-on-recreation required: bf16-safe npz
checkpoint round-trip (bf16 arrays can't go through `np.savez` directly;
kind-'V' views are restored via dtype metadata), adapter re-extraction
after restore, and orbax base-weight caching on GCS so a fresh machine
skips HF→orbax conversion (~40 GB, reused across all preemptions).

## 6. gpt-oss-20b: merge-on-load for fused MoE, and the NaN war

### 6.1 Why LoRA-on-MoE needs special serving machinery

The trainer's qwix LoRA factors MoE experts as shared/per-expert pairs
(`wi_0`/`wi_1`: shared A `(d,r)` + per-expert B `(r,E,f)`; `wo`:
per-expert A + shared B; router: `(A@B)ᵀ`). vLLM cannot serve LoRA on
**fused** expert weights — there is no per-expert linear to wrap. Industry
practice is merge-into-weights; we built **incremental merge-on-load**
([`apply_moe_lora_deltas`](https://github.com/SachinKonan/tpu-inference/blob/skyrl/v0.23.0-lora/tpu_inference/runner/tpu_runner.py)):

> Incremental semantics: `new_state = state + delta(new) − delta(prev)`,
> where `prev_factors` describe the previously merged factors (`None` on
> the first call). This keeps merged weights equal to
> `base + delta(latest)` **without a pristine copy of the base**.

Three production lessons, each earned the hard way
([`8b20a6749`](https://github.com/SachinKonan/tpu-inference/commit/8b20a6749)):

1. **Wire format**: `collective_rpc` args pass through vLLM's
   MsgpackEncoder, which rewrites ndarrays into typed triples the untyped
   target never reconstructs. Factors now travel as a worker-local
   **safetensors path** (frontend shares the host).
2. **HBM discipline**: on-device einsum deltas OOM'd at layer 8 (serving
   prealloc leaves a few GB). Deltas are computed on the **host** in f32
   BLAS, cast to weight dtype, and applied one component at a time via a
   **donated-buffer sliced add** — peak extra HBM = one component
   (~0.5 GiB), not 40 GiB of expert-tensor copies.
3. **Layout truth**: the merge assumed a 4-D `(E, 2, H, I)` stacked w13;
   the real torchax layout is 3-D `(E, H_pad, 2·I_pad)` — observed
   `(32, 2880, 5888)` — gate|up concatenated uninterleaved. Every gate/up
   delta had been **silently skipped** as "does not fit".

Unit tests ([`182bc4426`](https://github.com/SachinKonan/tpu-inference/commit/182bc4426))
prove incremental ≡ direct-from-base within 1.86× a single bf16 rounding
step over 3 simulated syncs, parametrized over both w13 layouts and all
three wire forms. In production: 180 consecutive merges, 39–51 s each.

### 6.2 The NaN chain (three defensive layers)

bf16 gpt-oss on TPU torchax emits occasional NaN logprobs **from the base
model at step 0** — this is a serving-numerics property, not a training
artifact. Defenses, in order of discovery:

1. **Clamp at source** (fork, §2): NaN → −1e4 sentinel, keeps the API alive.
2. **Neutralize sentinels before jit** (`skyrl/backends/tunix_backend.py`):
   zeroing the loss weight is *not enough* — `exp(train_lp − (−1e4)) = inf`
   still forms inside the jitted loss and `inf × 0 = nan` poisons gradients:

   ```python
   sentinel = sampling_logprobs <= -9.0e3
   loss_mask = np.where(sentinel, 0.0, loss_mask)
   # Neutralize the value itself: even with zero weight,
   # exp(train_lp - (-1e4)) = inf inside the jitted loss and inf * 0 = nan.
   sampling_logprobs = np.where(sentinel, 0.0, sampling_logprobs)
   ```

3. **Non-finite optimizer guard**: if the global grad norm is non-finite,
   skip the update entirely (never poison LoRA weights) and return finite
   metrics with `skyrl.ai/skipped_nonfinite_update=1.0`.

### 6.3 What the runs proved

The **full-MoE run** hit 76.8% at step 60, then declined to 51% by 179 —
tracked by the sentinel fraction growing from 32% to 92% of tokens
(signal starvation, not classic overfitting). The **attention-only
ablation** (`lora_mlp_regex: "(?!)"` — a never-match regex, giving a
63.7 MB adapter, 2.5 s native-vLLM syncs, no merge) matched the full run's
quality at step 20 (76.6%) and trained stably to step 78, then tipped into
a degenerate repeated-token attractor at the final step (coherent math →
infinite `!` until the token cap).

**Verdict:** the merge machinery is exonerated — the collapse reproduces
without it. The common denominator is bf16 torchax serving numerics
amplified by RL feedback. gpt-oss RL on this stack needs the serving-side
root cause fixed (mxfp4-vs-bf16 logit investigation) or much more
conservative KL control; the adapter plumbing was never the problem.

## 7. Smaller load-bearing fixes (this repo)

| Fix | Why it mattered |
|---|---|
| Sequence buckets {128, 256, 384, 512} | splash-attention requires `bkv_compute` multiple of 128; the power-of-2 bucket 192 crashed mid-run |
| Tokenizer never from orbax `model_path` (+ tiktoken support) | MaxText checkpoints don't carry tokenizers; gpt-oss needs tiktoken |
| bf16 npz checkpoint round-trip | `np.savez` mangles bf16 (kind 'V'); restore views via dtype metadata — required for preemption resume |
| `rollout_error_tolerance` (cookbook fork CLI) | one failed rollout group would hang stream-minibatch forever |
| Group-coalesced sampling + seed omission | vLLM TPU rejects per-request seeds and greedy `n>1` |
| `UV_CACHE_DIR` pinned in run envs | tmux server env silently overrides shell exports with a quota-limited path; killed two launches |

## 8. Repository and fork inventory

| Repo | Branch | Role |
|---|---|---|
| SkyRLTpu (this repo) | `agent/tunix-backend` (PR #4) | backend, launchers, supervision, docs, tests |
| [SachinKonan/tpu-inference](https://github.com/SachinKonan/tpu-inference/tree/skyrl/v0.23.0-lora) | `skyrl/v0.23.0-lora` | runtime LoRA + merge-on-load + NaN clamp (submodule, source-installed) |
| [SachinKonan/maxtext](https://github.com/SachinKonan/maxtext/tree/skyrl/qwen35-dense) | `skyrl/qwen35-dense` | Qwen3.5 dense arch + the three numerics fixes (via `TUNIX_MAXTEXT_PIP_SPEC`) |
| [SachinKonan/tinker-cookbook](https://github.com/SachinKonan/tinker-cookbook/tree/skyrl/rl-stream-tolerance) | `skyrl/rl-stream-tolerance` | tolerance CLI, coalesced sampling, gemma4 renderer, rolling checkpoints (submodule) |

The through-line of the project: **qwen3.5 was a numerics problem** (three
subtle math bugs found by parity bisection), **gemma4 was a topology
problem** (exotic weight layout onto standard PEFT naming), and **gpt-oss
was a systems problem** (merge-on-load worked; the model's own serving
numerics were the real adversary). The shared infrastructure — pipelined
on-policy RL, inflight adapter serving, and preemption-safe supervision —
carried all four.
