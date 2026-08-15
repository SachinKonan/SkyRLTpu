# Muse-Glimmer-30B: the vLLM-native (torch) model

Why there are now two implementations of the same model, what the torch one
buys, what has been proven about it, and what has not.

Read `SPEC.md` first for the architecture contract. Where this file and the HF
reference disagree, the reference wins. `E2E.md` is the JAX path's record; this
file does not replace it and does not claim its results.

---

## 1. The blocker

SkyRL's RL loop syncs weights by uploading **LoRA adapters** to vLLM. The
JAX-native model cannot receive them:

```python
# tpu_inference/models/common/model_loader.py, in _get_nnx_model
lora_manager, model = None, None
```

That line is unconditional on the `flax_nnx` branch, so `--enable-lora` reaches
tpu-inference's runner with `lora_manager is None` and dies on
`AssertionError: LoRA is not enabled`. Qwen and Gemma do not hit this because
they run `VLLM_MODEL_IMPL_TYPE=vllm`, the torch wrapper path, which has the
whole LoRA stack.

**The fix is not a LoRA feature. It is a second model.** Build Muse-Glimmer out
of vLLM's own parallel-linear classes and LoRA arrives for free:
`vllm_model_wrapper.py::load_weights` calls `load_lora_model` and then
`replace_set_lora`, which walk `model.named_modules()` and wrap every
`BaseLayerWithLoRA` under `torchax.default_env()`. `QKVParallelLinear`,
`MergedColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding` and
`ParallelLMHead` all have LoRA variants in vLLM 0.23.0. torchax runs the
resulting torch graph on TPU.

The MoE delta-merge path (`apply_moe_lora_deltas`) is **not** used. It is
gpt-oss-specific and was rejected.

## 2. Was vLLM 0.23.0's Gemma-4 usable as a template?

**Yes — it is the right template and it was used as one.** vLLM 0.23.0 ships
`vllm/model_executor/models/gemma4.py` (1714 lines). It gave, directly:

| what | where it came from |
|---|---|
| per-layer `layer_types` -> `per_layer_sliding_window` on `Attention` | `Gemma4Attention.__init__` |
| per-layer-type RoPE via `get_rope(..., rope_parameters=dict)` | same |
| per-head q/k norm as `unflatten(-1, (heads, dim)) -> norm -> flatten(-2,-1)` | `Gemma4Attention.forward` |
| sandwich norms, `residual = h; ... ; h = residual + x` twice | `Gemma4DecoderLayer.forward` |
| `_get_text_config`, `packed_modules_mapping`, `language_model.` stripping in `load_weights` | `Gemma4ForCausalLM` |
| `extract_layer_index(prefix)`, `maybe_prefix` | `.utils` |

Three places where Gemma-4 could **not** be followed, all of them because
Muse-Glimmer differs:

1. **Logit transform order.** Gemma-4 does
   `LogitsProcessor(vocab_size, soft_cap=final_logit_softcapping)`.
   vLLM's `LogitsProcessor.forward` applies `soft_cap` and *then* `scale`
   (logits_processor.py:65-71). Muse-Glimmer needs `* output_multiplier`
   **then** the tanh cap (SPEC trap 7) — the opposite order, and there is no
   knob for it. Resolved by using `LogitsProcessor(vocab_size)` as a plain
   gather (TP all-gather + vocab de-padding) and doing the transform in
   `compute_logits`. Not a structural limitation; just not expressible through
   that class.
2. **Parameter-free q/k norm.** Gemma-4's q/k norms carry weights
   (`RMSNorm(head_dim, eps=...)`); Muse-Glimmer's do not exist in the
   checkpoint at all. vLLM's `RMSNorm(..., has_weight=False)` exists and is
   close, but it uses `rsqrt` where HF's `MuseGlimmerRMSNorm` deliberately
   writes `pow(mean_sq + eps, -0.5)` "to address compiler differences between
   Torch and JAX". Local module, ~8 lines, mirroring HF exactly.
3. **The attention gate.** Gemma-4 has nothing like it.

The four per-layer centred norms did **not** need a local module:
vLLM's `GemmaRMSNorm` is `(x.float() * rsqrt(var + eps)) * (1 + w)` cast back,
with a zeros-initialised weight — byte for byte
`MuseGlimmerTextCenteredRMSNorm`. It is used as-is.

## 3. What the model looks like

`third_party/tpu-inference/tpu_inference/models/vllm/muse_glimmer.py`, on
branch `agent/muse-glimmer-text`.

```
MuseGlimmerForCausalLM(nn.Module, SupportsLoRA)
  .model.embed_tokens          VocabParallelEmbedding
  .model.embed_norm            parameter-free RMSNorm  (NO sqrt(d) -- trap 9)
  .model.layers[i]
      .input_layernorm             GemmaRMSNorm  eps=1e-5   n*(1+w)
      .self_attn.qkv_proj          QKVParallelLinear
      .self_attn.q_norm/.k_norm    parameter-free, per head over head_dim
      .self_attn.rotary_emb        get_rope(...) or None if layer_rope_theta==0
      .self_attn.attn              Attention(per_layer_sliding_window=...)
      .self_attn.attn_gate_proj    ColumnParallelLinear
      .self_attn.o_proj            RowParallelLinear
      .post_attention_layernorm    GemmaRMSNorm  eps=1e-8   <- post_norm_eps
      .pre_feedforward_layernorm   GemmaRMSNorm  eps=1e-5
      .mlp.gate_up_proj            MergedColumnParallelLinear
      .mlp.down_proj               RowParallelLinear
      .post_feedforward_layernorm  GemmaRMSNorm  eps=1e-8   <- post_norm_eps
  .model.norm                  MuseGlimmerScaledRMSNorm  n*w, ones init (trap 10)
  .lm_head                     ParallelLMHead  (NOT tied -- trap 8)
```

Registration: `models/common/oot_registration.py` now maps
`MuseGlimmerForConditionalGeneration` ->
`tpu_inference.models.vllm.muse_glimmer:MuseGlimmerForCausalLM` instead of the
raising `JaxOnlyTextGenerationShim` (which stays in the file for any future
JAX-only architecture).

### Why that registration cannot regress the JAX path

`model_loader.get_model` branches on `MODEL_IMPL_TYPE` **before** any registry
is consulted:

* `flax_nnx` -> `get_flax_model` -> `_get_model_architecture` -> tpu-inference's
  own `_MODEL_REGISTRY`, which still holds the nnx
  `MuseGlimmerForConditionalGeneration`. vLLM's `ModelRegistry` is never read on
  this branch.
* `auto` -> `resolve_model_architecture`, which returns `flax_nnx` for every
  architecture outside `_VLLM_PREFERRED_ARCHITECTURES`. Muse-Glimmer was **not**
  added to that set, deliberately: the default stays the proven JAX path and the
  torch model is opt-in via `VLLM_MODEL_IMPL_TYPE=vllm`.
* `vllm` -> `get_vllm_model` -> vLLM's `ModelRegistry` -> the torch class.

One user-visible behaviour change, and it is an improvement in honesty rather
than a regression: `--enable-lora` under `flax_nnx` used to be rejected early by
vLLM (the shim did not declare `SupportsLoRA`); it now passes vLLM's check and
fails later at tpu-inference's own `LoRA is not enabled` assertion. Either way
`flax_nnx` + LoRA does not work, which is the entire reason this file exists.

## 4. The four places trouble was expected

**The attention gate.** `o = o * sigmoid(attn_gate_proj(x))` before `o_proj`,
with `x` the post-input-layernorm activation. Implemented as a plain
`ColumnParallelLinear` with `gather_output=False`, and that is sufficient under
TP: tpu-inference shards `ColumnParallelLinear` as
`P(None, ShardingAxisName.ATTN_HEAD)` (quantization/configs.py:60), i.e. the
output is split into contiguous blocks on the same mesh axis the attention
output and `o_proj`'s input use. Rank *r* owns q heads
`[r*H/tp, (r+1)*H/tp)` in all three, so the elementwise product needs no
collective. Name kept as `attn_gate_proj`; see section 6.

**2 KV heads under TP=4.** This turned out to be **already solved in-tree** and
needed nothing from the model. `tpu_inference/layers/vllm/custom_ops/linear.py`
registers `VllmQKVParallelLinear` as the OOT override of `QKVParallelLinear`; it
inflates `total_num_kv_heads` to the mesh TP size and tiles each loaded K/V
tensor with `repeat_interleave` — `[h0, h0, h1, h1]`, explicitly not `tile`'s
`[h0, h1, h0, h1]` — then marks the replica sub-axis replicated via `shard_map`.
The one thing the model must do is **not recompute the head counts**: it reads
`self.qkv_proj.num_heads` / `.num_kv_heads` back off the layer after
construction, because those are inflated and the naive
`total_num_kv_heads // tp_size` would produce the wrong `split()` sizes. See
section 8 — this remains the least-proven part.

**Parameter-free qk-norm.** Local `MuseGlimmerRMSNormNoScale`, applied on the
`[T, heads, head_dim]` view. Mirrors HF's `pow(m, -0.5)`.

**Per-layer attention config.** 39 sliding (window 2048) + 13 full, NoPE on the
full ones. Copied from Gemma-4's per-layer pattern, with one addition Gemma-4
does not need:

```python
self.attn = Attention(..., per_layer_sliding_window=sliding_window, ...)
self.attn.sliding_window = None
```

The window must reach the **kernel** but must not reach the **KV-cache spec**.
`PallasAttentionBackendImpl` copies `sliding_window` at construction and
forwards it to the ragged-paged-attention kernel; `runner/kv_cache_manager.py`
separately reads `attn_module.sliding_window` and, when it is not None, emits a
`SlidingWindowSpec` — a hybrid KV cache. That layout was implemented, measured
and **reverted** on the JAX path (E2E.md section 5: 4.5% capacity regression at
4096, and one sequence in five corrupted under concurrency). Clearing the
attribute after construction keeps a uniform single-group cache, matching the
JAX model exactly. `kv_cache_manager` uses the same idiom for its own
`disable_sliding_window` workaround, and `Attention.sliding_window` has exactly
one other consumer in vLLM 0.23.0 — `Attention.get_kv_cache_spec`, which
tpu-inference does not call.

## 5. CPU parity

Harness: `tpu/muse_glimmer/vllm_parity_ref.py` (HF side) +
`vllm_parity_check.py` (torch side), driven by `vllm_parity.sbatch` and
`vllm_parity_real.sbatch`. Never the login node.

Two venvs, because vllm 0.23.0 and `transformers @ main` cannot coexist —
vllm pins transformers 5.15.0 and `muse_glimmer` needs the unreleased tree, and
this repo's `[tool.uv] override-dependencies` silently overrides even
`--no-deps --force-reinstall` (use `UV_NO_CONFIG=1`). They talk through files:

```
/n/fs/vision-mix/sk7524/muse-parity/hfvenv   transformers 5.16.0.dev0, torch 2.13.0+cpu
/n/fs/vision-mix/sk7524/muse-vllm/venv       vllm 0.23.0, torch 2.11.0+cpu, jax 0.11.0 (cpu), torchax 0.0.11
```

### 5.1 Tiny random weights — **PASS** (job 3697921)

4 layers keeping `[S, S, S, F]`, `layer_rope_theta = [t, t, t, 0]`, window 8,
`qk_scale_factor` 3.87, `output_multiplier` 0.19611613513818404, softcap 20.0.
Two shapes: `tiny` (4 q heads / 1 kv head / head_dim 64) and `gqa` (8 q heads /
2 kv heads / head_dim 32, matching the 30B's GQA grouping). float32 throughout.

| variant | T | hidden max_abs | logits max_abs |
|---|---|---|---|
| tiny | 24 | **0.000e+00** | **0.000e+00** |
| tiny | 7 | 9.894e-06 | 1.520e-06 |
| tiny | 23 | **0.000e+00** | **0.000e+00** |
| tiny | 37 | **0.000e+00** | **0.000e+00** |
| tiny | 129 | 4.888e-06 | 6.557e-07 |
| gqa | 24 | **0.000e+00** | **0.000e+00** |
| gqa | 7 | 8.583e-06 | 1.296e-06 |
| gqa | 23 | **0.000e+00** | **0.000e+00** |
| gqa | 37 | **0.000e+00** | **0.000e+00** |
| gqa | 129 | 3.576e-06 | 5.364e-07 |

Bit-identical at three of five lengths and `<= 9.9e-06` at the other two —
against the JAX port's 1.48e-05 on the same config. Weight loading:
**51 checkpoint tensors -> 39 parameters, 0 unloaded**, on a checkpoint written
with the 30B's own key layout (`model.language_model.*` + a top-level
`lm_head.weight`).

Structural probes, both variants:

```
layer 0..2 (sliding)      last-pos delta from token 0: 0.000e+00   blind past the window
layer 3    (full)         last-pos delta from token 0: 1.357e+00   sees everything
layer 0..2 (sliding/rope) delta under stride-3 positions: 4.8-6.3  RoPE is live
layer 3    (full/NoPE)    delta under stride-3 positions: 0.000e+00 provably NoPE
```

The NoPE probe changes position *differences*, not a constant shift — a shift
proves nothing because RoPE is relative (SPEC trap 12).

### 5.2 Real weights — **PASS, 3949/3949** (job 3697934)

`vllm_parity_real.sbatch`, 30B in float32 on CPU, compared against the recorded
HF dump `runs/muse_glimmer/hf_ref.npz` (transformers 5.16.0.dev0, float32,
eager) rather than recomputing it, so only one 30B copy is ever live.
neu311, 16 CPU / 160 GB, **39m09s**, MaxRSS 158.9 GiB.

Weight loading: **1436 checkpoint tensors -> 471 parameters, 0 unloaded** — the
vision stack dropped, `model.language_model.*` flattened, `self_attn.gate_proj`
renamed to `attn_gate_proj`, q/k/v fused into `qkv_proj` and gate/up into
`gate_up_proj`, and nothing left over on either side.

`max|lm_head - embed| = 3.0950`, i.e. **untied**, reproducing the 3.09 recorded
for the JAX gate. A tied `lm_head` would still produce fluent text; this is the
only cheap way to catch it (SPEC trap 8).

| prompt | tokens | teacher-forced argmax | top-8 set | logit rows max-abs | max-rel (>1% peak) |
|---|---|---|---|---|---|
| p1_tiny | 6 | **6/6** | 6/6 | **0.000e+00** | 0.000e+00 |
| p2_odd | 35 | **35/35** | 35/35 | 2.158e-05 | 8.272e-05 |
| p3_mid | 267 | **267/267** | 267/267 | 4.387e-05 | 1.240e-04 |
| p4_long_sliding | **3609** | **3609/3609** | 3609/3609 | 2.052e-03 | 7.745e-04 |
| p5_code | 32 | **32/32** | 32/32 | 2.241e-05 | 8.026e-05 |

**3949 of 3949 positions agree on argmax** — the same bar the JAX port hit, on
the same prompts, against the same reference dump. Final hidden states, where
the reference recorded them: max-abs 0.000e+00 (p1), 3.433e-05 (p2, p5).

p4 at 3609 tokens is 1.76x the 2048 sliding window, so both the 39 sliding and
the 13 full-attention layers are genuinely exercised, including the
sliding/full divergence that only appears past the window. Its larger absolute
error is the expected float32 accumulation over a longer reduction — the JAX
port measured 1.19e-03 on the same prompt, this one 2.05e-03, and both agree
with HF on every single argmax and every single top-8 set.

A mis-mapped tensor, a `tile`-for-`repeat_interleave` swap in a fused
projection, a wrong norm flavour or a swapped logit-transform order would all
show up here as argmax disagreements. None do.

### 5.2b Registry additivity — **PASS** (job 3697973)

"Additive" is a claim, so `--mode registry` checks it rather than assuming it:

```
vLLM ModelRegistry[MuseGlimmerForConditionalGeneration]
    -> tpu_inference.models.vllm.muse_glimmer.MuseGlimmerForCausalLM
tpu-inference _MODEL_REGISTRY[MuseGlimmerForConditionalGeneration]
    -> tpu_inference.models.jax.muse_glimmer.MuseGlimmerForConditionalGeneration
MuseGlimmerForConditionalGeneration in _VLLM_PREFERRED_ARCHITECTURES: False
resolve_model_architecture('auto') -> 'flax_nnx'
```

Both branches resolve to their own implementation and the default is unchanged.

### 5.3 What the CPU gate deliberately does not cover

The gate builds the model from **vLLM's stock layer classes**: tpu-inference's
OOT overrides are popped from `op_registry_oot` first, because every one of them
reaches for `vllm_config.quant_config.mesh` and shards onto a JAX device mesh
that does not exist on CPU. Consequences, all of them on-slice-only questions:

* **KV-head replication at TP=4 is not exercised.** With torch TP=1 and no mesh,
  `VllmQKVParallelLinear`'s inflation path is dead code. This is exactly the
  `repeat_interleave`-vs-`tile` hazard, and tiny weights could not catch it
  anyway.
* Weight sharding, the all-reduces, and the ragged-paged-attention kernel are
  not run. The gate substitutes an eager masked softmax for the paged
  `Attention`, installed by replacing `self_attn.attn` **after** construction —
  so the real `PallasAttentionBackendImpl` is still built, and the window/scale/
  head counts the harness asserts on are read straight off `impl`.

## 6. LoRA

### 6.1 Adapter count — **NON-ZERO** (job 3697928)

Counted on CPU by calling vLLM's own `create_lora_manager` — the same call
`VllmModelWrapper.load_weights` makes through
`tpu_inference.lora.lora_manager` — and counting `BaseLayerWithLoRA` instances
in the returned manager's model. Punica wrapper:
`tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU`, i.e. the real one.

```
supports_lora(model) = True
vLLM-discovered LoRA-capable suffixes: ['attn_gate_proj', 'down_proj',
                                        'gate_up_proj', 'o_proj', 'qkv_proj']

LoRA-wrapped modules: 20        (4-layer config -> 5 per layer)
   4  MergedQKVParallelLinearWithLoRA      qkv_proj
   4  RowParallelLinearWithLoRA            o_proj
   4  ColumnParallelLinearWithLoRA         attn_gate_proj
   4  MergedColumnParallelLinearWithLoRA   gate_up_proj
   4  RowParallelLinearWithLoRA            down_proj
```

At 52 layers that is **260 wrapped modules**. Compare the JAX/MaxText side,
which gets 56 adapter arrays on `self_attention` + `mlp` only and cannot adapt
the attention gate at all.

The zero-adapter failure mode is the one that matters — a name mismatch injects
nothing and still produces a plausible loss curve — so two things are asserted
explicitly rather than assumed:

* `attn_gate_proj` is wrapped once per layer (4/4);
* `is_supported_lora_module("...self_attn.attn_gate_proj", ["gate_proj"])` is
  **False**. vLLM's matcher is `re.match(r".*\.{target}$", name)`, which needs a
  literal dot before the target; `attn_gate_proj` has `_`. The MLP's
  `gate_proj` therefore cannot latch onto the attention gate, which is the whole
  reason for the name.

**The name is load-bearing in the other direction too.** An adapter trained
against raw HF module names — where the attention gate *is* `self_attn.gate_proj`
— would be routed by `packed_modules_mapping` into `gate_up_proj` and fail on
shape. Adapters produced by this stack use `attn_gate_proj` and are fine.
`make_lora_adapter.py` emits that layout.

### 6.2 On-slice LoRA smoke

<!-- LORA_SLICE -->

## 7. On-slice results

<!-- SLICE_RESULTS -->

## 8. What remains unproven

Ranked by how much it would cost to be wrong.

1. **KV-head replication at TP=4 on real weights.** 2 KV heads under TP=4 goes
   through `VllmQKVParallelLinear`'s `repeat_interleave` inflation. Nothing on
   CPU exercises it (torch TP=1, no mesh), and tiny random weights could not
   catch a `tile`-vs-`repeat_interleave` swap even if they did — this is
   exactly the trap the JAX port hit. Only the on-slice greedy comparison in
   section 7 tests it, and only at TP=4.
2. **TP != 4.** Untested, as on the JAX side. The 2x TP=2 split that E2E.md
   section 7 recommends for throughput is untested on this model.
3. **Concurrency.** The JAX path's most expensive lesson (E2E.md section 5.6)
   is that single-stream boundary testing cannot catch paged-KV bugs: five
   prompts fired *concurrently* broke one sequence that was byte-exact at batch
   1. The torch path uses the same uniform KV spec and the same kernel, so
   there is no specific reason to expect a difference, but "no reason to
   expect" is not a measurement.
4. **Context past what was decoded here.** The JAX path serves 32768. Nothing
   here probes the torch path past the lengths in section 7.
5. **LoRA under load, and LoRA correctness.** The smoke proves an adapter
   attaches, changes the output, swaps, and detaches. It does not prove the
   numerics of the adapted forward against a reference, nor multi-adapter
   serving (`--max-loras 1`), nor `attn_gate_proj` adapters specifically
   producing the *right* delta.
6. **Throughput and memory relative to the JAX path.** Not measured. The torch
   path runs the same kernels through torchax but with vLLM's linear layers
   and an extra `attn_gate_proj` matmul that the JAX model also has; whether it
   costs tokens/s is an open question, and it matters because the JAX path
   remains the default.
7. **Quantization, multi-host, speculative decoding, vision.** Out of scope,
   as on the JAX side.

## Reproducing

```bash
# 1. build the two CPU venvs (once, needs network)
sbatch /n/fs/vision-mix/sk7524/muse-vllm/build_venv.sbatch
sbatch /n/fs/vision-mix/sk7524/muse-vllm/add_jax.sbatch

# 2. tiny parity (both shapes) + the LoRA adapter count -- CPU, ~4 min
sbatch tpu/muse_glimmer/vllm_parity.sbatch tiny

# 3. real-weight parity vs the recorded HF dump -- CPU, 160 GB
sbatch tpu/muse_glimmer/vllm_parity_real.sbatch

# 4. ONE spot v5p-8: serve under MODEL_IMPL_TYPE=vllm, greedy-decode the five
#    recorded prompts, LoRA load/unload smoke, then re-check flax_nnx
sbatch tpu/muse_glimmer/vllm_impl_tpu.sbatch
```

### QR hygiene

`vllm_impl_tpu.sh` copies the QR lifecycle from `followup_tpu.sh` verbatim:
TERM/INT/HUP trapped explicitly (a bare `EXIT` trap does not fire on an
untrapped SIGTERM here), the delete issued `setsid nohup` so it outlives the
shell, re-issued and re-verified until `describe` returns empty, and swept
across **all three** zones because the landing phase rotates. The slurm time
limit is longer than the script's own `CAP_SEC` so the script's cleanup, not a
slurm kill, ends the run.
