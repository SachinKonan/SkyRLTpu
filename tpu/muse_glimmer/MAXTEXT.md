# Muse-Glimmer-30B on MaxText (training half)

Text-only MaxText/tunix support for `meta-models/Muse-Glimmer-30B`, with numerical
parity against the HF reference. Companion to `SPEC.md` (the contract) and to the
serving-side port in `third_party/tpu-inference` (a different agent's tree — nothing
here touches it).

**Status: parity PASS.** Worst error across every case, both layouts:
`max_abs 9.89e-06` (target `< 1e-4`), `rel_to_scale 7.48e-06` (target `< 1e-3`).
Slurm job **3686711**, log `/n/fs/vision-mix/sk7524/muse-parity/logs/parity-3686711.out`.

---

## 1. Where the code lives

The MaxText model code is **not** in this repo. It is a clone of the upstream MaxText
that the pinned `maxtext` PyPI package is built from:

| | |
|---|---|
| clone | `/n/fs/vision-mix/sk7524/maxtext-muse` |
| pushed to | `SachinKonan/maxtext` branch `muse-glimmer` @ `4f65ba509` (remote `fork`) — DONE 2026-08-12 |
| upstream | `https://github.com/AI-Hypercomputer/maxtext` |
| base | tag `maxtext-v0.2.3` = `3f36aef23439dd5875b48cbf294bbcba1996a726` (the released `maxtext==0.2.3` the sweep installs today) |
| branch | `muse-glimmer` |
| commit | `4f65ba509` |

**Nothing has been pushed.** To use this from `start_colocated_vllm_tinker.sh` the branch
must first be pushed to a fork you control — that is your call, not mine. See §5.

In *this* repo only `tpu/muse_glimmer/` gained files (this doc, the parity harness, the
converter smoke, the sizing script, the sbatch wrappers). No pins, no sweep files, no
`third_party/**` were touched, and `TUNIX_MAXTEXT_PIP_SPEC`'s default is unchanged.

## 2. What was added to MaxText

New files:
- `src/maxtext/models/muse_glimmer.py` — `MuseGlimmerDecoderLayer` and
  `MuseGlimmerScannableBlock` (+ their `*ToLinen` wrappers). One scanned block is one
  `[S,S,S,F]` cycle, so 52 layers = 13 scan steps.
- `src/maxtext/configs/models/muse-glimmer-30b.yml` — the production config.
- `src/maxtext/configs/models/muse-glimmer-tiny.yml` — same constants, tiny shapes, for
  the parity harness.

Touched files (all additive, all default-off for other models):
| file | change |
|---|---|
| `common/common_types.py` | `DecoderBlockType.MUSE_GLIMMER` |
| `configs/types.py` | `ModelName` += the two names; new fields `post_norm_layer_epsilon`, `qk_scale_factor`, `logits_output_multiplier` |
| `configs/base.yml` | defaults for those three (`0.0` / `1.0` / `1.0`, i.e. no-ops) |
| `layers/attentions.py` | generic `use_attn_output_gate` (default `False`) |
| `layers/nnx_decoders.py` | layer_map + `get_norm_layer` + unscanned layer kwargs + embedding norm + output head |
| `layers/decoders.py` | the same five sites on the linen path (the converter builds the linen tree) |
| `utils/globals.py` | `HF_IDS` |
| `checkpoint_conversion/utils/hf_model_configs.py` | `HF_MODEL_CONFIGS` |
| `checkpoint_conversion/utils/param_mapping.py` | `PARAM_MAPPING` + `HOOK_FNS` |

### Attention: MaxText's own kernel path, not a hand-rolled one

The layer builds a stock `attentions.Attention` and drives it with the knobs
gemma-2/3/4 and olmo-3 already use:

- `attention_type=LOCAL_SLIDING | GLOBAL` per sub-layer, from
  `MUSE_GLIMMER_ATTENTION_PATTERN`, plus `sliding_window_size=2048`.
- **Which kernel that is, confirmed by reading the code:** `attention: autoselected`
  (the base default) dispatches in `attention_op.py:940-990` to
  `apply_attention_dot` when `model_mode == AUTOREGRESSIVE` or `length < 128`, and
  otherwise to `AttentionOp.tpu_flash_attention`, which is
  `jax.experimental.pallas.ops.tpu.splash_attention` (`attention_op.py:32`, kernel built
  at `:1336`). The sliding window is expressed to that kernel as
  `splash_attention_mask.LocalMask(window_size=(W, W))` AND-ed with `CausalMask`
  (`attention_op.py:1277-1283`). So on TPU at our training lengths Muse-Glimmer runs on
  **splash**, on exactly the code path MaxText's other local/global hybrids run on.
  On CPU (and for `length < 128`) it falls back to the dot-product path, whose sliding
  mask is `col > row - W` (`attention_op.py:715-724`). Nothing model-specific was added
  to either.
- **NoPE** on the full layers via the existing `is_nope_layer` — it only skips the rope
  step (`attentions.py:1166-1173`); nothing changes kernel-side.
- **`qk_scale_factor` (3.87)** is folded with `head_dim**-0.5` into
  `query_pre_attn_scalar` (`muse_glimmer.get_query_pre_attn_scalar`). This is not a
  shortcut: MaxText's kernels all run with `sm_scale=1.0` (`attention_op.py:1028, 1109,
  1646, 1705, 1716`) and expect `1/sqrt(d)` folded into q. Both factors are scalars on q
  and RoPE is linear in q, so applying their product after RoPE is the same computation.
  Neither is dropped and neither is applied twice — the parity numbers are the proof.
- **The sigmoid gate and `o_proj` are outside the kernel.** `use_attn_output_gate=True`
  adds an `[emb] -> [heads, head_dim]` projection of the *attention input* whose sigmoid
  multiplies the attention output immediately before `out_projection`
  (`attentions.py`, right after the qwen3-hybrid gate). Same logical axes as the query
  projection, so it shards identically.

> One upstream oddity worth knowing, **not introduced here and not specific to this
> model**: the dot-product sliding mask admits `W` keys (`col > row - W`) while splash's
> `LocalMask(window_size=(W, W))` is a `<=`-style bound, so the two paths can differ by
> one key at the window edge. That affects gemma-2/3/4 and olmo-3 identically. The
> CPU parity harness necessarily tests the dot-product path; a TPU-side check of the
> splash path is listed under "what remains".

### The three norm flavours

Per `SPEC.md` (as corrected): parameter-free, scaled `n*w`, and centred `(1+w)`.

- centred `(1+w)` — the four per-layer norms — `RMSNorm(scale_init=zeros, scale_offset=1.0)`.
  The stored zero-centred HF tensor is copied through unchanged; MaxText adds the 1.
- scaled `n*w` — **the final `model.norm` only** — a plain `RMSNorm`
  (`scale_offset=0`), registered via `get_norm_layer`. I found this independently before
  the spec was corrected: the reference builds it as `MuseGlimmerRMSNorm(hidden_size,
  eps)` with `with_scale=True`, ones-init.
- parameter-free — the qk norm (`use_qk_norm: true` + `qk_norm_with_scale: false`, giving
  a per-head `RMSNorm` over `head_dim` with no scale) and the embedding norm
  (`muse_glimmer.normed_embedding`, called from both decoders' `_apply_embedding`).
  There is **no** `sqrt(hidden_size)` on the embedding, and it cannot be folded into the
  table anyway — it is a per-token nonlinearity and the raw table is still what the
  (untied) head would need.

### Output head

`lm_head` is **not** tied. `text_config.tie_word_embeddings` is `false` and the released
`model.safetensors.index.json` carries a separate `lm_head.weight` in shard 2; the
parameter count corroborates it (27.855B only closes with a separate head). So the config
sets `logits_via_embedding: false` and the converter maps `lm_head.weight` onto
`decoder/logits_dense/kernel`. `apply_output_head`'s `logits_dense` branch now applies
`logits_output_multiplier` and *then* the tanh softcap (previously the softcap was only
reachable through the `logits_via_embedding` branch — no shipped model hits the changed
code, all seven models that set `final_logits_soft_cap` are `logits_via_embedding: true`).

## 3. Running it here

```bash
# One-time: build the two isolated CPU venvs (never on the login node).
sbatch tpu/muse_glimmer/build_parity_venvs.sbatch   # jobs 3686552 + 3686628
sbatch tpu/muse_glimmer/finish_mtvenv.sbatch        # MaxText declares no deps; this
                                                    # imports in a loop and installs
                                                    # whatever is missing next
# Parity + converter smoke + sizing, all in one job:
sbatch tpu/muse_glimmer/run_maxtext_parity.sbatch   # job 3686711 -> ALL CHECKS PASSED
```

Venvs live at `/n/fs/vision-mix/sk7524/muse-parity/{hfvenv,mtvenv}`. `hfvenv` holds
`transformers 5.16.0.dev0` from git main (Muse-Glimmer is unreleased) and never comes
near the project environment — the live RL sweep's transformers pin is untouched.

### Training launch (once the fork is pushed)

```bash
TINKER_BACKEND=tunix \
TUNIX_MODEL_SOURCE=maxtext \
TUNIX_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba509" \
TUNIX_MAXTEXT_MODEL_NAME=muse-glimmer-30b \
TUNIX_MAXTEXT_KWARGS='{"remat_policy":"full","ici_fsdp_parallelism":4}' \
TUNIX_MAX_TARGET_LENGTH=24576 \
TUNIX_FLCE_TILE_SIZE=2048 \
  bash tpu/start_colocated_vllm_tinker.sh ...
```

`TUNIX_MAXTEXT_PIP_SPEC`'s default in `start_colocated_vllm_tinker.sh` is deliberately
**not** changed — that script is live for the running sweep. Override it per-launch.

The backend then converts HF -> orbax on first start by shelling out to:

```bash
python -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=muse-glimmer-30b base_output_directory=<cache>/muse-glimmer-30b \
    scan_layers=True use_multimodal=false skip_jax_distributed_system=True \
    --lazy_load_tensors=True \
    checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True
```

(add `--hf_model_path <dir>` to convert from a local HF directory — note the `--`, it is
an argparse flag, not a pyconfig key, and without it pyconfig rejects it). `HF_IDS` maps
`muse-glimmer-30b -> meta-models/Muse-Glimmer-30B`, so no extra plumbing is needed.
`tpu/muse_glimmer/convert_smoke.py` runs that exact command end to end on tiny weights
and loads the result back through `model_creation_utils.from_pretrained`.

## 4. Verification

### Parity harness

Two stages in two isolated venvs, so torch and jax never share a process:

1. `parity_hf_dump.py` (hfvenv) builds a tiny random-weight `MuseGlimmerTextModel`
   (4 layers `[S,S,S,F]`, hidden 256, 4 heads, 1 kv head, head_dim 64, vocab 512,
   window 8; every non-shape constant identical to the 30B) plus an untied `lm_head`,
   and dumps the weights under **the exact HF names from the released index**
   (`model.language_model.*` + `lm_head.weight`).
2. `parity_maxtext_check.py` (mtvenv) pushes those tensors through MaxText's **real**
   converter functions — `PARAM_MAPPING`, `HOOK_FNS`, `validate_and_filter_param_map_keys`,
   `apply_hook_fns`, `_build_single_axis_stacked_tensor` — and compares.

Weights are initialised so each trap is observable: the embedding gets σ=1 (so the
parameter-free embedding norm bites), the four centred norms get zero-mean σ=0.1, and the
final norm gets **one**-mean σ=0.1 — so mixing up `(1+w)` and `n*w` fails loudly in either
direction.

Sequence lengths `7, 24, 37, 100, 129`: deliberately not powers of two and not multiples
of the splash query block (512), spanning both sides of the window and of the
`length < 128` kernel-dispatch boundary. Plus a `seq=37` sequence padded into a 128-long
row with `decoder_segment_ids`, which is what the tunix backend actually feeds.

Results (job 3686711), worst over both `scan_layers=True` and `False`:

| metric | worst | target |
|---|---|---|
| `max_abs` (raw uncapped logits) | **9.89e-06** | < 1e-4 |
| `rel_to_scale` = max_abs / max\|ref\| | **7.48e-06** | < 1e-3 |
| `rel_significant` (entries > 1% of peak) | 3.09e-04 | — |
| single-layer vs HF (sliding / full) | 4.77e-06 / 6.86e-06 | < 1e-4 |
| orbax round-trip via `to_maxtext` + `from_pretrained` | **8.01e-07** | < 1e-4 |

The tanh softcap *compresses* error, so capped logits alone would be a soft gate; the
harness therefore also re-runs with `final_logits_soft_cap=0` and
`logits_output_multiplier=1` (`override_model_config=True`) and gates on those raw
logits. Unrestricted element-wise relative error is not reported on purpose — logits
contain near-zero entries and it blows up for a numerically perfect port.

### Structural assertions (all PASS)

- sliding layer: a perturbation at distance `>= window` changes the output by exactly
  `0.0`; inside the window it changes it by `8.0` (probe not vacuous); **and the HF
  reference shows the identical boundary**.
- full layer: sees a perturbation arbitrarily far back (`min |delta| = 1.2e-01`).
- full layer is **rope-free**: `max |delta| = 0.0` under a stride-3 position ramp. Per
  `SPEC.md` trap 12 a *constant* shift proves nothing (RoPE is relative), so the probe
  changes position *differences*; the sliding layer moves by `5.5` under the same change,
  which is what makes the assertion non-vacuous. Per trap 11 every probe passes an
  explicit index-based mask to the HF layer rather than letting `create_causal_mask`
  derive packed-sequence boundaries from `position_ids` — MaxText's masks are
  `broadcasted_iota` over indices, so the two sides agree by construction.
- converter mapping covers **exactly** the model's 51 parameters: 0 unmapped, 0 stale,
  in both layouts.
- every parameter carries a logical sharding annotation.
- scanned and unscanned layouts both match HF and each other.

### LoRA targeting (verified, not assumed)

tunix targets LoRA with `_MAXTEXT_ATTN_REGEX` / `_MAXTEXT_MLP_REGEX` in
`skyrl/backends/tunix_backend.py`. Checked against the actual module paths this model
produces:

```
ATTN  decoder/layers/layers_0/self_attention/{query,key,value,out}     <- adapters
MLP   decoder/layers/layers_0/mlp/{wi_0,wi_1,wo}                       <- adapters
----  decoder/layers/layers_0/self_attention/attn_gate                 <- no adapter
```

and the same for the unscanned `decoder/layers_{0..3}/...` layout. That is why the
attention module attribute is named **`self_attention`** and not `attention` (olmo-3, the
structural template, uses `attention` and would silently get zero adapters here).

**The `gate_proj` collision is real and is handled by construction.** This model has two
tensors whose HF names end in `gate_proj`:

| HF name | MaxText param | LoRA |
|---|---|---|
| `self_attn.gate_proj` (sigmoid attention gate) | `self_attention/attn_gate/kernel` | no |
| `mlp.gate_proj` (SwiGLU) | `mlp/wi_0/kernel` | yes |

MaxText renames the MLP one to `wi_0` and the converter maps them from full dotted paths,
so nothing can match on the last component alone (which is how the serving side got bitten
and why they renamed theirs to `attn_gate_proj`). Adapting the attention gate too would
need a one-line regex change (`(query|key|value|out|attn_gate)`) plus a
`_MAXTEXT_PROJ_TO_HF[("self_attention","attn_gate")] = ("self_attn","gate_proj")` entry for
PEFT export — **not done**, because `tunix_backend.py` is outside my mandate and no
existing model adapts a gate.

### Sharding / memory at the real training shape

From `sizing.py` (job 3686711, `model_name=muse-glimmer-30b`, `ici_fsdp_parallelism=4`,
`scan_layers=True`, S=24576):

```
PARAMETERS  : 27,854,780,928  (27.855 B)
bf16 total  : 51.88 GiB
bf16 /chip  : 12.97 GiB      (4-way FSDP)
all parameters carry a logical sharding annotation
```

- Comparable to the Qwen3.5-27B currently on these slices (~13.5 GB/chip), so the weight
  side of the fb arena is a known quantity.
- Largest params are the scanned MLP kernels, `(6656, 13, 19968)` with axes
  `('embed', 'layers', 'mlp')` — `embed` maps to `fsdp` in `logical_axis_rules`, so they
  shard. The new `attn_gate` carries `('embed', 'q_heads', 'kv')`, identical to the query
  projection.
- **The attention output gate is 1,417,674,752 params = 5.1% of the model (2.64 GiB
  bf16).** Budget for it: a same-shape model without the gate would be ~26.4B.
- Activation floor at S=24576 under `remat_policy=full`: 13 blocks × 24576 × 6656 × 2B =
  3.96 GiB (≈0.99 GiB/chip). The real peak also holds one recomputed block plus the
  `[B,S,V]` logits — **18.50 GiB in f32 at S=24576, V=202048**, which is the binding term.
  Use `TUNIX_FLCE_TILE_SIZE` (needs `maxtext_kwargs.num_vocab_tiling > 1`), as the sweep
  already does for gpt-oss.

## 5. What remains before a real training run

1. ~~**Push the fork.**~~ **DONE 2026-08-12** — branch `muse-glimmer` is on
   `SachinKonan/maxtext` at `4f65ba509`; use the pip spec above verbatim.
2. **Real-weight spot check.** Parity is on tiny random weights. Converting a real shard
   and checking a handful of tokens needs the 60 GB download, which I did not do. The
   converter is name-for-name against the real `model.safetensors.index.json`, and the
   CLI round-trip is proven, so this is a confirmation rather than a risk.
3. **TPU-side numeric check of the splash path.** CPU can only exercise dot-product.
   Worth one v5p smoke comparing splash vs dot-product outputs at a length that straddles
   the query block size, which would also settle the `LocalMask` window-edge question
   above. Needs a QR — out of scope for this phase.
4. **Tokenizer.** `HF_IDS` points at `meta-models/Muse-Glimmer-30B`; the tunix backend
   loads the tokenizer from `base_model`, so the repo has to be reachable (or a local
   path supplied) at launch. Untested here — everything ran offline.
5. **LoRA smoke on TPU.** The regexes are verified to match the right module paths, but no
   adapter has actually been injected and stepped. First bring-up should log the LoRA
   param count and assert it is non-zero (a zero count is the classic silent failure).
6. **Vision tower, DFlash drafter, tool/reasoning parsers** — out of scope, per `SPEC.md`.

## 6. Where SPEC.md was wrong

Both found independently here before the corrected spec arrived, and both are now fixed
in `SPEC.md`:

1. **The final `model.norm` is not a centred norm.** The original forward pass applied
   `CenteredRMSNorm` with `(1 + w)` to `model.norm`; the reference builds it as
   `MuseGlimmerRMSNorm(hidden_size, eps=rms_norm_eps)` — `with_scale=True`, ones-init,
   plain `n * w`. Would have silently corrupted every output.
2. **Trap 8 was backwards: `lm_head` is not tied.** `tie_word_embeddings` is `false` and
   the index ships a separate tensor; the class-level `_tied_weights_keys` is gated on the
   config flag and does not fire. The converter maps `lm_head.weight` explicitly.

Everything else in the spec held up. The one thing worth adding for a future reader:
`output_multiplier` is not a magic constant, it is `1/sqrt(hidden_size/256)`
(= `1/sqrt(26)` at hidden 6656), per the docstring on `MuseGlimmerTextConfig`. The tiny
parity config keeps the 30B's literal value on both sides rather than recomputing it, so
the multiplier is genuinely exercised.
