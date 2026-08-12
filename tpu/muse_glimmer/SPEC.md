# Muse-Glimmer-30B on TPU — text-only port spec

Goal: JAX-native **text-only** support for `meta-models/Muse-Glimmer-30B`, for BOTH
serving (tpu-inference) and training (MaxText/tunix), with **numerical parity against
the HF reference**. No vision tower.

Reference implementation (ground truth, fetched 2026-08-12):
`transformers` main → `src/transformers/models/muse_glimmer/modeling_muse_glimmer.py`
(local copies in the session scratchpad: `modeling_muse_glimmer.py`, `modular_muse_glimmer.py`).
The model ships **inside transformers 5.15.0.dev0** — the HF repo contains no `.py`,
so there is no `trust_remote_code` path. Weight map: `model.safetensors.index.json`.

## Status of upstream support (checked 2026-08-12)
- vLLM support is **PR #51655, OPEN, not merged** → in no released wheel.
- The vLLM recipe requires **0.27.0+**; this repo pins **vllm==0.23.0**.
- No TPU/JAX mention in the recipe or the PR. `tpu_inference/models/jax/` has no
  muse_glimmer and the loader registry does not know the arch.
⇒ The JAX-native path does not depend on any of that landing. Chosen deliberately.

## Config (text_config), verbatim values
hidden_size 6656 · num_hidden_layers 52 · num_attention_heads 32 · num_key_value_heads 2
head_dim 128 · intermediate_size 19968 · hidden_activation "silu" · attention_bias false
max_position_embeddings 131072 · sliding_window 2048
rms_norm_eps 1e-5 · **post_norm_eps 1e-8** (different! post norms only)
qk_scale_factor 3.87 · final_logit_softcapping 20.0 · output_multiplier 0.19611613513818404
layer_types: `[sliding, sliding, sliding, full]` × 13 → 39 sliding + 13 full
layer_rope_theta: 500000.0 on sliding layers, **0 on full layers** (= NoPE)
rope_parameters.rope_theta 500000.0, rope_type "default"
`tie_word_embeddings` **false** → `lm_head.weight` is a **separate** tensor, NOT tied to
`model.language_model.embed_tokens.weight`. See trap 8; measured on the real 30B
checkpoint, `max|lm_head − embed| = 3.09`.

## THREE norm flavours (get these wrong and nothing else matters)

> CORRECTED 2026-08-12: this section originally listed two flavours and the
> forward pass below applied `CenteredRMSNorm` to the final `model.norm`.
> That is wrong — the reference builds `model.norm` as
> `MuseGlimmerRMSNorm(hidden_size, eps=rms_norm_eps)`, i.e. `with_scale=True`,
> ones-init, plain `n * w`. Verified against the reference and by parity.

```
RMSNorm_noscale(x, eps):        # MuseGlimmerRMSNorm(with_scale=False) — NO parameters
    xf = x.float()
    return (xf * pow(mean(xf^2, -1, keepdim) + eps, -0.5)).astype(x.dtype)

RMSNorm_scaled(x, w, eps):      # MuseGlimmerRMSNorm(with_scale=True) — weight init ONES
    xf = x.float()                                     # THE FINAL model.norm ONLY
    n  = xf * pow(mean(xf^2, -1, keepdim) + eps, -0.5)
    return (n * w.float()).astype(x.dtype)             # <-- n * w, NOT (1 + w)

CenteredRMSNorm(x, w, eps):     # MuseGlimmerTextCenteredRMSNorm — weight init ZEROS
    xf = x.float()                                     # the four per-layer norms
    n  = xf * rsqrt(mean(xf^2, -1, keepdim) + eps)
    return (n * (1.0 + w.float())).astype(x.dtype)     # <-- (1 + w), Gemma convention
```

Note every flavour reduces in **float32** regardless of the activation dtype
(`hidden_states.float()`), so the model carries float32 precision even in
float64. That sets the achievable parity floor — see the acceptance section.
HF deliberately uses `pow(m, -0.5)` over `rsqrt` "to address compiler differences
between Torch and JAX" — mirror that choice if parity drifts.

## Forward (exact)

```
h = RMSNorm_noscale(embed_tokens[input_ids], eps=norm_eps)   # NormedEmbedding.
                                                             # NOTE: no sqrt(d) scaling.
for i in range(52):
    residual = h
    x = CenteredRMSNorm(h, layers[i].input_layernorm.weight, eps=rms_norm_eps)

    q = q_proj(x).reshape(..., 32, 128)
    k = k_proj(x).reshape(...,  2, 128)
    v = v_proj(x).reshape(...,  2, 128)
    q = RMSNorm_noscale(q, eps=rms_norm_eps) * qk_scale_factor   # scale on q ONLY
    k = RMSNorm_noscale(k, eps=rms_norm_eps)                     # per-head, over head_dim
    if layer_rope_theta[i] != 0:                                 # full layers: NO rope
        q, k = apply_rope(q, k, theta=500000.0)
    o = attention(q, k, v,
                  scaling = head_dim ** -0.5,                    # 128**-0.5, separate from 3.87
                  causal  = True,
                  window  = 2048 if layer_types[i]=="sliding_attention" else None)
    o = o.reshape(..., 32*128)
    o = o * sigmoid(gate_proj(x))          # gate from x = POST input_layernorm, not from o
    o = o_proj(o)
    o = CenteredRMSNorm(o, layers[i].post_attention_layernorm.weight, eps=post_norm_eps)
    h = residual + o

    residual = h
    y = CenteredRMSNorm(h, layers[i].pre_feedforward_layernorm.weight, eps=rms_norm_eps)
    y = mlp.down_proj(silu(mlp.gate_proj(y)) * mlp.up_proj(y))
    y = CenteredRMSNorm(y, layers[i].post_feedforward_layernorm.weight, eps=post_norm_eps)
    h = residual + y

h = RMSNorm_scaled(h, model.norm.weight, eps=rms_norm_eps)   # CORRECTED: n * w
logits = lm_head(h)                              # NOT tied — see trap 8
logits = logits * output_multiplier              # 0.19611613513818404
logits = final_logit_softcapping * tanh(logits / final_logit_softcapping)   # T=20
```

## Parity traps (each one silently produces plausible-but-wrong output)
1. `(1.0 + w)` in CenteredRMSNorm. Weights are stored zero-centred; `n * w` gives ~0.
2. `qk_scale_factor` multiplies **q only**, and is *separate from* the `head_dim**-0.5`
   attention scaling. Effective q scale = 3.87 × 0.0884.
3. qk_norm is **parameter-free** (absent from the checkpoint) and per-head over head_dim.
4. Attention gate is `sigmoid(gate_proj(x))` where `x` is the **input_layernorm output**,
   NOT the attention output. Elementwise over `heads*head_dim`, applied **before** o_proj.
5. **NoPE on the 13 full-attention layers** (`layer_rope_theta[i] == 0`).
6. `post_norm_eps = 1e-8` on the two POST norms; `rms_norm_eps = 1e-5` everywhere else.
7. Logits: multiply by `output_multiplier` **then** softcap. Order matters.
8. ~~`lm_head` is tied to the embedding matrix.~~ **WRONG — CORRECTED 2026-08-12.**
   `text_config.tie_word_embeddings` is `false`, the 30B index ships a separate
   `lm_head.weight`, and although `MuseGlimmerForConditionalGeneration` declares
   `_tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}`,
   the runtime tie is gated on the config flag. Measured on a real instance:
   `lm_head.weight.data_ptr() != embed_tokens.weight.data_ptr()`. Load
   `lm_head.weight` explicitly; fall back to the embedding matrix only when the
   tensor is absent.
9. The embedding is RMS-normed and there is **no** `sqrt(hidden_size)` multiplier
   (unlike Gemma) — do not copy that from gemma4.py.
10. **The final `model.norm` is `RMSNorm_scaled` (`n * w`), not the centred
    `(1 + w)` variant.** Added 2026-08-12; the original spec had this wrong.
11. **HF's attention mask depends on `position_ids`, not just on token index.**
    `create_causal_mask` runs `find_packed_sequence_indices(position_ids)`,
    which starts a new "packed sequence" wherever consecutive position_ids do
    not differ by exactly 1. Feed it a stride-3 position ramp and the mask
    collapses to the diagonal. Anything comparing against HF with
    non-contiguous positions must pass an explicit mask dict via
    `attention_mask=` or it will be comparing against a different model.
12. **A constant shift of `position_ids` is not a NoPE test.** RoPE is
    relative, so a rope layer is shift-invariant too. Discriminate with a
    change in position *differences* (and pin the mask, per trap 11).

## Closest in-tree templates
- `tpu_inference/models/jax/gemma4.py` — sandwich norms + sliding/full hybrid + logit
  softcapping. Structurally the nearest relative; start here.
- `tpu_inference/models/jax/qwen3.py` — GQA + per-head norms plumbing.
Registry: add the arch to `tpu_inference/models/common/model_loader.py`.

## Acceptance: parity harness is the spec
Primary gate, run on CPU with a **tiny random-weight config** (e.g. 4 layers keeping the
[S,S,S,F] pattern, hidden 256, heads 4, kv 1, head_dim 64, vocab 512, window 8) so the
whole thing fits in RAM and exercises both layer types + NoPE:
- build HF `MuseGlimmerTextModel` and the JAX model from the SAME random weights;
- compare final hidden states and logits in float32;
- target **max abs err < 1e-4 / rel < 1e-3** on tiny; report actual numbers.
- separately assert: sliding layers attend at most `window` back; full layers see all;
  full layers are rope-free (rotating positions must not change their output).
Then a real-weight check on a handful of tokens once weights are staged.

**Implemented**: `tpu/muse_glimmer/parity_check.py`, run via `parity_check.sbatch`
(CPU, never the login node) from a throwaway venv holding transformers@main.
Status as of 2026-08-12: **PASS**, job 3686544.

  last_hidden_state   max_abs 1.48e-05   rel_to_scale 2.35e-06
  logits (softcapped) max_abs 1.69e-06   rel_to_scale 2.57e-06

Beware the relative-error metric: unrestricted element-wise `|d|/|ref|` reports
2.4e-01 on the logits purely because some entries sit near zero. Gate on
`max_abs`, on `max_abs / max|ref|`, and on the element-wise ratio restricted to
entries above 1% of peak.

The float32 gap is *not* reducible: every norm reduces in float32 by
construction. The harness therefore justifies it by triangulating on a float64
reference — HF's own float32 run sits 1.86e-05 from float64 truth while ours
sits 1.76e-05, i.e. the port is at least as close to the truth as the reference
implementation's own float32 path.

## Out of scope (this pass)
Vision tower, projector, image/video tokens, DFlash speculative drafter, tool/reasoning
parsers. Text-only.
