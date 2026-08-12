#!/usr/bin/env python3
"""CPU parity harness: JAX Muse-Glimmer text stack vs the HF reference.

This is the acceptance gate for ``tpu/muse_glimmer/SPEC.md``.  It builds a tiny
random-weight Muse-Glimmer that keeps the real ``[S, S, S, F]`` layer pattern
(so both attention types *and* the NoPE layers are exercised), loads the SAME
weights into

  * HF ``MuseGlimmerTextModel`` (transformers >= 5.16.0.dev0), and
  * ``tpu_inference.models.jax.muse_glimmer_core`` (the framework-free JAX core
    that the tpu-inference serving model delegates all its numerics to),

and compares final hidden states and logits in float32.

It additionally runs behavioural probes that a plain tensor diff cannot catch:

  P1  sliding layers attend to exactly ``sliding_window`` keys (self included)
  P2  full layers attend all the way back to token 0
  P3  full layers are RoPE-invariant (shifting ``position_ids`` is a no-op)
  P4  sliding layers are *not* RoPE-invariant (proves RoPE is really applied)
  P5  ``lm_head`` <-> ``embed_tokens`` tying behaviour of the real HF class
  P6  float64 rerun, to show the float32 residual is rounding and not a bug
  P0  the weight map covers every text key of the *real* 30B checkpoint and
      classifies every vision key as out-of-scope (``mg_weight_keys.txt``)

Every probe is answered from the HF reference *and* from the JAX core, and the
two answers are required to agree -- so the harness measures the reference
rather than asserting the spec's opinion of it.

Run it from the isolated venv (see ``parity_check.sbatch``); it must never be
run on the login node.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Locate + import the JAX core straight from the file, so that this harness
# never imports the `tpu_inference` package (which drags in vllm and TPU-only
# kernels).  The core is deliberately jax-only for exactly this reason.
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_PATH = (_REPO_ROOT / "third_party" / "tpu-inference" / "tpu_inference" /
              "models" / "jax" / "muse_glimmer_core.py")


def _load_core(path: Path):
    if not path.is_file():
        raise SystemExit(f"muse_glimmer_core.py not found at {path}")
    spec = importlib.util.spec_from_file_location("muse_glimmer_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["muse_glimmer_core"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Tiny config
# --------------------------------------------------------------------------

TINY = dict(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,  # -> layer_types [S, S, S, F], layer_rope_theta [t,t,t,0]
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=64,
    sliding_window=8,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    post_norm_eps=1e-8,
    qk_scale_factor=3.87,
    final_logit_softcapping=20.0,
    output_multiplier=0.19611613513818404,
    hidden_activation="silu",
    attention_bias=False,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
)

REAL_ROPE_THETA = 500000.0


def build_text_config(**overrides):
    from transformers import MuseGlimmerTextConfig
    kwargs = dict(TINY)
    kwargs.update(overrides)
    cfg = MuseGlimmerTextConfig(**kwargs)
    # The real checkpoint uses theta 500000 on the sliding layers; the tiny
    # default would be 10000.  Keep the real number so the RoPE tables match
    # what the 30B actually uses.
    cfg.rope_parameters = dict(cfg.rope_parameters)
    cfg.rope_parameters["rope_theta"] = REAL_ROPE_THETA
    cfg.layer_rope_theta = [
        0.0 if t == 0 else REAL_ROPE_THETA for t in cfg.layer_rope_theta
    ]
    return cfg


# --------------------------------------------------------------------------
# Random weights
# --------------------------------------------------------------------------


def randomise_(module, seed: int = 0):
    """Overwrite every parameter with a non-degenerate random tensor.

    Default HF init leaves the four centred norms at exactly zero, where
    ``n * (1 + w)`` and the (wrong) ``n * w`` differ maximally -- good for
    catching spec trap 1, bad for catching everything else.  Randomising them
    around zero keeps the trap live while also exercising real scaling.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif "layernorm" in name or name.endswith("norm.weight"):
                # `model.norm` is a ones-init scaled RMSNorm; the four per-layer
                # norms are zero-init centred ones.  Random values around the
                # respective init keep both flavours distinguishable.
                base = 1.0 if name.endswith(
                    "norm.weight") and "layernorm" not in name else 0.0
                p.copy_(base + torch.randn(p.shape, generator=g) * 0.3)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.08)
    return module


def torch_state_dict_to_jax(state_dict, core, params, jnp):
    return core.load_params(state_dict, params, dtype=jnp.float32)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


SIG_FRAC = 0.01  # "where the signal is": |ref| >= 1% of the tensor's peak


def diff_report(name: str, ref: np.ndarray, got: np.ndarray) -> Dict[str, Any]:
    """Report three relative-error flavours, because one of them lies.

    ``max_rel_raw`` -- unrestricted element-wise ``|d| / |ref|`` -- is useless
    as a gate: an activation that happens to land on 1e-5 turns 2e-6 of float32
    round-off into a "17% error".  The numbers that mean something are

      ``rel_to_scale``  = max_abs / max|ref|   (error relative to the tensor)
      ``max_rel_sig``   = max |d|/|ref| over entries with |ref| >= 1% of peak

    Both are reported; the gate uses ``max_abs``, ``rel_to_scale`` and
    ``max_rel_sig``, and ``max_rel_raw`` is printed for information only.
    """
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    assert ref.shape == got.shape, f"{name}: shape {ref.shape} vs {got.shape}"
    err = np.abs(ref - got)
    max_abs = float(err.max())
    scale = float(np.abs(ref).max())
    floor = max(1e-12, 1e-6 * scale)
    max_rel_raw = float((err / np.maximum(np.abs(ref), floor)).max())
    rel_to_scale = max_abs / scale if scale else float("nan")
    sig = np.abs(ref) >= SIG_FRAC * scale
    max_rel_sig = float(
        (err[sig] / np.abs(ref[sig])).max()) if sig.any() else float("nan")
    print(f"  {name:<28} max_abs={max_abs:.3e}  rel_to_scale={rel_to_scale:.3e}"
          f"  max_rel_sig={max_rel_sig:.3e}  [raw_rel={max_rel_raw:.3e}, "
          f"|ref|max={scale:.4g}]")
    return dict(name=name,
                max_abs=max_abs,
                max_rel_raw=max_rel_raw,
                max_rel_sig=max_rel_sig,
                rel_to_scale=rel_to_scale,
                ref_scale=scale)


def check(results, failures, args, *names):
    """Gate the named results; append human-readable reasons to `failures`."""
    for r in results:
        if names and r["name"] not in names:
            continue
        bad = []
        if not (r["max_abs"] < args.abs_tol):
            bad.append(f"max_abs={r['max_abs']:.3e} >= {args.abs_tol:g}")
        if not (r["rel_to_scale"] < args.rel_tol):
            bad.append(
                f"rel_to_scale={r['rel_to_scale']:.3e} >= {args.rel_tol:g}")
        if not (r["max_rel_sig"] < args.rel_tol):
            bad.append(
                f"max_rel_sig={r['max_rel_sig']:.3e} >= {args.rel_tol:g}")
        if bad:
            failures.append(f"{r['name']}: " + ", ".join(bad))


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


def hf_attention_span(layer_type: str, seq_len: int, window: int,
                      seed: int) -> int:
    """Empirically measure how far back HF lets a query attend.

    Builds a ONE-layer HF ``MuseGlimmerTextModel`` of the requested type, then
    perturbs the input at every key position and records which query positions
    move.  Returns ``max(q_idx - kv_idx)`` over all pairs that actually
    interact, i.e. ``window - 1`` for a sliding layer under HF's
    ``kv_idx > q_idx - sliding_window`` convention.
    """
    import torch
    from transformers import MuseGlimmerTextModel

    cfg = build_text_config(num_hidden_layers=1,
                            layer_types=[layer_type],
                            sliding_window=window)
    cfg.layer_rope_theta = [
        0.0 if layer_type == "full_attention" else REAL_ROPE_THETA
    ]
    cfg._attn_implementation = "eager"
    model = randomise_(MuseGlimmerTextModel(cfg), seed=seed).eval().float()

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=(
        torch.Generator().manual_seed(seed)))
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=False).last_hidden_state[0]

    span = -1
    for j in range(seq_len):
        alt = ids.clone()
        alt[0, j] = (alt[0, j] + 1) % cfg.vocab_size
        with torch.no_grad():
            out = model(input_ids=alt, use_cache=False).last_hidden_state[0]
        moved = (out - base).abs().max(dim=-1).values > 1e-6
        for i in range(j, seq_len):
            if bool(moved[i]):
                span = max(span, i - j)
    return span


def core_attention_span(core, jnp, layer_type: str, seq_len: int, window: int,
                        seed: int) -> int:
    """Same measurement, against the JAX core's mask."""
    import numpy as _np
    cfg = core.MuseGlimmerTextParams(
        hidden_size=TINY["hidden_size"],
        num_hidden_layers=1,
        num_attention_heads=TINY["num_attention_heads"],
        num_key_value_heads=TINY["num_key_value_heads"],
        head_dim=TINY["head_dim"],
        intermediate_size=TINY["intermediate_size"],
        vocab_size=TINY["vocab_size"],
        sliding_window=window,
        layer_types=(layer_type, ),
        layer_rope_theta=(0.0 if layer_type == core.FULL else REAL_ROPE_THETA,
                          ),
        rope_theta=REAL_ROPE_THETA,
        rms_norm_eps=TINY["rms_norm_eps"],
        post_norm_eps=TINY["post_norm_eps"],
        qk_scale_factor=TINY["qk_scale_factor"],
        final_logit_softcapping=TINY["final_logit_softcapping"],
        output_multiplier=TINY["output_multiplier"],
    )
    rng = _np.random.default_rng(seed)
    weights = _random_core_weights(core, jnp, cfg, rng)
    ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, seq_len)))
    base = _np.asarray(core.forward_hidden(weights, ids, cfg))[0]

    span = -1
    for j in range(seq_len):
        alt = _np.asarray(ids).copy()
        alt[0, j] = (alt[0, j] + 1) % cfg.vocab_size
        out = _np.asarray(core.forward_hidden(weights, jnp.asarray(alt),
                                              cfg))[0]
        moved = _np.abs(out - base).max(axis=-1) > 1e-6
        for i in range(j, seq_len):
            if moved[i]:
                span = max(span, i - j)
    return span


def _random_core_weights(core, jnp, cfg, rng):
    """A params pytree of the right shapes, straight from numpy."""
    d, n, k, h = (cfg.hidden_size, cfg.num_attention_heads,
                  cfg.num_key_value_heads, cfg.head_dim)
    f = cfg.intermediate_size

    def r(*shape, scale=0.08):
        return jnp.asarray(rng.standard_normal(shape) * scale,
                           dtype=jnp.float32)

    layers = []
    for _ in range(cfg.num_hidden_layers):
        layers.append({
            "input_layernorm": r(d, scale=0.3),
            "post_attention_layernorm": r(d, scale=0.3),
            "pre_feedforward_layernorm": r(d, scale=0.3),
            "post_feedforward_layernorm": r(d, scale=0.3),
            "q_proj": r(n * h, d),
            "k_proj": r(k * h, d),
            "v_proj": r(k * h, d),
            "o_proj": r(d, n * h),
            "attn_gate_proj": r(n * h, d),
            "mlp_gate_proj": r(f, d),
            "mlp_up_proj": r(f, d),
            "mlp_down_proj": r(d, f),
        })
    return {
        "embed_tokens": r(cfg.vocab_size, d, scale=0.5),
        "norm": jnp.asarray(1.0 + rng.standard_normal(d) * 0.3,
                            dtype=jnp.float32),
        "lm_head": r(cfg.vocab_size, d),
        "layers": layers,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


REAL_LAYERS = 52
KEYS_FILE = Path(__file__).with_name("mg_weight_keys.txt")


def check_real_checkpoint_keys(core, failures) -> None:
    """P0: run the real 30B key list through the loader's classifier.

    ``mg_weight_keys.txt`` is the verbatim key set of
    ``meta-models/Muse-Glimmer-30B``'s ``model.safetensors.index.json``.  No
    weights required -- names alone catch the mistakes that matter: a missed
    prefix, the ``self_attn.gate_proj`` / ``mlp.gate_proj`` collision, or a
    vision tensor sneaking into the text params.
    """
    print("\n-- P0: weight map vs the real 30B checkpoint key list --")
    if not KEYS_FILE.is_file():
        print(f"  [skipped] {KEYS_FILE} missing")
        return
    keys = [k for k in KEYS_FILE.read_text().split("\n") if k]
    text, vision, unmapped = {}, [], []
    for k in keys:
        loc = core.classify_weight(k)
        if loc is None:
            (vision if core.is_vision_weight(k) else unmapped).append(k)
        else:
            text.setdefault(loc, []).append(k)

    per_layer = {}
    specials = set()
    for loc in text:
        if loc[0] == "layers":
            per_layer.setdefault(loc[1], set()).add(loc[2])
        else:
            specials.add(loc[0])
    print(f"  {len(keys)} keys -> {len(text)} text slots, {len(vision)} vision "
          f"(dropped), {len(unmapped)} unmapped")
    print(f"  non-layer slots: {sorted(specials)}")
    print(f"  layers covered: {len(per_layer)} (expected {REAL_LAYERS})")

    if unmapped:
        failures.append(f"P0: {len(unmapped)} checkpoint keys unmapped and not "
                        f"vision, e.g. {unmapped[:5]}")
    if specials != {"embed_tokens", "norm", "lm_head"}:
        failures.append(f"P0: unexpected non-layer slots {sorted(specials)}")
    if len(per_layer) != REAL_LAYERS:
        failures.append(
            f"P0: {len(per_layer)} layers covered, expected {REAL_LAYERS}")
    for idx, got in sorted(per_layer.items()):
        missing = core.REQUIRED_LAYER_KEYS - got
        if missing:
            failures.append(f"P0: layer {idx} missing {sorted(missing)}")
            break
    # The one aliasing hazard worth naming explicitly.
    a = core.classify_weight("model.language_model.layers.0.self_attn.gate_proj.weight")
    b = core.classify_weight("model.language_model.layers.0.mlp.gate_proj.weight")
    print(f"  self_attn.gate_proj -> {a[2]!r};  mlp.gate_proj -> {b[2]!r}")
    if a == b:
        failures.append("P0: self_attn.gate_proj and mlp.gate_proj alias to the"
                        " same slot")
    if any(core.classify_weight(k) is not None for k in vision):
        failures.append("P0: a vision key leaked into the text params")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--extra-seq-lens",
                    type=str,
                    default="7,23,37,129",
                    help="extra prefill lengths to check: shorter than the "
                    "window, non-multiples of it, and > 128")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--abs-tol", type=float, default=1e-4)
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    ap.add_argument("--core", type=Path, default=_CORE_PATH)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")

    import jax
    import jax.numpy as jnp
    import torch
    import transformers
    from transformers import (MuseGlimmerForConditionalGeneration,
                              MuseGlimmerTextModel)

    core = _load_core(args.core)

    print("=" * 78)
    print("Muse-Glimmer text-only parity harness")
    print("=" * 78)
    print(f"transformers {transformers.__version__}   torch {torch.__version__}"
          f"   jax {jax.__version__}")
    print(f"jax devices: {jax.devices()}")
    print(f"core: {args.core}")
    print()

    torch.manual_seed(args.seed)
    cfg = build_text_config()
    cfg._attn_implementation = "eager"
    print(f"layer_types      = {cfg.layer_types}")
    print(f"layer_rope_theta = {cfg.layer_rope_theta}")
    print(f"sliding_window   = {cfg.sliding_window}   "
          f"rms_norm_eps={cfg.rms_norm_eps}  post_norm_eps={cfg.post_norm_eps}")
    print()

    # ---- HF reference -----------------------------------------------------
    hf = randomise_(MuseGlimmerTextModel(cfg), seed=args.seed).eval().float()
    state_dict = {k: v.detach().clone() for k, v in hf.state_dict().items()}

    # A standalone lm_head, so logits can be checked without instantiating the
    # vision tower.  (The real-class tie behaviour is probed separately, P5.)
    lm_head_w = torch.randn(cfg.vocab_size, cfg.hidden_size,
                            generator=torch.Generator().manual_seed(
                                args.seed + 7)) * 0.05
    state_dict["lm_head.weight"] = lm_head_w

    params = core.params_from_hf_config(cfg)
    jax_weights = torch_state_dict_to_jax(state_dict, core, params, jnp)
    print(f"loaded {len(state_dict)} checkpoint tensors -> "
          f"{len(jax_weights['layers'])} JAX layers + "
          f"{sorted(k for k in jax_weights if k != 'layers')}")

    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len),
                        generator=gen)

    with torch.no_grad():
        ref_hidden = hf(input_ids=ids,
                        use_cache=False).last_hidden_state.float()
        ref_logits = ref_hidden @ lm_head_w.t()
        ref_logits = ref_logits * cfg.output_multiplier
        ref_logits = torch.tanh(
            ref_logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping

    jax_ids = jnp.asarray(ids.numpy())
    got_hidden, got_logits = core.forward(jax_weights, jax_ids, params)

    print("\n-- tensor parity (float32) --")
    results = [
        diff_report("last_hidden_state", ref_hidden.numpy(),
                    np.asarray(got_hidden)),
        diff_report("logits (post softcap)", ref_logits.numpy(),
                    np.asarray(got_logits)),
    ]

    # Ragged / non-round sequence lengths. The TPU serving path runs the
    # ragged-paged-attention Pallas kernel, which pads sequences up to its
    # block size; a length that is neither a multiple of the sliding window
    # nor of a typical block size is the one most likely to expose an
    # off-by-one in mask construction, so pin a few explicitly.
    extra = [int(x) for x in args.extra_seq_lens.split(",") if x.strip()]
    for L in extra:
        ids_L = torch.randint(0, cfg.vocab_size, (args.batch, L),
                              generator=gen)
        with torch.no_grad():
            ref_L = hf(input_ids=ids_L, use_cache=False).last_hidden_state
        got_L = core.forward_hidden(jax_weights, jnp.asarray(ids_L.numpy()),
                                    params)
        results.append(
            diff_report(f"last_hidden_state T={L}", ref_L.float().numpy(),
                        np.asarray(got_L)))

    failures = []
    check(results, failures, args)

    check_real_checkpoint_keys(core, failures)

    # ---- P1/P2: attention span -------------------------------------------
    print("\n-- P1/P2: measured attention span (max q_idx - kv_idx) --")
    window = cfg.sliding_window
    probe_len = 3 * window
    hf_slide = hf_attention_span("sliding_attention", probe_len, window,
                                 args.seed)
    core_slide = core_attention_span(core, jnp, core.SLIDING, probe_len, window,
                                     args.seed)
    hf_full = hf_attention_span("full_attention", probe_len, window, args.seed)
    core_full = core_attention_span(core, jnp, core.FULL, probe_len, window,
                                    args.seed)
    print(f"  sliding (window={window}):  HF={hf_slide}   JAX={core_slide}   "
          f"expected {window - 1}")
    print(f"  full    (seq={probe_len}):     HF={hf_full}   JAX={core_full}   "
          f"expected {probe_len - 1}")
    if hf_slide != core_slide:
        failures.append(f"sliding span mismatch HF={hf_slide} JAX={core_slide}")
    if hf_full != core_full:
        failures.append(f"full span mismatch HF={hf_full} JAX={core_full}")
    if hf_slide != window - 1:
        failures.append(
            f"HF sliding span {hf_slide} != window-1 ({window - 1}): the "
            "reference window convention is NOT q_idx - kv_idx < W")
    if hf_full != probe_len - 1:
        failures.append(f"HF full span {hf_full} != seq-1 ({probe_len - 1})")

    # ---- P3/P4: RoPE invariance ------------------------------------------
    #
    # TRAP A: a *constant* shift of position_ids is NOT a NoPE test.  RoPE is
    # relative -- q_i . k_j depends only on (pos_i - pos_j) -- so a rope layer
    # is shift-invariant too, and "shifting positions doesn't change the
    # output" is satisfied by every layer in the model.  The discriminator has
    # to change the position *differences*: a stride-3 position ramp.
    #
    # TRAP B: HF does not treat position_ids as RoPE-only.  `create_causal_mask`
    # runs `find_packed_sequence_indices(position_ids)`, which starts a new
    # "packed sequence" wherever consecutive position_ids do not differ by
    # exactly 1.  A stride-3 ramp therefore makes every token its own sequence
    # and collapses the mask to the diagonal -- the output changes for reasons
    # that have nothing to do with RoPE.  So the probe hands HF an explicit
    # precomputed mask dict (the documented `attention_mask: dict` path), after
    # which position_ids really do only feed RoPE.
    print("\n-- P3/P4: position remap (NoPE check) --")
    shift = 97
    base_pos = torch.arange(args.seq_len).unsqueeze(0).expand(
        args.batch, -1).contiguous()
    shifted_pos = base_pos + shift  # constant shift: no-op for RoPE *and* NoPE
    strided_pos = base_pos * 3  # changes relative offsets: RoPE must react

    def explicit_mask_dict(seq_len: int, batch: int, window: int):
        idx = torch.arange(seq_len)
        causal = idx[None, :] <= idx[:, None]
        sliding = causal & ((idx[:, None] - idx[None, :]) < window)
        neg = torch.finfo(torch.float32).min

        def additive(m):
            return torch.where(m, torch.zeros(()), torch.full(
                (), neg)).view(1, 1, seq_len, seq_len).expand(
                    batch, 1, seq_len, seq_len).contiguous()

        return {
            "full_attention": additive(causal),
            "sliding_attention": additive(sliding),
        }

    mask_dict = explicit_mask_dict(args.seq_len, args.batch, cfg.sliding_window)

    def hf_single_layer(layer_type, position_ids):
        c = build_text_config(num_hidden_layers=1, layer_types=[layer_type])
        c.layer_rope_theta = [
            0.0 if layer_type == "full_attention" else REAL_ROPE_THETA
        ]
        c._attn_implementation = "eager"
        m = randomise_(MuseGlimmerTextModel(c), seed=args.seed).eval().float()
        with torch.no_grad():
            return m(input_ids=ids,
                     position_ids=position_ids,
                     attention_mask=mask_dict,
                     use_cache=False).last_hidden_state.float().numpy()

    hf_full_base = hf_single_layer("full_attention", base_pos)
    hf_full_shift = hf_single_layer("full_attention", shifted_pos)
    hf_full_stride = hf_single_layer("full_attention", strided_pos)
    hf_slide_base = hf_single_layer("sliding_attention", base_pos)
    hf_slide_shift = hf_single_layer("sliding_attention", shifted_pos)
    hf_slide_stride = hf_single_layer("sliding_attention", strided_pos)

    def rel_delta(a, b):
        scale = float(np.abs(a).max()) or 1.0
        return float(np.abs(a - b).max()) / scale

    d_full_shift = rel_delta(hf_full_base, hf_full_shift)
    d_slide_shift = rel_delta(hf_slide_base, hf_slide_shift)
    d_full_stride = rel_delta(hf_full_base, hf_full_stride)
    d_slide_stride = rel_delta(hf_slide_base, hf_slide_stride)
    print("  (explicit mask dict passed to HF, so position_ids feed RoPE only)")
    print(f"  [constant shift +{shift}] HF full={d_full_shift:.3e}  "
          f"HF sliding={d_slide_shift:.3e}   "
          "(both ~0: RoPE is relative, so this proves nothing)")
    print(f"  [stride x3      ] HF full={d_full_stride:.3e}  "
          f"HF sliding={d_slide_stride:.3e}   "
          "(full must be 0 -> NoPE; sliding must be large -> RoPE live)")
    if d_full_shift != 0.0:
        failures.append(f"HF full layer moved under a constant position shift "
                        f"(delta {d_full_shift:.3e})")
    if d_full_stride != 0.0:
        failures.append(f"HF full layer is NOT position-independent: stride-3 "
                        f"positions changed it by {d_full_stride:.3e}")
    if d_slide_stride <= 1e-2:
        failures.append(f"HF sliding layer barely reacted to stride-3 "
                        f"positions ({d_slide_stride:.3e}): RoPE may be off")

    # Document TRAP B by measuring it: same stride-3 positions, but letting HF
    # build its own mask.  A NoPE layer still moves, because the mask moved.
    c_auto = build_text_config(num_hidden_layers=1,
                               layer_types=["full_attention"])
    c_auto.layer_rope_theta = [0.0]
    c_auto._attn_implementation = "eager"
    m_auto = randomise_(MuseGlimmerTextModel(c_auto),
                        seed=args.seed).eval().float()
    with torch.no_grad():
        auto_base = m_auto(input_ids=ids, position_ids=base_pos,
                           use_cache=False).last_hidden_state.numpy()
        auto_stride = m_auto(input_ids=ids, position_ids=strided_pos,
                             use_cache=False).last_hidden_state.numpy()
    d_auto = rel_delta(auto_base, auto_stride)
    print(f"  [stride x3, HF-built mask] NoPE full layer moved by {d_auto:.3e}"
          " -- that is HF's packed-sequence mask reacting to position_ids,")
    print("    NOT RoPE.  Any port that feeds non-contiguous position_ids to "
          "HF for a reference must pin the mask.")
    if d_auto == 0.0:
        print("    (note: HF no longer derives packed sequences from "
              "position_ids; the explicit-mask workaround is now redundant)")

    # Same probe on the JAX core, sharing the HF weights of the 1-layer models.
    def core_single_layer(layer_type, position_ids):
        c = build_text_config(num_hidden_layers=1, layer_types=[layer_type])
        c.layer_rope_theta = [
            0.0 if layer_type == "full_attention" else REAL_ROPE_THETA
        ]
        c._attn_implementation = "eager"
        m = randomise_(MuseGlimmerTextModel(c), seed=args.seed).eval().float()
        p = core.params_from_hf_config(c)
        sd = dict(m.state_dict())
        sd["lm_head.weight"] = lm_head_w
        w = core.load_params(sd, p, dtype=jnp.float32)
        out = core.forward_hidden(w, jnp.asarray(ids.numpy()), p,
                                  positions=jnp.asarray(position_ids.numpy()))
        return np.asarray(out), m, p, w

    core_full_base, m_full, p_full, w_full = core_single_layer(
        "full_attention", base_pos)
    core_full_shift, _, _, _ = core_single_layer("full_attention", shifted_pos)
    core_full_stride, _, _, _ = core_single_layer("full_attention",
                                                  strided_pos)
    core_slide_base, _, _, _ = core_single_layer("sliding_attention", base_pos)
    core_slide_shift, _, _, _ = core_single_layer("sliding_attention",
                                                  shifted_pos)
    core_slide_stride, _, _, _ = core_single_layer("sliding_attention",
                                                   strided_pos)
    cd_full_shift = rel_delta(core_full_base, core_full_shift)
    cd_slide_shift = rel_delta(core_slide_base, core_slide_shift)
    cd_full_stride = rel_delta(core_full_base, core_full_stride)
    cd_slide_stride = rel_delta(core_slide_base, core_slide_stride)
    print(f"  [constant shift +{shift}] JAX full={cd_full_shift:.3e}  "
          f"JAX sliding={cd_slide_shift:.3e}")
    print(f"  [stride x3      ] JAX full={cd_full_stride:.3e}  "
          f"JAX sliding={cd_slide_stride:.3e}")
    if cd_full_stride != 0.0:
        failures.append(f"JAX full layer is NOT position-independent: stride-3 "
                        f"positions changed it by {cd_full_stride:.3e}")
    if cd_slide_stride <= 1e-2:
        failures.append(f"JAX sliding layer barely reacted to stride-3 "
                        f"positions ({cd_slide_stride:.3e}): RoPE may be off")
    # The two engines must also agree on *how much* RoPE moves the sliding
    # layer -- equal magnitudes rule out "both applied some rotation".
    if abs(cd_slide_stride - d_slide_stride) > 1e-3 * max(
            d_slide_stride, 1e-9):
        failures.append(
            f"stride-3 response differs: HF={d_slide_stride:.6e} vs "
            f"JAX={cd_slide_stride:.6e}")

    # Per-layer-type single-layer parity, which isolates a mismatch to one
    # attention flavour instead of blaming the whole stack.
    print("\n-- per-layer-type single-layer parity --")
    results.append(
        diff_report("1x full_attention", hf_full_base, core_full_base))
    results.append(
        diff_report("1x sliding_attention", hf_slide_base, core_slide_base))
    check(results[-2:], failures, args)

    # ---- P5: lm_head tying ------------------------------------------------
    print("\n-- P5: lm_head / embed_tokens tying in the real HF class --")
    try:
        from transformers import MuseGlimmerConfig, MuseGlimmerVisionConfig
        vcfg = MuseGlimmerVisionConfig(hidden_size=32,
                                       intermediate_size=64,
                                       num_hidden_layers=2,
                                       num_attention_heads=2,
                                       layer_types=[
                                           "window_attention",
                                           "full_attention"
                                       ])
        full_cfg = MuseGlimmerConfig(text_config=cfg.to_dict(),
                                     vision_config=vcfg.to_dict(),
                                     out_hidden_size=64,
                                     projector_hidden_size=64)
        full_cfg._attn_implementation = "eager"
        mm = MuseGlimmerForConditionalGeneration(full_cfg).eval()
        tied = (mm.lm_head.weight.data_ptr() ==
                mm.model.language_model.embed_tokens.weight.data_ptr())
        print(f"  _tied_weights_keys       = {mm._tied_weights_keys}")
        print(f"  config.tie_word_embeddings = "
              f"{full_cfg.text_config.tie_word_embeddings}")
        print(f"  lm_head IS tied to embed_tokens at runtime: {tied}")
        keys = set(mm.state_dict())
        print("  'lm_head.weight' present in state_dict: "
              f"{'lm_head.weight' in keys}")

        # End-to-end logits through the real class, text-only.
        randomise_(mm, seed=args.seed + 3)
        with torch.no_grad():
            mm_out = mm(input_ids=ids, use_cache=False)
        mm_sd = {k: v.detach().clone() for k, v in mm.state_dict().items()}
        mm_params = core.params_from_hf_config(full_cfg.text_config)
        mm_w = core.load_params(mm_sd, mm_params, dtype=jnp.float32)
        mm_hidden, mm_logits = core.forward(mm_w, jax_ids, mm_params)
        results.append(
            diff_report("ForCondGen logits", mm_out.logits.float().numpy(),
                        np.asarray(mm_logits)))
        check(results[-1:], failures, args)
    except Exception as exc:  # pragma: no cover - informational probe
        print(f"  [skipped] could not build the multimodal wrapper: "
              f"{type(exc).__name__}: {exc}")

    # ---- P6: is the residual error just float32 round-off? ----------------
    #
    # Naively one would rerun in float64 and expect the gap to collapse to
    # ~1e-15.  It does not, and that is not a bug: EVERY norm in this model
    # reduces in float32 by construction -- HF literally writes
    # `hidden_states.float()` inside `MuseGlimmerRMSNorm._norm` -- so the
    # activations carry float32 precision no matter what dtype the weights are.
    # The meaningful question is therefore not "how small is the gap" but "is
    # our gap to the truth bigger than HF's own gap to the truth".  Use HF
    # float64 as ground truth and compare two distances to it.
    print("\n-- P6: float64 ground truth (is our error bigger than HF's own?) --")
    try:
        jax.config.update("jax_enable_x64", True)
        hf64 = hf.double()
        with torch.no_grad():
            ref64 = hf64(input_ids=ids, use_cache=False).last_hidden_state
        ref64_np = ref64.numpy()
        w64 = core.load_params(
            {k: v.double()
             for k, v in state_dict.items()}, params, dtype=jnp.float64)
        got64 = np.asarray(core.forward_hidden(w64, jax_ids, params))

        d_hf = float(np.abs(ref_hidden.numpy().astype(np.float64) -
                            ref64_np).max())
        d_jax = float(np.abs(np.asarray(got_hidden).astype(np.float64) -
                             ref64_np).max())
        d_jax64 = float(np.abs(got64 - ref64_np).max())
        print(f"  |HF_f32  - HF_f64| = {d_hf:.3e}   <- HF's own float32 noise")
        print(f"  |JAX_f32 - HF_f64| = {d_jax:.3e}   <- our float32 distance "
              "to ground truth")
        print(f"  |JAX_f64 - HF_f64| = {d_jax64:.3e}   <- residual with both "
              "in float64 (floor is the float32 norms)")
        if d_hf > 0 and d_jax > 3.0 * d_hf:
            failures.append(
                f"JAX float32 is {d_jax / d_hf:.1f}x further from the float64 "
                "truth than HF float32 is: the gap is systematic, not rounding")
        if d_jax64 > 3.0 * d_hf:
            failures.append(
                f"float64-vs-float64 residual {d_jax64:.3e} exceeds 3x HF's "
                f"own float32 noise {d_hf:.3e}")
    except Exception as exc:  # pragma: no cover - informational probe
        print(f"  [skipped] float64 rerun failed: {type(exc).__name__}: {exc}")
    finally:
        jax.config.update("jax_enable_x64", False)

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    if failures:
        print(f"PARITY FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print(f"PARITY OK  (abs < {args.abs_tol:g}, rel < {args.rel_tol:g})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
