#!/usr/bin/env python3
"""Stage 2 of the Muse-Glimmer HF <-> MaxText parity harness: the MaxText side.

Reads stage 1's dump, pushes the HF tensors through MaxText's REAL checkpoint
conversion machinery (``PARAM_MAPPING`` / ``HOOK_FNS`` /
``validate_and_filter_param_map_keys`` / ``apply_hook_fns`` -- the exact functions
``python -m maxtext.checkpoint_conversion.to_maxtext`` calls), builds the MaxText
model from the resulting parameter tree, and compares final hidden states and logits
against the reference in float32.

Also asserts the structural properties the numbers alone would not pin down:
  * sliding layers cannot see further back than ``sliding_window_size``;
  * full layers can;
  * full layers are ROPE-INVARIANT (they are the NoPE layers);
  * the converter's mapping covers exactly the model's parameter set;
  * every parameter carries a logical sharding annotation (FSDP/TP would silently
    replicate otherwise);
  * the scanned and unscanned layouts agree.

Runs on CPU. Must run in the MaxText venv (jax cpu + the muse-glimmer clone).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import linen as nn  # noqa: E402
from flax import nnx  # noqa: E402


FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}{(' :: ' + detail) if detail else ''}")
  if not ok:
    FAILURES.append(f"{name} :: {detail}")
  return ok


@contextlib.contextmanager
def maxtext_config_cwd():
  """chdir into a temp dir whose ``src`` symlink points at maxtext's configs.

  Mirrors ``skyrl/backends/tunix_backend.py::_maxtext_config_cwd`` so the bare
  ``base.yml`` argv entry resolves the same way it does in production.
  """
  import maxtext
  from pathlib import Path

  configs_dir = Path(maxtext.__file__).parent / "configs"
  prev = os.getcwd()
  with tempfile.TemporaryDirectory() as td:
    # pyconfig resolves a bare `base.yml` against <cwd>/src/, and derives the
    # models/ directory from the *parent* of the base config's directory. Lay the
    # temp dir out so both land inside the installed package's configs/.
    src = Path(td) / "src"
    src.mkdir()
    (src / "base.yml").symlink_to(configs_dir / "base.yml")
    (src / "models").symlink_to(configs_dir / "models")
    (src / "maxtext").symlink_to(configs_dir.parent)
    os.chdir(td)
    try:
      yield
    finally:
      os.chdir(prev)


def make_config(model_name: str, seq_len: int, scan_layers: bool, **extra):
  import maxtext.configs.pyconfig as pyconfig

  overrides = dict(
      model_name=model_name,
      per_device_batch_size=1,
      max_target_length=seq_len,
      max_prefill_predict_length=max(1, seq_len // 2),
      scan_layers=scan_layers,
      pure_nnx=True,
      pure_nnx_decoder=True,
      # float32 end to end -- this is a numerical parity test, not a perf test.
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      float32_qk_product=True,
      float32_logits=True,
      logits_dot_in_fp32=True,
      cast_logits_to_fp32=True,
      # dot_product is the only kernel available on CPU; splash/flash are TPU-only.
      attention="dot_product",
      enable_dropout=False,
      dropout_rate=0.0,
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
  )
  overrides.update(extra)
  argv = ["", "base.yml"] + [
      f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()
  ]
  with maxtext_config_cwd():
    return pyconfig.initialize(argv)


def build_params_from_hf(cfg, hf_weights: dict) -> tuple[dict, dict]:
  """Run the HF tensors through MaxText's real converter mapping.

  Returns (linen params tree, diagnostics).
  """
  from maxtext.checkpoint_conversion.to_maxtext import (
      _build_single_axis_stacked_tensor,
      get_maxtext_model_info,
  )
  from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
  from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
  from maxtext.checkpoint_conversion.utils.utils import (
      apply_hook_fns,
      validate_and_filter_param_map_keys,
  )

  model_key = cfg.model_name
  hf_config_dict = HF_MODEL_CONFIGS[model_key].to_dict()
  param_map = PARAM_MAPPING[model_key](hf_config_dict, cfg, cfg.scan_layers)
  hooks = HOOK_FNS[model_key](hf_config_dict, cfg, cfg.scan_layers, saving_to_hf=False)

  abstract_dict, treedef = get_maxtext_model_info(cfg)

  # --- the mapping must cover the model exactly -----------------------------
  mapped = set(param_map)
  present = set(abstract_dict)
  missing = sorted(present - mapped)  # model params with no HF source -> random weights
  extra = sorted(mapped - present)  # mapping entries the model does not have
  diag = {"missing": missing, "extra": extra, "n_params": len(present)}

  filtered = validate_and_filter_param_map_keys(param_map.keys(), abstract_dict.keys())
  flat = [None] * len(abstract_dict)
  for mt_key in filtered:
    hf_src = param_map[mt_key]
    hook = hooks.get(mt_key)
    idx, target_shape = abstract_dict[mt_key]
    if isinstance(hf_src, list):
      # scanned: MaxText stacks along `config.param_scan_axis` (1, not 0). Use the
      # converter's own stacker so a bug here cannot diverge from production.
      flat[idx] = np.asarray(
          _build_single_axis_stacked_tensor(
              hf_src,
              lambda k: hf_weights[k].astype(np.float32),
              hook,
              tuple(target_shape),
              cfg,
          ),
          np.float32,
      )
    else:
      flat[idx] = np.asarray(
          apply_hook_fns(hf_weights[hf_src].astype(np.float32), tuple(target_shape), hook), np.float32
      )

  # anything unmapped stays at its (random) init -- record it loudly
  for mt_key in missing:
    idx, target_shape = abstract_dict[mt_key]
    flat[idx] = np.zeros(target_shape, np.float32)

  params = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in flat])
  return params, diag


def build_linen_model(cfg):
  import jax
  from maxtext.layers import quantizations
  from maxtext.models import models
  from maxtext.utils import maxtext_utils

  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
  quant = quantizations.configure_quantization(cfg)
  return models.transformer_as_linen(cfg, mesh, quant=quant), mesh


def run_model(model, params, ids, positions=None, segment_ids=None):
  from maxtext.common.common_types import MODEL_MODE_TRAIN

  b, s = ids.shape
  if positions is None:
    positions = jnp.arange(s, dtype=jnp.int32)[None, :].repeat(b, axis=0)
  if segment_ids is None:
    segment_ids = jnp.ones((b, s), dtype=jnp.int32)
  out = model.apply(
      {"params": params},
      jnp.asarray(ids, jnp.int32),
      positions,
      decoder_segment_ids=segment_ids,
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )
  return out[0] if isinstance(out, tuple) else out


def errs(a, b):
  """(max_abs, rel_to_scale, rel_significant).

  `rel_to_scale` = max_abs / max|ref| -- the metric to gate on.
  `rel_significant` = element-wise |d|/|ref| restricted to entries above 1% of
  peak. Unrestricted element-wise relative error is meaningless here: logits
  contain near-zero entries and it blows up to ~1e-1 for a numerically perfect
  port (this bit the serving-side agent; see SPEC.md acceptance section).
  """
  a = np.asarray(a, np.float64)
  b = np.asarray(b, np.float64)
  abs_err = np.abs(a - b)
  peak = max(float(np.abs(b).max()), 1e-30)
  sig = np.abs(b) > 0.01 * peak
  rel_sig = float((abs_err[sig] / np.abs(b)[sig]).max()) if sig.any() else 0.0
  return float(abs_err.max()), float(abs_err.max() / peak), rel_sig


# ---------------------------------------------------------------------------
# Structural probes: single decoder layer, so the assertion is about ONE layer.
# ---------------------------------------------------------------------------
def load_layer_from_hf(layer, layer_idx: int, hf_weights: dict):
  """Copy one HF decoder layer's tensors into a standalone MuseGlimmerDecoderLayer.

  Uses the same reshape convention as the converter (`kernel.T.reshape(target)`),
  so a bug here and a bug in `param_mapping.py` would have to agree to hide.

  Note the naming trap this makes explicit: HF `self_attn.gate_proj` (the sigmoid
  attention gate) lands on `self_attention/attn_gate`, while HF `mlp.gate_proj`
  (SwiGLU) lands on `mlp/wi_0`. Two different tensors that share a last dotted
  component -- any matcher keyed on that component alone will confuse them.
  """
  pre = f"model.language_model.layers.{layer_idx}."
  kernels = {
      ("self_attention", "query", "kernel"): "self_attn.q_proj.weight",
      ("self_attention", "key", "kernel"): "self_attn.k_proj.weight",
      ("self_attention", "value", "kernel"): "self_attn.v_proj.weight",
      ("self_attention", "out", "kernel"): "self_attn.o_proj.weight",
      ("self_attention", "attn_gate", "kernel"): "self_attn.gate_proj.weight",
      ("mlp", "wi_0", "kernel"): "mlp.gate_proj.weight",
      ("mlp", "wi_1", "kernel"): "mlp.up_proj.weight",
      ("mlp", "wo", "kernel"): "mlp.down_proj.weight",
  }
  norms = {
      ("pre_self_attention_norm", "scale"): "input_layernorm.weight",
      ("post_self_attention_norm", "scale"): "post_attention_layernorm.weight",
      ("pre_ffw_norm", "scale"): "pre_feedforward_layernorm.weight",
      ("post_ffw_norm", "scale"): "post_feedforward_layernorm.weight",
  }
  state = nnx.state(layer, nnx.Param)
  flat = state.flat_state()
  seen = set()
  for path, var in flat:
    key = tuple(str(k) for k in path)
    tgt = tuple(k for k in key if k in ("kernel", "scale") or not k.isdigit())
    if tgt in kernels:
      src = hf_weights[pre + kernels[tgt]].astype(np.float32)
      var.value = jnp.asarray(src.T.reshape(var.value.shape))
    elif tgt in norms:
      src = hf_weights[pre + norms[tgt]].astype(np.float32)
      var.value = jnp.asarray(src.reshape(var.value.shape))
    else:
      raise KeyError(f"unmapped layer parameter {key}")
    seen.add(tgt)
  missing = (set(kernels) | set(norms)) - seen
  assert not missing, f"HF tensors with no MaxText home: {sorted(missing)}"
  nnx.update(layer, state)


def layer_probes(cfg, hf_weights, dump, atol, rtol):
  """Single-layer probes, mirrored 1:1 against the HF reference layers.

  MaxText's `generate_attention_mask` builds masks from `broadcasted_iota` over
  INDICES, not from `decoder_positions`, so changing positions changes RoPE only.
  The HF side is driven the same way: stage 1 calls the decoder layer directly
  with an explicit index-based mask, never letting `create_causal_mask` derive
  packed-sequence boundaries from position_ids (SPEC trap 11).
  """
  from maxtext.common.common_types import MODEL_MODE_TRAIN, AttentionType
  from maxtext.models import muse_glimmer
  from maxtext.utils import maxtext_utils

  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  h_in = jnp.asarray(dump["probe_hidden_in"])
  h_pert = jnp.asarray(dump["probe_hidden_in_perturbed"])
  s = h_in.shape[1]
  pos = jnp.asarray(dump["probe_positions"], jnp.int32)
  pos_s3 = jnp.asarray(dump["probe_positions_stride3"], jnp.int32)
  seg = jnp.ones((1, s), jnp.int32)
  perturb_at = 5
  window = cfg.sliding_window_size

  def make_layer(attention_type, is_nope, layer_idx):
    layer = muse_glimmer.MuseGlimmerDecoderLayer(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        attention_type=attention_type,
        is_nope_layer=is_nope,
        quant=None,
        rngs=nnx.Rngs(0),
    )
    load_layer_from_hf(layer, layer_idx, hf_weights)
    return layer

  def call(layer, h, positions):
    out = layer(h, seg, positions, True, MODEL_MODE_TRAIN)
    return out[0] if isinstance(out, tuple) else out

  sliding = make_layer(AttentionType.LOCAL_SLIDING, False, 0)
  full = make_layer(AttentionType.GLOBAL, True, 3)

  # ---- numeric parity of a single layer against HF -----------------------
  base = np.asarray(call(sliding, h_in, pos))
  fbase = np.asarray(call(full, h_in, pos))
  for tag, got in (("sliding", base), ("full", fbase)):
    a, r, rs = errs(got, dump[f"probe_{tag}_base"])
    check(
        f"single-layer parity vs HF ({tag} layer)",
        a < atol and r < rtol,
        f"max_abs={a:.3e} rel_to_scale={r:.3e} rel_sig={rs:.3e}",
    )

  # ---- sliding layer respects the window --------------------------------
  pert = np.asarray(call(sliding, h_pert, pos))
  delta = np.abs(base - pert).max(axis=-1)[0]
  inside = [i for i in range(s) if perturb_at <= i < perturb_at + window]
  outside = [i for i in range(s) if i >= perturb_at + window]
  check(
      "sliding layer: positions further back than the window are unaffected",
      max(delta[i] for i in outside) < 1e-9,
      f"window={window} max|delta| at distance>=window = {max(delta[i] for i in outside):.3e}",
  )
  check(
      "sliding layer: positions inside the window ARE affected (probe is not vacuous)",
      max(delta[i] for i in inside) > 1e-4,
      f"max|delta| inside window = {max(delta[i] for i in inside):.3e}",
  )
  # the same window behaviour must hold in the reference
  hf_delta = np.abs(dump["probe_sliding_base"] - dump["probe_sliding_perturbed"]).max(axis=-1)[0]
  check(
      "HF sliding layer shows the SAME window boundary (probe agrees with the reference)",
      max(hf_delta[i] for i in outside) < 1e-9 and max(hf_delta[i] for i in inside) > 1e-4,
      f"hf outside={max(hf_delta[i] for i in outside):.3e} inside={max(hf_delta[i] for i in inside):.3e}",
  )

  # ---- full layer sees everything ---------------------------------------
  fpert = np.asarray(call(full, h_pert, pos))
  fdelta = np.abs(fbase - fpert).max(axis=-1)[0]
  check(
      "full layer: sees a perturbation arbitrarily far back",
      min(fdelta[i] for i in outside) > 1e-4,
      f"min|delta| at distance>=window = {min(fdelta[i] for i in outside):.3e}",
  )

  # ---- full layer is rope-free (NoPE) -----------------------------------
  # SPEC trap 12: use a change in position DIFFERENCES (stride 3), not a constant
  # shift -- RoPE is relative, so a constant shift leaves rope layers unchanged too.
  fs3 = np.asarray(call(full, h_in, pos_s3))
  a, _, _ = errs(fs3, fbase)
  check(
      "full layer is ROPE-FREE (stride-3 positions change nothing; layer_rope_theta == 0)",
      a < 1e-9,
      f"max|delta| under stride-3 positions = {a:.3e}",
  )
  ss3 = np.asarray(call(sliding, h_in, pos_s3))
  a2, _, _ = errs(ss3, base)
  check(
      "sliding layer IS position-dependent (the stride-3 probe is not vacuous)",
      a2 > 1e-4,
      f"max|delta| under stride-3 positions = {a2:.3e}",
  )
  # and both stride-3 runs must still match HF
  for tag, got in (("sliding", ss3), ("full", fs3)):
    a3, r3, _ = errs(got, dump[f"probe_{tag}_stride3"])
    check(
        f"single-layer parity vs HF at stride-3 positions ({tag} layer)",
        a3 < atol and r3 < rtol,
        f"max_abs={a3:.3e} rel_to_scale={r3:.3e}",
    )


def sharding_check(cfg, model, params):
  """Every parameter must carry a logical axis annotation, or FSDP silently replicates."""
  from maxtext.utils import maxtext_utils

  abstract = maxtext_utils.get_abstract_param(model, cfg)["params"]
  unannotated = []
  for path, leaf in jax.tree_util.tree_flatten_with_path(
      abstract, is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned)
  )[0]:
    key = "-".join(k.key for k in path if hasattr(k, "key"))
    names = getattr(leaf, "names", None)
    if not names or all(n is None for n in names):
      unannotated.append((key, names))
  check(
      "every parameter carries a logical sharding annotation",
      not unannotated,
      f"{len(unannotated)} unannotated: {unannotated[:6]}",
  )
  return unannotated


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True, help="stage 1 output directory")
  ap.add_argument("--model_name", default="muse-glimmer-tiny")
  ap.add_argument("--atol", type=float, default=1e-4)
  ap.add_argument("--rtol", type=float, default=1e-3)
  args = ap.parse_args()

  weights = dict(np.load(os.path.join(args.dump, "weights.npz")))
  ref = dict(np.load(os.path.join(args.dump, "ref.npz")))
  meta = json.load(open(os.path.join(args.dump, "meta.json")))
  cases = [c["seq"] for c in meta["cases"]]
  print(f"[mt] jax {jax.__version__} on {jax.devices()}")
  print(f"[mt] reference from transformers {meta['transformers_version']}, cases={cases}\n")

  results = []
  for scan_layers in (True, False):
    print(f"\n===== scan_layers={scan_layers} =====")
    # Build params once at the longest length; the parameter tree is length-independent.
    cfg0 = make_config(args.model_name, max(cases), scan_layers)
    params, diag = build_params_from_hf(cfg0, weights)
    check(
        f"[scan={scan_layers}] converter mapping covers every MaxText parameter",
        not diag["missing"],
        f"{len(diag['missing'])} unmapped of {diag['n_params']}: {diag['missing'][:8]}",
    )
    check(
        f"[scan={scan_layers}] converter mapping has no stale entries",
        not diag["extra"],
        f"{len(diag['extra'])} stale: {diag['extra'][:8]}",
    )

    model0, _ = build_linen_model(cfg0)
    if scan_layers:
      sharding_check(cfg0, model0, params)

    for seq in cases:
      cfg = make_config(args.model_name, seq, scan_layers)
      model, _ = build_linen_model(cfg)
      p, _ = build_params_from_hf(cfg, weights)
      ids = ref[f"ids_{seq}"]
      logits = np.asarray(run_model(model, p, ids))
      la, lr, ls = errs(logits, ref[f"logits_{seq}"])
      results.append((scan_layers, seq, "logits", la, lr, ls))
      check(
          f"[scan={scan_layers}] logits parity seq={seq}",
          la < args.atol and lr < args.rtol,
          f"max_abs={la:.3e} rel_to_scale={lr:.3e} rel_sig={ls:.3e}",
      )

      # The tanh softcap COMPRESSES error, so the capped logits are a weak gate.
      # Re-run with the multiplier and cap disabled: that is the hidden state's
      # error amplified by the head, i.e. the real numeric gate.
      cfg_raw = make_config(
          args.model_name,
          seq,
          scan_layers,
          # override_model_config is required to contradict the model yml.
          override_model_config=True,
          final_logits_soft_cap=0.0,
          logits_output_multiplier=1.0,
      )
      model_raw, _ = build_linen_model(cfg_raw)
      p_raw, _ = build_params_from_hf(cfg_raw, weights)
      raw = np.asarray(run_model(model_raw, p_raw, ids))
      ra, rr, rs = errs(raw, ref[f"logits_raw_{seq}"])
      results.append((scan_layers, seq, "raw_logits", ra, rr, rs))
      check(
          f"[scan={scan_layers}] RAW (uncapped, unscaled) logits parity seq={seq}",
          ra < args.atol and rr < args.rtol,
          f"max_abs={ra:.3e} rel_to_scale={rr:.3e} rel_sig={rs:.3e}",
      )

    # padded-row case: a non-block-divisible sequence inside a longer, padded row.
    # This is what the tunix backend actually feeds (bucketed / padded rows), and it
    # is the closest CPU analogue of the splash-kernel block padding on TPU.
    seq = 37
    padded_len = 128
    cfg = make_config(args.model_name, padded_len, scan_layers)
    model, _ = build_linen_model(cfg)
    p, _ = build_params_from_hf(cfg, weights)
    ids = np.zeros((1, padded_len), np.int32)
    ids[0, :seq] = ref[f"ids_{seq}"][0]
    seg = np.zeros((1, padded_len), np.int32)
    seg[0, :seq] = 1
    pos = np.concatenate([np.arange(seq), np.zeros(padded_len - seq, np.int32)])[None, :]
    logits = np.asarray(run_model(model, p, ids, jnp.asarray(pos, jnp.int32), jnp.asarray(seg)))
    la, lr, ls = errs(logits[:, :seq], ref[f"logits_{seq}"])
    check(
        f"[scan={scan_layers}] logits parity seq=37 padded into a {padded_len}-long row",
        la < args.atol and lr < args.rtol,
        f"max_abs={la:.3e} rel_to_scale={lr:.3e} rel_sig={ls:.3e}",
    )
    results.append((scan_layers, f"37-in-{padded_len}", "logits", la, lr, ls))

  # structural probes (scanned config; the layer class is the same either way)
  print("\n===== structural probes =====")
  cfg = make_config(args.model_name, meta["probe_seq"], True)
  layer_probes(cfg, weights, ref, args.atol, args.rtol)

  print("\n===== summary =====")
  print(f"{'scan':>6} {'seq':>14} {'what':>11} {'max_abs':>12} {'rel_to_scale':>13} {'rel_sig':>12}")
  for scan, seq, what, a, r, rs in results:
    print(f"{str(scan):>6} {str(seq):>14} {what:>11} {a:12.4e} {r:13.4e} {rs:12.4e}")
  worst_a = max(x[3] for x in results)
  worst_r = max(x[4] for x in results)
  print(f"\nWORST: max_abs={worst_a:.4e} (target < {args.atol})  "
        f"rel_to_scale={worst_r:.4e} (target < {args.rtol})")

  if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
      print("  - " + f)
    sys.exit(1)
  print("\nALL CHECKS PASSED")


if __name__ == "__main__":
  main()
