#!/usr/bin/env python3
"""Stage 1 of the Muse-Glimmer HF <-> MaxText parity harness: the REFERENCE side.

Builds a tiny random-weight ``MuseGlimmerTextModel`` (+ an untied ``lm_head``),
runs it in float32 on CPU for several sequence lengths, and dumps

  * ``weights.npz``  -- the state dict under the EXACT HuggingFace parameter names
                        from ``model.safetensors.index.json`` (text tower under
                        ``model.language_model.*`` plus a top-level ``lm_head.weight``),
                        so stage 2 can feed it through MaxText's real
                        ``PARAM_MAPPING`` / ``HOOK_FNS`` rather than a bespoke loader;
  * ``ref.npz``      -- final hidden states and logits per case;
  * ``meta.json``    -- the config and the case list.

MUST run in an isolated venv with ``transformers`` from git main -- Muse-Glimmer only
exists in the unreleased 5.15.0.dev0. Never install that into the SkyRL project env.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


# The tiny config. Every non-shape constant is identical to the released 30B so the
# parity traps (qk_scale_factor, post_norm_eps, output_multiplier, softcap) are all
# live; only the shapes shrink. Keeps the [S,S,S,F] cycle exactly once so both
# attention types and the NoPE layers are exercised.
TINY = dict(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=64,
    hidden_activation="silu",
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    post_norm_eps=1e-8,
    qk_scale_factor=3.87,
    output_multiplier=0.19611613513818404,
    final_logit_softcapping=20.0,
    sliding_window=8,
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=False,
)

# Sequence lengths. Deliberately includes lengths that are NOT multiples of the
# splash/flash query block size (512) nor powers of two, plus lengths on both sides
# of the sliding window (8) so the window mask actually binds.
CASES = [7, 24, 37, 100, 129]


def build_config():
  """Instantiate MuseGlimmerTextConfig with the explicit [S,S,S,F] / NoPE pattern."""
  try:
    from transformers import MuseGlimmerTextConfig  # type: ignore
  except ImportError:  # older layouts
    from transformers.models.muse_glimmer.configuration_muse_glimmer import (  # type: ignore
        MuseGlimmerTextConfig,
    )

  n = TINY["num_hidden_layers"]
  # HF's own default is "every 4th layer counted BACKWARD from the last is full
  # attention / NoPE", which for any n divisible by 4 is exactly `i % 4 == 3`.
  # We pass both lists explicitly so the harness never depends on that default.
  layer_types = ["full_attention" if (i % 4) == 3 else "sliding_attention" for i in range(n)]
  layer_rope_theta = [0 if (i % 4) == 3 else 500000.0 for i in range(n)]
  cfg = MuseGlimmerTextConfig(
      **TINY,
      layer_types=layer_types,
      layer_rope_theta=layer_rope_theta,
      rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
  )
  cfg._attn_implementation = "eager"  # deterministic, and the sliding mask is explicit
  return cfg


def random_init(model, lm_head, seed: int = 0):
  """Fill every parameter with something that makes each parity trap observable.

  In particular the four per-layer sandwich norms are ZERO-centred (HF applies
  ``n * (1 + w)``) while the final ``model.norm`` is ONE-centred (``n * w``). Init
  them differently on purpose: a converter that gets the ``(1 + w)`` convention
  wrong, or that "helpfully" adds 1 to the final norm, then fails loudly.
  """
  g = torch.Generator().manual_seed(seed)

  def rand(t, std, mean=0.0):
    with torch.no_grad():
      t.copy_(torch.empty(t.shape, dtype=torch.float32).normal_(mean, std, generator=g))

  centred_norms = (
      "input_layernorm.weight",
      "post_attention_layernorm.weight",
      "pre_feedforward_layernorm.weight",
      "post_feedforward_layernorm.weight",
  )
  for name, p in model.named_parameters():
    if name == "embed_tokens.weight":
      rand(p, 1.0)  # big, so the parameter-free embedding RMSNorm actually bites
    elif any(name.endswith(s) for s in centred_norms):
      rand(p, 0.10)  # zero-centred -> exercises (1 + w)
    elif name == "norm.weight":
      rand(p, 0.10, mean=1.0)  # one-centred -> plain n * w
    else:
      rand(p, 0.02)
  rand(lm_head.weight, 0.02)


def hf_names(model, lm_head):
  """State dict under the exact names in the released ``model.safetensors.index.json``.

  The checkpoint is the multimodal ``MuseGlimmerForConditionalGeneration``, so the
  text tower sits under ``model.language_model.*`` and ``lm_head.weight`` is a
  SEPARATE top-level tensor (``tie_word_embeddings: false``).
  """
  out = {}
  for name, p in model.state_dict().items():
    out[f"model.language_model.{name}"] = p.detach().float().numpy()
  out["lm_head.weight"] = lm_head.weight.detach().float().numpy()
  return out


def apply_head(hidden, lm_head, cfg):
  """lm_head -> * output_multiplier -> tanh softcap. Order matters (spec trap 7)."""
  logits = lm_head(hidden)
  logits = logits * cfg.output_multiplier
  logits = logits / cfg.final_logit_softcapping
  logits = torch.tanh(logits)
  logits = logits * cfg.final_logit_softcapping
  return logits


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  import transformers

  try:
    from transformers import MuseGlimmerTextModel  # type: ignore
  except ImportError:
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (  # type: ignore
        MuseGlimmerTextModel,
    )

  cfg = build_config()
  torch.manual_seed(args.seed)
  model = MuseGlimmerTextModel(cfg).to(torch.float32).eval()
  lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(torch.float32)
  random_init(model, lm_head, seed=args.seed)

  # --- sanity assertions on the reference itself -----------------------------
  assert cfg.layer_types == ["sliding_attention"] * 3 + ["full_attention"]
  assert cfg.layer_rope_theta == [500000.0, 500000.0, 500000.0, 0]
  # qk-norm is parameter-free -> must NOT be in the state dict
  bad = [k for k in model.state_dict() if "q_norm" in k or "k_norm" in k or "qk_norm" in k]
  assert not bad, f"unexpected qk-norm parameters in the reference: {bad}"
  # lm_head is untied in the released config
  assert cfg.tie_word_embeddings is False

  weights = hf_names(model, lm_head)

  rng = np.random.default_rng(args.seed + 1)
  ref = {}
  case_meta = []
  with torch.no_grad():
    for seq in CASES:
      ids = rng.integers(0, cfg.vocab_size, size=(1, seq)).astype(np.int64)
      t_ids = torch.from_numpy(ids)
      out = model(input_ids=t_ids, use_cache=False)
      h = out.last_hidden_state
      logits = apply_head(h, lm_head, cfg)
      ref[f"ids_{seq}"] = ids
      ref[f"hidden_{seq}"] = h.float().numpy()
      ref[f"logits_{seq}"] = logits.float().numpy()
      # Pre-multiplier, pre-softcap head output. The tanh softcap compresses
      # errors, so the capped logits are a WEAK gate; this raw version is the
      # hidden-state error amplified by the head and is the real numeric gate.
      ref[f"logits_raw_{seq}"] = lm_head(h).float().numpy()
      case_meta.append({"seq": seq})
      print(f"[hf] seq={seq:4d} hidden={tuple(h.shape)} |h|max={h.abs().max():.4f} "
            f"logits|max={logits.abs().max():.4f}")

    # --- per-layer probes, mirrored exactly on the MaxText side --------------
    # Same fixed hidden-state input into a single decoder layer, so the window /
    # NoPE assertions are about ONE layer and are not smeared by later layers.
    probe_seq = 24
    probe_h = torch.from_numpy(rng.normal(0, 1.0, size=(1, probe_seq, cfg.hidden_size)).astype(np.float32))
    ref["probe_hidden_in"] = probe_h.numpy()
    # perturbed copy: token 5 replaced
    probe_h2 = probe_h.clone()
    probe_h2[0, 5, :] = torch.from_numpy(rng.normal(0, 1.0, size=(cfg.hidden_size,)).astype(np.float32))
    ref["probe_hidden_in_perturbed"] = probe_h2.numpy()

    pos = torch.arange(probe_seq)[None, :]
    cos, sin = model.rotary_emb(probe_h, pos)
    # SPEC trap 12: a CONSTANT shift of position_ids is not a NoPE test -- RoPE is
    # relative, so a rope layer is shift-invariant too. Discriminate with a change
    # in position DIFFERENCES: a stride-3 ramp.
    pos_stride3 = torch.arange(probe_seq)[None, :] * 3
    cos_s3, sin_s3 = model.rotary_emb(probe_h, pos_stride3)
    ref["probe_positions"] = pos.numpy()
    ref["probe_positions_stride3"] = pos_stride3.numpy()

    # SPEC trap 11: HF's `create_causal_mask` derives packed-sequence boundaries
    # from position_ids, so a stride-3 ramp would collapse its mask to the
    # diagonal. We therefore never let HF build the mask -- every probe calls the
    # decoder layer directly with an EXPLICIT index-based mask, which is exactly
    # what MaxText's `generate_attention_mask` builds (broadcasted_iota over
    # indices, NOT over decoder_positions).
    neg = torch.finfo(torch.float32).min
    q = torch.arange(probe_seq)[:, None]
    k = torch.arange(probe_seq)[None, :]
    causal = torch.where(k <= q, 0.0, neg)[None, None]
    sliding = torch.where((k <= q) & (k > q - cfg.sliding_window), 0.0, neg)[None, None]

    def run_layer(layer_idx, h, position_embeddings, mask, position_ids=pos):
      return model.layers[layer_idx](
          h, position_embeddings=position_embeddings, attention_mask=mask, position_ids=position_ids
      )

    for tag, layer_idx, mask, pe in (
        ("sliding", 0, sliding, (cos, sin)),
        ("full", 3, causal, None),  # NoPE: the model passes position_embeddings=None
    ):
      ref[f"probe_{tag}_base"] = run_layer(layer_idx, probe_h, pe, mask).float().numpy()
      ref[f"probe_{tag}_perturbed"] = run_layer(layer_idx, probe_h2, pe, mask).float().numpy()

    # stride-3 positions, same explicit masks. The NoPE (full) layer must be
    # bit-identical to its contiguous-position run; the sliding layer must not.
    ref["probe_full_stride3"] = run_layer(3, probe_h, None, causal, pos_stride3).float().numpy()
    ref["probe_sliding_stride3"] = (
        run_layer(0, probe_h, (cos_s3, sin_s3), sliding, pos_stride3).float().numpy()
    )

  np.savez(os.path.join(args.out, "weights.npz"), **weights)
  np.savez(os.path.join(args.out, "ref.npz"), **ref)
  meta = {
      "transformers_version": transformers.__version__,
      "config": cfg.to_dict(),
      "tiny": TINY,
      "cases": case_meta,
      "probe_seq": 24,
      "probe_perturb_index": 5,
      "seed": args.seed,
  }
  with open(os.path.join(args.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

  print(f"\n[hf] transformers {transformers.__version__}")
  print(f"[hf] wrote {len(weights)} tensors + {len(ref)} arrays to {args.out}")
  # Show the exact HF names so the mapping in param_mapping.py can be eyeballed.
  for k in sorted(weights)[:6]:
    print(f"[hf]   {k}  {weights[k].shape}")
  print(f"[hf]   ... ({len(weights)} total)")


if __name__ == "__main__":
  main()
