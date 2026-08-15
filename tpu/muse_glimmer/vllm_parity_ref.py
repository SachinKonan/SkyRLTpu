#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HF side of the **torch/vLLM** Muse-Glimmer parity gate: dump the reference.

Runs in the throwaway `transformers @ main` venv (transformers 5.16.0.dev0),
which cannot hold vllm at the same time -- vllm 0.23.0 pins an older
transformers, and the SkyRL repo's `override-dependencies` pin makes an
in-place upgrade silently wrong (see tpu/muse_glimmer/README.md).  So the two
halves talk through files:

    vllm_parity_ref.py   (hfvenv)   -> tiny_ref.npz + tiny_weights.safetensors
    vllm_parity_check.py (mgvllm)   <- reads both, builds the torch model,
                                        compares.

The tiny config keeps the real [S, S, S, F] layer pattern, the real
`layer_rope_theta` NoPE marker on the full layer, and the real
`qk_scale_factor` / `output_multiplier` / softcap, so every parity trap in
SPEC.md is live at 4 layers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Identical to tpu/muse_glimmer/parity_check.py::TINY so the two gates (JAX and
# torch) are comparable number for number.
TINY = dict(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,  # -> [S, S, S, F], layer_rope_theta [t, t, t, 0]
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

# A second shape whose kv-head count is *smaller than* a plausible TP size, so
# the checkpoint->QKVParallelLinear replication path is exercised on CPU as
# well (2 kv heads, as the 30B has).  TP is still 1 on CPU; the real 2-vs-TP4
# replication can only be proven on a slice, but this at least pins the GQA
# grouping (q heads 0-3 -> kv0, 4-7 -> kv1).
TINY_GQA = dict(
    TINY,
    hidden_size=256,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=32,
)


def build_text_config(variant: str):
    from transformers import MuseGlimmerTextConfig
    kwargs = dict(TINY if variant == "tiny" else TINY_GQA)
    cfg = MuseGlimmerTextConfig(**kwargs)
    cfg.rope_parameters = dict(cfg.rope_parameters)
    cfg.rope_parameters["rope_theta"] = REAL_ROPE_THETA
    cfg.layer_rope_theta = [
        0.0 if t == 0 else REAL_ROPE_THETA for t in cfg.layer_rope_theta
    ]
    return cfg


def randomise_(module, seed: int = 0):
    """Same scheme as parity_check.py: keep the two norm flavours
    distinguishable while making every other tensor non-degenerate."""
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif "layernorm" in name or name.endswith("norm.weight"):
                base = 1.0 if name.endswith(
                    "norm.weight") and "layernorm" not in name else 0.0
                p.copy_(base + torch.randn(p.shape, generator=g) * 0.3)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.08)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=("tiny", "gqa"), default="tiny")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--extra-seq-lens", type=str, default="7,23,37,129")
    args = ap.parse_args()

    import transformers
    from safetensors.torch import save_file
    from transformers import MuseGlimmerTextModel

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cfg = build_text_config(args.variant)
    cfg._attn_implementation = "eager"
    print(f"transformers {transformers.__version__}  torch {torch.__version__}")
    print(f"variant={args.variant}  layer_types={cfg.layer_types}")
    print(f"layer_rope_theta={cfg.layer_rope_theta}  window={cfg.sliding_window}")

    hf = randomise_(MuseGlimmerTextModel(cfg), seed=args.seed).eval().float()
    state_dict = {k: v.detach().clone() for k, v in hf.state_dict().items()}

    # Untied lm_head, exactly as the 30B ships one (spec trap 8).
    lm_head_w = torch.randn(cfg.vocab_size,
                            cfg.hidden_size,
                            generator=torch.Generator().manual_seed(args.seed +
                                                                    7)) * 0.05
    state_dict["lm_head.weight"] = lm_head_w

    # Save the checkpoint under the *30B's* key layout, so the torch model's
    # load_weights() is exercised on the same names it will see in production
    # (`model.language_model.*` + a top-level `lm_head.weight`).
    ckpt = {}
    for k, v in state_dict.items():
        if k == "lm_head.weight":
            ckpt[k] = v.contiguous()
        else:
            ckpt[f"model.language_model.{k}"] = v.contiguous()
    save_file(ckpt, str(args.out_dir / f"{args.variant}_weights.safetensors"))

    gen = torch.Generator().manual_seed(args.seed)
    out = {}
    lens = [args.seq_len] + [
        int(x) for x in args.extra_seq_lens.split(",") if x.strip()
    ]
    for L in lens:
        ids = torch.randint(0, cfg.vocab_size, (args.batch, L), generator=gen)
        with torch.no_grad():
            hidden = hf(input_ids=ids, use_cache=False).last_hidden_state.float()
            logits = hidden @ lm_head_w.t()
            logits = logits * cfg.output_multiplier
            logits = torch.tanh(logits / cfg.final_logit_softcapping)
            logits = logits * cfg.final_logit_softcapping
        out[f"T{L}/ids"] = ids.numpy().astype(np.int64)
        out[f"T{L}/hidden"] = hidden.numpy()
        out[f"T{L}/logits"] = logits.numpy()

    # NoPE discrimination (spec trap 12): rotate positions by a *non-constant*
    # amount and confirm the full layer's output is unchanged while a sliding
    # layer's is not.  Done here on the HF side so the torch side is compared
    # against a reference that provably distinguishes the two.
    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=gen)
    out["nope/ids"] = ids.numpy().astype(np.int64)

    np.savez(args.out_dir / f"{args.variant}_ref.npz", **out)

    # A real on-disk model directory (config.json only) so the torch side can
    # build a genuine `ModelConfig` through vLLM's own constructor rather than
    # hand-populating one -- the hand-populated version drifts from vLLM's
    # internals every release.  The architecture string is the checkpoint's
    # real one, so the registry lookup is exercised too.
    model_dir = args.out_dir / f"{args.variant}_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = cfg.to_dict()
    cfg_dict["architectures"] = ["MuseGlimmerForConditionalGeneration"]
    (model_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2,
                                                      default=str))

    meta = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "variant": args.variant,
        "seed": args.seed,
        "batch": args.batch,
        "seq_lens": lens,
        "config": cfg.to_dict(),
        "n_ckpt_tensors": len(ckpt),
    }
    (args.out_dir / f"{args.variant}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))
    print(f"wrote {len(ckpt)} tensors + {len(out)} reference arrays to "
          f"{args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
