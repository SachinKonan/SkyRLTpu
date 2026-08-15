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
"""Emit a PEFT-format LoRA adapter for Muse-Glimmer, shaped like the ones the
RL loop uploads.

Used by the on-slice smoke: an adapter is only proof of a working LoRA stack if
loading it *changes the output*, so ``lora_B`` is deliberately non-zero (a real
PEFT init would zero it, and a zero-B adapter is indistinguishable from an
adapter that was silently dropped).

Module naming follows this port's tree, not raw HF: the attention gate is
``attn_gate_proj``.  ``q/k/v_proj`` and ``gate/up_proj`` stay unpacked here --
vLLM's ``packed_modules_mapping`` routes them into ``qkv_proj`` / ``gate_up_proj``
at load time, which is exactly the path an RL adapter takes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# leaf name -> (out_features_expr, in_features_expr) in terms of the config
_SHAPES = {
    "q_proj": lambda c: (c.num_attention_heads * c.head_dim, c.hidden_size),
    "k_proj": lambda c: (c.num_key_value_heads * c.head_dim, c.hidden_size),
    "v_proj": lambda c: (c.num_key_value_heads * c.head_dim, c.hidden_size),
    "o_proj": lambda c: (c.hidden_size, c.num_attention_heads * c.head_dim),
    "attn_gate_proj":
    lambda c: (c.num_attention_heads * c.head_dim, c.hidden_size),
    "gate_proj": lambda c: (c.intermediate_size, c.hidden_size),
    "up_proj": lambda c: (c.intermediate_size, c.hidden_size),
    "down_proj": lambda c: (c.hidden_size, c.intermediate_size),
}

_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--scale", type=float, default=0.02,
                    help="lora_B magnitude; must be non-zero so the adapter "
                         "provably changes the output")
    ap.add_argument(
        "--targets",
        default="q_proj,k_proj,v_proj,o_proj,attn_gate_proj,"
        "gate_proj,up_proj,down_proj")
    ap.add_argument("--layers", default="",
                    help="comma-separated layer indices; default = all")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    cfg_path = Path(args.model_dir) / "config.json"
    raw = json.loads(cfg_path.read_text())
    text = raw.get("text_config", raw)

    class C:
        hidden_size = text["hidden_size"]
        num_attention_heads = text["num_attention_heads"]
        num_key_value_heads = text["num_key_value_heads"]
        head_dim = text.get("head_dim",
                            text["hidden_size"] // text["num_attention_heads"])
        intermediate_size = text["intermediate_size"]
        num_hidden_layers = text["num_hidden_layers"]

    targets = [t for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in _SHAPES]
    if unknown:
        raise SystemExit(f"unknown target modules: {unknown}")

    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = list(range(C.num_hidden_layers))

    dtype = getattr(torch, args.dtype)
    g = torch.Generator().manual_seed(args.seed)
    tensors: dict[str, torch.Tensor] = {}
    for i in layers:
        for leaf in targets:
            out_f, in_f = _SHAPES[leaf](C)
            sub = "self_attn" if leaf in _ATTN else "mlp"
            base = (f"base_model.model.model.layers.{i}.{sub}.{leaf}")
            a = torch.randn(args.rank, in_f, generator=g) * (in_f**-0.5)
            b = torch.randn(out_f, args.rank, generator=g) * args.scale
            tensors[f"{base}.lora_A.weight"] = a.to(dtype)
            tensors[f"{base}.lora_B.weight"] = b.to(dtype)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": args.rank,
                "lora_alpha": args.alpha,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": targets,
                "fan_in_fan_out": False,
                "modules_to_save": None,
            },
            indent=2))

    n_params = sum(t.numel() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors ({n_params:,} params) for "
          f"{len(layers)} layers x {len(targets)} targets -> {out}")
    print(f"targets: {targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
