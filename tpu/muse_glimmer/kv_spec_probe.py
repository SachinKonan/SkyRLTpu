#!/usr/bin/env python3
"""What does the KV-cache-group change actually do, before a slice is spent?

Calls the REAL `KVCacheManager.get_kv_cache_spec` against a duck-typed runner
(everything it touches is config, a mesh and a dtype), then hands the resulting
spec dict to vLLM's own `get_kv_cache_config` -- the function that decides
`num_kv_cache_groups`.  So the group count, the per-group layer lists and the
resulting block count are all answerable on a CPU node.

Two model sources:
  --model-dir   a real HF checkpoint directory (muse-glimmer)
  --synthetic   a config built from literal attribute values, for models whose
                weights are not staged here (the gemma-4 variants)

The point is to know, before booting an engine, whether the gate fires, how
many groups vLLM forms, and therefore whether the served model's forward pass
-- which may index `kv_caches` by layer index -- is still being handed the
layout it was written against.
"""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace


def force_cpu_platform() -> str:
    import vllm.platforms as vp
    vp.builtin_platform_plugins["cpu"] = \
        lambda: "vllm.platforms.cpu.CpuPlatform"
    vp._current_platform = None
    vp._init_trace = ""
    return type(vp.current_platform).__name__


SYNTHETIC = {
    # Gemma-4 E2B/E4B: num_global_key_value_heads / global_head_dim are absent
    # (None) -> the uniform-dims gate FIRES.
    "gemma4-e2b": dict(num_hidden_layers=30,
                       num_attention_heads=8,
                       num_key_value_heads=4,
                       head_dim=256,
                       hidden_size=2048,
                       sliding_window=512,
                       num_global_key_value_heads=None,
                       global_head_dim=None,
                       pattern=("sliding_attention", ) * 5 +
                       ("full_attention", )),
    # Gemma-4 26B/31B: both set and DIFFERENT -> the gate does not fire.
    "gemma4-31b": dict(num_hidden_layers=48,
                       num_attention_heads=32,
                       num_key_value_heads=16,
                       head_dim=256,
                       hidden_size=4096,
                       sliding_window=1024,
                       num_global_key_value_heads=4,
                       global_head_dim=512,
                       pattern=("sliding_attention", ) * 5 +
                       ("full_attention", )),
    # Muse-Glimmer-30B, from config.json, for a weightless cross-check.
    "muse-glimmer": dict(num_hidden_layers=52,
                         num_attention_heads=32,
                         num_key_value_heads=2,
                         head_dim=128,
                         hidden_size=6656,
                         sliding_window=2048,
                         num_global_key_value_heads=None,
                         global_head_dim=None,
                         pattern=("sliding_attention", ) * 3 +
                         ("full_attention", )),
}


def config_from_json(path: str) -> dict:
    """Read the REAL config.json into the synthetic builder's field dict.

    vLLM's own `ModelConfig` cannot be constructed for gemma-4 in this CPU
    venv: transformers' heterogeneity guard raises
    `AmbiguousGlobalPerLayerAttributeError` on `head_dim` before any of our
    code runs.  That is a transformers/vLLM-version artefact of dry-running on
    CPU, not anything to do with the change under test -- and
    `get_kv_cache_spec` only ever reads the handful of attributes below, so
    feeding it the real values directly is faithful for this question.
    """
    import json
    c = json.load(open(path))
    t = c.get("text_config", c)
    lt = t.get("layer_types") or []
    return dict(num_hidden_layers=t["num_hidden_layers"],
                num_attention_heads=t.get("num_attention_heads"),
                num_key_value_heads=t.get("num_key_value_heads"),
                head_dim=t.get("head_dim"),
                hidden_size=t.get("hidden_size"),
                sliding_window=t.get("sliding_window"),
                num_global_key_value_heads=t.get("num_global_key_value_heads"),
                global_head_dim=t.get("global_head_dim"),
                pattern=tuple(lt) if lt else ("full_attention", ),
                layer_types=lt)


def build_synthetic(name: str, tp: int, block_size: int,
                    max_model_len: int = 4096):
    from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
    from vllm.config.parallel import ParallelConfig

    spec = SYNTHETIC[name] if name in SYNTHETIC else config_from_json(name)
    L = spec["num_hidden_layers"]
    layer_types = (list(spec["layer_types"]) if spec.get("layer_types") else
                   [spec["pattern"][i % len(spec["pattern"])]
                    for i in range(L)])
    text_config = SimpleNamespace(
        num_hidden_layers=L,
        num_attention_heads=spec["num_attention_heads"],
        num_key_value_heads=spec["num_key_value_heads"],
        head_dim=spec["head_dim"],
        hidden_size=spec["hidden_size"],
        sliding_window=spec["sliding_window"],
        num_global_key_value_heads=spec["num_global_key_value_heads"],
        global_head_dim=spec["global_head_dim"],
        layer_types=layer_types,
    )
    model_config = SimpleNamespace(
        hf_text_config=text_config,
        hf_config=text_config,
        use_mla=False,
        architectures=[f"{name}ForCausalLM"],
        get_total_num_kv_heads=lambda: spec["num_key_value_heads"],
        get_head_size=lambda: spec["head_dim"],
        get_num_layers=lambda pc: L,
        dtype="bfloat16",
    )
    parallel_config = ParallelConfig(tensor_parallel_size=tp)
    cache_config = CacheConfig(block_size=block_size)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        scheduler_config=SchedulerConfig(max_model_len=max_model_len,
                                        is_encoder_decoder=False),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
    )
    del VllmConfig
    return vllm_config, model_config, parallel_config, cache_config


def build_real(model_dir: str, tp: int, block_size: int, max_model_len: int):
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.parallel import ParallelConfig

    force_cpu_platform()
    mc = ModelConfig(model=model_dir,
                     tokenizer=model_dir,
                     dtype="bfloat16",
                     seed=0,
                     max_model_len=max_model_len)
    vc = VllmConfig(model_config=mc,
                    parallel_config=ParallelConfig(tensor_parallel_size=tp),
                    cache_config=CacheConfig(block_size=block_size))
    vc.compilation_config.static_forward_context = {}
    return vc, vc.model_config, vc.parallel_config, vc.cache_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--available-gib", type=float, default=300.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import jax
    import numpy as np

    if args.synthetic:
        vc, mc, pc, cc = build_synthetic(args.synthetic, args.tp,
                                         args.block_size,
                                         args.max_model_len)
        label = args.synthetic
    else:
        vc, mc, pc, cc = build_real(args.model_dir, args.tp, args.block_size,
                                    args.max_model_len)
        label = args.model_dir

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("data", "attn_dp", "model", "expert"))

    from tpu_inference.runner.kv_cache_manager import KVCacheManager

    runner = SimpleNamespace(vllm_config=vc,
                             model_config=mc,
                             parallel_config=pc,
                             cache_config=cc,
                             speculative_config=None,
                             mesh=mesh,
                             # vLLM sizes pages with torch's element_size(),
                             # so the spec dtype has to be a torch dtype even
                             # though the arrays themselves are JAX.
                             kv_cache_dtype=__import__("torch").bfloat16)
    mgr = KVCacheManager(runner)
    spec = mgr.get_kv_cache_spec()

    kinds = {}
    for name, s in spec.items():
        k = type(s).__name__
        w = getattr(s, "sliding_window", None)
        kinds.setdefault((k, w), []).append(name)
    print(f"\n=== {label} ===")
    print(f"layers with a spec: {len(spec)}")
    for (k, w), names in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:20s} sliding_window={w!s:8s} n={len(names):3d} "
              f"e.g. {names[:3]}")

    result = {
        "label": label,
        "n_layer_specs": len(spec),
        "spec_kinds": {
            f"{k}|{w}": len(v)
            for (k, w), v in kinds.items()
        },
    }

    # --- what vLLM would do with that spec -------------------------------
    try:
        # vLLM 0.23 splits this in two: group the layers, then size the pool.
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups, get_kv_cache_groups)
        avail = int(args.available_gib * 2**30)
        groups_spec = get_kv_cache_groups(vc, spec)
        cfg = get_kv_cache_config_from_groups(vc, groups_spec, avail)
        groups = [{
            "index": i,
            "spec": type(g.kv_cache_spec).__name__,
            "sliding_window": getattr(g.kv_cache_spec, "sliding_window", None),
            "n_layers": len(g.layer_names),
            "layers": g.layer_names[:4],
        } for i, g in enumerate(cfg.kv_cache_groups)]
        result["num_kv_cache_groups"] = len(cfg.kv_cache_groups)
        result["num_blocks"] = cfg.num_blocks
        result["num_kv_cache_tensors"] = len(cfg.kv_cache_tensors)
        result["kv_token_capacity"] = cfg.num_blocks * args.block_size
        result["groups"] = groups
        print(f"  -> num_kv_cache_groups = {len(cfg.kv_cache_groups)}, "
              f"num_blocks = {cfg.num_blocks}, "
              f"kv tokens = {cfg.num_blocks * args.block_size} "
              f"(at {args.available_gib} GiB)")
        for g in groups:
            print(f"     group {g['index']}: {g['spec']:20s} "
                  f"window={g['sliding_window']} layers={g['n_layers']} "
                  f"{g['layers']}")
        print(f"  -> len(kv_cache_tensors) = {len(cfg.kv_cache_tensors)} "
              f"(the model is handed this many arrays; a model that indexes "
              f"kv_caches[layer_idx] needs {len(spec)})")
    except Exception as exc:  # noqa: BLE001
        result["kv_cache_config_error"] = repr(exc)
        print(f"  -> get_kv_cache_config FAILED: {exc!r}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
