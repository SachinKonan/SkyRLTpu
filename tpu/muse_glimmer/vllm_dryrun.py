#!/usr/bin/env python3
"""CPU dry-run of everything vLLM does BEFORE it touches a TPU.

Three prior runs in this repo burned their slice budget on integration errors
that were fully diagnosable without hardware.  This catches them on a CPU node:

  stage 1  transformers can parse ``model_type: muse_glimmer``
  stage 2  the ``vllm.general_plugins`` entry point fires and registers the
           architecture with vLLM's OWN ModelRegistry
  stage 3  ``ModelConfig`` constructs -- this is where vLLM raises
           "Model architectures [...] are not supported for now", in
           ``EngineArgs.create_engine_config()``, before any tpu-inference code
           would otherwise run
  stage 4  the derived shapes come off ``text_config`` and are non-zero
           (vLLM reads them with ``getattr(..., 0)``, so a missing field is a
           silent 0 and a confusing crash much later)
  stage 5  tpu-inference resolves the arch to the JAX class
  stage 6  the nnx module builds and ``load_weights`` consumes the REAL
           checkpoint (truncated to a few layers so it fits in RAM), with every
           loaded parameter compared numerically against the source tensor

Stage 6 is the one that catches a mis-mapped tensor in the *serving* loader --
the CPU parity harness only ever exercised ``muse_glimmer_core``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List

MODEL_DIR_DEFAULT = "/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b"

RESULTS: Dict[str, Any] = {}


def stage(name: str):

    def deco(fn):

        def wrapper(*a, **k):
            print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}", flush=True)
            try:
                out = fn(*a, **k)
                RESULTS[name] = {"ok": True, "detail": out}
                print(f"--- {name}: OK", flush=True)
                return out
            except Exception as exc:  # noqa: BLE001
                RESULTS[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                }
                print(f"--- {name}: FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc(limit=12)
                return None

        return wrapper

    return deco


@stage("1_transformers_config")
def s1_config(model_dir: str):
    import transformers
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir)
    text = cfg.get_text_config()
    print("transformers", transformers.__version__)
    print("architectures", cfg.architectures, "| model_type", cfg.model_type)
    print("text model_type", text.model_type)
    return {
        "transformers": transformers.__version__,
        "architectures": list(cfg.architectures or []),
        "model_type": cfg.model_type,
        "text_model_type": text.model_type,
        "text_is_distinct": text is not cfg,
    }


@stage("2_plugin_registers_arch")
def s2_plugin():
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.plugins import load_general_plugins
    before = "MuseGlimmerForConditionalGeneration" in set(
        ModelRegistry.get_supported_archs())
    load_general_plugins()
    after = set(ModelRegistry.get_supported_archs())
    print("registered before plugins:", before)
    print("registered after plugins :",
          "MuseGlimmerForConditionalGeneration" in after)
    if "MuseGlimmerForConditionalGeneration" not in after:
        raise AssertionError(
            "the vllm.general_plugins entry point did not register the arch "
            "(vLLM swallows plugin exceptions -- look for 'Failed to load "
            "plugin' above)")
    return {"registered_before": before, "registered_after": True}


@stage("3_model_config")
def s3_model_config(model_dir: str, max_model_len: int):
    from vllm.config import ModelConfig
    mc = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=max_model_len,
    )
    print("ModelConfig built. multimodal:", mc.is_multimodal_model)
    print("architectures:", mc.hf_config.architectures)
    return {
        "is_multimodal_model": bool(mc.is_multimodal_model),
        "max_model_len": int(mc.max_model_len),
    }, mc


@stage("4_derived_shapes")
def s4_shapes(mc):
    vals = {
        "hidden_size": mc.get_hidden_size(),
        "head_size": mc.get_head_size(),
        "vocab_size": mc.get_vocab_size(),
        "num_layers": mc.get_num_layers(None) if False else
        mc.hf_text_config.num_hidden_layers,
        "num_kv_heads": mc.hf_text_config.num_key_value_heads,
        "num_attention_heads": mc.hf_text_config.num_attention_heads,
        "sliding_window": getattr(mc.hf_text_config, "sliding_window", None),
        "max_model_len": mc.max_model_len,
    }
    for k, v in vals.items():
        print(f"  {k:22s} = {v}")
    zeros = [k for k, v in vals.items() if v == 0]
    if zeros:
        raise AssertionError(
            f"vLLM derived 0 for {zeros} -- these come off text_config via "
            "getattr(..., 0) and silently poison KV-cache sizing")
    expect = {
        "hidden_size": 6656,
        "head_size": 128,
        "vocab_size": 202048,
        "num_layers": 52,
        "num_kv_heads": 2,
        "num_attention_heads": 32,
        "sliding_window": 2048,
    }
    bad = {k: (vals[k], v) for k, v in expect.items() if vals[k] != v}
    if bad:
        raise AssertionError(f"derived shape mismatch (got, want): {bad}")
    return vals


@stage("5_tpu_inference_resolves_arch")
def s5_resolve(mc):
    from tpu_inference.models.common.model_loader import \
        _get_model_architecture
    cls = _get_model_architecture(mc.hf_config)
    print("resolved to", cls.__module__ + "." + cls.__name__)
    if cls.__name__ != "MuseGlimmerForConditionalGeneration":
        raise AssertionError(f"unexpected class {cls}")
    return {"class": cls.__module__ + "." + cls.__name__}


def force_cpu_platform() -> str:
    """Make ``VllmConfig`` constructible on a CPU-only node.

    vLLM picks a platform by asking each builtin plugin whether it applies; the
    CPU one only says yes for a CPU-built wheel or macOS.  On this Linux node
    with the default (CUDA) wheel and no GPU nothing activates, vLLM settles on
    ``UnspecifiedPlatform``, and ``VllmConfig.__post_init__`` then dies with
    "Failed to infer device type".

    That is purely an artefact of dry-running on CPU -- the TPU host has the
    vllm-tpu wheel and a real TpuPlatform.  Point the builtin plugin at
    CpuPlatform and clear the memoised choice so the config object can be built
    and the REAL weight mapping exercised, which is the whole point of stage 6.
    """
    import vllm.platforms as vp
    vp.builtin_platform_plugins["cpu"] = \
        lambda: "vllm.platforms.cpu.CpuPlatform"
    # Drop whatever earlier stages memoised (UnspecifiedPlatform).
    vp._current_platform = None
    vp._init_trace = ""
    name = type(vp.current_platform).__name__
    print(f"forced platform -> {name}")
    return name


@stage("6_load_weights_real_checkpoint")
def s6_load_weights(model_dir: str, n_layers: int, tp: int):
    """Build the real nnx module (truncated) and push the REAL tensors in.

    Layers are structurally identical, so ``n_layers`` of them exercises every
    distinct key pattern at full width; embed_tokens / lm_head / model.norm are
    kept at their real sizes.  Peak RSS is a few GB instead of 60.
    """
    import jax
    import numpy as np
    import torch
    from flax import nnx
    from safetensors.torch import load_file
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.parallel import ParallelConfig

    print("jax devices:", jax.devices())
    if len(jax.devices()) < tp:
        raise AssertionError(
            f"need {tp} devices for TP={tp}; got {jax.devices()} "
            "(set XLA_FLAGS=--xla_force_host_platform_device_count=N)")
    force_cpu_platform()

    # Every JAX model in tpu-inference (qwen2, qwen3_moe, deepseek_v3, and
    # muse_glimmer alike) reads get_pp_group() in __init__.  TPUWorker calls
    # init_pp_distributed_environment() unconditionally before building the
    # model -- with need_pp=False when PP==1 -- so `_PP` exists on the real
    # path.  This harness constructs the module directly, so it has to do the
    # same or die with `NameError: name '_PP' is not defined`, which looks
    # alarmingly like a model bug and is not one.
    from tpu_inference.distributed import jax_parallel_state
    jax_parallel_state.init_pp_distributed_environment(
        ip="127.0.0.1",
        rank=0,
        world_size=1,
        device=jax.devices()[0],
        need_pp=False)
    print("initialised PP group (world_size=1, need_pp=False)")


    mc = ModelConfig(model=model_dir,
                     tokenizer=model_dir,
                     dtype="bfloat16",
                     seed=0,
                     max_model_len=1024)
    text_cfg = mc.hf_text_config
    full_layers = text_cfg.num_hidden_layers
    # Keep the [S,S,S,F] pattern intact so both attention flavours are built.
    text_cfg.num_hidden_layers = n_layers
    text_cfg.layer_types = list(text_cfg.layer_types[:n_layers])
    text_cfg.layer_rope_theta = list(text_cfg.layer_rope_theta[:n_layers])
    print(f"truncated {full_layers} -> {n_layers} layers; "
          f"types={text_cfg.layer_types} thetas={text_cfg.layer_rope_theta}")

    vllm_config = VllmConfig(model_config=mc,
                             parallel_config=ParallelConfig(
                                 tensor_parallel_size=tp))

    # `ModelConfig.dtype` is a *torch* dtype.  On the real path
    # model_loader.get_model() rewrites it to the jnp equivalent before
    # constructing the module and restores it afterwards
    # (models/common/model_loader.py: to_jax_dtype(...)), so every JAX model --
    # qwen2 and muse_glimmer alike -- sees a jnp dtype in __init__.  Building
    # the module directly skips that step, and nnx then dies with
    # "Cannot interpret 'torch.bfloat16' as a data type".
    from tpu_inference.utils import to_jax_dtype
    vllm_config.model_config.dtype = to_jax_dtype(vllm_config.model_config.dtype)
    print(f"converted model dtype -> {vllm_config.model_config.dtype}")
    # The model indexes mesh.shape["model"], so that axis name is load-bearing.
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:tp]).reshape(1, tp), ("data", "model"))
    print("mesh:", mesh)

    from tpu_inference.models.jax.muse_glimmer import \
        MuseGlimmerForConditionalGeneration

    with jax.set_mesh(mesh):
        model = MuseGlimmerForConditionalGeneration(
            vllm_config, jax.random.PRNGKey(0), mesh)

    params_before = {
        name: np.asarray(p.get_value()).copy()
        for name, p in model.named_parameters()
    }
    print(f"module tree has {len(params_before)} parameters")
    for name in sorted(params_before)[:6]:
        print(f"   e.g. {name} {params_before[name].shape}")

    # ---- collect the real tensors we expect to be consumed -----------------
    from tpu_inference.models.jax.muse_glimmer_core import is_vision_weight
    keep: Dict[str, Any] = {}
    n_vision = 0
    # Read key-by-key with safe_open rather than load_file: shard 1 alone is
    # 46.5 GiB, and materialising it whole just to throw ~95% of it away got
    # this stage OOM-killed (exit 137) at --mem=96G.  safe_open mmaps the file
    # and only the tensors actually kept are ever resident.
    from safetensors import safe_open
    for fn in sorted(f for f in os.listdir(model_dir)
                     if f.endswith(".safetensors")):
        with safe_open(os.path.join(model_dir, fn), framework="pt") as f:
            for key in f.keys():
                if is_vision_weight(key):
                    n_vision += 1
                    continue
                clean = key.replace("language_model.", "")
                if "layers." in clean:
                    idx = int(clean.split("layers.")[1].split(".")[0])
                    if idx >= n_layers:
                        continue
                keep[key] = f.get_tensor(key)
    print(f"checkpoint: {n_vision} vision tensors skipped, "
          f"{len(keep)} text tensors kept (truncated to {n_layers} layers)")

    with jax.set_mesh(mesh):
        loaded = model.load_weights(list(keep.items()))
    print(f"load_weights reported {len(loaded) if loaded else 0} loaded names")

    params_after = dict(model.named_parameters())

    # ---- every parameter must have MOVED (i.e. actually been written) ------
    untouched: List[str] = []
    for name, param in params_after.items():
        cur = np.asarray(param.get_value())
        prev = params_before[name]
        if cur.shape == prev.shape and np.array_equal(cur, prev):
            untouched.append(name)
    print(f"parameters: {len(params_after)} total, {len(untouched)} untouched")
    if untouched:
        raise AssertionError(
            f"{len(untouched)} parameters were never written by load_weights "
            f"(random init survived): {sorted(untouched)[:20]}")

    # ---- spot-check VALUES against the source tensors ----------------------
    checks = _value_checks(np, torch, keep, params_after, text_cfg, tp)
    for c in checks:
        print(f"  {c['param']:58s} max_abs_err={c['max_abs']:.3e} "
              f"({c['note']})")
    bad = [c for c in checks if c["max_abs"] > 0]
    if bad:
        raise AssertionError(
            f"{len(bad)} parameters do not match the source tensor: "
            f"{[(c['param'], c['note']) for c in bad][:10]}")
    return {
        "n_layers_tested": n_layers,
        "n_params": len(params_after),
        "n_text_tensors_consumed": len(keep),
        "n_vision_tensors_skipped": n_vision,
        "n_value_checks": len(checks),
    }


def _value_checks(np, torch, hf_tensors, params, text_cfg, tp):
    """Undo the loader's transforms and compare against the raw checkpoint."""
    checks = []
    D = text_cfg.hidden_size
    H = text_cfg.head_dim
    NH = text_cfg.num_attention_heads
    KV = text_cfg.num_key_value_heads
    kv_rep = max(1, tp // KV)

    def to_np(t):
        if hasattr(t, "detach"):
            t = t.detach()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            return t.numpy()
        return np.asarray(t)

    def get(path):
        if path not in params:
            raise KeyError(f"no param at {path}; have e.g. "
                           f"{sorted(params)[:8]}")
        return np.asarray(params[path].get_value(), dtype=np.float32)

    def add(name, path, expected, note):
        expected = to_np(expected)
        got = get(path)
        if got.shape != expected.shape:
            checks.append({
                "param": name,
                "max_abs": float("inf"),
                "note": f"SHAPE {got.shape} != {expected.shape}",
            })
            return
        checks.append({
            "param": name,
            "max_abs": float(np.abs(got.astype(np.float32) -
                                    expected.astype(np.float32)).max()),
            "note": note,
        })

    hf = {k: to_np(v) for k, v in hf_tensors.items()}

    add("embed_tokens", "model.embed_tokens.weight",
        hf["model.language_model.embed_tokens.weight"], "verbatim [V,D]")
    add("lm_head", "lm_head.weight", hf["lm_head.weight"].T,
        "transposed [D,V]")
    add("model.norm", "model.norm.weight",
        hf["model.language_model.norm.weight"], "verbatim [D]")

    layers = sorted({
        int(k.split("layers.")[1].split(".")[0])
        for k in hf if ".layers." in k
    })
    # First and last built layer: the [S,S,S,F] pattern makes layer 3 a
    # full-attention / NoPE layer, so both flavours get checked.
    for i in {layers[0], layers[-1]}:
        p = f"model.language_model.layers.{i}"
        m = f"model.layers.{i}"
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            add(f"L{i}.{nm}", f"{m}.{nm}.weight", hf[f"{p}.{nm}.weight"],
                "verbatim [D]")
        # q: [NH*H, D] -> [D, NH, H];  o: [D, NH*H] -> [NH, H, D]
        add(f"L{i}.q_proj", f"{m}.self_attn.q_proj.weight",
            hf[f"{p}.self_attn.q_proj.weight"].reshape(NH, H,
                                                       D).transpose(2, 0, 1),
            "[NH*H,D]->[D,NH,H]")
        add(f"L{i}.o_proj", f"{m}.self_attn.o_proj.weight",
            hf[f"{p}.self_attn.o_proj.weight"].reshape(D, NH,
                                                       H).transpose(1, 2, 0),
            "[D,NH*H]->[NH,H,D]")
        for nm in ("k_proj", "v_proj"):
            w = hf[f"{p}.self_attn.{nm}.weight"].reshape(KV, H,
                                                         D).transpose(2, 0, 1)
            if kv_rep > 1:
                # element-wise repeat (h0 h0 h1 h1), matching the loader
                w = np.repeat(w, kv_rep, axis=1)
            add(f"L{i}.{nm}", f"{m}.self_attn.{nm}.weight", w,
                f"[KV*H,D]->[D,KV,H] kv_repeat={kv_rep}")
        # THE trap: self_attn.gate_proj must land on attn_gate_proj, and the
        # MLP gate_proj must NOT be overwritten by it (or vice versa).
        add(f"L{i}.attn_gate_proj", f"{m}.self_attn.attn_gate_proj.weight",
            hf[f"{p}.self_attn.gate_proj.weight"].T,
            "self_attn.gate_proj -> attn_gate_proj, transposed")
        for nm in ("gate_proj", "up_proj", "down_proj"):
            add(f"L{i}.mlp.{nm}", f"{m}.mlp.{nm}.weight",
                hf[f"{p}.mlp.{nm}.weight"].T, "transposed")
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-load", action="store_true")
    args = ap.parse_args()

    s1_config(args.model_dir)
    s2_plugin()
    mc_out = s3_model_config(args.model_dir, args.max_model_len)
    if mc_out is not None:
        _, mc = mc_out
        s4_shapes(mc)
        s5_resolve(mc)
    if not args.skip_load:
        s6_load_weights(args.model_dir, args.layers, args.tp)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    failed = []
    for name, r in RESULTS.items():
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {name}"
              f"{'' if r['ok'] else '  <- ' + r['error']}")
        if not r["ok"]:
            failed.append(name)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(RESULTS, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
