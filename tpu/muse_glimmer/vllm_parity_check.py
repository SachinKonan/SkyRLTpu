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
"""CPU parity gate for the **torch/vLLM** Muse-Glimmer port.

Runs entirely on CPU, no TPU, no chips.  Builds
``tpu_inference.models.vllm.muse_glimmer.MuseGlimmerForCausalLM`` out of vLLM's
real layer classes with ``tensor_parallel_size=1``, substitutes an eager
attention implementation for the paged ``Attention`` layer (which needs a TPU
kernel and a KV cache), loads a checkpoint, and compares against the HF
reference.

Two modes:

``--mode tiny``
    4 layers keeping the real ``[S, S, S, F]`` pattern and the real NoPE marker
    on the full layer.  Reference produced by ``vllm_parity_ref.py`` in the
    ``transformers @ main`` venv.  Also runs the structural probes: sliding
    layers must not see past the window, full layers must see everything, and
    full layers must be position-invariant under a *non-constant* position
    change (SPEC trap 12 -- a constant shift proves nothing, RoPE is relative).

``--mode real``
    The 30B checkpoint against the recorded HF dump
    (``runs/muse_glimmer/hf_ref.npz``): teacher-forced argmax agreement over
    every position, top-8 sets, and the sampled full logit rows.  This is the
    gate the JAX port passed at 3949/3949.

Never run on the login node -- use ``vllm_parity_check.sbatch``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TPU_INFERENCE = REPO_ROOT / "third_party" / "tpu-inference"


# --------------------------------------------------------------------------
# vLLM bring-up on CPU
# --------------------------------------------------------------------------


#: OOT layer classes tpu-inference registers on import. They are TPU-only:
#: every one of them reaches for ``vllm_config.quant_config.mesh`` and shards
#: the weight onto a JAX device mesh, which does not exist on CPU. Popping them
#: makes the gate build the model out of vLLM's stock classes, which is what
#: this gate is for -- the model's *arithmetic*. Sharding, KV replication at
#: TP=4 and the Pallas kernel are only testable on a slice, and are recorded as
#: such.
_TPU_OOT_LAYER_NAMES = (
    "RowParallelLinear",
    "ColumnParallelLinear",
    "ReplicatedLinear",
    "QKVParallelLinear",
    "VocabParallelEmbedding",
    "ParallelLMHead",
)


def drop_tpu_oot_layers() -> list[str]:
    from vllm.model_executor.custom_op import op_registry_oot
    dropped = []
    for name in _TPU_OOT_LAYER_NAMES:
        if op_registry_oot.pop(name, None) is not None:
            dropped.append(name)
    return dropped


def force_pallas_attention_backend():
    """Make ``Attention.__init__`` pick tpu-inference's Pallas backend on CPU.

    vLLM selects the attention backend from ``current_platform``, and on a
    plain CPU box there is none ("Invalid attention backend for None").  Force
    the backend this model will actually run on, so the ``AttentionImpl`` that
    gets constructed -- and therefore the head counts, the softmax scale and
    the sliding window it is handed -- is the real
    ``PallasAttentionBackendImpl``.  Its ``forward`` is never called here; the
    harness swaps the layer for :class:`EagerAttentionShim` and reads the
    window straight off ``impl``.

    Returns the undo callable.
    """
    from vllm.model_executor.layers.attention import attention as attn_mod

    from tpu_inference.layers.vllm.backends.flash_attn import \
        PallasAttentionBackend

    original = attn_mod.get_attn_backend
    attn_mod.get_attn_backend = lambda *a, **k: PallasAttentionBackend

    def undo():
        attn_mod.get_attn_backend = original

    return undo


def init_cpu_distributed():
    """vLLM's parallel-linear layers need a (world size 1) process group.

    ``initialize_model_parallel`` reads ``get_current_vllm_config()``, so this
    must be called *inside* a ``set_current_vllm_config`` context.
    """
    import torch.distributed as dist
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:{}".format(
                os.environ.get("MG_PARITY_PORT", "29591")),
            local_rank=0,
            backend="gloo",
        )
    initialize_model_parallel(tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1)


class EagerAttentionShim(__import__("torch").nn.Module):
    """Stands in for ``vllm.model_executor.layers.attention.Attention``.

    The real layer routes into ``PallasAttentionBackendImpl`` -> a ragged paged
    attention Pallas kernel and a KV cache handed in through the wrapper
    context.  None of that exists on CPU, and none of it is what this gate is
    testing: the point is the *model's* arithmetic around attention.  So the
    shim implements the same masked softmax the kernel does -- causal, plus the
    sliding window when the layer has one -- over a single unbatched prefill.

    Installed by replacing each ``self_attn.attn`` module after construction,
    which also proves the window really did reach the kernel-facing object
    (``impl.sliding_window``) even though the model clears
    ``attn.sliding_window`` to keep the KV-cache spec uniform.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 scale: float, sliding_window: int | None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window

    def forward(self, q, k, v):
        import torch
        # [T, N*H] -> [N, T, H]
        t = q.shape[0]
        qh = q.view(t, self.num_heads, self.head_dim).transpose(0, 1)
        kh = k.view(t, self.num_kv_heads, self.head_dim).transpose(0, 1)
        vh = v.view(t, self.num_kv_heads, self.head_dim).transpose(0, 1)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            # HF `repeat_kv` semantics: each kv head repeated CONTIGUOUSLY.
            kh = kh.repeat_interleave(n_rep, dim=0)
            vh = vh.repeat_interleave(n_rep, dim=0)

        idx = torch.arange(t, device=q.device)
        allowed = idx[:, None] >= idx[None, :]
        if self.sliding_window is not None:
            allowed = allowed & ((idx[:, None] - idx[None, :])
                                 < self.sliding_window)

        logits = torch.einsum("nth,nsh->nts", qh, kh) * self.scale
        logits = torch.where(allowed, logits,
                             torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
        out = torch.einsum("nts,nsh->nth", probs, vh)
        return out.transpose(0, 1).reshape(t, self.num_heads * self.head_dim)


def install_eager_attention(model) -> list[EagerAttentionShim]:
    """Replace every paged ``Attention`` with :class:`EagerAttentionShim`."""
    shims = []
    for layer in model.model.layers:
        attn = layer.self_attn
        real = attn.attn
        shim = EagerAttentionShim(
            num_heads=real.impl.num_heads,
            num_kv_heads=real.impl.num_kv_heads,
            head_dim=real.impl.head_size,
            scale=real.impl.scale,
            # Read the window off the IMPL, not off the layer: the model
            # deliberately clears `attn.sliding_window` so the KV-cache spec
            # stays uniform, and this is the assertion that the window still
            # reached the kernel-facing object.
            sliding_window=real.impl.sliding_window,
        )
        attn.attn = shim
        shims.append(shim)
    return shims


def make_vllm_config(model_dir: str,
                     dtype: str = "float32",
                     max_model_len: int = 4096,
                     lora_config=None):
    """A real ``VllmConfig`` for CPU model construction.

    ``ModelConfig`` is built through its own constructor from a real directory
    so that everything ``__post_init__`` derives (``model_arch_config``,
    ``hf_text_config``, ...) is what vLLM itself would derive.  The
    architecture lookup goes through tpu-inference's out-of-tree registration,
    so this also proves the registration resolves to the torch class.
    """
    from vllm.config import (CacheConfig, CompilationConfig, DeviceConfig,
                             LoadConfig, ModelConfig, ParallelConfig,
                             SchedulerConfig, VllmConfig)

    from tpu_inference.models.common.oot_registration import \
        register_out_of_tree_architectures
    register_out_of_tree_architectures()

    model_config = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
        enforce_eager=True,
    )

    # Construct the real thing rather than ``__new__``-ing it: half of
    # ``VllmConfig``'s fields (attention_config, quant_config, ...) are filled
    # in by ``__post_init__``, and hand-populating them chases a new
    # AttributeError every vLLM release.
    #
    # ``custom_ops=["none"]`` routes every CustomOp (here: GemmaRMSNorm) to
    # ``forward_native``, which is also what the TPU path does -- there is no
    # ``forward_tpu``.
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(max_model_len=max_model_len,
                                         is_encoder_decoder=False),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        lora_config=lora_config,
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )
    return vllm_config


def build_model(model_dir: str,
                dtype: str = "float32",
                max_model_len: int = 4096,
                lora_config=None):
    """Instantiate the torch model on CPU under a real ``VllmConfig``."""
    import torch
    from vllm.config import set_current_vllm_config

    from tpu_inference.models.vllm.muse_glimmer import MuseGlimmerForCausalLM

    vllm_config = make_vllm_config(model_dir, dtype, max_model_len,
                                   lora_config)
    dropped = drop_tpu_oot_layers()
    if dropped:
        print(f"using vLLM's stock layer classes on CPU (dropped TPU OOT "
              f"overrides: {', '.join(dropped)})")
    undo_backend = force_pallas_attention_backend()
    try:
        with set_current_vllm_config(vllm_config):
            init_cpu_distributed()
            with torch.device("cpu"):
                torch.set_default_dtype(getattr(torch, dtype))
                try:
                    model = MuseGlimmerForCausalLM(vllm_config=vllm_config,
                                                   prefix="")
                finally:
                    torch.set_default_dtype(torch.float32)
    finally:
        undo_backend()
    return model, vllm_config


# --------------------------------------------------------------------------
# Diff reporting -- same metrics as parity_check.py so the two gates compare
# --------------------------------------------------------------------------


def diff_report(name: str, ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape:
        return {"name": name, "shape_mismatch": (ref.shape, got.shape)}
    d = np.abs(ref - got)
    peak = float(np.max(np.abs(ref))) or 1.0
    big = np.abs(ref) > 0.01 * peak
    rel_big = float(np.max(d[big] / np.abs(ref[big]))) if big.any() else 0.0
    out = {
        "name": name,
        "shape": tuple(ref.shape),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rel_to_scale": float(d.max() / peak),
        "rel_above_1pct_peak": rel_big,
    }
    print(f"  {name:34s} max_abs {out['max_abs']:.3e}  "
          f"rel_to_scale {out['rel_to_scale']:.3e}  "
          f"rel(>1% peak) {out['rel_above_1pct_peak']:.3e}")
    return out


# --------------------------------------------------------------------------
# Tiny mode
# --------------------------------------------------------------------------


def load_tiny_config(meta_path: Path):
    """Rebuild the HF text config from the reference dump's metadata.

    The reference venv holds transformers@main (which has `muse_glimmer`); this
    venv holds whatever transformers vllm 0.23.0 pinned.  If it also has
    `muse_glimmer` we use the real config class, otherwise a duck-typed stand-in
    -- the model only ever reads plain attributes off it.
    """
    meta = json.loads(meta_path.read_text())
    cfg_dict = meta["config"]
    try:
        from transformers import MuseGlimmerTextConfig
        cfg = MuseGlimmerTextConfig(**{
            k: v
            for k, v in cfg_dict.items()
            if k not in ("architectures", "model_type", "transformers_version")
        })
        # Pin the exact per-layer values the reference used; constructors have
        # been known to re-derive these.
        cfg.layer_types = list(cfg_dict["layer_types"])
        cfg.layer_rope_theta = list(cfg_dict["layer_rope_theta"])
        cfg.rope_parameters = dict(cfg_dict["rope_parameters"])
        source = "transformers.MuseGlimmerTextConfig"
    except Exception as exc:  # pragma: no cover - fallback path
        cfg = argparse.Namespace(**cfg_dict)
        source = f"duck-typed namespace ({type(exc).__name__})"
    return cfg, meta, source


def run_tiny(args, failures: list[str]) -> list[dict]:
    import torch
    from safetensors.torch import load_file

    ref_dir = Path(args.ref_dir)
    variant = args.variant
    cfg, meta, cfg_source = load_tiny_config(ref_dir / f"{variant}_meta.json")
    print(f"config from: {cfg_source}")
    print(f"layer_types      = {list(cfg.layer_types)}")
    print(f"layer_rope_theta = {list(cfg.layer_rope_theta)}")
    print(f"sliding_window   = {cfg.sliding_window}  "
          f"rms_norm_eps={cfg.rms_norm_eps}  post_norm_eps={cfg.post_norm_eps}")
    print()

    model, vllm_config = build_model(
        str(ref_dir / f"{variant}_model"),
        dtype="float32",
        max_model_len=int(cfg.max_position_embeddings))
    model.eval()
    arch = vllm_config.model_config.architectures
    print(f"vLLM resolved architectures={arch} -> {type(model).__name__}")
    if type(model).__name__ != "MuseGlimmerForCausalLM":
        failures.append(
            f"tiny: registry resolved to {type(model).__name__}, not "
            "MuseGlimmerForCausalLM")

    ckpt = load_file(str(ref_dir / f"{variant}_weights.safetensors"))
    loaded = model.load_weights(list(ckpt.items()))
    expected = {n for n, _ in model.named_parameters()}
    missing = expected - set(loaded)
    print(f"loaded {len(ckpt)} checkpoint tensors -> {len(loaded)} params; "
          f"{len(missing)} unloaded")
    if missing:
        print(f"  UNLOADED: {sorted(missing)[:10]}")
        failures.append(f"tiny: {len(missing)} parameters never loaded")

    shims = install_eager_attention(model)
    windows = [s.sliding_window for s in shims]
    print(f"per-layer kernel windows = {windows}")
    expect_windows = [
        cfg.sliding_window if t == "sliding_attention" else None
        for t in cfg.layer_types
    ]
    if windows != expect_windows:
        failures.append(
            f"tiny: kernel windows {windows} != expected {expect_windows}")
    ropes = [l.self_attn.rotary_emb is not None for l in model.model.layers]
    expect_ropes = [bool(t) for t in cfg.layer_rope_theta]
    print(f"per-layer rope enabled   = {ropes}")
    if ropes != expect_ropes:
        failures.append(f"tiny: rope flags {ropes} != expected {expect_ropes}")

    ref = np.load(ref_dir / f"{variant}_ref.npz")
    results = []
    print("\n-- tensor parity (float32) --")
    for L in meta["seq_lens"]:
        ids = ref[f"T{L}/ids"]
        ref_hidden = ref[f"T{L}/hidden"]
        ref_logits = ref[f"T{L}/logits"]
        got_hidden = np.zeros_like(ref_hidden)
        got_logits = np.zeros_like(ref_logits)
        for b in range(ids.shape[0]):
            t_ids = torch.from_numpy(ids[b]).long()
            positions = torch.arange(ids.shape[1], dtype=torch.long)
            with torch.no_grad():
                h = model(input_ids=t_ids, positions=positions)
                lg = model.compute_logits(h)
            got_hidden[b] = h.float().numpy()
            got_logits[b] = lg.float().numpy()
        results.append(diff_report(f"hidden T={L}", ref_hidden, got_hidden))
        results.append(diff_report(f"logits T={L}", ref_logits, got_logits))

    # ---- structural probes -------------------------------------------------
    print("\n-- structural probes --")
    results += tiny_probes(model, cfg, failures)
    return results


def tiny_probes(model, cfg, failures: list[str]) -> list[dict]:
    """Attention span + NoPE probes, run directly on the built torch model."""
    import torch
    out = []
    window = int(cfg.sliding_window)
    seq = max(4 * window, 32)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (seq, ), generator=g)
    positions = torch.arange(seq, dtype=torch.long)

    # (a) span: perturb token 0 and see which layers' outputs move at the last
    #     position.  Sliding layers must NOT see it (seq-1 - 0 >= window);
    #     full layers must.
    for layer_idx, layer in enumerate(model.model.layers):
        is_sliding = cfg.layer_types[layer_idx] == "sliding_attention"
        h = torch.randn(seq, cfg.hidden_size, generator=g)
        h2 = h.clone()
        h2[0] += 10.0
        with torch.no_grad():
            a = layer(positions, h)
            b = layer(positions, h2)
        moved = float((a[-1] - b[-1]).abs().max())
        ok = (moved < 1e-6) if is_sliding else (moved > 1e-4)
        tag = "sliding" if is_sliding else "full"
        print(f"  layer {layer_idx} ({tag:8s}) last-pos delta from token 0: "
              f"{moved:.3e}  {'OK' if ok else 'FAIL'}")
        out.append({
            "name": f"span_layer{layer_idx}",
            "moved": moved,
            "ok": ok
        })
        if not ok:
            failures.append(
                f"tiny: layer {layer_idx} ({tag}) attention span wrong "
                f"(delta {moved:.3e}, window {window}, seq {seq})")

    # (b) NoPE: change the position DIFFERENCES (not a constant shift -- RoPE is
    #     relative, so a shift proves nothing; SPEC trap 12).  Full layers must
    #     be unchanged; sliding layers must move.
    stride = torch.arange(seq, dtype=torch.long) * 3
    for layer_idx, layer in enumerate(model.model.layers):
        is_sliding = cfg.layer_types[layer_idx] == "sliding_attention"
        h = torch.randn(seq, cfg.hidden_size, generator=g)
        with torch.no_grad():
            a = layer(positions, h)
            b = layer(stride, h)
        moved = float((a - b).abs().max())
        ok = (moved > 1e-4) if is_sliding else (moved < 1e-6)
        tag = "sliding/rope" if is_sliding else "full/NoPE"
        print(f"  layer {layer_idx} ({tag:12s}) delta under stride-3 "
              f"positions: {moved:.3e}  {'OK' if ok else 'FAIL'}")
        out.append({"name": f"nope_layer{layer_idx}", "moved": moved,
                    "ok": ok})
        if not ok:
            failures.append(
                f"tiny: layer {layer_idx} ({tag}) failed the NoPE probe "
                f"(delta {moved:.3e})")
    return out


# --------------------------------------------------------------------------
# Real-weight mode
# --------------------------------------------------------------------------


def run_real(args, failures: list[str]) -> list[dict]:
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    model_dir = Path(args.model_dir)
    cfg_full = AutoConfig.from_pretrained(model_dir)
    text_cfg = getattr(cfg_full, "text_config", cfg_full)
    print(f"num_hidden_layers={text_cfg.num_hidden_layers}  "
          f"hidden={text_cfg.hidden_size}  kv_heads={text_cfg.num_key_value_heads}")

    model, _ = build_model(str(model_dir), dtype="float32",
                           max_model_len=args.max_model_len)
    model.eval()

    files = sorted(model_dir.glob("*.safetensors"))
    print(f"loading {len(files)} safetensors shards (float32, CPU)")
    n_loaded = 0

    def _iter():
        nonlocal n_loaded
        for f in files:
            with safe_open(str(f), framework="pt") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    if tensor.dtype in (torch.bfloat16, torch.float16):
                        tensor = tensor.float()
                    n_loaded += 1
                    yield key, tensor

    loaded = model.load_weights(_iter())
    expected = {n for n, _ in model.named_parameters()}
    missing = expected - set(loaded)
    print(f"read {n_loaded} checkpoint tensors -> {len(loaded)} params; "
          f"{len(missing)} unloaded")
    if missing:
        print(f"  UNLOADED: {sorted(missing)[:10]}")
        failures.append(f"real: {len(missing)} parameters never loaded")

    # lm_head must NOT be the embedding matrix (SPEC trap 8).
    lm = model.lm_head.weight.detach()
    emb = model.model.embed_tokens.weight.detach()
    delta = float((lm - emb).abs().max())
    print(f"max|lm_head - embed| = {delta:.4f}  "
          f"(expected ~3.09, i.e. UNTIED)")
    if delta < 1e-6:
        failures.append("real: lm_head is tied to embed_tokens; it must not be")

    install_eager_attention(model)

    ref = np.load(args.hf_ref)
    results = []
    prompts = [p for p in args.prompts.split(",") if p.strip()]
    for name in prompts:
        key = f"{name}/ids"
        if key not in ref:
            print(f"  {name}: absent from the reference dump, skipping")
            continue
        ids = np.asarray(ref[f"{name}/ids"]).reshape(-1)
        if args.max_prompt_tokens and len(ids) > args.max_prompt_tokens:
            print(f"  {name}: {len(ids)} tokens > --max-prompt-tokens "
                  f"{args.max_prompt_tokens}, skipping")
            continue
        t_ids = torch.from_numpy(ids).long()
        positions = torch.arange(len(ids), dtype=torch.long)
        with torch.no_grad():
            hidden = model(input_ids=t_ids, positions=positions)
            logits = model.compute_logits(hidden)
        got = logits.float().numpy()

        ref_argmax = np.asarray(ref[f"{name}/argmax"]).reshape(-1)
        got_argmax = got.argmax(-1)
        agree = int((ref_argmax == got_argmax).sum())
        total = int(ref_argmax.size)
        print(f"  {name:16s} tokens {len(ids):5d}  "
              f"argmax {agree}/{total}"
              f"{'  ALL AGREE' if agree == total else '  *** MISMATCH ***'}")
        entry = {
            "name": name,
            "tokens": int(len(ids)),
            "argmax_agree": agree,
            "argmax_total": total,
        }
        if agree != total:
            failures.append(
                f"real: {name} teacher-forced argmax {agree}/{total}")

        rows_idx = np.asarray(ref[f"{name}/full_rows_idx"]).reshape(-1)
        ref_rows = np.asarray(ref[f"{name}/full_rows"])
        got_rows = got[rows_idx]
        entry["rows"] = diff_report(f"{name} full logit rows", ref_rows,
                                    got_rows)

        ref_top8 = np.asarray(ref[f"{name}/top8_ids"])
        got_top8 = np.argsort(-got, axis=-1)[:, :8]
        set_match = int(
            sum(
                set(a.tolist()) == set(b.tolist())
                for a, b in zip(ref_top8, got_top8)))
        entry["top8_set_match"] = set_match
        print(f"  {name:16s} top-8 set match {set_match}/{len(ref_top8)}")
        if set_match != len(ref_top8):
            failures.append(
                f"real: {name} top-8 set match {set_match}/{len(ref_top8)}")

        if f"{name}/hs_final" in ref:
            entry["hidden"] = diff_report(f"{name} final hidden",
                                          np.asarray(ref[f"{name}/hs_final"]),
                                          hidden.float().numpy())
        results.append(entry)
    return results


# --------------------------------------------------------------------------
# Registry mode -- "additive" is a claim, so check it
# --------------------------------------------------------------------------


def run_registry(args, failures: list[str]) -> list[dict]:
    """Both dispatch branches must resolve to their own implementation.

    The worry with registering a real torch class under the checkpoint's
    architecture name is that it displaces the JAX model.  It cannot:
    `model_loader.get_model` branches on MODEL_IMPL_TYPE *before* any registry
    lookup, and the two branches read different registries.  Assert exactly
    that, on CPU, rather than only observing it on a slice.
    """
    out = []
    from vllm.model_executor.models.registry import ModelRegistry

    from tpu_inference.models.common.model_loader import (
        _MODEL_REGISTRY, _VLLM_PREFERRED_ARCHITECTURES,
        _get_model_architecture, resolve_model_architecture)
    from tpu_inference.models.common.oot_registration import \
        register_out_of_tree_architectures

    register_out_of_tree_architectures()
    arch = "MuseGlimmerForConditionalGeneration"

    # 1. vLLM's registry -> the torch model (the `vllm` branch).
    torch_cls = ModelRegistry._try_load_model_cls(arch)
    if torch_cls is None:
        failures.append(f"registry: vLLM cannot load a class for {arch}")
        return out
    print(f"vLLM ModelRegistry[{arch}] -> {torch_cls.__module__}."
          f"{torch_cls.__name__}")
    out.append({"vllm_registry": f"{torch_cls.__module__}.{torch_cls.__name__}"})
    if torch_cls.__name__ != "MuseGlimmerForCausalLM":
        failures.append(f"registry: vLLM resolves {arch} to {torch_cls!r}")

    # 2. tpu-inference's registry -> the JAX model (the `flax_nnx` branch).
    cfg = argparse.Namespace(architectures=[arch], model_type="muse_glimmer")
    jax_cls = _get_model_architecture(cfg)
    print(f"tpu-inference _MODEL_REGISTRY[{arch}] -> {jax_cls.__module__}."
          f"{jax_cls.__name__}")
    out.append({"jax_registry": f"{jax_cls.__module__}.{jax_cls.__name__}"})
    if not jax_cls.__module__.startswith("tpu_inference.models.jax"):
        failures.append(
            f"registry: flax_nnx branch now resolves {arch} to "
            f"{jax_cls.__module__}.{jax_cls.__name__} -- the JAX path is "
            "REGRESSED")
    if _MODEL_REGISTRY.get(arch) is not jax_cls:
        failures.append("registry: _MODEL_REGISTRY entry is not the JAX class")

    # 3. MODEL_IMPL_TYPE=auto must still choose flax_nnx.
    in_preferred = arch in _VLLM_PREFERRED_ARCHITECTURES
    print(f"{arch} in _VLLM_PREFERRED_ARCHITECTURES: {in_preferred} "
          f"(must be False, or `auto` flips to the torch model)")
    out.append({"vllm_preferred": in_preferred})
    if in_preferred:
        failures.append(
            "registry: the architecture was added to "
            "_VLLM_PREFERRED_ARCHITECTURES, which changes the DEFAULT away "
            "from the proven JAX path")

    fake_vllm_config = argparse.Namespace(
        load_config=argparse.Namespace(load_format="auto"),
        model_config=argparse.Namespace(hf_config=cfg),
        speculative_config=None,
    )
    impl = resolve_model_architecture(fake_vllm_config, is_draft_model=False)
    print(f"resolve_model_architecture('auto') -> {impl!r} (must be 'flax_nnx')")
    out.append({"auto_resolves_to": impl})
    if impl != "flax_nnx":
        failures.append(
            f"registry: MODEL_IMPL_TYPE=auto now resolves to {impl!r}")
    return out


# --------------------------------------------------------------------------
# LoRA mode -- the assertion that actually matters
# --------------------------------------------------------------------------


def run_lora(args, failures: list[str]) -> list[dict]:
    """Count the LoRA-wrapped layers this model produces.

    This is the whole point of the torch port, so it is asserted directly
    rather than inferred from "the server started".  A name mismatch injects
    **zero** adapters and trains nothing while still producing a plausible
    loss curve -- the failure mode the JAX/MaxText side was bitten by.

    Runs on CPU: ``create_lora_manager`` -> ``_create_lora_modules`` is pure
    torch module surgery (``from_layer``), the same call
    ``VllmModelWrapper.load_weights`` makes via
    ``tpu_inference.lora.lora_manager``.  No TPU, no kernels.
    """
    import torch
    from vllm.config import LoRAConfig, set_current_vllm_config
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.lora.model_manager import LoRAModelManager, create_lora_manager
    from vllm.model_executor.models.interfaces import supports_lora

    ref_dir = Path(args.ref_dir)
    cfg, _meta, _src = load_tiny_config(ref_dir / f"{args.variant}_meta.json")

    lora_config = LoRAConfig(
        max_lora_rank=args.lora_rank,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype=torch.float32,
    )
    if args.lora_targets:
        lora_config.target_modules = [
            t for t in args.lora_targets.split(",") if t.strip()
        ]

    model, vllm_config = build_model(str(ref_dir / f"{args.variant}_model"),
                                     dtype="float32",
                                     max_model_len=int(
                                         cfg.max_position_embeddings),
                                     lora_config=lora_config)

    declares = bool(supports_lora(model))
    print(f"supports_lora(model) = {declares}")
    if not declares:
        failures.append("lora: the model does not satisfy vLLM's SupportsLoRA")

    from vllm.lora.utils import get_supported_lora_modules
    supported = sorted(set(get_supported_lora_modules(model)))
    print(f"vLLM-discovered LoRA-capable module suffixes: {supported}")

    # The punica wrapper comes from `current_platform`, which on a bare CPU box
    # raises NotImplementedError. Point it at the same wrapper the TPU platform
    # uses; if that one insists on a device, fall back to vLLM's CPU wrapper.
    # Either way it does not affect *which modules get wrapped*, which is the
    # number this mode exists to report.
    from vllm.platforms import current_platform
    wrappers = [
        "tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU",
        "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU",
    ]
    manager = None
    last_exc: Exception | None = None
    for qual in wrappers:
        current_platform.get_punica_wrapper = (lambda q=qual: q)
        try:
            with set_current_vllm_config(vllm_config):
                manager = create_lora_manager(
                    model,
                    max_num_seqs=8,
                    max_num_batched_tokens=1024,
                    vocab_size=int(cfg.vocab_size),
                    lora_config=lora_config,
                    vllm_config=vllm_config,
                    device=torch.device("cpu"),
                )
            print(f"punica wrapper: {qual}")
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"punica wrapper {qual} unusable on CPU: "
                  f"{type(exc).__name__}: {exc}")
    if manager is None:
        raise RuntimeError(
            f"no usable punica wrapper on CPU: {last_exc!r}")
    assert isinstance(manager, LoRAModelManager)

    wrapped = {
        name: type(mod).__name__
        for name, mod in manager.model.named_modules()
        if isinstance(mod, BaseLayerWithLoRA)
    }
    n = len(wrapped)
    print(f"\nLoRA-wrapped modules: {n}")
    by_kind: dict[str, int] = {}
    by_leaf: dict[str, int] = {}
    for name, kind in wrapped.items():
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_leaf[name.split(".")[-1]] = by_leaf.get(name.split(".")[-1], 0) + 1
    for kind, c in sorted(by_kind.items()):
        print(f"  {c:4d}  {kind}")
    print("  by leaf name:")
    for leaf, c in sorted(by_leaf.items()):
        print(f"    {c:4d}  {leaf}")

    if n == 0:
        failures.append("lora: ZERO adapter modules -- the whole point of this "
                        "port is that this number is not zero")

    # The attention gate must be adapted under its own name, and must NOT be
    # confused with the MLP's gate_proj.
    gate_leaf = by_leaf.get("attn_gate_proj", 0)
    n_layers = int(cfg.num_hidden_layers)
    print(f"\nattention gates wrapped: {gate_leaf} (expected {n_layers})")
    if gate_leaf != n_layers:
        failures.append(
            f"lora: attn_gate_proj wrapped {gate_leaf}x, expected {n_layers}")
    if "gate_proj" in by_leaf:
        failures.append(
            "lora: a bare `gate_proj` module exists; it should be packed into "
            "`gate_up_proj`, and the attention gate should be `attn_gate_proj`")

    # Targeting `gate_proj` (the MLP gate) must not latch onto the attention
    # gate: vLLM's suffix regex requires a literal dot before the target, and
    # `attn_gate_proj`'s preceding character is `_`.
    from vllm.lora.utils import is_supported_lora_module
    collides = is_supported_lora_module(
        "model.layers.0.self_attn.attn_gate_proj", ["gate_proj"])
    print(f"`gate_proj` target matches `self_attn.attn_gate_proj`: {collides} "
          f"(must be False)")
    if collides:
        failures.append("lora: `gate_proj` collides with `attn_gate_proj`")

    return [{
        "name": "lora",
        "n_wrapped": n,
        "by_kind": by_kind,
        "by_leaf": by_leaf,
        "supports_lora": declares,
        "supported_suffixes": supported,
        "gate_proj_collides": bool(collides),
    }]


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=("tiny", "real", "lora", "registry"),
                    default="tiny")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-targets", default="")
    ap.add_argument("--variant", default="tiny")
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--model-dir",
                    default="/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b")
    ap.add_argument("--hf-ref",
                    default=str(REPO_ROOT / "runs/muse_glimmer/hf_ref.npz"))
    ap.add_argument("--prompts",
                    default="p1_tiny,p2_odd,p3_mid,p4_long_sliding,p5_code")
    ap.add_argument("--max-prompt-tokens", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.ref_dir is None:
        args.ref_dir = str(REPO_ROOT / "runs" / "muse_glimmer" / "vllm_parity")

    sys.path.insert(0, str(TPU_INFERENCE))
    # vLLM inspects the model class in a SUBPROCESS, which inherits the
    # environment but not sys.path. In production tpu_inference is pip
    # installed; here it is a source tree, so it has to go on PYTHONPATH or
    # the inspection fails with "architectures failed to be inspected".
    py_path = os.environ.get("PYTHONPATH", "")
    if str(TPU_INFERENCE) not in py_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = (f"{TPU_INFERENCE}{os.pathsep}{py_path}"
                                    if py_path else str(TPU_INFERENCE))
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import transformers
    import vllm
    print("=" * 78)
    print("Muse-Glimmer torch/vLLM parity harness")
    print("=" * 78)
    print(f"vllm {vllm.__version__}  torch {torch.__version__}  "
          f"transformers {transformers.__version__}")
    print(f"mode={args.mode}")
    print()

    failures: list[str] = []
    if args.mode == "tiny":
        results = run_tiny(args, failures)
    elif args.mode == "lora":
        results = run_lora(args, failures)
    elif args.mode == "registry":
        results = run_registry(args, failures)
    else:
        results = run_real(args, failures)

    print()
    print("=" * 78)
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASS")
    print("=" * 78)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "vllm": vllm.__version__,
                    "torch": torch.__version__,
                    "results": results,
                    "failures": failures,
                },
                indent=2,
                default=str))
        print(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
