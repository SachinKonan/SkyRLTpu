"""Reproduce the empty-``v`` crash from the 2026-08-17 TPU run — on CPU.

The crash (job 3710865, ``runs/muse_glimmer/crash-torch.log``): the first real
request (6 tokens, padded to a 16-token bucket) died in
``tpu_inference/layers/vllm/backends/flash_attn.py:259`` with

    TypeError: cannot reshape array of shape (0, 4, 128) (size 0)
               into shape (16, 4, 128) (size 8192)

``q`` and ``k`` arrived full; ``v`` arrived with ZERO width.  A zero-width
split piece is impossible in stock torch (``split`` validates sizes loudly),
but this model executes under torchax, where ``split`` lowers to JAX slicing —
and JAX **clamps** out-of-range slices silently.  So if the runtime qkv output
is narrower than ``q_size + 2*kv_size``, the last piece quietly comes out
empty instead of erroring.

ROOT CAUSE (found by this repro's stage A): tpu-inference's OOT
``VllmQKVParallelLinear`` inflates its WEIGHT BUFFER to ``mesh TP`` KV heads
when TP > num_key_value_heads, but its ``forward`` then collapses the replica
sub-axis back out of the global view via a ``shard_map`` re-spec
(``out_specs`` without the ``replica`` axis halves the global kv width) and
returns the STOCK widths ``q + 2 * total_num_kv_heads * head_dim``.  Three
width conventions therefore exist:

    buffer width  = q + 2 * kv_inflated   (weight rows, quant apply output)
    stock width   = q + 2 * kv_real       (OOT forward output — the contract)
    model width   = q_size + 2 * kv_size  (whatever the model file declares)

The muse model file read the INFLATED attrs off the layer (model width =
buffer width = 5120 for the 30B), while the executed forward returned the
stock width (4608).  torchax's clamped split then silently emptied ``v``.
The LoRA-wrapped path (``_mcp_apply`` calls ``quant_method.apply`` directly)
bypasses the dedup and returns the BUFFER width — inconsistent with the
non-LoRA path, which the seam fix also patches.

This script fakes the production mesh on CPU (6 axes, ATTN_HEAD product 4,
``--xla_force_host_platform_device_count=4``), keeps the OOT registrations,
builds the model, and drives the qkv path stage by stage, printing every
width.  Stages:

    A.  process_weights_after_loading + quant apply + the OOT replication
        forward (the exact non-LoRA production path — where the TPU run died).
    A2. full self_attn.forward with NO LoRA (the crash configuration), with
        vLLM ``Attention.forward`` views live and a shape-recording stub for
        the paged-attention op.
    B.  production LoRA wrap (load_lora_model) + no-adapter forward.
    C.  self_attn.forward again through the LoRA-wrapped qkv.
    D.  full model forward via shard_model_to_tpu + functional_call under
        jax.jit (production step-function shape).

Exits nonzero at the FIRST stage whose executed width diverges from the
model's declared geometry.  ``MG_REPRO_KEEP_GOING=1`` continues through
later stages after a divergence (used to show the model's post-split width
assert firing on pre-fix code).

Run via ``repro_qkv_width.sbatch`` (compute node, MGVENV python,
``PYTHONPATH`` carrying the repo's tpu-inference).
"""
from __future__ import annotations

import os

# Must precede any jax import: 4 fake CPU devices so the mesh (and with it the
# OOT KV replication) is real.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_force_host_platform_device_count=4")

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KEEP_GOING = os.environ.get("MG_REPRO_KEEP_GOING") == "1"

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = "OK  " if ok else "FAIL"
    print(f"[{tag}] {label}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(label)


def main() -> int:
    import torch

    import vllm_parity_check as H
    import vllm_parity_ref as R

    # ------------------------------------------------------------------
    # CPU-only harness shim: `process_weights_after_loading` frees the CPU
    # copy with `layer.weight.untyped_storage().resize_(0)`.  On this box the
    # storages backing the loaded params refuse resizing ("Trying to resize
    # storage that is not resizable") — a CPU artifact, not the bug under
    # test (on TPU hosts the line is fine).  Make resize_ lenient for the
    # repro only; the library is NOT modified.
    # ------------------------------------------------------------------
    _orig_resize = torch.UntypedStorage.resize_

    def _lenient_resize(self, n):
        try:
            return _orig_resize(self, n)
        except RuntimeError as e:
            if "not resizable" in str(e):
                return self
            raise

    torch.UntypedStorage.resize_ = _lenient_resize

    # ------------------------------------------------------------------
    # 0. Tiny GQA config on disk (2 kv heads < mesh TP 4 -> replicas 2,
    #    the 30B's exact situation at TP=4).
    # ------------------------------------------------------------------
    workdir = Path(tempfile.mkdtemp(prefix="mg-reproqkv-"))
    cfg = R.build_text_config("gqa")
    model_dir = workdir / "gqa_model"
    model_dir.mkdir(parents=True)
    cfg_dict = cfg.to_dict()
    cfg_dict["architectures"] = ["MuseGlimmerForConditionalGeneration"]
    (model_dir / "config.json").write_text(
        json.dumps(cfg_dict, indent=2, default=str))
    print(f"tiny gqa config: heads={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} head_dim={cfg.head_dim} "
          f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")

    # ------------------------------------------------------------------
    # 1. Real mesh over 4 fake CPU devices + the real TPU quant config.
    #    PRODUCTION AXIS NAMES (tpu_runner uses MESH_AXIS_NAMES, 6 axes):
    #    the OOT forward's replication branch walks ATTN_HEAD =
    #    ('model','expert','dcp') and builds shard_map specs over
    #    'expert'/'dcp' — a 2-axis fake mesh cannot execute that branch at
    #    all, which is exactly how the previous CPU gates missed the bug.
    # ------------------------------------------------------------------
    import jax
    from jax.sharding import Mesh

    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

    devices = jax.devices()
    check("4 cpu devices", len(devices) == 4, f"got {len(devices)}")
    mesh_shape = tuple(4 if a == "model" else 1 for a in MESH_AXIS_NAMES)
    mesh = Mesh(
        np.asarray(devices).reshape(mesh_shape), MESH_AXIS_NAMES)
    print(f"mesh axes: {dict(zip(MESH_AXIS_NAMES, mesh_shape))}")

    vllm_config = H.make_vllm_config(str(model_dir), dtype="float32")

    from tpu_inference.layers.vllm.quantization import \
        get_tpu_quantization_config
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config, mesh)
    check("quant_config carries mesh",
          getattr(vllm_config.quant_config, "mesh", None) is mesh,
          type(vllm_config.quant_config).__name__)

    # ------------------------------------------------------------------
    # 2. Build the model with the OOT layers KEPT (the one thing every
    #    previous CPU gate dropped).
    # ------------------------------------------------------------------
    from vllm.config import set_current_vllm_config

    from tpu_inference.models.vllm.muse_glimmer import MuseGlimmerForCausalLM

    undo_backend = H.force_pallas_attention_backend()
    try:
        with set_current_vllm_config(vllm_config):
            H.init_cpu_distributed()
            with torch.device("cpu"):
                model = MuseGlimmerForCausalLM(vllm_config=vllm_config,
                                               prefix="")
    finally:
        undo_backend()

    sa = model.model.layers[0].self_attn
    qkv = sa.qkv_proj
    head = sa.head_dim

    # ------------------------------------------------------------------
    # 3. Construction-time widths.  THREE width conventions live here:
    #      buffer_width : weight rows / quant apply output (inflated kv)
    #      stock_width  : the OOT forward's output contract (real kv)
    #      model_width  : what the model file declares (q_size + 2*kv_size)
    #    Correctness requires model_width == stock_width.
    # ------------------------------------------------------------------
    print("\n--- construction ---")
    print(f"qkv type              : {type(qkv).__name__}")
    print(f"qkv.num_heads         : {qkv.num_heads}")
    print(f"qkv.num_kv_heads      : {qkv.num_kv_heads}   (inflated)")
    print(f"qkv.total_num_kv_heads: {qkv.total_num_kv_heads}   (real)")
    print(f"qkv.num_kv_head_replicas: {getattr(qkv, 'num_kv_head_replicas', 'n/a')}")
    print(f"qkv.output_sizes      : {getattr(qkv, 'output_sizes', 'n/a')}")
    print(f"qkv.weight.shape      : {tuple(qkv.weight.shape)}")
    print(f"model q_size / kv_size: {sa.q_size} / {sa.kv_size}")
    print(f"model num_kv_heads    : {sa.num_kv_heads}")
    print(f"Attention num_kv_heads: {sa.attn.num_kv_heads}")
    impl = getattr(sa.attn, "impl", None)
    if impl is not None:
        print(f"impl num_kv_heads     : {impl.num_kv_heads}  head_size {head}")

    buffer_width = sum(qkv.output_sizes)
    stock_width = (qkv.num_heads + 2 * qkv.total_num_kv_heads) * head
    model_width = sa.q_size + 2 * sa.kv_size
    print(f"buffer_width={buffer_width}  stock_width={stock_width}  "
          f"model_width={model_width}")
    if model_width != stock_width:
        print(f"[NOTE] model declares width {model_width} but the OOT forward "
              f"contract is {stock_width} -- expect the executed divergence "
              f"at stage A (the 30B analogue: 5120 declared vs 4608 executed)")

    check("weight rows == sum(layer.output_sizes) (buffer contract)",
          qkv.weight.shape[0] == buffer_width,
          f"weight={qkv.weight.shape[0]} expected={buffer_width}")
    check("OOT replication engaged (kv heads inflated to mesh TP)",
          type(qkv).__name__ == "VllmQKVParallelLinear"
          and getattr(qkv, "num_kv_head_replicas", 1) > 1,
          f"type={type(qkv).__name__} replicas="
          f"{getattr(qkv, 'num_kv_head_replicas', 'n/a')}")

    # ------------------------------------------------------------------
    # 4. Weight-buffer forward, plain torch (buffer layout only: the real
    #    runtime path goes through quant apply + the OOT forward below).
    # ------------------------------------------------------------------
    print("\n--- weight-buffer sanity (plain torch) ---")
    x = torch.randn(16, cfg.hidden_size)
    out = torch.nn.functional.linear(x, qkv.weight.detach().cpu())
    print(f"F.linear output width : {out.shape[-1]}")
    check("F.linear width == buffer width", out.shape[-1] == buffer_width,
          f"{out.shape[-1]} vs {buffer_width}")
    q, k, v = out.split(list(qkv.output_sizes), dim=-1)
    print(f"split by layer sizes  : q={q.shape[-1]} k={k.shape[-1]} "
          f"v={v.shape[-1]}")
    check("v nonempty after buffer split", v.numel() > 0)

    # ------------------------------------------------------------------
    # 5. torchax split semantics: prove the silent clamp that hides a width
    #    mismatch at runtime on TPU.
    # ------------------------------------------------------------------
    print("\n--- torchax split clamp probe ---")
    import torchax
    sizes_model = [sa.q_size, sa.kv_size, sa.kv_size]
    narrow = model_width - sa.kv_size  # one kv piece short
    with torchax.default_env():
        t = torch.randn(16, narrow).to("jax")
        try:
            qq, kk, vv = t.split(sizes_model, dim=-1)
            print(f"torchax split of width-{narrow} by {sizes_model}: "
                  f"q={qq.shape[-1]} k={kk.shape[-1]} v={vv.shape[-1]}")
            check("torchax clamps oversized split silently (masking bug)",
                  vv.shape[-1] == 0,
                  "if this fails, torchax validates and the TPU failure "
                  "mode is elsewhere")
        except Exception as e:  # noqa: BLE001
            check("torchax split raised (no silent clamp)", False,
                  f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # 6. The load path: real checkpoint layout through load_weights, then
    #    re-check the width and that the tiled kv rows are populated.
    # ------------------------------------------------------------------
    print("\n--- load_weights (real 30B key layout, tiled kv) ---")
    from transformers import MuseGlimmerTextModel
    hf = R.randomise_(MuseGlimmerTextModel(cfg), seed=0).eval().float()
    state = {f"model.language_model.{k}": v.detach().clone()
             for k, v in hf.state_dict().items()}
    state["lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.05
    q_sz_l, k_sz_l, v_sz_l = qkv.output_sizes
    try:
        model.load_weights(iter(state.items()))
        w = model.model.layers[0].self_attn.qkv_proj.weight.detach().cpu()
        print(f"post-load weight shape: {tuple(w.shape)}")
        check("post-load rows unchanged", w.shape[0] == buffer_width,
              f"{w.shape[0]} vs {buffer_width}")
        v_rows = w[q_sz_l + k_sz_l:q_sz_l + k_sz_l + v_sz_l]
        check("v region populated (nonzero) after tiling",
              v_rows.numel() > 0 and float(v_rows.abs().sum()) > 0,
              f"rows={v_rows.shape[0]} sum|.|={float(v_rows.abs().sum()):.4f}")
        # replication correctness: with 2 real kv heads tiled to mesh TP=4,
        # kv head pairs must be exact copies (repeat_interleave: h0 h0 h1 h1).
        reps = getattr(qkv, "num_kv_head_replicas", 1)
        if reps > 1:
            k_rows = w[q_sz_l:q_sz_l + k_sz_l]
            kh = k_rows.reshape(-1, head, k_rows.shape[-1])
            same01 = torch.allclose(kh[0], kh[1])
            distinct = kh.shape[0] < 2 * reps or not torch.allclose(
                kh[0], kh[2 * reps - 1])
            check("k replication is repeat_interleave (h0 h0 h1 h1)",
                  same01 and distinct,
                  f"kv-head blocks={kh.shape[0]} replicas={reps}")
    except Exception as e:  # noqa: BLE001
        check("load_weights on OOT-built model", False,
              f"{type(e).__name__}: {e}")

    DIVERGED: list[tuple[str, list[str]]] = []

    def verdict_if_diverged(stage: str) -> int | None:
        """Stop at the FIRST stage whose width diverges."""
        if FAILURES:
            DIVERGED.append((stage, list(FAILURES)))
            if len(DIVERGED) == 1:
                print(f"\nFIRST-DIVERGENCE-STAGE: {stage}")
            else:
                print(f"\nALSO-DIVERGED-STAGE: {stage}")
            print("REPRO-FAILURES: " + "; ".join(FAILURES))
            FAILURES.clear()
            if not KEEP_GOING:
                print("\n--- verdict ---")
                return 1
            print("(MG_REPRO_KEEP_GOING=1: continuing through later stages)")
        return None

    if (rc := verdict_if_diverged("construction/load")) is not None:
        return rc

    def run_maybe_jit(tag: str, fn, *tensors):
        """Run a torchax computation eagerly; if the eager path refuses (the
        production path is always jitted), retry under torchax's jax_jit.
        Returns the output tensor or None (with a FAIL recorded)."""
        from torchax import interop as ti
        try:
            return fn(*tensors)
        except Exception as e_eager:  # noqa: BLE001
            try:
                out = ti.jax_jit(fn)(*tensors)
                print(f"[info] {tag}: eager failed "
                      f"({type(e_eager).__name__}), succeeded under jit")
                return out
            except Exception as e_jit:  # noqa: BLE001
                check(f"{tag} ran", False,
                      f"eager: {type(e_eager).__name__}: {e_eager} | "
                      f"jit: {type(e_jit).__name__}: {e_jit}")
                traceback.print_exc()
                return None

    def report_width(tag: str, out, expected: int, split_by=None) -> None:
        width = out.shape[-1]
        msg = f"{tag}: width={width}"
        if split_by is not None:
            qq, kk, vv = out.split(split_by, dim=-1)
            msg += (f" split q={qq.shape[-1]} k={kk.shape[-1]} "
                    f"v={vv.shape[-1]}")
        print(msg)
        check(f"{tag} width == {expected}", width == expected,
              f"got {width}")
        if split_by is not None:
            check(f"{tag} v piece == kv_size", vv.shape[-1] == split_by[2],
                  f"v width {vv.shape[-1]} vs {split_by[2]}")

    # ------------------------------------------------------------------
    # 7. STAGE A -- the real quant processing (transpose + reorder + shard
    #    to the mesh), the forward through quant_method.apply, and the OOT
    #    layer's own __call__ (the replication forward: reorder + slice +
    #    shard_map replica-collapse).  This is exactly the non-LoRA
    #    production path the crashing request took.
    # ------------------------------------------------------------------
    print("\n--- stage A: process_weights_after_loading + quant apply + "
          "OOT forward ---")
    from vllm.model_executor.layers.quantization.base_config import \
        QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        n_processed = 0
        for name, module in model.named_modules():
            qm = getattr(module, "quant_method", None)
            if isinstance(qm, QuantizeMethodBase):
                qm.process_weights_after_loading(module)
                n_processed += 1
    print(f"processed {n_processed} quant-method layers onto the mesh")

    lc = qkv.quant_method.linear_config
    print(f"linear_config.output_sizes : {lc.output_sizes} "
          f"(is layer.output_sizes: {lc.output_sizes is qkv.output_sizes})")
    print(f"linear_config.n_shards     : {lc.n_shards}   "
          f"fuse_matmuls: {lc.fuse_matmuls}")

    def _apply_quant(t):
        return qkv.quant_method.apply(qkv, t)

    def _call_qkv(t):
        r = qkv(t)
        return r[0] if isinstance(r, tuple) else r

    with torchax.default_env():
        x = torch.randn(16, cfg.hidden_size, dtype=torch.float32).to("jax")
        with jax.set_mesh(mesh):
            out_apply = run_maybe_jit("quant_method.apply", _apply_quant, x)
            if out_apply is not None:
                # apply is the LAYER-INTERNAL path: inflated buffer width.
                report_width("quant_method.apply (buffer contract)",
                             out_apply, buffer_width)
            out_fwd = run_maybe_jit("OOT qkv.__call__ (replication forward)",
                                    _call_qkv, x)
            if out_fwd is not None:
                # The model-facing path: must equal the model's declared
                # width, or the model's split silently truncates on TPU.
                report_width("OOT qkv.__call__ vs model declaration",
                             out_fwd, model_width, split_by=sizes_model)
                print(f"OOT qkv.__call__ absolute width: {out_fwd.shape[-1]} "
                      f"(stock contract {stock_width})")

    if (rc := verdict_if_diverged("A: quant apply / OOT forward")) is not None:
        return rc

    # ------------------------------------------------------------------
    # 7b. STAGE A2 -- full self_attn.forward BEFORE any LoRA wrap: the
    #     exact configuration of the crashed TPU run (no --enable-lora, the
    #     OOT replication forward live).  vLLM's Attention.forward views are
    #     kept live; only the paged-attention custom op is stubbed with a
    #     shape recorder.  A zero-width v shows up here exactly the way the
    #     TPU backend saw it: a (0, kv_heads, head_dim) value tensor -- or,
    #     with the model's post-split width assert in place, as that assert
    #     firing with the actual width in the message.
    # ------------------------------------------------------------------
    print("\n--- stage A2: self_attn.forward, NO LoRA (the crash config) ---")
    import vllm.model_executor.layers.attention.attention as attn_mod

    # Production moves every remaining CPU tensor (rotary cache, norm scales,
    # embeddings) to the mesh via shard_model_to_tpu before the first step.
    # Convert them in place with the SAME helper so the forward runs entirely
    # on jax tensors; this is device placement only, not width logic.
    from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
        _shard_tensor_to_tpu_replicated, _tensor_is_in_cpu, shard_model_to_tpu)

    n_conv = 0
    for holder in model.modules():
        for coll in (holder._parameters, holder._buffers):
            for pname, tsr in list(coll.items()):
                if tsr is None or not _tensor_is_in_cpu(tsr):
                    continue
                conv = _shard_tensor_to_tpu_replicated(tsr, mesh)
                if coll is holder._parameters:
                    conv = torch.nn.Parameter(conv, requires_grad=False)
                coll[pname] = conv
                n_conv += 1
    print(f"converted {n_conv} remaining CPU tensors to the mesh")

    rec: dict[str, tuple] = {}

    def fake_kv_update(key, value, layer_name, *a, **k):
        rec["kv_update"] = (tuple(key.shape), tuple(value.shape))
        return None

    def fake_uawo(query, key, value, output, layer_name, **k):
        rec["q"] = tuple(query.shape)
        rec["k"] = tuple(key.shape)
        rec["v"] = tuple(value.shape)
        return None

    orig_uawo = attn_mod.unified_attention_with_output
    orig_kvu = attn_mod.unified_kv_cache_update
    sa.attn.use_direct_call = True

    # vLLM's IR ops (rms_norm, rotary) route through torch.library wrappers
    # whose dynamo-disable shim breaks torchax's function-mode interception on
    # CPU.  tpu-inference's own aux paths run torch modules under torchax
    # with the wrap disabled (`enable_torch_wrap(False)`), which dispatches
    # the native python impls instead -- identical arithmetic, no wrapper.
    from vllm.ir import enable_torch_wrap

    def _call_sa(h, pos):
        with enable_torch_wrap(False):
            return sa.forward(positions=pos, hidden_states=h)

    def run_self_attn(stage_tag: str):
        rec.clear()
        attn_mod.unified_attention_with_output = fake_uawo
        attn_mod.unified_kv_cache_update = fake_kv_update
        try:
            with torchax.default_env():
                h = torch.randn(16, cfg.hidden_size,
                                dtype=torch.float32).to("jax")
                pos = torch.arange(16).to("jax")
                with jax.set_mesh(mesh):
                    out_sa = run_maybe_jit(f"{stage_tag} self_attn.forward",
                                           _call_sa, h, pos)
            if out_sa is not None:
                print(f"attention views seen by the backend: q={rec.get('q')} "
                      f"k={rec.get('k')} v={rec.get('v')}")
                check(f"{stage_tag} backend q view is (16, heads, head_dim)",
                      rec.get("q") == (16, sa.num_heads, head),
                      f"{rec.get('q')}")
                check(f"{stage_tag} backend k view is (16, kv_heads, "
                      f"head_dim)",
                      rec.get("k") == (16, sa.num_kv_heads, head),
                      f"{rec.get('k')}")
                check(f"{stage_tag} backend v view is (16, kv_heads, "
                      "head_dim) -- TPU crash was (0, kv_heads, head_dim)",
                      rec.get("v") == (16, sa.num_kv_heads, head),
                      f"{rec.get('v')}")
                check(f"{stage_tag} self_attn output width == hidden",
                      out_sa.shape[-1] == cfg.hidden_size,
                      f"{tuple(out_sa.shape)}")
            else:
                print(f"attention views recorded before failure: {rec}")
        finally:
            attn_mod.unified_attention_with_output = orig_uawo
            attn_mod.unified_kv_cache_update = orig_kvu

    run_self_attn("A2 (no LoRA)")
    check("model num_kv_heads == real (stock) kv heads",
          sa.num_kv_heads == qkv.total_num_kv_heads,
          f"model {sa.num_kv_heads} vs real {qkv.total_num_kv_heads}")

    if (rc := verdict_if_diverged("A2: self_attn, no LoRA")) is not None:
        return rc

    # ------------------------------------------------------------------
    # 8. STAGE B -- the production LoRA wrap (vllm_model_wrapper
    #    load_lora_model), then the wrapped forward with NO active adapter
    #    (the crashing request had lora_request=None).
    # ------------------------------------------------------------------
    print("\n--- stage B: LoRA wrap (production load_lora_model) ---")
    import vllm.envs as vllm_envs
    from vllm.config import LoRAConfig
    from vllm.platforms import current_platform

    print(f"VLLM_LORA_ENABLE_DUAL_STREAM = "
          f"{vllm_envs.VLLM_LORA_ENABLE_DUAL_STREAM}")

    lora_config = LoRAConfig(max_lora_rank=8,
                             max_loras=1,
                             max_cpu_loras=1,
                             lora_dtype=torch.float32)
    vllm_config.lora_config = lora_config

    # Production runs on the TPU platform, whose get_punica_wrapper returns
    # exactly this class; the CPU platform has none, so pin it.
    current_platform.get_punica_wrapper = (
        lambda *a, **k: "tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU")

    from tpu_inference.models.vllm.vllm_model_wrapper import (load_lora_model,
                                                              replace_set_lora)

    with torchax.default_env():
        lora_manager, model_wrapped = load_lora_model(model,
                                                      vllm_config,
                                                      device="jax")
    replace_set_lora(model_wrapped)

    wqkv = model.model.layers[0].self_attn.qkv_proj
    print(f"wrapped qkv type   : {type(wqkv).__name__}")
    check("qkv got LoRA-wrapped",
          type(wqkv).__name__ == "MergedQKVParallelLinearWithLoRA",
          type(wqkv).__name__)
    print(f"wrapper n_slices   : {getattr(wqkv, 'n_slices', 'n/a')}")
    print(f"wrapper output_slices: {getattr(wqkv, 'output_slices', 'n/a')}")
    print(f"enable_aux_cuda_stream on wrapper: "
          f"{getattr(wqkv, '_enable_aux_cuda_stream', 'n/a')}")

    # lora_request=None => every token maps to LoRA index -1 (no adapter).
    def _call_wqkv(t):
        r = wqkv(t)
        return r[0] if isinstance(r, tuple) else r

    with torchax.default_env():
        pw = wqkv.punica_wrapper
        pw._token_lora_indices = torch.full(
            tuple(pw._token_lora_indices.shape), -1,
            dtype=torch.int32).to("jax")

        x = torch.randn(16, cfg.hidden_size, dtype=torch.float32).to("jax")
        with jax.set_mesh(mesh):
            out_l = run_maybe_jit("LoRA-wrapped qkv forward (no adapter)",
                                  _call_wqkv, x)
            if out_l is not None:
                # Must match BOTH the model declaration and the non-LoRA
                # (stock) contract, or LoRA on/off silently changes geometry.
                report_width("LoRA-wrapped qkv forward vs model declaration",
                             out_l, model_width, split_by=sizes_model)
                check("LoRA-wrapped width == non-LoRA (stock) width",
                      out_l.shape[-1] == stock_width,
                      f"{out_l.shape[-1]} vs stock {stock_width} -- LoRA "
                      f"path bypasses the OOT forward's replica collapse")

    if (rc := verdict_if_diverged("B: LoRA wrap")) is not None:
        return rc

    # ------------------------------------------------------------------
    # 9. STAGE C -- full self_attn.forward again, now through the
    #    LoRA-WRAPPED qkv (same live Attention.forward views and stubs as
    #    stage A2).  With the seam consistent, wrapped and unwrapped
    #    forwards must show identical geometry.
    # ------------------------------------------------------------------
    print("\n--- stage C: self_attn.forward through the LoRA wrap ---")
    run_self_attn("C (LoRA-wrapped)")

    if (rc := verdict_if_diverged("C: full self_attn")) is not None:
        return rc

    # ------------------------------------------------------------------
    # 10. STAGE D -- the full 4-layer model forward, in PRODUCTION SHAPE:
    #     params flow as jax pytrees into a jax.jit'd function that enters
    #     torchax inside the trace and runs torch.func.functional_call --
    #     exactly vllm_model_wrapper's step function.  (Purely eager
    #     execution trips vLLM's compiled custom-op shims on CPU; production
    #     never runs eager.)
    # ------------------------------------------------------------------
    print("\n--- stage D: full model forward (shard_model_to_tpu + "
          "functional_call under jax.jit, as production) ---")
    from torchax.interop import jax_view as jview
    from torchax.interop import torch_view as tview

    rec.clear()
    attn_mod.unified_attention_with_output = fake_uawo
    attn_mod.unified_kv_cache_update = fake_kv_update
    for layer in model.model.layers:
        layer.self_attn.attn.use_direct_call = True

    def _model_fn(params_jax, ids_jax, pos_jax):
        with torchax.default_env(), enable_torch_wrap(False):
            out = torch.func.functional_call(model,
                                             tview(params_jax),
                                             kwargs={
                                                 "input_ids": tview(ids_jax),
                                                 "positions": tview(pos_jax),
                                             },
                                             tie_weights=False)
            return jview(out)

    try:
        params_and_buffers = shard_model_to_tpu(model, mesh)
        print(f"params_and_buffers entries: {len(params_and_buffers)}")
        with torchax.default_env():
            ids = torch.arange(16).to("jax")
            pos = torch.arange(16).to("jax")
            params_jax = jview(params_and_buffers)
            ids_jax, pos_jax = jview(ids), jview(pos)
        with jax.set_mesh(mesh):
            hs_jax = jax.jit(_model_fn)(params_jax, ids_jax, pos_jax)
        check("model forward output width == hidden",
              hs_jax.shape[-1] == cfg.hidden_size, f"{tuple(hs_jax.shape)}")
        print(f"last attention views: q={rec.get('q')} k={rec.get('k')} "
              f"v={rec.get('v')}")
        check("last layer backend v view nonempty",
              rec.get("v") == (16, sa.num_kv_heads, head),
              f"{rec.get('v')}")
    except Exception as e:  # noqa: BLE001
        check("full model forward ran", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        attn_mod.unified_attention_with_output = orig_uawo
        attn_mod.unified_kv_cache_update = orig_kvu

    if FAILURES:
        DIVERGED.append(("D: full model", list(FAILURES)))
        if len(DIVERGED) == 1:
            print("\nFIRST-DIVERGENCE-STAGE: D: full model")
        else:
            print("\nALSO-DIVERGED-STAGE: D: full model")
        print("REPRO-FAILURES: " + "; ".join(FAILURES))
        FAILURES.clear()

    # ------------------------------------------------------------------
    print("\n--- verdict ---")
    if DIVERGED:
        print(f"FIRST-DIVERGENCE-STAGE: {DIVERGED[0][0]}")
        for stage, fails in DIVERGED:
            print(f"  [{stage}] " + "; ".join(fails))
        return 1
    print("ALL-STAGES-GREEN (construction, load, quant processing, quant "
          "apply, OOT forward, no-LoRA self_attn, LoRA wrap, no-adapter "
          "forward, wrapped self_attn, full model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
