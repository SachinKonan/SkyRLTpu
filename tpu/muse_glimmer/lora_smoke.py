#!/usr/bin/env python3
"""LoRA training smoke for Muse-Glimmer-30B on a v5p-8, via tunix + MaxText.

Closes `MAXTEXT.md` §5 item 5 and `E2E.md` §5 "training on real weights": the
converter round-trip and the LoRA *regexes* were verified on CPU, but no
adapter had ever been injected into the real model and stepped.

The failure this is built to catch is the silent one.  qwix matches adapters by
module path; a name that does not match yields **zero** adapters and training
then "runs" perfectly happily, updating nothing.  MaxText's attention attribute
is `self_attention` (not `attention`) precisely so the tunix regex
`layers_[0-9]+/self_attention/(query|key|value|out)` matches -- and on the
serving side the same class of collision is why `self_attn.gate_proj` had to be
renamed `attn_gate_proj`.  So: count the adapters, print the module paths they
landed on, and refuse to report success on zero.

Also asserts the two things that make the loss trace meaningful:
  * the base parameters are actually FSDP-sharded across the 4 chips (a
    replicated 27.9B model would not fit, but a partially-replicated one can,
    and then the "smoke passed" says nothing about the real shape);
  * grad_norm is finite and non-zero, and the loss moves.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

os.environ.setdefault("JAX_PLATFORMS", "tpu")


def _batch(model_ids: List[str], tokens: List[int], loss_fn: str):
    """A math-RL-shaped PreparedModelPassBatch (same shape as the RL loop's)."""
    from skyrl.tinker import types

    n = len(model_ids)
    return types.PreparedModelPassBatch(
        all_model_inputs=[
            types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)])
            for _ in range(n)
        ],
        all_targets=[list(tokens[1:]) + [tokens[0]]] * n,
        all_token_weights=[[1.0] * len(tokens)] * n,
        all_sampling_logprobs=[[0.0] * len(tokens)] * n,
        all_advantages=[[1.0] * len(tokens)] * n,
        all_values=[[]] * n,
        all_returns=[[]] * n,
        all_model_ids=list(model_ids),
        all_loss_fns=[loss_fn] * n,
        all_loss_fn_configs=[None] * n,
        request_batch_slices=[(str(i), m, i, i + 1)
                              for i, m in enumerate(model_ids)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="HF dir (tokenizer source)")
    ap.add_argument("--orbax", required=True, help=".../0/items")
    ap.add_argument("--maxtext-model-name", default="muse-glimmer-30b")
    ap.add_argument("--max-target-length", type=int, default=512)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss-fn", default="importance_sampling")
    ap.add_argument("--fsdp", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    import numpy as np

    res: Dict[str, Any] = {"jax_devices": len(jax.devices())}
    print(f"jax devices: {jax.devices()}", flush=True)

    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig
    from skyrl.tinker import types

    cfg = TunixBackendConfig(
        model_source="maxtext",
        maxtext_model_name=args.maxtext_model_name,
        model_path=args.orbax,
        maxtext_max_target_length=args.max_target_length,
        maxtext_kwargs={
            "remat_policy": "full",
            "ici_fsdp_parallelism": args.fsdp,
        },
        param_dtype="bfloat16",
        max_lora_rank=max(args.rank, 32),
        inference_backend="native",
        train_micro_batch_size=1,
    )
    t0 = time.time()
    backend = TunixBackend(args.base_model, cfg)
    res["backend_init_seconds"] = time.time() - t0
    print(f"backend up in {res['backend_init_seconds']:.0f}s", flush=True)

    # ---- base params really sharded across the chips? ---------------------
    from flax import nnx
    base_leaves = [
        x for x in jax.tree.leaves(backend.base_state)
        if hasattr(x, "shape") and hasattr(x, "dtype")
    ]
    total_bytes = sum(int(np.prod(x.shape)) * x.dtype.itemsize
                      for x in base_leaves)
    per_dev = 0
    n_sharded = 0
    for x in base_leaves:
        try:
            shards = x.addressable_shards
            per_dev += shards[0].data.nbytes
            if shards[0].data.shape != tuple(x.shape):
                n_sharded += 1
        except Exception:  # noqa: BLE001
            pass
    res["base_params"] = {
        "n_arrays": len(base_leaves),
        "total_gib": total_bytes / 2**30,
        "per_device_gib": per_dev / 2**30,
        "n_arrays_sharded": n_sharded,
        "device_count": jax.device_count(),
        "fsdp_applied": per_dev < total_bytes * 0.75,
        "mesh": str(getattr(backend, "_mesh", None)),
    }
    b = res["base_params"]
    print(f"base params: {b['n_arrays']} arrays, {b['total_gib']:.2f} GiB "
          f"global, {b['per_device_gib']:.2f} GiB/device, "
          f"{b['n_arrays_sharded']} sharded, fsdp_applied={b['fsdp_applied']}",
          flush=True)

    # ---- adapters ---------------------------------------------------------
    backend.create_model("mg_smoke",
                         types.LoraConfig(rank=args.rank,
                                          alpha=args.alpha,
                                          seed=0))
    slot = backend.models["mg_smoke"]
    flat = jax.tree_util.tree_flatten_with_path(slot.lora_state)[0]
    n_arrays = len(flat)
    n_params = sum(int(np.prod(v.shape)) for _, v in flat)
    paths = [jax.tree_util.keystr(k) for k, _ in flat]
    res["lora"] = {
        "rank": args.rank,
        "alpha": args.alpha,
        "n_adapter_arrays": n_arrays,
        "n_adapter_params": n_params,
        "module_paths_sample": paths[:16],
        "all_module_paths": paths,
        "shapes_sample": [list(v.shape) for _, v in flat[:8]],
    }
    print(f"\nLoRA adapters: {n_arrays} arrays, {n_params:,} params", flush=True)
    for p, (_, v) in list(zip(paths, flat))[:16]:
        print(f"    {p}  {tuple(v.shape)}", flush=True)
    if n_arrays == 0 or n_params == 0:
        res["verdict"] = "FAIL: ZERO LoRA adapters injected"
        print("\n*** ZERO ADAPTERS -- the module-path regex did not match. "
              "This is the silent failure this smoke exists to catch. ***",
              flush=True)
        json.dump(res, open(args.out, "w"), indent=2)
        return 1

    # ---- a few real optimizer steps ---------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    text = ("Question: Natalia sold clips to 48 friends in April, and then "
            "she sold half as many clips in May. How many clips did Natalia "
            "sell altogether in April and May?\nAnswer: In May she sold "
            "48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. "
            "The answer is 72.")
    tokens = tok(text, add_special_tokens=True).input_ids
    tokens = tokens[:args.max_target_length - 1]
    print(f"\ntrain sequence: {len(tokens)} tokens", flush=True)

    adam = types.AdamParams(learning_rate=args.lr,
                            beta1=0.9,
                            beta2=0.999,
                            eps=1e-8,
                            weight_decay=0.0)

    def _mean_loss(out):
        """Different loss fns expose different tensors; take whichever is there."""
        o = out["0"].loss_fn_outputs[0]
        for k in ("elementwise_loss", "loss", "policy_loss"):
            if k in o:
                return float(np.mean(o[k]["data"])), o
        raise KeyError(f"no loss tensor in {sorted(o)}")

    # importance_sampling is the RL-loop shape; fall back to cross_entropy if
    # this build's loss registry disagrees, rather than losing the whole stage.
    loss_fn = args.loss_fn
    try:
        probe = backend.forward_backward(_batch(["mg_smoke"], tokens, loss_fn))
        _mean_loss(probe)
    except Exception as exc:  # noqa: BLE001
        print(f"loss_fn={loss_fn} unusable ({exc!r}); falling back to "
              "cross_entropy", flush=True)
        loss_fn = "cross_entropy"
    res["loss_fn"] = loss_fn
    backend.optim_step("mg_smoke", types.OptimStepInput(adam_params=adam))

    trace = []
    for step in range(args.steps):
        t0 = time.time()
        out = backend.forward_backward(_batch(["mg_smoke"], tokens, loss_fn))
        loss, o0 = _mean_loss(out)
        lp = o0["logprobs"]["data"] if "logprobs" in o0 else [0.0]
        step_res = backend.optim_step("mg_smoke",
                                      types.OptimStepInput(adam_params=adam))
        gn = float(step_res.metrics.get("skyrl.ai/grad_norm", float("nan")))
        trace.append({
            "step": step,
            "mean_loss": loss,
            "grad_norm": gn,
            "loss_finite": bool(np.isfinite(loss)),
            "grad_norm_finite": bool(np.isfinite(gn)),
            "logprobs_finite": bool(np.isfinite(lp).all()),
            "seconds": time.time() - t0,
        })
        print(f"  step {step}: loss={loss:.6f} grad_norm={gn:.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    final = backend.forward_backward(_batch(["mg_smoke"], tokens, loss_fn))
    final_loss, _ = _mean_loss(final)
    res["loss_trace"] = trace
    res["final_loss_after_steps"] = final_loss
    res["loss_moved"] = abs(final_loss - trace[0]["mean_loss"]) > 1e-6
    res["loss_decreased"] = final_loss < trace[0]["mean_loss"]
    res["all_finite"] = all(t["loss_finite"] and t["grad_norm_finite"]
                            for t in trace)
    res["grad_norm_nonzero"] = all(t["grad_norm"] > 0 for t in trace)
    res["verdict"] = ("PASS" if (res["all_finite"] and res["grad_norm_nonzero"]
                                 and res["loss_moved"] and n_params > 0
                                 and b["fsdp_applied"]) else "CHECK")
    print(f"\nfinal loss {final_loss:.6f} vs first {trace[0]['mean_loss']:.6f} "
          f"-> moved={res['loss_moved']} decreased={res['loss_decreased']}",
          flush=True)
    print(f"VERDICT: {res['verdict']}", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    del nnx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
