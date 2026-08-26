#!/usr/bin/env python3
"""Measure the muse fb HBM arena as a function of sequence length.

Motivation (2026-08-22): the RL cell's [1, 18432] jit_forward_backward_fn asks
to reserve 56.99G with 56.45G free, and the ask is bit-identical across
train_token_budget (45056/22528), num_vocab_tiling (8/32) and flce_tile_size
(2048/1024) -- only max_target_length moves it (60.47G @22528). Two points fit
any curve; this probe collects the curve. The fb program has only ever been
validated at the lora_smoke default of 512 tokens.

Run ON the trainer host with the TPU free (tinker down, jobman controller
paused), inside the trainer venv:

    for L in 4096 8192 12288 18432; do
      TUNIX_UNIFORM_SEQ_LEN=$L ~/SkyRLTpu/.venv/bin/python fb_mem_probe.py \
        --base-model ~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/<snap> \
        --orbax ~/skyrl-maxtext-ckpts-local/muse-glimmer-30b/0/items \
        --length $L --out /tmp/fbprobe_$L.json
    done

One length per process: a fresh process per point keeps XLA state and the
allocator clean, so the reserve numbers are comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

# Must match the cell (cell_worker m-* + pick_tiles muse) EXACTLY -- the point
# is to reproduce the failing program, smaller.
CELL_MAXTEXT_KWARGS = {
    "remat_policy": "full",
    "ici_fsdp_parallelism": 4,
    "num_vocab_tiling": 32,
}
CELL_FLCE_TILE = 1024


def _batch(tokens: List[int], loss_fn: str, n: int = 1):
    """n datums of identical length -- the packing path only engages with >1."""
    from skyrl.tinker import types

    return types.PreparedModelPassBatch(
        all_model_ids=["mg_probe"] * n,
        all_model_inputs=[types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)]) for _ in range(n)],
        all_targets=[list(tokens[1:]) + [tokens[0]] for _ in range(n)],
        all_token_weights=[[1.0] * len(tokens) for _ in range(n)],
        all_sampling_logprobs=[[0.0] * len(tokens) for _ in range(n)],
        all_advantages=[[1.0] * len(tokens) for _ in range(n)],
        all_values=[[0.0] * len(tokens) for _ in range(n)],
        all_returns=[[0.0] * len(tokens) for _ in range(n)],
        all_loss_fns=[loss_fn] * n,
        all_loss_fn_configs=[{}] * n,
        request_batch_slices=[("0", "mg_probe", 0, n)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--orbax", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=1,
                    help="fb calls to make; the FIRST includes JIT compile, so "
                         "report the later ones as steady state.")
    ap.add_argument("--datums", type=int, default=1,
                    help="How many datums to send; >1 exercises budget packing.")
    ap.add_argument("--budget", type=int, default=0,
                    help="train_token_budget; 0 = same as --length (1 seq/microbatch).")
    ap.add_argument("--remat", default="full",
                    help="maxtext remat_policy; 'none' omits the key entirely.")
    ap.add_argument("--nvt", type=int, default=32, help="num_vocab_tiling.")
    ap.add_argument("--flce", type=int, default=1024, help="flce_tile_size.")
    ap.add_argument("--rank", type=int, default=32,
                    help="LoRA rank. Sweeping this separates LoRA-sized cost "
                         "from base-weight duplication in the create_model delta.")
    ap.add_argument("--free-base", action="store_true",
                    help="Set free_base_state_after_template: release the pristine "
                         "base copy once the template exists.")
    ap.add_argument("--skip-fb", action="store_true",
                    help="Measure load/LoRA HBM only -- the create_model delta "
                         "is the interesting number and needs no fb.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = args.length
    assert os.environ.get("TUNIX_UNIFORM_SEQ_LEN") == str(L), (
        "set TUNIX_UNIFORM_SEQ_LEN to --length so the uniform-shape fb path "
        "compiles the same program family the cell does"
    )

    import jax
    import numpy as np

    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig
    from skyrl.tinker import types

    _kwargs = {"ici_fsdp_parallelism": 4, "num_vocab_tiling": args.nvt}
    if args.remat != "none":
        _kwargs["remat_policy"] = args.remat
    res: Dict[str, Any] = {
        "length": L, "kwargs": _kwargs, "flce": args.flce,
        "budget": args.budget or L, "seqs_per_microbatch": (args.budget or L) // L,
    }

    cfg = TunixBackendConfig(
        model_source="maxtext",
        maxtext_model_name="muse-glimmer-30b",
        model_path=args.orbax,
        maxtext_max_target_length=L,
        maxtext_kwargs=_kwargs,
        flce_tile_size=args.flce,
        train_token_budget=(args.budget or L),
        param_dtype="bfloat16",
        max_lora_rank=max(args.rank, 32),
        inference_backend="native",
        train_micro_batch_size=1,
        free_base_state_after_template=args.free_base,
    )
    t0 = time.time()
    backend = TunixBackend(args.base_model, cfg)
    res["backend_init_s"] = round(time.time() - t0, 1)

    def hbm() -> Dict[str, float]:
        d = jax.local_devices()[0]
        try:
            s = d.memory_stats()
            return {
                "in_use_gib": s.get("bytes_in_use", 0) / 2**30,
                "limit_gib": s.get("bytes_limit", 0) / 2**30,
                "peak_gib": s.get("peak_bytes_in_use", 0) / 2**30,
            }
        except Exception:  # noqa: BLE001
            return {}

    res["hbm_after_load"] = hbm()
    backend.create_model("mg_probe", types.LoraConfig(rank=args.rank, alpha=2.0 * args.rank, seed=0))
    res["hbm_after_lora"] = hbm()
    res["rank"] = args.rank
    res["datums"] = args.datums
    res["free_base"] = args.free_base
    res["create_model_delta_gib"] = round(
        res["hbm_after_lora"]["in_use_gib"] - res["hbm_after_load"]["in_use_gib"], 2)
    if args.skip_fb:
        json.dump(res, open(args.out, "w"), indent=2)
        print(json.dumps(res, indent=2), flush=True)
        return 0

    tokens = list(range(2, L + 1))  # exactly L-1 tokens + BOS-ish; padded to L by uniform mode
    tokens = tokens[: L - 1]
    tokens = [min(t, 199000) for t in tokens]

    try:
        _times = []
        for _i in range(max(1, args.repeat)):
            _t = time.time()
            out = backend.forward_backward(_batch(tokens, "importance_sampling", args.datums))
            _times.append(round(time.time() - _t, 1))
        res["fb_seconds_each"] = _times
        res["fb_seconds"] = _times[0]
        res["fb_steady"] = _times[-1]
        res["fb_steady_per_datum"] = round(_times[-1] / max(1, args.datums), 2)
        res["fit"] = True
        res["hbm_after_fb"] = hbm()
        o = out["0"].loss_fn_outputs[0]
        for k in ("elementwise_loss", "loss", "policy_loss"):
            if k in o:
                res["mean_loss"] = float(np.mean(o[k]["data"]))
                break
    except Exception as exc:  # noqa: BLE001
        res["fit"] = False
        msg = str(exc)
        res["error_head"] = msg[:400]
        m = re.search(r"reserve ([0-9.]+)G", msg)
        n = re.search(r"([0-9.]+)G free", msg)
        if m:
            res["reserve_gib"] = float(m.group(1))
        if n:
            res["free_gib"] = float(n.group(1))
        res["hbm_after_fail"] = hbm()

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "error_head"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
