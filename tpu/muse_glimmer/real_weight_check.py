#!/usr/bin/env python3
"""Real-weight teacher-forced parity + greedy reference for Muse-Glimmer-30B.

``tpu/muse_glimmer/parity_check.py`` proves the *math* on tiny random weights.
This proves the **weights themselves**: that the 627 text tensors of the real
30B checkpoint land where the port thinks they do.  A mis-mapped tensor shows
up here as an O(1) error at one layer instead of as a mystery on the TPU.

Two sides, run as two SEPARATE processes so neither has to hold both models:

  --side hf    torch ``MuseGlimmerForConditionalGeneration`` (text path),
               float32 on CPU.  Dumps, per prompt:
                 * the exact token ids (so the TPU is fed ids, never text --
                   no tokenizer/BOS ambiguity in the greedy comparison)
                 * per-position argmax + top-8 over the whole prompt
                 * full float32 logit rows at a sample of positions
                 * per-LAYER hidden states for the short prompts (localises a
                   mis-mapped tensor to a layer)
                 * greedy continuations (the reference for the TPU run)

  --side jax   ``tpu_inference.models.jax.muse_glimmer_core`` fed straight from
               the safetensors shards, float32, and compared against the dump.

The JAX side is imported straight from the file so this harness never imports
the ``tpu_inference`` package (which drags in vllm + TPU-only deps).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = (REPO_ROOT / "third_party" / "tpu-inference" / "tpu_inference" /
             "models" / "jax" / "muse_glimmer_core.py")

# Positions whose full float32 logit row is dumped (clipped to the prompt).
_FULL_ROW_STRIDE = 97


def _load_core():
    if not CORE_PATH.exists():
        raise SystemExit(f"muse_glimmer_core.py not found at {CORE_PATH}")
    spec = importlib.util.spec_from_file_location("muse_glimmer_core",
                                                  CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["muse_glimmer_core"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Prompts.  Lengths are chosen to exercise, on the TPU side:
#   * a very short prompt,
#   * a non-block-divisible length (block sizes are powers of two),
#   * a mid-length prompt,
#   * a prompt LONGER THAN THE 2048 SLIDING WINDOW, so the 39 sliding layers
#     and the 13 full-attention layers genuinely diverge.
# ---------------------------------------------------------------------------

_FILLER_PARAS = [
    "The kettle in the observatory kitchen had a habit of whistling exactly "
    "when the seeing was best, which the graduate students took as a personal "
    "insult and the postdocs took as a schedule.",
    "Cartography in the delta is a seasonal profession: the channels braid and "
    "rebraid every spring, so a map printed in March is a historical document "
    "by August and a liability by October.",
    "He kept the ledger in three colours of ink, one for money that had moved, "
    "one for money that had been promised, and one for money that existed only "
    "as a shared conviction among four people in a room.",
    "The restoration team argued for six weeks about a single millimetre of "
    "varnish, and in the end the decision was made by a conservator who had "
    "not spoken once during any of the meetings.",
    "Freight timetables are written in a dialect of optimism; the arrival "
    "column describes not when the train will arrive but when it would arrive "
    "in a world where nothing at all had gone wrong.",
    "Every language in the valley has a word for the particular grey of the "
    "sky an hour before hail, and no two of those words are cognate, which "
    "linguists find either delightful or infuriating.",
]


def build_prompts() -> List[Dict[str, Any]]:
    long_body = " ".join(_FILLER_PARAS[i % len(_FILLER_PARAS)]
                         for i in range(90))
    return [
        {
            "name": "p1_tiny",
            "text": "The capital of France is",
            "max_new_tokens": 32,
        },
        {
            "name": "p2_odd",
            "text":
            ("Explain, in one paragraph and without using bullet points, why a "
             "ragged paged attention kernel needs to know the sliding window "
             "size for some layers and not for others."),
            "max_new_tokens": 48,
        },
        {
            "name": "p3_mid",
            "text":
            ("A technical note.\n\n" + " ".join(_FILLER_PARAS) +
             "\n\nSummarise the passage above in exactly two sentences, then "
             "state what the six paragraphs have in common.\n\nSummary:"),
            "max_new_tokens": 48,
        },
        {
            "name": "p4_long_sliding",
            "text": ("Archive fragment.\n\n" + long_body +
                     "\n\nQuestion: taken together, what single idea do the "
                     "fragments above keep circling back to?\n\nAnswer:"),
            "max_new_tokens": 32,
        },
        {
            "name": "p5_code",
            "text":
            ("def rms_norm(x, eps):\n"
             "    # normalise x over its last axis, reducing in float32\n"
             "    xf = x.astype('float32')\n"),
            "max_new_tokens": 32,
        },
    ]


# ---------------------------------------------------------------------------
# Diff reporting
# ---------------------------------------------------------------------------


def diff_stats(ref: np.ndarray, got: np.ndarray) -> Dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    d = np.abs(ref - got)
    peak = float(np.max(np.abs(ref))) or 1.0
    # Element-wise relative error is meaningless near zero, so also report it
    # restricted to entries above 1% of peak (the SPEC's gating convention).
    big = np.abs(ref) > 0.01 * peak
    rel_big = float(np.max(d[big] / np.abs(ref[big]))) if big.any() else 0.0
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_abs_over_peak": float(d.max() / peak),
        "max_rel_above_1pct_peak": rel_big,
        "peak_ref": peak,
    }


# ---------------------------------------------------------------------------
# HF side
# ---------------------------------------------------------------------------


def run_hf(args) -> int:
    import torch
    import transformers
    from transformers import AutoTokenizer

    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    print(f"transformers {transformers.__version__} | torch {torch.__version__}",
          flush=True)

    from transformers import MuseGlimmerForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.time()
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"loaded HF model in {time.time() - t0:.1f}s", flush=True)

    text_cfg = model.config.get_text_config()
    print("text_config:", {
        k: getattr(text_cfg, k, None)
        for k in ("hidden_size", "num_hidden_layers", "sliding_window",
                  "vocab_size", "tie_word_embeddings")
    }, flush=True)
    # Trap 8: assert the head really is untied on this checkpoint.
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    print("lm_head untied:", emb.data_ptr() != head.data_ptr(),
          "| max|lm_head - embed| =",
          float((head - emb).abs().max()) if head.shape == emb.shape else "n/a",
          flush=True)

    out: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "attn_implementation": "eager",
        "lm_head_untied": bool(emb.data_ptr() != head.data_ptr()),
    }
    summary: Dict[str, Any] = {}

    for spec in build_prompts():
        name = spec["name"]
        ids = tok(spec["text"], return_tensors="pt",
                  add_special_tokens=True).input_ids
        T = ids.shape[1]
        print(f"\n=== {name}: {T} tokens ===", flush=True)

        want_layers = T <= args.layer_dump_max_len
        t0 = time.time()
        res = model(input_ids=ids, output_hidden_states=want_layers)
        logits = res.logits[0].to(torch.float32).numpy()  # [T, V]
        print(f"  forward {time.time() - t0:.1f}s  logits {logits.shape}",
              flush=True)

        order = np.argsort(-logits, axis=-1)[:, :8]
        out[f"{name}/ids"] = ids[0].numpy().astype(np.int32)
        out[f"{name}/argmax"] = order[:, 0].astype(np.int32)
        out[f"{name}/top8_ids"] = order.astype(np.int32)
        out[f"{name}/top8_vals"] = np.take_along_axis(logits, order,
                                                      axis=-1).astype(
                                                          np.float32)
        rows = sorted(set(list(range(0, T, _FULL_ROW_STRIDE)) + [T - 1]))
        out[f"{name}/full_rows_idx"] = np.asarray(rows, dtype=np.int32)
        out[f"{name}/full_rows"] = logits[rows].astype(np.float32)

        if want_layers:
            hs = res.hidden_states  # len = L + 1
            out[f"{name}/hs_embed"] = hs[0][0].to(torch.float32).numpy()
            stack = np.stack(
                [h[0].to(torch.float32).numpy() for h in hs[1:-1]], axis=0)
            out[f"{name}/hs_layers"] = stack  # [L-1, T, D] = layers 0..L-2
            out[f"{name}/hs_final"] = (
                res.hidden_states[-1][0].to(torch.float32).numpy())
        del res, logits

        # Greedy reference for the TPU comparison.
        t0 = time.time()
        gen = model.generate(input_ids=ids,
                             do_sample=False,
                             num_beams=1,
                             max_new_tokens=spec["max_new_tokens"],
                             use_cache=True,
                             pad_token_id=tok.pad_token_id
                             or tok.eos_token_id)
        new_ids = gen[0, T:].numpy().astype(np.int32)
        out[f"{name}/greedy_ids"] = new_ids
        txt = tok.decode(new_ids, skip_special_tokens=False)
        print(f"  greedy {len(new_ids)} tok in {time.time() - t0:.1f}s",
              flush=True)
        print(f"  greedy text: {txt!r}", flush=True)
        summary[name] = {
            "n_prompt_tokens": int(T),
            "greedy_ids": new_ids.tolist(),
            "greedy_text": txt,
            "prompt_text": spec["text"] if T < 200 else spec["text"][:200] +
            f"... [{T} tokens]",
            "layer_dump": bool(want_layers),
        }
        del gen

    meta["prompts"] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out} and {Path(args.out).with_suffix('.json')}",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# JAX side
# ---------------------------------------------------------------------------


class _LazySafetensors:
    """``.items()`` over the shards, one tensor at a time.

    ``core.load_params`` only ever calls ``.items()``, so streaming keeps peak
    RSS at (one tensor) + (the float32 params it decides to keep) instead of
    (whole bf16 checkpoint) + (float32 params).
    """

    def __init__(self, model_dir: Path):
        import safetensors.torch  # noqa: F401
        self.files = sorted(model_dir.glob("*.safetensors"))
        if not self.files:
            raise SystemExit(f"no safetensors under {model_dir}")

    def items(self):
        from safetensors import safe_open
        for f in self.files:
            with safe_open(str(f), framework="pt") as handle:
                for key in handle.keys():
                    yield key, handle.get_tensor(key)


def run_jax(args) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax
    import jax.numpy as jnp
    from transformers import AutoConfig

    core = _load_core()
    print(f"jax {jax.__version__} devices={jax.devices()}", flush=True)

    ref = np.load(args.ref, allow_pickle=False)
    hf_meta = json.loads(Path(args.ref).with_suffix(".json").read_text())

    cfg_full = AutoConfig.from_pretrained(args.model_dir)
    text_cfg = cfg_full.get_text_config()
    params = core.params_from_hf_config(text_cfg)
    print(f"params: L={params.num_hidden_layers} D={params.hidden_size} "
          f"window={params.sliding_window} "
          f"types={params.layer_types[:4]}...", flush=True)

    t0 = time.time()
    weights = core.load_params(_LazySafetensors(Path(args.model_dir)), params,
                              dtype=jnp.float32)
    print(f"loaded {sum(1 for _ in weights['layers'])} layers of weights in "
          f"{time.time() - t0:.1f}s", flush=True)

    results: Dict[str, Any] = {}
    failures: List[str] = []

    for spec in build_prompts():
        name = spec["name"]
        ids = np.asarray(ref[f"{name}/ids"])[None, :]
        T = ids.shape[1]
        print(f"\n=== {name}: {T} tokens ===", flush=True)
        jids = jnp.asarray(ids)

        t0 = time.time()
        want_layers = f"{name}/hs_layers" in ref.files
        if want_layers:
            # Replicate core.forward_hidden but keeping every layer output, so
            # a mis-mapped tensor is localised to a layer index.
            mask_index = jnp.broadcast_to(jnp.arange(T), (1, T))
            h = core.embed(weights, jids, params)
            per_layer = []
            masks = core.build_masks(mask_index, params, None)
            for i, layer in enumerate(weights["layers"]):
                h = core.decoder_layer(layer, h, mask_index,
                                       masks[params.layer_types[i]], i, params)
                per_layer.append(np.asarray(h[0]))
            hidden = core.rms_norm_scaled(h, weights["norm"],
                                          params.rms_norm_eps)
            emb_out = np.asarray(core.embed(weights, jids, params)[0])
        else:
            hidden = core.forward_hidden(weights, jids, params)
            per_layer = None
            emb_out = None
        logits = np.asarray(core.compute_logits(weights, hidden, params)[0])
        hidden_np = np.asarray(hidden[0])
        print(f"  forward {time.time() - t0:.1f}s", flush=True)

        entry: Dict[str, Any] = {"n_tokens": int(T)}

        if per_layer is not None:
            entry["embed"] = diff_stats(ref[f"{name}/hs_embed"], emb_out)
            hs_layers = ref[f"{name}/hs_layers"]  # [L-1, T, D]
            worst = None
            for i in range(hs_layers.shape[0]):
                st = diff_stats(hs_layers[i], per_layer[i])
                if worst is None or st["max_abs"] > worst[1]["max_abs"]:
                    worst = (i, st)
            entry["per_layer_worst"] = {"layer": int(worst[0]), **worst[1]}
            entry["per_layer_last"] = diff_stats(hs_layers[-1],
                                                 per_layer[hs_layers.shape[0] -
                                                           1])
            entry["hidden_final"] = diff_stats(ref[f"{name}/hs_final"],
                                               hidden_np)

        rows = np.asarray(ref[f"{name}/full_rows_idx"])
        entry["logits_sampled_rows"] = diff_stats(ref[f"{name}/full_rows"],
                                                  logits[rows])
        entry["logits_sampled_rows"]["rows"] = rows.tolist()

        # top-8 agreement over EVERY position (this is what greedy decoding
        # actually depends on).
        top8_ids = np.asarray(ref[f"{name}/top8_ids"])
        top8_vals = np.asarray(ref[f"{name}/top8_vals"])
        got_order = np.argsort(-logits, axis=-1)[:, :8]
        got_vals = np.take_along_axis(logits, got_order, axis=-1)
        argmax_match = int((got_order[:, 0] == top8_ids[:, 0]).sum())
        entry["argmax_match"] = f"{argmax_match}/{T}"
        entry["top8_set_match"] = int(
            sum(
                set(got_order[i].tolist()) == set(top8_ids[i].tolist())
                for i in range(T)))
        entry["top8_vals"] = diff_stats(top8_vals, got_vals)
        if argmax_match != T:
            bad = np.nonzero(got_order[:, 0] != top8_ids[:, 0])[0]
            gaps = [
                float(top8_vals[i, 0] - top8_vals[i, 1]) for i in bad[:10]
            ]
            entry["argmax_mismatch_positions"] = bad[:10].tolist()
            entry["argmax_mismatch_ref_top1_top2_gap"] = gaps
            failures.append(
                f"{name}: {T - argmax_match}/{T} teacher-forced argmax "
                f"mismatches (ref top1-top2 gaps {gaps})")

        thr = args.tol_abs
        if entry["logits_sampled_rows"]["max_abs"] > thr:
            failures.append(f"{name}: logits max_abs "
                            f"{entry['logits_sampled_rows']['max_abs']:.3e} > "
                            f"{thr:.1e}")
        results[name] = entry
        print(json.dumps(entry, indent=2, default=float), flush=True)

    out = {
        "hf_meta": hf_meta,
        "results": results,
        "failures": failures,
        "jax": jax.__version__,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}", flush=True)

    print("\n================ SUMMARY ================")
    for name, e in results.items():
        print(f"{name:20s} n={e['n_tokens']:5d} "
              f"logits max_abs={e['logits_sampled_rows']['max_abs']:.3e} "
              f"rel(>1%peak)={e['logits_sampled_rows']['max_rel_above_1pct_peak']:.3e} "
              f"argmax={e['argmax_match']}")
        if "per_layer_worst" in e:
            w = e["per_layer_worst"]
            print(f"{'':20s} worst layer {w['layer']:3d} "
                  f"max_abs={w['max_abs']:.3e}  "
                  f"embed max_abs={e['embed']['max_abs']:.3e}  "
                  f"final hidden max_abs={e['hidden_final']['max_abs']:.3e}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("hf", "jax"), required=True)
    ap.add_argument("--model-dir",
                    default="/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="hf .npz dump (jax side)")
    ap.add_argument("--layer-dump-max-len", type=int, default=64)
    ap.add_argument("--tol-abs", type=float, default=2e-3)
    args = ap.parse_args()
    if args.side == "hf":
        return run_hf(args)
    if not args.ref:
        raise SystemExit("--ref is required for --side jax")
    return run_jax(args)


if __name__ == "__main__":
    raise SystemExit(main())
