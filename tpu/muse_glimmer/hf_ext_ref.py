#!/usr/bin/env python3
"""Extra HF references: decode across the sliding window, and long context.

Two things `hf_ref.npz` cannot answer:

1. **Decode across the 2048 boundary.**  The prefill-only probes in
   `make_ext_ref.py` come free from the existing teacher-forced argmax, but
   they only exercise prefill.  A KV-retention bug could instead bite while
   *decoding* past the window, when the block table grows.  So: greedy
   continuations from prefixes that start just below 2048 and cross it.

2. **Context past 4096.**  The served run was capped at `max_model_len=4096`.
   These prompts are built to a target token count exactly, carry a needle in
   the first ~30 tokens and ask for it at the end, so the 13 full-attention
   (NoPE) layers -- the only ones that can see that far -- carry real load.
   For each we dump the teacher-forced argmax at EVERY position, which lets
   the TPU side check agreement at any prefix length with one request each.

Everything is written incrementally: the TPU run may start consuming the
short-prefix results while the 16k forward is still going.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

NEEDLE = ("ARCHIVE HEADER, read once and remember it. The archive key is "
          "TANGERINE-7741 and the custodian of record is Rasmussen.\n\n")
QUESTION = ("\n\nQuestion: repeat the archive key and name the custodian of "
            "record, both stated in the ARCHIVE HEADER at the very top of "
            "this archive.\n\nAnswer: The archive key is")

_FILLER = [
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


def flush(out: dict, path: str) -> None:
    Path(path).write_text(json.dumps(out))
    print(f"  [checkpointed -> {path}]", flush=True)


def topk_chunked(logits: torch.Tensor, k: int = 8, chunk: int = 1024):
    """topk over [T, V] in row chunks (a [16384, 202048] argsort is not free)."""
    ids, vals = [], []
    for s in range(0, logits.shape[0], chunk):
        v, i = torch.topk(logits[s:s + chunk].float(), k, dim=-1)
        ids.append(i.numpy().astype(np.int32))
        vals.append(v.numpy().astype(np.float32))
    return np.concatenate(ids), np.concatenate(vals)


def build_long_ids(tok, target: int) -> list:
    q_ids = tok(QUESTION, add_special_tokens=False).input_ids
    body = NEEDLE + " ".join(_FILLER[i % len(_FILLER)]
                             for i in range(target // 4))
    head = tok(body, add_special_tokens=True).input_ids
    keep = target - len(q_ids)
    assert len(head) >= keep, (len(head), keep)
    return list(head[:keep]) + list(q_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref-npz", required=True, help="hf_ref.npz (for p4 ids)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--decode-lens", default="2020,2040,2048")
    ap.add_argument("--decode-tokens", type=int, default=24)
    ap.add_argument("--long-lens", default="8192,16384")
    ap.add_argument("--long-tokens", default="16,8")
    args = ap.parse_args()

    # AutoModelForCausalLM does NOT accept MuseGlimmerConfig -- the checkpoint
    # advertises the multimodal ...ForConditionalGeneration class, which is the
    # same class real_weight_check.py used to produce hf_ref.npz.  Using the
    # same class keeps this reference commensurable with that one.
    from transformers import (AutoTokenizer,
                              MuseGlimmerForConditionalGeneration)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.time()
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=torch.float32,
        attn_implementation=args.attn,
        low_cpu_mem_usage=True)
    model.eval()
    print(f"model loaded in {time.time() - t0:.0f}s  attn={args.attn}",
          flush=True)

    z = np.load(args.ref_npz, allow_pickle=False)
    p4 = z["p4_long_sliding/ids"].astype(np.int64)
    p4_argmax = z["p4_long_sliding/argmax"].astype(np.int64)

    out = {"attn_implementation": args.attn, "decode": {}, "long": {}}

    # --- phase 0: the chosen attention impl must agree with the validated ---
    # eager reference on a stretch of p4 that already has a known answer.
    with torch.no_grad():
        probe = model(input_ids=torch.tensor(p4[:512])[None]).logits[0]
    got = probe.argmax(-1).numpy()
    agree = int((got == p4_argmax[:512]).sum())
    out["attn_sanity"] = {
        "positions": 512,
        "argmax_agree_with_eager_ref": agree,
        "ok": agree >= 511,
    }
    print(f"phase 0: {args.attn} vs the eager reference over the first 512 "
          f"p4 positions: {agree}/512 argmax agree", flush=True)
    del probe, got
    flush(out, args.out)

    # ------------------------------------------------- phase 1: decode -----
    for L in [int(x) for x in args.decode_lens.split(",") if x]:
        t0 = time.time()
        ids = torch.tensor(p4[:L])[None]
        with torch.no_grad():
            gen = model.generate(input_ids=ids,
                                 do_sample=False,
                                 num_beams=1,
                                 max_new_tokens=args.decode_tokens,
                                 use_cache=True,
                                 pad_token_id=tok.pad_token_id
                                 or tok.eos_token_id)
        new = gen[0, L:].numpy().astype(int).tolist()
        out["decode"][str(L)] = {
            "prefix_len": L,
            "greedy_ids": new,
            "greedy_text": tok.decode(new, skip_special_tokens=False),
            "crosses_window": L + len(new) > 2048,
            "seconds": time.time() - t0,
        }
        print(f"phase 1: prefix {L} -> {len(new)} greedy tok in "
              f"{time.time() - t0:.0f}s  {out['decode'][str(L)]['greedy_text']!r}",
              flush=True)
        del gen
        flush(out, args.out)

    # --------------------------------------------- phase 2/3: long ctx -----
    lens = [int(x) for x in args.long_lens.split(",") if x]
    ntok = [int(x) for x in args.long_tokens.split(",") if x]
    for L, n in zip(lens, ntok):
        key = str(L)
        try:
            ids_list = build_long_ids(tok, L)
            ids = torch.tensor(ids_list)[None]
            t0 = time.time()
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0]
            fwd = time.time() - t0
            top_ids, top_vals = topk_chunked(logits, 8)
            final_row = logits[-1].float().numpy().astype(np.float32)
            del logits
            print(f"phase 2: {L} tokens forward in {fwd:.0f}s", flush=True)

            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(input_ids=ids,
                                     do_sample=False,
                                     num_beams=1,
                                     max_new_tokens=n,
                                     use_cache=True,
                                     pad_token_id=tok.pad_token_id
                                     or tok.eos_token_id)
            new = gen[0, L:].numpy().astype(int).tolist()
            txt = tok.decode(new, skip_special_tokens=False)
            del gen
            out["long"][key] = {
                "n_tokens": L,
                "ids": ids_list,
                "argmax": top_ids[:, 0].astype(int).tolist(),
                "top1_top2_gap": (top_vals[:, 0] -
                                  top_vals[:, 1]).astype(float).tolist(),
                "greedy_ids": new,
                "greedy_text": txt,
                "needle_hit": ("TANGERINE" in txt or "7741" in txt),
                "forward_seconds": fwd,
                "generate_seconds": time.time() - t0,
            }
            np.save(Path(args.out).with_suffix(f".final_logits_{L}.npy"),
                    final_row)
            print(f"phase 2: {L} greedy {n} tok -> {txt!r} "
                  f"needle_hit={out['long'][key]['needle_hit']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            out["long"][key] = {"n_tokens": L, "error": repr(exc)}
            print(f"phase 2: {L} FAILED: {exc!r}", flush=True)
        flush(out, args.out)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
