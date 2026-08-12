#!/usr/bin/env python3
"""Follow-up TPU checks: window boundary, long context, temperature.

Runs inside the vLLM venv on the TPU VM.  Prompts go in as explicit
``list[int]`` and ids come back via ``return_token_ids``, so nothing
round-trips through text (same discipline as `mg_client.py`).

Phases, selectable with --phases:

  boundary  one-token greedy from prefixes of the 3609-token p4 prompt,
            densely across the 2048 sliding window.  HF's teacher-forced
            argmax at position L-1 IS the greedy next token for the prefix of
            length L, so the reference is free and exact.  Reports the
            SHORTEST prefix length at which the served model disagrees, and
            classifies each disagreement as an argmax tie or a real one using
            both sides' top1/top2 gaps.
  decode    multi-token greedy from prefixes that CROSS the window while
            decoding (prefill-only probes cannot see a bug that bites when
            the block table grows).
  longctx   teacher-forced prefix probes at long positions + a full-length
            greedy continuation with a needle planted in the first ~30
            tokens, which only the 13 full-attention (NoPE) layers can carry.
  temp      temperature > 0: seed determinism, seed sensitivity,
            degeneration, and -- the load-bearing one -- the returned
            logprobs against HF's own distribution at the same temperature.
            Greedy is invariant to any monotone rescaling of the logits, so
            `output_multiplier` and the tanh softcap are exactly the kind of
            bug greedy structurally cannot see.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np


def post(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(url,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Engine:

    def __init__(self, base: str, model: str):
        self.base = base.rstrip("/")
        self.model = model

    def complete(self,
                 prompt_ids: List[int],
                 max_tokens: int,
                 temperature: float = 0.0,
                 logprobs: Optional[int] = None,
                 seed: Optional[int] = 0,
                 top_p: float = 1.0,
                 timeout: int = 1800) -> dict:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "ignore_eos": True,
        }
        if seed is not None:
            payload["seed"] = seed
        if logprobs is not None:
            payload["logprobs"] = logprobs
        t0 = time.time()
        out = post(f"{self.base}/v1/completions", payload, timeout=timeout)
        out["_elapsed"] = time.time() - t0
        return out


def gen_ids(resp: dict) -> List[int]:
    ch = resp["choices"][0]
    if ch.get("token_ids"):
        return list(ch["token_ids"])
    raise RuntimeError(f"no token_ids in response; keys={sorted(ch)}")


def tops(resp: dict, step: int = 0) -> List[tuple]:
    lp = resp["choices"][0].get("logprobs") or {}
    t = lp.get("top_logprobs") or []
    if step >= len(t) or not t[step]:
        return []
    items = []
    for k, v in t[step].items():
        tid = int(str(k).split(":")[-1]) if ":" in str(k) else None
        items.append((tid, float(v)))
    items.sort(key=lambda x: -x[1])
    return items


def first_div(a: List[int], b: List[int]) -> Optional[int]:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# --------------------------------------------------------------- boundary ---
def phase_boundary(eng: Engine, ref: dict, res: dict, max_len: int) -> None:
    print("\n=== boundary: 1-token greedy across the 2048 window ===",
          flush=True)
    ids = ref["p4_ids"]
    rows = []
    shortest = None
    for p in ref["probes"]:
        L = p["prefix_len"]
        if L > max_len - 4:
            continue
        r = eng.complete(ids[:L], 1, 0.0, logprobs=8)
        got = gen_ids(r)[0]
        tp = tops(r)
        gap = (tp[0][1] - tp[1][1]) if len(tp) > 1 else None
        agree = got == p["hf_next_id"]
        rank = next((i for i, (t, _) in enumerate(tp) if t == p["hf_next_id"]),
                    None)
        row = {
            "prefix_len": L,
            "hf_next_id": p["hf_next_id"],
            "tpu_next_id": got,
            "agree": agree,
            "hf_top1_top2_gap": p["hf_top1_top2_gap"],
            "tpu_top1_top2_gap": gap,
            "hf_token_rank_on_tpu": rank,
            "past_window": L > 2048,
        }
        if not agree and shortest is None:
            shortest = L
        rows.append(row)
        mark = "ok " if agree else "DIFF"
        print(f"  L={L:5d} {mark} hf={p['hf_next_id']:6d} tpu={got:6d} "
              f"hf_gap={p['hf_top1_top2_gap']:.4f} "
              f"tpu_gap={'n/a' if gap is None else f'{gap:.4f}'} "
              f"rank_of_hf_on_tpu={rank}",
              flush=True)
    n_ok = sum(r["agree"] for r in rows)
    res["boundary"] = {
        "n_probes": len(rows),
        "n_agree": n_ok,
        "shortest_diverging_prefix_len": shortest,
        "probed_lengths": [r["prefix_len"] for r in rows],
        "probes": rows,
    }
    print(f"  {n_ok}/{len(rows)} probes agree with HF; shortest divergence: "
          f"{shortest}", flush=True)


# ----------------------------------------------------------------- decode ---
def phase_decode(eng: Engine, ref: dict, ext: dict, res: dict,
                 max_len: int) -> None:
    print("\n=== decode: greedy continuations that CROSS the window ===",
          flush=True)
    ids = ref["p4_ids"]
    out = {}
    for key, entry in sorted(ext.get("decode", {}).items(), key=lambda kv: int(kv[0])):
        L = entry["prefix_len"]
        want = entry["greedy_ids"]
        if L + len(want) > max_len - 4:
            continue
        r = eng.complete(ids[:L], len(want), 0.0)
        got = gen_ids(r)
        d = first_div(want, got)
        out[key] = {
            "prefix_len": L,
            "n_tokens": len(want),
            "crosses_window": entry.get("crosses_window"),
            "exact": d is None,
            "first_divergence": d,
        }
        print(f"  prefix {L:5d} +{len(want)} tok crosses={entry.get('crosses_window')} "
              f"-> {'EXACT' if d is None else f'diverges at {d}'}",
              flush=True)
    res["decode_crossing"] = out


# ---------------------------------------------------------------- longctx ---
def phase_longctx(eng: Engine, ext: dict, res: dict, max_len: int) -> None:
    print("\n=== long context ===", flush=True)
    out = {}
    for key, entry in sorted(ext.get("long", {}).items(),
                             key=lambda kv: int(kv[0])):
        if "error" in entry:
            continue
        L = entry["n_tokens"]
        if L + 32 > max_len:
            print(f"  {L}: skipped, exceeds max_model_len {max_len}",
                  flush=True)
            continue
        ids = entry["ids"]
        argmax = entry["argmax"]
        gaps = entry["top1_top2_gap"]
        # teacher-forced agreement at prefix lengths spread over the context
        probe_lens = sorted({
            256, 1024, 2047, 2048, 2049, 4095, 4096, 4097, 6144,
            L // 2, L - 1024, L - 1, L
        })
        probes = []
        for pl in probe_lens:
            if not (0 < pl <= L):
                continue
            r = eng.complete(ids[:pl], 1, 0.0, logprobs=4)
            got = gen_ids(r)[0]
            tp = tops(r)
            probes.append({
                "prefix_len": pl,
                "hf_next_id": argmax[pl - 1],
                "tpu_next_id": got,
                "agree": got == argmax[pl - 1],
                "hf_top1_top2_gap": gaps[pl - 1],
                "tpu_top1_top2_gap":
                (tp[0][1] - tp[1][1]) if len(tp) > 1 else None,
            })
            print(f"  {L}: prefix {pl:6d} {'ok ' if probes[-1]['agree'] else 'DIFF'} "
                  f"hf={argmax[pl-1]:6d} tpu={got:6d} "
                  f"hf_gap={gaps[pl-1]:.4f}", flush=True)
        want = entry["greedy_ids"]
        r = eng.complete(ids, len(want), 0.0, timeout=3600)
        got = gen_ids(r)
        d = first_div(want, got)
        text = r["choices"][0]["text"]
        out[key] = {
            "n_tokens": L,
            "teacher_forced_probes": probes,
            "n_probe_agree": sum(p["agree"] for p in probes),
            "n_probes": len(probes),
            "greedy_exact_vs_hf": d is None,
            "greedy_first_divergence": d,
            "tpu_text": text,
            "hf_text": entry["greedy_text"],
            "needle_hit_tpu": ("TANGERINE" in text or "7741" in text),
            "needle_hit_hf": entry.get("needle_hit"),
            "seconds": r["_elapsed"],
        }
        print(f"  {L}: greedy {'EXACT' if d is None else f'diverges at {d}'}; "
              f"needle tpu={out[key]['needle_hit_tpu']} "
              f"hf={out[key]['needle_hit_hf']}\n"
              f"       tpu text: {text!r}", flush=True)
    res["longctx"] = out


NEEDLE = ("ARCHIVE HEADER, read once and remember it. The archive key is "
          "TANGERINE-7741 and the custodian of record is Rasmussen.\n\n")
QUESTION = ("\n\nQuestion: repeat the archive key and name the custodian of "
            "record, both stated in the ARCHIVE HEADER at the very top of "
            "this archive.\n\nAnswer: The archive key is")


def phase_longfallback(eng: Engine, tokenizer_dir: str, lens: List[int],
                       res: dict, max_len: int) -> None:
    """Long context without an HF reference: needle recall + determinism.

    The HF side of a 16k float32 forward is expensive and may not be ready.
    This still answers the question that matters for the 13 NoPE
    full-attention layers -- can the model use information from the very
    start of a context far past the 2048 sliding window -- and whether the
    served path is self-consistent at that length.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    filler = (
        "The kettle in the observatory kitchen had a habit of whistling "
        "exactly when the seeing was best, which the graduate students took "
        "as a personal insult and the postdocs took as a schedule. "
        "Cartography in the delta is a seasonal profession: the channels "
        "braid and rebraid every spring, so a map printed in March is a "
        "historical document by August and a liability by October. ")
    q_ids = tok(QUESTION, add_special_tokens=False).input_ids
    out = {}
    print("\n=== long context (no HF ref): needle recall + determinism ===",
          flush=True)
    for L in lens:
        if L + 32 > max_len:
            continue
        body = NEEDLE + filler * (L // 40 + 8)
        head = tok(body, add_special_tokens=True).input_ids
        ids = list(head[:L - len(q_ids)]) + list(q_ids)
        assert len(ids) == L, (len(ids), L)
        r1 = eng.complete(ids, 24, 0.0, timeout=3600)
        r2 = eng.complete(ids, 24, 0.0, timeout=3600)
        t1, t2 = r1["choices"][0]["text"], r2["choices"][0]["text"]
        out[str(L)] = {
            "n_tokens": L,
            "needle_hit": ("TANGERINE" in t1 or "7741" in t1),
            "custodian_hit": ("Rasmussen" in t1),
            "deterministic": gen_ids(r1) == gen_ids(r2),
            "text": t1,
            "seconds": r1["_elapsed"],
        }
        o = out[str(L)]
        print(f"  {L:6d} tok: needle={o['needle_hit']} "
              f"custodian={o['custodian_hit']} determ={o['deterministic']} "
              f"({o['seconds']:.1f}s)\n         {t1!r}", flush=True)
    res["longctx_fallback"] = out


# ------------------------------------------------------------------- temp ---
def _logsoftmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def _degeneration(ids: List[int]) -> Dict[str, Any]:
    n = len(ids)
    d1 = len(set(ids)) / max(n, 1)
    bg = [(ids[i], ids[i + 1]) for i in range(n - 1)]
    d2 = len(set(bg)) / max(len(bg), 1)
    c = Counter(bg)
    top_bg = c.most_common(1)[0][1] if c else 0
    # longest run of a repeating k-gram back to back
    longest = 0
    for k in range(1, min(20, n // 2) + 1):
        run = 0
        for i in range(n - 2 * k + 1):
            if ids[i:i + k] == ids[i + k:i + 2 * k]:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
    return {
        "n": n,
        "distinct_1": d1,
        "distinct_2": d2,
        "max_bigram_count": top_bg,
        "longest_periodic_run": longest,
        "degenerate": (d1 < 0.25 or longest >= 8),
    }


def phase_temp(eng: Engine, rows_npz: str, res: dict, temps: List[float],
               gen_tokens: int) -> None:
    print("\n=== temperature > 0 ===", flush=True)
    z = np.load(rows_npz, allow_pickle=False)
    names = sorted({k.split("/")[0] for k in z.files})
    out: Dict[str, Any] = {"logprob_vs_hf": {}, "determinism": {},
                           "degeneration": {}}

    # ---- (d) logprobs vs HF's distribution, the check greedy cannot do ----
    for T in temps:
        per_prompt = {}
        for name in names:
            ids = z[f"{name}/ids"].astype(int).tolist()
            row = z[f"{name}/final_logits"].astype(np.float64)
            r = eng.complete(ids, 1, T, logprobs=16, seed=1234)
            tp = tops(r)
            if not tp:
                per_prompt[name] = {"error": "no logprobs returned"}
                continue
            hf_raw = _logsoftmax(row)
            hf_T = _logsoftmax(row / T)
            got_ids = [t for t, _ in tp if t is not None]
            got_lp = np.array([v for t, v in tp if t is not None])
            d_raw = float(np.abs(got_lp - hf_raw[got_ids]).max())
            d_T = float(np.abs(got_lp - hf_T[got_ids]).max())
            hf_order_raw = np.argsort(-row)[:len(got_ids)].tolist()
            set_match = len(set(got_ids) & set(hf_order_raw))
            # rank correlation of the returned set against HF ordering
            per_prompt[name] = {
                "n_returned": len(got_ids),
                "topk_set_overlap_with_hf": set_match,
                "top1_id_tpu": got_ids[0],
                "top1_id_hf": hf_order_raw[0],
                "max_abs_logprob_err_vs_hf_raw": d_raw,
                "max_abs_logprob_err_vs_hf_temperature_scaled": d_T,
                "matches": ("raw" if d_raw < d_T and d_raw < 5e-2 else
                            ("temperature_scaled" if d_T < 5e-2 else "NEITHER")),
                "hf_entropy_nats_raw":
                float(-(np.exp(hf_raw) * hf_raw).sum()),
                "hf_entropy_nats_T": float(-(np.exp(hf_T) * hf_T).sum()),
            }
            p = per_prompt[name]
            print(f"  T={T} {name:18s} overlap={set_match}/{len(got_ids)} "
                  f"err_vs_raw={d_raw:.3e} err_vs_T={d_T:.3e} "
                  f"-> {p['matches']}", flush=True)
        out["logprob_vs_hf"][str(T)] = per_prompt

    # ---- (a)/(b) seed determinism and seed sensitivity --------------------
    base_ids = z[f"{names[0]}/ids"].astype(int).tolist()
    for T in temps:
        same = [
            gen_ids(eng.complete(base_ids, 32, T, seed=4242)) for _ in range(3)
        ]
        diff = [
            gen_ids(eng.complete(base_ids, 32, T, seed=s))
            for s in (1, 2, 3, 4)
        ]
        # concurrent, same seed: a shared-RNG bug shows up here and not
        # sequentially
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [
                ex.submit(eng.complete, base_ids, 32, T, None, 4242)
                for _ in range(4)
            ]
            conc = [gen_ids(f.result()) for f in futs]
        out["determinism"][str(T)] = {
            "same_seed_repeats": len(same),
            "same_seed_all_identical": all(s == same[0] for s in same),
            "same_seed_concurrent_identical":
            all(c == same[0] for c in conc),
            "distinct_outputs_across_4_seeds":
            len({tuple(d) for d in diff}),
            "sample_ids": same[0][:12],
        }
        o = out["determinism"][str(T)]
        print(f"  T={T} same-seed identical={o['same_seed_all_identical']} "
              f"(concurrent {o['same_seed_concurrent_identical']}); "
              f"{o['distinct_outputs_across_4_seeds']}/4 distinct across seeds",
              flush=True)

    # ---- (c) degeneration -------------------------------------------------
    for T in temps:
        r = eng.complete(base_ids, gen_tokens, T, seed=7, timeout=3600)
        g = gen_ids(r)
        st = _degeneration(g)
        st["text_head"] = r["choices"][0]["text"][:400]
        out["degeneration"][str(T)] = st
        print(f"  T={T} {gen_tokens} tok: distinct1={st['distinct_1']:.3f} "
              f"distinct2={st['distinct_2']:.3f} "
              f"longest_periodic_run={st['longest_periodic_run']} "
              f"degenerate={st['degenerate']}", flush=True)

    res["temperature"] = out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-ref", default="")
    ap.add_argument("--hf-ext", default="")
    ap.add_argument("--logit-rows", default="")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="muse-glimmer-30b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--phases", default="boundary,decode,longctx,temp")
    ap.add_argument("--tokenizer", default="",
                    help="HF dir, for the reference-free long-context phase")
    ap.add_argument("--fallback-long", default="",
                    help="comma-separated token counts for that phase")
    ap.add_argument("--temps", default="0.7,1.0")
    ap.add_argument("--degen-tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    eng = Engine(f"http://127.0.0.1:{args.port}", args.model)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    res: Dict[str, Any] = {
        "label": args.label,
        "max_model_len": args.max_model_len,
        "phases_requested": phases,
    }

    ref = json.load(open(args.ext_ref)) if args.ext_ref else {}
    ext = json.load(open(args.hf_ext)) if args.hf_ext else {}

    # liveness before anything is measured
    probe = eng.complete(ref["p4_ids"][:8] if ref else [1, 2, 3], 4)
    print(f"liveness ok -> {gen_ids(probe)} ({probe['_elapsed']:.1f}s)",
          flush=True)
    res["liveness_ids"] = gen_ids(probe)

    try:
        if "boundary" in phases and ref:
            phase_boundary(eng, ref, res, args.max_model_len)
        if "decode" in phases and ref and ext:
            phase_decode(eng, ref, ext, res, args.max_model_len)
        if "longctx" in phases and ext.get("long"):
            phase_longctx(eng, ext, res, args.max_model_len)
        if "longfallback" in phases and args.tokenizer and args.fallback_long:
            phase_longfallback(
                eng, args.tokenizer,
                [int(x) for x in args.fallback_long.split(",") if x], res,
                args.max_model_len)
        if "temp" in phases and args.logit_rows:
            phase_temp(eng, args.logit_rows, res,
                       [float(t) for t in args.temps.split(",") if t],
                       args.degen_tokens)
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)
        print(f"PHASE FAILED: {exc!r}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
