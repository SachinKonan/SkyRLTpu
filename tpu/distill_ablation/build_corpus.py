"""Build a distillation corpus offline from a pool snapshot.

Selection is free (no Tinker); the teacher pass (writing the <improve> critiques)
is the only paid part. Held-out seed ids are removed from the pool BEFORE
selection, so no corpus datum trains on a held-out seed.

Modes:
  self  -> select_distill_pairs         (worse+betters from one run's pool)
  cross -> select_cross_distill_pairs    (worse=self-tag, betters=other member)

Teacher model is an axis: --teacher-model openai/gpt-oss-20b (default) or
openai/gpt-oss-120b (A5). The student tokenizer (20b) is used either way.

Usage (offline dry run, free):
  python build_corpus.py --snapshot ... --mode self --max-pairs 8 --dry-run

Usage (paid, build the self superset):
  python build_corpus.py --snapshot .../ctrl15/.../puct_sampler_step_000015.json \
    --mode self --max-pairs 300 --out corpora/self_superset.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import random
import ssl
import time

import common


def select_pairs(sampler, builder, mode, self_tag, max_pairs, num_betters, max_code_chars, rng):
    from ttt_discover.rl.context_distill import select_distill_pairs, select_cross_distill_pairs
    if mode == "self":
        return select_distill_pairs(sampler, [builder], max_pairs, num_betters, max_code_chars, rng)
    return select_cross_distill_pairs(sampler, [builder], self_tag, max_pairs, num_betters, max_code_chars, rng)


async def build(args):
    from ttt_discover.rl.context_distill import build_distill_data

    heldout_ids = set(common.read_json(args.heldout)["ids"]) if args.heldout else set()
    snap = sorted(glob.glob(args.snapshot))[-1]
    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_scratch/{args.name}"
    sampler = common.load_pool_sampler(snap, scratch)
    # Remove held-out states BEFORE selection: they can be neither worse nor better.
    with sampler._lock:
        n0 = len(sampler._states)
        sampler._states = [s for s in sampler._states if s.id not in heldout_ids]
    print(f"[corpus:{args.name}] pool {n0} -> {len(sampler._states)} after held-out removal")

    # Cross-MODEL: tag the gpt-oss pool as origin='gptoss' and merge in the foreign
    # betters (origin nemo/qwen); select_cross_distill_pairs then pairs a gpt-oss
    # 'worse' with a FOREIGN 'better'. --better-origin restricts to one model.
    if args.foreign_betters:
        from ttt_discover.tinker_utils.state import state_from_dict
        from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
        fb = common.read_json(args.foreign_betters)["betters"]
        if args.better_origin:
            fb = [b for b in fb if b["origin"] == args.better_origin]
        with sampler._lock:
            for s in sampler._states:
                s.origin = "gptoss"
            for b in fb:
                sampler._states.append(
                    state_from_dict(b, state_type=ErdosMinOverlapEnv.state_type))
        args.mode = "cross"
        args.self_tag = "gptoss"
        print(f"[corpus:{args.name}] merged {len(fb)} foreign betters "
              f"(origin={args.better_origin or 'all'}); worse=gptoss, betters=foreign")

    renderer, tokenizer, cfg = common.env_bits(scratch)
    builder = common.make_seed_builder(sampler, renderer, cfg, sampler._states[0], num_envs=1)
    rng = random.Random(args.seed)

    pairs, sel_metrics = select_pairs(
        sampler, builder, args.mode, args.self_tag,
        args.max_pairs, args.num_betters, args.max_code_chars, rng,
    )
    print(f"[corpus:{args.name}] selected {len(pairs)} pairs (mode={args.mode})")
    if not pairs:
        raise SystemExit("no pairs selected")

    if args.dry_run:
        summary = []
        for p in pairs[:5]:
            summary.append({
                "worse_id": p.worse.id, "worse_score": p.worse.value,
                "n_betters": len(p.betters),
                "better_ids": [b.id for b in p.betters],
                "better_origins": [getattr(b, "origin", None) for b in p.betters],
                "prompt_len_tokens": p.prompt_model_input.length,
                "teacher_prompt_chars": len(p.teacher_prompt),
            })
        common.write_json(args.out, {"dry_run": True, "n_pairs": len(pairs),
                                     "select_metrics": sel_metrics, "sample": summary})
        print(f"[dry-run] wrote {args.out}; first pair worse_id={pairs[0].worse.id}, "
              f"betters origins={summary[0]['better_origins'] if summary else None}")
        return

    # --- paid: teacher writes critiques (chunked to bound concurrency) ---
    common.load_dotenv_key()
    import tinker
    svc = tinker.ServiceClient()
    teacher = svc.create_sampling_client(base_model=args.teacher_model)
    print(f"[corpus:{args.name}] teacher = {args.teacher_model}")

    records, probe_holder = [], []
    agg = {}
    t0 = time.time()
    for i in range(0, len(pairs), args.chunk):
        chunk = pairs[i:i + args.chunk]
        batch = await build_distill_data(
            chunk, teacher, common.STUDENT_MODEL, renderer,
            distill_weight=1.0, teacher_phase1_tokens=args.teacher_phase1_tokens,
            context_window=common.CONTEXT_WINDOW, temperature=1.0,
            max_target_tokens=args.max_target_tokens,
            grounding_filter=True, leak_filter=True,
            probe_holder=probe_holder if not records else None, probe_size=8,
        )
        for d, w01 in zip(batch.datums, batch.weight_tensors):
            rec = common.datum_to_record(d, w01)
            if common.record_max_token_id(rec) > common.MAX_TRAIN_TOKEN_ID:
                agg["oov_dropped"] = agg.get("oov_dropped", 0) + 1
                continue
            records.append(rec)
        for k, v in batch.metrics.items():
            agg[k] = agg.get(k, 0.0) + v
        print(f"[corpus:{args.name}] chunk {i//args.chunk}: +{len(batch.datums)} datums "
              f"(total {len(records)}), {time.time()-t0:.0f}s")

    if not records:
        raise SystemExit(f"[corpus:{args.name}] 0 datums survived gates; metrics={agg}. "
                         "Check teacher-phase1-tokens (prompt must fit budget-2048).")

    # Inspect: decode the surviving critiques so we can judge whether the teacher
    # captured the (foreign) better's technique.
    if args.show_critiques:
        import re as _re
        FINAL = "<|channel|>final<|message|>"
        _IMP = _re.compile(r"<improve>(.*?)</improve>", _re.S)
        print(f"\n===== SAMPLE CRITIQUES ({min(args.show_critiques, len(records))}) =====")
        for i, rec in enumerate(records[: args.show_critiques]):
            txt = tokenizer.decode(rec["model_input_ids"]).split(FINAL)[-1]
            blocks = _IMP.findall(txt)
            print(f"\n--- critique {i}: {len(blocks)} <improve> blocks ---")
            for b in blocks:
                what = _re.search(r"<what>(.*?)</what>", b, _re.S)
                cite = _re.search(r"<cite>(.*?)</cite>", b, _re.S)
                print(f"  CITE: {' '.join((cite.group(1) if cite else '').split())[:80]}")
                print(f"  WHAT: {' '.join((what.group(1) if what else '').split())[:280]}")

    probe = [common.datum_to_record(mi_pack, mask) for mi_pack, mask in
             _probe_records(probe_holder)]
    common.write_json(args.out, {
        "name": args.name, "mode": args.mode, "teacher_model": args.teacher_model,
        "snapshot": snap, "n_pairs": len(pairs), "n_datums": len(records),
        "metrics": agg, "select_metrics": sel_metrics,
        "records": records, "probe": probe,
    })
    print(f"[corpus:{args.name}] DONE {len(records)} datums -> {args.out}  metrics={agg}")


def _probe_records(probe_holder):
    """probe_holder items are (model_input, targets_tensor, mask01_tensor);
    re-wrap as (datum-like, mask) for datum_to_record."""
    import tinker
    from tinker import TensorData

    class _D:
        def __init__(self, mi, targets):
            self.model_input = mi
            self.loss_fn_inputs = {"target_tokens": TensorData.from_torch(targets)}
    for mi, targets, mask in probe_holder:
        yield _D(mi, targets), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--mode", choices=["self", "cross"], default="self")
    ap.add_argument("--name", required=True, help="corpus name (scratch + logging)")
    ap.add_argument("--teacher-model", default=common.STUDENT_MODEL)
    ap.add_argument("--self-tag", default="alpha", help="cross mode: this member's origin tag")
    ap.add_argument("--max-pairs", type=int, default=300, help="over-generate; ~46% survive gates")
    ap.add_argument("--num-betters", type=int, default=3)
    ap.add_argument("--max-code-chars", type=int, default=10000)
    ap.add_argument("--teacher-phase1-tokens", type=int, default=26000)
    ap.add_argument("--max-target-tokens", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--foreign-betters", help="foreign_betters.json: merge as cross betters")
    ap.add_argument("--better-origin", help="restrict foreign betters to one model (qwen/nemo)")
    ap.add_argument("--show-critiques", type=int, default=0, help="decode+print N critiques")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="select pairs only, no Tinker")
    args = ap.parse_args()

    if not args.dry_run:
        ssl.create_default_context()
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
