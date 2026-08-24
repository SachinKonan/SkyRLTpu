"""Filter stored teacher outputs into a corpus — OFFLINE, FREE, repeatable. Apply
the grounding + leak gates (each toggleable), build the CE datums, and inspect
the dropped critiques (with the exact leak-offending identifiers). Re-run with
--no-leak-filter to see what the leak gate actually costs us — no regeneration.
"""

from __future__ import annotations

import argparse
import re

import common


class _B:  # minimal better with a .code attr for the leak gate
    def __init__(self, code): self.code = code


def build_datum(prompt_ids, target_tokens):
    import tinker, torch
    from tinker import TensorData
    from ttt_discover.rl.data_processing import (
        create_rightshifted_model_input_and_leftshifted_targets)
    prompt_mi = tinker.ModelInput.from_ints(prompt_ids)
    if prompt_mi.length + len(target_tokens) + 8 > common.CONTEXT_WINDOW:
        return None, None
    chunks = list(prompt_mi.chunks) + [tinker.types.EncodedTextChunk(tokens=target_tokens)]
    model_input, targets = create_rightshifted_model_input_and_leftshifted_targets(chunks)
    weights = ([0.0] * prompt_mi.length + [1.0] * len(target_tokens))[1:]
    datum = tinker.Datum(model_input=model_input, loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(targets, dtype=torch.int64)),
        "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32))})
    return datum, torch.tensor(weights, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-outputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leak-filter", dest="leak", action="store_true", default=True)
    ap.add_argument("--no-leak-filter", dest="leak", action="store_false")
    ap.add_argument("--grounding-filter", dest="grounding", action="store_true", default=True)
    ap.add_argument("--no-grounding-filter", dest="grounding", action="store_false")
    ap.add_argument("--max-target-tokens", type=int, default=8192)
    ap.add_argument("--show-dropped", type=int, default=0)
    args = ap.parse_args()

    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.rl import context_distill as cd
    tok = get_tokenizer(common.STUDENT_MODEL)

    data = common.read_json(args.teacher_outputs)
    cats = {"survived": 0, "grounding_dropped": 0, "leak_dropped": 0,
            "gen_failed": 0, "too_long": 0, "oov": 0}
    records, weight_masks, dropped = [], [], []

    for o in data["outputs"]:
        if o["status"] != "ok" or not o["teacher_tokens"]:
            cats["gen_failed"] += 1; continue
        final = tok.decode(o["teacher_tokens"]).split(cd.FINAL_SPLIT)[-1]
        if args.grounding:
            ok, _ = cd._grounding_ok(final, o["worse_code"])
            if not ok:
                cats["grounding_dropped"] += 1; continue
        betters = [_B(b["code"]) for b in o["betters"]]
        offenders = cd._leak_offenders(final, o["worse_code"], betters)
        if args.leak and offenders:
            cats["leak_dropped"] += 1
            if len(dropped) < args.show_dropped:
                whats = re.findall(r"<what>(.*?)</what>", final, re.S)
                dropped.append((sorted(offenders), whats))
            continue
        target = o["teacher_tokens"][: args.max_target_tokens]
        datum, mask = build_datum(o["prompt_ids"], target)
        if datum is None:
            cats["too_long"] += 1; continue
        rec = common.datum_to_record(datum, mask)
        if common.record_max_token_id(rec) > common.MAX_TRAIN_TOKEN_ID:
            cats["oov"] += 1; continue
        records.append(rec); weight_masks.append(mask)
        cats["survived"] += 1

    # probe = first 8 survivors
    probe = records[:8]
    common.write_json(args.out, {
        "name": data["name"], "teacher_model": data["teacher_model"],
        "better_origin": data.get("better_origin"),
        "leak_filter": args.leak, "grounding_filter": args.grounding,
        "n_datums": len(records), "cats": cats,
        "records": records, "probe": probe})

    print(f"[offline:{data['name']}] leak_filter={args.leak} grounding={args.grounding}")
    print(f"  {cats}")
    print(f"  -> {len(records)} datums -> {args.out}")
    for i, (off, whats) in enumerate(dropped):
        print(f"\n--- LEAK-DROPPED {i}: offenders={off} ---")
        for w in whats[:4]:
            print(f"  WHAT: {' '.join(w.split())[:240]}")


if __name__ == "__main__":
    main()
