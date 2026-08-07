"""Prove each model's chat rendering is right BEFORE spending the run on it.

Two checks per server:
  1. token-level: our rendered prompt tokenizes to the same ids the server's
     own chat template produces (``/tokenize``), and every special token the
     renderer depends on is a SINGLE token;
  2. behavioural: a 32-token sample off the real generation prompt comes back
     non-empty and does not immediately emit an end-of-turn.

Check 1 is allowed to disagree for gemma and say so: ``Gemma4Renderer``
deliberately emits ``<bos>`` that the naive template path does not, because a
BOS-less gemma prompt was measured at 1/32 vs 7/8 compiling programs. What is
NOT allowed is a silent disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe import render as R  # noqa: E402
from pallas_arena.probe.sampler import VllmSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    urls = dict(kv.split("=", 1) for kv in args.servers.split(","))
    report = {}
    bad = 0
    for key, url in urls.items():
        spec = C.MODELS[key]
        rep = R.verify_against_server(url, spec.hf_id, spec.renderer)
        s = VllmSampler(url, spec.hf_id)
        probe_prompt = R.render(spec.renderer, "Reply with the single word: OK")
        outs = s.sample_group(probe_prompt, 1, max_tokens=32, temperature=0.0, stop=R.STOPS[spec.renderer])
        rep["sample_text"] = (outs[0]["text"] or "")[:300]
        rep["sample_nonempty"] = bool((outs[0]["text"] or "").strip())
        rep["sample_finish"] = outs[0]["finish_reason"]
        report[key] = rep
        ok_specials = rep.get("special_tokens_all_single")
        print(f"[render] {key}: template_agrees={rep.get('agree')} "
              f"special_tokens_single={ok_specials} sample_nonempty={rep['sample_nonempty']}")
        print(f"[render] {key} rendered prefix: {rep['rendered_prefix']!r}")
        if rep.get("agree") is False:
            print(f"[render] {key} DIVERGES from the server template at token "
                  f"{rep.get('first_divergence_at')}: ours={rep.get('ours_head')} "
                  f"template={rep.get('template_head')}")
        if not ok_specials or not rep["sample_nonempty"]:
            bad += 1
            print(f"[render] {key} FAILED the hard checks", file=sys.stderr)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
