"""rf3 prompt smoke: generate N completions per (task, variant) cell from a
served model and dump RAW text to JSONL.

Measures what the evolution run will actually see at its sampling
distribution (temperature 1.0): validity rate and correctness are graded
OFFLINE by grade_smoke.py on CPU -- this script only generates and saves.
Full think+program text is saved, never just metrics (the c5 lesson:
regenerating is expensive; in_context_improve.py once threw away 240
rollouts).

Truncation rate at the completion cap is itself a measurement: qwen's
thinking channel ate the whole budget in the cold probe (3/80 finished), and
group-size-32 planning needs the real finish_reason distribution.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.request


# Two-phase forcing cue (byte-identical to the proven rq2/client/loop.py
# pattern): appended to the truncated assistant message, continued with
# continue_final_message. Extraction-friendly: the program is whatever
# follows this cue inside the opened fence.
FORCE = ("\n\nI have thought about this enough. Here is my final, complete, "
         "self-contained program:\n\n```python\n")


def _post(server: str, body: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.load(r)


def _chat(server: str, model: str, prompt: str, max_tokens: int, temperature: float,
          timeout_s: float, ctx: int = 32768, answer_cap: int = 8192) -> dict:
    """Two-phase completion. Thinking stays ON (by direction: we sample the
    policy we would train), but it is CAPPED: phase 1 gets `max_tokens`; if
    it truncates, the assistant message is continued past a forcing cue with
    an answer budget sized from ACTUAL usage (rq2 loop.py measured fixed p2
    overflowing the context -> vLLM 400 -- sizing from usage is load-bearing).
    Measured motivation: 16k/18k/27k single-phase caps truncated 53-97% of
    completions -- thinking expands to fill ANY budget it is given."""
    base = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1.0,
    }
    d = _post(server, {**base, "max_tokens": max_tokens}, timeout_s)
    ch = d["choices"][0]
    if ch.get("finish_reason") != "length":
        return d
    text = ch["message"].get("content") or ""
    u = d.get("usage") or {}
    used = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or max_tokens)
    room = ctx - used - 256
    p2 = min(answer_cap, room)
    if p2 < 512:
        return d  # no room to force; the parser gets the phase-1 text
    d2 = _post(server, {
        **base, "max_tokens": p2,
        "messages": base["messages"] + [{"role": "assistant", "content": text + FORCE}],
        "continue_final_message": True, "add_generation_prompt": False,
    }, timeout_s)
    ch2 = d2["choices"][0]
    merged = dict(d2)
    merged["choices"] = [dict(ch2)]
    merged["choices"][0]["message"] = dict(ch2["message"])
    merged["choices"][0]["message"]["content"] = (
        text + FORCE + (ch2["message"].get("content") or ""))
    merged["two_phase"] = True
    merged["phase1_usage"] = u
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", default="qwen35-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=2400.0)
    ap.add_argument("--cells", default="", help="comma list of task:variant to run (default: all)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--answer-cap", type=int, default=8192,
                    help="phase-2 code budget when phase 1 (--max-tokens) truncates")
    args = ap.parse_args()

    sys.path.insert(0, "tpu")
    from pallas_arena.probe.prompt_ref_first import build3, build3s
    from pallas_arena.probe.smoke_config import CELLS

    want = {tuple(c.split(":")) for c in args.cells.split(",") if c.strip()}
    jobs = []
    for (task, variant), (cases, example) in CELLS.items():
        if want and (task, variant) not in want:
            continue
        if example == "scaffold":
            prompt = build3s(task, cases)
        else:
            prompt = build3(task, cases, example=example)
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        for i in range(args.group_size):
            jobs.append((task, variant, i, prompt, ph))
    n_cells = len({(t, v) for t, v, *_ in jobs})
    print(f"[gen] {len(jobs)} generations over {n_cells} cells", flush=True)

    def run(job):
        task, variant, i, prompt, ph = job
        t0 = time.time()
        try:
            resp = _chat(args.server, args.model, prompt, args.max_tokens,
                         args.temperature, args.timeout_s,
                         ctx=args.ctx, answer_cap=args.answer_cap)
            ch = resp["choices"][0]
            return {
                "task": task, "variant": variant, "idx": i, "prompt_sha": ph,
                "two_phase": bool(resp.get("two_phase")),
                "phase1_usage": resp.get("phase1_usage"),
                "finish_reason": ch.get("finish_reason"),
                "text": ch["message"].get("content") or "",
                "reasoning": ch["message"].get("reasoning_content") or "",
                "usage": resp.get("usage"),
                "wall_s": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            return {"task": task, "variant": variant, "idx": i, "prompt_sha": ph,
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "wall_s": round(time.time() - t0, 1)}

    done = 0
    with open(args.out, "w") as f, cf.ThreadPoolExecutor(args.concurrency) as ex:
        for row in ex.map(run, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            done += 1
            if done % 8 == 0:
                print(f"[gen] {done}/{len(jobs)} at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[gen] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
