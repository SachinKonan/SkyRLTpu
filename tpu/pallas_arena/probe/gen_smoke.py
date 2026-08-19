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


def _chat(server: str, model: str, prompt: str, max_tokens: int, temperature: float,
          timeout_s: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.load(r)


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
    args = ap.parse_args()

    sys.path.insert(0, "tpu")
    from pallas_arena.probe.prompt_ref_first import build3
    from pallas_arena.probe.smoke_config import CELLS

    jobs = []
    for (task, variant), (cases, example) in CELLS.items():
        prompt = build3(task, cases, example=example)
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        for i in range(args.group_size):
            jobs.append((task, variant, i, prompt, ph))
    print(f"[gen] {len(jobs)} generations over {len(CELLS)} cells", flush=True)

    def run(job):
        task, variant, i, prompt, ph = job
        t0 = time.time()
        try:
            resp = _chat(args.server, args.model, prompt, args.max_tokens,
                         args.temperature, args.timeout_s)
            ch = resp["choices"][0]
            return {
                "task": task, "variant": variant, "idx": i, "prompt_sha": ph,
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
