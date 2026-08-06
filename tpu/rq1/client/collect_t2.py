# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""RQ1 T2 collector: N completions (thinking trace + code) from the self-hosted OSS farm.

Talks straight to a vLLM OpenAI-compatible endpoint (the tinker layer is not involved).
Every raw response is saved in full to raw/ (think + text) -- NEVER keep only metrics, raw
generations are the expensive artifact. Parsed code lands in the same solutions/ +
submissions.jsonl contract as T1, so grade_batch consumes both identically.

  uv run collect_t2.py --problem fc46 --n 200 --farm-url http://127.0.0.1:18001 \
      --model Qwen/Qwen3.5-27B --out runs/fc46_C

--resume skips indices whose raw/<i>.json already exists (safe across farm preemptions).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def sha(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def strip_fence(t):
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else ""


async def one(i, cli, args, prompt, out, lock):
    sid = f"t2_{i:03d}"
    rawf = out / "raw" / f"{sid}.json"
    for attempt in range(4):
        try:
            r = await cli.post(
                f"{args.farm_url.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {args.api_key}"},
                json={"model": args.model, "temperature": args.temperature,
                      "max_tokens": args.max_tokens,
                      "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 3:
                rawf.write_text(json.dumps({"id": sid, "error": str(e)[:400]}))
                return sid, "request-failed"
            await asyncio.sleep(10 * (attempt + 1))
    msg = data["choices"][0]["message"]
    think = msg.get("reasoning_content") or ""
    text = msg.get("content") or ""
    rawf.write_text(json.dumps({"id": sid, "think": think, "text": text,
                                "usage": data.get("usage"),
                                "finish": data["choices"][0].get("finish_reason")}))
    code = strip_fence(text) or strip_fence(think)
    if not code.strip():
        return sid, "no-code-block"
    h = sha(code)
    async with lock:
        (out / "solutions" / f"{h}.txt").write_text(code)
        with open(out / "submissions.jsonl", "a") as fh:
            fh.write(json.dumps({"session": sid, "agent_key": "t2", "sol_hash": h,
                                 "approach": "", "insight": "", "source": "t2",
                                 "ts": round(time.time(), 1)}) + "\n")
    return sid, "ok"


async def run(args):
    out = Path(args.out).resolve()
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "solutions").mkdir(exist_ok=True)
    d = HERE / "data" / args.problem
    prompt = (d / "prompt_completion.md").read_text()
    meta = json.loads((d / "meta.json").read_text())
    (out / "manifest.json").write_text(json.dumps({
        "cell": args.cell, "problem": args.problem, "n": args.n, "model": args.model,
        "farm_url": args.farm_url, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "seed_score": meta.get("seed_score"),
        "started": time.strftime("%F %T")}, indent=2))

    todo = [i for i in range(args.n)
            if not (args.resume and (out / "raw" / f"t2_{i:03d}.json").exists())]
    print(f"[t2] {args.problem} cell={args.cell}: {len(todo)}/{args.n} samples, "
          f"conc={args.concurrency}, model={args.model}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    n_ok = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=60)) as cli:
        async def guarded(i):
            async with sem:
                return await one(i, cli, args, prompt, out, lock)
        done = 0
        for fut in asyncio.as_completed([guarded(i) for i in todo]):
            sid, status = await fut
            done += 1
            n_ok += status == "ok"
            if done % 10 == 0 or status != "ok":
                print(f"[t2] {sid}: {status} ({done}/{len(todo)}, ok={n_ok})", flush=True)
    print(f"[t2] DONE: {n_ok}/{len(todo)} parsed programs -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--farm-url", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=28000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="C")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
