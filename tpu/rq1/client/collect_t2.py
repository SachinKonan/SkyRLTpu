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
ATTEMPTS = 8


def sha(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def strip_fence(t):
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else ""


FORCE = ("\n\nI have thought about this enough. Here is my final, complete, self-contained "
         "program:\n\n```{fence}\n")


async def _post(cli, args, body, tries=ATTEMPTS):
    """POST with patient retries (the farm tunnel can be down for minutes at a time)."""
    last = None
    for attempt in range(tries):
        try:
            r = await cli.post(f"{args.farm_url.rstrip('/')}/v1/chat/completions",
                               headers={"Authorization": f"Bearer {args.api_key}"}, json=body)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt == tries - 1:
                raise last
            await asyncio.sleep(min(60, 10 * (attempt + 1)))


async def one(i, cli, args, prompt, out, lock):
    sid = f"t2_{i:03d}"
    rawf = out / "raw" / f"{sid}.json"
    base = {"model": args.model, "temperature": args.temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if args.extra_body:
        base.update(json.loads(args.extra_body))
    p1 = args.phase1_tokens if args.two_phase else args.max_tokens
    # Patient retries live in _post: the farm tunnel can be down for minutes with a healthy TPU.
    try:
        data = await _post(cli, args, {**base, "max_tokens": p1})
    except Exception as e:
        rawf.write_text(json.dumps({"id": sid, "error": str(e)[:400]}))
        return sid, "request-failed"
    msg = data["choices"][0]["message"]
    think = msg.get("reasoning_content") or ""
    text = msg.get("content") or ""
    finish = data["choices"][0].get("finish_reason")

    # ---- phase 2: force the final program ----
    # A verbose reasoner (qwen: 83% truncated at 28k, median exactly at the cap) never reaches
    # its answer, and its transcript is littered with exploratory code blocks -- so "last fenced
    # block" grabs a stub like "// Greedy construction\n// ...". Mirroring ttt_discover's
    # TwoPhaseTokenCompleter, cap the thinking, then continue the assistant turn with an explicit
    # hand-off into an open fence, so the continuation IS the final program.
    forced = ""
    if args.two_phase and finish == "length":
        forced = FORCE.format(fence=args.fence)
        try:
            d2 = await _post(cli, args, {
                **base, "max_tokens": args.phase2_tokens,
                "messages": base["messages"] + [{"role": "assistant", "content": text + forced}],
                "continue_final_message": True, "add_generation_prompt": False})
            cont = d2["choices"][0]["message"].get("content") or ""
            text = text + forced + cont
            finish = f"length+{d2['choices'][0].get('finish_reason')}"
        except Exception as e:
            rawf.write_text(json.dumps({"id": sid, "think": think, "text": text,
                                        "usage": data.get("usage"), "finish": finish,
                                        "phase2_error": str(e)[:300]}))
            return sid, "phase2-failed"

    rawf.write_text(json.dumps({"id": sid, "think": think, "text": text,
                                "usage": data.get("usage"), "finish": finish,
                                "two_phase": bool(forced)}))
    if forced:
        # Everything after the hand-off is the answer; it may or may not close its fence.
        tail = text.split(forced, 1)[1]
        code = tail.split("```")[0]
    else:
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
    args.fence = meta["fence"]          # phase-2 hand-off opens a fence of the right language
    (out / "manifest.json").write_text(json.dumps({
        "cell": args.cell, "problem": args.problem, "n": args.n, "model": args.model,
        "farm_url": args.farm_url, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "extra_body": args.extra_body,
        "seed_score": meta.get("seed_score"),
        "started": time.strftime("%F %T")}, indent=2))

    def collected(i):
        """A raw file exists for FAILED requests too (it records the error), so resuming on
        mere existence silently abandons every sample lost to a preemption. Only a raw file
        with actual model output counts as collected."""
        f = out / "raw" / f"t2_{i:03d}.json"
        if not f.exists():
            return False
        try:
            d = json.loads(f.read_text())
        except Exception:
            return False
        return not d.get("error") and bool((d.get("text") or "") or (d.get("think") or ""))

    todo = [i for i in range(args.n) if not (args.resume and collected(i))]
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
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into the request body. gemma4 THINKING NEEDS "
                         '\'{"chat_template_kwargs": {"enable_thinking": true}}\' -- the '
                         "stock template only injects the <|think|> system turn with that "
                         "flag; without it the model emits an empty thought channel.")
    ap.add_argument("--two-phase", action="store_true",
                    help="cap thinking at --phase1-tokens, then continue the assistant turn "
                         "into an open code fence so a verbose model still emits a final "
                         "program (mirrors ttt_discover's TwoPhaseTokenCompleter)")
    ap.add_argument("--phase1-tokens", type=int, default=16000)
    ap.add_argument("--phase2-tokens", type=int, default=12000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="C")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
