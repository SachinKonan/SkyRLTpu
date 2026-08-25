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
         "self-contained program, with the compute inside a real "
         "`pl.pallas_call` kernel and every import included:\n\n```python\n")


def extract_completion(text: str, required_defs: list[str] | None = None) -> str | None:
    """The program in a completion: after the forcing cue when present
    (fence-closed or not), else the last fenced block defining kernel().

    required_defs (contract cells): models routinely spread the answer over
    4-9 fenced blocks (re-thinking in the answer phase); measured on bench
    3750899, ALL 7 'missing required defs' contract violations were this
    extractor taking the wrong single block while every required def existed
    in the text. With required_defs we pick the block defining the most of
    them, and if they are scattered, merge the parsable def/import/TUNABLE
    blocks into one payload."""
    import re
    cue = "I have thought about this enough"
    region = text.rsplit(cue, 1)[1] if cue in text else text
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", region, re.S)
    # An unclosed trailing fence (cap hit mid-block) still carries code.
    if region.count("```") % 2 == 1 and "```python\n" in region:
        blocks.append(region.rsplit("```python\n", 1)[1])
    if not blocks and region is not text:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)

    if required_defs and blocks:
        def score(b: str) -> int:
            return sum(1 for fn in required_defs if f"def {fn}" in b)
        best = max(blocks, key=score)
        if score(best) == len(required_defs):
            return best.strip()
        import ast
        keep = []
        for b in blocks:
            try:
                ast.parse(b)
            except SyntaxError:
                continue
            if score(b) or re.search(r"(^|\n)(import |from |def |[A-Z_][A-Z_0-9]*\s*=)", b):
                keep.append(b)
        merged = "\n\n".join(keep).strip()
        if merged and score(merged) > score(best):
            return merged
        if score(best):
            return best.strip()
        # fall through: no block carries any required def

    if cue in text:
        after = text.rsplit(cue, 1)[1]
        if "```python\n" in after:
            after = after.split("```python\n", 1)[1]
        return after.split("```", 1)[0].strip() or None
    with_kernel = [b for b in blocks if "def kernel" in b]
    if with_kernel:
        return with_kernel[-1].strip()
    return blocks[-1].strip() if blocks else None


def _post(server: str, body: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.load(r)


def _chat(server: str, model: str, prompt: str, max_tokens: int, temperature: float,
          timeout_s: float, ctx: int = 32768, answer_cap: int = 8192,
          extra_body: dict | None = None) -> dict:
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
    if extra_body:
        base.update(extra_body)
    try:
        d = _post(server, {**base, "max_tokens": max_tokens}, timeout_s)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        # Phase-1 400 = prompt + think cap exceeds ctx UNDER THIS MODEL'S
        # TOKENIZER (measured: gemma inflates the splash rf3c prompt past
        # what qwen-token estimates predicted; all 32 requests 400'd).
        # Measure the true prompt size with a 1-token probe, then clamp.
        probe = _post(server, {**base, "max_tokens": 1}, timeout_s)
        ptoks = ((probe.get("usage") or {}).get("prompt_tokens") or 0)
        # Reserve answer room: correct programs measure 2-4.4k tokens, and
        # thinking fills any cap it is given -- without the reserve, phase 2
        # would get sized to ~nothing and the reply ends as pure thinking.
        reserve = min(answer_cap, 5120)
        clamped = ctx - ptoks - reserve - 64
        if clamped < 512:
            raise
        d = _post(server, {**base, "max_tokens": clamped}, timeout_s)
        max_tokens = clamped
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
    ap.add_argument("--repair-from", default="", help="graded json of a prior round: run ONE repair turn per failed program (the RL improvement turn)")
    ap.add_argument("--gens-from", default="", help="gens jsonl matching --repair-from")
    ap.add_argument("--enable-thinking-kwarg", action="store_true",
                    help="send chat_template_kwargs enable_thinking=True (gemma needs it; qwen thinks by default)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--answer-cap", type=int, default=8192,
                    help="phase-2 code budget when phase 1 (--max-tokens) truncates")
    args = ap.parse_args()

    sys.path.insert(0, "tpu")
    from pallas_arena.probe.prompt_ref_first import build3, build3s
    from pallas_arena.probe.smoke_config import CELLS

    want = {tuple(c.split(":")) for c in args.cells.split(",") if c.strip()}

    def base_prompt(task, variant):
        cases, example = CELLS[(task, variant)]
        if example == "scaffold":
            return build3s(task, cases)
        if example == "contract":
            from pallas_arena.probe.prompt_ref_first import build3c
            return build3c(task, cases)
        return build3(task, cases, example=example)

    jobs = []
    if args.repair_from:
        # THE RL IMPROVEMENT TURN, simulated: base prompt + the candidate's
        # own program + the verbatim judge feedback it earned.
        from pallas_arena.probe.prompt_ref_first import IMPROVE_TEMPLATE
        graded = json.load(open(args.repair_from))
        prior = {}
        for line in open(args.gens_from):
            r = json.loads(line)
            if not r.get("error"):
                prior[(r["task"], r["variant"], r["idx"])] = r.get("text") or ""
        for cell_key, celld in graded.items():
            task, variant = cell_key.split(":")
            if want and (task, variant) not in want:
                continue
            for row in celld["rows"]:
                out = str(row.get("outcome") or "")
                text = prior.get((task, variant, row["idx"]))
                if text is None or out.startswith("gen_error") or out == "no_program":
                    continue
                program = extract_completion(text)
                if not program:
                    continue
                fb = out.replace("pregate: ", "", 1)
                prompt = IMPROVE_TEMPLATE.format(
                    base=base_prompt(task, variant),
                    reward="0.0 (failed validity)" if out != "correct" else "passed validity",
                    program=program,
                    observation=fb,
                )
                ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]
                jobs.append((task, f"{variant}+repair", row["idx"], prompt, ph))
    else:
        for (task, variant), (cases, example) in CELLS.items():
            if want and (task, variant) not in want:
                continue
            prompt = base_prompt(task, variant)
            ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            for i in range(args.group_size):
                jobs.append((task, variant, i, prompt, ph))
    n_cells = len({(t, v) for t, v, *_ in jobs})
    print(f"[gen] {len(jobs)} generations over {n_cells} cells", flush=True)

    def run(job):
        task, variant, i, prompt, ph = job
        t0 = time.time()
        try:
            extra = ({"chat_template_kwargs": {"enable_thinking": True}}
                     if args.enable_thinking_kwarg else None)
            resp = _chat(args.server, args.model, prompt, args.max_tokens,
                         args.temperature, args.timeout_s,
                         ctx=args.ctx, answer_cap=args.answer_cap, extra_body=extra)
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
