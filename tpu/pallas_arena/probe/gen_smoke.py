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
import http.client
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


def _canon_family(model: str) -> str:
    try:
        from pallas_arena.probe import canonical
    except ImportError:
        import canonical  # type: ignore
    return canonical.family_of(model)


def extract_completion(text: str, required_defs: list[str] | None = None,
                       family: str | None = None) -> str | None:
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
    # THE ANSWER, NOT THE REASONING. These models sketch code WHILE thinking:
    # one qwen completion carried 9 fenced blocks of which 8 were fragments
    # inside <think> and only the 9th, after it, was the program. The split
    # is delegated to ttt_discover's RosettaStone so the arena and the
    # training paths can never disagree about where reasoning ends; the
    # arena-side reconstitution of markers the chat endpoint drops lives in
    # probe/canonical.py. Falls through to the whole text when the family is
    # unknown or the markers are absent.
    region = text
    if family:
        try:
            from pallas_arena.probe import canonical
        except ImportError:                 # running from inside probe/
            import canonical               # type: ignore
        region, _split = canonical.answer(text, family)
    if cue in region:
        region = region.rsplit(cue, 1)[1]
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


# The client reaches the engine through an SSH port-forward. That tunnel
# flaps -- it has its own restart loop -- and while it is down every request
# fails at the CONNECTION level even though the engine is healthy. Measured
# 2026-08-27 on the gemma rg_lru arm: 22 of 32 generations were lost to
# `Connection refused`/`RemoteDisconnected` while the engine log showed
# `Running: 6 reqs` at 500 tok/s and returned 200s throughout. Those are
# transient by construction, so retry them; HTTP errors (400/500) are the
# engine's real answer and must NOT be retried here.
_CONN_ERRORS = (urllib.error.URLError, http.client.RemoteDisconnected,
                ConnectionResetError, ConnectionRefusedError, TimeoutError)


def _post(server: str, body: dict, timeout_s: float, conn_retries: int = 6) -> dict:
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(conn_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return json.load(r)
        except urllib.error.HTTPError:
            raise                      # the engine answered; that is a verdict
        except _CONN_ERRORS as e:
            if attempt >= conn_retries:
                raise
            wait = min(30, 5 * (attempt + 1))
            print(f"[conn] {type(e).__name__} on attempt {attempt + 1}/{conn_retries + 1}"
                  f" -- tunnel likely restarting; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _chat(server: str, model: str, prompt: str, max_tokens: int, temperature: float,
          timeout_s: float, ctx: int = 32768, answer_cap: int = 8192,
          extra_body: dict | None = None, phase1_total: int = 0,
          think_budget: int = 0) -> dict:
    """Two-phase completion. Thinking stays ON (by direction: we sample the
    policy we would train), but it is CAPPED: phase 1 gets `max_tokens`; if
    it truncates, the assistant message is continued past a forcing cue with
    an answer budget sized from ACTUAL usage (rq2 loop.py measured fixed p2
    overflowing the context -> vLLM 400 -- sizing from usage is load-bearing).
    Measured motivation: 16k/18k/27k single-phase caps truncated 53-97% of
    completions -- thinking expands to fill ANY budget it is given.

    ``phase1_total`` (when > 0) switches to the RESERVE-FIRST budget of
    ttt_discover's TwoPhaseTokenCompleter: it is the total context phase 1
    may occupy INCLUDING the prompt, which guarantees
    ``ctx - phase1_total - 64`` tokens for the answer no matter how much the
    model thinks. The old behaviour gives thinking a fixed cap and lets the
    answer take whatever survives, which is how 7 of 32 qwen rg_lru
    candidates (2026-08-27) finished normally having emitted only reasoning
    and no program at all -- the answer phase had nothing left to spend."""
    base = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1.0,
        # KEEP THE FAMILY SURFACE. gemma-4's '<|channel>' (id 100) and
        # '<channel|>' (id 101) are added tokens with special=True, so the
        # default detokenization strips them and RosettaStone -- the one
        # implementation of "where does reasoning end" -- has nothing to
        # match. qwen3.5's <think>/</think> are special=False and survive
        # either way, so this is safe for both.
        "skip_special_tokens": False,
    }
    if extra_body:
        base.update(extra_body)
    if think_budget and not phase1_total:
        # Probe the true prompt size, then let phase 1 have exactly
        # think_budget new tokens; everything else belongs to the answer.
        probe = _post(server, {**base, "max_tokens": 1}, timeout_s)
        ptoks = ((probe.get("usage") or {}).get("prompt_tokens") or 0)
        phase1_total = ptoks + think_budget
    if phase1_total:
        # RESERVE THE ANSWER FIRST. Measure the true prompt size (tokenizers
        # differ per model; qwen-token estimates already mispredicted gemma
        # badly enough to 400 every request), then let phase 1 have only what
        # is left of its total allowance.
        probe = _post(server, {**base, "max_tokens": 1}, timeout_s)
        ptoks = ((probe.get("usage") or {}).get("prompt_tokens") or 0)
        p1 = phase1_total - ptoks
        reserved = ctx - phase1_total - 64
        if p1 < 512:
            raise RuntimeError(
                f"prompt {ptoks} leaves {p1} phase-1 tokens of a {phase1_total} budget")
        print(f"[budget] prompt={ptoks} think<={p1} answer_reserved={reserved}", flush=True)
        max_tokens = p1
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
    p2 = min(answer_cap, room) if not phase1_total else max(room, 0)
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
    ap.add_argument("--seed-file", default="",
                    help="path to a WORKING annotated program: run one improvement turn per "
                         "sample on it (the seeded-RL one-step test). Cell tasks come from "
                         "--cells; variant is tagged +seed.")
    ap.add_argument("--seed-observation", default="",
                    help="path to the REAL judge observation text for the seed (from the parity run)")
    ap.add_argument("--seed-reward", default="1.0x (parity with the production kernel -- reward only accrues ABOVE this)",
                    help="reward line shown for the seed program")
    ap.add_argument("--no-think", action="store_true",
                    help="chat_template_kwargs enable_thinking=False and single-phase answer-cap "
                         "generation (the cheap-heal arm: no thinking budget at all)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--think-budget", type=int, default=0,
                    help="tokens of THINKING allowed; the answer is guaranteed "
                         "ctx - prompt - think_budget - 64. Preferred over "
                         "--phase1-total because it self-adjusts to the measured "
                         "prompt, which differs per model and per task.")
    ap.add_argument("--phase1-total", type=int, default=0,
                    help="RESERVE-FIRST budgeting (ttt_discover TwoPhaseTokenCompleter "
                         "semantics): total context phase 1 may occupy INCLUDING the "
                         "prompt. The answer is then guaranteed ctx - phase1_total - 64 "
                         "tokens regardless of how much the model thinks. 0 = legacy "
                         "behaviour (fixed think cap, answer gets the remainder).")
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
    if args.seed_file:
        # SEEDED ONE-STEP TEST: every sample is an improvement turn on ONE
        # known-good annotated program (production-structure seed). This is
        # the erdos/ac-inequalities initial-state pattern applied to kernels.
        from pallas_arena.probe.prompt_ref_first import SEED_IMPROVE_TEMPLATE, build3seed
        seed_program = open(args.seed_file).read()
        obs = ("passed: correct on every test shape, forward and backward. "
               "Reward accrues only for making it FASTER (uniformly across "
               "shapes, fwd and bwd).")
        if args.seed_observation:
            # The REAL judge observation for this seed (per-shape fwd/bwd
            # ratios) -- produced by the parity fleet run, exactly what the
            # RL loop would show.
            obs = open(args.seed_observation).read().strip()
        for (task, variant), (cases, _kind) in CELLS.items():
            if want and (task, variant) not in want:
                continue
            prompt = SEED_IMPROVE_TEMPLATE.format(
                base=build3seed(task, cases),
                reward=args.seed_reward,
                program=seed_program,
                observation=obs,
            )
            ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            for i in range(args.group_size):
                jobs.append((task, f"{variant}+seed", i, prompt, ph))
    elif args.repair_from:
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
                req = None
                if variant.startswith("rf3c"):
                    from pallas_arena.probe.contract_compose import scan_scaffold
                    from pallas_arena.probe.seam_scaffolds import RGLRU_SCAFFOLD, SPLASH_SCAFFOLD
                    _scaf = {"rg_lru": RGLRU_SCAFFOLD, "splash_attention": SPLASH_SCAFFOLD}[task]
                    req = list(scan_scaffold(_scaf).required_defs)
                program = extract_completion(text, required_defs=req)
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
        if args.no_think:
            variant = f"{variant}-nt"   # separate grading cell for the A/B
        t0 = time.time()
        try:
            if args.no_think:
                # Cheap heal: no thinking, answer only. Single phase -- the
                # two-phase machinery exists purely to cap thinking.
                extra = {"chat_template_kwargs": {"enable_thinking": False}}
                p1 = args.answer_cap
                resp = _chat(args.server, args.model, prompt, p1,
                             args.temperature, args.timeout_s,
                             ctx=args.ctx, answer_cap=0, extra_body=extra)
            else:
                extra = ({"chat_template_kwargs": {"enable_thinking": True}}
                         if args.enable_thinking_kwarg else None)
                resp = _chat(args.server, args.model, prompt, args.max_tokens,
                             args.temperature, args.timeout_s,
                             ctx=args.ctx, answer_cap=args.answer_cap, extra_body=extra,
                             phase1_total=args.phase1_total,
                             think_budget=args.think_budget)
            ch = resp["choices"][0]
            return {
                "task": task, "variant": variant, "idx": i, "prompt_sha": ph,
                # RECORD THE FAMILY. Grading must split reasoning from answer
                # exactly as sampling did, and it cannot infer the surface
                # from the text alone.
                "family": _canon_family(args.model),
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
