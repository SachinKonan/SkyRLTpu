"""Bootstrap a NEW problem: sample seedless completions, grade them, and answer two questions
at once --

  1. is the problem SATURATED?  (ud failed this: 10 programs tied at exactly 18.58, 0/198 beat
     it, 198 programs -> only 15 distinct scores. A problem whose ceiling one-shot sampling
     already reaches cannot discriminate between treatments.)
  2. what should the SEED be?   (a mid-range program with real headroom, same rule the other
     packs use -- not the best one, which would leave nothing to find.)

Chicken-and-egg it avoids: make_problem_pack needs a seed, and a seed needs samples. Here the
prompt is built from the env's own get_question with an EMPTY state, which emits the
"write code to optimize this" variant rather than the "improve this program" one.

  $PY bootstrap_problem.py --problem fc159 --n 40 --registry <registry.json>
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fleet"))
sys.path.insert(0, str(HERE.parent / "client"))
sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation")
import httpx  # noqa: E402
import registry as REG  # noqa: E402

FRONTIER = "/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover"
MAIN = "/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover"
# problem -> (root, module, class, problem_type, lang, maximize)
NEW = {
    "fc159": (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "159", "cpp", True),
}
MODEL_CFG = {"qwen35": {"model": "Qwen/Qwen3.5-27B", "ctx": 22528, "p1": 12000, "p2": 7000},
             "gemma4": {"model": "google/gemma-4-31B-it", "ctx": 16384, "p1": 9000, "p2": 6000}}
FORCE = ("\n\nI have thought about this enough. Here is my final, complete, self-contained "
         "program:\n\n```{fence}\n")


def load_env(root, mod, cls):
    for k in [k for k in sys.modules if k == "examples" or k.startswith("examples.")]:
        del sys.modules[k]
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return getattr(importlib.import_module(mod), cls)


def question(problem):
    root, mod, cls, ptype, lang, mx = NEW[problem]
    E = load_env(root, mod, cls)
    st = E.create_initial_state(ptype)
    st.code = ""                      # seedless -> "write code to optimize this" variant
    st.value = None
    env = E.__new__(E)
    env.initial_state = st
    env.problem_type = ptype
    env.eval_timeout = 1000
    return env.get_question(), lang, mx


def strip_fence(t):
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else ""


async def sample(n, urls, prompt, fence, out, conc=16):
    sem = asyncio.Semaphore(conc)
    res = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(2400, connect=30)) as cli:
        async def one(i):
            mkey = "qwen35" if i % 2 == 0 else "gemma4"
            pool = urls.get(mkey) or urls.get("qwen35") or urls.get("gemma4")
            if not pool:
                return None
            cfg = MODEL_CFG[mkey]
            url = pool[i % len(pool)]
            base = {"model": cfg["model"], "temperature": 1.0,
                    "messages": [{"role": "user", "content": prompt}]}
            if mkey == "gemma4":
                base["chat_template_kwargs"] = {"enable_thinking": True}
            async with sem:
                try:
                    r = await cli.post(f"{url}/v1/chat/completions",
                                       headers={"Authorization": "Bearer EMPTY"},
                                       json={**base, "max_tokens": cfg["p1"]})
                    r.raise_for_status()
                    d = r.json()
                except Exception as e:
                    return {"id": i, "error": str(e)[:200]}
                text = d["choices"][0]["message"].get("content") or ""
                fin = d["choices"][0].get("finish_reason")
                forced = ""
                if fin == "length":
                    u = d.get("usage") or {}
                    room = cfg["ctx"] - ((u.get("prompt_tokens") or 0)
                                         + (u.get("completion_tokens") or cfg["p1"])) - 256
                    if room >= 512:
                        forced = FORCE.format(fence=fence)
                        try:
                            r2 = await cli.post(f"{url}/v1/chat/completions",
                                headers={"Authorization": "Bearer EMPTY"},
                                json={**base, "max_tokens": min(cfg["p2"], room),
                                      "messages": base["messages"] + [
                                          {"role": "assistant", "content": text + forced}],
                                      "continue_final_message": True,
                                      "add_generation_prompt": False})
                            r2.raise_for_status()
                            text = text + forced + (r2.json()["choices"][0]["message"].get("content") or "")
                        except Exception:
                            forced = ""
                code = (text.split(forced, 1)[1].split("```")[0] if forced
                        else strip_fence(text))
                (out / "raw" / f"{i:03d}.json").write_text(json.dumps(
                    {"id": i, "model": mkey, "text": text, "finish": fin}))
                return {"id": i, "model": mkey, "program": code or None}
        for fut in asyncio.as_completed([one(i) for i in range(n)]):
            r = await fut
            if r:
                res.append(r)
                if len(res) % 10 == 0:
                    print(f"[boot] sampled {len(res)}/{n}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(NEW))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--registry",
                    default="/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq2/fleet/registry.json")
    ap.add_argument("--out", default="/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq2/bootstrap")
    ap.add_argument("--grade-concurrency", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out) / args.problem
    (out / "raw").mkdir(parents=True, exist_ok=True)
    q, lang, maximize = question(args.problem)
    fence = "cpp" if lang == "cpp" else "python"
    prompt = q + (f"\n\nThink it through, then output your single best COMPLETE program as ONE "
                  f"```{fence} block (the last such block is taken as your answer).\n")
    (out / "prompt_used.md").write_text(prompt)
    urls = {m: REG.urls(args.registry, model=m) for m in ("qwen35", "gemma4")}
    print(f"[boot] {args.problem}: prompt {len(prompt)} chars; "
          f"endpoints qwen={len(urls['qwen35'])} gemma={len(urls['gemma4'])}", flush=True)

    t0 = time.time()
    res = asyncio.run(sample(args.n, urls, prompt, fence, out))
    got = [r for r in res if r.get("program")]
    print(f"[boot] {len(got)}/{args.n} parsed programs in {int(time.time()-t0)}s", flush=True)

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from grading_mcp import _grade
    root, mod, cls, ptype, _l, _m = NEW[args.problem]
    with ProcessPoolExecutor(max_workers=args.grade_concurrency) as ex:
        futs = {ex.submit(_grade, root, mod, cls, ptype, lang, None, r["program"], False,
                          str(out / "gradelogs")): r for r in got}
        for f in as_completed(futs):
            r = futs[f]
            try:
                g = f.result()
                r["score"] = g["score"] if g.get("valid") else None
                r["detail"] = (g.get("detail") or "")[:120]
            except Exception as e:
                r["score"] = None
                r["detail"] = str(e)[:120]

    ok = sorted([r for r in got if r.get("score") is not None],
                key=lambda r: r["score"], reverse=maximize)
    scores = [r["score"] for r in ok]
    dist = Counter(round(s, 4) for s in scores)
    print(f"\n===== {args.problem} BOOTSTRAP =====")
    print(f"parsed {len(got)}/{args.n}, valid {len(ok)}")
    if not ok:
        print("NO VALID PROGRAMS -- cannot seed this problem"); return
    print(f"scores: best={scores[0]:.6g} median={scores[len(scores)//2]:.6g} worst={scores[-1]:.6g}")
    print(f"DISTINCT scores: {len(dist)} across {len(scores)} programs")
    print(f"  most common: {dist.most_common(4)}")
    tied = dist.most_common(1)[0][1]
    print(f"\nSATURATION READ: {'CONCERN' if len(dist) < max(5, len(scores)//4) or tied > len(scores)//3 else 'looks discriminative'}"
          f"  (ud for contrast: 15 distinct / 198, 120 tied on one value)")
    # seed = mid-range, same rule the other packs use
    seed = ok[len(ok) // 2]
    (out / "seed_candidate.cpp").write_text(seed["program"])
    (out / "bootstrap.json").write_text(json.dumps(
        {"problem": args.problem, "n": args.n, "parsed": len(got), "valid": len(ok),
         "scores": scores, "distinct": len(dist), "seed_score": seed["score"]}, indent=2))
    print(f"seed candidate (median): score={seed['score']:.6g} -> {out}/seed_candidate.cpp")


if __name__ == "__main__":
    main()
