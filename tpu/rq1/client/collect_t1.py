# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""RQ1 T1 collector: N independent `codex exec` rollouts on one problem. PORTABLE, stdlib-only.

Each rollout is an isolated agent: it gets the problem pack prompt (statement + seed program +
seed PRODUCTION SCORE + direction), full local shell (compile/run/test itself -- there is NO
grading service), and must finish with exactly one submit() on the capture MCP. The driver
never grades; solutions are graded later, offline, by server/grade_batch.py.

  uv run collect_t1.py --problem fc46 --n 200 --model gpt-5.6-sol --effort xhigh \
      --site auto --out runs/fc46_B

Concurrency is a bounded worker pool (default 25 slots): slot frees as soon as a rollout exits,
no wave barriers. --resume skips rollouts that already have a done-marker. Capture fallback:
if a rollout never submit()s, its final message (-o) / workspace files are salvaged into the
same submissions.jsonl marked source=salvage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import preflight

HERE = Path(__file__).resolve().parent

T1_CONTRACT = """HOW TO WORK: you have full local shell access in your workspace -- write your
program, compile it (g++ / python3), run it on small budgets, and measure it yourself. There is
NO grading service available to you: your final program will be graded LATER by the organizers
with ONE production run at the exact budget stated in the problem, no more, no less. So (a) make
your program honor its stated time budget and scale its work to it, and (b) verify locally that
it runs end-to-end and beats the seed before you submit. Embedding a precomputed answer as data
(literal arrays, base64 blobs) is INVALID and will be rejected.

FINISH (mandatory): call the `capture` MCP tool submit(program, approach, insight,
session="{sess}") EXACTLY ONCE with:
  program:  your complete final program (one self-contained source text)
  approach: one line -- what idea this solution tried
  insight:  one line -- what should be learned for future improvement
As a backup, also put the same program in one ```{fence} block in your final message.
You have ~{wall_min} minutes of wall clock. Budget your time; an unsubmitted rollout scores zero.
"""


def sha(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def strip_fence(t):
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else (t or "")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def base_env(ch: Path):
    env = os.environ.copy()
    npm = Path.home() / ".npm-global" / "bin"
    if npm.exists():
        env["PATH"] = f"{npm}:{env['PATH']}"
    env["CODEX_HOME"] = str(ch)
    return env


def start_capture(out: Path, port: int, extra=()):
    cmd = ["uv", "run", "--script", str(HERE / "capture_mcp.py"),
           "--port", str(port), "--logdir", str(out)] + list(extra)
    p = subprocess.Popen(cmd, stdout=open(out / "capture.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            return p
        except OSError:
            if p.poll() is not None:
                raise RuntimeError(f"capture server died; see {out}/capture.log")
            time.sleep(1)
    raise RuntimeError("capture server failed to bind")


def load_pack(problem: str):
    d = HERE / "data" / problem
    meta = json.loads((d / "meta.json").read_text())
    return (d / "prompt_agent.md").read_text(), meta


def run_one(i, prompt, out, ch, model, effort, wall, lock, subf, soldir):
    sess = f"r{i:03d}"
    wd = out / sess
    wd.mkdir(exist_ok=True)
    done = wd / "done.json"
    (wd / "prompt.txt").write_text(prompt)
    t0 = time.time()
    cmd = ["codex", "exec", "--strict-config", "-m", model,
           "-c", f"model_reasoning_effort={effort}",
           "-s", "workspace-write", "-c", "approval_policy=never",
           "--json", "-C", str(wd), "-o", str(wd / "final.txt")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=open(wd / "events.jsonl", "w"),
                         stderr=subprocess.STDOUT, env=base_env(ch), cwd=str(wd), text=True)
    try:
        p.stdin.write(prompt)
        p.stdin.close()
    except Exception:
        pass
    status = "ok"
    try:
        p.wait(timeout=wall + 120)
    except subprocess.TimeoutExpired:
        p.kill()
        status = "wall-killed"
    # ---- capture fallback: salvage a program if this session never submit()ed ----
    have = False
    if subf.exists():
        with lock:
            txt = subf.read_text()
        have = any(json.loads(l).get("session") == sess
                   for l in txt.splitlines() if l.strip())
    if not have:
        prog = None
        fin = wd / "final.txt"
        if fin.exists() and "```" in (fin.read_text() or ""):
            prog = strip_fence(fin.read_text())
        if not prog:
            for cand in ("solution.txt", "solution.py", "solution.cpp", "main.cpp"):
                f = wd / cand
                if f.exists() and f.read_text().strip():
                    prog = strip_fence(f.read_text())
                    break
        if prog and prog.strip():
            h = sha(prog)
            with lock:
                (soldir / f"{h}.txt").write_text(prog)
                with open(subf, "a") as fh:
                    fh.write(json.dumps({"session": sess, "agent_key": "salvage",
                                         "sol_hash": h, "approach": "", "insight": "",
                                         "source": "salvage",
                                         "ts": round(time.time(), 1)}) + "\n")
            status += "+salvaged"
        else:
            status += "+no-program"
    done.write_text(json.dumps({"status": status, "secs": round(time.time() - t0)}))
    return sess, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--wall", type=int, default=1200, help="per-rollout wall seconds")
    ap.add_argument("--site", choices=["default", "neuronic", "auto"], default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="B")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    soldir = out / "solutions"
    soldir.mkdir(exist_ok=True)
    subf = out / "submissions.jsonl"
    lock = threading.Lock()

    prompt_pack, meta = load_pack(args.problem)
    prof = preflight.resolve_site(args.site, out / "preflight")
    port = free_port()
    ch = preflight.write_codex_home(out, landlock=prof["landlock"],
                                    long_provider=prof["long_provider"],
                                    mcp_url=f"http://127.0.0.1:{port}/mcp")
    cap = start_capture(out, port)

    (out / "manifest.json").write_text(json.dumps({
        "cell": args.cell, "problem": args.problem, "n": args.n, "model": args.model,
        "effort": args.effort, "wall": args.wall, "site_profile": prof,
        "seed_score": meta.get("seed_score"), "started": time.strftime("%F %T")}, indent=2))

    todo = []
    for i in range(args.n):
        if args.resume and (out / f"r{i:03d}" / "done.json").exists():
            continue
        todo.append(i)
    print(f"[t1] {args.problem} cell={args.cell}: {len(todo)}/{args.n} rollouts to run, "
          f"{args.concurrency} concurrent, wall={args.wall}s, model={args.model}/{args.effort}",
          flush=True)

    n_done = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = []
            for i in todo:
                sess = f"r{i:03d}"
                prompt = "\n".join([
                    prompt_pack, "",
                    T1_CONTRACT.format(sess=sess, fence=meta["fence"],
                                       wall_min=args.wall // 60)])
                futs.append(ex.submit(run_one, i, prompt, out, ch, args.model,
                                      args.effort, args.wall, lock, subf, soldir))
            for f in futs:
                sess, status = f.result()
                n_done += 1
                print(f"[t1] {sess}: {status} ({n_done}/{len(todo)})", flush=True)
    finally:
        cap.terminate()
    n_subs = sum(1 for l in subf.read_text().splitlines() if l.strip()) if subf.exists() else 0
    print(f"[t1] DONE: {n_subs} submissions from {args.n} rollouts -> {out}", flush=True)


if __name__ == "__main__":
    main()
