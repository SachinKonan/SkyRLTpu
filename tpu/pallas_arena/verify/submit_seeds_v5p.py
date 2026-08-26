"""Submit the CANDIDATE SEEDS (both rg_lru variants + splash) to the live
queue, wait for real-silicon verdicts, and write the orchestrator's inputs.

Deliberately NOT the full banked corpus: the v6e run already established
that the old "correct" kernels do not compile (E2003 / timeout / VMEM), and
regrading known failures spends judge minutes for nothing.

Outputs on success:
  runs/pallas_arena/seed-parity-results.json
      keys SEED-rglru-timeblocked / SEED-splash-flash (the names the
      orchestrator gates on); the rg_lru entry is the WINNING variant and
      records which file won under "seed_file".
  tpu/pallas_arena/probe/seed_rglru_active.py
      a copy of the winning rg_lru seed -- the file the arms prompt with.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import time
import urllib.request

REPO = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu")
PROBE = REPO / "tpu/pallas_arena/probe"
OUT = REPO / "runs/pallas_arena/seed-parity-results.json"

CANDIDATES = {
    # name -> (problem, file, orchestrator key it may win)
    "rglru-logscan": ("rg_lru", PROBE / "seed_rglru_logscan.py"),
    "rglru-prod4d": ("rg_lru", PROBE / "seed_rglru_timeblocked.py"),
    "splash-flash": ("splash_attention", PROBE / "seed_splash_flash.py"),
}


def _post(base, path, payload):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def main() -> None:
    base = (REPO / "runs/pallas_arena/rl-queue-url.txt").read_text().strip()
    wids = {}
    for name, (problem, f) in CANDIDATES.items():
        wids[name] = _post(base, "/submit", {"problem": problem, "code": f.read_text(),
                                             "tag": f"seedv5p-{name}"})["work_id"]
        print(f"submitted {name} ({problem})", flush=True)

    results, deadline = {}, time.time() + 3 * 3600
    pending = dict(wids)
    while pending and time.time() < deadline:
        got = _post(base, "/results", {"work_ids": list(pending.values())})["results"]
        for name in list(pending):
            rec = got.get(pending[name]) or {}
            if rec.get("done"):
                r = rec.get("result") or rec
                results[name] = r
                print(f"[verdict] {name}: passed={r.get('passed')} gate={r.get('gate')} "
                      f"reward={r.get('reward_with_bwd') or r.get('reward')} "
                      f"boot={r.get('task_boot_s')}s", flush=True)
                if not r.get("passed"):
                    print("   ", str((r.get("violations") or [""])[0])[:220], flush=True)
                del pending[name]
        if pending:
            time.sleep(15)
    for name in pending:
        results[name] = {"error": "no verdict before deadline"}
        print(f"[timeout] {name}", flush=True)

    # Pick the winning rg_lru variant; expose under the orchestrator's key.
    out = {}
    rg = [(n, results.get(n) or {}) for n in ("rglru-logscan", "rglru-prod4d")]
    rg_pass = [(n, r) for n, r in rg if r.get("passed")]
    if rg_pass:
        rg_pass.sort(key=lambda nr: float(nr[1].get("reward_with_bwd")
                                          or nr[1].get("reward") or 0), reverse=True)
        win_name, win = rg_pass[0]
        win = {**win, "seed_file": str(CANDIDATES[win_name][1])}
        shutil.copyfile(CANDIDATES[win_name][1], PROBE / "seed_rglru_active.py")
        print(f"[winner] rg_lru seed = {win_name} -> seed_rglru_active.py", flush=True)
        out["SEED-rglru-timeblocked"] = win
    else:
        out["SEED-rglru-timeblocked"] = rg[0][1] or {"passed": False}
        print("[winner] NO rg_lru variant passed", flush=True)
    out["SEED-splash-flash"] = results.get("splash-flash") or {"passed": False}
    out["_variants"] = {k: {kk: v.get(kk) for kk in
                            ("passed", "gate", "reward", "reward_with_bwd", "task_boot_s")}
                        for k, v in results.items()}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
