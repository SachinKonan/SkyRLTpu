"""Submit the seed programs (+ the banked correct kernels) to a live arena
grading fleet and collect REAL-silicon verdicts: does each run at production
shapes, and where does it land vs the production baseline (seed target ~1.0x)?

Login-safe (pure urllib; no jax). Run AFTER rl_judges.sbatch publishes
runs/pallas_arena/rl-queue-url.txt:

    python3 -m pallas_arena.verify.submit_seed_parity          # from tpu/
or  python3 tpu/pallas_arena/verify/submit_seed_parity.py      # from repo root
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
import urllib.request

REPO = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu")
OUT = REPO / "runs/pallas_arena/seed-parity-results.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _post(base, path, payload):
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gather_candidates() -> dict[str, tuple[str, str]]:
    """name -> (problem, whole-program code). Contract blocks are composed
    into whole programs with the same jax-free machinery the RL env uses."""
    cc = _load("cc", REPO / "tpu/pallas_arena/probe/contract_compose.py")
    ss = _load("ss", REPO / "tpu/pallas_arena/probe/seam_scaffolds.py")
    scaf = {"rg_lru": ss.RGLRU_SCAFFOLD, "splash_attention": ss.SPLASH_SCAFFOLD}
    out: dict[str, tuple[str, str]] = {}

    # The two seeds (already whole programs).
    out["SEED-rglru-timeblocked"] = (
        "rg_lru", (REPO / "tpu/pallas_arena/probe/seed_rglru_timeblocked.py").read_text())
    out["SEED-splash-flash"] = (
        "splash_attention", (REPO / "tpu/pallas_arena/probe/seed_splash_flash.py").read_text())

    runs = REPO / "runs/pallas_arena"
    # Banked CONTRACT blocks (rf3c cold + heal) -> compose.
    for f in sorted(runs.glob("correct-*.py")):
        prob = "rg_lru" if ("rglru" in f.name or "rg_lru" in f.name) else "splash_attention"
        try:
            out[f.stem] = (prob, cc.compose_contract(scaf[prob], f.read_text()))
        except Exception as e:  # non-contract (whole-program) bank entries
            out[f.stem] = (prob, f.read_text())
            print(f"[gather] {f.name}: submitted as-is ({type(e).__name__})")
    # Campaign-era whole-program corrects.
    for f in sorted(runs.glob("first-correct-*.py")) + sorted(runs.glob("correct-rglru-kernel-*.py")):
        out.setdefault(f.stem, ("rg_lru", f.read_text()))
    return out


def main() -> None:
    url = (REPO / "runs/pallas_arena/rl-queue-url.txt").read_text().strip()
    cands = gather_candidates()
    print(f"queue {url}; submitting {len(cands)} candidates")
    wids = {}
    for name, (prob, code) in cands.items():
        wids[name] = _post(url, "/submit", {"problem": prob, "code": code})["work_id"]

    results, deadline = {}, time.time() + 4 * 3600
    while wids and time.time() < deadline:
        got = _post(url, "/results", {"work_ids": list(wids.values())})["results"]
        for name in list(wids):
            rec = got.get(wids[name]) or {}
            if rec.get("done"):
                r = rec.get("result") or {}
                results[name] = r
                rw = r.get("reward_with_bwd") or r.get("reward")
                print(f"[verdict] {name}: passed={r.get('passed')} reward={rw} gate={r.get('gate')}")
                del wids[name]
        if wids:
            time.sleep(15)
    for name in wids:
        results[name] = {"error": "no verdict before deadline"}
        print(f"[timeout] {name}")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
