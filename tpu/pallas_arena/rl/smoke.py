"""Grade the seed at two tile sizes on one SkyPilot TPU slice; no training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pallas_arena.rl.task import public_contract


def grade(case, tile, output):
    from pallas_arena.judge.worker import PersistentWorker

    source = Path(__file__).with_name("seed_rglru.py").read_text()
    source = source.replace("BLOCK_T = 512", f"BLOCK_T = {tile}")
    worker = PersistentWorker("rg_lru", cases=[case], timing_pairs=20,
                              compile_budget_s=180, grade_budget_s=900,
                              worker_id=f"smoke-rank{os.environ.get('SKYPILOT_NODE_RANK', '0')}")
    start = time.monotonic()
    boot = worker.boot()
    result = worker.grade_code(source, enforce_pallas=True) if boot.get("ok") else {
        "ok": False, "passed": False, "gate": "judge_fault", "violations": [str(boot)]}
    result.update(task_boot_s=boot.get("boot_s"), task_noise_floor=boot.get("noise_floor"),
                  smoke_wall_s=time.monotonic() - start, smoke_case=case, smoke_tile=tile,
                  source=source, versions={name: importlib.metadata.version(name)
                                          for name in ("jax", "jaxlib", "recurrentgemma")})
    output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({key: result.get(key) for key in (
        "smoke_case", "smoke_tile", "passed", "gate", "score", "grad_scores", "violations")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--tile", type=int, default=512)
    args = parser.parse_args()
    if args.case:
        grade(args.case, args.tile, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
    hosts = len(os.environ.get("SKYPILOT_NODE_IPS", "localhost").split())
    cases = [name for name, _ in public_contract()[1]]
    tasks = [(case, tile) for tile in (512, 256) for case in cases]
    for i, (case, tile) in enumerate(tasks):
        if i % hosts != rank:
            continue
        output = args.output / f"{case}-t{tile}.json"
        with output.with_suffix(".log").open("w") as log:
            try:
                result = subprocess.run([sys.executable, "-m", "pallas_arena.rl.smoke",
                                         "--case", case, "--tile", str(tile), "--output", str(output)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=1500)
                if result.returncode:
                    raise RuntimeError(f"grade process exited {result.returncode}")
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                output.write_text(json.dumps(dict(ok=False, passed=False, gate="judge_fault",
                                                  violations=[str(exc)], smoke_case=case, smoke_tile=tile)))
        print(output.read_text(), flush=True)


if __name__ == "__main__":
    main()
