"""CPU simulation of the phase-5 fleet demonstration — zero TPU.

Real queue process + N real worker processes + the real `fleet_driver`, with
grading mocked. The mock is not a stub that always says "pass": it looks up
each corpus tag's EXPECTED verdict, gate and phase-4 measured relative cost,
and adds a deterministic per-judge score offset. So the driver's whole
acceptance path — verdicts at the right gate, the terminal compile-bomb,
regrade spread, cross-judge agreement, cache-identical repeats, mid-lease
chaos, exactly-once accounting, throughput bookkeeping — is exercised end to
end before a single chip is provisioned. The phase-3 lesson: every bug the
simulation can find costs zero spot-TPU minutes to find.

What the simulation deliberately cannot show: real compile/chip times, real
numerics, real preemption. Those are the fleet run's job.

  sbatch tpu/pallas_arena/phase5/simulate_fleet.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ARENA_IMPORT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARENA_IMPORT_ROOT))

REPO_ROOT = ARENA_IMPORT_ROOT.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ARENA_IMPORT_ROOT), env.get("PYTHONPATH", "")])
    env["JAX_PLATFORMS"] = "cpu"
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", type=int, default=5)
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--dups", type=int, default=40)
    ap.add_argument("--mock-grade-s", type=float, default=0.5)
    ap.add_argument("--lease-timeout", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    port = _free_port()
    out = args.out or f"/tmp/phase5-sim-{os.getpid()}.json"
    procs: list[subprocess.Popen] = []
    names = [f"sim-judge-{i + 1}" for i in range(args.judges)]

    try:
        print(f"[sim] queue on 127.0.0.1:{port} lease={args.lease_timeout}s", flush=True)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable, "-m", "pallas_arena.judge.queue",
                    "--port", str(port), "--lease-timeout", str(args.lease_timeout),
                ],
                env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        )
        deadline = time.time() + 40
        healthy = False
        while time.time() < deadline and not healthy:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:
                    healthy = json.loads(r.read()).get("ok", False)
            except Exception:
                time.sleep(0.3)
        if not healthy:
            print("[sim] queue never became healthy")
            return 1

        for name in names:
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable, "-m", "pallas_arena.judge.worker",
                        "--problem", "rmsnorm", "--queue", f"http://127.0.0.1:{port}",
                        "--sim-mode", "mock", "--mock-grade-s", str(args.mock_grade_s),
                        "--worker-id", name, "--poll-s", "0.2",
                    ],
                    env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            )
        print(f"[sim] {len(names)} mock judges: {names}", flush=True)

        # The kill has to be a real SIGKILL of a real process holding a real
        # lease -- a simulated "pretend it died" would test nothing. pkill
        # never matches its own pid, so the pattern is safe.
        kill_cmd = "pkill -9 -f -- '--worker-id {judge}'"
        rc = subprocess.run(
            [
                sys.executable, str(Path(__file__).with_name("fleet_driver.py")),
                "--queue", f"http://127.0.0.1:{port}",
                "--out", out,
                "--judges", ",".join(names),
                "--judges-expected", str(len(names)),
                "--judges-min", "2",
                "--fleet-wait-s", "120",
                "--target", str(args.target),
                "--dups", str(args.dups),
                "--grade-wait-s", "900",
                "--dup-wait-s", "300",
                "--poll-s", "0.5",
                "--chaos-after", "0.3",
                "--chaos-wait-s", "180",
                "--chaos-kill-cmd", kill_cmd,
            ],
            env=_env(),
            cwd=str(REPO_ROOT),
        ).returncode
        print(f"[sim] fleet_driver rc={rc}; results -> {out}", flush=True)
        try:
            res = json.loads(Path(out).read_text())
            bad = [k for k, v in res["invariants"].items() if not v["ok"]]
            print(f"[sim] invariants: {len(res['invariants']) - len(bad)} pass, {len(bad)} fail {bad}", flush=True)
        except Exception as e:
            print(f"[sim] could not read results: {e!r}", flush=True)
        return rc
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
