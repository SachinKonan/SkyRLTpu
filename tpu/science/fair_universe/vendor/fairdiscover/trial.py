"""One TTT-Discover code evaluation, producing scalar reward + logs.

Launch candidate in separately timed subprocesses. For untrusted model-generated
code, run this entire command INSIDE a properly isolated container/VM; Python
subprocess timeouts do not provide filesystem or network security.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .config import read_config


def trial(candidate, stage, run_dir, config=None, steps=None, batch_size=None,
          max_reward_ind=None, train_timeout=3600, reward_timeout=1800):
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    cfg = read_config(config, steps=steps, batch_size=batch_size,
                      reward_max_ind=max_reward_ind)
    if int(cfg["steps"]) <= 0 or int(cfg["batch_size"]) <= 0:
        raise ValueError("steps and batch_size must be positive")
    (run / "trial_config.json").write_text(json.dumps(cfg, indent=2))
    candidate = str(Path(candidate).resolve())
    stage = str(Path(stage).resolve())
    current_phase = "train"
    try:
        for phase, timeout in (("train", train_timeout), ("reward", reward_timeout)):
            current_phase = phase
            cmd = [sys.executable, "-m", "fairdiscover.worker", phase,
                   "--candidate", candidate, "--stage", stage,
                   "--run-dir", str(run)]
            start = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout, check=False)
            except subprocess.TimeoutExpired as err:
                raise RuntimeError(f"{phase} timed out after {timeout}s") from err
            (run / "logs" / (phase + ".stdout.txt")).write_text(proc.stdout)
            (run / "logs" / (phase + ".stderr.txt")).write_text(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(f"{phase} exited {proc.returncode}; see logs/{phase}.stderr.txt")
        return json.loads((run / "reward.json").read_text())
    except Exception as error:
        # Failed candidates get a fixed lower-than-valid score; useful for RL rollouts.
        report = {"reward": -1., "failed": True, "phase": current_phase,
                  "error": str(error), "reward_is_proxy_not_official": True}
        (run / "reward.json").write_text(json.dumps(report, indent=2))
        (run / "reward.txt").write_text("-1.0\n")
        return report


def main():
    p = argparse.ArgumentParser(description="Run a JAX FAIR Phase2 discovery trial")
    p.add_argument("--candidate", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-reward-ind", type=int, default=None)
    p.add_argument("--train-timeout", type=int, default=3600)
    p.add_argument("--reward-timeout", type=int, default=1800)
    a = p.parse_args()
    result = trial(a.candidate, a.stage, a.run_dir, a.config,
                   a.steps, a.batch_size, a.max_reward_ind,
                   a.train_timeout, a.reward_timeout)
    print(json.dumps(result, sort_keys=True))
    sys.exit(1 if result.get("failed") else 0)


if __name__ == "__main__":
    main()
