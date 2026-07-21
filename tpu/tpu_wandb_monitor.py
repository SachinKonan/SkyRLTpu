#!/usr/bin/env python3
"""TPU utilization -> wandb sidecar for the qwen35 tinker server hosts.

Samples per-chip HBM usage and TensorCore duty cycle every INTERVAL seconds
via the tpu_info library (the same libtpu SDK-monitoring gRPC service the
tpu-info CLI reads) and logs them to a dedicated wandb run per host, in the
same project as the training runs so the charts sit side by side.

Env: WANDB_API_KEY (required), TPUMON_RUN_NAME (e.g. qwen35-tpu-w0-trainer),
     TPUMON_PROJECT (default ttt-discover-gptoss20b), TPUMON_INTERVAL (30s).
Run:  ~/tpumon/bin/python tpu_wandb_monitor.py
"""
from __future__ import annotations

import os
import subprocess
import time


def sample_via_lib():
    """Preferred: tpu_info python API. Returns list of (used_gib, total_gib, duty_pct)."""
    from tpu_info import device, metrics  # type: ignore

    chip_type, _count = device.get_local_chips()
    out = []
    for u in metrics.get_chip_usage(chip_type):
        out.append((u.memory_usage / 2**30, u.total_memory / 2**30, float(u.duty_cycle_pct)))
    return out


def sample_via_cli():
    """Fallback: parse `tpu-info` table rows like
    '| 0    | 38.31 GiB / 95.74 GiB | 12.00%     |'."""
    txt = subprocess.run(
        [os.path.expanduser("~/tpumon/bin/tpu-info")], capture_output=True, text=True, timeout=30
    ).stdout
    out = []
    for line in txt.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[1].isdigit() and "GiB" in parts[2] and "%" in parts[3]:
            mem = parts[2].replace("GiB", "").split("/")
            out.append((float(mem[0]), float(mem[1]), float(parts[3].rstrip("%"))))
    return out


def main() -> None:
    import wandb

    interval = float(os.environ.get("TPUMON_INTERVAL", "30"))
    run = wandb.init(
        project=os.environ.get("TPUMON_PROJECT", "ttt-discover-gptoss20b"),
        name=os.environ.get("TPUMON_RUN_NAME", f"tpumon-{os.uname().nodename}"),
        config={"host": os.uname().nodename, "interval_s": interval},
        settings=wandb.Settings(console="off"),
    )
    print(f"[tpumon] logging to {run.url}", flush=True)

    failures = 0
    while True:
        try:
            try:
                chips = sample_via_lib()
            except Exception:
                chips = sample_via_cli()
            if chips:
                log = {}
                for i, (used, total, duty) in enumerate(chips):
                    log[f"tpu/chip{i}/hbm_used_gib"] = used
                    log[f"tpu/chip{i}/hbm_pct"] = 100.0 * used / max(total, 1e-9)
                    log[f"tpu/chip{i}/duty_cycle_pct"] = duty
                log["tpu/hbm_used_gib_max"] = max(u for u, _, _ in chips)
                log["tpu/hbm_pct_max"] = max(100.0 * u / max(t, 1e-9) for u, t, _ in chips)
                log["tpu/duty_cycle_pct_mean"] = sum(d for _, _, d in chips) / len(chips)
                run.log(log)
                failures = 0
        except Exception as e:  # keep the sidecar alive through blips
            failures += 1
            print(f"[tpumon] sample failed ({failures}): {e}", flush=True)
            if failures > 60:
                raise
        time.sleep(interval)


if __name__ == "__main__":
    main()
