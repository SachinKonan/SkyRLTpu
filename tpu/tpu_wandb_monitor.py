#!/usr/bin/env python3
"""TPU utilization -> wandb sidecar for the qwen35 tinker server hosts.

Samples per-chip HBM usage and TensorCore duty cycle every INTERVAL seconds
via the tpu_info library (the same libtpu SDK-monitoring gRPC service the
tpu-info CLI reads) and logs them INTO the training runs themselves: wandb
"shared mode" lets this process attach as a secondary writer to runs owned
by the training clients (which init with mode="shared" via TTD_WANDB_SHARED=1),
so tpu/* charts sit next to env/* on the same run page. One sidecar per TPU
host; metric keys carry the host tag (tpu/w0/..., tpu/w1/...).

Env: WANDB_API_KEY (required)
     TPUMON_ATTACH_IDS  comma-separated wandb run ids to write into (required)
     TPUMON_HOST_TAG    e.g. w0-trainer / w1-vllm (required)
     TPUMON_PROJECT     default tpu-tinker-exps
     TPUMON_INTERVAL    default 30 (seconds)
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
    project = os.environ.get("TPUMON_PROJECT", "tpu-tinker-exps")
    tag = os.environ["TPUMON_HOST_TAG"]
    attach_ids = [x for x in os.environ["TPUMON_ATTACH_IDS"].split(",") if x]

    runs = []
    for rid in attach_ids:
        run = wandb.init(
            project=project,
            id=rid,
            reinit="create_new",
            settings=wandb.Settings(
                mode="shared", x_primary=False, x_update_finish_state=False, console="off"
            ),
        )
        # tpu/* series get their own x-axis (sample index), independent of the
        # primary writer's RL-step axis.
        run.define_metric(f"tpu/{tag}/sample")
        run.define_metric(f"tpu/{tag}/*", step_metric=f"tpu/{tag}/sample")
        runs.append(run)
        print(f"[tpumon:{tag}] attached to {run.url}", flush=True)

    failures = 0
    sample_idx = 0
    while True:
        try:
            try:
                chips = sample_via_lib()
            except Exception:
                chips = sample_via_cli()
            if chips:
                log = {f"tpu/{tag}/sample": sample_idx}
                for i, (used, total, duty) in enumerate(chips):
                    log[f"tpu/{tag}/chip{i}/hbm_used_gib"] = used
                    log[f"tpu/{tag}/chip{i}/hbm_pct"] = 100.0 * used / max(total, 1e-9)
                    log[f"tpu/{tag}/chip{i}/duty_cycle_pct"] = duty
                log[f"tpu/{tag}/hbm_used_gib_max"] = max(u for u, _, _ in chips)
                log[f"tpu/{tag}/hbm_pct_max"] = max(100.0 * u / max(t, 1e-9) for u, t, _ in chips)
                log[f"tpu/{tag}/duty_cycle_pct_mean"] = sum(d for _, _, d in chips) / len(chips)
                for run in runs:
                    run.log(log)
                sample_idx += 1
                failures = 0
        except Exception as e:  # keep the sidecar alive through blips
            failures += 1
            print(f"[tpumon:{tag}] sample failed ({failures}): {e}", flush=True)
            if failures > 60:
                raise
        time.sleep(interval)


if __name__ == "__main__":
    main()
