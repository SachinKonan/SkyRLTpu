#!/usr/bin/env python3
"""Submit the GPT-OSS 120B v6e-32 acceptance gate through TPUSwarm."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tpuswarm.client import SwarmClient

from skyrl.tpu_swarm import gptoss120b_v6e32_smoke_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=os.environ.get("TPUSWARM_SERVER"))
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--result-gcs")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TPUSWARM_REMOTE_SKYRL_ROOT",
                "/home/sk7524_princeton_edu/SkyRLTpu-tpuswarm",
            )
        ),
    )
    parser.add_argument("--resource-class", default="gcp-tpu-v6e-32-asia")
    parser.add_argument("--pool", default="tpuswarm-v6e32-asia-gptoss120b")
    args = parser.parse_args()
    if not args.server:
        parser.error("--server or TPUSWARM_SERVER is required")

    task = gptoss120b_v6e32_smoke_task(
        task_id=args.task_id,
        run_name=args.run_name,
        result_gcs=args.result_gcs,
        repo_dir=args.repo_dir,
        resource_class=args.resource_class,
        pool=args.pool,
        secrets={"HF_TOKEN": None},
    )
    client = SwarmClient(
        args.server,
        bearer_token=os.environ.get("TPUSWARM_TOKEN"),
    )
    record = client.submit_task(task)
    print(record.task_id, record.status.value)


if __name__ == "__main__":
    main()
