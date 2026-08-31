"""Submit the fixed mixed-v6e-32 Qwen3.5 GRPO run through TPUSwarm."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tpuswarm.client import SwarmClient

from skyrl.tpu_swarm import qwen35_v6e32_grpo_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=os.environ.get("TPUSWARM_SERVER"))
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-dir", required=True)
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
    parser.add_argument("--pool", default="tpuswarm-v6e32-asia-qwen35")
    parser.add_argument("--gcs-run")
    parser.add_argument("--num-epochs", type=int, default=15)
    args = parser.parse_args()
    if not args.server:
        parser.error("--server or TPUSWARM_SERVER is required")

    env = {} if args.gcs_run is None else {"GCS_RUN": args.gcs_run}
    task = qwen35_v6e32_grpo_task(
        task_id=args.task_id,
        run_dir=args.run_dir,
        repo_dir=args.repo_dir,
        resource_class=args.resource_class,
        num_epochs=args.num_epochs,
        env=env,
        pool=args.pool,
        secrets={"HF_TOKEN": None, "WANDB_API_KEY": None},
    )
    client = SwarmClient(
        args.server,
        bearer_token=os.environ.get("TPUSWARM_TOKEN"),
    )
    record = client.submit_task(task)
    print(record.task_id, record.status.value)


if __name__ == "__main__":
    main()
