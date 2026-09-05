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
    parser.add_argument("--zone", default="asia-northeast1-b")
    parser.add_argument(
        "--storage-bucket",
        default="gs://sk7524-tinker-tpu-asia-northeast1",
    )
    parser.add_argument("--experiment-name")
    parser.add_argument("--gcs-run")
    parser.add_argument("--num-epochs", type=int, default=15)
    args = parser.parse_args()
    if not args.server:
        parser.error("--server or TPUSWARM_SERVER is required")
    if not args.storage_bucket.startswith("gs://"):
        parser.error("--storage-bucket must be a gs:// URL")
    if "/" not in args.zone:
        region = args.zone.rsplit("-", 1)[0]
    else:
        parser.error("--zone must be a zone name such as us-east5-b")

    bucket = args.storage_bucket.rstrip("/")
    gcs_run = args.gcs_run or f"{bucket}/skyrl-runs/{args.run_dir}"
    env = {
        "ZONE": args.zone,
        "GCS_RUN": gcs_run,
        "TUNIX_MAXTEXT_CKPT_CACHE_GCS": f"{bucket}/skyrl-maxtext-ckpts",
        "TUNIX_JAX_CACHE_GCS": (
            f"{bucket}/jax-compile-cache-v6e-qwen35-"
            "tp8-fsdp2-r32-s22528-b45056-v1"
        ),
        "VLLM_XLA_CACHE_GCS": (
            f"{bucket}/vllm-xla-cache-v6e-qwen35-tp4-s22528-v1"
        ),
        "HF_CACHE_GCS": f"{bucket}/hf-cache-qwen35-v1",
    }
    resources = {
        "cloud": "gcp",
        "region": region,
        "zone": args.zone,
        "accelerators": "tpu-v6e-32",
        "accelerator_args": {
            "gcp_queued_resource": True,
            "runtime_version": "v2-alpha-tpuv6e",
        },
        "use_spot": True,
        "disk_size": 150,
        "job_recovery": {
            "strategy": "FAILOVER",
            "max_restarts_on_errors": 3,
            "recover_on_exit_codes": [33, 34],
        },
    }
    task = qwen35_v6e32_grpo_task(
        task_id=args.task_id,
        run_dir=args.run_dir,
        repo_dir=args.repo_dir,
        resource_class=args.resource_class,
        experiment_name=args.experiment_name,
        num_epochs=args.num_epochs,
        env=env,
        pool=args.pool,
        resources=resources,
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
