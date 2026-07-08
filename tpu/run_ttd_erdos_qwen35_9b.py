#!/usr/bin/env python3
from __future__ import annotations

import os
import ssl
import sys
from datetime import datetime
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def main() -> None:
    # Force OpenSSL's one-time lazy init to run on the main thread. The tinker
    # client builds its first ssl.SSLContext inside a worker thread under a
    # running asyncio loop; if that is OpenSSL's first SSL_CTX_new in the
    # process it races the one-time provider/config init and dies with
    # "ssl.SSLError: unknown error (_ssl.c:3108)". Creating one context here
    # first makes the init deterministic.
    ssl.create_default_context()

    repo_root = Path(__file__).resolve().parents[1]
    discover_root = repo_root / "third_party" / "discover"
    sys.path.insert(0, str(discover_root))

    run_dir = Path(os.environ.get("TTD_RUN_DIR", repo_root / "runs" / "ttd_erdos")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")

    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover import DiscoverConfig, discover

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
    group_size = _env_int("GROUP_SIZE", 16)
    groups_per_batch = _env_int("GROUPS_PER_BATCH", 64)
    learning_rate = _env_float("LEARNING_RATE", 2e-5)
    experiment_name = os.environ.get(
        "EXPERIMENT_NAME",
        (
            "erdos-min-overlap-"
            f"{model_name.replace('/', '-')}-32rank-"
            f"{learning_rate:g}lr-{group_size}group-{groups_per_batch}batch-"
            f"32k-seed{os.environ.get('SEED', '0')}-{date}"
        ),
    )

    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        model_name=model_name,
        renderer_name=os.environ.get("RENDERER_NAME", "qwen3"),
        lora_rank=_env_int("LORA_RANK", 32),
        save_every=_env_int("SAVE_EVERY", 2),
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=learning_rate,
        num_epochs=_env_int("NUM_EPOCHS", 180),
        temperature=_env_float("TEMPERATURE", 1.0),
        kl_penalty_coef=_env_float("KL_PENALTY_COEF", 0.1),
        phase1_max_tokens=_env_int("PHASE1_MAX_TOKENS", 26000),
        context_window=_env_int("CONTEXT_WINDOW", 32768),
        completion_max_tokens=_env_optional_int("COMPLETION_MAX_TOKENS"),
        experiment_name=experiment_name,
        log_root=str(run_dir / "tinker_log"),
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        tinker_base_url=os.environ.get("TINKER_BASE_URL", "http://127.0.0.1:18000"),
        num_cpus_per_task=_env_int("NUM_CPUS_PER_TASK", 1),
        eval_backend=os.environ.get("TTD_EVAL_BACKEND", "submitit"),
        eval_timeout=_env_int("EVAL_TIMEOUT", 1100),
    )
    print(f"Running Erdos minimum overlap discover in {run_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Tinker API: {config.tinker_base_url}")
    discover(config)


if __name__ == "__main__":
    main()
