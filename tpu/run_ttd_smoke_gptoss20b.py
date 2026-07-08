#!/usr/bin/env python3
"""Minimal ttt-discover smoke test against the real Thinking Machines Tinker API.

Goal: validate the full discover round-trip (env build -> sample from Tinker ->
local reward eval -> one train step) end-to-end at minimal cost, using the
cheapest gpt-oss model. It is NOT meant to learn anything: with group_size=1 the
within-group advantage is 0 (discover logs "All rewards are uniform. There will
be no gradient" and takes a no-op step), which is exactly what we want for a
plumbing check.

Cost knobs (all overridable via env):
  - openai/gpt-oss-20b + gpt_oss_low_reasoning renderer (fewest sampled tokens)
  - group_size=1, groups_per_batch=1  -> 1 rollout per step
  - num_epochs=1                      -> a single train step
  - small context / max tokens        -> minimal generated tokens
  - kl_penalty_coef=0                 -> skips the extra base-model logprob pass
  - eval_backend=local                -> no Ray, no SLURM (runs in-process)
"""

from __future__ import annotations

import os
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
    repo_root = Path(__file__).resolve().parents[1]
    discover_root = repo_root / "third_party" / "discover"
    sys.path.insert(0, str(discover_root))

    # Load the discover-dir .env (TINKER_API_KEY etc.) — discover itself never
    # calls load_dotenv, so we must do it explicitly.
    try:
        from dotenv import load_dotenv

        load_dotenv(discover_root / ".env")
    except Exception as exc:  # pragma: no cover - dotenv is a hard dep, but be loud
        print(f"WARNING: could not load {discover_root/'.env'}: {exc}", file=sys.stderr)

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Put it in third_party/discover/.env or export it."
        )
    # Local, Ray-free eval by default; keep it quiet on wandb.
    os.environ.setdefault("TTD_EVAL_BACKEND", "local")
    os.environ.setdefault("WANDB_MODE", "disabled")

    run_dir = Path(
        os.environ.get("TTD_RUN_DIR", repo_root / "runs" / "ttd_smoke_gptoss20b")
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover import DiscoverConfig, discover

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")
    group_size = _env_int("GROUP_SIZE", 1)
    groups_per_batch = _env_int("GROUPS_PER_BATCH", 1)
    experiment_name = os.environ.get(
        "EXPERIMENT_NAME",
        f"smoke-{model_name.replace('/', '-')}-g{group_size}-b{groups_per_batch}-{date}",
    )

    # tinker_base_url stays None so the SDK hits the real prod service.
    tinker_base_url = os.environ.get("TINKER_BASE_URL") or None

    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        model_name=model_name,
        renderer_name=os.environ.get("RENDERER_NAME", "gpt_oss_low_reasoning"),
        lora_rank=_env_int("LORA_RANK", 1),
        save_every=_env_int("SAVE_EVERY", 1000),  # effectively skip checkpoint writes
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=_env_float("LEARNING_RATE", 1e-5),
        num_epochs=_env_int("NUM_EPOCHS", 1),
        temperature=_env_float("TEMPERATURE", 1.0),
        kl_penalty_coef=_env_float("KL_PENALTY_COEF", 0.0),
        phase1_max_tokens=_env_int("PHASE1_MAX_TOKENS", 2048),
        context_window=_env_int("CONTEXT_WINDOW", 4096),
        completion_max_tokens=_env_optional_int("COMPLETION_MAX_TOKENS"),
        experiment_name=experiment_name,
        log_root=str(run_dir / "tinker_log"),
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        tinker_base_url=tinker_base_url,
        num_cpus_per_task=_env_int("NUM_CPUS_PER_TASK", 1),
        eval_backend=os.environ.get("TTD_EVAL_BACKEND", "local"),
        eval_timeout=_env_int("EVAL_TIMEOUT", 120),
    )
    print(f"[smoke] run_dir      : {run_dir}")
    print(f"[smoke] model        : {model_name}  (renderer={config.renderer_name})")
    print(f"[smoke] batch shape  : groups_per_batch={groups_per_batch}, group_size={group_size}")
    print(f"[smoke] tokens       : context={config.context_window}, phase1_max={config.phase1_max_tokens}")
    print(f"[smoke] eval_backend : {config.eval_backend}  (kl_coef={config.kl_penalty_coef})")
    print(f"[smoke] tinker_url   : {tinker_base_url or 'SDK default (prod)'}")
    print(f"[smoke] experiment   : {experiment_name}")
    discover(config)
    print("[smoke] discover() returned without raising — pipeline round-trip OK.")


if __name__ == "__main__":
    main()
