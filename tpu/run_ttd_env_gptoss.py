#!/usr/bin/env python3
"""Generic ttt-discover entrypoint against the real Tinker API.

Same env-var contract as tpu/run_ttd_smoke_gptoss20b.py, plus:
  TTD_ENV          which example environment to run:
                   erdos | circle_packing | ac_inequalities
  TTD_PROBLEM_TYPE env-specific problem id (circle_packing: "26"/"32";
                   ac_inequalities: "ac1"/"ac2"; erdos: "")

Defaults preserve the gpt-oss-20b canonical config used for the Erdos runs.
"""

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


ENV_REGISTRY = {
    # name -> (module path, class name, default problem_type, default eval_timeout)
    "erdos": ("examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "", 1100),
    "circle_packing": ("examples.circle_packing.env", "CirclePackingEnv", "26", 530),
    "ac_inequalities": ("examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac2", 1100),
    # Frontier-CS 2.0 planar unit-distance problem; problem_type = N points
    # ("65536" real task, "10" mirrors their erdos_demo smoke variant).
    "frontier_erdos_ud": ("examples.frontier_erdos_ud.env", "FrontierErdosUDEnv", "65536", 1100),
    # Pallas kernel arena, RG-LRU: remote TPU judging via ARENA_QUEUE_URL.
    # eval_timeout is the QUEUE-WAIT bound, not a sandbox budget (nothing runs
    # locally); a 32-group on a few judges is queue-depth dominated.
    "pallas_rglru": ("examples.pallas_rglru.env", "PallasRgLruEnv", "", 3600),
}


def main() -> None:
    # OpenSSL first-init on the main thread (see run_ttd_smoke_gptoss20b.py).
    ssl.create_default_context()

    repo_root = Path(__file__).resolve().parents[1]
    discover_root = repo_root / "third_party" / "discover"
    sys.path.insert(0, str(discover_root))

    try:
        from dotenv import load_dotenv

        load_dotenv(discover_root / ".env")
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not load {discover_root/'.env'}: {exc}", file=sys.stderr)

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set.")

    os.environ.setdefault("TTD_EVAL_BACKEND", "local")
    os.environ.setdefault("WANDB_MODE", "disabled")

    env_name = os.environ.get("TTD_ENV", "erdos")
    if env_name not in ENV_REGISTRY:
        raise SystemExit(f"Unknown TTD_ENV={env_name!r}; options: {sorted(ENV_REGISTRY)}")
    module_path, class_name, default_ptype, default_timeout = ENV_REGISTRY[env_name]

    import importlib

    env_type = getattr(importlib.import_module(module_path), class_name)
    problem_type = os.environ.get("TTD_PROBLEM_TYPE", default_ptype)

    run_dir = Path(
        os.environ.get("TTD_RUN_DIR", repo_root / "runs" / f"ttd_{env_name}_gptoss")
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    from ttt_discover import DiscoverConfig, discover

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")
    group_size = _env_int("GROUP_SIZE", 8)
    groups_per_batch = _env_int("GROUPS_PER_BATCH", 64)
    experiment_name = os.environ.get(
        "EXPERIMENT_NAME",
        f"{env_name}{problem_type}-{model_name.replace('/', '-')}-{date}",
    )
    tinker_base_url = os.environ.get("TINKER_BASE_URL") or None

    config = DiscoverConfig(
        env_type=env_type,
        problem_type=problem_type,
        model_name=model_name,
        renderer_name=os.environ.get("RENDERER_NAME", "gpt_oss_high_reasoning"),
        lora_rank=_env_int("LORA_RANK", 32),
        save_every=_env_int("SAVE_EVERY", 0),
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=_env_float("LEARNING_RATE", 4e-5),
        num_epochs=_env_int("NUM_EPOCHS", 15),
        temperature=_env_float("TEMPERATURE", 1.0),
        kl_penalty_coef=_env_float("KL_PENALTY_COEF", 0.1),
        phase1_max_tokens=_env_int("PHASE1_MAX_TOKENS", 26000),
        context_window=_env_int("CONTEXT_WINDOW", 32768),
        completion_max_tokens=_env_optional_int("COMPLETION_MAX_TOKENS"),
        experiment_name=experiment_name,
        log_root=str(run_dir / "tinker_log"),
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        tinker_base_url=tinker_base_url,
        num_cpus_per_task=_env_int("NUM_CPUS_PER_TASK", 1),
        eval_backend=os.environ.get("TTD_EVAL_BACKEND", "local"),
        eval_timeout=_env_int("EVAL_TIMEOUT", default_timeout),
        distill_enabled=os.environ.get("TTD_DISTILL_ENABLED", "0") == "1",
        distill_pairs_per_step=_env_int("TTD_DISTILL_PAIRS_PER_STEP", 16),
        distill_num_betters=_env_int("TTD_DISTILL_NUM_BETTERS", 3),
        distill_weight=_env_float("TTD_DISTILL_WEIGHT", 0.1),
        distill_grounding_filter=os.environ.get("TTD_DISTILL_GROUNDING_FILTER", "1") == "1",
        distill_leak_filter=os.environ.get("TTD_DISTILL_LEAK_FILTER", "1") == "1",
        distill_teacher_phase1_tokens=_env_int("TTD_DISTILL_TEACHER_PHASE1_TOKENS", 0),
        distill_max_target_tokens=_env_int("TTD_DISTILL_MAX_TARGET_TOKENS", 8192),
        distill_max_code_chars=_env_int("TTD_DISTILL_MAX_CODE_CHARS", 10000),
        distill_teacher_model=os.environ.get("TTD_DISTILL_TEACHER_MODEL") or None,
    )
    print(f"[ttd] env          : {env_name} (problem_type={problem_type!r})")
    print(f"[ttd] run_dir      : {run_dir}")
    print(f"[ttd] model        : {model_name}  (renderer={config.renderer_name})")
    print(f"[ttd] batch shape  : groups_per_batch={groups_per_batch}, group_size={group_size}")
    print(f"[ttd] eval         : backend={config.eval_backend} timeout={config.eval_timeout}")
    print(f"[ttd] distill      : enabled={config.distill_enabled} weight={config.distill_weight} "
          f"teacher={config.distill_teacher_model or 'policy'}")
    print(f"[ttd] elite slots  : {os.environ.get('TTD_ELITE_SLOTS', '0')}")
    print(f"[ttd] experiment   : {experiment_name}")
    discover(config)
    print("[ttd] discover() returned without raising — run complete.")


if __name__ == "__main__":
    main()
