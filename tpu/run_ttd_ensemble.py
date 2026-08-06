#!/usr/bin/env python3
"""Ensemble ttt-discover run: N generators (default gpt-oss-20b + Nemotron-3-Nano),
each with its own LoRA/GRPO/KL objective, sharing ONE PUCT state pool.

Member spec via TTD_ENSEMBLE_MODELS="model:renderer[:tag],model:renderer[:tag]".
Per-member overrides: TTD_M{i}_{LORA_RANK,PHASE1_MAX_TOKENS,COMPLETION_MAX_TOKENS,
GROUPS_PER_BATCH,LEARNING_RATE,KL_PENALTY_COEF,BASE_URL,CONTEXT_WINDOW,
TRAIN_MAX_SEQ} (i = member index, 0-based).

League mode (rl/cross_import.py): TTD_CROSS_WEIGHT (lambda, default 0.1; 0
disables) + TTD_CROSS_MAX_IMPORTS (per problem/direction, default 4).
Problem env via TTD_ENV (erdos_min_overlap | circle_packing | ac_inequalities |
frontier_erdos_ud), matching run_ttd_smoke_gptoss20b.py.
"""

from __future__ import annotations

import os
import re
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


DEFAULT_MODELS = (
    "openai/gpt-oss-20b:gpt_oss_high_reasoning:gptoss,"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:qwen3:nemotron"
)


def _default_tag(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", base.lower())[:12] or "member"


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
        raise SystemExit("TINKER_API_KEY is not set (third_party/discover/.env).")
    os.environ.setdefault("TTD_EVAL_BACKEND", "local")
    os.environ.setdefault("WANDB_MODE", "online")

    run_dir = Path(
        os.environ.get("TTD_RUN_DIR", repo_root / "runs" / "ttd_ensemble15")
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Problem env from TTD_ENV (default: Erdős), same map as the smoke runner.
    import importlib
    _ENVS = {
        "erdos_min_overlap": ("examples.erdos_min_overlap.env", "ErdosMinOverlapEnv"),
        "ac_inequalities": ("examples.ac_inequalities.env", "AutoCorrInequalityEnv"),
        "circle_packing": ("examples.circle_packing.env", "CirclePackingEnv"),
        "frontier_erdos_ud": ("examples.frontier_erdos_ud.env", "FrontierErdosUDEnv"),
        # Frontier-CS algorithmic track (ported from the frontiercs worktree);
        # problem id via TTD_PROBLEM_TYPE (46=JSSP), knobs TTD_FCALGO_*
        "frontier_algo": ("examples.frontier_algo.env", "FrontierAlgoEnv"),
    }
    _mod, _cls = _ENVS[os.environ.get("TTD_ENV", "erdos_min_overlap")]
    env_type = getattr(importlib.import_module(_mod), _cls)

    from ttt_discover.rl.ensemble import (
        EnsembleConfig,
        EnsembleMemberConfig,
        ensemble_discover,
    )

    members = []
    for i, spec in enumerate(os.environ.get("TTD_ENSEMBLE_MODELS", DEFAULT_MODELS).split(",")):
        parts = spec.strip().split(":")
        model, renderer = parts[0], parts[1]
        tag = parts[2] if len(parts) > 2 else _default_tag(model)
        is_gptoss = model.startswith("openai/gpt-oss")
        is_gemma = "gemma" in model.lower()
        members.append(EnsembleMemberConfig(
            model_name=model,
            renderer_name=renderer,
            metric_prefix=tag,
            lora_rank=_env_int(f"TTD_M{i}_LORA_RANK", _env_int("LORA_RANK", 32)),
            phase1_max_tokens=_env_int(f"TTD_M{i}_PHASE1_MAX_TOKENS",
                                       _env_int("PHASE1_MAX_TOKENS", 26000)),
            # Nemotron/ChatML has no phase-2 forcing: a hard cap turns long thinking
            # into truncated zero-reward samples, so default None (remaining context).
            completion_max_tokens=(None if is_gptoss
                                   else _env_optional_int(f"TTD_M{i}_COMPLETION_MAX_TOKENS")),
            # Nemotron's HF chat template emits an empty leading system turn;
            # qwen3.5 proper runs without one (matches the single-model sweep),
            # and Gemma4Renderer injects its own <|think|> system turn.
            convo_prefix=([{"role": "system", "content": ""}]
                          if "nemotron" in model.lower() else None),
            groups_per_batch=_env_int(f"TTD_M{i}_GROUPS_PER_BATCH",
                                      _env_int("GROUPS_PER_BATCH", 32)),
            learning_rate=_env_float(f"TTD_M{i}_LEARNING_RATE",
                                     _env_float("LEARNING_RATE", 4e-5)),
            kl_penalty_coef=_env_float(f"TTD_M{i}_KL_PENALTY_COEF",
                                       _env_float("KL_PENALTY_COEF", 0.1)),
            tinker_base_url=os.environ.get(f"TTD_M{i}_BASE_URL") or None,
            context_window=_env_optional_int(f"TTD_M{i}_CONTEXT_WINDOW"),
            train_max_seq=_env_optional_int(f"TTD_M{i}_TRAIN_MAX_SEQ"),
        ))

    if len(members) < 2:
        raise SystemExit(
            f"TTD_ENSEMBLE_MODELS parsed to {len(members)} member(s): "
            f"{[m.model_name for m in members]}. An ensemble needs >=2. NOTE: if you "
            "passed this var inside 'sbatch --export=ALL,...', sbatch split it on "
            "commas — export it into the shell instead and rely on --export=ALL."
        )

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = os.environ.get(
        "EXPERIMENT_NAME", f"erdos-ensemble-{'-'.join(m.metric_prefix for m in members)}-{date}"
    )
    log_path = str(run_dir / "tinker_log" / experiment_name)

    cfg = EnsembleConfig(
        members=members,
        env_type=env_type,
        log_path=log_path,
        group_size=_env_int("GROUP_SIZE", 8),
        num_epochs=_env_int("NUM_EPOCHS", 15),
        temperature=_env_float("TEMPERATURE", 1.0),
        context_window=_env_int("CONTEXT_WINDOW", 32768),
        save_every=_env_int("SAVE_EVERY", 0),
        loss_fn=os.environ.get("TTD_LOSS_FN", "importance_sampling"),
        adv_estimator=os.environ.get("TTD_ADV_ESTIMATOR", "entropic_adaptive_beta"),
        adv_estimator_beta=_env_float("TTD_ADV_ESTIMATOR_BETA", 2.0),
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        wandb_name=experiment_name,
        tinker_base_url=os.environ.get("TINKER_BASE_URL") or None,
        num_cpus_per_task=_env_int("NUM_CPUS_PER_TASK", 1),
        problem_type=os.environ.get("TTD_PROBLEM_TYPE", ""),
        eval_backend=os.environ.get("TTD_EVAL_BACKEND", "local"),
        eval_timeout=_env_int("EVAL_TIMEOUT", 1100),
        # League cross-imitation (frontier-gated imports; lambda=0 disables)
        cross_import_weight=_env_float("TTD_CROSS_WEIGHT", 0.1),
        cross_max_imports_per_seed=_env_int("TTD_CROSS_MAX_IMPORTS", 4),
        # v2 dataflow executor (stream per-group fb + per-problem cross-CE)
        pipeline_dataflow=os.environ.get("TTD_LEAGUE_PIPELINE", "0") == "1",
        # Cross-model contrastive distillation (symmetric; off by default)
        distill_enabled=os.environ.get("TTD_DISTILL_ENABLED", "0") == "1",
        distill_pairs_per_step=_env_int("TTD_DISTILL_PAIRS_PER_STEP", 16),
        distill_num_betters=_env_int("TTD_DISTILL_NUM_BETTERS", 3),
        distill_weight=_env_float("TTD_DISTILL_WEIGHT", 0.1),
        distill_grounding_filter=os.environ.get("TTD_DISTILL_GROUNDING_FILTER", "1") == "1",
        distill_leak_filter=os.environ.get("TTD_DISTILL_LEAK_FILTER", "1") == "1",
        distill_teacher_phase1_tokens=_env_int("TTD_DISTILL_TEACHER_PHASE1_TOKENS", 0),
        distill_max_target_tokens=_env_int("TTD_DISTILL_MAX_TARGET_TOKENS", 8192),
        distill_max_code_chars=_env_int("TTD_DISTILL_MAX_CODE_CHARS", 10000),
    )

    print(f"[ensemble] run_dir    : {run_dir}")
    print(f"[ensemble] env        : {env_type.__name__}  league lambda={cfg.cross_import_weight} "
          f"max_imports={cfg.cross_max_imports_per_seed} "
          f"pipeline={int(cfg.pipeline_dataflow)} eval_backend={cfg.eval_backend}")
    for i, m in enumerate(members):
        print(f"[ensemble] member {i}   : {m.model_name} renderer={m.renderer_name} "
              f"tag={m.metric_prefix} groups={m.groups_per_batch} lr={m.learning_rate} "
              f"kl={m.kl_penalty_coef} phase1={m.phase1_max_tokens} "
              f"cmax={m.completion_max_tokens} base_url={m.tinker_base_url or 'shared'} "
              f"ctx={m.context_window or cfg.context_window} tms={m.train_max_seq}")
    print(f"[ensemble] shared     : group_size={cfg.group_size} steps={cfg.num_epochs} "
          f"ctx={cfg.context_window} eval={cfg.eval_backend}@{cfg.eval_timeout}s "
          f"save_every={cfg.save_every}")
    print(f"[ensemble] distill    : enabled={cfg.distill_enabled} weight={cfg.distill_weight} "
          f"pairs={cfg.distill_pairs_per_step} betters={cfg.distill_num_betters} "
          f"elite_slots={os.environ.get('TTD_ELITE_SLOTS', '0')}")
    print(f"[ensemble] experiment : {experiment_name}")
    ensemble_discover(cfg)
    print("[ensemble] done — loop returned without raising.")


if __name__ == "__main__":
    main()
