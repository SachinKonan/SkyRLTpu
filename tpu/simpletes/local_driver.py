"""SimpleTES on the RQ4 farms, run from della as the child of ``local_notrain --driver simpletes``.

The runner provides what the RQ4 frozen client gets: a local remote-only ingress that
leases this model's inference farms (``TTD_FROZEN_BASE_URL``) and the farm grading
transport (``SKYRL_GRADING_URL``, no local slots). This process renders prompts with the
run's RL renderer, generates through the ingress with the RQ4 caps, and grades every
candidate through ``Environment._safe_grade`` of the run's RQ4 environment.

Usage (by the runner): python -m tpu.simpletes.local_driver --config <profile.json>
    --spec <simpletes spec.json> --snapshot <tokenizer snapshot>
Outputs: <profile root>/simpletes/ (run.json, checkpoints/, llm_calls.jsonl, summary.json).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

# Transition the server forces when thinking hits its budget (completers.*TwoPhaseTokenCompleter).
COMPLETERS = {"qwen": "QwenTwoPhaseTokenCompleter", "gemma": "GemmaTwoPhaseTokenCompleter",
              "muse": "MuseTwoPhaseTokenCompleter"}
ENVS = {"qubit": "science_routing", "circuit": "science_placement", "erdos": "erdos_min_overlap"}


async def in_generation_phase(work, expected_n: int, phase=None):
    """Run ``work()`` inside one borrowing phase, as the RQ4 client wraps its sampling.

    The remote-only ingress holds /v1/completions until a client opens a phase
    (/skyrl/v1/borrowing/begin; SKYRL_BORROWING_BASE_MODEL=1 makes it a base-model phase).
    SimpleTES generates continuously rather than in steps, so the whole search is one phase,
    kept alive by the helper's heartbeat and ended on exit.
    """
    if phase is None:
        from tpu.swarm.ray_train.borrowing_phase import sampling_phase as phase
    async with phase(expected_n=expected_n):
        return await work()


def transition_tokens(tokenizer, family: str) -> int:
    from ttt_discover.tinker_utils import completers
    completer = getattr(completers, COMPLETERS[family])
    return len(tokenizer.encode(completer.THINK_CLOSE + completer.ANSWER_CUE, add_special_tokens=False))


async def run(config, spec: dict, snapshot: str) -> int:
    from transformers import AutoTokenizer
    from tpu.swarm.ray_train.frozen_benchmark import check_choice
    from .backend import Caps, NativeThinkingBackend, family_of, make_renderer
    from .driver import event, prepare_run, save_json, search, write_inputs
    from .evaluator import RLEnvironmentEvaluator
    from .tasks import get_task

    task = get_task(spec["task"])
    if ENVS.get(task.name) != config.client_env.get("TTD_ENV"):
        raise ValueError(f"SimpleTES task {task.name} does not match the profile's TTD_ENV "
                         f"{config.client_env.get('TTD_ENV')}")
    folder = Path(config.root).expanduser() / "simpletes"
    folder.mkdir(parents=True, exist_ok=True)
    renderer_name = config.client_member_spec.split(":")[1]
    identity, root = prepare_run(folder, spec, task, config.model, config.client_member_spec)
    # Science tasks grade inside their own reward (farm transport); math tasks take the run's
    # grading backend, CPUs and timeout exactly as the frozen client does.
    math_grading = {} if task.science else dict(
        eval_backend=os.environ.get("TTD_EVAL_BACKEND"),
        num_cpus_per_task=int(os.environ.get("NUM_CPUS_PER_TASK", task.num_cpus_per_task)),
        eval_timeout=int(os.environ.get("EVAL_TIMEOUT", task.eval_timeout)))
    evaluator = RLEnvironmentEvaluator(task, root, log_dir=folder / "grading", **math_grading)
    write_inputs(folder, task, root, evaluator)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    family = family_of(renderer_name)
    renderer = make_renderer(renderer_name, tokenizer, identity["renderer_date"])
    fixed = config.client_env.get("TTD_FIXED_THINK_TOKENS")
    caps = Caps(context_window=int(config.client_context_window),
                phase1_max_tokens=int(config.client_phase1_max_tokens),
                fixed_thinking=int(fixed) if fixed else None,
                fixed_total=int(config.client_env["TTD_FIXED_TOTAL_TOKENS"]) if fixed else None,
                fixed_fit=config.client_env.get("TTD_FIXED_BUDGET_FIT") == "1")
    backend = NativeThinkingBackend(base_url=os.environ["TTD_FROZEN_BASE_URL"], model=config.model,
                                    renderer=renderer, family=family, caps=caps,
                                    log_path=folder / "llm_calls.jsonl",
                                    request_timeout=config.inference.request_timeout,
                                    check_choice=check_choice,
                                    transition_tokens=transition_tokens(tokenizer, family))
    save_json(folder / "caps.json", dict(context_window=caps.context_window,
                                          phase1_max_tokens=caps.phase1_max_tokens,
                                          context_buffer=caps.context_buffer,
                                          fixed_thinking=caps.fixed_thinking, fixed_total=caps.fixed_total,
                                          transition_tokens=backend.transition_tokens,
                                          renderer=renderer_name, renderer_date=identity["renderer_date"]))

    async def work():
        event("simpletes_phase_open", run_id=config.run_id)
        warm = await backend.generate_text("Reply with the word ready.", thinking_tokens=8, answer_tokens=64,
                                           temperature=1.0, kind="preflight")
        event("simpletes_preflight_passed", run_id=config.run_id, reply=warm[:40])
        return await search(folder, spec, task, evaluator, backend, caps, config.model, config.run_id,
                            seed=int(spec.get("seed", config.bootstrap_seed or 0)))
    return await in_generation_phase(work, int(spec["k_candidates"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    # The shared grading pool is sized when ttt_discover is first imported.
    os.environ["TTD_SAFE_GRADE_MAX_WORKERS"] = str(spec.get("eval_concurrency", 512))
    from tpu.swarm.ray_train.config import Config
    config = Config.load(args.config)
    if not (config.inference.remote_only and config.client_env.get("TTD_FROZEN_POLICY") == "1"):
        raise SystemExit("SimpleTES local runs need a remote-only frozen (base model) profile")
    return asyncio.run(run(config, spec, args.snapshot))


if __name__ == "__main__":
    raise SystemExit(main())
