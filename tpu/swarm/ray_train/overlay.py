"""Explicit source overlays for opt-in features, isolated by digest."""
import hashlib
import json
from pathlib import Path
import shutil

FILES = (
    "skyrl/backends/tunix_backend.py",
    "skyrl/backends/lora_init.py",
    "skyrl/backends/stacked_lora.py",
    "skyrl/tinker/loss_fns.py",
    "skyrl/tinker/types.py",
    "skyrl/tinker/api.py",
    "skyrl/tinker/engine.py",
    "tpu/run_ttd_ensemble.py",
    "tpu/vllm_tpu_server.py",
    "third_party/discover/ttt_discover/rl/ensemble.py",
    "third_party/discover/ttt_discover/rl/multi_lora.py",
    "third_party/discover/ttt_discover/rl/multi_lora_request.py",
    "third_party/discover/ttt_discover/tinker_utils/completers.py",
)

ARENA_FILES = (
    "tpu/run_ttd_ensemble.py",
    # The frozen client's ray_train package shadows the executor bundle's
    # package. Ship task definitions/cleanup plus legacy actor compatibility
    # so both drivers and workers import the same grading implementation.
    "tpu/swarm/ray_train/__init__.py",
    "tpu/swarm/ray_train/grader_actor.py",
    "tpu/swarm/ray_train/grader_tasks.py",
    "tpu/swarm/ray_train/grader_child.py",
    "tpu/swarm/ray_train/process.py",
    "tpu/pallas_arena/__init__.py",
    "tpu/pallas_arena/judge/__init__.py",
    "tpu/pallas_arena/judge/client.py",
    "tpu/pallas_arena/judge/collect.py",
    "tpu/pallas_arena/judge/timing.py",
    "tpu/pallas_arena/judge/observation.py",
    "tpu/pallas_arena/judge/problems/rg_lru.py",
    "tpu/pallas_arena/rl/__init__.py",
    "tpu/pallas_arena/rl/task.py",
    "tpu/pallas_arena/rl/env.py",
    "tpu/pallas_arena/rl/seed_rglru.py",
)

SMOKE_FILES = (
    'tpu/run_ttd_ensemble.py',
    'third_party/discover/ttt_discover/rl/ensemble.py',
    'third_party/discover/ttt_discover/tinker_utils/completers.py',
)

PROBLEM_PROMPT_FILES = {
    "ac_inequalities": "third_party/discover/examples/ac_inequalities/env.py",
    "circle_packing": "third_party/discover/examples/circle_packing/env.py",
}

NATIVE_FILES = {"skyrl/tinker/dispatch.py", "skyrl/tinker/extra/external_inference.py",
                "skyrl/tinker/extra/skyrl_train_inference_forwarding.py", "skyrl/tinker/types.py", "skyrl/tinker/api.py", "skyrl/backends/vllm_sampling.py",
                "skyrl/backends/native_completion.py", "third_party/discover/ttt_discover/rl/train.py",
                "third_party/discover/ttt_discover/tinker_utils/completers.py"}

ADAPTIVE_PWC_FILE = "third_party/discover/ttt_discover/rl/train.py"
ANSWER_ONLY_FILE = "third_party/discover/ttt_discover/tinker_utils/dataset_builder.py"
DATABASE_FILES = {"skyrl/tinker/db_models.py"}
REPEATED_KV_FILES = {"skyrl/backends/tunix_backend.py", "skyrl/backends/lora_init.py"}
WARMUP_FILES = REPEATED_KV_FILES | {"skyrl/backends/backward_warmup.py", "tpu/swarm/ray_train/warmup_contract.py"}

SCIENCE_FILES = {'tpu/run_ttd_ensemble.py'} | {
    'tpu/science/' + name for name in (
        '__init__.py', 'bootstrap.py', 'seed_pool.py', 'training_env.py', 'feedback.py', 'training_setup.py', 'ray_cpu.py', 'worker.py',
        'routing.py', 'contracts.py', 'rewards.py', 'isolation.py', 'cgroup_limits.py', 'cpu_slots.py',
        'placement_ray.py', 'placement_slots.py', 'placement_task.py', 'challenge_contract.py',
        'seed_routing.py', 'challenge_seed.py', 'challenge_seed_jax.py', 'requirements-cpu.lock',
        'prompts/rendered/routing.txt', 'prompts/placement-jax-v6e.txt')}


def manifest(repo, config=None):
    names = set(FILES) if config is None or config.adapter_count > 1 else set()
    if config is not None and config.inference.hosts_per_engine > 1:
        names.add("tpu/vllm_tpu_server.py")
    # The updated backend imports the helper only for an explicitly enabled
    # warmup. Always ship the helper whenever that backend is selected.
    if config is not None and config.trainer.backward_warmup:
        names.update(WARMUP_FILES)
    if config is None or not config.inference_only:
        names.update(DATABASE_FILES)
    if config is not None and config.is_recurrent_gemma:
        names.update(ARENA_FILES)
        names.update(SMOKE_FILES)  # Propagate fatal grader errors through the training loop.
    if config is not None and config.training_smoke:
        names.update(SMOKE_FILES)
    if config is not None and config.science_task:
        names.update(SCIENCE_FILES | set(SMOKE_FILES))
    native_kv_heads = {'qwen3.5-27b': 4, 'muse-glimmer-30b': 2}
    if (config is not None and not config.inference_only
            and (config.attention_replay
                 or config.science_task and config.model_preset == "gemma4-31b"
                 or config.model_preset in native_kv_heads
                 and config.trainer.logical_kv_heads > native_kv_heads[config.model_preset])):
        # Gemma science runs need the same distributed-input backend as their
        # validated replay, even though they do not repeat native KV heads.
        names.update(REPEATED_KV_FILES)
    if config is not None and config.has_problem_prompt_overlay:
        names.add(PROBLEM_PROMPT_FILES[config.client_env["TTD_ENV"]])
    if config is not None and config.has_adaptive_pwc_overlay:
        names.add(ADAPTIVE_PWC_FILE)
    if config is not None and config.has_answer_only_overlay:
        names.add(ANSWER_ONLY_FILE)
        # Carry the validated Ray pipeline/shutdown-compatible ensemble path
        # into full runs as well as one-step smokes.
        names.update(SMOKE_FILES)
    if config is not None and config.inference.native_thinking_budget and not config.inference_only:
        names.update(NATIVE_FILES | set(SMOKE_FILES))
    return {name: hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest() for name in sorted(names)}


def identity(directory):
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise RuntimeError("this feature requires a built, pinned source overlay")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(directory, destination):
    directory, destination = Path(directory), Path(destination)
    records = json.loads((directory / "manifest.json").read_text())
    allowed = [set(FILES), set(ARENA_FILES), set(FILES) | set(ARENA_FILES),
               set(SMOKE_FILES), set(SMOKE_FILES) | set(ARENA_FILES)]
    # Each mathematical task gets only its own prompt file. Erdős retains
    # the pinned base-bundle prompt and does not receive either override.
    for name in PROBLEM_PROMPT_FILES.values():
        allowed.extend([base | {name} for base in [set(), *allowed[:5]]])
    allowed.extend([base | {ADAPTIVE_PWC_FILE} for base in [set(), *allowed]])
    allowed.extend([base | {ANSWER_ONLY_FILE} for base in [set(), *allowed]])
    allowed.extend([base | NATIVE_FILES | set(SMOKE_FILES) for base in [set(), *allowed]])
    allowed.extend([base | DATABASE_FILES for base in [set(), *allowed]])
    allowed.extend([base | SCIENCE_FILES | set(SMOKE_FILES) for base in [set(), *allowed]])
    allowed.extend([base | REPEATED_KV_FILES for base in [set(), *allowed]])
    allowed.extend([base | WARMUP_FILES for base in [set(), *allowed]])
    allowed.extend([base | {"tpu/vllm_tpu_server.py"} for base in [set(), *allowed]])
    if set(records) not in allowed:
        raise RuntimeError("unexpected source overlay file set")
    for name, expected in records.items():
        source = directory / name
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source overlay checksum mismatch: {name}")
    for name in records:
        target = destination / name
        if target.is_symlink():
            raise RuntimeError(f"refusing source overlay through symlink: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / name, target)
