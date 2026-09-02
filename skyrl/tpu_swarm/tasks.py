"""TPUSwarm adapters for existing SkyRL TPU training jobs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tpuswarm.builtin import CommandAutoResumable
from tpuswarm.handlers import TaskRegistry
from tpuswarm.serialization import to_jsonable
from tpuswarm.types import (
    ConsistencyMode,
    Priority,
    TaskSpec,
    WorkflowSpec,
)

ERDOS_MIN_OVERLAP_KIND = "skyrl.ttt.erdos_min_overlap.v1"
QWEN35_V6E32_GRPO_KIND = "skyrl.qwen35.v6e32_grpo.v1"
GPTOSS120B_V6E32_SMOKE_KIND = "skyrl.gptoss120b.v6e32_smoke.v1"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class ErdosMinOverlapTask(CommandAutoResumable):
    """Runs one existing Erdős cell as an independently recoverable leaf."""

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        cell = payload.get("cell", "ttd-n")
        run_dir = payload.get("run_dir", "stageA2-ttd-n")
        if not isinstance(cell, str) or _SAFE_NAME.fullmatch(cell) is None:
            raise ValueError(
                "cell must contain only letters, digits, dot, underscore, or dash"
            )
        if not isinstance(run_dir, str) or _SAFE_NAME.fullmatch(run_dir) is None:
            raise ValueError(
                "run_dir must contain only letters, digits, dot, underscore, or dash"
            )

        repo_dir = Path(str(payload.get("repo_dir", Path.cwd()))).expanduser()
        env = {
            "CELL": cell,
            "RUN_DIR_NAME": run_dir,
            "EXPERIMENT_NAME": str(payload.get("experiment_name", run_dir)),
            "NUM_EPOCHS": str(payload.get("num_epochs", 15)),
            "TTD_ENV": "erdos_min_overlap",
            "SKYRL_REPO_DIR": str(repo_dir),
        }
        overrides = payload.get("env", {})
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise ValueError("env must map strings to strings")
        env.update(overrides)
        secrets = payload.get("secrets", {})
        if not isinstance(secrets, dict) or not all(
            isinstance(key, str) and (value is None or isinstance(value, str))
            for key, value in secrets.items()
        ):
            raise ValueError("secrets must map strings to strings or null")

        wrapper = repo_dir / "tpu" / "swarm" / "run_erdos_min_overlap.sh"
        probe = repo_dir / "tpu" / "jobman" / "cell_probe.sh"
        sync = repo_dir / "tpu" / "jobman" / "cell_sync.sh"
        resources = payload.get(
            "resources",
            {
                "cloud": "gcp",
                "accelerators": "tpu-v5p-32",
                "use_spot": True,
                "job_recovery": {
                    "strategy": "EAGER_NEXT_REGION",
                    "max_restarts_on_errors": 3,
                },
            },
        )
        return super().validate_payload(
            {
                "argv": ["bash", str(wrapper)],
                "completion_probe_argv": ["bash", str(probe)],
                "preemption_argv": ["bash", str(sync)],
                "pool": payload.get("pool", "tpuswarm-v5p32"),
                "resources": resources,
                "envs": env,
                "secrets": secrets,
                "cell": cell,
                "run_dir": run_dir,
            }
        )


class Qwen35V6e32GrpoTask(CommandAutoResumable):
    """Runs one resumable 22,528-context Qwen GRPO job on a warm v6e-32."""

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or _SAFE_NAME.fullmatch(run_dir) is None:
            raise ValueError(
                "run_dir must contain only letters, digits, dot, underscore, or dash"
            )
        repo_dir = Path(str(payload.get("repo_dir", Path.cwd()))).expanduser()
        env = {
            # Proven league GRPO arm.
            "CELL": "grpo-n",
            "RUN_DIR_NAME": run_dir,
            "EXPERIMENT_NAME": str(payload.get("experiment_name", run_dir)),
            "NUM_EPOCHS": str(payload.get("num_epochs", 15)),
            "TTD_ENV": "erdos_min_overlap",
            "TTD_ADV_ESTIMATOR": "mean_baseline",
            "TTD_ELITE_SLOTS": "2",
            "GROUPS_PER_BATCH": "16",
            "GROUP_SIZE": "32",
            "LEARNING_RATE": "1.5e-4",
            "LORA_RANK": "32",
            "KL_PENALTY_COEF": "0",
            "TTD_RESTART_RATIO": "0",
            "TTD_CROSS_WEIGHT": "0",
            "TTD_LEAGUE_PIPELINE": "1",
            "TTD_ALLOW_SINGLE_MEMBER": "1",
            # 22,528-token rollouts, streamed into two-row trainer calls.
            "CONTEXT_WINDOW": "22528",
            "PHASE1_MAX_TOKENS": "20480",
            "TTD_TRAIN_MAX_SEQ": "22528",
            "TUNIX_MAX_TARGET_LENGTH": "22528",
            "TUNIX_UNIFORM_SEQ_LEN": "22528",
            "TUNIX_TRAIN_TOKEN_BUDGET": "45056",
            # Four half-host TPU VMs train; four each serve one TP4 engine.
            "TRAIN_WORKERS": "0,1,2,3",
            "VLLM_WORKERS": "4,5,6,7",
            "TRAIN_TP_SIZE": "8",
            "TRAIN_FSDP_SIZE": "2",
            "TUNIX_ROW_SHARD": "2",
            "TRAIN_TPU_PROCESS_BOUNDS": "2,2,1",
            "TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
            "VLLM_TP_SIZE": "4",
            "VLLM_ENGINES_PER_HOST": "1",
            "VLLM_MAX_MODEL_LEN": "22528",
            "CELL_START_VLLM": "1",
            "CELL_SYNC_SKYRL": "0",
            "VLLM_RAY_EXECUTOR": "0",
            "VLLM_CLIENT_SIDE_ROUND_ROBIN": "1",
            "ZONE": "asia-northeast1-b",
            "GCS_RUN": (f"gs://sk7524-tinker-tpu-asia-northeast1/skyrl-runs/{run_dir}"),
            "TUNIX_MAXTEXT_CKPT_CACHE_GCS": (
                "gs://sk7524-tinker-tpu-asia-northeast1/skyrl-maxtext-ckpts"
            ),
            "TUNIX_JAX_CACHE_GCS": (
                "gs://sk7524-tinker-tpu-asia-northeast1/"
                "jax-compile-cache-v6e-qwen35-tp8-fsdp2-r32-s22528-b45056-v1"
            ),
            "VLLM_XLA_CACHE_GCS": (
                "gs://sk7524-tinker-tpu-asia-northeast1/"
                "vllm-xla-cache-v6e-qwen35-tp4-s22528-v1"
            ),
            "HF_CACHE_GCS": (
                "gs://sk7524-tinker-tpu-asia-northeast1/hf-cache-qwen35-v1"
            ),
            "SKYRL_REPO_DIR": str(repo_dir),
        }
        overrides = payload.get("env", {})
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise ValueError("env must map strings to strings")
        env.update(overrides)
        secrets = payload.get("secrets", {})
        if not isinstance(secrets, dict) or not all(
            isinstance(key, str) and (value is None or isinstance(value, str))
            for key, value in secrets.items()
        ):
            raise ValueError("secrets must map strings to strings or null")

        resources = payload.get(
            "resources",
            {
                "cloud": "gcp",
                "region": "asia-northeast1",
                "zone": "asia-northeast1-b",
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
            },
        )
        wrapper = repo_dir / "tpu" / "swarm" / "run_qwen35_v6e32_grpo.sh"
        probe = repo_dir / "tpu" / "jobman" / "cell_probe.sh"
        sync = repo_dir / "tpu" / "jobman" / "cell_sync.sh"
        return super().validate_payload(
            {
                "argv": ["bash", str(wrapper)],
                "completion_probe_argv": ["bash", str(probe)],
                "preemption_argv": ["bash", str(sync)],
                "pool": payload.get("pool", "tpuswarm-v6e32-asia-qwen35"),
                "resources": resources,
                "envs": env,
                "secrets": secrets,
                "cell": "grpo-n",
                "run_dir": run_dir,
            }
        )


class GptOss120BV6e32SmokeTask(CommandAutoResumable):
    """Runs the checkpoint/replay acceptance gate on a full v6e-32 slice."""

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        run_name = payload.get("run_name")
        if not isinstance(run_name, str) or _SAFE_NAME.fullmatch(run_name) is None:
            raise ValueError(
                "run_name must contain only letters, digits, dot, underscore, or dash"
            )
        repo_dir = Path(str(payload.get("repo_dir", Path.cwd()))).expanduser()
        result_gcs = str(
            payload.get(
                "result_gcs",
                "gs://sk7524-tinker-tpu-asia-northeast1/"
                f"v6e-smoke-results/{run_name}.json",
            )
        )
        if not result_gcs.startswith("gs://") or result_gcs.endswith("/"):
            raise ValueError("result_gcs must be a GCS object URL")

        env = {
            "MODEL_NAME": "openai/gpt-oss-120b",
            "TUNIX_MAXTEXT_MODEL_NAME": "gpt-oss-120b",
            "TUNIX_MAXTEXT_PIP_SPEC": (
                "maxtext @ git+https://github.com/SachinKonan/maxtext.git@"
                "d388c5478b18b2322ab36c032deb87b9a4ff065f"
            ),
            "TUNIX_MAXTEXT_KWARGS": (
                '{"sparse_matmul":true,"megablox":true,'
                '"num_vocab_tiling":64,"remat_policy":"full",'
                '"allow_split_physical_axes":true}'
            ),
            "TRAIN_WORKERS": "0,1,2,3,4,5,6,7",
            "VLLM_WORKERS": "",
            "TRAIN_TP_SIZE": "8",
            "TRAIN_FSDP_SIZE": "4",
            "TUNIX_ROW_SHARD": "4",
            "TRAIN_TPU_PROCESS_BOUNDS": "4,2,1",
            "TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
            "TUNIX_LORA_RANK": "32",
            "TUNIX_SMOKE_ROWS": "4",
            "TUNIX_SMOKE_REPLAYS": "1",
            "TUNIX_SMOKE_EXTRA_UPDATES": "1",
            "TUNIX_REPLAY_DIAGNOSTICS": "1",
            "TUNIX_REQUIRE_SPARSE_EXPERT_GRADIENTS": "1",
            "TUNIX_FREE_BASE_STATE": "1",
            "TUNIX_MAX_TARGET_LENGTH": "1024",
            "TUNIX_UNIFORM_SEQ_LEN": "1024",
            "TUNIX_TRAIN_TOKEN_BUDGET": "4096",
            "TUNIX_FLCE_TILE_SIZE": "512",
            "TUNIX_SMOKE_TIMEOUT_SECONDS": "14400",
            "TUNIX_MAXTEXT_CKPT_REQUIRE_MARKER": "1",
            "TUNIX_MAXTEXT_CKPT_CACHE_GCS": (
                "gs://sk7524-tinker-tpu-asia-northeast1/"
                "skyrl-maxtext-ckpts-gptoss120b-bf16-d388"
            ),
            "TUNIX_JAX_CACHE_GCS": (
                "gs://sk7524-tinker-tpu-asia-northeast1/"
                "jax-compile-cache-v6e-gptoss120b-tp8-fsdp4-r32-s1024-v1"
            ),
            "TUNIX_JAX_CACHE_LOCAL": (
                "/home/sk7524_princeton_edu/jax_cache_gptoss120b_tp8_fsdp4_r32_s1024_v1"
            ),
            "SMOKE_RESULT_GCS": result_gcs,
            "ZONE": "asia-northeast1-b",
            "SKYRL_REPO_DIR": str(repo_dir),
        }
        overrides = payload.get("env", {})
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise ValueError("env must map strings to strings")
        env.update(overrides)
        secrets = payload.get("secrets", {})
        if not isinstance(secrets, dict) or not all(
            isinstance(key, str) and (value is None or isinstance(value, str))
            for key, value in secrets.items()
        ):
            raise ValueError("secrets must map strings to strings or null")

        resources = payload.get(
            "resources",
            {
                "cloud": "gcp",
                "region": "asia-northeast1",
                "zone": "asia-northeast1-b",
                "accelerators": "tpu-v6e-32",
                "accelerator_args": {
                    "gcp_queued_resource": True,
                    "runtime_version": "v2-alpha-tpuv6e",
                },
                "use_spot": True,
                # The immutable BF16 Orbax checkpoint is much larger than the
                # 65 GB MXFP4 release. Leave room for venvs and XLA caches.
                "disk_size": 400,
                "job_recovery": {
                    "strategy": "FAILOVER",
                    "max_restarts_on_errors": 3,
                    "recover_on_exit_codes": [33, 34],
                },
            },
        )
        wrapper = repo_dir / "tpu" / "swarm" / "run_gptoss120b_v6e32_smoke.sh"
        probe = repo_dir / "tpu" / "swarm" / "gptoss120b_smoke_probe.sh"
        return super().validate_payload(
            {
                "argv": ["bash", str(wrapper)],
                "completion_probe_argv": ["bash", str(probe)],
                "pool": payload.get("pool", "tpuswarm-v6e32-asia-gptoss120b"),
                "resources": resources,
                "envs": env,
                "secrets": secrets,
                "run_name": run_name,
                "result_gcs": result_gcs,
            }
        )


def erdos_min_overlap_task(
    *,
    task_id: str,
    cell: str,
    run_dir: str,
    repo_dir: str | Path,
    resource_class: str = "gcp-tpu-v5p-32",
    experiment_name: str | None = None,
    num_epochs: int = 15,
    env: Mapping[str, str] | None = None,
    secrets: Mapping[str, str | None] | None = None,
    pool: str = "tpuswarm-v5p32",
    resources: Mapping[str, Any] | None = None,
    priority: Priority = Priority.NORMAL,
) -> TaskSpec:
    """Builds the stable JSON contract for one current SkyRL Erdős run."""

    payload: dict[str, Any] = {
        "cell": cell,
        "run_dir": run_dir,
        "repo_dir": str(Path(repo_dir).expanduser()),
        "num_epochs": num_epochs,
        "env": dict(env or {}),
        "secrets": dict(secrets or {}),
        "pool": pool,
    }
    if resources is not None:
        payload["resources"] = dict(resources)
    if experiment_name is not None:
        payload["experiment_name"] = experiment_name
    return TaskSpec(
        task_id=task_id,
        idempotency_key=task_id,
        kind=ERDOS_MIN_OVERLAP_KIND,
        resource_class=resource_class,
        payload=payload,
        priority=priority,
        recovery_priority=Priority.BLOCKING_RECOVERY,
    )


def qwen35_v6e32_grpo_task(
    *,
    task_id: str,
    run_dir: str,
    repo_dir: str | Path,
    resource_class: str = "gcp-tpu-v6e-32-asia",
    experiment_name: str | None = None,
    num_epochs: int = 15,
    env: Mapping[str, str] | None = None,
    secrets: Mapping[str, str | None] | None = None,
    pool: str = "tpuswarm-v6e32-asia-qwen35",
    resources: Mapping[str, Any] | None = None,
    priority: Priority = Priority.NORMAL,
) -> TaskSpec:
    """Build the fixed mixed-v6e-32 Qwen3.5 GRPO task contract."""

    payload: dict[str, Any] = {
        "run_dir": run_dir,
        "repo_dir": str(Path(repo_dir).expanduser()),
        "num_epochs": num_epochs,
        "env": dict(env or {}),
        "secrets": dict(secrets or {}),
        "pool": pool,
    }
    if resources is not None:
        payload["resources"] = dict(resources)
    if experiment_name is not None:
        payload["experiment_name"] = experiment_name
    return TaskSpec(
        task_id=task_id,
        idempotency_key=task_id,
        kind=QWEN35_V6E32_GRPO_KIND,
        resource_class=resource_class,
        payload=payload,
        priority=priority,
        recovery_priority=Priority.BLOCKING_RECOVERY,
    )


def gptoss120b_v6e32_smoke_task(
    *,
    task_id: str,
    repo_dir: str | Path,
    run_name: str | None = None,
    result_gcs: str | None = None,
    resource_class: str = "gcp-tpu-v6e-32-asia",
    env: Mapping[str, str] | None = None,
    secrets: Mapping[str, str | None] | None = None,
    pool: str = "tpuswarm-v6e32-asia-gptoss120b",
    resources: Mapping[str, Any] | None = None,
    priority: Priority = Priority.NORMAL,
) -> TaskSpec:
    """Build the fixed GPT-OSS 120B TP8/FSDP4 acceptance contract."""

    payload: dict[str, Any] = {
        "run_name": run_name or task_id,
        "repo_dir": str(Path(repo_dir).expanduser()),
        "env": dict(env or {}),
        "secrets": dict(secrets or {}),
        "pool": pool,
    }
    if result_gcs is not None:
        payload["result_gcs"] = result_gcs
    if resources is not None:
        payload["resources"] = dict(resources)
    return TaskSpec(
        task_id=task_id,
        idempotency_key=task_id,
        kind=GPTOSS120B_V6E32_SMOKE_KIND,
        resource_class=resource_class,
        payload=payload,
        priority=priority,
        recovery_priority=Priority.BLOCKING_RECOVERY,
    )


def erdos_ensemble_workflow(
    *,
    workflow_id: str,
    members: Mapping[str, TaskSpec] | Sequence[tuple[str, TaskSpec]],
    consistency_mode: ConsistencyMode = ConsistencyMode.INDEPENDENT,
) -> WorkflowSpec:
    """Builds a workflow whose failed members recover independently."""

    items = members.items() if isinstance(members, Mapping) else members
    components = [{"key": key, "task": to_jsonable(task)} for key, task in items]
    if not components:
        raise ValueError("an ensemble must contain at least one member")
    return WorkflowSpec(
        workflow_id=workflow_id,
        kind="static_multi.v1",
        consistency_mode=consistency_mode,
        payload={"components": components},
    )


def register(registry: TaskRegistry) -> None:
    """Registers SkyRL handlers with a TPUSwarm server or worker."""

    registry.register_task(ERDOS_MIN_OVERLAP_KIND, ErdosMinOverlapTask)
    registry.register_task(QWEN35_V6E32_GRPO_KIND, Qwen35V6e32GrpoTask)
    registry.register_task(GPTOSS120B_V6E32_SMOKE_KIND, GptOss120BV6e32SmokeTask)
