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
