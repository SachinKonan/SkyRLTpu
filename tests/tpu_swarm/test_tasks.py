from pathlib import Path

import pytest
from tpuswarm.builtin import register_builtin_handlers
from tpuswarm.handlers import TaskRegistry, WorkflowEngine
from tpuswarm.store import SwarmStore
from tpuswarm.types import ConsistencyMode, Priority

from skyrl.tpu_swarm import (
    ERDOS_MIN_OVERLAP_KIND,
    erdos_ensemble_workflow,
    erdos_min_overlap_task,
    register,
)


def registry() -> TaskRegistry:
    result = TaskRegistry()
    register_builtin_handlers(result)
    register(result)
    return result


def test_erdos_task_normalizes_existing_cell_contract(tmp_path: Path):
    store = SwarmStore(tmp_path / "swarm.db")
    engine = WorkflowEngine(store, registry())
    task = erdos_min_overlap_task(
        task_id="erdos-qwen",
        cell="ttd-n",
        run_dir="test-run",
        repo_dir="/repo",
        env={"GCS_RUN": "gs://bucket/test-run"},
        secrets={"WANDB_API_KEY": None},
    )

    engine.submit(
        erdos_ensemble_workflow(workflow_id="erdos-workflow", members={"qwen": task})
    )

    record = store.get_task("erdos-qwen")
    assert record.kind == ERDOS_MIN_OVERLAP_KIND
    assert record.payload["argv"] == [
        "bash",
        "/repo/tpu/swarm/run_erdos_min_overlap.sh",
    ]
    assert record.payload["completion_probe_argv"] == [
        "bash",
        "/repo/tpu/jobman/cell_probe.sh",
    ]
    assert record.payload["envs"]["TTD_ENV"] == "erdos_min_overlap"
    assert record.payload["envs"]["GCS_RUN"] == "gs://bucket/test-run"
    assert record.payload["secrets"] == {"WANDB_API_KEY": None}
    assert record.payload["pool"] == "tpuswarm-v5p32"
    assert record.payload["resources"]["job_recovery"]["strategy"] == (
        "EAGER_NEXT_REGION"
    )
    assert record.recovery_priority is Priority.BLOCKING_RECOVERY


def test_erdos_ensemble_preserves_independent_members():
    first = erdos_min_overlap_task(
        task_id="first", cell="ttd-n", run_dir="first", repo_dir="/repo"
    )
    second = erdos_min_overlap_task(
        task_id="second", cell="ttd-r", run_dir="second", repo_dir="/repo"
    )

    workflow = erdos_ensemble_workflow(
        workflow_id="ensemble",
        members={"first": first, "second": second},
        consistency_mode=ConsistencyMode.ALL_AT_BARRIER,
    )

    assert workflow.consistency_mode is ConsistencyMode.ALL_AT_BARRIER
    assert [item["key"] for item in workflow.payload["components"]] == [
        "first",
        "second",
    ]


def test_erdos_task_rejects_shell_metacharacters():
    handler = registry().task(ERDOS_MIN_OVERLAP_KIND)

    with pytest.raises(ValueError, match="cell"):
        handler.validate_payload(
            {"cell": "ttd-n;shutdown", "run_dir": "run", "repo_dir": "/repo"}
        )
