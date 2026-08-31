from pathlib import Path

import pytest
from tpuswarm.builtin import register_builtin_handlers
from tpuswarm.handlers import TaskRegistry, WorkflowEngine
from tpuswarm.store import SwarmStore
from tpuswarm.types import ConsistencyMode, Priority

from skyrl.tpu_swarm import (
    ERDOS_MIN_OVERLAP_KIND,
    QWEN35_V6E32_GRPO_KIND,
    erdos_ensemble_workflow,
    erdos_min_overlap_task,
    qwen35_v6e32_grpo_task,
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


def test_qwen35_v6e32_task_pins_mixed_topology_and_grpo_contract():
    task = qwen35_v6e32_grpo_task(
        task_id="qwen35-v6e32",
        run_dir="qwen35-grpo-run",
        repo_dir="/repo",
        secrets={"WANDB_API_KEY": None},
    )

    payload = registry().task(QWEN35_V6E32_GRPO_KIND).validate_payload(task.payload)
    assert task.kind == QWEN35_V6E32_GRPO_KIND
    assert payload["argv"] == [
        "bash",
        "/repo/tpu/swarm/run_qwen35_v6e32_grpo.sh",
    ]
    envs = payload["envs"]
    assert envs["TRAIN_WORKERS"] == "0,1,2,3"
    assert envs["VLLM_WORKERS"] == "4,5,6,7"
    assert envs["TRAIN_TP_SIZE"] == "8"
    assert envs["TRAIN_FSDP_SIZE"] == "2"
    assert envs["VLLM_TP_SIZE"] == "4"
    assert envs["VLLM_ENGINES_PER_HOST"] == "1"
    assert envs["CONTEXT_WINDOW"] == "22528"
    assert envs["TUNIX_UNIFORM_SEQ_LEN"] == "22528"
    assert envs["TUNIX_TRAIN_TOKEN_BUDGET"] == "45056"
    assert envs["GROUPS_PER_BATCH"] == "16"
    assert envs["GROUP_SIZE"] == "32"
    assert envs["LEARNING_RATE"] == "1.5e-4"
    assert envs["LORA_RANK"] == "32"
    assert envs["TTD_LEAGUE_PIPELINE"] == "1"
    assert payload["pool"] == "tpuswarm-v6e32-asia-qwen35"
    resources = payload["resources"]
    assert resources["accelerators"] == "tpu-v6e-32"
    assert resources["zone"] == "asia-northeast1-b"
    assert resources["accelerator_args"] == {
        "gcp_queued_resource": True,
        "runtime_version": "v2-alpha-tpuv6e",
    }
    assert resources["disk_size"] == 150
    assert resources["job_recovery"] == {
        "strategy": "FAILOVER",
        "max_restarts_on_errors": 3,
        "recover_on_exit_codes": [33, 34],
    }
    assert task.recovery_priority is Priority.BLOCKING_RECOVERY


def test_qwen35_v6e32_task_rejects_unsafe_run_name():
    handler = registry().task(QWEN35_V6E32_GRPO_KIND)

    with pytest.raises(ValueError, match="run_dir"):
        handler.validate_payload({"run_dir": "run;shutdown", "repo_dir": "/repo"})
