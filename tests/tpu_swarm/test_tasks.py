import json
from pathlib import Path

import pytest
import yaml
from tpuswarm.builtin import register_builtin_handlers
from tpuswarm.handlers import TaskRegistry, WorkflowEngine
from tpuswarm.store import SwarmStore
from tpuswarm.types import ConsistencyMode, Priority

from skyrl.tpu_swarm import (
    ERDOS_MIN_OVERLAP_KIND,
    GPTOSS120B_V6E32_SMOKE_KIND,
    QWEN35_V6E32_GRPO_KIND,
    erdos_ensemble_workflow,
    erdos_min_overlap_task,
    gptoss120b_v6e32_smoke_task,
    qwen35_v6e32_grpo_task,
    register,
)

_REPO = Path(__file__).parents[2]
_GPTOSS120B_POOL = (
    _REPO / "tpu" / "swarm" / "examples" / "v6e32-gptoss120b-smoke-pool.yaml"
)
_GPTOSS120B_STAGE = _REPO / "tpu" / "swarm" / "stage_gptoss120b_orbax.sbatch"
_V6E_SMOKE_WORKER = _REPO / "tpu" / "jobman" / "v6e_tunix_smoke_worker.sh"
_ORBAX_PREP = _REPO / "tpu" / "jobman" / "ensure_orbax_ckpt.sh"


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


def test_gptoss120b_v6e32_task_pins_full_training_topology_and_checkpoint():
    task = gptoss120b_v6e32_smoke_task(
        task_id="gptoss120b-v6e32-smoke",
        repo_dir="/repo",
        secrets={"HF_TOKEN": None},
    )

    payload = (
        registry().task(GPTOSS120B_V6E32_SMOKE_KIND).validate_payload(task.payload)
    )
    assert task.kind == GPTOSS120B_V6E32_SMOKE_KIND
    assert payload["argv"] == [
        "bash",
        "/repo/tpu/swarm/run_gptoss120b_v6e32_smoke.sh",
    ]
    assert payload["completion_probe_argv"] == [
        "bash",
        "/repo/tpu/swarm/gptoss120b_smoke_probe.sh",
    ]
    envs = payload["envs"]
    assert envs["MODEL_NAME"] == "openai/gpt-oss-120b"
    assert envs["TRAIN_WORKERS"] == "0,1,2,3,4,5,6,7"
    assert envs["VLLM_WORKERS"] == ""
    assert envs["TRAIN_TP_SIZE"] == "8"
    assert envs["TRAIN_FSDP_SIZE"] == "4"
    assert envs["TUNIX_ROW_SHARD"] == "4"
    assert envs["TRAIN_TPU_PROCESS_BOUNDS"] == "4,2,1"
    assert envs["TUNIX_SMOKE_ROWS"] == "4"
    assert envs["TUNIX_UNIFORM_SEQ_LEN"] == "1024"
    assert envs["TUNIX_TRAIN_TOKEN_BUDGET"] == "4096"
    assert envs["TUNIX_SMOKE_TIMEOUT_SECONDS"] == "14400"
    assert envs["TUNIX_MAXTEXT_CKPT_REQUIRE_MARKER"] == "1"
    assert "gptoss120b-bf16-d388" in envs["TUNIX_MAXTEXT_CKPT_CACHE_GCS"]
    assert "d388c5478b18b2322ab36c032deb87b9a4ff065f" in envs["TUNIX_MAXTEXT_PIP_SPEC"]
    assert json.loads(envs["TUNIX_MAXTEXT_KWARGS"])["megablox"] is True
    assert payload["pool"] == "tpuswarm-v6e32-asia-gptoss120b"
    resources = payload["resources"]
    assert resources["accelerators"] == "tpu-v6e-32"
    assert resources["zone"] == "asia-northeast1-b"
    assert resources["disk_size"] == 400
    assert resources["job_recovery"] == {
        "strategy": "FAILOVER",
        "max_restarts_on_errors": 3,
        "recover_on_exit_codes": [33, 34],
    }
    assert task.recovery_priority is Priority.BLOCKING_RECOVERY


def test_gptoss120b_task_result_is_idempotent_and_overrideable():
    task = gptoss120b_v6e32_smoke_task(
        task_id="stable-task",
        repo_dir="/repo",
        result_gcs="gs://bucket/results/custom.json",
        env={"TUNIX_UNIFORM_SEQ_LEN": "256"},
    )

    payload = (
        registry().task(GPTOSS120B_V6E32_SMOKE_KIND).validate_payload(task.payload)
    )
    assert task.idempotency_key == "stable-task"
    assert payload["envs"]["SMOKE_RESULT_GCS"] == "gs://bucket/results/custom.json"
    assert payload["envs"]["TUNIX_UNIFORM_SEQ_LEN"] == "256"


@pytest.mark.parametrize(
    "payload",
    [
        {"run_name": "bad;name", "repo_dir": "/repo"},
        {
            "run_name": "safe",
            "repo_dir": "/repo",
            "result_gcs": "https://bucket/result.json",
        },
    ],
)
def test_gptoss120b_task_rejects_unsafe_identity_or_result(payload):
    handler = registry().task(GPTOSS120B_V6E32_SMOKE_KIND)

    with pytest.raises(ValueError):
        handler.validate_payload(payload)


def test_gptoss120b_pool_and_task_cannot_drift_on_machine_shape():
    pool = yaml.safe_load(_GPTOSS120B_POOL.read_text())
    task = gptoss120b_v6e32_smoke_task(task_id="shape-check", repo_dir="/repo")
    payload = (
        registry().task(GPTOSS120B_V6E32_SMOKE_KIND).validate_payload(task.payload)
    )

    assert pool["resources"]["accelerators"] == payload["resources"]["accelerators"]
    assert pool["resources"]["zone"] == payload["resources"]["zone"]
    assert pool["resources"]["disk_size"] == payload["resources"]["disk_size"]
    for key in (
        "MODEL_NAME",
        "TUNIX_MAXTEXT_MODEL_NAME",
        "TUNIX_MAXTEXT_PIP_SPEC",
        "TUNIX_MAXTEXT_KWARGS",
        "TRAIN_WORKERS",
        "VLLM_WORKERS",
        "TRAIN_TP_SIZE",
        "TRAIN_FSDP_SIZE",
        "TUNIX_ROW_SHARD",
        "TRAIN_TPU_PROCESS_BOUNDS",
        "TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TUNIX_SMOKE_ROWS",
        "TUNIX_UNIFORM_SEQ_LEN",
        "TUNIX_TRAIN_TOKEN_BUDGET",
        "TUNIX_SMOKE_TIMEOUT_SECONDS",
        "TUNIX_MAXTEXT_CKPT_CACHE_GCS",
    ):
        assert str(pool["envs"][key]) == payload["envs"][key], key


def test_gptoss120b_checkpoint_publish_and_restore_require_atomic_marker():
    stage = _GPTOSS120B_STAGE.read_text()
    prep = _ORBAX_PREP.read_text()

    assert "SLURM_SUBMIT_DIR" in stage
    assert "e7523373bc44b42296b43202e265a1eebf2ee16f" in stage
    assert "#SBATCH --mem=320G" in stage
    assert "5d2578e28b809bfd90062a56d5c27f5159eae46c" in stage
    assert "--lazy_load_tensors=True" in stage
    assert stage.index("gcloud storage rsync") < stage.index(
        'gcloud storage cp "$MARKER"'
    )
    assert "TUNIX_MAXTEXT_CKPT_REQUIRE_MARKER:-0" in prep
    assert '"$SRC/CHECKPOINT_COMPLETE"' in prep
    assert '"$DST/0/items/manifest.ocdbt"' in prep


def test_v6e_smoke_timeout_requests_managed_recovery():
    worker = _V6E_SMOKE_WORKER.read_text()

    assert 'SMOKE_TIMEOUT_SECONDS="${TUNIX_SMOKE_TIMEOUT_SECONDS:-14400}"' in worker
    assert "timeout --foreground --signal=TERM --kill-after=60s" in worker
    assert 'SMOKE_RC=33' in worker
