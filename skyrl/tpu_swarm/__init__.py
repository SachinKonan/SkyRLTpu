"""SkyRL handlers and task builders for the TPUSwarm runtime."""

from skyrl.tpu_swarm.tasks import (
    ERDOS_MIN_OVERLAP_KIND,
    QWEN35_V6E32_GRPO_KIND,
    ErdosMinOverlapTask,
    Qwen35V6e32GrpoTask,
    erdos_ensemble_workflow,
    erdos_min_overlap_task,
    qwen35_v6e32_grpo_task,
    register,
)

__all__ = [
    "ERDOS_MIN_OVERLAP_KIND",
    "QWEN35_V6E32_GRPO_KIND",
    "ErdosMinOverlapTask",
    "Qwen35V6e32GrpoTask",
    "erdos_ensemble_workflow",
    "erdos_min_overlap_task",
    "qwen35_v6e32_grpo_task",
    "register",
]
