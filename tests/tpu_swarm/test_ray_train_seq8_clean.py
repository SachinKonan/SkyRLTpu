from dataclasses import asdict

from tpu.swarm.ray_train.config import Config


def test_clean_restart_changes_only_run_identity_and_retired_tasks():
    old = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_seq8_training.json")
    new = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_seq8_clean.json")
    assert new.run_id == "qwen-ray-v4-64-seq8-clean-001"
    assert new.run_gcs != old.run_gcs
    assert len(new.retired_task_ids) == 4
    assert {task.rsplit("_", 1)[1] for task in new.retired_task_ids} == {
        "401-0", "408-0", "396-0", "389-0"
    }
    left, right = asdict(old), asdict(new)
    for key in ("run_id", "retired_task_ids"):
        left.pop(key)
        right.pop(key)
    assert left == right
