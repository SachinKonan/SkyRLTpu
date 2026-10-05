from pathlib import Path

from tpu.swarm.ray_train.commands import inference_command
from tpu.swarm.ray_train.config import Config


def test_retry_preserves_validated_serving_shape_and_full_training_split():
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_retry_after_concurrency.json")
    diagnostic = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_concurrency.json")
    previous = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_budget.json")
    assert not config.inference_only and config.inference_only_ranks is None
    assert config.trainer.hosts == config.inference_hosts == 4
    assert (config.trainer.tp, config.trainer.fsdp, config.trainer.remat) == (8, 2, "full")
    assert config.base_bundle_sha256 == diagnostic.base_bundle_sha256
    assert config.cache.inference_compile == diagnostic.cache.inference_compile
    assert config.cache.orbax == previous.cache.orbax
    assert config.cache.trainer_compile == previous.cache.trainer_compile
    assert config.client_sampling_environment() == previous.client_sampling_environment()
    assert config.run_id not in (previous.run_id, diagnostic.run_id)
    command = inference_command(config, *[Path(p) for p in ("/root", "/source", "/hf", "/run")])
    for flag, value in (("--max-num-seqs", "16"), ("--tensor-parallel-size", "4"),
                        ("--gpu-memory-utilization", "0.8"), ("--max-loras", "1")):
        assert command[command.index(flag)+1] == value
    assert config.retired_task_ids == [
        "sky-managed-2026-09-07-11-37-58-605508_qwen-ray-v4-64-budget-003_382-0"]
