from pathlib import Path

from tpu.swarm.ray_train.commands import client_environment, inference_command
from tpu.swarm.ray_train.config import Config


def test_full_training_uses_eight_sequences_per_independent_engine():
    c = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_seq8_training.json")
    previous = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_retry_after_concurrency.json")
    assert not c.inference_only and c.inference_only_ranks is None
    assert c.hosts == 8 and c.trainer.hosts == c.inference_hosts == 4
    assert c.trainer == previous.trainer
    assert c.base_bundle_sha256 == previous.base_bundle_sha256
    command = inference_command(c, *[Path(p) for p in ("/root", "/source", "/hf", "/run")])
    for flag, value in (("--max-num-seqs", "8"), ("--tensor-parallel-size", "4"),
                        ("--max-model-len", "22528"), ("--max-loras", "1"),
                        ("--gpu-memory-utilization", "0.8")):
        assert command[command.index(flag) + 1] == value
    env = client_environment(c, Path("/root"), "127.0.0.1")
    assert env["GROUP_SIZE"] == "32" and env["GROUPS_PER_BATCH"] == "16"
    assert c.client_sampling_environment() == previous.client_sampling_environment()


def test_fresh_state_reuses_models_and_seeds_successful_diagnostic_cache():
    c = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_seq8_training.json")
    previous = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_retry_after_concurrency.json")
    assert c.run_id != previous.run_id and c.run_gcs != previous.run_gcs
    assert c.cache.hf == previous.cache.hf and c.cache.orbax == previous.cache.orbax
    assert c.cache.trainer_compile == previous.cache.trainer_compile
    assert c.cache.inference_compile_seed == previous.cache.inference_compile
    assert "seq8-mem80" in c.cache.inference_compile
    assert c.retired_task_ids == [
        "sky-managed-2026-09-07-14-04-01-220617_qwen-ray-v4-64-budget-004_401-0"]
