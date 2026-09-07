import json
import subprocess
from pathlib import Path
import pytest
import yaml

from tpu.swarm.ray_train.commands import (
    client_environment, inference_command, inference_environment,
    trainer_backend_config, trainer_command, trainer_environment,
)
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.build import build


def config():
    return Config.from_dict(dict(run_id="ray-test", accelerator="tpu-v4-64", hosts=8,
        bucket="gs://test", base_bundle="gs://test/base.tar.gz", base_bundle_sha256="a"*64,
        cache=dict(hf="gs://test/hf", orbax="gs://test/orbax", trainer_compile="gs://test/train",
                   inference_compile="gs://test/infer")))


ROOT = Path("/private/ray-test")
IPS = ["10.0.0.1", "10.0.0.3", "10.0.0.5", "10.0.0.7"]


@pytest.mark.parametrize("profile,zone,hosts", [
    ("qwen_v4_32", "us-central2-b", 4),
    ("qwen_v4_64", "us-central2-b", 8),
    ("qwen_v5p_32", "us-east5-a", 4),
])
def test_profile_build_targets_correct_tpu_family(tmp_path, profile, zone, hosts):
    path = Path("tpu/swarm/ray_train/profiles") / (profile + ".json")
    cfg = Config.load(path)
    _, _, task_path = build(path, tmp_path)
    task = yaml.safe_load(task_path.read_text())
    assert cfg.hosts == hosts
    assert task["resources"]["zone"] == zone
    assert task["resources"]["accelerators"] == cfg.accelerator
    if "v4" in profile:
        assert task["resources"]["accelerator_args"]["runtime_version"] == "tpu-ubuntu2204-base"
    if hosts == 4:
        assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp) == (1, 1, 4)
        assert cfg.inference_hosts == 3
        assert cfg.trainer.process_bounds == "1,1,1"
        env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:1], 0)
        assert env["TPU_PROCESS_ADDRESSES"] == "10.0.0.1:19804"


@pytest.mark.parametrize("profile,sequences,memory,cache_tag", [
    ("qwen_v5p_32", 32, 0.9, "seq32-mem90"),
    ("qwen_v4_64", 16, 0.87, "seq16-mem87"),
])
def test_tuned_profiles_preserve_training_and_checkpoint_identity(profile, sequences, memory, cache_tag):
    cfg = Config.load(f"tpu/swarm/ray_train/profiles/{profile}.json")
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    assert cmd[cmd.index("--max-num-seqs") + 1] == str(sequences)
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == str(memory)
    assert cache_tag in cfg.cache.inference_compile
    assert "seq16-mem90" in cfg.cache.inference_compile_seed
    if profile == "qwen_v4_64":
        assert cfg.run_id == "qwen-ray-v4-64-001"
        assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp, cfg.inference_hosts) == (4, 8, 2, 4)
    else:
        assert cfg.run_id == "qwen-ray-v5p-32-001"
        assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp, cfg.inference_hosts) == (1, 1, 4, 3)
    assert cfg.trainer.remat == "full"


@pytest.mark.parametrize("profile", ["qwen_v4_32", "qwen_v4_64", "qwen_v5p_32"])
def test_launch_checkpoint_path_matches_synchronous_mirror(tmp_path, monkeypatch, profile):
    from skyrl.utils.checkpoint_mirror import mirror_checkpoint_to_gcs
    cfg = Config.load(f"tpu/swarm/ray_train/profiles/{profile}.json")
    command = trainer_command(cfg, tmp_path, tmp_path / "source", IPS[0], IPS[:1], 0)
    staging = command[command.index("--checkpoints-base") + 1]
    assert "://" not in staging
    checkpoint = Path(staging) / "model_test" / "sampler_weights" / "ss0_seq1.tar.gz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sampler checkpoint")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0,
            stdout=str(checkpoint.stat().st_size) if "describe" in command else "", stderr="")

    monkeypatch.setattr("skyrl.utils.checkpoint_mirror.subprocess.run", run)
    backend = json.loads(command[-1])
    destination = mirror_checkpoint_to_gcs(checkpoint, backend["checkpoint_mirror_gcs"],
                                            "model_test", "sampler_weights")
    assert destination == cfg.run_gcs + "/checkpoints/model_test/sampler_weights/ss0_seq1.tar.gz"
    assert calls[0] == ["gcloud", "storage", "cp", str(checkpoint), destination]
    assert calls[1][2:4] == ["objects", "describe"]


def test_trainer_has_proven_qwen_mesh_and_direct_upload():
    cfg = config()
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS)
    assert backend["maxtext_kwargs"] == dict(ici_tensor_parallelism=8, ici_fsdp_parallelism=2,
        ici_context_parallelism=1, remat_policy="full", num_vocab_tiling=64,
        attention="autoselected", use_tokamax_splash=True, allow_split_physical_axes=True,
        override_model_config=True, base_num_kv_heads=8, jax_cache_dir=str(ROOT / "ram/compile"))
    assert backend["vllm_lora_upload_endpoint"] == "/skyrl/v1/upload_lora_adapter"
    assert backend["num_processes"] == 4
    assert not backend["vllm_client_side_round_robin"]
    cmd = trainer_command(cfg, ROOT, ROOT / "source", IPS[0], IPS, 0)
    assert cmd[cmd.index("--checkpoints-base")+1] == str(ROOT / "runs" / cfg.run_id / "checkpoints")
    assert backend["checkpoint_mirror_gcs"] == cfg.run_gcs + "/checkpoints"
    assert json.loads(cmd[-1]) == backend


def test_trainer_rank_and_bounds_use_selected_physical_row():
    env = trainer_environment(config(), ROOT, ROOT / "run", IPS, 2)
    assert env["CLOUD_TPU_TASK_ID"] == "2"
    assert env["TPU_PROCESS_ADDRESSES"] == ",".join(ip+":19804" for ip in IPS)
    assert env["TPU_PROCESS_BOUNDS"] == "1,1,4"
    cmd = trainer_command(config(), ROOT, ROOT / "source", IPS[0], IPS, 2)
    assert "skyrl.backends.rpc" in cmd
    assert cmd[cmd.index("--process-id")+1] == "2"


def test_inference_exact_requested_settings_and_single_host_isolation(monkeypatch):
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "bad:7777")
    monkeypatch.setenv("TPU_MULTIPROCESS_DP", "1")
    cfg = config()
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    for key, value in (("--tensor-parallel-size", "4"), ("--max-num-seqs", "16"),
                       ("--gpu-memory-utilization", "0.9"), ("--max-num-batched-tokens", "4096")):
        assert cmd[cmd.index(key)+1] == value
    env = inference_environment(cfg, ROOT, ROOT / "run")
    assert env["TPU_PROCESS_BOUNDS"] == "1,1,1"
    assert env["TPU_PROCESS_ADDRESSES"] == "localhost:19805"
    assert "JAX_COORDINATOR_ADDRESS" not in env
    assert "TPU_MULTIPROCESS_DP" not in env
    assert env["VLLM_XLA_CACHE_PATH"].endswith("ram/compile")


def test_client_uses_existing_renderer_and_workload_ray():
    env = client_environment(config(), ROOT, IPS[0])
    assert env["TTD_ENSEMBLE_MODELS"] == "Qwen/Qwen3.5-27B:qwen3:qwen"
    assert env["RAY_ADDRESS"] == "10.0.0.1:19679"
    assert env["TTD_RAY_PAYLOAD"] == "1"
    assert env["GROUPS_PER_BATCH"] == "16"
    assert env["GROUP_SIZE"] == "32"
    assert env["EVAL_TIMEOUT"] == "1100"


def test_both_roles_enable_persistent_compilation_cache(monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")
    cfg = config()
    for env in (trainer_environment(cfg, ROOT, ROOT / "run", IPS, 0),
                inference_environment(cfg, ROOT, ROOT / "run")):
        assert env["JAX_ENABLE_COMPILATION_CACHE"] == "true"
        assert env["JAX_COMPILATION_CACHE_DIR"] == str(ROOT / "ram/compile")
        assert env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
        assert env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "0"


def test_v5p_profile_preserves_single_host_fsdp_and_memory_saving():
    cfg = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json")
    assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp) == (1, 1, 4)
    assert cfg.inference.tp == 4 and cfg.inference_hosts == 3
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:1])
    assert backend["maxtext_kwargs"]["ici_tensor_parallelism"] == 1
    assert backend["maxtext_kwargs"]["ici_fsdp_parallelism"] == 4
    assert backend["maxtext_kwargs"]["jax_cache_dir"] == str(ROOT / "ram/compile")
    assert backend["free_base_state_after_template"]
    assert "tp1-fsdp4" in cfg.cache.trainer_compile
    env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:1], 0)
    assert env["TUNIX_ROW_SHARD"] == "4"
    assert env["TUNIX_UNIFORM_SEQ_LEN"] == "18432"
    assert backend["maxtext_max_target_length"] == 18432
    assert backend["train_token_budget"] == 4 * 18432
    assert "s18432" in cfg.cache.trainer_compile
    client = client_environment(cfg, ROOT, IPS[0])
    assert client["TTD_M0_TRAIN_MAX_SEQ"] == "18432"
    assert client["TTD_M0_CONTEXT_WINDOW"] == "18432"
    assert cfg.inference.max_model_length == 22528
    assert (cfg.inference.max_sequences, cfg.inference.memory_utilization) == (32, 0.9)


def test_v4_64_trainer_shape_is_unchanged():
    cfg = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64.json")
    assert (cfg.trainer.tp, cfg.trainer.fsdp) == (8, 2)
    assert (cfg.trainer.sequence_length, cfg.trainer.token_budget) == (22528, 45056)
