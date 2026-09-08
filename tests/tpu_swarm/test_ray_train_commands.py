import json
import subprocess
from pathlib import Path
import pytest
import yaml

from tpu.swarm.ray_train.commands import (
    client_environment, inference_command, inference_environment, inference_urls,
    trainer_backend_config, trainer_command, trainer_environment,
)
from tpu.swarm.ray_train.config import Config, PRESETS
from tpu.swarm.ray_train.build import build


def config(**overrides):
    """A v5p-32 profile with executor defaults only (the legacy qwen cell shape)."""
    raw = dict(run_id="ray-test", accelerator="tpu-v5p-32", hosts=4,
        bucket="gs://test", base_bundle="gs://test/base.tar.gz", base_bundle_sha256="a"*64,
        cache=dict(hf="gs://test/hf", orbax="gs://test/orbax", trainer_compile="gs://test/train",
                   inference_compile="gs://test/infer"))
    raw.update(overrides)
    return Config.from_dict(raw)


def v5p_config(**overrides):
    """A v5p-32 profile that names only the model preset: must equal a legacy cell."""
    raw = dict(run_id="ray-v5p", accelerator="tpu-v5p-32", hosts=4,
        bucket="gs://test", base_bundle="gs://test/base.tar.gz", base_bundle_sha256="a"*64,
        cache=dict(hf="gs://test/hf", orbax="gs://test/orbax", trainer_compile="gs://test/train",
                   inference_compile="gs://test/infer"),
        trainer=dict(hosts=1, tp=1, fsdp=4, process_bounds="1,1,1"))
    raw.update(overrides)
    return Config.from_dict(raw)


ROOT = Path("/private/ray-test")
IPS = ["10.0.0.1", "10.0.0.3", "10.0.0.5", "10.0.0.7"]
ENGINE_IPS = IPS[1:]


@pytest.mark.parametrize("profile,zone,hosts", [
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
    assert task["resources"]["accelerator_args"]["runtime_version"] == "v2-alpha-tpuv5"
    if hosts == 4:
        assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp) == (1, 1, 4)
        assert cfg.inference_hosts == 3
        assert cfg.trainer.process_bounds == "1,1,1"
        env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:1], 0)
        assert env["TPU_PROCESS_ADDRESSES"] == "10.0.0.1:19804"


def test_v5p_profile_preserves_training_and_checkpoint_identity():
    cfg = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json")
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    # Legacy v5p-32 qwen engine: 128 sequences, 90% HBM, 8192-token batches.
    assert cmd[cmd.index("--max-num-seqs") + 1] == "128"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "8192"
    assert "seq128-mem90-chunk8192" in cfg.cache.inference_compile
    assert cfg.cache.inference_compile_seed is None or cfg.cache.inference_compile_seed == ""
    assert cfg.run_id == "qwen-ray-v5p-32-001"
    assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp, cfg.inference_hosts) == (1, 1, 4, 3)
    assert cfg.trainer.remat == "full"


@pytest.mark.parametrize("profile", ["qwen_v5p_32", "gptoss120b_v5p_32_grpo"])
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
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:1])
    # Legacy single-host v5p-32 qwen cell: tp 1 x fsdp 4 on one host.
    assert backend["maxtext_kwargs"] == dict(ici_tensor_parallelism=1, ici_fsdp_parallelism=4,
        ici_context_parallelism=1, remat_policy="full", num_vocab_tiling=64,
        attention="autoselected", use_tokamax_splash=True, jax_cache_dir=str(ROOT / "ram/compile"))
    assert backend["vllm_lora_upload_endpoint"] == "/skyrl/v1/upload_lora_adapter"
    assert backend.get("num_processes", 1) == 1
    cmd = trainer_command(cfg, ROOT, ROOT / "source", IPS[0], IPS[:1], 0)
    assert cmd[cmd.index("--checkpoints-base")+1] == str(ROOT / "runs" / cfg.run_id / "checkpoints")
    assert backend["checkpoint_mirror_gcs"] == cfg.run_gcs + "/checkpoints"
    assert json.loads(cmd[-1]) == backend


def test_trainer_rank_and_bounds_use_selected_physical_row():
    # Two-host trainer (gpt-oss shape): rank 1 of a 1,1,2 process grid.
    cfg = config(trainer=dict(hosts=2, tp=4, fsdp=2, process_bounds="1,1,2"))
    env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:2], 1)
    assert env["CLOUD_TPU_TASK_ID"] == "1"
    assert env["TPU_PROCESS_ADDRESSES"] == ",".join(ip+":19804" for ip in IPS[:2])
    assert env["TPU_PROCESS_BOUNDS"] == "1,1,2"
    cmd = trainer_command(cfg, ROOT, ROOT / "source", IPS[0], IPS[:2], 1)
    assert "skyrl.backends.rpc" in cmd
    assert cmd[cmd.index("--process-id")+1] == "1"


def test_inference_isolation_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "bad:7777")
    monkeypatch.setenv("TPU_MULTIPROCESS_DP", "1")
    cfg = config(inference=dict(max_sequences=16, chunk_tokens=4096, chunked_prefill=True, prefix_caching=False))
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    for key, value in (("--tensor-parallel-size", "4"), ("--max-num-seqs", "16"),
                       ("--gpu-memory-utilization", "0.9"), ("--max-num-batched-tokens", "4096")):
        assert cmd[cmd.index(key)+1] == value
    assert "--enable-chunked-prefill" in cmd and "--enable-prefix-caching" not in cmd
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
    # Legacy qwen cell: 18432-token rows on a 22528 MaxText length.
    assert backend["maxtext_max_target_length"] == 22528
    assert backend["train_token_budget"] == 4 * 18432
    assert "s18432" in cfg.cache.trainer_compile
    client = client_environment(cfg, ROOT, IPS[0])
    assert client["TTD_M0_TRAIN_MAX_SEQ"] == "18432"
    assert client["TTD_M0_CONTEXT_WINDOW"] == "18432"
    assert cfg.inference.max_model_length == 22528
    assert (cfg.inference.max_sequences, cfg.inference.memory_utilization) == (128, 0.9)


@pytest.mark.parametrize("profile", sorted(p.stem for p in Path("tpu/swarm/ray_train/profiles").glob("qwen_v5p_32*.json")))
def test_v5p_profiles_inherit_the_legacy_cell_defaults(profile):
    """Every v5p-32 qwen profile is the legacy v5p-32 cell plus only what defines
    its experiment (mix rank, carry source, bundle). No profile may re-enable the
    pre-parity executor behaviour (ingress routing, no prefix caching, 4096-token
    chunked prefill, one LoRA slot, experimental kernels, no seq buckets)."""
    cfg = Config.load(f"tpu/swarm/ray_train/profiles/{profile}.json")
    assert cfg.model_preset == "qwen3.5-27b"
    assert cfg.inference.routing == "direct"
    assert cfg.inference.prefix_caching and not cfg.inference.chunked_prefill
    assert cfg.inference.chunk_tokens == 8192 and cfg.inference.max_loras == 8
    assert not (cfg.inference.skip_precompile or cfg.inference.batched_rpa_kernel or cfg.inference.ragged_conv1d)
    assert cfg.inference.max_sequences == 128
    assert cfg.trainer.seq_buckets == "4096,8192,12288,16384,20480" and cfg.trainer.minimal_fb_output
    assert cfg.trainer.effective_max_target_length == 22528
    assert (cfg.trainer.lora_load_retries, cfg.trainer.lora_load_retry_sleep, cfg.trainer.request_timeout) == (3, 2.0, 300)
    assert cfg.trainer.max_concurrent_requests == 256
    assert not cfg.client_hf_offline
    assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp, cfg.trainer.process_bounds) == (1, 1, 4, "1,1,1")
    assert "v5p32-cells" in cfg.base_bundle
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:cfg.trainer.hosts], ENGINE_IPS)
    assert backend["vllm_base_url"] == ",".join(f"http://{ip}:19801" for ip in ENGINE_IPS)
    assert backend["vllm_client_side_round_robin"]


# ---------------------------------------------------------------- legacy parity


def test_qwen_preset_reproduces_the_legacy_vllm_command_line():
    """tpu/start_vllm_tpu.sh + cell_worker.sh qwen case: prefix caching on, 8 LoRA
    slots, 128 sequences, 22528 context, 8192 batched tokens at 0.90, no chunked
    prefill flag, the text-only multimodal limit."""
    cfg = v5p_config()
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    assert "--enable-prefix-caching" in cmd and "--enable-chunked-prefill" not in cmd
    for key, value in (("--tensor-parallel-size", "4"), ("--max-model-len", "22528"),
                       ("--max-num-seqs", "128"), ("--max-loras", "8"), ("--max-lora-rank", "32"),
                       ("--max-num-batched-tokens", "8192"), ("--gpu-memory-utilization", "0.9"),
                       ("--limit-mm-per-prompt", '{"image":0,"video":0}'),
                       ("--download-dir", str(ROOT / "ram/hf/hub"))):
        assert cmd[cmd.index(key)+1] == value, key
    env = inference_environment(cfg, ROOT, ROOT / "run")
    assert (env["SKIP_JAX_PRECOMPILE"], env["USE_BATCHED_RPA_KERNEL"], env["USE_JAX_RAGGED_CONV1D"]) == ("0", "0", "0")
    assert (env["TPU_BACKEND_TYPE"], env["MODEL_IMPL_TYPE"]) == ("torchax", "vllm")
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in env


def test_trainer_and_engine_match_the_live_v5p_cell_process_environment():
    """Diffed 2026-09-08 against legacy cell 390 (gemma) on worker 154: the
    trainer API ran with --external-inference-timeout-sec 7200 and no
    SKYRL_EXTERNAL_WATCHDOG_* overrides; the engine ran with
    VLLM_USE_RAY_EXECUTOR=0 and MODEL_IMPL_TYPE=vllm."""
    cfg = v5p_config(model_preset="gemma4-31b")
    cmd = trainer_command(cfg, ROOT, ROOT / "source", IPS[0], IPS[:1], 0, ENGINE_IPS)
    assert cmd[cmd.index("--external-inference-timeout-sec") + 1] == "7200"
    env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:1], 0)
    assert not any(k.startswith("SKYRL_EXTERNAL_WATCHDOG_") for k in env)
    engine = inference_environment(cfg, ROOT, ROOT / "run")
    assert engine["VLLM_USE_RAY_EXECUTOR"] == "0" and engine["MODEL_IMPL_TYPE"] == "vllm"
    assert (engine["SKIP_JAX_PRECOMPILE"], engine["USE_BATCHED_RPA_KERNEL"], engine["USE_JAX_RAGGED_CONV1D"]) == ("0", "0", "0")


def test_qwen_preset_reproduces_the_legacy_trainer_contract():
    """start_colocated_vllm_tinker.sh backend config for a qwen cell: 22528 MaxText
    length over 18432-token rows, 73728 budget, tile 512, client-side round robin to
    every engine, 256 in flight, 300 s requests, 3 x 2 s adapter retries, and the
    seq-bucket / minimal-output env cell_worker.sh forwards."""
    cfg = v5p_config()
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:1], ENGINE_IPS)
    assert backend["maxtext_max_target_length"] == 22528
    assert backend["train_token_budget"] == 73728 and backend["flce_tile_size"] == 512
    assert backend["vllm_base_url"] == ",".join(f"http://{ip}:19801" for ip in ENGINE_IPS)
    assert backend["vllm_client_side_round_robin"] and not backend["vllm_route_by_prompt_prefix"]
    assert backend["vllm_max_concurrent_requests"] == 256
    assert backend["vllm_request_timeout_sec"] == 300
    assert (backend["vllm_lora_load_retries"], backend["vllm_lora_load_retry_sleep_sec"]) == (3, 2.0)
    assert backend["vllm_lora_load_endpoint"] == "/v1/load_lora_adapter"
    assert backend["maxtext_kwargs"] == dict(ici_tensor_parallelism=1, ici_fsdp_parallelism=4,
        ici_context_parallelism=1, remat_policy="full", num_vocab_tiling=64,
        attention="autoselected", use_tokamax_splash=True, jax_cache_dir=str(ROOT / "ram/compile"))
    cmd = trainer_command(cfg, ROOT, ROOT / "source", IPS[0], IPS[:1], 0, ENGINE_IPS)
    assert cmd[cmd.index("--external-inference-url")+1] == backend["vllm_base_url"]
    env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:1], 0)
    assert env["TUNIX_SEQ_BUCKETS"] == "4096,8192,12288,16384,20480"
    assert env["TUNIX_MINIMAL_FB_OUTPUT"] == "1"
    assert env["TUNIX_UNIFORM_SEQ_LEN"] == "18432"


def test_qwen_preset_reproduces_the_legacy_client_environment():
    """launch_cell.sh: qwen member spec, 1.5e-4, 18432/13824 budgets, HF online."""
    env = client_environment(v5p_config(), ROOT, IPS[0])
    assert env["TTD_ENSEMBLE_MODELS"] == "Qwen/Qwen3.5-27B:qwen3:qwen"
    assert env["LEARNING_RATE"] == "1.5e-4"
    assert (env["TTD_M0_CONTEXT_WINDOW"], env["TTD_M0_PHASE1_MAX_TOKENS"]) == ("18432", "13824")
    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["TTD_QWEN_TWO_PHASE"] == "1" and env["TTD_PROBLEM_TYPE"] == ""


def test_gemma_preset_carries_its_legacy_values():
    cfg = v5p_config(model_preset="gemma4-31b")
    assert cfg.model == "google/gemma-4-31B-it"
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    assert cmd[cmd.index("--max-num-seqs")+1] == "32" and cmd[cmd.index("--max-model-len")+1] == "16384"
    assert "--disable-chunked-mm-input" in cmd
    assert cmd[cmd.index("--limit-mm-per-prompt")+1] == '{"image":0,"audio":0,"video":0}'
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:1], ENGINE_IPS)
    assert (backend["maxtext_max_target_length"], backend["train_token_budget"], backend["flce_tile_size"]) == (10240, 40960, 1024)
    assert backend["maxtext_kwargs"]["num_vocab_tiling"] == 32 and backend["maxtext_kwargs"]["allow_split_physical_axes"]
    client = client_environment(cfg, ROOT, IPS[0])
    assert client["TTD_ENSEMBLE_MODELS"] == "google/gemma-4-31B-it:gemma4:gemma"
    assert (client["TTD_M0_CONTEXT_WINDOW"], client["TTD_M0_PHASE1_MAX_TOKENS"], client["LEARNING_RATE"]) == ("10240", "6656", "4e-5")


def test_muse_preset_carries_its_legacy_engine_flags():
    cfg = v5p_config(model_preset="muse-glimmer-30b")
    env = inference_environment(cfg, ROOT, ROOT / "run")
    assert (env["SKIP_JAX_PRECOMPILE"], env["USE_BATCHED_RPA_KERNEL"], env["USE_JAX_RAGGED_CONV1D"]) == ("1", "1", "1")
    assert env["TPU_BACKEND_TYPE"] == "jax"
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:1], ENGINE_IPS)
    assert (backend["vllm_lora_load_retries"], backend["vllm_lora_load_retry_sleep_sec"], backend["vllm_request_timeout_sec"]) == (20, 30.0, 1800)
    assert backend["maxtext_kwargs"]["parameter_memory_host_offload"]
    assert cfg.inference.transformers_version == ""


def test_gptoss_preset_two_trainer_hosts_and_engine_requantize_env():
    cfg = v5p_config(model_preset="gpt-oss-120b",
                     trainer=dict(hosts=2, tp=4, fsdp=2, process_bounds="1,1,2"))
    assert cfg.model == "openai/gpt-oss-120b" and cfg.inference_hosts == 2
    backend = trainer_backend_config(cfg, ROOT, IPS[0], IPS[:2], IPS[2:])
    kwargs = backend["maxtext_kwargs"]
    assert kwargs["sparse_matmul"] and kwargs["megablox"] and kwargs["allow_split_physical_axes"]
    assert "use_tokamax_splash" not in kwargs
    assert (kwargs["ici_tensor_parallelism"], kwargs["ici_fsdp_parallelism"]) == (4, 2)
    assert backend["num_processes"] == 2 and backend["coordinator_address"] == "10.0.0.1:19803"
    assert backend["vllm_base_url"] == "http://10.0.0.5:19801,http://10.0.0.7:19801"
    assert backend["train_token_budget"] == 2 * 18432
    assert cfg.trainer.maxtext_spec.endswith("@d388c5478b18b2322ab36c032deb87b9a4ff065f")
    assert cfg.trainer.ckpt_require_marker
    env = inference_environment(cfg, ROOT, ROOT / "run")
    assert env["MOE_REQUANTIZE_WEIGHT_DTYPE"] == "fp8" and env["USE_MOE_EP_KERNEL"] == "0"
    trainer_env = trainer_environment(cfg, ROOT, ROOT / "run", IPS[:2], 1)
    assert (trainer_env["TPU_PROCESS_BOUNDS"], trainer_env["CLOUD_TPU_TASK_ID"]) == ("1,1,2", "1")
    assert trainer_env["TUNIX_ROW_SHARD"] == "2"
    client = client_environment(cfg, ROOT, IPS[0])
    assert client["TTD_ENSEMBLE_MODELS"] == "openai/gpt-oss-120b:gpt_oss_high_reasoning:gptoss"
    assert client["LEARNING_RATE"] == "4e-5"
    cmd = inference_command(cfg, ROOT, ROOT / "source", ROOT / "model", ROOT / "run")
    assert "--limit-mm-per-prompt" not in cmd and cmd[cmd.index("--max-num-seqs")+1] == "32"


def test_preset_values_stay_overridable_and_validated():
    cfg = v5p_config(inference=dict(engine_env={"MOE_REQUANTIZE_WEIGHT_DTYPE": "bf16"}, max_sequences=64))
    assert inference_environment(cfg, ROOT, ROOT / "run")["MOE_REQUANTIZE_WEIGHT_DTYPE"] == "bf16"
    assert cfg.inference.max_sequences == 64
    with pytest.raises(ValueError, match="model_preset"):
        v5p_config(model_preset="nope")
    with pytest.raises(ValueError, match="process_bounds"):
        v5p_config(trainer=dict(hosts=2, tp=4, fsdp=2, process_bounds="1,1,1"))
    with pytest.raises(ValueError, match="one or two trainer hosts"):
        v5p_config(trainer=dict(hosts=3, tp=4, fsdp=3, process_bounds="1,1,3"))
    with pytest.raises(ValueError, match="routing"):
        v5p_config(inference=dict(routing="sideways"))
    with pytest.raises(ValueError, match="client_member_spec"):
        v5p_config(client_member_spec="Other/Model:qwen3:qwen")
    assert inference_urls(v5p_config(inference=dict(routing="ingress")), IPS[0], ENGINE_IPS) == "http://10.0.0.1:19800"
    assert set(PRESETS) == {"qwen3.5-27b", "gemma4-31b", "muse-glimmer-30b", "gpt-oss-120b"}


def test_trainer_env_reaches_only_the_trainer_process():
    cfg = _mix_config()
    env = trainer_environment(cfg, ROOT, ROOT / "runs/ray-mix", IPS[:1], 0)
    assert env["TUNIX_LORA_MIX_GAMMA"] == "0.9" and env["TUNIX_LORA_MIX_GAMMA_LR"] == "0.02"
    assert env["TUNIX_ROW_SHARD"] == "4"  # launcher settings still present
    client = client_environment(cfg, ROOT, IPS[0])
    assert "TUNIX_LORA_MIX_GAMMA" not in client
    assert "TUNIX_LORA_MIX_GAMMA" not in inference_environment(cfg, ROOT, ROOT / "runs/ray-mix")


def _mix_config(**overrides):
    raw = dict(run_id="ray-mix", accelerator="tpu-v5p-32", hosts=4,
        bucket="gs://test", base_bundle="gs://test/base.tar.gz", base_bundle_sha256="a"*64,
        cache=dict(hf="gs://test/hf", orbax="gs://test/orbax", trainer_compile="gs://test/train",
                   inference_compile="gs://test/infer"),
        trainer=dict(hosts=1, tp=1, fsdp=4, process_bounds="1,1,1", sequence_length=18432,
                     token_budget=73728, max_lora_rank=64),
        inference=dict(max_lora_rank=64),
        client_context_window=18432, client_phase1_max_tokens=13824,
        trainer_env={"TUNIX_LORA_MIX_GAMMA": "0.9", "TUNIX_LORA_MIX_GAMMA_LR": "0.02"})
    raw.update(overrides)
    return Config.from_dict(raw)


def test_trainer_max_lora_rank_defaults_to_lora_rank_and_covers_the_mix():
    plain = config()
    assert plain.trainer.effective_max_lora_rank == plain.trainer.lora_rank
    assert trainer_backend_config(plain, ROOT, IPS[0], IPS[:4])["max_lora_rank"] == plain.trainer.lora_rank
    mix = _mix_config()
    assert trainer_backend_config(mix, ROOT, IPS[0], IPS[:1])["max_lora_rank"] == 64
    cmd = inference_command(mix, ROOT, ROOT / "src", ROOT / "snap", ROOT / "run")
    assert cmd[cmd.index("--max-lora-rank") + 1] == "64"


def test_mix_requires_doubled_rank_on_both_sides():
    with pytest.raises(ValueError, match="2 x lora_rank"):
        _mix_config(trainer=dict(hosts=1, tp=1, fsdp=4, process_bounds="1,1,1", sequence_length=18432,
                                 token_budget=73728, max_lora_rank=0), inference=dict(max_lora_rank=64))
    with pytest.raises(ValueError, match="inference cannot load"):
        _mix_config(inference=dict(max_lora_rank=32))
    with pytest.raises(ValueError, match="max_lora_rank must be"):
        _mix_config(trainer=dict(hosts=1, tp=1, fsdp=4, process_bounds="1,1,1", sequence_length=18432,
                                 token_budget=73728, max_lora_rank=16), trainer_env={})
    with pytest.raises(ValueError, match="trainer environment"):
        _mix_config(trainer_env={"TUNIX_LORA_MIX_GAMMA": 0.9})
