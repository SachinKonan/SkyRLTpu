"""Serializable configuration; no TPU, Ray or JAX imports in the controller.

Defaults reproduce the legacy v5p-32 cell launcher (tpu/jobman/cell_worker.sh,
tpu/launch_cell.sh, tpu/start_colocated_vllm_tinker.sh, tpu/start_vllm_tpu.sh)
for the same model: a profile that names only the model preset produces the
same vLLM command line, engine environment, trainer backend config and client
environment as a legacy cell. Every knob stays overridable per profile; the
2026-09-07 v4-64 profiles pin their earlier values explicitly so their runs are
unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Ports:
    # Above SkyPilot workers (11002-19999), below Linux ephemeral ports.
    # Bootstrap validates the actual host ranges before starting Ray.
    ray: int = 24679
    dashboard: int = 24680
    client: int = 24681
    object_manager: int = 24700
    node_manager: int = 24701
    dashboard_agent: int = 24702
    dashboard_agent_grpc: int = 24703
    runtime_env: int = 24704
    metrics: int = 24705
    worker_min: int = 22000
    worker_max: int = 22999
    inference: int = 24800
    engine: int = 24801
    trainer: int = 24802
    trainer_jax: int = 24803
    trainer_tpu: int = 24804
    inference_tpu: int = 24805
    topology_jax: int = 24806
    topology_subset: int = 24807


@dataclass(frozen=True)
class Cache:
    hf: str = ""
    orbax: str = ""
    trainer_compile: str = ""
    inference_compile: str = ""
    inference_compile_seed: str = ""
    trainer_gib: int = 128
    inference_gib: int = 96
    reserve_gib: int = 128
    process_count: int = 4
    thread_count: int = 8
    slice_threshold: str = "0"
    sync_seconds: int = 60


@dataclass(frozen=True)
class Trainer:
    # Legacy v5p-32 cell shape: one trainer host (4 chips, fsdp 4) and three
    # single-host engines. Multi-host trainers (gpt-oss) override these.
    hosts: int = 1
    tp: int = 1
    fsdp: int = 4
    process_bounds: str = "1,1,1"
    chip_bounds: str = "2,2,1"
    maxtext_model: str = "qwen3.5-27b"
    maxtext_spec: str = "maxtext @ git+https://github.com/SachinKonan/maxtext.git@0fd409939977ac0ab79a4e64d21730936f253567"
    stacked_lora_training: bool = False
    stacked_lora_verify: bool = False
    lora_rank: int = 32
    # Largest adapter rank the trainer backend accepts; 0 means lora_rank.
    # The learnable carried/fresh LoRA mix (skyrl.backends.lora_mix) stores
    # both halves in one rank-2r adapter, so it needs 2 x lora_rank here and
    # on the inference side.
    max_lora_rank: int = 0
    # Uniform training row length (TUNIX_UNIFORM_SEQ_LEN).
    sequence_length: int = 22528
    # MaxText max_target_length. Legacy qwen cells train 18432-token rows on a
    # 22528 MaxText length; 0 means "same as sequence_length".
    max_target_length: int = 0
    token_budget: int = 45056
    flce_tile: int = 512
    remat: str = "full"
    logical_kv_heads: int = 8
    # Legacy cell_worker.sh forwards these to every trainer.
    seq_buckets: str = "4096,8192,12288,16384,20480"
    minimal_fb_output: bool = True
    free_base_state: bool = True
    # Opt in only after TPU validation for the model/mesh. Runs before rollout
    # generation and leaves weights, optimizer and gradient accumulators intact.
    backward_warmup: bool = False
    # Extra MaxText kwargs merged over the launcher-derived set (gpt-oss:
    # sparse_matmul/megablox; gemma: allow_split_physical_axes; muse: host
    # offload).
    maxtext_kwargs: dict = field(default_factory=dict)
    # Legacy qwen/gemma/muse kwargs carry tokamax splash attention; presets for
    # models that never ran with it (gpt-oss) turn it off.
    tokamax_splash: bool = True
    num_vocab_tiling: int = 64
    # Backend request fan-out to the engines (legacy VLLM_MAX_CONCURRENT_REQUESTS).
    max_concurrent_requests: int = 256
    # Adapter load retry policy (legacy VLLM_LORA_LOAD_RETRIES / _SLEEP_SEC).
    lora_load_retries: int = 3
    lora_load_retry_sleep: float = 2.0
    # Backend-side per-request timeout to the engines (legacy VLLM_REQUEST_TIMEOUT_SEC).
    request_timeout: int = 300
    # Require the orbax CHECKPOINT_COMPLETE marker (gpt-oss 120B conversion contract).
    ckpt_require_marker: bool = False
    # Extra exact pins installed with the MaxText fork. The legacy launcher adds
    # drjax next to MaxText because the d388 fork's DiLoCo sharding helper
    # imports it unconditionally on the model-creation path; the Qwen pin does
    # not need it, so it stays a preset choice rather than a global pin.
    extra_pins: list = field(default_factory=list)

    @property
    def effective_max_lora_rank(self):
        return self.max_lora_rank or self.lora_rank

    @property
    def effective_max_target_length(self):
        return self.max_target_length or self.sequence_length


@dataclass(frozen=True)
class Inference:
    # This is the backend validated by job 328, not native vLLM multi-host DP.
    backend: str = "ray_serve"
    tp: int = 4
    max_sequences: int = 128
    memory_utilization: float = 0.9
    max_model_length: int = 22528
    chunk_tokens: int = 8192
    max_lora_rank: int = 32
    # Hosts per vLLM engine. 1 = one single-host engine per inference host
    # (tp over its 4 chips). 2 = pipeline-parallel pairs: each host runs tp
    # over its own chips as an independent JAX cluster and vLLM's Ray executor
    # hands activations between the two (tpu-inference's TPU_MULTIHOST_BACKEND=ray
    # is pipeline parallelism, not cross-host tensor parallelism). gpt-oss-120b
    # needs 8 chips on 32 GiB v6e; 4 chips hold it on 95 GiB v5p.
    hosts_per_engine: int = 1
    # --external-inference-timeout-sec on the trainer API (legacy v5p-32 cell:
    # EXTERNAL_INFERENCE_TIMEOUT_SEC=7200; 21600 was the v4-64 launcher's).
    request_timeout: int = 7200
    restart_limit: int = 3
    max_loras: int = 8
    max_adapter_upload_bytes: int = 2 * 1024**3
    # Legacy start_vllm_tpu.sh always passes --enable-prefix-caching and never
    # --enable-chunked-prefill; the 2026-09-07 profiles had the opposite.
    # False means omit the flag, retaining the installed vLLM default.
    prefix_caching: bool = True
    chunked_prefill: bool = False
    # Engine environment (legacy per-model values: qwen/gemma 0/0/0, muse 1/1/1).
    skip_precompile: bool = False
    batched_rpa_kernel: bool = False
    ragged_conv1d: bool = False
    tpu_backend: str = "torchax"
    model_impl: str = "vllm"
    limit_mm_per_prompt: str = '{"image":0,"video":0}'
    extra_args: list[str] = field(default_factory=list)
    # Extra engine environment (e.g. MOE_REQUANTIZE_WEIGHT_DTYPE for gpt-oss).
    engine_env: dict[str, str] = field(default_factory=dict)
    custom_token_buckets: str = ""
    serialize_model_and_sampling: bool = False
    # Muse needs the native model registration plugin as well as LoRA plugins.
    # Remove the allow-list after applying inherited/profile environment values.
    unset_plugins: bool = False
    plugins: str = "lora_filesystem_resolver"
    native_thinking_budget: bool = False
    transformers_version: str = "5.8.0"
    # "direct": the trainer round-robins straight to the engines and pushes
    # adapters to each one (legacy). "ingress": every request and adapter goes
    # through the Ray Serve ingress (the 2026-09-07 executor behaviour).
    routing: str = "direct"


@dataclass(frozen=True)
class ModelPreset:
    hf_model: str
    maxtext_model: str
    maxtext_spec: str
    member_spec: str
    learning_rate: str
    trainer: dict
    inference: dict
    client_context_window: int
    client_phase1_max_tokens: int


_QWEN_SPEC = "maxtext @ git+https://github.com/SachinKonan/maxtext.git@0fd409939977ac0ab79a4e64d21730936f253567"
_MUSE_SPEC = "maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba509"
_GPTOSS_SPEC = "maxtext @ git+https://github.com/SachinKonan/maxtext.git@d388c5478b18b2322ab36c032deb87b9a4ff065f"

# Zones each accelerator is launched in (first = default) and the TPU VM runtime.
ACCELERATOR_ZONES = {
    "tpu-v4-64": ("us-central2-b",),
    "tpu-v4-32": ("us-central2-b",),
    "tpu-v5p-32": ("us-east5-a",),
    "tpu-v6e-32": ("asia-northeast1-b", "us-east5-b", "europe-west4-a"),
}
ACCELERATOR_RUNTIME = {
    "tpu-v4-64": "tpu-ubuntu2204-base",
    "tpu-v4-32": "tpu-ubuntu2204-base",
    "tpu-v5p-32": "v2-alpha-tpuv5",
    "tpu-v6e-32": "v2-alpha-tpuv6e",
}

# Values transcribed from tpu/jobman/cell_worker.sh (model cases + pick_tiles)
# and tpu/launch_cell.sh (member spec, context, phase-1 budget). Single-trainer
# v5p-32 shapes; multi-host profiles override tp/fsdp/bounds.
PRESETS = {
    "qwen3.5-27b": ModelPreset(
        hf_model="Qwen/Qwen3.5-27B", maxtext_model="qwen3.5-27b", maxtext_spec=_QWEN_SPEC,
        member_spec="Qwen/Qwen3.5-27B:qwen3:qwen", learning_rate="1.5e-4",
        trainer=dict(sequence_length=18432, max_target_length=22528, token_budget=73728,
                     flce_tile=512, num_vocab_tiling=64),
        inference=dict(max_sequences=128, max_model_length=22528, chunk_tokens=8192,
                       limit_mm_per_prompt='{"image":0,"video":0}'),
        client_context_window=18432, client_phase1_max_tokens=13824),
    "gemma4-31b": ModelPreset(
        hf_model="google/gemma-4-31B-it", maxtext_model="gemma4-31b", maxtext_spec=_QWEN_SPEC,
        member_spec="google/gemma-4-31B-it:gemma4:gemma", learning_rate="4e-5",
        trainer=dict(sequence_length=10240, max_target_length=10240, token_budget=40960,
                     flce_tile=1024, num_vocab_tiling=32,
                     maxtext_kwargs={"allow_split_physical_axes": True}),
        inference=dict(max_sequences=32, max_model_length=16384, chunk_tokens=8192,
                       limit_mm_per_prompt='{"image":0,"audio":0,"video":0}',
                       extra_args=["--disable-chunked-mm-input"]),
        client_context_window=10240, client_phase1_max_tokens=6656),
    "muse-glimmer-30b": ModelPreset(
        # NOTE: legacy muse serves 2 x TP2 engines per host; the executor runs
        # one TP4 engine per host, so this preset is not byte-identical there.
        hf_model="meta-models/Muse-Glimmer-30B", maxtext_model="muse-glimmer-30b", maxtext_spec=_MUSE_SPEC,
        member_spec="meta-models/Muse-Glimmer-30B:muse_glimmer_high_reasoning:muse", learning_rate="4e-5",
        trainer=dict(sequence_length=18432, max_target_length=18432, token_budget=73728,
                     flce_tile=1024, num_vocab_tiling=32,
                     maxtext_kwargs={"parameter_memory_host_offload": True},
                     lora_load_retries=20, lora_load_retry_sleep=30.0, request_timeout=1800),
        inference=dict(max_sequences=64, max_model_length=22528, chunk_tokens=8192,
                       skip_precompile=True, batched_rpa_kernel=True, ragged_conv1d=True,
                       tpu_backend="jax", limit_mm_per_prompt="", transformers_version="",
                       unset_plugins=True),
        client_context_window=18432, client_phase1_max_tokens=13824),
    "gpt-oss-120b": ModelPreset(
        # Trainer: orbax export from the d388 MaxText fork with sparse expert
        # LoRA (docs/gpt-oss-mxfp4-lora.md). Inference: native MXFP4 checkpoint,
        # experts requantized at load; v5p has no FP8 MXU, so the requantize
        # dtype is the knob to try (fp8 storage first, bf16 fallback).
        hf_model="openai/gpt-oss-120b", maxtext_model="gpt-oss-120b", maxtext_spec=_GPTOSS_SPEC,
        member_spec="openai/gpt-oss-120b:gpt_oss_high_reasoning:gptoss", learning_rate="4e-5",
        trainer=dict(sequence_length=18432, max_target_length=18432, token_budget=36864,
                     flce_tile=512, num_vocab_tiling=64, tokamax_splash=False,
                     maxtext_kwargs={"sparse_matmul": True, "megablox": True,
                                     "allow_split_physical_axes": True},
                     ckpt_require_marker=True, extra_pins=["drjax==0.2.1"]),
        inference=dict(max_sequences=32, max_model_length=22528, chunk_tokens=8192,
                       limit_mm_per_prompt="",
                       engine_env={"MOE_REQUANTIZE_WEIGHT_DTYPE": "fp8",
                                   "MOE_REQUANTIZE_BLOCK_SIZE": "512",
                                   "USE_MOE_EP_KERNEL": "0"}),
        client_context_window=18432, client_phase1_max_tokens=13824),
}


@dataclass(frozen=True)
class Config:
    run_id: str
    accelerator: str
    hosts: int
    bucket: str
    base_bundle: str
    base_bundle_sha256: str
    model_preset: str = "qwen3.5-27b"
    # GCP zone the task targets; empty selects the accelerator's default
    # (ACCELERATOR_ZONES[accelerator][0]).
    zone: str = ""
    model: str = "Qwen/Qwen3.5-27B"
    root: str = "~/.cache/skyrl-ray"
    adapter_count: int = 1
    pooled_group_size: int = 32
    importance_cap: float = 2.0
    # Reviewed inventory files relative to ray_train/runtime_baselines. Empty
    # means record/reuse verification only, never a claim of legacy parity.
    runtime_baselines: dict[str, str] = field(default_factory=dict)
    inference_only: bool = False
    inference_only_ranks: list[int] | None = None
    arena_samples: int = 0
    frozen_benchmark: dict = field(default_factory=dict)
    training_smoke: bool = False
    # Opt-in, pre-optimizer science seed discovery. Existing profiles stay unchanged.
    bootstrap_layers: int = 0
    bootstrap_all_hosts: bool = False
    bootstrap_only: bool = False
    seed_pool_sha256: str = ""
    arena_service_only: bool = False
    arena_grader_rank: int | None = None
    # One independent base model on each inference rank, for arena comparisons.
    # Each entry specifies model_preset plus its own inference/cache overrides.
    arena_models: list[dict] = field(default_factory=list)
    arena_replicas_per_model: int = 1
    arena_concurrency: int = 8
    arena_max_tokens: int = 8192
    arena_thinking_tokens: int | None = None
    ports: Ports = field(default_factory=Ports)
    systemd_runtime: bool = False
    systemd_port: int = 24900
    systemd_peer_timeout: int = 30
    systemd_shutdown_grace: int = 90
    max_restarts_on_errors: int = 0
    checkpoint_resume: bool = False
    resume_min_checkpoint_step: int = 0
    cache: Cache = field(default_factory=Cache)
    trainer: Trainer = field(default_factory=Trainer)
    inference: Inference = field(default_factory=Inference)
    log_seconds: int = 30
    setup_timeout: int = 7200
    ready_timeout: int = 3600
    checkpoint_cleanup_timeout: int = 600
    client_context_window: int = 18432
    client_phase1_max_tokens: int = 13824
    # TTD_ENSEMBLE_MODELS "model:renderer:tag" (from the preset unless set).
    client_member_spec: str = "Qwen/Qwen3.5-27B:qwen3:qwen"
    client_learning_rate: str = "1.5e-4"
    stacked_probe: bool = False
    attention_replay: bool = False
    attention_replay_fixture: str = "gemma_attention_replay.json"
    # Legacy launch_cell.sh runs the client with HF_HUB_OFFLINE=0 and fetches
    # the tokenizer; the executor's restored snapshot can serve it offline.
    client_hf_offline: bool = False
    client_env: dict[str, str] = field(default_factory=dict)
    # Opt-in throughput setting; candidate CPU/RAM/time budgets are unchanged.
    science_routing_slots_per_host: int = 2
    # Opt-in CPU placement keeps historical TPU profiles reproducible.
    science_placement_backend: str = "tpu"
    science_placement_slots_per_host: int = 2
    # Extra environment for the trainer (Tinker API server) process only,
    # e.g. TUNIX_LORA_MIX_GAMMA. Applied after the launcher's own settings.
    trainer_env: dict[str, str] = field(default_factory=dict)
    retired_task_ids: list[str] = field(default_factory=list)
    retired_processes: dict = field(default_factory=dict)

    @property
    def science_task(self):
        return {"science_routing": "routing", "science_placement": "placement"}.get(self.client_env.get("TTD_ENV"))

    @property
    def ray_cpus_per_host(self):
        # Leave scheduler capacity for controller/Serve actors as well as graders.
        if self.science_task == 'placement' and self.science_placement_backend == 'cpu':
            return max(32, 4 * self.science_placement_slots_per_host + 8)
        return max(32, 4 * self.science_routing_slots_per_host + 8) if self.science_task == 'routing' else 32

    @property
    def is_recurrent_gemma(self):
        return self.client_env.get("TTD_ENV") in (
            "recurrent_gemma", "recurrentgemma", "pallas_rglru", "rg_lru")

    @property
    def requires_source_overlay(self):
        return (not self.inference_only or self.adapter_count > 1 or self.is_recurrent_gemma or self.training_smoke
                or self.has_problem_prompt_overlay or self.has_adaptive_pwc_overlay
                or self.has_answer_only_overlay or self.trainer.backward_warmup
                or self.inference.hosts_per_engine > 1
                or (self.inference.native_thinking_budget and not self.inference_only))

    @property
    def has_answer_only_overlay(self):
        return self.client_env.get("TTD_ANSWER_ONLY_CODE") == "1"

    @property
    def has_adaptive_pwc_overlay(self):
        return self.client_env.get("TTD_ADV_ESTIMATOR") == "piecewise_valid_entropic_centered_adaptive"

    @property
    def has_problem_prompt_overlay(self):
        env = self.client_env.get("TTD_ENV")
        return (env == "circle_packing" or
                env == "ac_inequalities" and self.client_env.get("TTD_PROBLEM_TYPE") == "ac2")

    @classmethod
    def from_dict(cls, raw):
        data = dict(raw)
        preset_name = data.get("model_preset", "qwen3.5-27b")
        if preset_name not in PRESETS:
            raise ValueError(f"unknown model_preset {preset_name!r}; known: {sorted(PRESETS)}")
        preset = PRESETS[preset_name]
        data.setdefault("model", preset.hf_model)
        data.setdefault("client_context_window", preset.client_context_window)
        data.setdefault("client_phase1_max_tokens", preset.client_phase1_max_tokens)
        data.setdefault("client_member_spec", preset.member_spec)
        data.setdefault("client_learning_rate", preset.learning_rate)
        trainer = dict(maxtext_model=preset.maxtext_model, maxtext_spec=preset.maxtext_spec)
        trainer.update(preset.trainer)
        trainer.update(data.get("trainer") or {})
        data["trainer"] = trainer
        inference = dict(preset.inference)
        inference.update(data.get("inference") or {})
        data["inference"] = inference
        for key, kind in (("ports", Ports), ("cache", Cache), ("trainer", Trainer),
                          ("inference", Inference)):
            data[key] = kind(**(data.get(key) or {}))
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.science_placement_backend not in ('cpu', 'tpu'):
            raise ValueError('placement backend must be cpu or tpu')
        placement_slots = self.science_placement_slots_per_host
        if type(placement_slots) is not int or not 1 <= placement_slots <= 16:
            raise ValueError('placement CPU slots must be an integer in [1,16]')
        if self.science_placement_backend == 'cpu':
            if self.science_task != 'placement':
                raise ValueError('CPU placement backend requires science placement')
            if max(self.cache.trainer_gib, self.cache.inference_gib) > 128:
                raise ValueError('CPU placement requires RAM caches capped at 128 GiB')
        elif placement_slots != 2:
            raise ValueError('placement CPU slots require the CPU backend')
        slots = self.science_routing_slots_per_host
        if type(slots) is not int or not 1 <= slots <= 16:
            raise ValueError('routing grading slots must be an integer in [1,16] (128 GiB maximum)')
        if slots != 2 and self.science_task != 'routing':
            raise ValueError('routing grading slots only apply to science routing')
        if slots > 2 and max(self.cache.trainer_gib, self.cache.inference_gib) > 128:
            raise ValueError('expanded routing grading requires RAM caches capped at 128 GiB')
        if type(self.bootstrap_only) is not bool:
            raise ValueError('bootstrap_only must be a boolean')
        if self.seed_pool_sha256 and (not re.fullmatch(r'[0-9a-f]{64}', self.seed_pool_sha256)
                or not self.science_task or self.inference_only or self.bootstrap_layers):
            raise ValueError('imported seed checksum requires science training without local bootstrap')
        if type(self.bootstrap_layers) is not int or self.bootstrap_layers not in (0, 1, 2):
            raise ValueError('bootstrap_layers must be 0, 1, or 2')
        if self.bootstrap_all_hosts and not self.bootstrap_layers:
            raise ValueError('bootstrap_all_hosts requires bootstrap_layers')
        if self.bootstrap_only and (not self.bootstrap_layers or not self.inference_only
                or self.science_task != 'routing' or self.accelerator != 'tpu-v4-32'
                or self.trainer.hosts != 0 or self.inference_only_ranks is not None
                or self.frozen_benchmark or self.arena_samples or self.arena_service_only):
            raise ValueError('bootstrap-only requires routing on all four v4-32 inference hosts and no trainer')
        if self.bootstrap_layers and (not self.science_task
                or (self.inference_only and not self.bootstrap_only)
                or self.adapter_count != 1 or (self.accelerator not in ('tpu-v4-64', 'tpu-v6e-32') and not self.bootstrap_only)
                or self.inference.hosts_per_engine != 1 or self.inference.tp != 4
                or not self.inference.native_thinking_budget or self.inference.routing != 'ingress'
                or self.client_env.get('TTD_MIN_VALID_PER_GROUP', '0') != '0'):
            raise ValueError('bootstrap requires single-model v4-64/v6e-32 science training or v4-32 routing seeds with native TP4 ingress')
        if self.has_answer_only_overlay:
            if not (self.has_problem_prompt_overlay or self.is_recurrent_gemma or self.science_task):
                raise ValueError("answer-only extraction requires a supported math, RG-LRU or science environment")
            if self.client_env.get("TTD_ANSWER_MODEL_FAMILY") not in ("qwen", "gemma", "muse"):
                raise ValueError("answer-only extraction requires an explicit model family")
        if self.has_adaptive_pwc_overlay and not (self.has_problem_prompt_overlay or self.is_recurrent_gemma):
            raise ValueError("adaptive PWC is enabled only for AC2, circle packing and RG-LRU")
        if self.training_smoke and (self.inference_only or self.adapter_count != 1 or self.client_env.get('NUM_EPOCHS') != '1'):
            raise ValueError('training smoke requires one adapter and exactly one training step')
        if self.arena_grader_rank is not None:
            if (type(self.arena_grader_rank) is not int or self.arena_grader_rank != 1
                    or self.accelerator != "tpu-v5p-32" or self.hosts != 4
                    or self.trainer.hosts != 1 or not self.is_recurrent_gemma
                    or self.inference_only or self.inference_only_ranks is not None
                    or self.arena_samples or self.arena_service_only or self.frozen_benchmark):
                raise ValueError("Dedicated RG grader requires v5p-32 training: trainer 0, grader 1, inference 2/3")
            if self.client_env.get("ARENA_QUEUE_URL"):
                raise ValueError("Dedicated RG grader uses workload Ray tasks; remove ARENA_QUEUE_URL")
        if self.arena_service_only and (not self.inference_only or not self.arena_samples):
            raise ValueError('arena service requires an inference-only judge profile')
        if type(self.arena_samples) is not int or not 0 <= self.arena_samples <= 256:
            raise ValueError("arena_samples must be an integer between 0 and 256")
        if type(self.arena_replicas_per_model) is not int or self.arena_replicas_per_model not in (1, 2):
            raise ValueError("arena_replicas_per_model must be 1 or 2")
        arena_ranks = list(range(1, 3 * self.arena_replicas_per_model + 1))
        if self.arena_samples and (not self.inference_only or self.trainer.hosts != 0
                or self.inference_only_ranks != arena_ranks or self.hosts not in (4, 8)
                or max(arena_ranks) >= self.hosts):
            raise ValueError("Arena sampling requires one judge host and three equally replicated models")
        if self.arena_models:
            if not self.arena_samples or len(self.arena_models) != 3:
                raise ValueError("arena_models requires three models and arena sampling")
            names = [entry.get("model_preset") for entry in self.arena_models]
            if len(set(names)) != 3 or any(n not in PRESETS for n in names):
                raise ValueError("arena_models requires three distinct known presets")
            for rank in self.inference_only_ranks:
                child = self.for_inference_rank(rank)
                if child.inference.hosts_per_engine != 1 or child.inference.tp != 4:
                    raise ValueError("arena_models requires a single-host TP4 engine per model")
                if self.arena_concurrency > child.inference.max_sequences:
                    raise ValueError("arena concurrency exceeds engine capacity")
        if (type(self.arena_concurrency) is not int or self.arena_concurrency < 1
                or type(self.arena_max_tokens) is not int or self.arena_max_tokens < 1):
            raise ValueError("arena concurrency and token limit must be positive integers")
        if type(self.inference.native_thinking_budget) is not bool:
            raise ValueError("native_thinking_budget must be a boolean")
        if self.inference.native_thinking_budget:
            if self.model_preset not in ("qwen3.5-27b", "gemma4-31b", "muse-glimmer-30b"):
                raise ValueError("unsupported native thinking model")
            if not self.inference_only and (self.adapter_count != 1 or self.client_env.get("TTD_MIN_THINK_TOKENS", "0") != "0"):
                raise ValueError("native training currently requires one adapter and min_think_tokens=0")
        if self.arena_thinking_tokens is not None:
            if (type(self.arena_thinking_tokens) is not int or self.arena_thinking_tokens < 0
                    or self.arena_thinking_tokens + 128 >= self.arena_max_tokens):
                raise ValueError("thinking cap must reserve transition and answer tokens")
            if not self.arena_samples or not self.inference.native_thinking_budget:
                raise ValueError("arena thinking cap requires native thinking budget serving")
        import math
        if type(self.adapter_count) is not int or self.adapter_count < 1:
            raise ValueError("adapter_count must be a positive integer")
        if (type(self.pooled_group_size) is not int or self.pooled_group_size < self.adapter_count
                or self.pooled_group_size % self.adapter_count):
            raise ValueError("pooled_group_size must be divisible by adapter_count")
        if not math.isfinite(self.importance_cap) or self.importance_cap < 1:
            raise ValueError("importance_cap must be finite and >= 1")
        if self.inference.max_loras < self.adapter_count:
            raise ValueError("inference max_loras must cover every sampling adapter")
        if self.trainer.stacked_lora_training and self.adapter_count < 2:
            raise ValueError("stacked training requires multiple adapters")
        if self.trainer.stacked_lora_verify and not self.trainer.stacked_lora_training:
            raise ValueError("stacked replay verification requires stacked training")
        if self.stacked_probe and (not self.trainer.stacked_lora_verify or self.adapter_count != 2):
            raise ValueError("short stacked probe requires two adapters and replay verification")
        if self.attention_replay and (self.adapter_count != 1 or self.inference_only
                                      or self.stacked_probe or self.is_recurrent_gemma
                                      or self.model_preset not in ("gemma4-31b", "qwen3.5-27b", "muse-glimmer-30b")):
            raise ValueError("attention replay requires one Gemma, Qwen or Muse adapter and a training executor")
        if self.attention_replay and (Path(self.attention_replay_fixture).name != self.attention_replay_fixture
                                      or not self.attention_replay_fixture.endswith('.json')):
            raise ValueError("attention replay fixture must be a local JSON filename")
        if self.adapter_count > 1:
            if self.model not in ("Qwen/Qwen3.5-27B", "openai/gpt-oss-120b", "openai/gpt-oss-20b"):
                raise ValueError("pooled multi-LoRA supports Qwen3.5 and GPT-OSS")
            if self.trainer.minimal_fb_output or self.trainer_env.get("TUNIX_MINIMAL_FB_OUTPUT", "0") != "0":
                raise ValueError("pooled multi-LoRA requires full backward logprobs; set trainer.minimal_fb_output=false")
            if self.inference.routing != "ingress":
                raise ValueError("pooled multi-LoRA requires ingress routing for adapter publication")
            if self.trainer_env.get("TUNIX_LORA_MIX_GAMMA", "").strip():
                raise ValueError("pooled multi-LoRA cannot be combined with the carried/fresh LoRA mix")
            fields = self.client_member_spec.split(":")
            if len(fields) != 3 or fields[0] != self.model:
                raise ValueError("pooled multi-LoRA requires one model:renderer:tag member spec")
        if self.inference_only_ranks is not None:
            ranks = self.inference_only_ranks
            if (not self.inference_only or not isinstance(ranks, list) or not ranks
                    or any(type(r) is not int or not 0 <= r < self.hosts for r in ranks)
                    or len(set(ranks)) != len(ranks)):
                raise ValueError("inference_only_ranks requires unique valid ranks in inference-only mode")
        placement_ranks = self.placement_ranks
        if self.science_task:
            if ((not self.bootstrap_only and (self.inference_only
                    or self.accelerator not in ("tpu-v6e-32", "tpu-v4-64") or self.trainer.hosts != 4))
                    or self.arena_grader_rank is not None or placement_ranks or self.adapter_count != 1):
                raise ValueError("Science training requires v6e-32 or v4-64, four trainer hosts and dynamically assigned grading")
            # TP8 uses the checkpoint loader's repeat-interleaved KV heads,
            # with tied gradients/Adam moments and native-shape adapter export.
            if (self.model_preset == "muse-glimmer-30b" and self.trainer.tp not in (1, 2)
                    and not (self.trainer.tp == 8 and self.trainer.logical_kv_heads == 8)):
                raise ValueError("Muse has two KV heads: use TP1/TP2 or TP8 with eight tied logical KV heads")
            if (self.accelerator == "tpu-v4-64" and self.model_preset == "qwen3.5-27b"
                    and not self.inference.ragged_conv1d):
                raise ValueError("v4 Qwen science requires ragged_conv1d: true; the pinned Pallas convolution uses unsupported v4 unpacking")
            if self.client_env.get("TTD_EVAL_BACKEND") != "local" or self.client_env.get("NUM_CPUS_PER_TASK") != "4":
                raise ValueError("Science environments dispatch their own Ray tasks; use local outer evaluation and four CPUs")
        if placement_ranks:
            if self.arena_grader_rank is not None:
                raise ValueError("cannot combine RG-LRU and placement grading roles in one profile")
            from tpu.science.placement_slots import chips_from_env
            chips_from_env(self.client_env)
            if len(set(placement_ranks)) != len(placement_ranks) or any(r < 0 or r >= self.hosts for r in placement_ranks):
                raise ValueError("placement ranks must be unique valid host ranks")
            if self.inference_only:
                if not self.inference_only_ranks or set(placement_ranks) & set(self.inference_only_ranks):
                    raise ValueError("placement requires explicit disjoint inference ranks")
            elif self.accelerator != "tpu-v5p-32" or any(r < self.trainer.hosts for r in placement_ranks):
                raise ValueError("training with placement requires disjoint v5p-32 grading hosts")
            if self.inference_hosts < 1:
                raise ValueError("placement must leave at least one inference host")
        if type(self.checkpoint_cleanup_timeout) is not int or self.checkpoint_cleanup_timeout < 0:
            raise ValueError("checkpoint_cleanup_timeout must be a nonnegative integer (0 disables)")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", self.run_id):
            raise ValueError("run_id must be a safe unique name")
        expected = {"tpu-v4-64": 8, "tpu-v4-32": 4, "tpu-v5p-32": 4, "tpu-v6e-32": 8}
        if expected.get(self.accelerator) != self.hosts:
            raise ValueError("accelerator/host count mismatch")
        if self.zone and self.zone not in ACCELERATOR_ZONES.get(self.accelerator, ()):
            raise ValueError(f"zone {self.zone!r} is not a known zone for {self.accelerator}")
        if self.inference_only and self.trainer.hosts != 0:
            raise ValueError("inference-only profiles must declare zero trainer hosts")
        if not self.inference_only and not 0 < self.trainer.hosts < self.hosts:
            raise ValueError("require disjoint nonempty trainer and inference roles")
        if not self.inference_only and self.accelerator == "tpu-v4-64" and self.trainer.hosts != 4:
            raise ValueError("v4-64 currently requires the validated four-host row")
        if not self.inference_only and self.accelerator in ("tpu-v5p-32", "tpu-v4-32") and self.trainer.hosts not in (1, 2):
            raise ValueError("32-core profiles support one or two trainer hosts")
        if not self.inference_only and self.accelerator == "tpu-v6e-32" and (
                self.trainer.hosts != 4 or self.trainer.process_bounds != "2,2,1"):
            # The legacy v6e-32 cell (tpu/jobman/cell_worker.sh via
            # run_qwen35_v6e32_grpo.sh) trains on hosts 0-3 as a 2,2,1 process
            # grid and serves on 4-7; that is the only split validated on v6e.
            raise ValueError("v6e-32 profiles use the validated four-host 2,2,1 trainer block")
        if not self.inference_only and self.trainer.tp * self.trainer.fsdp != self.trainer.hosts * 4:
            raise ValueError("trainer TP x FSDP must cover exactly the trainer chips")
        if type(self.inference.hosts_per_engine) is not int or self.inference.hosts_per_engine not in (1, 2):
            raise ValueError("inference.hosts_per_engine must be 1 or 2")
        if self.inference_hosts % self.inference.hosts_per_engine:
            raise ValueError("inference hosts must form whole engines (hosts_per_engine)")
        if not self.inference_only:
            bounds = [int(x) for x in self.trainer.process_bounds.split(",")]
            if len(bounds) != 3 or bounds[0] * bounds[1] * bounds[2] != self.trainer.hosts:
                raise ValueError("trainer process_bounds must multiply to the trainer host count")
        if self.inference.tp not in (2, 4) or not 0 < self.inference.memory_utilization < 1:
            raise ValueError("require TP2/TP4 engines and a memory reserve")
        if self.inference.tp == 2 and (self.model_preset != "muse-glimmer-30b" or self.accelerator != "tpu-v5p-32"
                or self.inference.hosts_per_engine != 1 or self.inference.routing != "ingress" or self.arena_models):
            raise ValueError("TP2 requires single-host Muse v5p-32 engines through ingress")
        if self.inference.backend != "ray_serve":
            raise ValueError("native vLLM DP is not validated; no silent backend substitution")
        if self.inference.routing not in ("direct", "ingress"):
            raise ValueError("inference.routing must be 'direct' or 'ingress'")
        if type(self.inference.max_loras) is not int or self.inference.max_loras < 1:
            raise ValueError("inference.max_loras must be a positive integer")
        if type(self.inference.max_adapter_upload_bytes) is not int or self.inference.max_adapter_upload_bytes < 1:
            raise ValueError("max_adapter_upload_bytes must be a positive integer")
        if self.inference.max_sequences < 1 or self.inference.restart_limit < 0:
            raise ValueError("invalid inference limits")
        if self.inference.chunk_tokens < 1:
            raise ValueError("inference.chunk_tokens must be positive")
        if any(str(k) != k or str(v) != v for k, v in self.inference.engine_env.items()):
            raise ValueError("inference.engine_env must contain strings")
        if type(self.inference.unset_plugins) is not bool:
            raise ValueError("inference.unset_plugins must be a boolean")
        if not isinstance(self.inference.plugins, str):
            raise ValueError("inference.plugins must be a string")
        if not all(isinstance(a, str) for a in self.inference.extra_args):
            raise ValueError("inference.extra_args must contain strings")
        managed_flags = {"--tensor-parallel-size", "--pipeline-parallel-size", "--max-model-len",
                         "--max-num-seqs", "--max-num-batched-tokens", "--gpu-memory-utilization",
                         "--max-loras", "--max-lora-rank", "--port", "--host", "--served-model-name",
                         "--enable-prefix-caching", "--no-enable-prefix-caching",
                         "--enable-chunked-prefill", "--no-enable-chunked-prefill",
                         "--enable-lora", "--no-enable-lora", "--download-dir",
                         "--limit-mm-per-prompt", "--distributed-executor-backend",
                         "--data-parallel-size", "--skyrl-lora-dir", "--skyrl-ray-placement-hosts",
                         "--model"}
        aliases = {"-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size",
                   "-dp": "--data-parallel-size"}
        def normalized_flag(arg):
            flag = arg.split("=", 1)[0].replace("_", "-")
            return aliases.get(flag, flag)
        if any(normalized_flag(arg) in managed_flags for arg in self.inference.extra_args):
            raise ValueError("inference.extra_args conflicts with managed inference settings")
        if type(self.inference.serialize_model_and_sampling) is not bool:
            raise ValueError("serialize_model_and_sampling must be boolean")
        if not isinstance(self.inference.custom_token_buckets, str):
            raise ValueError("custom_token_buckets must be a string")
        if self.trainer.max_lora_rank < 0 or 0 < self.trainer.max_lora_rank < self.trainer.lora_rank:
            raise ValueError("trainer max_lora_rank must be 0 (= lora_rank) or >= lora_rank")
        if self.trainer.effective_max_lora_rank > self.inference.max_lora_rank:
            raise ValueError("inference cannot load the requested trainer LoRA rank")
        if self.trainer_env.get("TUNIX_LORA_MIX_GAMMA") and (
            self.trainer.effective_max_lora_rank < 2 * self.trainer.lora_rank
        ):
            raise ValueError("the LoRA mix stores two rank-r halves: set trainer.max_lora_rank and "
                             "inference.max_lora_rank to at least 2 x lora_rank")
        if self.trainer.max_target_length < 0 or 0 < self.trainer.max_target_length < self.trainer.sequence_length:
            raise ValueError("trainer max_target_length must be 0 (= sequence_length) or >= sequence_length")
        if self.trainer.sequence_length > self.inference.max_model_length:
            raise ValueError("inference context must cover trainer context")
        if min(self.trainer.max_concurrent_requests, self.trainer.lora_load_retries,
               self.trainer.request_timeout) < 1 or self.trainer.lora_load_retry_sleep < 0:
            raise ValueError("trainer request/retry settings must be positive")
        if not isinstance(self.trainer.maxtext_kwargs, dict):
            raise ValueError("trainer.maxtext_kwargs must be a mapping")
        if type(self.systemd_runtime) is not bool:
            raise ValueError("systemd_runtime must be boolean")
        if type(self.systemd_port) is not int or not 1024 <= self.systemd_port <= 65535:
            raise ValueError("invalid systemd coordinator port")
        if (type(self.systemd_peer_timeout) is not int or self.systemd_peer_timeout <= 0
                or type(self.systemd_shutdown_grace) is not int or self.systemd_shutdown_grace < 0):
            raise ValueError("invalid systemd lifecycle timeouts")
        if self.systemd_runtime:
            if (self.systemd_port in asdict(self.ports).values()
                    or self.ports.worker_min <= self.systemd_port <= self.ports.worker_max):
                raise ValueError("systemd coordinator port overlaps workload ports")
        if type(self.max_restarts_on_errors) is not int or not 0 <= self.max_restarts_on_errors <= 3:
            raise ValueError("max_restarts_on_errors must be an integer from 0 to 3")
        if type(self.checkpoint_resume) is not bool:
            raise ValueError("checkpoint_resume must be a boolean")
        if (type(self.resume_min_checkpoint_step) is not int or self.resume_min_checkpoint_step < 0
                or self.resume_min_checkpoint_step and not self.checkpoint_resume):
            raise ValueError("resume_min_checkpoint_step requires checkpoint_resume and a nonnegative integer")
        if not isinstance(self.runtime_baselines, dict) or set(self.runtime_baselines) - {"trainer", "serving", "client"}:
            raise ValueError("runtime_baselines supports trainer, serving and client inventories")
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name)
               for name in self.runtime_baselines.values()):
            raise ValueError("runtime baseline filenames must be plain JSON filenames")
        # commands.maxtext_kwargs merges these extras after the validated mesh.
        # Matching legacy declarations are harmless; conflicting ones would
        # silently launch a different mesh from the profile and row sharding.
        for key, expected in (("ici_tensor_parallelism", self.trainer.tp),
                              ("ici_fsdp_parallelism", self.trainer.fsdp),
                              ("ici_context_parallelism", 1)):
            if key in self.trainer.maxtext_kwargs:
                value = self.trainer.maxtext_kwargs[key]
                if type(value) is not int or value != expected:
                    raise ValueError(f"trainer.maxtext_kwargs.{key}={value!r} "
                                     f"conflicts with the trainer mesh (expected {expected})")
        if not isinstance(self.trainer.extra_pins, list) or not all(
                isinstance(p, str) and "==" in p for p in self.trainer.extra_pins):
            raise ValueError("trainer.extra_pins must be a list of exact 'name==version' pins")
        if min(self.cache.trainer_gib, self.cache.inference_gib) < 64 or self.cache.reserve_gib < 128:
            raise ValueError("RAM cache requires >=64 GiB cap and >=128 GiB runtime reserve")
        if min(self.cache.process_count, self.cache.thread_count, self.cache.sync_seconds,
               self.log_seconds, self.setup_timeout, self.ready_timeout) <= 0:
            raise ValueError("concurrency and timeouts must be positive")
        for path in (self.bucket, self.base_bundle, self.cache.hf, self.cache.orbax,
                     self.cache.trainer_compile, self.cache.inference_compile):
            if not path.startswith("gs://") or len(path.split("/")) < 3:
                raise ValueError(f"expected a GCS URI: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.base_bundle_sha256):
            raise ValueError("pin the frozen base bundle by SHA256")
        values = asdict(self.ports)
        lo, hi = values.pop("worker_min"), values.pop("worker_max")
        if not 1024 <= lo <= hi <= 65535:
            raise ValueError("invalid Ray worker port range")
        ports = list(values.values())
        if len(ports) != len(set(ports)) or any(not 1024 <= p <= 65535 for p in ports):
            raise ValueError("port collision or invalid port")
        if any(lo <= p <= hi or p in (6379, 6380, 16379) for p in ports):
            raise ValueError("ports overlap worker range or existing workload/SkyPilot Ray")
        if any(str(k) != k or str(v) != v for k, v in self.client_env.items()):
            raise ValueError("client environment must contain strings")
        if self.is_recurrent_gemma:
            from urllib.parse import urlsplit
            url = urlsplit(self.client_env.get("ARENA_QUEUE_URL", ""))
            if self.arena_grader_rank is None and (url.scheme not in ("http", "https") or not url.hostname
                    or url.username or url.password or url.query or url.fragment):
                raise ValueError("RecurrentGemma requires an ARENA_QUEUE_URL HTTP(S) endpoint without credentials")
            if self.client_env.get("TTD_PROBLEM_TYPE", "") not in ("", "rg_lru"):
                raise ValueError("RecurrentGemma only supports rg_lru")
            if self.stacked_probe or self.inference_only:
                raise ValueError("RecurrentGemma requires a training client, not an inference/stacked probe")
            timeout = float(self.client_env.get("EVAL_TIMEOUT", "3600"))
            wait = float(self.client_env.get("ARENA_WAIT_TIMEOUT", str(timeout)))
            if not 0 < wait <= timeout < float("inf"):
                raise ValueError("Arena requires 0 < ARENA_WAIT_TIMEOUT <= EVAL_TIMEOUT < infinity")
        if any(str(k) != k or str(v) != v for k, v in self.trainer_env.items()):
            raise ValueError("trainer environment must contain strings")
        if type(self.trainer.backward_warmup) is not bool:
            raise ValueError("backward_warmup must be boolean")
        if self.trainer.backward_warmup:
            from .warmup_contract import shapes
            if self.adapter_count != 1 or self.trainer_env.get("TUNIX_LORA_MIX_GAMMA"):
                raise ValueError("backward warmup requires a single unmixed adapter")
            if self.client_env.get("TTD_WARMUP_FB", "0") != "0":
                raise ValueError("disable legacy dummy FB when using state-preserving warmup")
            if self.client_env.get("TTD_LOSS_FN", "importance_sampling") != "importance_sampling":
                raise ValueError("backward warmup currently requires importance_sampling")
            shapes(self.trainer.sequence_length, self.trainer.token_budget, self.trainer.fsdp,
                   uniform=int(self.trainer_env.get("TUNIX_UNIFORM_SEQ_LEN", self.trainer.sequence_length)),
                   buckets=[int(n) for n in self.trainer_env.get("TUNIX_SEQ_BUCKETS", self.trainer.seq_buckets).split(",") if n.strip()])
        if "TUNIX_ROW_SHARD" in self.trainer_env:
            if self.trainer_env["TUNIX_ROW_SHARD"] != str(self.trainer.fsdp):
                raise ValueError("TUNIX_ROW_SHARD conflicts with trainer.fsdp")
        # Topology and request/database identities are executor-owned. Allow
        # compiler and diagnostic knobs, but never a second source of topology.
        owned = {"TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_PROCESS_ADDRESSES",
                 "TPU_PROCESS_PORT", "TPU_VISIBLE_CHIPS", "CLOUD_TPU_TASK_ID",
                 "SKYRL_TRAIN_PROCESS_ID", "SKYRL_DATABASE_URL", "SKYRL_FUTURE_BLOB_DIR"}
        if owned.intersection(self.trainer_env):
            raise ValueError("trainer_env cannot override executor topology or request storage")
        if owned.intersection(self.inference.engine_env):
            raise ValueError("engine_env cannot override executor topology")
        if {"VLLM_XLA_CACHE_PATH", "JAX_COMPILATION_CACHE_DIR", "VLLM_LORA_RESOLVER_CACHE_DIR",
            "VLLM_PLUGINS", "HF_HOME", "HF_HUB_OFFLINE", "RAY_ADDRESS", "RAY_NAMESPACE",
            "TPU_MULTIHOST_BACKEND", "VLLM_USE_RAY_EXECUTOR",
            "SKYRL_RAY_PLACEMENT_HOSTS"}.intersection(self.inference.engine_env):
            raise ValueError("engine_env cannot override executor cache, plugin or routing settings")
        flags = {"SKIP_JAX_PRECOMPILE": str(int(self.inference.skip_precompile)),
                 "USE_BATCHED_RPA_KERNEL": str(int(self.inference.batched_rpa_kernel)),
                 "USE_JAX_RAGGED_CONV1D": str(int(self.inference.ragged_conv1d)),
                 "CUSTOM_NUM_TOKENS_BUCKETS": self.inference.custom_token_buckets,
                 "SERIALIZE_MODEL_AND_SAMPLING": str(int(self.inference.serialize_model_and_sampling)),
                 "TPU_BACKEND_TYPE": self.inference.tpu_backend,
                 "MODEL_IMPL_TYPE": self.inference.model_impl}
        for key, expected in flags.items():
            if key in self.inference.engine_env and self.inference.engine_env[key] != expected:
                raise ValueError(f"engine_env {key} conflicts with inference configuration")
        if len(self.client_member_spec.split(":")) != 3 or not self.client_member_spec.startswith(self.model + ":"):
            raise ValueError("client_member_spec must be '<model>:<renderer>:<tag>' for the configured model")
        try:
            float(self.client_learning_rate)
        except ValueError:
            raise ValueError("client_learning_rate must be a float string") from None
        if not self.inference_only:
            budget = {key: int(value) for key, value in self.client_sampling_environment().items()}
            context = budget["TTD_M0_CONTEXT_WINDOW"]
            phase1 = budget["TTD_M0_PHASE1_MAX_TOKENS"]
            train_max = budget["TTD_M0_TRAIN_MAX_SEQ"]
            if not 0 < context <= train_max <= self.trainer.sequence_length:
                raise ValueError("client context must fit the client training limit and trainer sequence length")
            if not 0 < budget["CONTEXT_WINDOW"] <= self.inference.max_model_length:
                raise ValueError("global client context must fit inference context")
            # The two-phase completer adds a closing cue and a 50-token buffer.
            if phase1 <= 0 or context - phase1 < 128:
                raise ValueError("phase-one cap must reserve at least 128 tokens for the answer cue and buffer")
        if not isinstance(self.retired_task_ids, list) or any(
            not isinstance(task, str) or not re.fullmatch(r"sky-managed-[A-Za-z0-9_.-]+_\d+-\d+", task)
            for task in self.retired_task_ids
        ):
            raise ValueError("retired_task_ids must contain exact managed task IDs verified terminal")
        for ip, audit in self.retired_processes.items():
            import ipaddress
            ipaddress.ip_address(ip)
            if not re.fullmatch(r"[a-f0-9-]{36}", audit.get("boot_id", "")):
                raise ValueError("retired processes require a host boot identity")
            for process in audit.get("processes", []):
                if (not isinstance(process.get("pid"), int) or process["pid"] <= 1 or
                    not isinstance(process.get("created"), (float, int)) or process["created"] <= 0 or
                    not re.fullmatch(r"[a-f0-9]{64}", process.get("command_sha256", ""))):
                    raise ValueError("retired processes require exact PID, start time and command hash")

    def for_inference_rank(self, rank):
        if not self.arena_models or rank not in self.inference_only_ranks:
            return self
        entry = self.arena_models[self.inference_only_ranks.index(rank) // self.arena_replicas_per_model]
        raw = self.to_dict()
        for key in ("model", "client_member_spec", "client_context_window",
                    "client_phase1_max_tokens", "client_learning_rate"):
            raw.pop(key)
        raw.update(model_preset=entry["model_preset"], arena_models=[],
                   trainer={"hosts": 0},
                   inference=entry.get("inference", {}),
                   cache={**raw["cache"], **entry.get("cache", {})})
        return Config.from_dict(raw)

    @property
    def served_models(self):
        return ([PRESETS[e["model_preset"]].hf_model for e in self.arena_models]
                if self.arena_models else [self.model])

    @property
    def engines_per_host(self):
        return 4 // self.inference.tp

    def engine_slots(self, inference_ips):
        return [dict(ip=ip, slot=slot, port=self.ports.engine + slot,
                     key=ip if self.engines_per_host == 1 else f"{ip}:{self.ports.engine + slot}")
                for ip in inference_ips for slot in range(self.engines_per_host)]

    def engine_groups(self, inference_ips):
        """Inference hosts grouped into engines, in order; each group's first
        host runs the vLLM server and is the engine's address."""
        n = self.inference.hosts_per_engine
        return [list(inference_ips[i:i + n]) for i in range(0, len(inference_ips), n)]

    @property
    def effective_zone(self):
        return self.zone or ACCELERATOR_ZONES[self.accelerator][0]

    @property
    def placement_ranks(self):
        value = self.client_env.get("PLACEMENT_TPU_RANKS", "")
        return [int(r) for r in value.split(",")] if value else []

    @property
    def inference_hosts(self):
        if self.inference_only_ranks is not None:
            return len(self.inference_only_ranks)
        return (self.hosts - self.trainer.hosts - int(self.arena_grader_rank is not None)
                - len(self.placement_ranks) - int(self.science_task == "placement"
                                                and self.science_placement_backend == 'tpu'))

    def client_sampling_environment(self):
        defaults = {
            "CONTEXT_WINDOW": str(self.client_context_window),
            "TTD_M0_CONTEXT_WINDOW": str(self.client_context_window),
            "TTD_M0_TRAIN_MAX_SEQ": str(self.client_context_window),
            "TTD_M0_PHASE1_MAX_TOKENS": str(self.client_phase1_max_tokens),
        }
        return {key: self.client_env.get(key, value) for key, value in defaults.items()}

    @property
    def run_gcs(self):
        return f"{self.bucket.rstrip('/')}/ray-training/{self.run_id}"
