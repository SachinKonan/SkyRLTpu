"""Serializable configuration; no TPU, Ray or JAX imports in the controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Ports:
    ray: int = 19679
    dashboard: int = 19680
    client: int = 19681
    object_manager: int = 19700
    node_manager: int = 19701
    dashboard_agent: int = 19702
    dashboard_agent_grpc: int = 19703
    runtime_env: int = 19704
    metrics: int = 19705
    worker_min: int = 42000
    worker_max: int = 42999
    inference: int = 19800
    engine: int = 19801
    trainer: int = 19802
    trainer_jax: int = 19803
    trainer_tpu: int = 19804
    inference_tpu: int = 19805
    topology_jax: int = 19806
    topology_subset: int = 19807


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
    hosts: int = 4
    tp: int = 8
    fsdp: int = 2
    process_bounds: str = "1,1,4"
    chip_bounds: str = "2,2,1"
    maxtext_model: str = "qwen3.5-27b"
    maxtext_spec: str = "maxtext @ git+https://github.com/SachinKonan/maxtext.git@0fd409939977ac0ab79a4e64d21730936f253567"
    lora_rank: int = 32
    # Largest adapter rank the trainer backend accepts; 0 means lora_rank.
    # The learnable carried/fresh LoRA mix (skyrl.backends.lora_mix) stores
    # both halves in one rank-2r adapter, so it needs 2 x lora_rank here and
    # on the inference side.
    max_lora_rank: int = 0
    sequence_length: int = 22528
    token_budget: int = 45056
    flce_tile: int = 512
    remat: str = "full"
    logical_kv_heads: int = 8

    @property
    def effective_max_lora_rank(self):
        return self.max_lora_rank or self.lora_rank


@dataclass(frozen=True)
class Inference:
    # This is the backend validated by job 328, not native vLLM multi-host DP.
    backend: str = "ray_serve"
    tp: int = 4
    max_sequences: int = 16
    memory_utilization: float = 0.9
    max_model_length: int = 22528
    chunk_tokens: int = 4096
    max_lora_rank: int = 32
    request_timeout: int = 21600
    restart_limit: int = 3
    max_loras: int = 1


@dataclass(frozen=True)
class Config:
    run_id: str
    accelerator: str
    hosts: int
    bucket: str
    base_bundle: str
    base_bundle_sha256: str
    model: str = "Qwen/Qwen3.5-27B"
    root: str = "~/.cache/skyrl-ray"
    inference_only: bool = False
    inference_only_ranks: list[int] | None = None
    ports: Ports = field(default_factory=Ports)
    cache: Cache = field(default_factory=Cache)
    trainer: Trainer = field(default_factory=Trainer)
    inference: Inference = field(default_factory=Inference)
    log_seconds: int = 30
    setup_timeout: int = 7200
    ready_timeout: int = 3600
    checkpoint_cleanup_timeout: int = 600
    client_context_window: int = 18432
    client_phase1_max_tokens: int = 13824
    client_env: dict[str, str] = field(default_factory=dict)
    # Extra environment for the trainer (Tinker API server) process only,
    # e.g. TUNIX_LORA_MIX_GAMMA. Applied after the launcher's own settings.
    trainer_env: dict[str, str] = field(default_factory=dict)
    retired_task_ids: list[str] = field(default_factory=list)
    retired_processes: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw):
        data = dict(raw)
        for key, kind in (("ports", Ports), ("cache", Cache), ("trainer", Trainer),
                          ("inference", Inference)):
            data[key] = kind(**data.get(key, {}))
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.inference_only_ranks is not None:
            ranks = self.inference_only_ranks
            if (not self.inference_only or not isinstance(ranks, list) or not ranks
                    or any(type(r) is not int or not 0 <= r < self.hosts for r in ranks)
                    or len(set(ranks)) != len(ranks)):
                raise ValueError("inference_only_ranks requires unique valid ranks in inference-only mode")
        if type(self.checkpoint_cleanup_timeout) is not int or self.checkpoint_cleanup_timeout < 0:
            raise ValueError("checkpoint_cleanup_timeout must be a nonnegative integer (0 disables)")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", self.run_id):
            raise ValueError("run_id must be a safe unique name")
        expected = {"tpu-v4-64": 8, "tpu-v4-32": 4, "tpu-v5p-32": 4}
        if expected.get(self.accelerator) != self.hosts:
            raise ValueError("accelerator/host count mismatch")
        if self.inference_only and self.trainer.hosts != 0:
            raise ValueError("inference-only profiles must declare zero trainer hosts")
        if not self.inference_only and not 0 < self.trainer.hosts < self.hosts:
            raise ValueError("require disjoint nonempty trainer and inference roles")
        if not self.inference_only and self.accelerator == "tpu-v4-64" and self.trainer.hosts != 4:
            raise ValueError("v4-64 currently requires the validated four-host row")
        if not self.inference_only and self.accelerator in ("tpu-v5p-32", "tpu-v4-32") and self.trainer.hosts != 1:
            raise ValueError("32-core profiles currently require one trainer host")
        if not self.inference_only and self.trainer.tp * self.trainer.fsdp != self.trainer.hosts * 4:
            raise ValueError("trainer TP x FSDP must cover exactly the trainer chips")
        if self.inference.tp != 4 or not 0 < self.inference.memory_utilization < 1:
            raise ValueError("require independent four-chip engines and a memory reserve")
        if self.inference.backend != "ray_serve":
            raise ValueError("native vLLM DP is not validated; no silent backend substitution")
        if type(self.inference.max_loras) is not int or self.inference.max_loras < 1:
            raise ValueError("inference.max_loras must be a positive integer")
        if self.inference.max_sequences < 1 or self.inference.restart_limit < 0:
            raise ValueError("invalid inference limits")
        if self.trainer.max_lora_rank < 0 or 0 < self.trainer.max_lora_rank < self.trainer.lora_rank:
            raise ValueError("trainer max_lora_rank must be 0 (= lora_rank) or >= lora_rank")
        if self.trainer.effective_max_lora_rank > self.inference.max_lora_rank:
            raise ValueError("inference cannot load the requested trainer LoRA rank")
        if self.trainer_env.get("TUNIX_LORA_MIX_GAMMA") and (
            self.trainer.effective_max_lora_rank < 2 * self.trainer.lora_rank
        ):
            raise ValueError("the LoRA mix stores two rank-r halves: set trainer.max_lora_rank and "
                             "inference.max_lora_rank to at least 2 x lora_rank")
        if self.trainer.sequence_length > self.inference.max_model_length:
            raise ValueError("inference context must cover trainer context")
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
        if any(str(k) != k or str(v) != v for k, v in self.trainer_env.items()):
            raise ValueError("trainer environment must contain strings")
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

    @property
    def inference_hosts(self):
        if self.inference_only_ranks is not None:
            return len(self.inference_only_ranks)
        return self.hosts - self.trainer.hosts

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
