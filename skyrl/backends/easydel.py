"""EasyDeL learner and eSurge inference backend for the Tinker engine.

SkyRL remains responsible for request preparation, objective definitions, and
response schemas. EasyDeL supplies pretrained model conversion, model/sequence
sharding, rematerialized execution, memory-efficient target-logprob scoring,
LoRA state, optimizer state, and colocated eSurge generation.
"""

from __future__ import annotations

import base64
import gc
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
import optax
from cloudpathlib import AnyPath

# EasyDeL otherwise initializes JAX distributed as an import side effect. SkyRL
# owns process IDs and coordinator setup so every host joins the same group.
os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from easydel.infra import EasyDeLBaseConfigDict, EasyDeLState
from easydel.layers import ParallelLinear, eLoRA
from easydel.modules.auto import (
    AutoEasyDeLModelForCausalLM,
    AutoEasyDeLModelForImageTextToText,
)
from easydel.trainers._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
)
from easydel.utils.traversals import (
    get_module_from_path,
    iter_module_search,
    set_module_from_path,
)
from flax import nnx as nn
from flax.training import checkpoints
from huggingface_hub import snapshot_download
from jax.experimental import multihost_utils
from pydantic import BaseModel, Field, TypeAdapter
from transformers import AutoTokenizer

from skyrl.backends.backend import AbstractBackend
from skyrl.backends.multihost import (
    CommandClient,
    CommandServer,
    RpcPayload,
    local_ipv4_address,
)
from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import pad_batch, pad_to_fsdp
from skyrl.tinker import types
from skyrl.tinker.loss_fns import LossFnConfig, compute_per_token_losses
from skyrl.tinker.types import LOSS_TYPES
from skyrl.utils.log import logger

_DEFAULT_PPO_CLIP_LOW_THRESHOLD = 0.8
_DEFAULT_PPO_CLIP_HIGH_THRESHOLD = 1.2
_AXIS_NAMES = ("dp", "fsdp", "ep", "tp", "sp")
_DEFAULT_EASYDEL_CHECKPOINTS = {
    "Qwen/Qwen3.5-9B": "EasyDeL/Qwen3.5-9B",
}


class EasyDeLBackendConfig(BaseModel, extra="forbid"):
    """Configuration for EasyDeL execution behind the Tinker API."""

    max_lora_adapters: int = Field(default=32, ge=1)
    max_lora_rank: int = Field(default=64, ge=1)
    model_name_or_path: str | None = Field(
        default=None,
        description="Optional EasyDeL/HF checkpoint override; the Tinker base model remains the API identity.",
    )
    tokenizer_name_or_path: str | None = Field(
        default=None,
        description="Tokenizer override. By default the tokenizer is loaded from model_name_or_path.",
    )
    model_task: Literal["auto", "causal_lm", "image_text_to_text"] = Field(
        default="auto",
        description="EasyDeL auto-model family. Auto inspects a local config and known converted checkpoints.",
    )
    from_torch: bool | None = Field(
        default=None,
        description="Force loading HF/PyTorch weights when true or EasyDeL weights when false; auto-detect by default.",
    )
    dtype: str = Field(default="bfloat16")
    data_parallel_size: int = Field(default=1)
    fully_sharded_data_parallel_size: int = Field(default=1)
    expert_parallel_size: int = Field(default=1)
    tensor_parallel_size: int = Field(default=-1)
    sequence_parallel_size: int = Field(default=1)
    train_micro_batch_size: int = Field(default=1, ge=1)
    enforce_eager: bool = Field(default=False)
    attention_mechanism: str = Field(
        default="auto",
        description="EasyDeL attention mechanism. 'auto' selects Splash/blocksparse on TPU.",
    )
    gradient_checkpointing: str = Field(
        default="nothing_saveable",
        description="EasyDeL rematerialization policy; nothing_saveable gives maximum memory savings.",
    )
    use_scan_mlp: bool = Field(default=True)
    scan_mlp_chunk_size: int = Field(default=1024, ge=1)
    lmhead_token_chunk_size: int = Field(default=256, ge=1)
    lmhead_vocab_chunk_size: int = Field(default=32768, ge=1)
    sample_max_num_sequences: int = Field(default=32, ge=1)
    sample_max_model_len: int = Field(default=131072, ge=2)
    sample_hbm_utilization: float = Field(default=0.80, gt=0.0, le=1.0)
    sample_page_size: int = Field(default=128, ge=1)
    sample_distributed_service_name: str | None = Field(
        default=None,
        description="DNS name resolving every TPU worker for eSurge lockstep serving.",
    )
    sample_distributed_hosts: list[str] | None = Field(
        default=None,
        description=(
            "Optional worker hostnames in SkyRL process-id order. Jobman TPU VMs need this because "
            "EasyDeL's DNS discovery sorts IPs, which need not match worker IDs."
        ),
    )
    sample_distributed_auth_token: str = Field(default="skyrl-easydel-esurge")
    sample_distributed_control_port: int = Field(default=19666, ge=1024, le=65535)
    coordinator_address: str | None = None
    num_processes: int | None = None

    @property
    def axis_dims(self) -> tuple[int, int, int, int, int]:
        return (
            self.data_parallel_size,
            self.fully_sharded_data_parallel_size,
            self.expert_parallel_size,
            self.tensor_parallel_size,
            self.sequence_parallel_size,
        )


class _ScaledELoRA(eLoRA):
    """EasyDeL LoRA with Tinker's explicit alpha/rank residual scaling."""

    def __init__(self, *args, scale: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = float(scale)

    def __call__(self, x: jax.Array, *args, **kwargs):
        x, lora_a, lora_b = self.promote_dtype((x, self.lora_a[...], self.lora_b[...]), dtype=self.dtype)
        out = (x @ lora_a @ lora_b) * self.scale
        if self.base_module is not None:
            out += self.base_module(x, *args, **kwargs)
        return out

    def native_forward(self, inputs: jax.Array, *, w: jax.Array | None = None) -> jax.Array:
        inputs, lora_a, lora_b = self.promote_dtype(
            (inputs, self.lora_a[...], self.lora_b[...]),
            dtype=self.dtype,
        )
        out = (inputs @ lora_a @ lora_b) * self.scale
        if self.base_module is not None:
            if hasattr(self.base_module, "native_forward"):
                out += self.base_module.native_forward(inputs=inputs, w=w)
            elif w is None:
                out += self.base_module(inputs)
            else:
                out += self.base_module(inputs, w=w)
        return out


def _should_train_lora_path(path: tuple[Any, ...], config: types.LoraConfig) -> bool:
    name = ".".join(str(part) for part in path).lower()
    if any(marker in name for marker in (".visual.", ".vision.", "vision_tower")):
        return False
    is_unembed = "lm_head" in name
    is_attention = any(
        marker in name
        for marker in (
            "self_attn",
            "attention",
            "linear_attn",
            "in_proj",
            "out_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
        )
    )
    is_mlp = any(marker in name for marker in ("mlp", "experts", "gate_proj", "up_proj", "down_proj"))
    return (
        (is_unembed and config.train_unembed) or (is_attention and config.train_attn) or (is_mlp and config.train_mlp)
    )


def _apply_scaled_lora(model, config: types.LoraConfig) -> int:
    if config.train_unembed and getattr(model.config.get_text_config(), "tie_word_embeddings", False):
        raise ValueError("train_unembed=True is incompatible with tied embeddings in the EasyDeL backend")

    matches = list(iter_module_search(model, ParallelLinear))
    wrapped = 0
    rngs = nn.Rngs(config.seed)
    for path, _ in matches:
        if not _should_train_lora_path(path, config):
            continue
        base_module: ParallelLinear = get_module_from_path(model=model, path=path)
        set_module_from_path(
            model=model,
            path=path,
            new_value=_ScaledELoRA(
                base_module=base_module,
                rngs=rngs,
                dtype=base_module.dtype,
                param_dtype=base_module.param_dtype,
                in_features=base_module.in_features,
                lora_rank=config.rank,
                out_features=base_module.out_features,
                scale=config.alpha / config.rank,
            ),
        )
        wrapped += 1
    if wrapped == 0:
        raise ValueError(
            "The EasyDeL LoRA selection matched no ParallelLinear modules; "
            "check train_attn/train_mlp/train_unembed and the model architecture"
        )
    return wrapped


def _merge_scaled_lora_for_sampling(model):
    """Clone a learner model and bake its scaled LoRA updates into base kernels."""
    merged = nn.clone(model)
    lora_paths = [path for path, _ in iter_module_search(merged, _ScaledELoRA)]
    for path in lora_paths:
        adapter: _ScaledELoRA = get_module_from_path(model=merged, path=path)
        if adapter.base_module is None or not hasattr(adapter.base_module, "kernel"):
            raise ValueError(f"Cannot merge LoRA adapter at {path}: missing base kernel")
        kernel = adapter.base_module.kernel[...]
        with jax.default_matmul_precision("float32"):
            delta = adapter.lora_a[...].astype(jnp.float32) @ adapter.lora_b[...].astype(jnp.float32)
        adapter.base_module.kernel[...] = (kernel.astype(jnp.float32) + delta * adapter.scale).astype(kernel.dtype)
        set_module_from_path(model=merged, path=path, new_value=adapter.base_module)
    return merged, len(lora_paths)


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _remove_checkpoint_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _lora_configs_compatible(saved: types.LoraConfig, current: types.LoraConfig) -> bool:
    """Compare checkpoint-relevant LoRA structure, excluding the initialization seed."""
    return saved.model_dump(exclude={"seed"}) == current.model_dump(exclude={"seed"})


def _dtype_from_name(name: str):
    normalized = name.lower().replace("torch.", "").replace("jnp.", "")
    values = {
        "bf16": jnp.bfloat16,
        "bfloat16": jnp.bfloat16,
        "fp16": jnp.float16,
        "float16": jnp.float16,
        "fp32": jnp.float32,
        "float32": jnp.float32,
    }
    if normalized not in values:
        raise ValueError(f"Unsupported EasyDeL dtype {name!r}; choose one of {sorted(values)}")
    return values[normalized]


def _round_up_seq_len(seq_len: int) -> int:
    """Use the same two-significant-bit compile buckets as the JAX backend."""
    if seq_len <= 32:
        return 32
    msb_pos = seq_len.bit_length() - 1
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    result = seq_len & mask
    return result if result == seq_len else result + (1 << (msb_pos - 1))


def _resolve_cached_snapshot(path_or_repo: str) -> str:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return str(path)
    try:
        return snapshot_download(path_or_repo, local_files_only=True)
    except Exception:
        return path_or_repo


def _model_task(source: str, base_model: str, configured: str) -> str:
    if configured != "auto":
        return configured

    config_path = Path(source) / "config.json"
    if config_path.is_file():
        try:
            model_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read model config at {config_path}: {exc}") from exc
        architectures = " ".join(model_config.get("architectures") or []).lower()
        model_type = str(model_config.get("model_type", "")).lower()
        if "conditionalgeneration" in architectures or model_type == "qwen3_5":
            return "image_text_to_text"
        return "causal_lm"

    known_multimodal_sources = set(_DEFAULT_EASYDEL_CHECKPOINTS.values())
    requested_source = _DEFAULT_EASYDEL_CHECKPOINTS.get(base_model)
    if source in known_multimodal_sources or requested_source in known_multimodal_sources:
        return "image_text_to_text"
    return "causal_lm"


@dataclass
class _ModelRuntime:
    metadata: types.ModelMetadata
    state: EasyDeLState
    grad_sum: Any
    grad_count: int = 0
    forward_fn: Callable | None = field(default=None, repr=False)
    forward_backward_fn: Callable | None = field(default=None, repr=False)
    apply_gradients_fn: Callable | None = field(default=None, repr=False)
    sampling_model: Any | None = field(default=None, repr=False)
    sampling_engine: Any | None = field(default=None, repr=False)
    sampling_scope: str = field(default="", repr=False)


class EasyDeLBackendImpl(AbstractBackend):
    """EasyDeL implementation with one independent LoRA state per Tinker model."""

    def __init__(self, base_model: str, config: EasyDeLBackendConfig, process_id: int = 0):
        self.base_model = base_model
        self.config = config
        self.process_id = process_id
        self.metrics = types.EngineMetrics()
        self.models: dict[str, types.ModelMetadata] = {}
        self._runtimes: dict[str, _ModelRuntime] = {}
        self._base_scoring_runtime: _ModelRuntime | None = None
        self._base_sampling_engine: Any | None = None

        dtype = _dtype_from_name(config.dtype)
        source = _resolve_cached_snapshot(
            config.model_name_or_path or _DEFAULT_EASYDEL_CHECKPOINTS.get(base_model, base_model)
        )
        tokenizer_source = _resolve_cached_snapshot(config.tokenizer_name_or_path or source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        config_kwargs = EasyDeLBaseConfigDict(
            attn_mechanism=config.attention_mechanism,
            gradient_checkpointing=config.gradient_checkpointing,
            use_scan_mlp=config.use_scan_mlp,
            scan_mlp_chunk_size=config.scan_mlp_chunk_size,
            lmhead_chunksize=config.lmhead_token_chunk_size,
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "param_dtype": dtype,
            "sharding_axis_dims": config.axis_dims,
            "sharding_axis_names": _AXIS_NAMES,
            "config_kwargs": config_kwargs,
            "auto_shard_model": True,
        }
        if config.from_torch is not None:
            load_kwargs["from_torch"] = config.from_torch

        logger.info(
            "Loading EasyDeL base model %s for API model %s with mesh (dp,fsdp,ep,tp,sp)=%s",
            source,
            base_model,
            config.axis_dims,
        )
        model_task = _model_task(source, base_model, config.model_task)
        loader = {
            "causal_lm": AutoEasyDeLModelForCausalLM,
            "image_text_to_text": AutoEasyDeLModelForImageTextToText,
        }[model_task]
        logger.info("Using EasyDeL %s loader and tokenizer %s", model_task, tokenizer_source)
        base = loader.from_pretrained(str(source), **load_kwargs)
        self.mesh = base.mesh
        self._base_graphdef, self._base_graphstate, self._base_graphother = base.split_module()
        model_key = hashlib.sha1(base_model.encode("utf-8")).hexdigest()[:16]
        self._base_esurge_scope = f"skyrl-easydel-base-{model_key}"
        self._dtype = dtype

        if config.sequence_parallel_size > 1 and self.mesh.shape["sp"] <= 1:
            raise ValueError(
                f"sequence_parallel_size={config.sequence_parallel_size} was requested, "
                f"but EasyDeL created mesh {self.mesh.shape}"
            )
        logger.info("Initialized EasyDeL backend with concrete mesh=%s", self.mesh.shape)

    def _base_model_instance(self):
        model = nn.merge(self._base_graphdef, self._base_graphstate, self._base_graphother)
        model._esurge_cache_scope_key = self._base_esurge_scope
        return model

    @staticmethod
    def _build_loss_fn_config(configs: list[dict[str, float] | None]) -> LossFnConfig:
        configs = [config or {} for config in configs]
        return LossFnConfig(
            clip_low_threshold=np.asarray(
                [float(c.get("clip_low_threshold", _DEFAULT_PPO_CLIP_LOW_THRESHOLD)) for c in configs],
                dtype=np.float32,
            ),
            clip_high_threshold=np.asarray(
                [float(c.get("clip_high_threshold", _DEFAULT_PPO_CLIP_HIGH_THRESHOLD)) for c in configs],
                dtype=np.float32,
            ),
        )

    def _make_runtime_functions(self, runtime: _ModelRuntime) -> None:
        graphdef = runtime.state.graphdef
        token_chunk_size = self.config.lmhead_token_chunk_size
        vocab_chunk_size = self.config.lmhead_vocab_chunk_size

        def loss_for_lora(
            graphstate,
            graphother,
            input_ids,
            attention_mask,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
            loss_fn_config,
        ):
            frozen_other = jax.tree_util.tree_map(
                lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                graphother,
            )
            model = nn.merge(graphdef, graphstate, frozen_other)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                apply_lm_head=False,
            )
            target_logprobs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
                model,
                output.last_hidden_state,
                target_ids,
                token_chunk_size=token_chunk_size,
                vocab_chunk_size=vocab_chunk_size,
                return_entropy=False,
            )
            per_token_losses = compute_per_token_losses(
                loss_fn_types,
                target_logprobs,
                loss_mask,
                sampling_logprobs,
                advantages,
                loss_fn_config,
            )
            per_seq_loss = per_token_losses.sum(axis=-1) / jnp.maximum(loss_mask.sum(axis=-1), 1e-9)
            return per_seq_loss.sum(), (target_logprobs, per_token_losses)

        loss_and_grad = jax.value_and_grad(loss_for_lora, argnums=0, has_aux=True)

        def forward_backward(
            grad_sum,
            graphstate,
            graphother,
            input_ids,
            attention_mask,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
            loss_fn_config,
        ):
            (_, (target_logprobs, per_token_losses)), grads = loss_and_grad(
                graphstate,
                graphother,
                input_ids,
                attention_mask,
                target_ids,
                loss_mask,
                loss_fn_types,
                sampling_logprobs,
                advantages,
                loss_fn_config,
            )
            grad_sum = jax.tree_util.tree_map(lambda total, grad: total + grad, grad_sum, grads)
            return grad_sum, per_token_losses, target_logprobs

        def forward_only(
            grad_sum,
            graphstate,
            graphother,
            input_ids,
            attention_mask,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
            loss_fn_config,
        ):
            _, (target_logprobs, per_token_losses) = loss_for_lora(
                graphstate,
                graphother,
                input_ids,
                attention_mask,
                target_ids,
                loss_mask,
                loss_fn_types,
                sampling_logprobs,
                advantages,
                loss_fn_config,
            )
            return grad_sum, per_token_losses, target_logprobs

        def apply_gradients(state, grads):
            return state.apply_gradients(grads=grads)

        if self.config.enforce_eager:
            runtime.forward_backward_fn = forward_backward
            runtime.forward_fn = forward_only
            runtime.apply_gradients_fn = apply_gradients
        else:
            runtime.forward_backward_fn = jax.jit(forward_backward, donate_argnums=(0,))
            runtime.forward_fn = jax.jit(forward_only)
            runtime.apply_gradients_fn = jax.jit(apply_gradients)

    def has_model(self, model_id: str) -> bool:
        return model_id in self._runtimes

    def create_model(
        self,
        model_id: str,
        lora_config: types.LoraConfig,
        model_role: str = "policy",
    ) -> None:
        if model_role != "policy":
            raise ValueError(f"EasyDeLBackend only supports model_role='policy', got {model_role!r}")
        if model_id in self._runtimes:
            raise ValueError(f"Model {model_id} already exists")
        if len(self._runtimes) >= self.config.max_lora_adapters - 1:
            raise ValueError(f"Maximum number of LoRA adapters ({self.config.max_lora_adapters}) reached")
        if not 0 < lora_config.rank <= self.config.max_lora_rank:
            raise ValueError(f"LoRA rank {lora_config.rank} must be between 1 and {self.config.max_lora_rank}")

        adapter_index = min(
            set(range(1, self.config.max_lora_adapters)) - {m.adapter_index for m in self.models.values()}
        )
        metadata = types.ModelMetadata(adapter_index=adapter_index, lora_config=lora_config)
        runtime, wrapped = self._create_runtime(metadata)
        self.models[model_id] = metadata
        self._runtimes[model_id] = runtime
        logger.info(
            "Created EasyDeL model %s adapter_index=%d rank=%d alpha=%s wrapped_layers=%d",
            model_id,
            adapter_index,
            lora_config.rank,
            lora_config.alpha,
            wrapped,
        )

    def _create_runtime(self, metadata: types.ModelMetadata) -> tuple[_ModelRuntime, int]:
        model = nn.merge(self._base_graphdef, self._base_graphstate, self._base_graphother)
        wrapped = _apply_scaled_lora(model, metadata.lora_config)
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
        state = EasyDeLState.create(model=model)
        state = state.shard_model()
        state = state.init_tx(tx)
        runtime = _ModelRuntime(
            metadata=metadata,
            state=state,
            grad_sum=_tree_zeros_like(state.graphstate),
            sampling_scope=f"{self._base_esurge_scope}-adapter-{metadata.adapter_index}",
        )
        self._make_runtime_functions(runtime)
        return runtime, wrapped

    def _get_base_scoring_runtime(self) -> _ModelRuntime:
        if self._base_scoring_runtime is None:
            metadata = types.ModelMetadata(
                adapter_index=0,
                lora_config=types.LoraConfig(rank=1, alpha=1, seed=0),
            )
            self._base_scoring_runtime, wrapped = self._create_runtime(metadata)
            logger.info("Created exact zero-LoRA base scoring view with %d wrapped layers", wrapped)
        return self._base_scoring_runtime

    def delete_model(self, model_id: str) -> None:
        if model_id not in self._runtimes:
            raise ValueError(f"Model {model_id} not found")
        runtime = self._runtimes[model_id]
        if runtime.sampling_engine is not None:
            runtime.sampling_engine.terminate()
        del self._runtimes[model_id]
        del self.models[model_id]

    def shutdown(self) -> None:
        """Release colocated inference engines owned by this process."""
        engines = [getattr(self, "_base_sampling_engine", None)]
        engines.extend(runtime.sampling_engine for runtime in getattr(self, "_runtimes", {}).values())
        for engine in engines:
            if engine is None:
                continue
            try:
                engine.terminate()
            finally:
                controller = getattr(engine, "_distributed_controller", None)
                if controller is not None:
                    controller.shutdown()

    def _input_shardings(self):
        batch_axes = tuple(axis for axis in ("dp", "fsdp") if self.mesh.shape[axis] > 1)
        batch_axis: str | tuple[str, ...] | None
        if not batch_axes:
            batch_axis = None
        elif len(batch_axes) == 1:
            batch_axis = batch_axes[0]
        else:
            batch_axis = batch_axes
        sequence_axis = "sp" if self.mesh.shape["sp"] > 1 else None
        return (
            jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec(batch_axis, sequence_axis)),
            jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec(batch_axis)),
        )

    def _run_model_indices(
        self,
        runtime: _ModelRuntime,
        prepared_batch: types.PreparedModelPassBatch,
        indices: list[int],
        backward: bool,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        rendered = render_model_input([prepared_batch.all_model_inputs[i] for i in indices])
        input_rows = [item.prompt_ids for item in rendered]
        if any(item.multi_modal_kwargs is not None for item in rendered):
            raise NotImplementedError("EasyDeL Tinker backend currently accepts encoded-text model inputs only")

        max_len = _round_up_seq_len(max(len(row) for row in input_rows))
        sp_size = int(self.mesh.shape["sp"])
        max_len = math.ceil(max_len / sp_size) * sp_size
        input_ids = pad_batch(input_rows, max_len, np.int32)
        attention_mask = pad_batch([[1] * len(row) for row in input_rows], max_len, np.int32)
        target_ids = pad_batch([prepared_batch.all_targets[i] for i in indices], max_len, np.int32)
        loss_mask = pad_batch([prepared_batch.all_token_weights[i] for i in indices], max_len, np.float32)
        sampling_logprobs = pad_batch(
            [prepared_batch.all_sampling_logprobs[i] for i in indices],
            max_len,
            np.float32,
        )
        advantages = pad_batch([prepared_batch.all_advantages[i] for i in indices], max_len, np.float32)
        loss_fn_types = np.asarray([LOSS_TYPES[prepared_batch.all_loss_fns[i]] for i in indices], dtype=np.int32)
        loss_fn_config = self._build_loss_fn_config([prepared_batch.all_loss_fn_configs[i] for i in indices])

        batch_sharding, scalar_sharding = self._input_shardings()
        batch_parallel = int(self.mesh.shape["dp"]) * int(self.mesh.shape["fsdp"])
        micro_size = self.config.train_micro_batch_size
        token_losses: dict[int, np.ndarray] = {}
        target_logprobs: dict[int, np.ndarray] = {}
        model_fn = runtime.forward_backward_fn if backward else runtime.forward_fn
        assert model_fn is not None

        # EasyDeL/eformer 0.0.99 still reads JAX's physical mesh context.
        with self.mesh:
            for start in range(0, len(indices), micro_size):
                end = min(start + micro_size, len(indices))
                actual = end - start
                arrays_2d = (
                    pad_to_fsdp(input_ids[start:end], batch_parallel),
                    pad_to_fsdp(attention_mask[start:end], batch_parallel),
                    pad_to_fsdp(target_ids[start:end], batch_parallel),
                    pad_to_fsdp(loss_mask[start:end], batch_parallel),
                    pad_to_fsdp(sampling_logprobs[start:end], batch_parallel),
                    pad_to_fsdp(advantages[start:end], batch_parallel),
                )
                arrays_1d = (
                    pad_to_fsdp(loss_fn_types[start:end], batch_parallel),
                    pad_to_fsdp(loss_fn_config.clip_low_threshold[start:end], batch_parallel),
                    pad_to_fsdp(loss_fn_config.clip_high_threshold[start:end], batch_parallel),
                )
                device_2d = jax.device_put(arrays_2d, (batch_sharding,) * len(arrays_2d))
                device_1d = jax.device_put(arrays_1d, (scalar_sharding,) * len(arrays_1d))
                mb_config = LossFnConfig(
                    clip_low_threshold=device_1d[1],
                    clip_high_threshold=device_1d[2],
                )
                runtime.grad_sum, losses_device, logprobs_device = model_fn(
                    runtime.grad_sum,
                    runtime.state.graphstate,
                    runtime.state.graphother,
                    device_2d[0],
                    device_2d[1],
                    device_2d[2],
                    device_2d[3],
                    device_1d[0],
                    device_2d[4],
                    device_2d[5],
                    mb_config,
                )
                losses_host, logprobs_host = jax.device_get((losses_device[:actual], logprobs_device[:actual]))
                for offset in range(actual):
                    global_index = indices[start + offset]
                    seq_len = len(input_rows[start + offset])
                    token_losses[global_index] = np.asarray(losses_host[offset, :seq_len], dtype=np.float32)
                    target_logprobs[global_index] = np.asarray(logprobs_host[offset, :seq_len], dtype=np.float32)
                if backward:
                    runtime.grad_count += actual
        return token_losses, target_logprobs

    def _pause_sampling_engines(self) -> None:
        """Release eSurge weights and KV pages before learner execution."""
        engine_runtimes = [(getattr(self, "_base_sampling_engine", None), None)]
        engine_runtimes.extend(
            (runtime.sampling_engine, runtime) for runtime in getattr(self, "_runtimes", {}).values()
        )
        released_model = False
        for engine, runtime in engine_runtimes:
            if engine is None or (runtime is not None and runtime.sampling_model is None):
                continue
            if not getattr(engine, "_paused", False):
                engine.pause()
                # Distributed workers intentionally have no scheduler thread,
                # so upstream pause() returns before destroying their pages.
                if (
                    getattr(engine, "_kv_cache_valid", False)
                    and engine.num_running_requests == 0
                    and engine.num_pending_requests == 0
                ):
                    engine.runner.destroy_kv_cache()
                    engine._kv_cache_valid = False
            engine.release_model_state(clear_compiled_cache=False)
            if runtime is not None:
                runtime.sampling_model = None
            released_model = True
        if released_model:
            # nn.clone materializes a merged inference copy of the base model.
            # Dropping all Python references lets JAX recycle that HBM for the
            # learner while retaining eSurge's compiled executables.
            gc.collect()

    def _model_pass(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        *,
        backward: bool,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_model_inputs:
            return {}
        if "ppo_critic" in prepared_batch.all_loss_fns:
            raise ValueError("ppo_critic is only supported by the SkyRL-Train backend")
        by_model: dict[str, list[int]] = {}
        for i, model_id in enumerate(prepared_batch.all_model_ids):
            by_model.setdefault(model_id, []).append(i)

        losses_by_index: dict[int, np.ndarray] = {}
        logprobs_by_index: dict[int, np.ndarray] = {}
        for model_id, indices in by_model.items():
            if model_id not in self._runtimes:
                raise ValueError(f"Model {model_id} not found")
            losses, logprobs = self._run_model_indices(
                self._runtimes[model_id],
                prepared_batch,
                indices,
                backward,
            )
            losses_by_index.update(losses)
            logprobs_by_index.update(logprobs)

        results: dict[str, types.ForwardBackwardOutput | types.ErrorResponse] = {}
        for request_id, _, start, end in prepared_batch.request_batch_slices:
            outputs = []
            for i in range(start, end):
                losses = losses_by_index[i]
                logprobs = logprobs_by_index[i]
                outputs.append(
                    {
                        "elementwise_loss": {
                            "data": losses.tolist(),
                            "dtype": "float32",
                            "shape": [losses.shape[0]],
                        },
                        "logprobs": {
                            "data": logprobs.tolist(),
                            "dtype": "float32",
                            "shape": [logprobs.shape[0]],
                        },
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=outputs,
                metrics={},
            )
        return results

    def forward_backward(self, prepared_batch: types.PreparedModelPassBatch):
        self._pause_sampling_engines()
        return self._model_pass(prepared_batch, backward=True)

    def forward(self, prepared_batch: types.PreparedModelPassBatch):
        self._pause_sampling_engines()
        return self._model_pass(prepared_batch, backward=False)

    @staticmethod
    def _set_optimizer_hyperparameters(state: EasyDeLState, params: types.AdamParams) -> EasyDeLState:
        if state.opt_state is None or not hasattr(state.opt_state, "hyperparams"):
            raise RuntimeError("EasyDeL optimizer was not initialized with injectable hyperparameters")
        hyperparams = state.opt_state.hyperparams
        replacements = {
            "learning_rate": params.learning_rate,
            "b1": params.beta1,
            "b2": params.beta2,
            "eps": params.eps,
            "weight_decay": params.weight_decay,
        }
        for key, value in replacements.items():
            if key not in hyperparams:
                raise RuntimeError(f"Injected AdamW optimizer is missing hyperparameter {key!r}")
            old = hyperparams[key]
            hyperparams[key] = jnp.asarray(value, dtype=getattr(old, "dtype", jnp.float32))
        return state

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        runtime = self._runtimes[model_id]
        count = max(runtime.grad_count, 1)
        mean_grads = jax.tree_util.tree_map(lambda grad: grad / count, runtime.grad_sum)
        grad_norm = optax.global_norm(mean_grads)
        runtime.state = self._set_optimizer_hyperparameters(runtime.state, request_data.adam_params)
        assert runtime.apply_gradients_fn is not None
        with self.mesh:
            runtime.state = runtime.apply_gradients_fn(runtime.state, mean_grads)
        runtime.grad_sum = _tree_zeros_like(runtime.state.graphstate)
        runtime.grad_count = 0
        metrics = {
            "skyrl.ai/grad_norm": float(jax.device_get(grad_norm)),
            "skyrl.ai/learning_rate": request_data.adam_params.learning_rate,
        }
        return types.OptimStepOutput(metrics=metrics)

    def _checkpoint_target(self, model_id: str) -> dict[str, Any]:
        runtime = self._runtimes[model_id]
        return {
            "graphstate": runtime.state.graphstate,
            "opt_state": runtime.state.opt_state,
            "step": runtime.state.step,
            "lora_config": runtime.metadata.lora_config.model_dump(),
        }

    def _gather_checkpoint_arrays(self, model_id: str) -> dict[str, Any]:
        target = self._checkpoint_target(model_id)
        arrays = {key: target[key] for key in ("graphstate", "opt_state", "step")}
        gathered = multihost_utils.process_allgather(arrays, tiled=True)

        def collapse_local_replica(original, value):
            value = np.asarray(value)
            if isinstance(original, jax.Array) and not original.is_fully_addressable:
                return value
            original_shape = np.shape(original)
            if not original_shape:
                return value[0] if value.ndim else value
            if value.shape[0] == jax.process_count() * original_shape[0]:
                return value[: original_shape[0]]
            return value

        return jax.tree.map(collapse_local_replica, arrays, gathered)

    def _checkpoint_host_template(self, model_id: str) -> dict[str, Any]:
        """Build host placeholders without launching a pre-restore collective."""
        target = self._checkpoint_target(model_id)
        arrays = {key: target[key] for key in ("graphstate", "opt_state", "step")}

        def placeholder(value):
            if isinstance(value, jax.Array):
                return np.empty(value.shape, dtype=value.dtype)
            if isinstance(value, np.ndarray):
                return np.empty_like(value)
            return value

        return jax.tree.map(placeholder, arrays)

    @staticmethod
    def _place_checkpoint_arrays(restored: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
        def place(value, target):
            if isinstance(target, jax.Array):
                host_value = np.asarray(value)
                local_arrays = [
                    jax.device_put(host_value[shard.index], shard.device) for shard in target.addressable_shards
                ]
                return jax.make_array_from_single_device_arrays(
                    target.shape,
                    target.sharding,
                    local_arrays,
                )
            return value

        return jax.tree.map(place, restored, template)

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        if jax.process_count() > 1:
            host_arrays = self._gather_checkpoint_arrays(model_id)
            is_source = self.process_id == 0
            logger.info(
                "Saving EasyDeL checkpoint %s: process_id=%d jax_process_index=%d is_source=%s",
                output_path,
                self.process_id,
                jax.process_index(),
                is_source,
            )
            if is_source:
                local_output = Path(str(output_path))
                local_output.parent.mkdir(parents=True, exist_ok=True)
                _remove_checkpoint_path(local_output)
                payload = {
                    "arrays": host_arrays,
                    "lora_config": self.models[model_id].lora_config.model_dump(),
                }
                compressed = gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=1)
                temporary = local_output.with_name(f".{local_output.name}.tmp")
                temporary.write_bytes(compressed)
                temporary.replace(local_output)
                logger.info("Wrote EasyDeL checkpoint %s (%d bytes)", local_output, local_output.stat().st_size)
            multihost_utils.sync_global_devices(f"easydel-checkpoint-save-{model_id}")
            return
        checkpoints.save_checkpoint_multiprocess(
            target=self._checkpoint_target(model_id),
            ckpt_dir=output_path,
            step=0,
            prefix="checkpoint_",
            overwrite=True,
        )

    def _restore_checkpoint_bytes(self, payload: bytes, model_id: str) -> None:
        restored = pickle.loads(gzip.decompress(payload))
        loaded = types.LoraConfig.model_validate(restored["lora_config"])
        expected = self.models[model_id].lora_config
        if not _lora_configs_compatible(loaded, expected):
            raise ValueError(f"LoRA config mismatch: checkpoint={loaded}, model={expected}")
        template = self._checkpoint_host_template(model_id)
        arrays = self._place_checkpoint_arrays(restored["arrays"], template)
        runtime = self._runtimes[model_id]
        runtime.state = runtime.state.replace(**arrays)
        runtime.grad_sum = _tree_zeros_like(runtime.state.graphstate)
        runtime.grad_count = 0

    def load_checkpoint_payload(self, checkpoint_payload: str, model_id: str) -> None:
        """Restore a process-zero checkpoint payload without JAX collectives."""
        logger.info(
            "Loading EasyDeL checkpoint payload: process_id=%d jax_process_index=%d",
            self.process_id,
            jax.process_index(),
        )
        self._restore_checkpoint_bytes(base64.b64decode(checkpoint_payload), model_id)

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        if jax.process_count() > 1:
            local_checkpoint = Path(str(checkpoint_path))
            if not local_checkpoint.is_file():
                raise FileNotFoundError(f"EasyDeL checkpoint not found in {checkpoint_path}")
            self._restore_checkpoint_bytes(local_checkpoint.read_bytes(), model_id)
            return
        target = self._checkpoint_target(model_id)
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=target,
            prefix="checkpoint_",
        )
        if restored is None:
            raise FileNotFoundError(f"EasyDeL checkpoint not found in {checkpoint_path}")
        expected = self.models[model_id].lora_config
        loaded = types.LoraConfig.model_validate(restored["lora_config"])
        if not _lora_configs_compatible(loaded, expected):
            raise ValueError(f"LoRA config mismatch: checkpoint={loaded}, model={expected}")
        runtime = self._runtimes[model_id]
        runtime.state = runtime.state.replace(
            graphstate=restored["graphstate"],
            opt_state=restored["opt_state"],
            step=restored["step"],
        )
        runtime.grad_sum = _tree_zeros_like(runtime.state.graphstate)
        runtime.grad_count = 0

    def _refresh_esurge(self, model_id: str) -> None:
        runtime = self._runtimes[model_id]
        engine = runtime.sampling_engine
        sampling_model, merged_layers = _merge_scaled_lora_for_sampling(runtime.state.model)
        sampling_model._esurge_cache_scope_key = runtime.sampling_scope
        runtime.sampling_model = sampling_model
        if engine is not None:
            sampling_model._refresh_esurge_engine_weights(engine)
        logger.info("Refreshed merged EasyDeL sampler model %s with %d LoRA layers", model_id, merged_layers)

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        self._refresh_esurge(model_id)
        checkpoint_id = Path(str(output_path)).name.removesuffix(".tar.gz")
        self._runtimes[model_id].metadata.loaded_checkpoint_id = checkpoint_id
        if not persist:
            return
        local_output = Path(str(output_path))
        local_output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = local_output.with_name(f".{local_output.name}.easydel")
        _remove_checkpoint_path(checkpoint_dir)
        # Call the implementation directly. On the coordinator ``self`` is the
        # RPC wrapper; virtual dispatch here would issue a nested command while
        # workers are still executing this outer sampler-checkpoint command.
        EasyDeLBackendImpl.save_checkpoint(self, AnyPath(checkpoint_dir), model_id)
        is_source = self.process_id == 0
        if is_source:
            with tarfile.open(local_output, "w:gz") as archive:
                archive.add(checkpoint_dir, arcname="checkpoint")
        multihost_utils.sync_global_devices(f"easydel-sampler-save-{model_id}")
        _remove_checkpoint_path(checkpoint_dir)

    def _restore_sampler_checkpoint_bytes(self, model_id: str, checkpoint_id: str, payload: bytes) -> None:
        runtime = self._runtimes[model_id]
        if runtime.metadata.loaded_checkpoint_id == checkpoint_id:
            return
        with tempfile.TemporaryDirectory(prefix="skyrl-easydel-sampler-") as directory:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                archive.extractall(directory, filter="data")
            checkpoint_payload = (Path(directory) / "checkpoint").read_bytes()
            self._restore_checkpoint_bytes(checkpoint_payload, model_id)
        self._refresh_esurge(model_id)
        runtime.metadata.loaded_checkpoint_id = checkpoint_id

    def _load_sampler_checkpoint(self, model_id: str, checkpoint_id: str, checkpoint_path: str) -> None:
        self._restore_sampler_checkpoint_bytes(model_id, checkpoint_id, Path(checkpoint_path).read_bytes())

    def load_sampler_checkpoint_payload(self, model_id: str, checkpoint_id: str, checkpoint_payload: str) -> None:
        """Restore a sampler archive transported over the CPU command plane."""
        self._restore_sampler_checkpoint_bytes(model_id, checkpoint_id, base64.b64decode(checkpoint_payload))

    def load_sampler_checkpoint(self, model_id: str, checkpoint_id: str, checkpoint_path: str) -> None:
        """Restore a persisted sampler archive on every backend process."""
        self._load_sampler_checkpoint(model_id, checkpoint_id, checkpoint_path)

    def _sampling_model(self, model_id: str):
        if not model_id:
            return self._base_model_instance()
        runtime = self._runtimes[model_id]
        if runtime.sampling_model is None:
            self._refresh_esurge(model_id)
        return runtime.sampling_model

    def _sampling_engine(self, model_id: str):
        runtime = self._runtimes.get(model_id) if model_id else None
        engine = runtime.sampling_engine if runtime is not None else self._base_sampling_engine
        if engine is not None:
            if runtime is not None and runtime.sampling_model is None:
                self._refresh_esurge(model_id)
            elif runtime is None and getattr(engine.runner, "model", None) is None:
                sampling_model = self._base_model_instance()
                sampling_model._refresh_esurge_engine_weights(engine)
            if getattr(engine, "_paused", False):
                engine.resume()
            return engine

        distributed = jax.process_count() > 1
        if distributed and not self.config.sample_distributed_service_name:
            raise ValueError(
                "sample_distributed_service_name is required for multi-host eSurge; " "it must resolve every TPU worker"
            )

        from easydel.inference import eSurge

        if distributed and self.config.sample_distributed_hosts:
            hosts = list(self.config.sample_distributed_hosts)
            if len(hosts) != self.config.num_processes:
                raise ValueError(
                    "sample_distributed_hosts must contain one hostname per process: "
                    f"hosts={hosts!r} num_processes={self.config.num_processes}"
                )
            from easydel.inference.esurge.distributed import (
                controller as distributed_controller,
            )
            from easydel.inference.esurge.distributed.discovery import DiscoveryResult

            def resolve_ordered_hosts(service_name: str, world_size: int | None = None) -> DiscoveryResult:
                del service_name
                if world_size is not None and int(world_size) != len(hosts):
                    raise ValueError(f"Distributed world size mismatch: hosts={hosts!r} expected={int(world_size)}")
                return DiscoveryResult(hosts=hosts)

            distributed_controller.resolve_service_hosts = resolve_ordered_hosts

        adapter_index = runtime.metadata.adapter_index if runtime is not None else 0
        engine = eSurge(
            model=self._sampling_model(model_id),
            tokenizer=self.tokenizer,
            max_model_len=self.config.sample_max_model_len,
            max_num_seqs=self.config.sample_max_num_sequences,
            hbm_utilization=self.config.sample_hbm_utilization,
            page_size=self.config.sample_page_size,
            enable_prefix_caching=True,
            sampler_metrics=True,
            overlap_execution=False,
            distributed_mode=distributed,
            distributed_role="auto",
            distributed_service_name=self.config.sample_distributed_service_name,
            distributed_world_size=self.config.num_processes if distributed else None,
            distributed_rank=self.process_id if distributed else None,
            distributed_control_port=self.config.sample_distributed_control_port + adapter_index,
            distributed_advertise_addr=(
                self.config.sample_distributed_hosts[self.process_id]
                if distributed and self.config.sample_distributed_hosts
                else local_ipv4_address() if distributed else None
            ),
            distributed_auth_token=self.config.sample_distributed_auth_token if distributed else None,
            distributed_step_timeout_s=120.0,
            distributed_connect_timeout_s=120.0,
        )
        if runtime is None:
            self._base_sampling_engine = engine
        else:
            runtime.sampling_engine = engine
        return engine

    def _score_generated_tokens(
        self,
        model_id: str,
        prompt_ids: list[int],
        generated_ids: list[int],
    ) -> list[float]:
        if not generated_ids:
            return []
        full_ids = prompt_ids + generated_ids
        model_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=full_ids[:-1])])
        prepared = types.PreparedModelPassBatch(
            all_model_inputs=[model_input],
            all_targets=[full_ids[1:]],
            all_token_weights=[[1.0] * (len(full_ids) - 1)],
            all_sampling_logprobs=[[]],
            all_advantages=[[]],
            all_values=[[]],
            all_returns=[[]],
            all_model_ids=[model_id],
            all_loss_fns=["cross_entropy"],
            all_loss_fn_configs=[None],
            request_batch_slices=[("score", model_id, 0, 1)],
        )
        runtime = self._runtimes[model_id] if model_id else self._get_base_scoring_runtime()
        _, logprobs = self._run_model_indices(runtime, prepared, [0], backward=False)
        generated_start = len(prompt_ids) - 1
        values = logprobs[0][generated_start : generated_start + len(generated_ids)]
        if len(values) != len(generated_ids):
            raise RuntimeError(
                f"Generated-token score length mismatch: expected {len(generated_ids)}, got {len(values)}"
            )
        return values.tolist()

    def score_generated(self, model_id: str, prompt_ids: list[int], generated_ids: list[int]) -> list[float]:
        """Teacher-force generated tokens when the inference sampler omits metrics."""
        self._pause_sampling_engines()
        return self._score_generated_tokens(model_id, prompt_ids, generated_ids)

    def sample(self, prepared_batch: types.PreparedSampleBatch):
        if prepared_batch.needs_prompt_logprobs:
            raise NotImplementedError("eSurge does not currently expose Tinker-compatible prompt logprobs")
        if not prepared_batch.all_model_inputs:
            return {}

        rendered = render_model_input(prepared_batch.all_model_inputs)
        if jax.process_count() > 1 and self.process_id > 0:
            # EasyDeL's distributed eSurge worker executes model steps through
            # its own CPU control plane. It must not enqueue user requests.
            for model_id in dict.fromkeys(prepared_batch.all_model_ids):
                self._sampling_engine(model_id)
            return {}

        generated: list[types.GeneratedSequence] = []
        for i, item in enumerate(rendered):
            model_id = prepared_batch.all_model_ids[i]
            checkpoint_id = prepared_batch.all_checkpoint_ids[i]
            checkpoint_path = prepared_batch.all_checkpoint_paths[i]
            if model_id and checkpoint_id and self.models[model_id].loaded_checkpoint_id != checkpoint_id:
                self._load_sampler_checkpoint(model_id, checkpoint_id, checkpoint_path)

            engine = self._sampling_engine(model_id)
            params = prepared_batch.all_sampling_params[i]
            from easydel.inference import SamplingParams as EasyDeLSamplingParams

            easy_params = EasyDeLSamplingParams(
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                seed=params.seed,
                stop_token_ids=params.stop_tokens or [],
                stop=params.stop_strings or [],
                top_k=0 if params.top_k < 0 else params.top_k,
                top_p=params.top_p,
                logprobs=1,
            )
            prompt_text = self.tokenizer.decode(
                item.prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            roundtrip = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if roundtrip != item.prompt_ids:
                raise ValueError("Pretokenized prompt does not round-trip through the eSurge tokenizer")
            output = engine.generate(prompt_text, sampling_params=easy_params)[0].outputs[0]
            token_logprobs = []
            for token_id, values in zip(output.token_ids, output.logprobs or []):
                token_logprobs.append(float(values.get(token_id, next(iter(values.values()), 0.0))))
            if len(token_logprobs) != len(output.token_ids):
                if jax.process_count() == 1:
                    token_logprobs = self._score_generated_tokens(model_id, item.prompt_ids, list(output.token_ids))
                else:
                    token_logprobs = []
            stop_reason = "length" if output.finish_reason == "length" else "stop"
            generated.append(
                types.GeneratedSequence(
                    stop_reason=stop_reason,
                    tokens=list(output.token_ids),
                    logprobs=token_logprobs,
                )
            )

        results = {}
        for request_id, _, start, end, _ in prepared_batch.request_batch_slices:
            results[request_id] = types.SampleOutput(sequences=generated[start:end], prompt_logprobs=None)
        return results


class EasyDeLBackend(EasyDeLBackendImpl):
    """Coordinator wrapper that dispatches each Tinker operation to JAX hosts."""

    def __init__(self, base_model: str, config: EasyDeLBackendConfig):
        self.process_id = 0
        self._command_server: CommandServer | None = None
        if config.coordinator_address is not None:
            if config.num_processes is None:
                raise ValueError("num_processes is required when coordinator_address is set")
            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=self.process_id,
            )
            # Every host must enter PJRT initialization before any worker
            # blocks on the out-of-band command channel.
            jax.devices("tpu")
            self._command_server = CommandServer(config.coordinator_address, config.num_processes)
            logger.info(
                "EasyDeL JAX distributed initialized: process_id=0, process_count=%d, local_devices=%d",
                jax.process_count(),
                jax.local_device_count(),
            )
        self._broadcast_and_call("__init__", base_model=base_model, config=config, process_id=self.process_id)

    def _broadcast_and_call(self, method: str, **kwargs):
        if jax.process_count() > 1:
            hints = get_type_hints(getattr(EasyDeLBackendImpl, method))

            def serialize(key, value):
                if hints.get(key) is AnyPath:
                    return str(value)
                return TypeAdapter(hints[key]).dump_python(value, mode="json") if key in hints else value

            if self._command_server is None:
                raise RuntimeError("EasyDeL command server was not initialized")
            self._command_server.send(
                RpcPayload(method=method, kwargs={key: serialize(key, value) for key, value in kwargs.items()}),
            )
        local_error: BaseException | None = None
        result = None
        try:
            result = getattr(super(), method)(**kwargs)
        except BaseException as exc:
            local_error = exc
        worker_error: BaseException | None = None
        if jax.process_count() > 1:
            try:
                assert self._command_server is not None
                self._command_server.wait()
            except BaseException as exc:
                worker_error = exc
        if local_error is not None:
            if worker_error is not None:
                local_error.add_note(f"A remote worker also failed: {worker_error}")
            raise local_error
        if worker_error is not None:
            raise worker_error
        return result

    def create_model(
        self,
        model_id: str,
        lora_config: types.LoraConfig,
        model_role: str = "policy",
    ) -> None:
        self._broadcast_and_call(
            "create_model",
            model_id=model_id,
            lora_config=lora_config,
            model_role=model_role,
        )

    def delete_model(self, model_id: str) -> None:
        self._broadcast_and_call("delete_model", model_id=model_id)

    def shutdown(self) -> None:
        try:
            self._broadcast_and_call("shutdown")
        finally:
            if self._command_server is not None:
                self._command_server.close()
                self._command_server = None
            if jax.process_count() > 1:
                jax.distributed.shutdown()

    def forward_backward(self, prepared_batch: types.PreparedModelPassBatch):
        return self._broadcast_and_call("forward_backward", prepared_batch=prepared_batch)

    def forward(self, prepared_batch: types.PreparedModelPassBatch):
        return self._broadcast_and_call("forward", prepared_batch=prepared_batch)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput):
        return self._broadcast_and_call("optim_step", model_id=model_id, request_data=request_data)

    def sample(self, prepared_batch: types.PreparedSampleBatch):
        result = self._broadcast_and_call("sample", prepared_batch=prepared_batch)
        if jax.process_count() == 1:
            return result

        rendered = render_model_input(prepared_batch.all_model_inputs)
        for request_id, _, start, end, _ in prepared_batch.request_batch_slices:
            output = result[request_id]
            for offset, sequence in enumerate(output.sequences):
                if sequence.logprobs:
                    continue
                index = start + offset
                if index >= end:
                    raise RuntimeError(f"Sample result {request_id!r} has more sequences than prepared inputs")
                sequence.logprobs = self._broadcast_and_call(
                    "score_generated",
                    model_id=prepared_batch.all_model_ids[index],
                    prompt_ids=rendered[index].prompt_ids,
                    generated_ids=sequence.tokens,
                )
        return result

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        self._broadcast_and_call("save_checkpoint", output_path=output_path, model_id=model_id)

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        if jax.process_count() > 1:
            local_checkpoint = Path(str(checkpoint_path))
            if not local_checkpoint.is_file():
                raise FileNotFoundError(f"EasyDeL checkpoint not found in {checkpoint_path}")
            self._broadcast_and_call(
                "load_checkpoint_payload",
                checkpoint_payload=base64.b64encode(local_checkpoint.read_bytes()).decode("ascii"),
                model_id=model_id,
            )
            return
        self._broadcast_and_call("load_checkpoint", checkpoint_path=checkpoint_path, model_id=model_id)

    def load_sampler_checkpoint(self, model_id: str, checkpoint_id: str, checkpoint_path: str) -> None:
        if jax.process_count() > 1:
            local_checkpoint = Path(checkpoint_path)
            if not local_checkpoint.is_file():
                raise FileNotFoundError(f"EasyDeL sampler checkpoint not found in {checkpoint_path}")
            self._broadcast_and_call(
                "load_sampler_checkpoint_payload",
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                checkpoint_payload=base64.b64encode(local_checkpoint.read_bytes()).decode("ascii"),
            )
            return
        self._broadcast_and_call(
            "load_sampler_checkpoint",
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
        )

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_name(output_path.name + ".probe").write_text("write_probe")
        self._broadcast_and_call(
            "save_sampler_checkpoint",
            output_path=output_path,
            model_id=model_id,
            persist=persist,
        )


def run_worker(coordinator_address: str, num_processes: int, process_id: int) -> None:
    """Run a non-coordinator EasyDeL process in lockstep with Tinker process 0."""
    if process_id <= 0:
        raise ValueError("EasyDeL worker process_id must be greater than zero")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )
    jax.devices("tpu")

    command_client = CommandClient(coordinator_address, process_id)
    init_payload = command_client.receive()
    if init_payload.method != "__init__":
        raise RuntimeError(f"Expected EasyDeL __init__ command, got {init_payload.method}")
    config = EasyDeLBackendConfig.model_validate(init_payload.kwargs["config"])
    try:
        backend = EasyDeLBackendImpl(init_payload.kwargs["base_model"], config, process_id)
    except BaseException as exc:
        command_client.acknowledge(exc)
        raise
    command_client.acknowledge()
    logger.info("EasyDeL worker %d entered command loop", process_id)

    try:
        while True:
            payload: RpcPayload = command_client.receive()
            try:
                if not hasattr(backend, payload.method):
                    raise RuntimeError(f"Unknown EasyDeL worker method {payload.method!r}")
                method = getattr(backend, payload.method)
                hints = get_type_hints(method)
                kwargs = {
                    key: TypeAdapter(hints[key]).validate_python(value) if key in hints else value
                    for key, value in payload.kwargs.items()
                }
                method(**kwargs)
            except BaseException as exc:
                command_client.acknowledge(exc)
                raise
            command_client.acknowledge()
            if payload.method == "shutdown":
                break
    finally:
        command_client.close()
        jax.distributed.shutdown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SkyRL EasyDeL Tinker worker")
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", required=True, type=int)
    parser.add_argument("--process-id", required=True, type=int)
    args = parser.parse_args()
    run_worker(args.coordinator_address, args.num_processes, args.process_id)


if __name__ == "__main__":
    main()
