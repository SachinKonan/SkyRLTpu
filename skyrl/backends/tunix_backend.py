"""Tunix-powered backend for TinkerEngine.

Trains LoRA adapters (via qwix) on tunix models — native flax-NNX model
implementations today, MaxText-wrapped models via ``model_source="maxtext"``.

Design notes:
  - One frozen base model is loaded once. For each LoRA rank in use, a
    qwix-wrapped "template" module is created lazily; templates share the
    base weight arrays by construction (``nnx.merge`` references, never
    copies, the underlying jax.Arrays).
  - Each Tinker model (model_id) owns a ModelSlot: its LoRA parameter state,
    its optimizer (AdamW with injected hyperparams), and its gradient
    accumulation buffer. States are swapped into the rank-template with
    ``nnx.update`` — same shapes, so no recompilation.
  - forward/forward_backward take Tinker's pre-shifted inputs/targets:
    ``target_logprobs[t] = log_softmax(logits[t])[target_ids[t]]`` with NO
    internal shifting, matching the JaxBackend semantics exactly.
  - Sampling runs either through an in-process pure-JAX generation loop
    (``inference_backend="native"``, used for CPU tests and single-slice
    runs) or through an external/colocated vLLM server with inflight LoRA
    adapter updates (``inference_backend="vllm"``, reusing VllmSamplingClient).
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from cloudpathlib import AnyPath
from flax import nnx
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer

from skyrl.backends.backend import AbstractBackend
from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import pad_batch, pad_to_fsdp
from skyrl.backends.vllm_sampling import GroupedCompletion, VllmSamplingClient
from skyrl.tinker import types
from skyrl.tinker.loss_fns import LOSS_FUNCTIONS, LossFnConfig
from skyrl.tinker.types import LOSS_TYPES
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack, pack_and_upload

_DEFAULT_PPO_CLIP_LOW_THRESHOLD = 0.8
_DEFAULT_PPO_CLIP_HIGH_THRESHOLD = 1.2

# qwix targets tunix-native module names. Attention projections and MLP
# projections are separate groups so LoraConfig.train_attn/train_mlp map
# onto them.
_NATIVE_ATTN_REGEX = r".*q_proj|.*k_proj|.*v_proj|.*o_proj"
_NATIVE_MLP_REGEX = r".*gate_proj|.*up_proj|.*down_proj"

_SELF_CHECKPOINT_FILE = "tunix_lora_checkpoint.msgpack.npz"
_CHECKPOINT_META_FILE = "tunix_checkpoint_meta.json"
_EPHEMERAL_MARKER_FILE = "tunix_ephemeral_marker.json"


class TunixBackendConfig(BaseModel, extra="forbid"):
    """Configuration specific to the tunix backend."""

    model_source: Literal["huggingface", "maxtext"] = Field(
        default="huggingface",
        description="Where the training model implementation comes from: native tunix modules loaded "
        "from HF safetensors, or a MaxText model wrapped with the tunix adapter.",
    )
    model_path: str | None = Field(
        default=None,
        description="Optional local checkpoint dir (huggingface source) or load_parameters_path (maxtext source).",
    )
    maxtext_model_name: str | None = Field(
        default=None,
        description="Override for the MaxText model_name when model_source='maxtext'.",
    )
    maxtext_max_target_length: int = Field(
        default=4096,
        description="MaxText max_target_length (must cover the longest train/sample sequence).",
    )
    maxtext_kwargs: dict = Field(
        default_factory=dict,
        description="Extra MaxTextConfig field overrides forwarded to the MaxText pyconfig.",
    )
    maxtext_ckpt_cache_dir: str | None = Field(
        default=None,
        description="Cache dir for HF->orbax converted MaxText checkpoints "
        "(default ~/.cache/skyrl_tunix/maxtext). Conversion runs once per model.",
    )
    inference_backend: Literal["native", "vllm"] = Field(
        default="native",
        description="Sampling backend: 'native' runs an in-process JAX generation loop; "
        "'vllm' forwards sampling to a vLLM server with inflight LoRA updates.",
    )
    max_lora_rank: int = Field(default=32, description="Maximum LoRA rank accepted from clients")
    lora_attn_regex: str | None = Field(
        default=None,
        description="Override for the qwix module_path regex used for attention projections.",
    )
    lora_mlp_regex: str | None = Field(
        default=None,
        description="Override for the qwix module_path regex used for MLP projections.",
    )
    train_micro_batch_size: int = Field(
        default=0,
        description="Micro-batch size (sequences) for gradient accumulation; 0 means full batch.",
    )
    train_token_budget: int = Field(
        default=0,
        description=(
            "When >0, ignore train_micro_batch_size and pack variable-size "
            "micro-batches: sort examples by length and fill each micro-batch "
            "while padded_rows * padded_seq_len <= budget. Memory-safe iff a "
            "single sequence of `budget` tokens fits the fb graph (activations "
            "and logits scale with total padded tokens; attention only shrinks "
            "when the same tokens are split across shorter rows). Turns "
            "hundreds of singleton fb calls per RL batch into a few dozen."
        ),
    )
    sample_max_num_sequences: int = Field(
        default=0,
        description="Maximum concurrent sequences per native generation call; 0 means full batch.",
    )
    flce_tile_size: int = Field(
        default=0,
        description=(
            "When >0 (MaxText only), compute target log-probs with fused-linear "
            "cross-entropy: the model returns hidden [B,T,H] (needs "
            "maxtext_kwargs.num_vocab_tiling>1 so the decoder skips the output "
            "head) and the loss projects the flattened B*T token axis in tiles "
            "of THIS MANY TOKENS, so the full [B*T, V] logits are never "
            "materialized in the forward or backward. A fixed token size (not a "
            "tile count) keeps the per-tile [tile_size, V] footprint bounded at "
            "every call site — the KL forward runs far larger batches than the "
            "train fb. ~2048 fits comfortably. 0 keeps the single-shot path."
        ),
    )
    enforce_eager: bool = Field(default=False, description="Disable JAX JIT compilation")
    param_dtype: str = Field(default="float32", description="Parameter dtype for the base model (float32/bfloat16)")

    # vLLM sampling client (mirrors JaxBackendConfig so launch scripts stay uniform)
    vllm_base_url: str | None = Field(default=None, description="Base URL(s) for the vLLM server, comma-separated")
    vllm_model_name: str | None = Field(default=None, description="vLLM model name for base-model requests")
    vllm_api_key: str = Field(default="EMPTY", description="Bearer token for the vLLM OpenAI API")
    vllm_lora_base_dir: Path = Field(
        default=Path("/tmp/skyrl_tunix_vllm_loras"),
        description="Directory where LoRA sampler checkpoints are extracted before loading into vLLM.",
    )
    vllm_lora_load_endpoint: str = Field(default="/v1/load_lora_adapter")
    vllm_lora_unload_endpoint: str = Field(default="/v1/unload_lora_adapter")
    vllm_lora_upload_endpoint: str = Field(
        default="",
        description="When set (e.g. /skyrl/v1/upload_lora_adapter), ephemeral adapters are pushed "
        "over HTTP to the vLLM server's local disk instead of via shared storage.",
    )
    vllm_lora_load_retries: int = Field(default=3, ge=1)
    vllm_lora_load_retry_sleep_sec: float = Field(default=2.0, ge=0.0)
    vllm_request_timeout_sec: float = Field(default=300.0)
    vllm_max_concurrent_requests: int = Field(default=64)
    vllm_client_side_round_robin: bool = Field(default=False)
    vllm_group_completions: bool = Field(default=True)


def round_up_seq_len(seq_len: int) -> int:
    """Round up to two significant bits to bound the number of JIT shapes.

    (Same bucketing scheme as skyrl.tx.utils.models.round_up_seq_len, kept
    local so this backend has no dependency on the tx library.)
    """
    if seq_len <= 4:
        return max(seq_len, 1)
    msb = 1 << (seq_len - 1).bit_length() - 1
    step = msb // 2
    return ((seq_len + step - 1) // step) * step


def _keystr_map(state) -> dict[str, Any]:
    """Flatten a pytree/nnx.State into {path-string: leaf}."""
    return {jax.tree_util.keystr(p): v for p, v in jax.tree.flatten_with_path(state)[0]}


class _MaxTextAdapterShim(nnx.Module):
    """Adapts tunix's MaxText wrapper to the native-tunix model interface.

    The TunixMaxTextAdapter returns final logits from ``__call__`` and has no
    ``skip_lm_head`` / ``compute_final_logits`` / ``get_model_input``. It also
    ignores ``attention_mask`` entirely (MaxText applies causal masking
    internally); with right-padded batches that is safe — real tokens never
    attend forward into padding — and padded positions are excluded via the
    loss mask / generation bookkeeping instead.
    """

    def __init__(self, adapter: nnx.Module):
        self.adapter = adapter

    def __call__(
        self,
        input_tokens,
        positions,
        cache,
        attention_mask,
        output_hidden_states: bool = False,
        segment_ids=None,
        skip_lm_head: bool = False,
    ):
        logits, new_cache = self.adapter(input_tokens, positions, cache, attention_mask)
        return logits, new_cache

    def compute_final_logits(self, x):
        # __call__ already produced final logits; just fix the dtype.
        return x.astype(jnp.float32)

    def logits_from_hidden(self, hidden_chunk):
        """Project a hidden tile [tile, H] -> logits [tile, V] via MaxText's
        output head (decoder_norm + tied/untied lm-head, exactly what
        apply_output_head does). Used by the FLCE loss to project token tiles
        without ever forming the full [B, T, V]. Requires num_vocab_tiling>1 so
        __call__ returns hidden instead of logits."""
        return self.adapter.base.logits_from_hidden_states_for_vocab_tiling(hidden_chunk, True, "train")

    def get_model_input(self):
        # Batch must be divisible by every mesh axis the decoder shards it
        # over. qwen3.5's GatedDeltaNet uses a shard_map kernel with batch
        # sharded over 'fsdp' (= device count on a single-host trainer), so a
        # fixed batch of 2 fails on 4-chip meshes. device_count is a multiple
        # of any batch-sharding axis in the meshes we build (single-axis fsdp).
        batch, seq_len = max(2, jax.device_count()), 4
        return {
            "input_tokens": jnp.ones((batch, seq_len), dtype=jnp.int32),
            "positions": jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len)),
            "cache": None,
            "attention_mask": jnp.ones((batch, seq_len, seq_len), dtype=jnp.bool_),
        }


# MaxText's attention kernels require q_seq_len to be a multiple of the query
# block size once it exceeds one block (default sa_block_q=512): shorter
# sequences run as a single block and may use the usual power-of-2 buckets.
_MAXTEXT_SEQ_BLOCK = 512

# qwix module_path regex for MaxText decoders (pure-NNX). The qwen-family arm
# matches MaxText's own verified LoRA targets
# (configs/post_train/lora_module_path.yml); the layers_N arm covers decoders
# whose layers are attributes named layers_<i> — gemma4's grouped scan
# (decoder/scanned_blocks/layers_{0..5}, stacked over scan steps) and any
# unscanned decoder (decoder/layers_<i>) — with the standard
# self_attention/mlp submodule names; the GptOss arm covers gpt-oss,
# whose blocks are named GptOssAttention/{query,key,value,out} and GptOssMlp
# (experts held as raw stacked params — scoping the whole module lets qwix
# intercept the expert einsums; the router `gate` is deliberately included,
# its delta merges into the gate kernel like any other). The layer_N arm
# covers qwen3.5's inhomogeneous scan (decoder/layers/layer_{0..3}, stacked
# over scan steps): full-attention layers nest as
# layer_N/attention/attention/{query,key,value,out} (DecoderLayer.attention =
# Qwen3_5FullAttention wrapping attentions.Attention; query kernel fuses
# query+output-gate), dense MLP as layer_N/mlp/{wi_0,wi_1,wo}. GDN layers
# (layer_N/attention/{in_proj_qkvz,in_proj_ba,conv1d,out_proj,...}) are
# deliberately EXCLUDED for v1 — neither arm matches their module names.
_MAXTEXT_ATTN_REGEX = (
    r"(?:.*/)?(?:decoder/)?layers/(?:[0-9]+/)?self_attention/(?:query|key|value|out)(?:/.*)?"
    r"|(?:.*/)?layers_[0-9]+/self_attention/(?:query|key|value|out)(?:/.*)?"
    r"|(?:.*/)?layer_[0-9]+/attention/attention/(?:query|key|value|out)(?:/.*)?"
    r"|(?:.*/)?GptOssAttention/(?:query|key|value|out)(?:/.*)?"
)
_MAXTEXT_MLP_REGEX = (
    r"(?:.*/)?(?:decoder/)?layers/(?:[0-9]+/)?mlp/(?:wi_0|wi_1|wo)(?:/.*)?"
    r"|(?:.*/)?layers_[0-9]+/mlp/(?:wi_0|wi_1|wo)(?:/.*)?"
    r"|(?:.*/)?layer_[0-9]+/mlp/(?:wi_0|wi_1|wo)(?:/.*)?"
    r"|(?:.*/)?GptOssMlp(?:/.*)?"
)

# MaxText projection name -> HF (block, module) for PEFT export.
_MAXTEXT_PROJ_TO_HF = {
    ("self_attention", "query"): ("self_attn", "q_proj"),
    ("self_attention", "key"): ("self_attn", "k_proj"),
    ("self_attention", "value"): ("self_attn", "v_proj"),
    ("self_attention", "out"): ("self_attn", "o_proj"),
    ("mlp", "wi_0"): ("mlp", "gate_proj"),
    ("mlp", "wi_1"): ("mlp", "up_proj"),
    ("mlp", "wo"): ("mlp", "down_proj"),
}


def _sample_token_row(logits_row, temp, top_k, top_p, key):
    """Sample one token from one row of raw logits (greedy when temp<=0).

    Returns (token, raw_logprob_of_token). Logprobs are from the raw
    (untempered) distribution so they compare directly to training logprobs.
    """
    raw_logps = jax.nn.log_softmax(logits_row)
    greedy = jnp.argmax(logits_row)

    scaled = logits_row / jnp.maximum(temp, 1e-6)
    # top_k: mask logits below the k-th largest (top_k <= 0 disables)
    sorted_desc = jnp.sort(scaled)[::-1]
    k_idx = jnp.clip(top_k - 1, 0, scaled.shape[0] - 1)
    kth = jnp.where(top_k > 0, sorted_desc[k_idx], -jnp.inf)
    masked = jnp.where(scaled >= kth, scaled, -jnp.inf)
    # top_p nucleus filtering (always keeps the argmax)
    probs = jax.nn.softmax(masked)
    sort_idx = jnp.argsort(-probs)
    sorted_probs = probs[sort_idx]
    cumulative = jnp.cumsum(sorted_probs) - sorted_probs
    keep_sorted = cumulative < top_p
    keep = jnp.zeros_like(keep_sorted).at[sort_idx].set(keep_sorted)
    masked = jnp.where(keep, masked, -jnp.inf)

    sampled = jax.random.categorical(key, masked)
    token = jnp.where(temp <= 0.0, greedy, sampled)
    return token.astype(jnp.int32), raw_logps[token]


_sample_token_rows = jax.vmap(_sample_token_row)


def _step_keys(seeds, step_idx):
    return jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), step_idx))(seeds)


@contextmanager
def _maxtext_config_cwd():
    """chdir into a temp dir whose ``src`` symlink points at maxtext's configs.

    pip-installed MaxText resolves the bare ``base.yml`` argv entry (which the
    tunix AutoModel path passes) against ``<cwd>/src``.
    """
    import os
    import tempfile

    import maxtext

    configs_dir = Path(maxtext.__file__).parent / "configs"
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "src").symlink_to(configs_dir)
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(old_cwd)


@dataclass
class ModelSlot:
    """Per-model_id state hosted by the backend."""

    lora_config: types.LoraConfig
    template_key: tuple
    lora_state: nnx.State
    optimizer: nnx.Optimizer
    accum_grads: Any = None  # pytree matching lora_state, or None
    accum_count: int = 0
    loaded_sampler_checkpoint_id: str | None = None
    sampler_lora_states: dict = field(default_factory=dict)  # checkpoint_id -> lora state


@dataclass
class _Template:
    """A qwix-wrapped copy of the base model for one (rank, attn, mlp) combo."""

    model: nnx.Module
    graphdef: nnx.GraphDef
    rest_state: Any  # non-LoRA state (shares base arrays)
    lora_shape: nnx.State  # reference LoRA state (shapes/structure)
    forward_backward_fn: Callable
    forward_fn: Callable
    # "native": pure split/merge fns taking (lora_state, rest_state, ...).
    # "maxtext": nnx-lifted fns taking (model, ...) — the MaxText decoder
    # self-mutates state during forward, which breaks raw jax transforms.
    kind: str = "native"
    generate_fn: dict = field(default_factory=dict)  # (B, prompt_len, steps) -> jitted fn


class TunixBackend(AbstractBackend):
    """Tinker backend that trains qwix-LoRA adapters on tunix models."""

    def __init__(self, base_model: str, config: TunixBackendConfig):
        self.base_model = base_model
        self.config = config
        self.metrics = types.EngineMetrics()

        self.vllm_client: VllmSamplingClient | None = None
        if config.inference_backend == "vllm":
            if not config.vllm_base_url:
                raise ValueError("TunixBackendConfig.vllm_base_url is required when inference_backend='vllm'")
            self.vllm_client = VllmSamplingClient(
                base_url=config.vllm_base_url,
                model_name=config.vllm_model_name or base_model,
                api_key=config.vllm_api_key,
                lora_base_dir=config.vllm_lora_base_dir,
                lora_load_endpoint=config.vllm_lora_load_endpoint,
                lora_unload_endpoint=config.vllm_lora_unload_endpoint,
                lora_upload_endpoint=config.vllm_lora_upload_endpoint,
                lora_load_retries=config.vllm_lora_load_retries,
                lora_load_retry_sleep_sec=config.vllm_lora_load_retry_sleep_sec,
                request_timeout_sec=config.vllm_request_timeout_sec,
                max_concurrent_requests=config.vllm_max_concurrent_requests,
                client_side_round_robin=config.vllm_client_side_round_robin,
            )

        # For maxtext, model_path is an orbax weights dir with no tokenizer
        # files — the tokenizer always comes from the HF id.
        tokenizer_src = base_model if config.model_source == "maxtext" else (config.model_path or base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

        base = self._load_base_model()
        self.base_graphdef, self.base_state = nnx.split(base)

        self.templates: dict[tuple, _Template] = {}
        self.models: dict[str, ModelSlot] = {}
        self._generate_fn_cache: dict = {}

        logger.info(
            f"Initialized tunix backend for {base_model} "
            f"(model_source={config.model_source}, inference_backend={config.inference_backend})"
        )

    # ------------------------------------------------------------------ model loading

    def _load_base_model(self) -> nnx.Module:
        if self.config.model_source == "maxtext":
            return self._load_maxtext_model()
        return self._load_native_model()

    def _native_model_config(self, hf_config) -> Any:
        """Build a tunix qwen3 ModelConfig from an HF config object."""
        from tunix.models.qwen3 import model as qwen3_model

        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
            rope_theta = rope_params.get("rope_theta", 1_000_000)
        param_dtype = jnp.bfloat16 if self.config.param_dtype == "bfloat16" else jnp.float32
        return qwen3_model.ModelConfig(
            num_layers=hf_config.num_hidden_layers,
            vocab_size=hf_config.vocab_size,
            embed_dim=hf_config.hidden_size,
            hidden_dim=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            head_dim=head_dim,
            num_kv_heads=hf_config.num_key_value_heads,
            rope_theta=int(rope_theta),
            norm_eps=hf_config.rms_norm_eps,
            use_tied_embedding=bool(getattr(hf_config, "tie_word_embeddings", False)),
            num_experts=getattr(hf_config, "num_experts", None),
            num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", None),
            dtype=param_dtype,
            param_dtype=param_dtype,
        )

    def _load_native_model(self) -> nnx.Module:
        from huggingface_hub import snapshot_download
        from tunix.models.qwen3 import params as qwen3_params

        hf_config = AutoConfig.from_pretrained(self.config.model_path or self.base_model)
        model_type = getattr(hf_config, "model_type", "")
        if model_type not in ("qwen3", "qwen3_moe"):
            raise ValueError(
                f"tunix backend currently supports qwen3-family native models, got model_type={model_type!r}. "
                "Use model_source='maxtext' for other families."
            )

        checkpoint_dir = self.config.model_path
        if checkpoint_dir is None or not Path(checkpoint_dir).exists():
            checkpoint_dir = snapshot_download(self.base_model, allow_patterns=["*.safetensors", "*.json"])

        model_config = self._native_model_config(hf_config)
        model = qwen3_params.create_model_from_safe_tensors(
            str(checkpoint_dir), model_config, dtype=model_config.param_dtype
        )
        logger.info(f"Loaded native tunix qwen3 model from {checkpoint_dir}")
        return model

    def _maxtext_model_name(self) -> str:
        if self.config.maxtext_model_name:
            return self.config.maxtext_model_name
        try:
            from tunix.models import naming

            return naming.ModelNaming(model_id=self.base_model).model_name
        except Exception as e:
            raise ValueError(
                f"Cannot derive a MaxText model_name for {self.base_model!r} "
                "(family unknown to tunix naming). Set maxtext_model_name in the "
                "backend config to the MaxText config name, e.g. 'gpt-oss-20b'."
            ) from e

    def _ensure_maxtext_orbax_checkpoint(self) -> str:
        """Convert the HF checkpoint to MaxText's orbax format (cached).

        MaxText's ``load_parameters_path`` only accepts orbax checkpoints; raw
        HF safetensors silently yield a random-weight model. We run MaxText's
        own converter once and cache the result.
        """
        import subprocess
        import sys

        mt_name = self._maxtext_model_name()
        cache_root = Path(
            self.config.maxtext_ckpt_cache_dir or Path.home() / ".cache" / "skyrl_tunix" / "maxtext"
        ).expanduser() / mt_name
        items_dir = cache_root / "0" / "items"
        if items_dir.exists():
            logger.info(f"Using cached MaxText orbax checkpoint at {items_dir}")
            return str(items_dir)

        logger.info(f"Converting {self.base_model} to MaxText orbax format at {cache_root} (one-time)")
        cache_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "maxtext.checkpoint_conversion.to_maxtext",
            f"model_name={mt_name}",
            f"base_output_directory={cache_root}",
            "scan_layers=True",
            "use_multimodal=false",
            "skip_jax_distributed_system=True",
            "--lazy_load_tensors=True",
            "checkpoint_storage_use_ocdbt=True",
            "checkpoint_storage_use_zarr3=True",
        ]
        import os

        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"  # conversion must not grab the TPU held by this process
        with _maxtext_config_cwd():
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"MaxText checkpoint conversion failed (exit {result.returncode}):\n{result.stderr[-4000:]}"
            )
        if not items_dir.exists():
            raise RuntimeError(f"MaxText conversion succeeded but {items_dir} does not exist")
        logger.info(f"Converted MaxText checkpoint cached at {items_dir}")
        return str(items_dir)

    def _load_maxtext_model(self) -> nnx.Module:
        """Load a MaxText model via pyconfig directly.

        Deliberately bypasses tunix's AutoModel: its naming layer only resolves
        a fixed set of families (no gpt-oss, no qwen3.5) and it ignores
        model_path for the MAXTEXT source. Any model with a MaxText config
        (configs/models/<name>.yml) works here via maxtext_model_name.
        """
        try:
            import maxtext.configs.pyconfig as pyconfig
            from maxtext.utils import model_creation_utils
        except ImportError as e:  # pragma: no cover
            raise ImportError("model_source='maxtext' requires MaxText installed") from e

        mt_name = self._maxtext_model_name()
        # model_path override wins (e.g. a pre-converted orbax dir); otherwise
        # convert-and-cache from HF.
        orbax_path = self.config.model_path or self._ensure_maxtext_orbax_checkpoint()

        overrides: dict[str, Any] = {
            "model_name": mt_name,
            "load_parameters_path": orbax_path,
            "per_device_batch_size": 1,
            "max_target_length": self.config.maxtext_max_target_length,
            # Required True when load_parameters_path is set (pyconfig validation).
            "enable_checkpointing": True,
            # qwix LoRA only works on the pure-NNX decoder; the linen/ToNNX
            # hybrid silently yields zero LoRA params.
            "pure_nnx": True,
            "pure_nnx_decoder": True,
            # Activation dtype must equal weight dtype: the f32 LoRA delta
            # otherwise changes the nnx.scan carry dtype mid-scan.
            "dtype": self.config.param_dtype,
            "weight_dtype": self.config.param_dtype,
            "skip_jax_distributed_system": True,
        }
        if "gpt-oss" in mt_name:
            # qwix cannot inject LoRA into the megablox Pallas gmm kernel;
            # expert adapters require the dense einsum MoE path (costs
            # E/top_k more MoE FLOPs — acceptable at current utilization).
            overrides["sparse_matmul"] = False
            overrides["megablox"] = False
        overrides.update(self.config.maxtext_kwargs)
        argv = ["", "base.yml"] + [
            f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()
        ]

        with _maxtext_config_cwd():
            maxtext_config = pyconfig.initialize(argv)
            model = model_creation_utils.from_pretrained(
                maxtext_config, mesh=None, wrap_with_tunix_adapter=True
            )
        if isinstance(model, tuple):  # maxtext returns (model, mesh) when mesh=None
            model, mesh = model
            # KEEP the mesh. It used to be dropped on the floor, which meant every
            # jitted model pass ran with no mesh context and no sharding
            # annotations -- the likely reason per-device memory looks like the
            # weights are replicated rather than sharded across the chips.
            self._mesh = mesh
            logger.info(f"MaxText mesh: shape={getattr(mesh, 'shape', None)} axes={getattr(mesh, 'axis_names', None)}")
        logger.info(f"Loaded MaxText model {mt_name} for {self.base_model} (pure-NNX)")
        self._log_param_sharding(model)
        return _MaxTextAdapterShim(model)

    def _log_param_sharding(self, model) -> None:
        """One-shot diagnostic: are params actually sharded, and how much HBM is live?

        We have been unable to account for ~48GB/chip: sharded 27B weights should
        occupy ~13.5GB of a ~95GB chip, yet only ~32GB was free. This prints the
        real per-device footprint and the sharding of the largest parameters, which
        distinguishes "weights replicated" from "compiled-program arenas piling up".
        """
        try:
            import numpy as _np
            from flax import nnx as _nnx

            state = _nnx.state(model, _nnx.Param)
            leaves = [x for x in jax.tree.leaves(state) if hasattr(x, "shape") and hasattr(x, "sharding")]
            total = sum(int(_np.prod(x.shape)) * x.dtype.itemsize for x in leaves)
            # per-device bytes = addressable shards only
            per_dev = 0
            for x in leaves:
                try:
                    per_dev += sum(s.data.nbytes for s in x.addressable_shards[:1])
                except Exception:
                    pass
            logger.info(
                f"PARAM DIAG: {len(leaves)} arrays, global={total / 1e9:.1f}GB, "
                f"per-device={per_dev / 1e9:.1f}GB "
                f"(sharded/{jax.device_count()} would be ~{total / 1e9 / max(1, jax.device_count()):.1f}GB, "
                f"replicated would be ~{total / 1e9:.1f}GB)"
            )
            for x in sorted(leaves, key=lambda a: -int(_np.prod(a.shape)))[:3]:
                logger.info(f"PARAM DIAG: shape={tuple(x.shape)} sharding={x.sharding}")
            for d in jax.devices()[:2]:
                s = d.memory_stats() or {}
                lim = s.get("bytes_limit") or s.get("bytes_reservable_limit") or 0
                logger.info(
                    f"HBM DIAG dev{d.id}: in_use={s.get('bytes_in_use', 0) / 1e9:.1f}GB "
                    f"peak={s.get('peak_bytes_in_use', 0) / 1e9:.1f}GB limit={lim / 1e9:.1f}GB"
                )
        except Exception as e:  # diagnostics must never break model load
            logger.info(f"PARAM DIAG unavailable: {e}")

    # ------------------------------------------------------------------ templates

    def _template_key(self, lora_config: types.LoraConfig) -> tuple:
        return (lora_config.rank, lora_config.train_attn, lora_config.train_mlp)

    def _module_path_regex(self, lora_config: types.LoraConfig) -> str:
        is_maxtext = self.config.model_source == "maxtext"
        default_attn = _MAXTEXT_ATTN_REGEX if is_maxtext else _NATIVE_ATTN_REGEX
        default_mlp = _MAXTEXT_MLP_REGEX if is_maxtext else _NATIVE_MLP_REGEX
        parts = []
        if lora_config.train_attn:
            parts.append(self.config.lora_attn_regex or default_attn)
        if lora_config.train_mlp:
            parts.append(self.config.lora_mlp_regex or default_mlp)
        if not parts:
            raise ValueError("At least one of train_attn/train_mlp must be True")
        return "|".join(parts)

    def _wrap_with_lora(self, lora_config: types.LoraConfig, seed: int) -> nnx.Module:
        """Apply qwix LoRA to a fresh merge of the base model (shares base arrays)."""
        import qwix

        model = nnx.merge(self.base_graphdef, self.base_state)
        provider = qwix.LoraProvider(
            module_path=self._module_path_regex(lora_config),
            rank=lora_config.rank,
            alpha=lora_config.alpha,
        )
        model_input = model.get_model_input()
        return qwix.apply_lora_to_model(model, provider, rngs=nnx.Rngs(seed), **model_input)

    def _get_template(self, lora_config: types.LoraConfig) -> _Template:
        key = self._template_key(lora_config)
        if key in self.templates:
            return self.templates[key]

        model = self._wrap_with_lora(lora_config, seed=0)
        graphdef, lora_state, rest_state = nnx.split(model, nnx.LoRAParam, ...)

        kind = "maxtext" if self.config.model_source == "maxtext" else "native"
        if kind == "maxtext":
            forward_backward_fn, forward_fn = self._build_model_pass_fns_nnx()
        else:
            forward_backward_fn, forward_fn = self._build_model_pass_fns(graphdef)
        template = _Template(
            model=model,
            graphdef=graphdef,
            rest_state=rest_state,
            lora_shape=lora_state,
            forward_backward_fn=forward_backward_fn,
            forward_fn=forward_fn,
            kind=kind,
        )
        self.templates[key] = template
        logger.info(f"Created LoRA template for key={key}")
        return template

    def _init_lora_state(self, lora_config: types.LoraConfig) -> nnx.State:
        """Fresh, independently-seeded LoRA state for a new model."""
        model = self._wrap_with_lora(lora_config, seed=lora_config.seed)
        return nnx.state(model, nnx.LoRAParam)

    # ------------------------------------------------------------------ jitted model passes

    @staticmethod
    def _flce_target_logprobs(project_fn, hidden, target_ids, tile_size, use_checkpoint=True):
        """Fused-linear-CE target log-probs: ``logit[target] - logsumexp(logits)``
        computed by tiling the flattened ``B*T`` token axis via a ``custom_vjp``,
        so the full ``[B*T, V]`` logits are never materialized — only
        ``[tile_size, V]`` at a time, in BOTH forward and backward.

        The ``custom_vjp`` is the load-bearing part (the flash-CE / MaxText
        ``vocab_tiling`` pattern): the FORWARD scans token tiles, accumulates
        per-tile log-probs, and DISCARDS each tile's ``[tile, V]`` logits (the
        residual is the hidden tiles, not the logits); the BACKWARD recomputes
        one tile's logits at a time with ``jax.vjp``. A plain ``lax.scan`` — even
        with ``jax.checkpoint`` — does NOT achieve this: autodiff-through-scan
        stashes every tile's logits, stacking back to the full ``[B*T, V]`` and
        OOMing (the failure hit at 22528, 52.89G). ``project_fn`` (the frozen
        output head) is captured; only ``d/d(hidden)`` flows (LoRA-only
        training), reaching the LoRA params through the transformer backward.
        ``use_checkpoint`` is accepted for signature compatibility and ignored —
        the custom_vjp subsumes it.

        ``tile_size`` is a fixed number of TOKENS per tile (not a tile count):
        the per-tile ``[tile_size, V]`` footprint stays constant across call
        sites regardless of ``B*T``.
        """
        del use_checkpoint  # subsumed by the custom_vjp
        *lead, hdim = hidden.shape
        n = 1
        for d in lead:
            n *= d
        flat = hidden.reshape(n, hdim)
        tgt = target_ids.reshape(n)
        num_tiles = -(-n // tile_size)  # ceil
        pad = num_tiles * tile_size - n
        if pad:
            flat = jnp.pad(flat, ((0, pad), (0, 0)))
            tgt = jnp.pad(tgt, ((0, pad),))
        hf = flat.reshape(num_tiles, tile_size, hdim)
        tf = tgt.reshape(num_tiles, tile_size)

        def _tile_lp(h_tile, t_tile):  # [tile, H], [tile] -> [tile]
            logits = project_fn(h_tile).astype(jnp.float32)  # [tile, V], transient
            tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
            return tl - jax.nn.logsumexp(logits, axis=-1)

        # custom_vjp over the hidden tiles: fwd discards per-tile logits (residual
        # = hidden tiles only); bwd recomputes each tile's logits via jax.vjp so
        # peak logits memory is one [tile, V] chunk in both passes. tf (targets,
        # int) and project_fn (frozen head) are captured, not differentiated.
        @jax.custom_vjp
        def _tiled(hf):
            _, lp = jax.lax.scan(lambda c, x: (c, _tile_lp(x[0], x[1])), None, (hf, tf))
            return lp  # [num_tiles, tile]

        def _tiled_fwd(hf):
            _, lp = jax.lax.scan(lambda c, x: (c, _tile_lp(x[0], x[1])), None, (hf, tf))
            return lp, hf  # residual = hidden tiles only (never the logits)

        def _tiled_bwd(hf_res, cot):  # cot: [num_tiles, tile]
            def _bwd_body(_carry, xs):
                h_tile, t_tile, c_tile = xs
                _, vjp_fn = jax.vjp(lambda h: _tile_lp(h, t_tile), h_tile)
                (grad_h_tile,) = vjp_fn(c_tile)
                return None, grad_h_tile
            _, grad_hf = jax.lax.scan(_bwd_body, None, (hf_res, tf, cot))
            return (grad_hf,)

        _tiled.defvjp(_tiled_fwd, _tiled_bwd)
        return _tiled(hf).reshape(n + pad)[:n].reshape(*lead)

    @staticmethod
    def _loss_from_logits(
        logits,
        target_ids,
        loss_mask,
        loss_fn_types,
        sampling_logprobs,
        advantages,
        loss_fn_config,
        target_logprobs=None,
    ):
        """Shared per-token loss/logprob computation on final [B, T, V] logits.

        Tinker pre-shifts inputs/targets: no internal shift here. When
        ``target_logprobs`` is supplied (FLCE path) the [B, T, V] logits are
        never formed and this only runs the RL loss on the [B, T] log-probs.
        """
        # Memory-efficient cross-entropy: target_logprob = logit[target] -
        # logsumexp(logits), in the logits' native dtype (bf16 here, matching
        # the original log_softmax). log_softmax(x)[i] == x[i] - logsumexp(x),
        # so numerics are identical — but logsumexp reduces the vocab axis away
        # (XLA fuses it) instead of returning a full [B, T, V] tensor, dropping
        # the ~7.5 GB/chip log_softmax result that pushed the fb over HBM.
        # Standard large-vocab CE trick (cf. Liger / cut-cross-entropy).
        if target_logprobs is None:
            target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)[..., 0]
            target_logprobs = target_logits - jax.nn.logsumexp(logits, axis=-1)

        def compute_loss_per_example(loss_fn_type, tl, lm, sl, adv, cfg):
            return jax.lax.switch(loss_fn_type, LOSS_FUNCTIONS, tl, lm, sl, adv, cfg)

        per_token_losses = jax.vmap(compute_loss_per_example)(
            loss_fn_types, target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config
        )
        per_seq_loss = per_token_losses.sum(axis=-1) / jnp.maximum(loss_mask.sum(axis=-1), 1e-9)
        # Sum (not mean): gradients are divided by the accumulated example count at optim_step.
        return per_seq_loss.sum(), (target_logprobs, per_token_losses)

    def _build_model_pass_fns(self, graphdef: nnx.GraphDef) -> tuple[Callable, Callable]:
        """Pure split/merge model-pass fns for native tunix models."""

        def loss_for_lora(
            lora_state,
            rest_state,
            input_ids,
            positions,
            attention_mask,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
            loss_fn_config,
        ):
            model = nnx.merge(graphdef, lora_state, rest_state)
            hidden, _ = model(input_ids, positions, None, attention_mask, skip_lm_head=True)
            logits = model.compute_final_logits(hidden)  # [B, T, V] float32
            return self._loss_from_logits(
                logits, target_ids, loss_mask, loss_fn_types, sampling_logprobs, advantages, loss_fn_config
            )

        loss_and_grad = jax.value_and_grad(loss_for_lora, argnums=0, has_aux=True)

        def forward_backward_fn(*args):
            (_, (target_logprobs, per_token_losses)), lora_grads = loss_and_grad(*args)
            return per_token_losses, target_logprobs, lora_grads

        def forward_fn(*args):
            _, (target_logprobs, per_token_losses) = loss_for_lora(*args)
            return per_token_losses, target_logprobs, None

        if self.config.enforce_eager:
            return forward_backward_fn, forward_fn
        return jax.jit(forward_backward_fn), jax.jit(forward_fn)

    def _build_model_pass_fns_nnx(self) -> tuple[Callable, Callable]:
        """nnx-lifted model-pass fns (module-passing) for MaxText models.

        The MaxText pure-NNX decoder mutates its own state during forward
        (scan bookkeeping), so raw split/merge inside jax.value_and_grad hits
        TraceContextError; nnx's lifted transforms handle the mutation.
        """
        flce_tile_size = self.config.flce_tile_size

        def loss_fn(
            model,
            input_ids,
            positions,
            attention_mask,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
            loss_fn_config,
        ):
            hidden, _ = model(input_ids, positions, None, attention_mask, skip_lm_head=True)
            if flce_tile_size > 0:
                # FLCE: model returned hidden [B,T,H] (num_vocab_tiling>1 skips
                # the output head). Project token tiles inside the loss so the
                # full [B,T,V] logits never form. compute_final_logits is NOT
                # called here — hidden is [B,T,H], not logits.
                target_logprobs = self._flce_target_logprobs(model.logits_from_hidden, hidden, target_ids, flce_tile_size)
                return self._loss_from_logits(
                    None,
                    target_ids,
                    loss_mask,
                    loss_fn_types,
                    sampling_logprobs,
                    advantages,
                    loss_fn_config,
                    target_logprobs=target_logprobs,
                )
            logits = model.compute_final_logits(hidden)
            return self._loss_from_logits(
                logits, target_ids, loss_mask, loss_fn_types, sampling_logprobs, advantages, loss_fn_config
            )

        loss_and_grad = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam), has_aux=True)

        def forward_backward_fn(model, accum_grads, *args):
            # Gradient accumulation lives INSIDE the jit with the accumulator
            # DONATED (the tunix peft_trainer convention). The old python-side
            # `accum = tree_map(add, accum, grads)` kept three LoRA-grad pytrees
            # live at once (old accum + fresh grads + new sum) and re-allocated
            # every tile; over a ~90-tile pass that device-memory churn grew until
            # a mid-pass allocation failed (deterministically around tile 55 at
            # [4,16384]). Donating accum_grads makes the update in-place: one
            # resident grads pytree, zero per-tile growth.
            (_, (target_logprobs, per_token_losses)), lora_grads = loss_and_grad(model, *args)
            new_accum = jax.tree.map(jnp.add, accum_grads, lora_grads)
            return per_token_losses, target_logprobs, new_accum

        def forward_fn(model, *args):
            _, (target_logprobs, per_token_losses) = loss_fn(model, *args)
            return per_token_losses, target_logprobs, None

        if self.config.enforce_eager:
            return forward_backward_fn, forward_fn
        return nnx.jit(forward_backward_fn, donate_argnums=1), nnx.jit(forward_fn)

    # ------------------------------------------------------------------ AbstractBackend: models

    def has_model(self, model_id: str) -> bool:
        return model_id in self.models

    def create_model(self, model_id: str, lora_config: types.LoraConfig, model_role: str = "policy") -> None:
        if model_role != "policy":
            raise ValueError(f"TunixBackend only supports model_role='policy', got {model_role!r}")
        if model_id in self.models:
            raise ValueError(f"Model {model_id} already exists")
        if not (0 < lora_config.rank <= self.config.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_config.rank} must be between 1 and {self.config.max_lora_rank}")
        if lora_config.train_unembed:
            raise ValueError("TunixBackend does not support train_unembed=True yet")

        template = self._get_template(lora_config)
        lora_state = self._init_lora_state(lora_config)

        # hyperparam_dtype must be float32: inject_hyperparams otherwise follows
        # the (possibly bfloat16) param dtype, which NaNs adamw's bias correction.
        tx = optax.inject_hyperparams(optax.adamw, hyperparam_dtype=jnp.float32)(learning_rate=0.0)
        optimizer = nnx.Optimizer(template.model, tx, wrt=nnx.LoRAParam)

        self.models[model_id] = ModelSlot(
            lora_config=lora_config,
            template_key=self._template_key(lora_config),
            lora_state=lora_state,
            optimizer=optimizer,
        )
        logger.info(f"Created model {model_id} with lora rank={lora_config.rank}, alpha={lora_config.alpha}")

    def delete_model(self, model_id: str) -> None:
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        del self.models[model_id]
        logger.info(f"Deleted model {model_id}")

    # ------------------------------------------------------------------ batch prep

    @staticmethod
    def _build_loss_fn_config(all_loss_fn_configs: list[dict[str, float] | None]) -> LossFnConfig:
        configs = [c or {} for c in all_loss_fn_configs]
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

    @staticmethod
    def _positions_and_masks(all_input_ids: list[list[int]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Positions [B, T] and causal+padding attention mask [B, T, T]."""
        batch = len(all_input_ids)
        pad_mask = np.zeros((batch, max_len), dtype=bool)
        for i, seq in enumerate(all_input_ids):
            pad_mask[i, : len(seq)] = True
        positions = np.maximum(np.cumsum(pad_mask, axis=-1) - 1, 0)
        causal = np.tril(np.ones((max_len, max_len), dtype=bool))
        attn_mask = causal[None, :, :] & pad_mask[:, None, :]
        return positions.astype(np.int32), attn_mask

    @contextmanager
    def _jit_timing_context(self, seq_len: int, mode: str):
        jit_times = self.metrics.train_seq_len_jit_times if mode == "train" else self.metrics.sample_seq_len_jit_times
        if not self.config.enforce_eager and seq_len not in jit_times:
            logger.info(f"JIT compiling for {mode} seq_len={seq_len} in progress...")
            start = time.time()
            yield
            jit_times[seq_len] = time.time() - start
            logger.info(f"JIT compilation for {mode} seq_len={seq_len} took {jit_times[seq_len]:.2f}s")
        else:
            yield

    def _row_shard(self) -> int:
        """Multiple the microbatch row count must be padded up to.

        MaxText shards the batch dim over the data/fsdp mesh axes, so rows must be
        a multiple of *that* axis -- not of the total chip count. Under pure FSDP
        the two coincide, which is why device_count was used. With tensor
        parallelism (e.g. ici_tensor_parallelism=2 x ici_fsdp_parallelism=2 on 4
        chips) the batch axis is only 2, so padding to 4 doubles the rows -- and
        therefore the activation footprint of every tile -- for nothing.

        TUNIX_ROW_SHARD lets the deployment declare the real batch axis. It MUST
        match the MaxText mesh: too small and the batch will not shard evenly.
        """
        try:
            override = int(os.environ.get("TUNIX_ROW_SHARD", "0") or "0")
        except ValueError:
            override = 0
        return override if override > 0 else jax.device_count()

    def _shard_batch_arrays(self, *arrays):
        """Place microbatch arrays on the MaxText mesh, batch dim sharded.

        Mirrors tunix sft/sharding_utils.shard_input (data_sharding_axis=("fsdp",)).
        Row counts are already padded to the batch axis size (_row_shard), so dim 0
        always divides. Kill switch: TUNIX_SHARD_INPUTS=0 restores the old
        replicated-input behavior. Any failure falls back to the unsharded arrays —
        sharding is an optimization, never a correctness requirement.
        """
        mesh = getattr(self, "_mesh", None)
        if mesh is None or os.environ.get("TUNIX_SHARD_INPUTS", "1") != "1":
            return arrays
        try:
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P

            batch_axes = tuple(
                a for a in ("data", "fsdp") if a in getattr(mesh, "axis_names", ()) and mesh.shape[a] > 1
            )
            if not batch_axes:
                return arrays
            nrows = arrays[0].shape[0]
            row_sharding = NamedSharding(mesh, P(batch_axes))
            if not getattr(self, "_shard_inputs_logged", False):
                self._shard_inputs_logged = True
                logger.info(f"Sharding microbatch inputs over mesh axes {batch_axes} (rows={nrows})")
            return tuple(
                jax.device_put(a, row_sharding)
                if getattr(a, "ndim", 0) >= 1 and a.shape[0] == nrows
                else a
                for a in arrays
            )
        except Exception as e:
            logger.warning(f"input sharding failed ({e}); falling back to replicated inputs")
            return arrays

    def _micro_batch_size(self, total: int) -> int:
        mb = self.config.train_micro_batch_size
        return total if mb <= 0 else max(1, min(mb, total))

    @staticmethod
    def _round_seq_len(seq_len: int, kind: str) -> int:
        if kind == "maxtext":
            if seq_len > _MAXTEXT_SEQ_BLOCK:
                return -(-seq_len // _MAXTEXT_SEQ_BLOCK) * _MAXTEXT_SEQ_BLOCK
            # Splash attention requires kv block sizes (min(block, seq_len)) to
            # be a multiple of 128 (NUM_LANES); power-of-2 buckets like 192
            # crash with "bkv_compute=192 must be a multiple of 128". Round
            # short batches to 128-multiples instead ({128, 256, 384, 512}).
            return max(128, -(-seq_len // 128) * 128)
        return round_up_seq_len(seq_len)

    # ------------------------------------------------------------------ forward / forward_backward

    def _model_pass(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        with_grads: bool,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_model_inputs:
            return {}
        if "ppo_critic" in prepared_batch.all_loss_fns:
            raise ValueError("ppo_critic is only supported by the SkyRL-Train backend")

        # Packed training loads large-arena programs (up to ~57G for the
        # longest bucket). Arena reservation needs CONTIGUOUS memory at the
        # bottom of HBM, and programs loaded earlier in the step (KL scoring
        # shapes, sampler stubs) fragment it — reactive eviction-on-OOM then
        # retries into the same fragmented layout and still fails. Evict
        # everything up front so the fb programs load largest-first into clean
        # memory; the recompile cost is a few minutes per step.
        if with_grads and self.config.train_token_budget > 0:
            jax.clear_caches()

        all_input_ids = [r.prompt_ids for r in render_model_input(prepared_batch.all_model_inputs)]
        n_examples = len(all_input_ids)
        seq_lens = [len(seq) for seq in all_input_ids]
        loss_fn_config = self._build_loss_fn_config(prepared_batch.all_loss_fn_configs)
        loss_fn_types = np.array([LOSS_TYPES[name] for name in prepared_batch.all_loss_fns], dtype=np.int32)

        # Group example indices by model_id (order-preserving within a group).
        groups: dict[str, list[int]] = {}
        for i, model_id in enumerate(prepared_batch.all_model_ids):
            groups.setdefault(model_id, []).append(i)

        token_losses_out: list[np.ndarray | None] = [None] * n_examples
        logprobs_out: list[np.ndarray | None] = [None] * n_examples

        for model_id, indices in groups.items():
            slot = self.models[model_id]
            template = self.templates[slot.template_key]
            pass_fn = template.forward_backward_fn if with_grads else template.forward_fn

            budget = self.config.train_token_budget
            # Uniform-shape mode: TUNIX_UNIFORM_SEQ_LEN>0 forces EVERY microbatch to
            # ONE shape [shard, _ubkt] so the fb compiles a single program instead of
            # ~28+ distinct [rows, len] shapes. Variable shapes both recompile (~50s
            # each) AND leave resident memory that accumulates across microbatches
            # within a step -> free HBM shrinks -> OOM mid-step. Over-pads short
            # datums (some wasted compute) but makes the step fit AND fast. The client
            # must cap trained datums to <= TUNIX_UNIFORM_SEQ_LEN (TTD_TRAIN_MAX_SEQ);
            # set budget == shard * _ubkt so exactly `shard` datums pack per microbatch.
            _uniform = int(os.environ.get("TUNIX_UNIFORM_SEQ_LEN", "0") or "0")
            _ubkt = self._round_seq_len(_uniform, template.kind) if _uniform > 0 else 0
            # Length bucketing (TUNIX_SEQ_BUCKETS="4096,8192,12288,18432"): quantize
            # each datum's padded length to a short ladder instead of one global max.
            # A single uniform shape compiles exactly once but makes a 3k datum cost
            # the same as an 18k one; exact per-datum lengths avoid that waste but
            # explode the XLA shape count (compile storm + memory churn). Buckets keep
            # compiles to a handful -- cached after the first step -- while short
            # datums stay cheap. Takes precedence over nothing: UNIFORM still wins if
            # both are set, so the uniform escape hatch remains available.
            _buckets = sorted(
                {
                    self._round_seq_len(int(b), template.kind)
                    for b in os.environ.get("TUNIX_SEQ_BUCKETS", "").split(",")
                    if b.strip().isdigit() and int(b) > 0
                }
            )

            def _bucket_len(n: int) -> int:
                if _ubkt:
                    return _ubkt
                for b in _buckets:
                    if n <= b:
                        return b
                return self._round_seq_len(n, template.kind)
            if budget > 0:
                # Token-budget packing: longest-first, greedy fill while the
                # padded (rows x seq_len) area stays under the budget. The
                # longest sequences run as singletons; short buckets pack
                # many-up. Row count is quantized to the fsdp multiple that
                # pad_to_fsdp will impose so the budget check matches reality.
                shard = self._row_shard() if template.kind == "maxtext" else 1
                order = sorted(indices, key=lambda i: len(all_input_ids[i]), reverse=True)
                mb_groups: list[list[int]] = []
                cur: list[int] = []
                cur_len = 0
                for i in order:
                    _ln = _bucket_len(len(all_input_ids[i]))
                    cand_len = cur_len if cur else _ln
                    cand_rows = -(-(len(cur) + 1) // shard) * shard
                    # Break the tile on a bucket change as well as on the budget:
                    # `order` is longest-first, so a smaller bucket here means every
                    # remaining datum would otherwise be padded up to this tile's
                    # longer bucket -- exactly the waste bucketing exists to avoid.
                    if cur and (_ln != cur_len or cand_rows * cand_len > budget):
                        mb_groups.append(cur)
                        cur = [i]
                        cur_len = _ln
                    else:
                        cur.append(i)
                        cur_len = cand_len
                if cur:
                    mb_groups.append(cur)
            else:
                micro_bs = self._micro_batch_size(len(indices))
                mb_groups = [indices[s : s + micro_bs] for s in range(0, len(indices), micro_bs)]
            # Per-tile progress. A training fb over a full batch is ~100 microbatch
            # tiles but only logs once, on completion -- which makes a long-running
            # fb indistinguishable from a hang (we chased exactly that). Log one
            # line per tile with its wall time and an ETA for the rest of the pass.
            _mb_total = len(mb_groups)
            _mb_start = time.time()
            for _mb_n, mb_idx in enumerate(mb_groups, 1):
                _mb_t0 = time.time()
                mb_inputs = [all_input_ids[i] for i in mb_idx]
                max_len = _bucket_len(max(len(seq) for seq in mb_inputs))

                input_ids = pad_batch(mb_inputs, max_len, np.int32)
                target_ids = pad_batch([prepared_batch.all_targets[i] for i in mb_idx], max_len, np.int32)
                loss_mask = pad_batch([prepared_batch.all_token_weights[i] for i in mb_idx], max_len, np.float32)
                sampling_logprobs = pad_batch(
                    [prepared_batch.all_sampling_logprobs[i] for i in mb_idx], max_len, np.float32
                )
                # vLLM clamps NaN logprobs to -1e4; those tokens have unknown true
                # probability — exclude them from the loss (for every loss type) to
                # avoid inf importance ratios exp(train_lp - (-1e4)).
                sentinel = sampling_logprobs <= -9.0e3
                loss_mask = np.where(sentinel, 0.0, loss_mask).astype(np.float32)
                # Also neutralize the sentinel value itself: even with zero weight,
                # exp(train_lp - (-1e4)) = inf inside the jitted loss and
                # inf * 0 = nan, which poisons the loss and the gradients.
                sampling_logprobs = np.where(sentinel, 0.0, sampling_logprobs).astype(np.float32)
                advantages = pad_batch([prepared_batch.all_advantages[i] for i in mb_idx], max_len, np.float32)
                positions, attn_mask = self._positions_and_masks(mb_inputs, max_len)
                mb_loss_fn_types = loss_fn_types[mb_idx]
                mb_clip_low = loss_fn_config.clip_low_threshold[mb_idx]
                mb_clip_high = loss_fn_config.clip_high_threshold[mb_idx]

                if template.kind == "maxtext":
                    # MaxText shards the batch dim over the data/fsdp mesh axes:
                    # pad rows to that axis's size. Padded rows have all-zero loss
                    # weights and are excluded from outputs and the
                    # gradient-accumulation count.
                    shard = self._row_shard()
                    (
                        input_ids,
                        target_ids,
                        loss_mask,
                        sampling_logprobs,
                        advantages,
                        positions,
                        attn_mask,
                        mb_loss_fn_types,
                        mb_clip_low,
                        mb_clip_high,
                    ) = (
                        pad_to_fsdp(arr, shard)
                        for arr in (
                            input_ids,
                            target_ids,
                            loss_mask,
                            sampling_logprobs,
                            advantages,
                            positions,
                            attn_mask,
                            mb_loss_fn_types,
                            mb_clip_low,
                            mb_clip_high,
                        )
                    )

                if template.kind == "maxtext":
                    # Shard the batch dim over the mesh BEFORE the jitted pass —
                    # exactly what tunix's trainer does via sharding_utils.shard_input
                    # (data_sharding_axis=("fsdp",)). We used to hand jit raw numpy
                    # arrays, which GSPMD treats as REPLICATED: every chip then
                    # materializes activations for the whole batch. Measured cost at
                    # [4,20480]: 36.8G/chip of remat checkpoints — exactly 4x the
                    # ~9.2G a chip needs for its own row — plus 4x redundant compute
                    # (the ~12% MFU). Sharding dim 0 makes each chip own one row.
                    (
                        input_ids,
                        target_ids,
                        loss_mask,
                        sampling_logprobs,
                        advantages,
                        positions,
                        attn_mask,
                        mb_loss_fn_types,
                        mb_clip_low,
                        mb_clip_high,
                    ) = self._shard_batch_arrays(
                        input_ids,
                        target_ids,
                        loss_mask,
                        sampling_logprobs,
                        advantages,
                        positions,
                        attn_mask,
                        mb_loss_fn_types,
                        mb_clip_low,
                        mb_clip_high,
                    )

                mb_config = LossFnConfig(
                    clip_low_threshold=mb_clip_low,
                    clip_high_threshold=mb_clip_high,
                )

                common_args = (
                    input_ids,
                    positions,
                    attn_mask,
                    target_ids,
                    loss_mask,
                    mb_loss_fn_types,
                    sampling_logprobs,
                    advantages,
                    mb_config,
                )
                with self._jit_timing_context(max_len, mode="train"):
                    if template.kind == "maxtext":
                        # Swap this model's LoRA values into the shared template
                        # and run the module-passing (nnx-lifted) fns.
                        nnx.update(template.model, slot.lora_state)
                        if with_grads:
                            # In-jit donated accumulation (see forward_backward_fn).
                            # slot.accum_grads carries across fb REQUESTS until the
                            # optim step; zeros on the very first tile. NOTE: the
                            # donated buffer is consumed even by a FAILED execution,
                            # so an OOM retry of this lambda raises a loud
                            # deleted-array error instead of silently corrupting
                            # the accumulated gradient -- that is intentional.
                            if slot.accum_grads is None:
                                slot.accum_grads = jax.tree.map(
                                    jnp.zeros_like, nnx.state(template.model, nnx.LoRAParam)
                                )
                            per_token_losses, target_logprobs, slot.accum_grads = self._with_oom_recovery(
                                lambda: pass_fn(template.model, slot.accum_grads, *common_args),
                                "model_pass",
                            )
                            slot.accum_count += len(mb_idx)
                        else:
                            per_token_losses, target_logprobs, _ = self._with_oom_recovery(
                                lambda: pass_fn(template.model, *common_args), "model_pass"
                            )
                    else:
                        per_token_losses, target_logprobs, lora_grads = self._with_oom_recovery(
                            lambda: pass_fn(slot.lora_state, template.rest_state, *common_args),
                            "model_pass",
                        )
                        # Non-maxtext (plain jax.jit) path keeps python-side
                        # accumulation; its models are small enough not to care.
                        if with_grads and lora_grads is not None:
                            if slot.accum_grads is None:
                                slot.accum_grads = lora_grads
                            else:
                                slot.accum_grads = jax.tree.map(jnp.add, slot.accum_grads, lora_grads)
                            slot.accum_count += len(mb_idx)

                per_token_losses, target_logprobs = jax.device_get((per_token_losses, target_logprobs))
                for row, i in enumerate(mb_idx):
                    token_losses_out[i] = per_token_losses[row, : seq_lens[i]].astype(np.float32)
                    logprobs_out[i] = target_logprobs[row, : seq_lens[i]].astype(np.float32)
                # device_get above forces JAX's async dispatch to finish, so this
                # elapsed time is the tile's real cost (not just enqueue time).
                _mb_dt = time.time() - _mb_t0
                _mb_el = time.time() - _mb_start
                try:
                    _ms = jax.devices()[0].memory_stats() or {}
                    _mem = (
                        f" hbm={_ms.get('bytes_in_use', 0) / 1e9:.1f}G"
                        f"/peak={_ms.get('peak_bytes_in_use', 0) / 1e9:.1f}G"
                    )
                except Exception:
                    _mem = ""
                logger.info(
                    f"{'fb' if with_grads else 'fwd'} tile {_mb_n}/{_mb_total} "
                    f"shape=[{input_ids.shape[0]},{max_len}] {_mb_dt:.2f}s "
                    f"(elapsed {_mb_el:.1f}s, eta {(_mb_el / _mb_n) * (_mb_total - _mb_n):.0f}s){_mem}"
                )
                if template.kind == "maxtext":
                    # THE LEAK (measured by the live-array profiler): one
                    # [rows, seq, hidden] bf16 tensor is retained per tile --
                    # +1 array of (4,16384,5120) every tile, 168MB/chip -- until
                    # the allocator evicts the fb program to make room and its
                    # ~38G reload fails (deterministically ~tile 56). The MaxText
                    # pure-NNX decoder mutates module state during forward; under
                    # nnx.jit those mutations PERSIST on template.model, so each
                    # tile's sown intermediates pile up. Drop them eagerly.
                    try:
                        nnx.pop(template.model, nnx.Intermediate)
                    except Exception:
                        pass
                # Leak hunter: device memory grows ~170MB/tile even with donated
                # accumulation (measured 29.4G -> 38.7G over 55 tiles, then the
                # allocator evicts the fb program to make room and its ~38G
                # reload fails). Enumerate live arrays by shape every 10 tiles so
                # the growth names itself instead of being theorized about.
                if _mb_n % 10 == 0 or _mb_n == _mb_total:
                    try:
                        _by: dict = {}
                        for _a in jax.live_arrays():
                            _k = (str(_a.shape), str(_a.dtype))
                            _c, _b = _by.get(_k, (0, 0))
                            _by[_k] = (_c + 1, _b + _a.nbytes)
                        _top = sorted(_by.items(), key=lambda kv: -kv[1][1])[:6]
                        logger.info(
                            f"live_arrays n={sum(v[0] for v in _by.values())} "
                            + "; ".join(f"{k[0]}/{k[1]}x{v[0]}={v[1] / 1e9:.2f}G" for k, v in _top)
                        )
                        if template.kind == "maxtext":
                            # Name the retainer: any module-state leaf shaped like a
                            # per-batch activation ([rows, ...]) is the smoking gun.
                            _st = list(nnx.state(template.model).flat_state())
                            _hits = [
                                ("/".join(map(str, p)), tuple(v.value.shape))
                                for p, v in _st
                                if hasattr(v, "value")
                                and getattr(v.value, "ndim", 0) >= 2
                                and v.value.shape[0] == input_ids.shape[0]
                            ]
                            logger.info(
                                f"module_state leaves={len(_st)} batch-shaped={len(_hits)} "
                                f"sample={_hits[:3]}"
                            )
                    except Exception:
                        pass

        # A training fb over a large batch returns per-token elementwise_loss +
        # logprobs for EVERY datum (e.g. 376 datums x ~10k tokens = millions of
        # floats). Serialized to JSON that is a huge REST body, and the client's
        # result_async STALLS transferring it (diagnosed 2026-07-27: retrieve 200
        # then blocked at low CPU on the body). ttt-discover discards these
        # per-token outputs anyway (train_step training_logprobs_D is unused). With
        # TUNIX_MINIMAL_FB_OUTPUT=1, return empty per-token arrays for the training
        # pass so the retrieve is tiny+fast and the lazy server queue advances to
        # the optim step. Only for with_grads (the forward-only pass may be a real
        # logprob consumer, e.g. KL scoring).
        _minimal_out = with_grads and os.environ.get("TUNIX_MINIMAL_FB_OUTPUT", "0") == "1"
        results: dict[str, types.ForwardBackwardOutput | types.ErrorResponse] = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                if _minimal_out:
                    loss_fn_outputs.append(
                        {
                            "elementwise_loss": {"data": [], "dtype": "float32", "shape": [0]},
                            "logprobs": {"data": [], "dtype": "float32", "shape": [0]},
                        }
                    )
                    continue
                # Belt-and-braces: never let a non-finite value reach the JSON
                # response (json serialization rejects inf/nan).
                token_losses = np.nan_to_num(token_losses_out[i], nan=0.0, posinf=0.0, neginf=0.0)
                token_logprobs = np.nan_to_num(logprobs_out[i], nan=0.0, posinf=0.0, neginf=0.0)
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": token_losses.tolist(),
                            "dtype": "float32",
                            "shape": [token_losses.shape[0]],
                        },
                        "logprobs": {
                            "data": token_logprobs.tolist(),
                            "dtype": "float32",
                            "shape": [token_logprobs.shape[0]],
                        },
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def forward_backward(
        self, prepared_batch: types.PreparedModelPassBatch
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        return self._model_pass(prepared_batch, with_grads=True)

    def forward(
        self, prepared_batch: types.PreparedModelPassBatch
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        return self._model_pass(prepared_batch, with_grads=False)

    # ------------------------------------------------------------------ optim step

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        slot = self.models[model_id]
        template = self.templates[slot.template_key]
        adam = request_data.adam_params

        if slot.accum_count == 0:
            logger.warning(f"No accumulated gradients for model {model_id}; applying step with zero gradients")
            mean_grads = jax.tree.map(jnp.zeros_like, slot.lora_state)
        else:
            count = float(slot.accum_count)
            mean_grads = jax.tree.map(lambda g: g / count, slot.accum_grads)

        hp = slot.optimizer.opt_state.hyperparams
        hp["learning_rate"][...] = adam.learning_rate
        hp["b1"][...] = adam.beta1
        hp["b2"][...] = adam.beta2
        hp["eps"][...] = adam.eps
        hp["weight_decay"][...] = adam.weight_decay

        grad_norm = float(jax.device_get(optax.global_norm(mean_grads)))

        if not np.isfinite(grad_norm):
            # Belt-and-braces: never apply non-finite gradients (they would
            # permanently poison the LoRA weights) and never return a
            # non-finite metric (json serialization rejects inf/nan).
            logger.warning(
                f"Non-finite grad norm ({grad_norm}) for model {model_id}; skipping optimizer update"
            )
            slot.accum_grads = None
            slot.accum_count = 0
            return types.OptimStepOutput(
                metrics={
                    "skyrl.ai/grad_norm": 0.0,
                    "skyrl.ai/learning_rate": adam.learning_rate,
                    "skyrl.ai/skipped_nonfinite_update": 1.0,
                }
            )

        # Swap this model's LoRA values into the shared template, apply the
        # update in place, then snapshot the new state back into the slot.
        nnx.update(template.model, slot.lora_state)
        slot.optimizer.update(template.model, mean_grads)
        slot.lora_state = nnx.state(template.model, nnx.LoRAParam)

        slot.accum_grads = None
        slot.accum_count = 0

        metrics = {
            "skyrl.ai/grad_norm": grad_norm,
            "skyrl.ai/learning_rate": adam.learning_rate,
        }
        logger.info(f"Applied optimizer step for model {model_id}, metrics={metrics}")
        return types.OptimStepOutput(metrics=metrics)

    # ------------------------------------------------------------------ sampling

    def sample(
        self, prepared_batch: types.PreparedSampleBatch
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        if not prepared_batch.all_model_inputs:
            return {}
        # prompt_logprobs batches (compute_logprobs) always run natively on the
        # trainer chips: the vLLM TPU backend cannot serve prompt_logprobs, and
        # the API only routes such requests to the engine. Mixed batches cannot
        # occur in that deployment (plain sampling never reaches the engine
        # when an external client is configured).
        if self.config.inference_backend == "vllm" and not prepared_batch.needs_prompt_logprobs:
            return self._sample_vllm(prepared_batch)
        return self._sample_native(prepared_batch)

    # ----- native sampling -----

    def _resolve_sampler_lora_state(self, model_id: str, checkpoint_id: str, checkpoint_path: str) -> nnx.State:
        """LoRA state snapshot for a (model, checkpoint) pair, from memory or disk."""
        slot = self.models[model_id]
        if checkpoint_id in slot.sampler_lora_states:
            return slot.sampler_lora_states[checkpoint_id]

        assert checkpoint_path, f"No checkpoint path for model {model_id} checkpoint {checkpoint_id}"
        logger.info(f"Loading LoRA sampler checkpoint from {checkpoint_path}")
        payload = self._read_checkpoint_archive(AnyPath(checkpoint_path))
        if payload.get("ephemeral"):
            raise ValueError(
                f"Sampler checkpoint {checkpoint_id} for model {model_id} was ephemeral and is no longer "
                "in memory (engine restarted?). Re-run save_weights_for_sampler."
            )
        lora_state = self._state_from_flat(slot.lora_state, payload["lora_weights"])
        slot.sampler_lora_states[checkpoint_id] = lora_state
        return lora_state

    def _sample_native(
        self, prepared_batch: types.PreparedSampleBatch
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        all_input_ids = [r.prompt_ids for r in render_model_input(prepared_batch.all_model_inputs)]
        n = len(all_input_ids)

        # Group sample indices by (model_id, checkpoint_id): one merged model per group.
        groups: dict[tuple[str, str], list[int]] = {}
        for i in range(n):
            key = (prepared_batch.all_model_ids[i], prepared_batch.all_checkpoint_ids[i])
            groups.setdefault(key, []).append(i)

        sequences_out: list[types.GeneratedSequence | None] = [None] * n
        prompt_logprobs_out: list[list[float] | None] = [None] * n

        for (model_id, checkpoint_id), indices in groups.items():
            if model_id:
                slot = self.models[model_id]
                template = self.templates[slot.template_key]
                lora_state = self._resolve_sampler_lora_state(
                    model_id, checkpoint_id, prepared_batch.all_checkpoint_paths[indices[0]]
                )
            else:
                # Base-model sampling: any template with zero LoRA delta is
                # exactly the base model.
                template, lora_state = self._base_sampling_states()

            max_bs = self.config.sample_max_num_sequences or len(indices)
            for chunk_start in range(0, len(indices), max_bs):
                chunk = indices[chunk_start : chunk_start + max_bs]
                prompts = [all_input_ids[i] for i in chunk]
                params = [prepared_batch.all_sampling_params[i] for i in chunk]
                needs_plp = prepared_batch.needs_prompt_logprobs
                # compute_logprobs requests (prompt_logprobs with max_tokens<=1)
                # only consume the prompt scores; skip the decode entirely
                # rather than running a padded full-batch generation for tokens
                # the client throws away.
                plp_only = needs_plp and all(p.max_tokens is not None and p.max_tokens <= 1 for p in params)

                if template.kind == "maxtext":
                    nnx.update(template.model, lora_state)
                    gen = (
                        self._plp_dummy_sequences(len(chunk))
                        if plp_only
                        else self._generate_eager(template.model, prompts, params)
                    )
                    plps = (
                        self._prompt_logprobs(self._module_logps_fn(template.model), prompts)
                        if needs_plp
                        else [None] * len(chunk)
                    )
                else:
                    gen = (
                        self._plp_dummy_sequences(len(chunk))
                        if plp_only
                        else self._generate(template.graphdef, lora_state, template.rest_state, prompts, params)
                    )
                    plps = (
                        self._prompt_logprobs(
                            self._pure_logps_fn(template.graphdef, lora_state, template.rest_state), prompts
                        )
                        if needs_plp
                        else [None] * len(chunk)
                    )
                for row, i in enumerate(chunk):
                    sequences_out[i] = gen[row]
                    prompt_logprobs_out[i] = plps[row]

        results: dict[str, types.SampleOutput | types.ErrorResponse] = {}
        for request_id, _, start_idx, end_idx, plp_requested in prepared_batch.request_batch_slices:
            results[request_id] = types.SampleOutput(
                sequences=[sequences_out[i] for i in range(start_idx, end_idx)],
                prompt_logprobs=prompt_logprobs_out[start_idx] if plp_requested else None,
            )
        return results

    def _base_sampling_states(self):
        """(template, zero-lora state) for base-model sampling."""
        if not self.templates:
            # No LoRA model created yet: build a rank-1 throwaway template
            # (zero delta == exact base model outputs).
            self._get_template(types.LoraConfig(rank=1, alpha=1.0, seed=0))
        template = next(iter(self.templates.values()))
        zero_lora = jax.tree.map(jnp.zeros_like, template.lora_shape)
        return template, zero_lora

    def _get_generate_fn(self, graphdef, batch: int, total_len: int, steps: int) -> Callable:
        """Build (and cache per shape) the jitted generation scan."""

        def generate_scan(lora_state, rest_state, tokens, filled, seeds, temps, top_ks, top_ps):
            causal = jnp.tril(jnp.ones((total_len, total_len), dtype=bool))

            def step(carry, step_idx):
                tokens, filled = carry
                # Merge inside the scan body: creating the module at the outer
                # trace level and calling it in the scan trace trips flax's
                # TraceContextError (module mutation across trace levels).
                model = nnx.merge(graphdef, lora_state, rest_state)
                pad_mask = jnp.arange(total_len)[None, :] < filled[:, None]
                positions = jnp.maximum(jnp.cumsum(pad_mask, axis=-1) - 1, 0)
                attn_mask = causal[None, :, :] & pad_mask[:, None, :]
                hidden, _ = model(tokens, positions, None, attn_mask, skip_lm_head=True)
                logits = model.compute_final_logits(hidden)
                last = jnp.take_along_axis(logits, (filled - 1)[:, None, None], axis=1)[:, 0, :]
                keys = _step_keys(seeds, step_idx)
                next_tokens, logps = _sample_token_rows(last, temps, top_ks, top_ps, keys)
                tokens = jax.vmap(lambda row, pos, tok: row.at[pos].set(tok))(tokens, filled, next_tokens)
                return (tokens, filled + 1), (next_tokens, logps)

            (_, _), (gen_tokens, gen_logps) = jax.lax.scan(step, (tokens, filled), jnp.arange(steps))
            return gen_tokens.T, gen_logps.T  # [B, steps]

        if self.config.enforce_eager:
            return generate_scan
        key = (id(graphdef), batch, total_len, steps)
        if key not in self._generate_fn_cache:
            self._generate_fn_cache[key] = jax.jit(generate_scan)
        return self._generate_fn_cache[key]

    def _generate(
        self,
        graphdef,
        lora_state,
        rest_state,
        prompts: list[list[int]],
        params: list[types.SamplingParams],
    ) -> list[types.GeneratedSequence]:
        batch = len(prompts)
        max_prompt_len = round_up_seq_len(max(len(p) for p in prompts))
        max_new = max(p.max_tokens for p in params)
        steps = max_new
        total_len = round_up_seq_len(max_prompt_len + max_new)

        tokens = pad_batch(prompts, total_len, np.int32)
        filled = np.array([len(p) for p in prompts], dtype=np.int32)
        seeds = np.array([p.seed for p in params], dtype=np.uint32)
        temps = np.array([p.temperature for p in params], dtype=np.float32)
        top_ks = np.array([p.top_k for p in params], dtype=np.int32)
        top_ps = np.array([p.top_p for p in params], dtype=np.float32)

        gen_fn = self._get_generate_fn(graphdef, batch, total_len, steps)
        with self._jit_timing_context(total_len, mode="sample"):
            gen_tokens, gen_logps = jax.device_get(
                gen_fn(lora_state, rest_state, jnp.asarray(tokens), jnp.asarray(filled), seeds, temps, top_ks, top_ps)
            )
        return self._assemble_sequences(gen_tokens, gen_logps, params)

    def _generate_eager(
        self,
        model: nnx.Module,
        prompts: list[list[int]],
        params: list[types.SamplingParams],
    ) -> list[types.GeneratedSequence]:
        """Eager (non-jitted) generation loop, used for MaxText templates.

        The MaxText decoder self-mutates state during forward, which breaks
        raw jit+scan; eager execution is fine for CPU tests and small runs —
        TPU-scale sampling goes through vLLM.
        """
        batch = len(prompts)
        max_new = max(p.max_tokens for p in params)
        total_len = max(len(p) for p in prompts) + max_new

        tokens = jnp.asarray(pad_batch(prompts, total_len, np.int32))
        filled = jnp.asarray(np.array([len(p) for p in prompts], dtype=np.int32))
        seeds = jnp.asarray(np.array([p.seed for p in params], dtype=np.uint32))
        temps = jnp.asarray(np.array([p.temperature for p in params], dtype=np.float32))
        top_ks = jnp.asarray(np.array([p.top_k for p in params], dtype=np.int32))
        top_ps = jnp.asarray(np.array([p.top_p for p in params], dtype=np.float32))
        causal = jnp.tril(jnp.ones((total_len, total_len), dtype=bool))

        step_tokens, step_logps = [], []
        for step_idx in range(max_new):
            pad_mask = jnp.arange(total_len)[None, :] < filled[:, None]
            positions = jnp.maximum(jnp.cumsum(pad_mask, axis=-1) - 1, 0)
            attn_mask = causal[None, :, :] & pad_mask[:, None, :]
            hidden, _ = model(tokens, positions, None, attn_mask, skip_lm_head=True)
            logits = model.compute_final_logits(hidden)
            last = jnp.take_along_axis(logits, (filled - 1)[:, None, None], axis=1)[:, 0, :]
            keys = _step_keys(seeds, step_idx)
            next_tokens, logps = _sample_token_rows(last, temps, top_ks, top_ps, keys)
            tokens = jax.vmap(lambda row, pos, tok: row.at[pos].set(tok))(tokens, filled, next_tokens)
            filled = filled + 1
            step_tokens.append(next_tokens)
            step_logps.append(logps)

        gen_tokens = np.stack(jax.device_get(step_tokens), axis=1)  # [B, steps]
        gen_logps = np.stack(jax.device_get(step_logps), axis=1)
        return self._assemble_sequences(gen_tokens, gen_logps, params)

    def _assemble_sequences(
        self, gen_tokens: np.ndarray, gen_logps: np.ndarray, params: list[types.SamplingParams]
    ) -> list[types.GeneratedSequence]:
        sequences = []
        for row, p in enumerate(params):
            row_tokens = [int(t) for t in gen_tokens[row, : p.max_tokens]]
            row_logps = [float(x) for x in gen_logps[row, : p.max_tokens]]
            row_tokens, row_logps, stop_reason = self._apply_stops(row_tokens, row_logps, p)
            sequences.append(
                types.GeneratedSequence(stop_reason=stop_reason, tokens=row_tokens, logprobs=row_logps)
            )
        return sequences

    def _apply_stops(
        self, tokens: list[int], logprobs: list[float], params: types.SamplingParams
    ) -> tuple[list[int], list[float], str]:
        """Post-hoc stop-token / stop-string truncation (inclusive of the stop match)."""
        stop_reason = "length"
        end = len(tokens)

        if params.stop_tokens:
            stop_set = set(params.stop_tokens)
            for i, tok in enumerate(tokens):
                if tok in stop_set:
                    end = i + 1
                    stop_reason = "stop"
                    break

        if params.stop_strings:
            text = ""
            for i in range(end):
                text = self.tokenizer.decode(tokens[: i + 1])
                if any(s in text for s in params.stop_strings):
                    if i + 1 < end or stop_reason == "length":
                        end = i + 1
                        stop_reason = "stop"
                    break

        return tokens[:end], logprobs[:end], stop_reason

    def _pure_logps_fn(self, graphdef, lora_state, rest_state) -> Callable:
        """Target-gathered log-softmax forward for native templates (jitted).

        Returns fn(input_ids, positions, attn_mask, target_ids) -> [B, L]
        logprobs of target_ids; the gather lives inside the jit so the full
        [B, L, V] log-softmax never escapes the program.
        """

        def fwd(lora_state, rest_state, input_ids, positions, attn_mask, target_ids):
            model = nnx.merge(graphdef, lora_state, rest_state)
            hidden, _ = model(input_ids, positions, None, attn_mask, skip_lm_head=True)
            logits = model.compute_final_logits(hidden)
            # logsumexp identity (see _loss_from_logits): avoid materializing
            # the full [B, L, V] log_softmax; KL forward-scoring path.
            tgt = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)[..., 0]
            return tgt - jax.nn.logsumexp(logits, axis=-1)

        jitted = fwd if self.config.enforce_eager else jax.jit(fwd)
        return lambda input_ids, positions, attn_mask, target_ids: jitted(
            lora_state, rest_state, input_ids, positions, attn_mask, target_ids
        )

    def _module_logps_fn(self, model: nnx.Module) -> Callable:
        """Target-gathered log-softmax forward for MaxText templates.

        Must be nnx.jit-lifted, not eager: the qwen3.5 delta-rule attention
        shard_maps the batch axis over the fsdp mesh, and only the jitted path
        handles batch sizes that don't divide the mesh (the eager call fails
        with 'axis sizes not evenly divisible'). Same [B, L] contract as
        _pure_logps_fn.
        """

        flce_tile_size = self.config.flce_tile_size
        flce = TunixBackend._flce_target_logprobs

        def fwd(model, input_ids, positions, attention_mask, target_ids):
            hidden, _ = model(input_ids, positions, None, attention_mask, skip_lm_head=True)
            if flce_tile_size > 0:
                # FLCE: model returned hidden [B,T,H] (num_vocab_tiling skips the
                # output head). Project token tiles to get target log-probs
                # without forming [B,L,V]. Same KL forward-scoring contract.
                # Forward-only (no backward): plain scan rolls, so skip the
                # checkpoint that would otherwise materialize all tiles here.
                return flce(model.logits_from_hidden, hidden, target_ids, flce_tile_size, use_checkpoint=False)
            logits = model.compute_final_logits(hidden)
            # logsumexp identity (see _loss_from_logits): avoid materializing
            # the full [B, L, V] log_softmax; KL forward-scoring path.
            tgt = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)[..., 0]
            return tgt - jax.nn.logsumexp(logits, axis=-1)

        jitted = fwd if self.config.enforce_eager else nnx.jit(fwd)
        return lambda input_ids, positions, attn_mask, target_ids: jitted(
            model, input_ids, positions, attn_mask, target_ids
        )

    def _log_hbm(self, tag: str) -> None:
        """One-line per-device HBM telemetry (in_use/peak), best-effort."""
        try:
            s = jax.devices()[0].memory_stats() or {}
            logger.info(
                f"HBM[{tag}] in_use={s.get('bytes_in_use', 0) / 1e9:.2f}G "
                f"peak={s.get('peak_bytes_in_use', 0) / 1e9:.2f}G "
                f"limit={(s.get('bytes_limit') or 0) / 1e9:.2f}G"
            )
        except Exception:
            pass

    def _with_oom_recovery(self, fn, desc: str):
        """Run fn; on RESOURCE_EXHAUSTED, retry GENTLY first, destructively last.

        The old behavior -- jax.clear_caches() then retry once -- converted a
        single transient OOM into an unrecoverable spiral: clearing evicts the
        loaded fb program, whose ~37G arena then cannot be re-reserved into the
        now-fragmented HBM, so the one retry dies too, the request fails, the
        client resubmits, and every subsequent attempt recompiles into the same
        wall (observed as 20+ identical E0101 loads across four configs).

        Order matters: (1) plain retry after gc -- the usual cause is a
        transient data-buffer spike and the program is STILL LOADED, so a bare
        retry is cheap and safe; (2) only if that fails, clear caches and pay
        the recompile as the last resort.
        """
        import gc

        try:
            return fn()
        except Exception as e:
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            self._log_hbm(f"{desc}/oom1")
            logger.warning(f"{desc}: RESOURCE_EXHAUSTED — gc + gentle retry (program kept loaded)")
            gc.collect()
            try:
                return fn()
            except Exception as e2:
                if "RESOURCE_EXHAUSTED" not in str(e2):
                    raise
                self._log_hbm(f"{desc}/oom2")
                logger.warning(f"{desc}: still exhausted — jax.clear_caches() and final retry")
                jax.clear_caches()
                gc.collect()
                return fn()

    def _prompt_logprobs(
        self, logps_fn: Callable, prompts: list[list[int]], sub_batch: int | None = None
    ) -> list[list[float]]:
        """Per-token logprobs of the prompt tokens themselves (first token gets 0.0).

        The full [B, L, V] log-softmax at long context is tens of GB, so
        logps_fn gathers it down to [B, L] in-program, and prompts are
        processed in small sub-batches to bound the transient logits tensor
        (KL-scoring batches arrive hundreds of sequences at a time). Rows are
        padded to a device-count multiple exactly like _model_pass: MaxText
        shard_maps the batch dim over the fsdp mesh axis and rejects
        non-divisible batches.
        """
        shard = max(1, jax.device_count())
        if sub_batch is None:
            sub_batch = shard
        # Few program shapes (coarse 8192-buckets of the ACTUAL length), evict
        # first so each scoring arena loads into clean memory. Pinning the full
        # maxtext_max_target_length (32768) needed a ~57.5G contiguous arena vs
        # ~56.9G free — off by ~670MB — while a forward at the real lengths
        # (datums are pre-filtered to <=TTD_TRAIN_MAX_SEQ, ~24576) fits with
        # room to spare. Cap defensively at the fb ceiling too.
        jax.clear_caches()
        score_cap = min(
            int(self.config.maxtext_max_target_length or 32768),
            int(os.environ.get("TTD_TRAIN_MAX_SEQ", "24576") or 24576),
        )
        # SKYRL_SCORE_FIXED_LEN: pin scoring to ONE bucket so exactly one
        # scorer program ever exists. The 8192-bucket ladder compiles a NEW
        # program (each pinning its own arena) as sequences grow across a run;
        # on memory-edge cells (KL x Erdos) the accumulated scorer arenas
        # eventually leave the fb unloadable -- observed as ENGINE-SICK trips
        # arriving one step later per mitigation. Wastes pad compute on short
        # sequences; only set it where the fb/scorer budget is tight.
        _fixed = int(os.environ.get("SKYRL_SCORE_FIXED_LEN", "0") or 0)
        out: list[list[float]] = []
        for start in range(0, len(prompts), sub_batch):
            chunk = prompts[start : start + sub_batch]
            raw_len = max(len(p) for p in chunk)
            if _fixed > 0:
                max_len = min(score_cap, _fixed)
            else:
                max_len = min(score_cap, max(8192, -(-raw_len // 8192) * 8192))
            input_ids = pad_batch(chunk, max_len, np.int32)
            positions, attn_mask = self._positions_and_masks(chunk, max_len)

            # targets[t] = input_ids[t+1]: the wrapped final column is never
            # read (loop stops at len-1). Padded rows are dropped on output.
            targets = np.roll(input_ids, -1, axis=1)
            input_ids, positions, attn_mask, targets = (
                pad_to_fsdp(arr, shard) for arr in (input_ids, positions, attn_mask, targets)
            )
            gathered = self._with_oom_recovery(
                lambda: jax.device_get(logps_fn(input_ids, positions, attn_mask, targets)),
                "prompt_logprobs",
            )
            for row, prompt in enumerate(chunk):
                row_out = [0.0]
                for t in range(1, len(prompt)):
                    row_out.append(float(gathered[row, t - 1]))
                out.append(row_out)
        return out

    @staticmethod
    def _plp_dummy_sequences(n: int) -> list["types.GeneratedSequence"]:
        """Empty sequences for compute_logprobs-style requests (no decode ran)."""
        return [
            types.GeneratedSequence(tokens=[], logprobs=[], stop_reason="length") for _ in range(n)
        ]

    # ----- vLLM sampling (mirrors JaxBackend; the client is backend-agnostic) -----

    def _sample_vllm(
        self, prepared_batch: types.PreparedSampleBatch
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        assert self.vllm_client is not None

        all_input_ids = [r.prompt_ids for r in render_model_input(prepared_batch.all_model_inputs)]
        model_names: list[str] = []
        for model_id, checkpoint_id, checkpoint_path in zip(
            prepared_batch.all_model_ids,
            prepared_batch.all_checkpoint_ids,
            prepared_batch.all_checkpoint_paths,
        ):
            if model_id:
                if not checkpoint_id or not checkpoint_path:
                    raise ValueError(f"LoRA sample for model {model_id} is missing checkpoint information")
                model_names.append(
                    self.vllm_client.ensure_lora_loaded(model_id, AnyPath(checkpoint_path), checkpoint_id=checkpoint_id)
                )
            else:
                model_names.append(self.vllm_client.model_name)

        if self.config.vllm_group_completions:
            groups: list[GroupedCompletion] = []
            group_meta: list[tuple[str, bool]] = []
            for request_id, _, start_idx, end_idx, plp_requested in prepared_batch.request_batch_slices:
                groups.append(
                    GroupedCompletion(
                        prompt_ids=all_input_ids[start_idx],
                        sampling_params=prepared_batch.all_sampling_params[start_idx],
                        model_name=model_names[start_idx],
                        n=end_idx - start_idx,
                        session_id=prepared_batch.all_session_ids[start_idx],
                    )
                )
                group_meta.append((request_id, plp_requested))
            group_sequences, group_plps = self.vllm_client.sample_groups(
                groups, prompt_logprobs=prepared_batch.needs_prompt_logprobs
            )
            return {
                request_id: types.SampleOutput(
                    sequences=sequences,
                    prompt_logprobs=plps if plp_requested else None,
                )
                for (request_id, plp_requested), sequences, plps in zip(group_meta, group_sequences, group_plps)
            }

        all_sequences, all_plps = self.vllm_client.sample_many(
            all_input_ids,
            prepared_batch.all_sampling_params,
            model_names,
            prepared_batch.all_session_ids,
            prompt_logprobs=prepared_batch.needs_prompt_logprobs,
        )
        results: dict[str, types.SampleOutput | types.ErrorResponse] = {}
        for request_id, _, start_idx, end_idx, plp_requested in prepared_batch.request_batch_slices:
            results[request_id] = types.SampleOutput(
                sequences=[all_sequences[i] for i in range(start_idx, end_idx)],
                prompt_logprobs=all_plps[start_idx] if plp_requested and all_plps else None,
            )
        return results

    # ------------------------------------------------------------------ checkpointing

    @staticmethod
    def _flat_numpy(state) -> dict[str, np.ndarray]:
        out = {}
        for k, v in _keystr_map(state).items():
            arr = np.asarray(jax.device_get(v))
            if arr.dtype.kind == "V":
                # npz cannot represent extension dtypes: bfloat16 saves as raw
                # void bytes that np.load returns untyped. Store float32
                # instead (exact for bf16); the load side casts back to the
                # target leaf dtype.
                arr = arr.astype(np.float32)
            out[k] = arr
        return out

    @staticmethod
    def _state_from_flat(target_state, flat: dict[str, np.ndarray]):
        """Rebuild a pytree with target structure from a {keystr: array} dict."""
        target_map = _keystr_map(target_state)
        missing = set(target_map) - set(flat)
        extra = set(flat) - set(target_map)
        if missing or extra:
            raise ValueError(f"Checkpoint state mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
        leaves, treedef = jax.tree.flatten_with_path(target_state)

        def restore(path, leaf):
            arr = flat[jax.tree_util.keystr(path)]
            dtype = np.dtype(leaf.dtype)
            if arr.dtype.kind == "V" and arr.dtype.itemsize == dtype.itemsize:
                # Legacy checkpoint written before the float32 conversion in
                # _flat_numpy: the void bytes are the target dtype verbatim.
                arr = arr.view(dtype)
            return jnp.asarray(arr, dtype=dtype)

        return jax.tree.unflatten(treedef, [restore(p, leaf) for p, leaf in leaves])

    @staticmethod
    def _write_npz(path: Path, flat: dict[str, np.ndarray]) -> None:
        np.savez(path, **flat)

    @staticmethod
    def _read_npz(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        slot = self.models[model_id]
        with pack_and_upload(AnyPath(output_path)) as tmp:
            self._write_npz(tmp / "lora_weights.npz", self._flat_numpy(slot.lora_state))
            self._write_npz(tmp / "optimizer_state.npz", self._flat_numpy(nnx.state(slot.optimizer)))
            (tmp / _CHECKPOINT_META_FILE).write_text(
                json.dumps({"lora_config": slot.lora_config.model_dump(), "format": "tunix_backend_v1"})
            )
        logger.info(f"Saved training checkpoint to {output_path}")

    def _read_checkpoint_archive(self, checkpoint_path: AnyPath) -> dict[str, Any]:
        with download_and_unpack(checkpoint_path) as tmp:
            payload: dict[str, Any] = {}
            marker = tmp / _EPHEMERAL_MARKER_FILE
            if marker.exists():
                payload["ephemeral"] = True
                payload.update(json.loads(marker.read_text()))
                return payload
            meta = tmp / _CHECKPOINT_META_FILE
            if meta.exists():
                payload.update(json.loads(meta.read_text()))
            lora_file = tmp / "lora_weights.npz"
            if lora_file.exists():
                payload["lora_weights"] = self._read_npz(lora_file)
            opt_file = tmp / "optimizer_state.npz"
            if opt_file.exists():
                payload["optimizer_state"] = self._read_npz(opt_file)
            return payload

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        slot = self.models[model_id]
        payload = self._read_checkpoint_archive(AnyPath(checkpoint_path))
        if "lora_weights" not in payload:
            raise FileNotFoundError(f"Training checkpoint not found or incomplete at {checkpoint_path}")

        ckpt_rank = payload.get("lora_config", {}).get("rank")
        if ckpt_rank is not None and ckpt_rank != slot.lora_config.rank:
            raise ValueError(
                f"Rank mismatch: checkpoint has rank {ckpt_rank}, model configured with rank {slot.lora_config.rank}"
            )

        slot.lora_state = self._state_from_flat(slot.lora_state, payload["lora_weights"])
        if "optimizer_state" in payload:
            opt_state = self._state_from_flat(nnx.state(slot.optimizer), payload["optimizer_state"])
            nnx.update(slot.optimizer, opt_state)
        slot.accum_grads = None
        slot.accum_count = 0
        logger.info(f"Loaded training checkpoint from {checkpoint_path}")

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        slot = self.models[model_id]
        output_path = AnyPath(output_path)
        checkpoint_id = output_path.name.removesuffix(".tar.gz")

        # Snapshot the current LoRA state in memory for the native sampler hot path.
        slot.sampler_lora_states[checkpoint_id] = slot.lora_state
        slot.loaded_sampler_checkpoint_id = checkpoint_id

        if self.vllm_client is not None:
            if self.config.vllm_lora_upload_endpoint:
                # Push the adapter over HTTP straight to the vLLM server's
                # local disk — no shared filesystem in the hot path.
                self.vllm_client.push_adapter(model_id, checkpoint_id, self._peft_adapter_tar_bytes(slot))
            else:
                # Fallback: write the PEFT dir into the shared lora_base_dir
                # under the exact adapter name ensure_lora_loaded derives; its
                # extractor early-returns on existing dirs, skipping the
                # tar/extract round-trip through shared storage.
                self._publish_peft_adapter(slot, model_id, checkpoint_id)

        if persist:
            with pack_and_upload(output_path) as tmp:
                self._write_npz(tmp / "lora_weights.npz", self._flat_numpy(slot.lora_state))
                (tmp / _CHECKPOINT_META_FILE).write_text(
                    json.dumps({"lora_config": slot.lora_config.model_dump(), "format": "tunix_backend_v1"})
                )
        else:
            with pack_and_upload(output_path) as tmp:
                (tmp / _EPHEMERAL_MARKER_FILE).write_text(
                    json.dumps({"checkpoint_id": checkpoint_id, "model_id": model_id})
                )
        logger.info(f"Saved sampler checkpoint for model {model_id} to {output_path} (persist={persist})")

        if self.vllm_client is not None and not self.config.vllm_lora_upload_endpoint:
            # push_adapter already loaded the adapter on the push path.
            self.vllm_client.ensure_lora_loaded(model_id, output_path, checkpoint_id=checkpoint_id)

    def _peft_adapter_tar_bytes(self, slot: ModelSlot) -> bytes:
        """Serialize the HF-PEFT adapter export as an uncompressed in-memory tar."""
        import io
        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._export_peft_adapter(slot, Path(tmp))
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:") as tar:
                for f in sorted(Path(tmp).iterdir()):
                    tar.add(f, arcname=f.name)
            return buf.getvalue()

    def _publish_peft_adapter(self, slot: ModelSlot, model_id: str, checkpoint_id: str) -> None:
        """Atomically place the exported PEFT dir at <lora_base_dir>/<adapter-name>."""
        import shutil
        from skyrl.backends.vllm_sampling import _sanitize_lora_name

        target = Path(self.config.vllm_lora_base_dir) / _sanitize_lora_name(model_id, checkpoint_id)
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        self._export_peft_adapter(slot, staging)
        import os

        os.replace(staging, target)
        self._prune_published_adapters(model_id, keep=8)

    def _prune_published_adapters(self, model_id: str, keep: int) -> None:
        """Delete old published adapter dirs beyond the newest ``keep`` for a model.

        Matches vLLM's --max-loras LRU window: anything older is already
        evicted server-side and only wastes bucket storage.
        """
        import shutil
        from skyrl.backends.vllm_sampling import _sanitize_lora_name

        base = Path(self.config.vllm_lora_base_dir)
        prefix = _sanitize_lora_name(model_id, "")
        dirs = [p for p in base.glob(f"{prefix}*") if p.is_dir() and not p.name.endswith(".staging")]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in dirs[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
            logger.info(f"Pruned stale published adapter {stale.name}")

    # ------------------------------------------------------------------ HF-PEFT export

    def _export_peft_adapter(self, slot: ModelSlot, out_dir: Path) -> None:
        """Write adapter_model.safetensors + adapter_config.json (HF-PEFT layout).

        gpt-oss additionally writes moe_lora.safetensors + moe_lora.json: the
        raw expert/router LoRA factors, which standard PEFT cannot express —
        the vLLM server's merge-on-load path consumes them.
        """
        import safetensors.numpy as st_numpy

        if self.config.model_source == "maxtext":
            if self._is_gptoss_lora(slot):
                tensors, moe_tensors, moe_meta = self._peft_tensors_gptoss(slot)
                if moe_tensors:
                    st_numpy.save_file(moe_tensors, str(out_dir / "moe_lora.safetensors"))
                    (out_dir / "moe_lora.json").write_text(json.dumps(moe_meta, indent=2))
            else:
                tensors = self._peft_tensors_maxtext(slot)
        else:
            tensors = self._peft_tensors_native(slot)

        st_numpy.save_file(tensors, str(out_dir / "adapter_model.safetensors"))
        # ...self_attn.q_proj.lora_A.weight -> "q_proj"
        target_modules = sorted({k.rsplit(".", 2)[0].rsplit(".", 1)[-1] for k in tensors})
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": self.base_model,
            "r": slot.lora_config.rank,
            "lora_alpha": slot.lora_config.alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
        }
        (out_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    def _peft_tensors_native(self, slot: ModelSlot) -> dict[str, np.ndarray]:
        tensors: dict[str, np.ndarray] = {}
        for path, leaf in jax.tree.flatten_with_path(slot.lora_state)[0]:
            keystr = jax.tree_util.keystr(path)
            hf_module = self._qwix_path_to_hf_module(keystr)
            if hf_module is None:
                logger.warning("Skipping unmapped LoRA param %s in PEFT export", keystr)
                continue
            arr = np.asarray(jax.device_get(leaf), dtype=np.float32)
            if "_lora_a" in keystr:
                # qwix lora_a: (in, r) -> PEFT lora_A.weight: (r, in)
                tensors[f"base_model.model.{hf_module}.lora_A.weight"] = np.ascontiguousarray(arr.T)
            else:
                # qwix lora_b: (r, ...) -> PEFT lora_B.weight: (out, r)
                out_matrix = arr.reshape(arr.shape[0], -1)
                tensors[f"base_model.model.{hf_module}.lora_B.weight"] = np.ascontiguousarray(out_matrix.T)
        return tensors

    @staticmethod
    def _is_gptoss_lora(slot: ModelSlot) -> bool:
        return any(
            "GptOssMlp" in jax.tree_util.keystr(p) or "GptOssAttention" in jax.tree_util.keystr(p)
            for p, _ in jax.tree.flatten_with_path(slot.lora_state)[0]
        )

    def _peft_tensors_gptoss(
        self, slot: ModelSlot
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
        """Split gpt-oss LoRA state into (attention PEFT tensors, MoE factors, meta).

        gpt-oss decoders scan layers in ``layer_cycle_interval`` groups
        (``layers_{g}``: sliding/full attention alternation), stacked over
        axis 1; global HF layer index = stack_idx * n_groups + g (matches
        MaxText's convert_gpt_oss_ckpt: ``divmod(layer_idx, interval)``).

        MoE factor shapes (rank r, E experts, d model dim, f ff dim):
          wi_0/wi_1: lora_a (d, L, r) shared over experts; lora_b (r, L, E, f)
          wo:        lora_a (E, L, f, r) per-expert;       lora_b (r, L, d)
          router:    lora_a (d, L, r);                     lora_b (r, L, E)
        Per-expert delta = scale * A @ B_e (see meta["contraction"]).
        """
        import re

        attn: dict[str, np.ndarray] = {}
        moe: dict[str, np.ndarray] = {}
        entries = []
        groups: set[int] = set()
        for path, leaf in jax.tree.flatten_with_path(slot.lora_state)[0]:
            keystr = jax.tree_util.keystr(path)
            names = [a or b for a, b in re.findall(r"\['([A-Za-z_0-9]+)'\]|\[(\d+)\]", keystr)]
            group_name = next((n for n in names if re.fullmatch(r"layers_[0-9]+", n)), None)
            if group_name is None:
                logger.warning("Skipping gpt-oss LoRA param without layer group: %s", keystr)
                continue
            group = int(group_name.split("_")[1])
            groups.add(group)
            arr = np.asarray(jax.device_get(leaf), dtype=np.float32)
            entries.append((group, names, arr, keystr))
        n_groups = max(groups) + 1 if groups else 1

        for group, names, arr, keystr in entries:
            if "GptOssAttention" in names:
                proj = names[names.index("GptOssAttention") + 1]
                hf = {"query": "q_proj", "key": "k_proj", "value": "v_proj", "out": "o_proj"}.get(proj)
                leafname = names[-1]
                if hf is None or arr.ndim != 3 or "lora" not in leafname:
                    logger.warning("Skipping unmapped gpt-oss attention LoRA param %s", keystr)
                    continue
                is_a = leafname.endswith("lora_a")
                for j in range(arr.shape[1]):
                    gl = j * n_groups + group
                    per = arr[:, j, :]  # lora_a: (in, r); lora_b: (r, out)
                    name = f"base_model.model.model.layers.{gl}.self_attn.{hf}"
                    suffix = "lora_A.weight" if is_a else "lora_B.weight"
                    attn[f"{name}.{suffix}"] = np.ascontiguousarray(per.T)
            elif "GptOssMlp" in names:
                after = names[names.index("GptOssMlp") + 1 :]
                if after[0] == "gate":
                    comp, leafname = "router", after[-1]
                else:
                    m = re.fullmatch(r"(wi_0|wi_1|wo)_(lora_[ab])", after[0])
                    if m is None:
                        logger.warning("Skipping unmapped gpt-oss MoE LoRA param %s", keystr)
                        continue
                    comp, leafname = m.group(1), m.group(2)
                if "lora" not in leafname:
                    logger.warning("Skipping non-LoRA gpt-oss MoE param %s", keystr)
                    continue
                is_a = leafname.endswith("a")
                for j in range(arr.shape[1]):  # stack axis is 1 for every factor
                    gl = j * n_groups + group
                    per = np.ascontiguousarray(np.take(arr, j, axis=1))
                    moe[f"layers.{gl}.{comp}.{'lora_a' if is_a else 'lora_b'}"] = per
            else:
                logger.warning("Skipping unrecognized gpt-oss LoRA param %s", keystr)

        meta = {
            "format": "gptoss-moe-lora/v1",
            "rank": slot.lora_config.rank,
            "alpha": slot.lora_config.alpha,
            "scale": slot.lora_config.alpha / slot.lora_config.rank,
            "num_layer_groups": n_groups,
            "contraction": {
                "wi_0": "delta[e,d,f] = scale * sum_r A[d,r] * B[r,e,f]  (gate half)",
                "wi_1": "delta[e,d,f] = scale * sum_r A[d,r] * B[r,e,f]  (up half)",
                "wo": "delta[e,f,d] = scale * sum_r A[e,f,r] * B[r,d]",
                "router": "delta[d,e] = scale * sum_r A[d,r] * B[r,e]",
            },
        }
        return attn, moe, meta

    def _peft_tensors_maxtext(self, slot: ModelSlot) -> dict[str, np.ndarray]:
        """MaxText LoRA params are stacked over the scanned layer axis (axis 1).

        Two layer layouts are handled, distinguished per-path by
        ``_maxtext_path_to_hf``:
          - homogeneous scan (qwen-style): path component ``layers``; the HF
            layer index is the stack position directly.
          - grouped scan (gemma4-style): component ``layers_<k>`` inside the
            scanned block (k = position in the repeating attention pattern);
            the HF layer index is ``stack_pos * pattern_len + k`` — matching
            MaxText's own converter (hf_indices = range(k, n, pattern_len)).
        """
        entries: list[tuple[str, str, str, bool, int | None, np.ndarray]] = []
        inner_ids: set[int] = set()
        import re as _re

        # gemma4 HF checkpoints are ConditionalGeneration composites too: the
        # decoder lives at model.language_model.layers and vLLM's gemma4_mm
        # hf_to_vllm_mapper maps "model.language_model." ->
        # "language_model.model." exactly like qwen3.5, so gemma4 adapters
        # must also carry the composite-model HF names.
        composite_model = self._maxtext_model_name().startswith("gemma4")

        for path, leaf in jax.tree.flatten_with_path(slot.lora_state)[0]:
            keystr = jax.tree_util.keystr(path)
            parsed = self._maxtext_path_to_hf(keystr)
            if parsed is None:
                logger.warning("Skipping unmapped MaxText LoRA param %s in PEFT export", keystr)
                continue
            hf_block, hf_proj, is_a, inner = parsed
            arr = np.asarray(jax.device_get(leaf), dtype=np.float32)
            if arr.ndim != 3:
                logger.warning("Unexpected MaxText LoRA shape %s for %s; skipping", arr.shape, keystr)
                continue
            if inner is not None:
                inner_ids.add(inner)
            # qwen3.5's HF checkpoints are ConditionalGeneration wrappers: the
            # decoder lives at model.language_model.layers, and vLLM's LoRA
            # loader applies the model's hf_to_vllm_mapper
            # ("model.language_model." -> "language_model.model."), so the
            # adapter must use the composite-model HF names. The layer_<k>
            # path component is the qwen3.5 signature (gemma4 uses layers_<k>).
            hf_base = (
                "model.language_model.layers"
                if composite_model or _re.search(r"\['layer_[0-9]+'\]", keystr)
                else "model.layers"
            )
            entries.append((hf_base, hf_block, hf_proj, is_a, inner, arr))
        pattern_len = max(inner_ids) + 1 if inner_ids else 1

        tensors: dict[str, np.ndarray] = {}
        for hf_base, hf_block, hf_proj, is_a, inner, arr in entries:
            for j in range(arr.shape[1]):
                layer = j if inner is None else j * pattern_len + inner
                per_layer = arr[:, j, :]  # lora_a: (in, r); lora_b: (r, out)
                name = f"base_model.model.{hf_base}.{layer}.{hf_block}.{hf_proj}"
                suffix = "lora_A.weight" if is_a else "lora_B.weight"
                tensors[f"{name}.{suffix}"] = np.ascontiguousarray(per_layer.T)
        return tensors

    @staticmethod
    def _maxtext_path_to_hf(keystr: str) -> tuple[str, str, bool, int | None] | None:
        """Parse a MaxText qwix LoRA path into (hf_block, hf_proj, is_lora_a, inner_idx).

        e.g. "['adapter']['base']['decoder']['layers']['self_attention']['query']['kernel_lora_a'].value"
          -> ("self_attn", "q_proj", True, None)
        e.g. "...['decoder']['scanned_blocks']['layers_5']['self_attention']['query']['kernel_lora_a'].value"
          -> ("self_attn", "q_proj", True, 5)  # gemma4 grouped scan
        e.g. "...['decoder']['layers']['layer_3']['attention']['attention']['query']['kernel_lora_a'].value"
          -> ("self_attn", "q_proj", True, 3)  # qwen3.5 inhomogeneous scan

        ``inner_idx`` is None for homogeneous scans (``layers``); for grouped
        scans it is the position inside the repeating block (``layers_<k>``
        gemma4-style, ``layer_<k>`` qwen3.5-style).
        """
        import re

        names = [a or b for a, b in re.findall(r"\['([a-zA-Z_0-9]+)'\]|\[(\d+)\]", keystr)]
        grouped = next((n for n in names if re.fullmatch(r"layers?_[0-9]+", n)), None)
        if grouped is not None:
            layers_pos = names.index(grouped)
            inner: int | None = int(grouped.rsplit("_", 1)[1])
        elif "layers" in names:
            layers_pos = names.index("layers")
            inner = None
        else:
            return None
        try:
            block, proj = names[layers_pos + 1], names[layers_pos + 2]
            leaf = names[layers_pos + 3]
        except (IndexError, ValueError):
            return None
        if (block, proj) == ("attention", "attention"):
            # qwen3.5 hybrid full-attention: DecoderLayer.attention
            # (Qwen3_5FullAttention) wraps attentions.Attention, so the
            # projection sits one level deeper. GDN layers never reach here:
            # their submodules (in_proj_qkvz/in_proj_ba/conv1d/out_proj/...)
            # don't produce an ("attention", "attention") pair.
            try:
                proj = names[layers_pos + 3]
                leaf = names[layers_pos + 4]
            except IndexError:
                return None
            block = "self_attention"
        mapped = _MAXTEXT_PROJ_TO_HF.get((block, proj))
        if mapped is None or "lora" not in leaf:
            return None
        return mapped[0], mapped[1], leaf.endswith("lora_a"), inner

    @staticmethod
    def _qwix_path_to_hf_module(keystr: str) -> str | None:
        """Map a qwix LoRA param path to the HF module name it decorates.

        e.g. "['layers'][0]['attn']['q_proj']['w_lora_a'].value"
          -> "model.layers.0.self_attn.q_proj"
        """
        import re

        parts = re.findall(r"\['([a-z_0-9]+)'\]|\[(\d+)\]", keystr)
        names = [a or b for a, b in parts]
        if not names or "layers" not in names:
            return None
        try:
            layer_pos = names.index("layers")
            layer_idx = names[layer_pos + 1]
            block = names[layer_pos + 2]
            proj = names[layer_pos + 3]
        except (IndexError, ValueError):
            return None
        block_map = {"attn": "self_attn", "mlp": "mlp"}
        if block not in block_map:
            return None
        return f"model.layers.{layer_idx}.{block_map[block]}.{proj}"
