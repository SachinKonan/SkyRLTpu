"""Integration coverage for the optional EasyDeL Tinker backend."""

from __future__ import annotations

import base64
import gzip
import json
import os
import pickle
import tarfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from cloudpathlib import AnyPath
from flax import nnx

pytest.importorskip("easydel")

from skyrl.backends.easydel import (
    EasyDeLBackend,
    EasyDeLBackendConfig,
    EasyDeLBackendImpl,
    _lora_configs_compatible,
    _model_task,
)
from skyrl.backends.jax import JaxBackend, JaxBackendConfig
from skyrl.tinker import types
from skyrl.tinker.engine import (
    get_backend_classes,
    prepare_model_pass_batch,
    prepare_sample_batch,
)

TINY_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
RUN_INTEGRATION = os.environ.get("SKYRL_RUN_EASYDEL_INTEGRATION") == "1"


def _make_input(tokens: list[int]) -> types.ForwardBackwardInput:
    return types.ForwardBackwardInput(
        data=[
            types.Datum(
                model_input=types.ModelInput(
                    chunks=[types.EncodedTextChunk(tokens=tokens)]
                ),
                loss_fn_inputs=types.LossFnInputs(
                    target_tokens=types.TensorData(data=tokens[1:] + [0]),
                    weights=types.TensorData(data=[1.0] * len(tokens)),
                    advantages=types.TensorData(data=[]),
                    logprobs=types.TensorData(data=[]),
                ),
            )
        ],
        loss_fn="cross_entropy",
    )


def _logprobs(output: types.ForwardBackwardOutput) -> np.ndarray:
    return np.asarray(output.loss_fn_outputs[0]["logprobs"]["data"], dtype=np.float32)


def _losses(output: types.ForwardBackwardOutput) -> np.ndarray:
    return np.asarray(
        output.loss_fn_outputs[0]["elementwise_loss"]["data"], dtype=np.float32
    )


def _optimizer_input(learning_rate: float = 1e-4) -> types.OptimStepInput:
    return types.OptimStepInput(
        adam_params=types.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
        )
    )


def _snapshot(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)).copy(), tree
    )


def _assert_trees_equal(left, right) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def _trees_differ(left, right) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    return any(
        not np.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _path_keys(path) -> tuple[object, ...]:
    return tuple(getattr(item, "key", getattr(item, "name", item)) for item in path)


def _canonical_lora_value(layer: int, group: str, shape: tuple[int, ...]) -> np.ndarray:
    group_seed = {"qkv": 11, "o": 23, "gate_up": 37, "down": 53}[group]
    rng = np.random.default_rng(1000 * layer + group_seed)
    return rng.normal(0.0, 0.02, shape).astype(np.float32)


def _initialize_easy_lora_for_parity(backend: EasyDeLBackend, model_id: str) -> None:
    runtime = backend._runtimes[model_id]

    def initialize(path, value):
        keys = _path_keys(path)
        if "lora_a" not in keys and "lora_b" not in keys:
            return value
        layer = int(keys[keys.index("layers") + 1])
        projection = str(keys[-3])
        group = (
            "qkv"
            if projection in {"q_proj", "k_proj", "v_proj"}
            else "gate_up"
            if projection in {"gate_proj", "up_proj"}
            else "o"
            if projection == "o_proj"
            else "down"
        )
        array = np.zeros(value.shape, dtype=np.float32)
        if "lora_a" in keys:
            array = _canonical_lora_value(layer, group, value.shape)
        return jnp.asarray(array, dtype=value.dtype)

    graphstate = jax.tree.map_with_path(initialize, runtime.state.graphstate)
    runtime.state = runtime.state.replace(graphstate=graphstate)


def _easy_adapter_arrays(tree) -> dict[tuple[int, str, str], np.ndarray]:
    arrays: dict[tuple[int, str, str], np.ndarray] = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = _path_keys(path)
        if "lora_a" not in keys and "lora_b" not in keys:
            continue
        layer = int(keys[keys.index("layers") + 1])
        projection = str(keys[-3]).removesuffix("_proj")
        side = "A" if "lora_a" in keys else "B"
        arrays[(layer, projection, side)] = np.asarray(
            jax.device_get(value), dtype=np.float32
        )
    return arrays


def _fused_group_sizes(backend: JaxBackend, projection: str) -> tuple[int, ...]:
    if projection == "gate_up":
        return (1, 1)
    config = backend.model.config
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    q_per_kv = config.num_attention_heads // config.num_key_value_heads
    return (q_per_kv * head_dim, head_dim, head_dim)


def _fuse_components(
    arrays: list[np.ndarray], group_sizes: tuple[int, ...]
) -> np.ndarray:
    num_groups = arrays[0].shape[-1] // group_sizes[0]
    grouped = [
        array.reshape(*array.shape[:-1], num_groups, size)
        for array, size in zip(arrays, group_sizes)
    ]
    return np.concatenate(grouped, axis=-1).reshape(*arrays[0].shape[:-1], -1)


def _split_components(
    array: np.ndarray, group_sizes: tuple[int, ...]
) -> list[np.ndarray]:
    num_groups = array.shape[-1] // sum(group_sizes)
    grouped = array.reshape(*array.shape[:-1], num_groups, sum(group_sizes))
    result = []
    offset = 0
    for size in group_sizes:
        result.append(
            grouped[..., offset : offset + size].reshape(
                *array.shape[:-1], num_groups * size
            )
        )
        offset += size
    return result


def _initialize_jax_lora_from_easy(
    backend: JaxBackend,
    easy: dict[tuple[int, str, str], np.ndarray],
    *,
    rank: int,
) -> None:
    adapter_index = backend.models["parity"].adapter_index

    def initialize(path, value):
        keys = _path_keys(path)
        if "lora_A" not in keys and "lora_B" not in keys:
            return value
        projection = str(keys[-3]).removesuffix("_proj")
        side = "A" if "lora_A" in keys else "B"
        if projection in {"lm_head", "embed_tokens"}:
            return value
        replacement = value
        for layer in range(value.shape[0]):
            if projection == "qkv":
                matrix = (
                    easy[(layer, "q", side)]
                    if side == "A"
                    else _fuse_components(
                        [easy[(layer, name, side)] for name in ("q", "k", "v")],
                        _fused_group_sizes(backend, projection),
                    )
                )
            elif projection == "gate_up":
                matrix = (
                    easy[(layer, "gate", side)]
                    if side == "A"
                    else _fuse_components(
                        [easy[(layer, name, side)] for name in ("gate", "up")],
                        _fused_group_sizes(backend, projection),
                    )
                )
            else:
                matrix = easy[(layer, projection, side)]
            matrix = jnp.asarray(matrix, dtype=value.dtype)
            if side == "A":
                replacement = replacement.at[layer, adapter_index, ..., :rank].set(
                    matrix
                )
            else:
                replacement = replacement.at[layer, adapter_index, :rank, ...].set(
                    matrix
                )
        return replacement

    nnx.update(
        backend.lora_params, jax.tree.map_with_path(initialize, backend.lora_params)
    )


def _jax_adapter_arrays(
    backend: JaxBackend,
    tree,
    *,
    rank: int,
) -> dict[tuple[int, str, str], np.ndarray]:
    adapter_index = backend.models["parity"].adapter_index
    arrays: dict[tuple[int, str, str], np.ndarray] = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = _path_keys(path)
        if "lora_A" not in keys and "lora_B" not in keys:
            continue
        projection = str(keys[-3]).removesuffix("_proj")
        side = "A" if "lora_A" in keys else "B"
        if projection in {"lm_head", "embed_tokens"}:
            continue
        active = np.asarray(
            jax.device_get(
                value[:, adapter_index, ..., :rank]
                if side == "A"
                else value[:, adapter_index, :rank, ...]
            ),
            dtype=np.float32,
        )
        for layer, matrix in enumerate(active):
            if projection == "qkv":
                if side == "A":
                    for name in ("q", "k", "v"):
                        arrays[(layer, name, side)] = matrix
                else:
                    components = _split_components(
                        matrix, _fused_group_sizes(backend, projection)
                    )
                    for name, component in zip(
                        ("q", "k", "v"), components, strict=True
                    ):
                        arrays[(layer, name, side)] = component
            elif projection == "gate_up":
                if side == "A":
                    arrays[(layer, "gate", side)] = matrix
                    arrays[(layer, "up", side)] = matrix
                else:
                    components = _split_components(
                        matrix, _fused_group_sizes(backend, projection)
                    )
                    for name, component in zip(("gate", "up"), components, strict=True):
                        arrays[(layer, name, side)] = component
            else:
                arrays[(layer, projection, side)] = matrix
    return arrays


def _flatten_mapped(
    arrays: dict[tuple[int, str, str], np.ndarray], side: str
) -> np.ndarray:
    return np.concatenate(
        [arrays[key].ravel() for key in sorted(arrays) if key[-1] == side]
    )


def _make_rl_datum(
    prompt_tokens: list[int], sequence: types.GeneratedSequence, advantage: float
) -> types.Datum:
    full_sequence = prompt_tokens + sequence.tokens
    response_mask = [0.0] * len(prompt_tokens) + [1.0] * len(sequence.tokens)
    sampling_logprobs = [0.0] * len(prompt_tokens) + sequence.logprobs
    advantages = [0.0] * len(prompt_tokens) + [advantage] * len(sequence.tokens)
    return types.Datum(
        model_input=types.ModelInput(
            chunks=[types.EncodedTextChunk(tokens=full_sequence[:-1])]
        ),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=full_sequence[1:]),
            weights=types.TensorData(data=response_mask[1:]),
            advantages=types.TensorData(data=advantages[1:]),
            logprobs=types.TensorData(data=sampling_logprobs[1:]),
        ),
    )


def test_model_task_detects_staged_qwen35_checkpoint(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
            }
        )
    )
    assert _model_task(str(tmp_path), "Qwen/Qwen3.5-9B", "auto") == "image_text_to_text"


def test_model_task_explicit_override_wins(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert _model_task(str(tmp_path), "Qwen/Qwen3.5-9B", "causal_lm") == "causal_lm"


def test_checkpoint_lora_compatibility_ignores_initialization_seed():
    saved = types.LoraConfig(rank=32, alpha=32, seed=1)
    fresh = types.LoraConfig(rank=32, alpha=32, seed=2)
    incompatible = types.LoraConfig(rank=16, alpha=32, seed=1)

    assert _lora_configs_compatible(saved, fresh)
    assert not _lora_configs_compatible(saved, incompatible)


def test_non_policy_model_role_is_rejected_before_model_creation():
    backend = object.__new__(EasyDeLBackendImpl)
    with pytest.raises(ValueError, match="only supports model_role='policy'"):
        backend.create_model(
            "critic",
            types.LoraConfig(rank=2, alpha=4, seed=0),
            model_role="critic",
        )


def test_critic_objective_is_rejected_before_dispatch():
    backend = object.__new__(EasyDeLBackendImpl)
    batch = types.PreparedModelPassBatch(
        all_model_inputs=[
            types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])])
        ],
        all_targets=[[2, 0]],
        all_token_weights=[[1.0, 1.0]],
        all_sampling_logprobs=[[]],
        all_advantages=[[]],
        all_values=[[0.0, 0.0]],
        all_returns=[[0.0, 0.0]],
        all_model_ids=["critic"],
        all_loss_fns=["ppo_critic"],
        all_loss_fn_configs=[None],
        request_batch_slices=[("request", "critic", 0, 1)],
    )
    with pytest.raises(ValueError, match="only supported by the SkyRL-Train backend"):
        backend.forward(batch)


def test_engine_selects_easydel_without_ray():
    backend_class, config_class = get_backend_classes("easydel")
    assert backend_class is EasyDeLBackend
    assert config_class is EasyDeLBackendConfig
    with pytest.raises(ValueError, match="does not support use_ray"):
        get_backend_classes("easydel", use_ray=True)


def test_learner_pass_releases_colocated_sampler_state():
    events = []

    class FakeRunner:
        model = object()

        @staticmethod
        def destroy_kv_cache():
            events.append("destroy-kv")

    class FakeEngine:
        _paused = False
        _kv_cache_valid = True
        num_running_requests = 0
        num_pending_requests = 0
        runner = FakeRunner()

        def pause(self):
            events.append("pause")
            self._paused = True

        @staticmethod
        def release_model_state(*, clear_compiled_cache: bool):
            events.append(("release-model", clear_compiled_cache))

    backend = object.__new__(EasyDeLBackendImpl)
    runtime = SimpleNamespace(sampling_engine=FakeEngine(), sampling_model=object())
    backend._base_sampling_engine = None
    backend._runtimes = {"policy": runtime}

    backend._pause_sampling_engines()

    assert events == ["pause", "destroy-kv", ("release-model", False)]
    assert runtime.sampling_model is None


def test_sampling_engine_restores_released_weights_before_resume():
    events = []

    class FakeEngine:
        _paused = True
        runner = SimpleNamespace(model=None)

        def resume(self):
            events.append("resume")
            self._paused = False

    backend = object.__new__(EasyDeLBackendImpl)
    runtime = SimpleNamespace(sampling_engine=FakeEngine(), sampling_model=None)
    backend._base_sampling_engine = None
    backend._runtimes = {"policy": runtime}

    def refresh(model_id: str):
        events.append(("refresh", model_id))
        runtime.sampling_model = object()

    backend._refresh_esurge = refresh

    assert backend._sampling_engine("policy") is runtime.sampling_engine
    assert events == [("refresh", "policy"), "resume"]


@pytest.fixture(scope="module")
def easydel_backend() -> EasyDeLBackend:
    if not RUN_INTEGRATION:
        pytest.skip(
            "set SKYRL_RUN_EASYDEL_INTEGRATION=1 to run model-backed EasyDeL tests"
        )
    base_model = os.environ.get("SKYRL_EASYDEL_BASE_MODEL", TINY_MODEL)
    source = os.environ.get("SKYRL_EASYDEL_MODEL_PATH")
    tokenizer = os.environ.get("SKYRL_EASYDEL_TOKENIZER_PATH")
    from_torch_value = os.environ.get("SKYRL_EASYDEL_FROM_TORCH")
    from_torch = (
        None
        if from_torch_value is None
        else from_torch_value.lower() in {"1", "true", "yes"}
    )
    return EasyDeLBackend(
        base_model,
        EasyDeLBackendConfig(
            max_lora_adapters=8,
            max_lora_rank=32,
            model_name_or_path=source,
            tokenizer_name_or_path=tokenizer,
            from_torch=from_torch,
            model_task=os.environ.get("SKYRL_EASYDEL_MODEL_TASK", "auto"),
            dtype=os.environ.get("SKYRL_EASYDEL_DTYPE", "bfloat16"),
            data_parallel_size=int(os.environ.get("SKYRL_EASYDEL_DP", "1")),
            fully_sharded_data_parallel_size=int(
                os.environ.get("SKYRL_EASYDEL_FSDP", "1")
            ),
            expert_parallel_size=int(os.environ.get("SKYRL_EASYDEL_EP", "1")),
            tensor_parallel_size=int(os.environ.get("SKYRL_EASYDEL_TP", "-1")),
            sequence_parallel_size=int(os.environ.get("SKYRL_EASYDEL_SP", "1")),
            train_micro_batch_size=1,
            enforce_eager=os.environ.get("SKYRL_EASYDEL_EAGER") == "1",
            use_scan_mlp=False,
            lmhead_token_chunk_size=int(
                os.environ.get("SKYRL_EASYDEL_LMHEAD_TOKEN_CHUNK", "256")
            ),
            lmhead_vocab_chunk_size=int(
                os.environ.get("SKYRL_EASYDEL_LMHEAD_VOCAB_CHUNK", "32768")
            ),
            sample_max_num_sequences=int(
                os.environ.get("SKYRL_EASYDEL_SAMPLE_MAX_SEQS", "2")
            ),
            sample_max_model_len=int(
                os.environ.get("SKYRL_EASYDEL_SAMPLE_MAX_LEN", "128")
            ),
            sample_hbm_utilization=float(
                os.environ.get("SKYRL_EASYDEL_SAMPLE_HBM", "0.5")
            ),
        ),
    )


@pytest.mark.skipif(not RUN_INTEGRATION, reason="EasyDeL model integration is opt-in")
def test_forward_parity_with_current_jax_backend(easydel_backend: EasyDeLBackend):
    if easydel_backend.base_model != TINY_MODEL:
        pytest.skip("numerical parity uses the shared tiny Qwen3 checkpoint")

    model_id = "parity"
    lora_config = types.LoraConfig(rank=2, alpha=4, seed=0)
    easydel_backend.create_model(model_id, lora_config)
    jax_backend = JaxBackend(
        TINY_MODEL,
        JaxBackendConfig(
            max_lora_adapters=8, max_lora_rank=32, train_micro_batch_size=1
        ),
    )
    jax_backend.create_model(model_id, lora_config)

    # The backends represent the same adapters differently: JAX fuses QKV and
    # gate/up and pads rank to max_lora_rank, while EasyDeL stores separate
    # requested-rank matrices. Inject one shared effective initialization so
    # this is a numerical backend comparison rather than an RNG comparison.
    _initialize_easy_lora_for_parity(easydel_backend, model_id)
    easy_initial = _easy_adapter_arrays(
        easydel_backend._runtimes[model_id].state.graphstate
    )
    _initialize_jax_lora_from_easy(jax_backend, easy_initial, rank=lora_config.rank)
    jax_initial = _jax_adapter_arrays(
        jax_backend,
        jax_backend.lora_params,
        rank=lora_config.rank,
    )
    assert easy_initial.keys() == jax_initial.keys()
    for key in easy_initial:
        np.testing.assert_array_equal(easy_initial[key], jax_initial[key])

    request = _make_input([1, 2, 3, 4, 5, 6, 7, 8])
    batch = prepare_model_pass_batch({"request": (model_id, request)})
    easy_before = _snapshot(easydel_backend._runtimes[model_id].state.graphstate)
    jax_before = _snapshot(jax_backend.lora_params)
    easy_output = easydel_backend.forward_backward(batch)["request"]
    jax_output = jax_backend.forward_backward(batch)["request"]
    expected = _logprobs(jax_output)
    actual = _logprobs(easy_output)

    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(
        _losses(easy_output), _losses(jax_output), rtol=5e-2, atol=5e-2
    )

    easy_grad_norm = float(
        optax.global_norm(easydel_backend._runtimes[model_id].grad_sum)
    )
    adapter_index = jax_backend.models[model_id].adapter_index
    jax_grads = jax_backend.accumulated_grads.get_mean(adapter_index)
    jax_grad_norm = float(optax.global_norm(jax_grads))
    assert easy_grad_norm > 0 and jax_grad_norm > 0
    easy_mapped_grads = _easy_adapter_arrays(
        easydel_backend._runtimes[model_id].grad_sum
    )
    jax_mapped_grads = _jax_adapter_arrays(
        jax_backend,
        jax_grads,
        rank=lora_config.rank,
    )
    easy_b_grads = _flatten_mapped(easy_mapped_grads, "B")
    jax_b_grads = _flatten_mapped(jax_mapped_grads, "B")
    cosine = float(
        np.dot(easy_b_grads, jax_b_grads)
        / (np.linalg.norm(easy_b_grads) * np.linalg.norm(jax_b_grads))
    )
    per_projection_cosines = {}
    for key in sorted(easy_mapped_grads):
        if key[-1] != "B":
            continue
        left = easy_mapped_grads[key].ravel()
        right = jax_mapped_grads[key].ravel()
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        per_projection_cosines[key] = (
            float(np.dot(left, right) / denominator) if denominator else 1.0
        )
    assert cosine > 0.95, (
        f"mapped gradient cosine={cosine}; per_projection={per_projection_cosines}"
    )
    assert np.linalg.norm(easy_b_grads) / np.linalg.norm(jax_b_grads) == pytest.approx(
        1.0, rel=0.25
    )

    optimizer_input = _optimizer_input(learning_rate=1e-4)
    easy_metrics = easydel_backend.optim_step(model_id, optimizer_input).metrics
    jax_metrics = jax_backend.optim_step(model_id, optimizer_input).metrics
    assert easy_metrics is not None and jax_metrics is not None
    assert easy_metrics["skyrl.ai/grad_norm"] == pytest.approx(easy_grad_norm, rel=2e-2)
    assert jax_metrics["skyrl.ai/grad_norm"] == pytest.approx(jax_grad_norm, rel=2e-2)
    assert _trees_differ(
        easy_before, _snapshot(easydel_backend._runtimes[model_id].state.graphstate)
    )
    assert _trees_differ(jax_before, _snapshot(jax_backend.lora_params))

    easy_parameters = _easy_adapter_arrays(
        easydel_backend._runtimes[model_id].state.graphstate
    )
    jax_parameters = _jax_adapter_arrays(
        jax_backend,
        jax_backend.lora_params,
        rank=lora_config.rank,
    )
    easy_b_update = _flatten_mapped(easy_parameters, "B") - _flatten_mapped(
        easy_initial, "B"
    )
    jax_b_update = _flatten_mapped(jax_parameters, "B") - _flatten_mapped(
        jax_initial, "B"
    )
    update_cosine = float(
        np.dot(easy_b_update, jax_b_update)
        / (np.linalg.norm(easy_b_update) * np.linalg.norm(jax_b_update))
    )
    assert update_cosine > 0.95
    assert np.linalg.norm(easy_b_update) / np.linalg.norm(
        jax_b_update
    ) == pytest.approx(1.0, rel=0.2)

    easy_after = _logprobs(easydel_backend.forward(batch)["request"])
    jax_after = _logprobs(jax_backend.forward(batch)["request"])
    np.testing.assert_allclose(easy_after, jax_after, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="EasyDeL model integration is opt-in")
def test_optimizer_and_checkpoint_roundtrip(
    easydel_backend: EasyDeLBackend, tmp_path: Path
):
    model_id = "checkpoint"
    easydel_backend.create_model(model_id, types.LoraConfig(rank=2, alpha=4, seed=1))
    batch = prepare_model_pass_batch(
        {"request": (model_id, _make_input([11, 12, 13, 14, 15, 16]))}
    )

    easydel_backend.forward_backward(batch)
    metrics = easydel_backend.optim_step(model_id, _optimizer_input()).metrics
    assert metrics is not None
    assert metrics["skyrl.ai/grad_norm"] > 0
    saved_state = _snapshot(easydel_backend._runtimes[model_id].state.graphstate)
    saved_step = int(jax.device_get(easydel_backend._runtimes[model_id].state.step))

    checkpoint_path = AnyPath(tmp_path / "checkpoint")
    easydel_backend.save_checkpoint(checkpoint_path, model_id)
    easydel_backend.forward_backward(batch)
    easydel_backend.optim_step(model_id, _optimizer_input())
    easydel_backend.load_checkpoint(checkpoint_path, model_id)

    restored = easydel_backend._runtimes[model_id]
    _assert_trees_equal(_snapshot(restored.state.graphstate), saved_state)
    assert int(jax.device_get(restored.state.step)) == saved_step
    assert restored.grad_count == 0


@pytest.mark.skipif(not RUN_INTEGRATION, reason="EasyDeL model integration is opt-in")
def test_checkpoint_payload_roundtrip_without_jax_collectives(
    easydel_backend: EasyDeLBackend, tmp_path: Path
):
    model_id = "checkpoint-payload"
    easydel_backend.create_model(model_id, types.LoraConfig(rank=2, alpha=4, seed=2))
    batch = prepare_model_pass_batch(
        {"request": (model_id, _make_input([21, 22, 23, 24, 25, 26]))}
    )

    easydel_backend.forward_backward(batch)
    easydel_backend.optim_step(model_id, _optimizer_input())
    saved_state = _snapshot(easydel_backend._runtimes[model_id].state.graphstate)
    target = easydel_backend._checkpoint_target(model_id)
    payload = {
        "arrays": jax.device_get(
            {key: target[key] for key in ("graphstate", "opt_state", "step")}
        ),
        "lora_config": easydel_backend.models[model_id].lora_config.model_dump(),
    }
    compressed = gzip.compress(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=1
    )

    easydel_backend.forward_backward(batch)
    easydel_backend.optim_step(model_id, _optimizer_input())
    easydel_backend.load_checkpoint_payload(
        base64.b64encode(compressed).decode("ascii"), model_id
    )
    _assert_trees_equal(
        _snapshot(easydel_backend._runtimes[model_id].state.graphstate), saved_state
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(compressed)
    archive_path = tmp_path / "sampler.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(checkpoint, arcname="checkpoint")
    easydel_backend.load_sampler_checkpoint_payload(
        model_id,
        "payload-checkpoint",
        base64.b64encode(archive_path.read_bytes()).decode("ascii"),
    )
    assert (
        easydel_backend._runtimes[model_id].metadata.loaded_checkpoint_id
        == "payload-checkpoint"
    )


@pytest.mark.skipif(not RUN_INTEGRATION, reason="EasyDeL model integration is opt-in")
def test_esurge_sampling_matches_tinker_schema(easydel_backend: EasyDeLBackend):
    if easydel_backend.base_model != TINY_MODEL:
        pytest.skip("the fast eSurge integration smoke uses the tiny Qwen3 checkpoint")

    prompt_tokens = easydel_backend.tokenizer.encode(
        "The capital of France is", add_special_tokens=False
    )
    request = types.SampleInput(
        base_model=easydel_backend.base_model,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=prompt_tokens)]),
        sampling_params=types.SamplingParams(
            temperature=0.7,
            max_tokens=2,
            seed=0,
            top_k=16,
            top_p=0.95,
        ),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
    )
    output = easydel_backend.sample(prepare_sample_batch({"request": ("", request)}))[
        "request"
    ]
    assert len(output.sequences) == 1
    assert 0 < len(output.sequences[0].tokens) <= 2
    assert len(output.sequences[0].logprobs) == len(output.sequences[0].tokens)
    assert np.isfinite(output.sequences[0].logprobs).all()


@pytest.mark.skipif(not RUN_INTEGRATION, reason="EasyDeL model integration is opt-in")
def test_rl_rollout_update_and_sampler_refresh(easydel_backend: EasyDeLBackend):
    if easydel_backend.base_model != TINY_MODEL:
        pytest.skip("the fast RL lifecycle regression uses the tiny Qwen3 checkpoint")

    model_id = "rl-lifecycle"
    easydel_backend.create_model(model_id, types.LoraConfig(rank=2, alpha=4, seed=3))
    prompt_tokens = easydel_backend.tokenizer.encode(
        "Continue this sequence:", add_special_tokens=False
    )
    sample_request = types.SampleInput(
        base_model=easydel_backend.base_model,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=prompt_tokens)]),
        sampling_params=types.SamplingParams(
            temperature=1.0,
            max_tokens=4,
            seed=17,
            top_k=32,
            top_p=0.95,
        ),
        num_samples=2,
        checkpoint_id="",
        prompt_logprobs=False,
    )
    rollout = easydel_backend.sample(
        prepare_sample_batch({"rollout": (model_id, sample_request)})
    )["rollout"]
    assert len(rollout.sequences) == 2
    assert all(sequence.tokens for sequence in rollout.sequences)
    assert all(np.isfinite(sequence.logprobs).all() for sequence in rollout.sequences)

    request = types.ForwardBackwardInput(
        data=[
            _make_rl_datum(prompt_tokens, rollout.sequences[0], 1.0),
            _make_rl_datum(prompt_tokens, rollout.sequences[1], -1.0),
        ],
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    )
    before = _snapshot(easydel_backend._runtimes[model_id].state.graphstate)
    output = easydel_backend.forward_backward(
        prepare_model_pass_batch({"rl": (model_id, request)})
    )["rl"]
    assert all(np.isfinite(_logprobs(output)))
    metrics = easydel_backend.optim_step(
        model_id, _optimizer_input(learning_rate=1e-5)
    ).metrics
    assert metrics is not None
    assert np.isfinite(metrics["skyrl.ai/grad_norm"])
    assert metrics["skyrl.ai/grad_norm"] > 0
    after = _snapshot(easydel_backend._runtimes[model_id].state.graphstate)
    assert _trees_differ(before, after)

    easydel_backend._refresh_esurge(model_id)
    runtime = easydel_backend._runtimes[model_id]
    comparison_tokens = jnp.asarray([prompt_tokens], dtype=jnp.int32)
    comparison_mask = jnp.ones_like(comparison_tokens)
    with easydel_backend.mesh:
        learner_logits = runtime.state.model(
            input_ids=comparison_tokens,
            attention_mask=comparison_mask,
        ).logits
        sampler_logits = runtime.sampling_model(
            input_ids=comparison_tokens,
            attention_mask=comparison_mask,
        ).logits
    np.testing.assert_allclose(
        np.asarray(sampler_logits),
        np.asarray(learner_logits),
        rtol=2e-2,
        atol=2e-2,
    )
    refreshed = easydel_backend.sample(
        prepare_sample_batch({"after": (model_id, sample_request)})
    )["after"]
    assert len(refreshed.sequences) == 2
    assert all(np.isfinite(sequence.logprobs).all() for sequence in refreshed.sequences)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="EasyDeL model integration is opt-in")
def test_configured_long_sequence_train_step(easydel_backend: EasyDeLBackend):
    sequence_length = int(os.environ.get("SKYRL_EASYDEL_LONG_SEQ_LEN", "0"))
    if sequence_length <= 0:
        pytest.skip(
            "set SKYRL_EASYDEL_LONG_SEQ_LEN to run the long-sequence train smoke"
        )

    model_id = "long-sequence"
    easydel_backend.create_model(model_id, types.LoraConfig(rank=2, alpha=4, seed=2))
    model = easydel_backend._base_model_instance()
    vocab_size = int(model.config.get_text_config().vocab_size)
    tokens = (
        (np.arange(sequence_length, dtype=np.int64) % (vocab_size - 10)) + 10
    ).tolist()
    output = easydel_backend.forward_backward(
        prepare_model_pass_batch({"request": (model_id, _make_input(tokens))})
    )["request"]
    assert np.isfinite(_logprobs(output)).all()
    metrics = easydel_backend.optim_step(
        model_id, _optimizer_input(learning_rate=1e-6)
    ).metrics
    assert metrics is not None
    assert np.isfinite(metrics["skyrl.ai/grad_norm"])
