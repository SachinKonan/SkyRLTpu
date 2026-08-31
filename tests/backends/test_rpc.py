"""Lightweight tests for the backend-neutral distributed RPC layer."""

import pytest
from cloudpathlib import AnyPath

from skyrl.backends.rpc import (
    RpcPayload,
    call_with_rpc_ack,
    deserialize_call_kwargs,
    serialize_call_kwargs,
    serialize_config,
    synchronize_rpc_result,
)
from skyrl.backends.tunix_backend import (
    DistributedTunixBackend,
    TunixBackend,
    TunixBackendConfig,
)
from skyrl.tinker import types
from skyrl.tinker.engine import get_backend_classes


def test_tunix_engine_uses_distributed_facade():
    backend_type, config_type = get_backend_classes("tunix")
    assert backend_type is DistributedTunixBackend
    assert config_type is TunixBackendConfig


def test_rpc_payload_carries_backend_selection():
    payload = RpcPayload(
        method="__init__",
        backend_name="tunix",
        kwargs={"base_model": "tiny", "config": {}},
    )
    assert RpcPayload.model_validate_json(payload.model_dump_json()) == payload


def test_tunix_config_serializes_distributed_fields():
    config = TunixBackendConfig(
        coordinator_address="127.0.0.1:7777",
        num_processes=2,
    )
    encoded = serialize_config(TunixBackendConfig, config)
    decoded = TunixBackendConfig.model_validate(encoded)
    assert decoded.coordinator_address == "127.0.0.1:7777"
    assert decoded.num_processes == 2


def test_call_kwargs_round_trip_pydantic_and_path():
    lora = types.LoraConfig(rank=8, alpha=16.0, seed=7)
    encoded = serialize_call_kwargs(
        TunixBackend,
        "create_model",
        {"model_id": "policy", "lora_config": lora, "model_role": "policy"},
    )
    decoded = deserialize_call_kwargs(TunixBackend.create_model, encoded)
    assert decoded["model_id"] == "policy"
    assert decoded["lora_config"] == lora

    encoded_path = serialize_call_kwargs(
        TunixBackend,
        "save_checkpoint",
        {"output_path": AnyPath("/tmp/test-checkpoint.tar.gz"), "model_id": "policy"},
    )
    decoded_path = deserialize_call_kwargs(TunixBackend.save_checkpoint, encoded_path)
    assert str(decoded_path["output_path"]) == "/tmp/test-checkpoint.tar.gz"


def test_rpc_ack_surfaces_local_failure_without_exiting_worker(monkeypatch):
    monkeypatch.setattr(
        "skyrl.backends.rpc.multihost_utils.process_allgather",
        lambda value, tiled: value,
    )
    assert synchronize_rpc_result(None) == {}

    def fail():
        raise ValueError("deliberate worker failure")

    with pytest.raises(ValueError, match="deliberate worker failure"):
        call_with_rpc_ack(fail, {})
