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
    assert synchronize_rpc_result(None) == {}

    def fail():
        raise ValueError("deliberate worker failure")

    with pytest.raises(ValueError, match="deliberate worker failure"):
        call_with_rpc_ack(fail, {})


class _FakeDistributedClient:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.barriers = []

    def key_value_set_bytes(self, key, value):
        self.values[key] = value

    def wait_at_barrier(self, barrier_id, timeout_in_ms):
        self.barriers.append((barrier_id, timeout_in_ms))

    def blocking_key_value_get_bytes(self, key, timeout_in_ms):
        del timeout_in_ms
        return self.values[key]


def test_rpc_ack_uses_host_coordination_instead_of_tpu_collective(monkeypatch):
    client = _FakeDistributedClient(
        {
            "skyrl_rpc_ack/0/1": b"\x00",
            "skyrl_rpc_ack/0/2": b"\x01worker two failed",
            "skyrl_rpc_ack/0/3": b"\x00",
        }
    )
    monkeypatch.setattr("skyrl.backends.rpc._RPC_ACK_SEQUENCE", iter([0]))
    monkeypatch.setattr("skyrl.backends.rpc.jax.process_count", lambda: 4)
    monkeypatch.setattr("skyrl.backends.rpc.jax.process_index", lambda: 0)
    monkeypatch.setattr("skyrl.backends.rpc.jax_distributed.global_state.client", client)
    monkeypatch.setattr(
        "skyrl.backends.rpc.multihost_utils.process_allgather",
        lambda *_args, **_kwargs: pytest.fail("RPC acknowledgement used a TPU collective"),
    )

    assert synchronize_rpc_result(None) == {2: "worker two failed"}
    assert client.values["skyrl_rpc_ack/0/0"] == b"\x00"
    assert client.barriers == [("skyrl_rpc_ack/0/barrier", 30 * 60 * 1000)]
