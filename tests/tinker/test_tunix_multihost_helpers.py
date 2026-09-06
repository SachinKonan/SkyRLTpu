"""CPU mesh tests for Tunix global-array placement and restore helpers."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from skyrl.backends.tunix_backend import TunixBackend

pytestmark = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)


def _helper_backend() -> TunixBackend:
    backend = object.__new__(TunixBackend)
    backend._mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("fsdp", "tensor"))
    return backend


def test_four_rows_shard_only_over_fsdp(monkeypatch):
    backend = _helper_backend()
    monkeypatch.setenv("TUNIX_ROW_SHARD", "2")
    source = np.arange(32, dtype=np.int32).reshape(4, 8)
    placed = backend._shard_batch_arrays(source)[0]

    assert placed.shape == source.shape
    assert placed.sharding.spec == P(("fsdp",))
    np.testing.assert_array_equal(np.asarray(placed), source)

    # TP peers hold the same batch rows; the two FSDP groups own disjoint
    # halves. This is the single-host analogue of two rows per v6e host.
    row_slices = [shard.index[0] for shard in placed.addressable_shards]
    assert row_slices.count(slice(0, 2, None)) == 2
    assert row_slices.count(slice(2, 4, None)) == 2


def test_restore_uses_target_sharding():
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("fsdp", "tensor"))
    sharding = NamedSharding(mesh, P(("fsdp",)))
    target = {"weight": jax.device_put(np.zeros((4, 8), dtype=np.float32), sharding)}
    restored = TunixBackend._state_from_flat(
        target,
        {"['weight']": np.arange(32, dtype=np.float32).reshape(4, 8)},
    )
    assert restored["weight"].sharding == sharding
    np.testing.assert_array_equal(
        np.asarray(restored["weight"]),
        np.arange(32, dtype=np.float32).reshape(4, 8),
    )


def test_restore_prefers_checkpoint_sharding_over_current_target():
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("fsdp", "tensor"))
    current_sharding = NamedSharding(mesh, P(("fsdp",)))
    target = {
        "weight": jax.device_put(
            np.zeros((4, 8), dtype=np.float32),
            current_sharding,
        )
    }
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    restored = TunixBackend._state_from_flat(
        target,
        {"['weight']": values},
        {"['weight']": {"spec": [], "committed": True}},
    )

    assert restored["weight"].sharding == NamedSharding(mesh, P())
    assert restored["weight"].committed
    np.testing.assert_array_equal(np.asarray(restored["weight"]), values)


def test_checkpoint_layouts_round_trip_tuple_axes():
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("fsdp", "tensor"))
    sharding = NamedSharding(mesh, P(("fsdp", "tensor")))
    state = {
        "weight": jax.device_put(
            np.arange(32, dtype=np.float32).reshape(4, 8),
            sharding,
        )
    }

    assert TunixBackend._checkpoint_layouts(state) == {
        "['weight']": {
            "spec": [["fsdp", "tensor"]],
            "committed": True,
        }
    }


@pytest.mark.parametrize("local_exists", [False, True])
def test_restore_fetches_source_model_not_destination_slot(monkeypatch, tmp_path, local_exists):
    backend = _helper_backend()
    checkpoint = tmp_path / "model_source" / "000003.tar.gz"
    backend.config = SimpleNamespace(checkpoint_mirror_gcs="gs://bucket/checkpoints")
    slot = SimpleNamespace(lora_state={}, accum_grads="stale", accum_count=1)
    backend.models = {"model_new_slot": slot}
    calls = []

    def restore(path, mirror, source_model):
        calls.append((path, mirror, source_model))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"checkpoint")
        return f"{mirror}/{source_model}/{path.name}"

    if local_exists:
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr("skyrl.backends.tunix_backend.restore_checkpoint_from_gcs", restore)
    monkeypatch.setattr(backend, "_read_checkpoint_archive", lambda path: {"lora_weights": {}})
    monkeypatch.setattr(backend, "_state_from_flat", lambda *args: {"restored": True})

    backend.load_checkpoint(checkpoint, "model_new_slot")

    assert calls == ([] if local_exists else [(checkpoint, "gs://bucket/checkpoints", "model_source")])
    assert slot.lora_state == {"restored": True}
    assert slot.accum_grads is None
    assert slot.accum_count == 0
