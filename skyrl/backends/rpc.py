"""Backend-neutral JAX multi-process command broadcast.

Process 0 owns the Tinker engine and broadcasts its ordered backend calls to
the remaining JAX processes.  The data plane (model collectives) stays inside
each backend implementation; this module only keeps every process on the same
method call in the same order.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

import jax
import numpy as np
from cloudpathlib import AnyPath
from jax.experimental import multihost_utils
from pydantic import BaseModel, TypeAdapter

from skyrl.utils.log import logger


class RpcPayload(BaseModel):
    """One ordered backend call broadcast by process 0."""

    method: str
    kwargs: dict[str, Any]
    backend_name: str | None = None


_RPC_PAYLOAD_ADAPTER: TypeAdapter[RpcPayload] = TypeAdapter(RpcPayload)
_MAX_ERROR_BYTES = 16 * 1024


def broadcast_command(cmd: RpcPayload | None, process_id: int) -> RpcPayload:
    """Broadcast one JSON RPC payload from process 0 to the JAX world."""
    is_source = process_id == 0
    if is_source:
        if cmd is None:
            raise ValueError("Coordinator must provide a command to broadcast")
        data = _RPC_PAYLOAD_ADAPTER.dump_json(cmd)
        size = np.array([len(data)], dtype=np.int64)
    else:
        size = np.array([0], dtype=np.int64)

    size = multihost_utils.broadcast_one_to_all(size, is_source=is_source)
    if is_source:
        data_array = np.frombuffer(data, dtype=np.uint8)
    else:
        data_array = np.zeros(int(size[0]), dtype=np.uint8)
    data_array = multihost_utils.broadcast_one_to_all(data_array, is_source=is_source)
    return _RPC_PAYLOAD_ADAPTER.validate_json(data_array.tobytes())


def serialize_config(config_type: type[BaseModel], config: BaseModel) -> dict[str, Any]:
    return TypeAdapter(config_type).dump_python(config, mode="json")


def serialize_call_kwargs(implementation_type: type, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize a backend call using the implementation method annotations."""
    hints = get_type_hints(getattr(implementation_type, method))
    serialized: dict[str, Any] = {}
    for key, value in kwargs.items():
        hint = hints.get(key)
        if hint is AnyPath:
            serialized[key] = str(value)
        elif hint is not None:
            serialized[key] = TypeAdapter(hint).dump_python(value, mode="json")
        else:
            serialized[key] = value
    return serialized


def deserialize_call_kwargs(method: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rehydrate a broadcast call using the bound implementation annotations."""
    hints = get_type_hints(method)
    hydrated: dict[str, Any] = {}
    for key, value in kwargs.items():
        hint = hints.get(key)
        if hint is AnyPath:
            hydrated[key] = AnyPath(value)
        elif hint is not None:
            hydrated[key] = TypeAdapter(hint).validate_python(value)
        else:
            hydrated[key] = value
    return hydrated


def synchronize_rpc_result(error: str | None) -> dict[int, str]:
    """Exchange per-process command failures before accepting another RPC.

    A worker must not silently exit after process 0 has broadcast a command:
    the next model collective would otherwise hang forever. Every rank reaches
    this small acknowledgement collective after a backend method returns or
    raises. Workers remain in the command loop; process 0 surfaces the combined
    failure to the Tinker request.
    """
    encoded = (error or "").encode("utf-8", errors="replace")[:_MAX_ERROR_BYTES]
    lengths = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([len(encoded)], dtype=np.int32),
            tiled=True,
        )
    ).reshape(-1)
    width = max(1, int(lengths.max(initial=0)))
    local = np.zeros(width, dtype=np.uint8)
    local[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(local, tiled=True)).reshape(-1, width)
    return {
        process_id: bytes(gathered[process_id, :length]).decode("utf-8", errors="replace")
        for process_id, length in enumerate(lengths.tolist())
        if length
    }


def call_with_rpc_ack(method: Callable, kwargs: dict[str, Any]):
    """Execute one local backend method and synchronize success/failure."""
    result = None
    local_exception: Exception | None = None
    local_error: str | None = None
    try:
        result = method(**kwargs)
    except Exception as exc:  # keep the worker alive long enough to report it
        local_exception = exc
        local_error = traceback.format_exc()

    errors = synchronize_rpc_result(local_error)
    if local_exception is not None:
        raise local_exception
    if errors:
        details = "\n".join(f"process {pid}:\n{message}" for pid, message in sorted(errors.items()))
        raise RuntimeError(f"Distributed backend command failed:\n{details}")
    return result


@dataclass(frozen=True)
class _BackendWorkerSpec:
    config_type: type[BaseModel]
    implementation_type: type
    create: Callable[[str, BaseModel, int], Any]
    prepare: Callable[[BaseModel, int], bool] = lambda _config, _process_id: True


def _load_worker_spec(backend_name: str) -> _BackendWorkerSpec:
    """Lazily import a backend only after JAX distributed initialization."""
    if backend_name == "jax":
        from skyrl.backends.jax import (
            JaxBackendConfig,
            JaxBackendImpl,
            _get_worker_process_index_map,
            _is_active_backend_worker,
            _needs_worker_process_index_map,
        )

        def prepare(config: BaseModel, process_id: int) -> bool:
            assert isinstance(config, JaxBackendConfig)
            if _needs_worker_process_index_map(config):
                _get_worker_process_index_map(process_id)
            return _is_active_backend_worker(config, process_id)

        return _BackendWorkerSpec(
            config_type=JaxBackendConfig,
            implementation_type=JaxBackendImpl,
            create=lambda base_model, config, process_id: JaxBackendImpl(
                base_model, config, process_id  # type: ignore[arg-type]
            ),
            prepare=prepare,
        )
    if backend_name == "tunix":
        from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig

        return _BackendWorkerSpec(
            config_type=TunixBackendConfig,
            implementation_type=TunixBackend,
            create=lambda base_model, config, _process_id: TunixBackend(
                base_model, config  # type: ignore[arg-type]
            ),
        )
    raise ValueError(f"Unsupported distributed backend {backend_name!r}")


def run_worker(
    coordinator_address: str,
    num_processes: int,
    process_id: int,
    *,
    expected_backend: str | None = None,
) -> None:
    """Initialize JAX and run the backend selected by the INIT payload."""
    if process_id == 0:
        raise ValueError("Worker process_id must be > 0 (process 0 is the coordinator)")

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )
    logger.info(
        "Worker process_id=%s (%s total) initialized; waiting for backend INIT",
        process_id,
        jax.process_count(),
    )

    init_payload = broadcast_command(None, process_id=process_id)
    if init_payload.method != "__init__":
        raise ValueError(f"Expected __init__, got {init_payload.method}")
    backend_name = init_payload.backend_name or expected_backend
    if backend_name is None:
        raise ValueError("Distributed INIT payload did not select a backend")
    if expected_backend is not None and backend_name != expected_backend:
        raise ValueError(f"Worker expected backend {expected_backend!r}, got {backend_name!r}")

    spec = _load_worker_spec(backend_name)
    config = spec.config_type.model_validate(init_payload.kwargs["config"])
    active = spec.prepare(config, process_id)
    backend = (
        spec.create(init_payload.kwargs["base_model"], config, process_id)
        if active
        else None
    )
    logger.info(
        "Worker process_id=%s entering %s command loop (active=%s)",
        process_id,
        backend_name,
        active,
    )

    while True:
        payload = broadcast_command(None, process_id=process_id)
        if payload.method == "__shutdown__":
            synchronize_rpc_result(None)
            logger.info("Worker process_id=%s received shutdown", process_id)
            return
        if backend is None:
            logger.info("Inactive worker %s ignoring method %s", process_id, payload.method)
            synchronize_rpc_result(None)
            continue
        try:
            if not hasattr(backend, payload.method):
                raise AttributeError(f"{backend_name} backend has no RPC method {payload.method!r}")
            method = getattr(backend, payload.method)
            kwargs = deserialize_call_kwargs(method, payload.kwargs)
        except Exception:
            errors = synchronize_rpc_result(traceback.format_exc())
            logger.error("Distributed command preparation failed: %s", errors)
            continue
        try:
            call_with_rpc_ack(method, kwargs)
        except Exception:
            # The coordinator receives the same acknowledgement and returns
            # the failure to its caller. Stay alive so cleanup/retry is atomic
            # instead of turning the next command into an unexplained hang.
            logger.exception("Worker process_id=%s command %s failed", process_id, payload.method)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SkyRL distributed backend worker")
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--backend", choices=("jax", "tunix"), default=None)
    args = parser.parse_args()
    run_worker(
        args.coordinator_address,
        args.num_processes,
        args.process_id,
        expected_backend=args.backend,
    )


if __name__ == "__main__":
    main()
