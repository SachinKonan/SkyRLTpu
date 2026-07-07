#!/usr/bin/env python3
"""Probe whether a full TPU slice can run independent JAX subset meshes."""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _device_array(devices: list[jax.Device], rows: int, cols: int) -> np.ndarray:
    if len(devices) != rows * cols:
        raise ValueError(f"Cannot reshape {len(devices)} devices into ({rows}, {cols})")
    return np.array(devices, dtype=object).reshape(rows, cols)


def _run_matmul_probe(mesh: Mesh, axis_names: tuple[str, str], seed: int) -> float:
    sharding = NamedSharding(mesh, P(*axis_names))
    shape = mesh.devices.shape
    host_value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    host_value = host_value + np.float32(seed)

    @jax.jit
    def compute(x: jax.Array) -> jax.Array:
        y = x @ x.T
        return jnp.sum(jnp.tanh(y))

    with jax.set_mesh(mesh):
        x = jax.device_put(host_value, sharding)
        result = compute(x)
    return float(jax.device_get(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--train-processes", type=int, default=6)
    args = parser.parse_args()

    started = time.time()
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )

    process_index = jax.process_index()
    worker_process_pairs = np.array(
        jax.device_get(
            multihost_utils.process_allgather(
                np.array([args.process_id, process_index], dtype=np.int32),
                tiled=False,
            )
        )
    )
    worker_to_process = {int(worker): int(process) for worker, process in worker_process_pairs.tolist()}
    train_process_indices = {worker_to_process[worker] for worker in range(args.train_processes)}
    infer_process_indices = {
        worker_to_process[worker] for worker in range(args.train_processes, args.num_processes)
    }

    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    local_devices = sorted(jax.local_devices(), key=lambda d: d.id)
    chips_per_process = len(local_devices)
    infer_processes = args.num_processes - args.train_processes
    if args.train_processes <= 0 or infer_processes <= 0:
        raise ValueError("Both train and inference process counts must be positive")

    train_devices = [d for d in devices if d.process_index in train_process_indices]
    infer_devices = [d for d in devices if d.process_index in infer_process_indices]
    train_mesh = Mesh(
        _device_array(train_devices, args.train_processes, chips_per_process),
        ("fsdp", "tp"),
    )
    infer_mesh = Mesh(
        _device_array(infer_devices, infer_processes, chips_per_process),
        ("replica", "tp"),
    )

    if args.process_id < args.train_processes:
        role = "train"
        result = _run_matmul_probe(train_mesh, ("fsdp", "tp"), process_index)
        active_mesh_shape = train_mesh.devices.shape
    else:
        role = "infer"
        result = _run_matmul_probe(infer_mesh, ("replica", "tp"), process_index)
        active_mesh_shape = infer_mesh.devices.shape

    from skyrl.backends.jax import JaxBackendConfig, _make_backend_mesh

    skyrl_mesh = _make_backend_mesh(
        JaxBackendConfig(
            tensor_parallel_size=chips_per_process,
            expert_parallel_size=1,
            fully_sharded_data_parallel_size=args.train_processes,
            active_worker_ids=list(range(args.train_processes)),
            mesh_worker_ids=list(range(args.train_processes)),
        ),
        args.process_id,
    )

    multihost_utils.sync_global_devices("subset_mesh_probe_done")
    elapsed = time.time() - started

    payload = {
        "role": role,
        "process_id_arg": args.process_id,
        "process_index": process_index,
        "worker_to_process": worker_to_process,
        "process_count": jax.process_count(),
        "host": os.uname().nodename,
        "global_device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "local_device_ids": [d.id for d in local_devices],
        "train_mesh_shape": list(train_mesh.devices.shape),
        "infer_mesh_shape": list(infer_mesh.devices.shape),
        "skyrl_backend_mesh_shape": list(skyrl_mesh.devices.shape),
        "active_mesh_shape": list(active_mesh_shape),
        "result": result,
        "elapsed_sec": round(elapsed, 3),
    }
    print("SUBSET_MESH_PROBE_RESULT " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
