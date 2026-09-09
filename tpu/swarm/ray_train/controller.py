"""Cache barrier -> concurrent train/serve startup -> GRPO client -> owned cleanup."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import threading
import time

import httpx
import ray
from ray import serve
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tpu.swarm.select_v4_64_topology import select_split
from tpu.swarm.select_v6e_32_topology import candidate_splits as v6e_32_candidates
from tpu.swarm.select_v5p_32_topology import candidate_splits as v5p_32_candidates
from .config import Config
from .events import emit
from .host import Host
from .serving import Catalog, deploy


@ray.remote(num_cpus=0)
class RuntimeStatus:
    def __init__(self):
        self.result = {"terminal": False, "exit_code": None}
        self.acknowledged = set()
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def finish(self, code):
        self.result = {"terminal": True, "exit_code": code}

    def read(self, rank=None):
        if self.result["terminal"] and rank is not None:
            self.acknowledged.add(rank)
        return dict(self.result, acknowledged=sorted(self.acknowledged), stop_requested=self.stop_requested)


@ray.remote(num_cpus=8, resources={"TPU": 4}, max_restarts=0)
class TrainerRank:
    """Reserve a trainer host's devices for the entire collective lifetime."""
    def __init__(self, host, rank):
        self.host, self.rank = host, rank

    def reserved(self):
        return self.rank

    def start(self, ranks, inference_ips=None):
        return ray.get(self.host.start_trainer.remote(ranks, inference_ips))


class Controller:
    def __init__(self, config, ips):
        self.config, self.ips = config, ips
        self.root = Path(config.root).expanduser()
        self.log = self.root / "runs" / config.run_id / "controller.jsonl"
        self.hosts, self.trainers = [], []
        self.stopping = threading.Event()
        self.failure = None
        self.monitor = None
        self.sync_refs = {}
        self.last_sync = {}
        self.prepared = {}
        self.catalog = None
        self.runtime_status = None

    def report(self, event, **fields):
        return emit(self.log, event, run_id=self.config.run_id, **fields)

    def validated_block(self, candidates):
        """Probe candidate trainer blocks in order; keep the first whose hosts
        form a libtpu mesh. A host that fails the mesh check is left to serve
        (single-host init works there)."""
        errors = []
        for train_ranks, inference_ranks in candidates:
            try:
                self.checked_get([self.hosts[r].probe.remote(train_ranks, self.config.ports.topology_subset, True)
                                  for r in train_ranks], 300)
            except (RuntimeError, ray.exceptions.RayError) as exc:
                if self.failure or self.stopping.is_set():
                    raise
                detail = str(exc)[-400:]
                errors.append((train_ranks, detail))
                self.report("topology_block_rejected", train_ranks=train_ranks, detail=detail)
                continue
            return train_ranks, inference_ranks
        raise RuntimeError(f"no candidate trainer block formed a mesh: {errors}")

    def checked_get(self, refs, timeout):
        deadline = time.monotonic() + timeout
        waiting = list(refs)
        results = {}
        while waiting:
            if self.failure or self.stopping.is_set():
                raise RuntimeError(self.failure or "controller interrupted")
            if time.monotonic() > deadline:
                raise TimeoutError("distributed phase exceeded deadline")
            ready, _ = ray.wait(waiting, num_returns=1, timeout=1)
            for ref in ready:
                results[ref] = ray.get(ref)
                waiting.remove(ref)
        return [results[ref] for ref in refs]

    def writeback_tick(self):
        """One in-flight call per host/kind; a stalled host cannot block peers."""
        if self.sync_refs:
            ready, _ = ray.wait(list(self.sync_refs), num_returns=len(self.sync_refs), timeout=0)
            for ref in ready:
                rank, kind = self.sync_refs.pop(ref)
                try:
                    result = ray.get(ref)
                    self.report("writeback_complete", rank=rank, kind=kind, result=result)
                except Exception as exc:
                    self.report("writeback_retry_pending", rank=rank, kind=kind, detail=str(exc))
        if self.stopping.is_set() or not self.prepared:
            return
        active = set(self.sync_refs.values())
        now = time.monotonic()
        for rank, host in enumerate(self.hosts):
            for kind in (("compile", "run") if rank == 0 else ("compile",)):
                key = (rank, kind)
                if key in active or now - self.last_sync.get(key, float("-inf")) < self.config.cache.sync_seconds:
                    continue
                method = host.sync_compile if kind == "compile" else host.sync_run
                self.sync_refs[method.remote()] = key
                self.last_sync[key] = now

    def _monitor(self):
        while not self.stopping.is_set():
            try:
                if self.runtime_status and ray.get(self.runtime_status.read.remote(), timeout=5)["stop_requested"]:
                    self.stopping.set()
                live = {node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]}
                if missing := set(self.ips) - live:
                    self.failure = f"workload Ray nodes lost: {sorted(missing)}"
                statuses = ray.get([host.heartbeat.remote() for host in self.hosts], timeout=15)
                self.report("hosts", statuses=statuses)
                for status in statuses:
                    if status["stopped"]:
                        self.failure = f"host {status['rank']} stopped"
                    if status["processes"].get("trainer") is not None:
                        self.failure = f"trainer rank {status['rank']} exited"
                if self.catalog:
                    state = ray.get(self.catalog.snapshot.remote(), timeout=10)
                    self.report("inference_status", **state)
                    if state["exhausted"]:
                        self.failure = f"inference recovery budget exhausted: {state['exhausted']}"
            except Exception as exc:
                self.report("monitor_error", detail=str(exc))
                # A transient polling timeout does not cancel a healthy workload.
            try:
                self.writeback_tick()
            except Exception as exc:
                self.report("writeback_schedule_error", detail=str(exc))
            self.stopping.wait(self.config.log_seconds)

    def setup(self):
        deadline = time.monotonic() + self.config.setup_timeout
        while True:
            nodes = {node["NodeManagerAddress"]: node for node in ray.nodes() if node["Alive"]}
            if set(nodes) == set(self.ips):
                break
            if time.monotonic() > deadline or self.stopping.is_set():
                raise TimeoutError("workload Ray cluster did not acquire all expected hosts")
            self.report("waiting_nodes", joined=sorted(nodes), expected=self.ips)
            time.sleep(5)
        actor = ray.remote(num_cpus=0, max_concurrency=8, max_restarts=0)(Host)
        for rank, ip in enumerate(self.ips):
            self.hosts.append(actor.options(name=f"host-{rank}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip]["NodeID"], soft=False)).remote(
                    self.config.to_dict(), rank, self.ips))
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()
        self.checked_get([host.preflight.remote() for host in self.hosts],
                         self.config.checkpoint_cleanup_timeout + 420)
        if self.config.inference_only:
            inference_ranks = self.config.inference_only_ranks or list(range(self.config.hosts))
            self.checked_get([self.hosts[r].source_ready.remote() for r in inference_ranks], self.config.setup_timeout)
        else:
            self.checked_get([host.topology_ready.remote() for host in self.hosts], self.config.setup_timeout)
        if self.config.inference_only:
            train_ranks = []
        elif self.config.accelerator == "tpu-v4-64":
            full = self.checked_get([host.probe.remote(list(range(8)), self.config.ports.topology_jax)
                                     for host in self.hosts], 300)
            train_ranks, inference_ranks = select_split(full)
            self.checked_get([self.hosts[r].probe.remote(train_ranks, self.config.ports.topology_subset, True)
                              for r in train_ranks], 300)
        elif self.config.accelerator == "tpu-v6e-32":
            # Ranks follow SkyPilot's node list, not the physical worker order:
            # probe every host's chip coordinates and take the 2x2 host block
            # that contains rank 0 (asia replica 72 aborted the slice when
            # ranks 0-3 were physically scattered, 2026-09-09).
            full = self.checked_get([host.probe.remote(list(range(8)), self.config.ports.topology_jax)
                                     for host in self.hosts], 300)
            train_ranks, inference_ranks = self.validated_block(v6e_32_candidates(full))
        elif self.config.accelerator == "tpu-v5p-32" and self.config.trainer.hosts == 2:
            # v5p-32 = four 2x2 host layers stacked along z. Sky ranks are not
            # the layer order (job 482, worker 200: ranks 0/1 two layers apart,
            # "Mesh build was incomplete"), so probe and take the layer next to
            # rank 0, ordered by z for TPU_PROCESS_BOUNDS=1,1,2.
            full = self.checked_get([host.probe.remote(list(range(4)), self.config.ports.topology_jax)
                                     for host in self.hosts], 300)
            train_ranks, inference_ranks = self.validated_block(v5p_32_candidates(full))
        else:
            # Single-host trainers (legacy v5p-32 cell shape): rank 0 trains
            # and hosts the API server and client, the rest serve.
            train_ranks = list(range(self.config.trainer.hosts))
            inference_ranks = list(range(self.config.trainer.hosts, self.config.hosts))
        self.train_ranks = train_ranks
        # The trainer API (process 0) runs on the block's first host, which is
        # not necessarily the SkyPilot head (v6e blocks are x-fastest, v5p
        # pairs z-ordered); the client and readiness must talk to it.
        self.api_host = self.ips[train_ranks[0]] if train_ranks else self.ips[0]
        self.report("topology_validated", train_ranks=train_ranks, inference_ranks=inference_ranks, api_host=self.api_host)
        prepared_ranks = sorted(train_ranks + inference_ranks)
        prepared = self.checked_get([self.hosts[rank].prepare.remote("trainer" if rank in train_ranks else "inference")
                                    for rank in prepared_ranks], self.config.setup_timeout)
        self.prepared = {self.ips[rank]: info for rank, info in zip(prepared_ranks, prepared)}
        groups = self.config.engine_groups([self.ips[r] for r in inference_ranks])
        for group in groups:
            for ip in group:
                self.prepared[ip] = dict(self.prepared[ip], group=list(group))
        engine_ips = [group[0] for group in groups]
        self.report("cache_barrier_complete", hosts=len(prepared), engines=groups)
        if not self.config.inference_only:
            self.checked_get([self.hosts[0].restore_run.remote()], 600)
        for rank in train_ranks:
            self.trainers.append(TrainerRank.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[self.ips[rank]]["NodeID"], soft=False))
                .remote(self.hosts[rank], rank))
        self.checked_get([trainer.reserved.remote() for trainer in self.trainers], 120)
        self.catalog = Catalog.options(name="inference-catalog").remote(engine_ips, self.config.inference.restart_limit)
        inference_ips = engine_ips
        trainer_starts = [trainer.start.remote(train_ranks, inference_ips) for trainer in self.trainers]
        # Both services start concurrently; our readiness loop owns the deadline.
        deploy(self.config, self.prepared, self.catalog, self.ips[0])
        self.checked_get(trainer_starts, self.config.ready_timeout)
        deadline = time.monotonic() + self.config.ready_timeout
        with httpx.Client(timeout=5) as client:
            while time.monotonic() < deadline:
                if self.failure or self.stopping.is_set():
                    raise RuntimeError(self.failure or "controller interrupted")
                trainer_ready = self.config.inference_only or ray.get(
                    self.hosts[self.train_ranks[0]].trainer_ready.remote(), timeout=10)
                state = ray.get(self.catalog.snapshot.remote(), timeout=10)
                try:
                    api_ready = self.config.inference_only
                    if not self.config.inference_only:
                        response = client.get(f"http://{self.api_host}:{self.config.ports.trainer}/api/v1/get_server_capabilities")
                        api_ready = response.status_code == 200
                    inference_ready = client.get(f"http://{self.ips[0]}:{self.config.ports.inference}/health").status_code == 200
                except httpx.HTTPError:
                    api_ready = inference_ready = False
                if trainer_ready and api_ready and inference_ready and len(state["replicas"]) == self.config.inference_hosts:
                    self.report("services_ready", trainer=not self.config.inference_only, inference_replicas=len(state["replicas"]))
                    return self
                time.sleep(5)
        raise TimeoutError("trainer/inference readiness deadline exceeded")

    def run(self):
        self.setup()
        if self.config.inference_only:
            self.report("inference_only_waiting", endpoint=f"http://{self.ips[0]}:{self.config.ports.inference}")
            while not self.stopping.wait(5):
                if self.failure:
                    raise RuntimeError(self.failure)
            return 143
        self.checked_get([self.hosts[0].start_client.remote(self.api_host)], 30)
        self.report("client_started")
        while not self.stopping.wait(5):
            if self.failure:
                raise RuntimeError(self.failure)
            state = ray.get(self.hosts[0].heartbeat.remote(), timeout=15)
            code = state["processes"].get("client")
            if code is not None:
                self.report("client_finished", exit_code=code)
                return code
        return 143

    def close(self):
        self.stopping.set()
        if self.monitor:
            self.monitor.join(timeout=40)
        # Stop new client requests first, stop services, then flush caches
        # before bootstrap shuts down the private Ray runtime.
        if self.hosts:
            self.drain_phase({self.hosts[0].stop_client.remote(): 0}, "client_stop", timeout=40)
        try:
            serve.shutdown()
        except Exception as exc:
            self.report("serve_cleanup_error", detail=str(exc))
        if self.hosts:
            self.drain_phase({host.stop.remote(): rank for rank, host in enumerate(self.hosts)},
                             "host_stop", timeout=90)
            # A failed stop or upload on one host must not skip the remaining
            # hosts. Host-local locks serialize final and periodic writeback.
            self.drain_phase({host.sync_compile.remote(): rank for rank, host in enumerate(self.hosts)},
                             "final_compile_writeback", timeout=360)
            self.drain_phase({self.hosts[0].sync_run.remote(): 0}, "final_run_writeback", timeout=360)

    def drain_phase(self, refs, phase, timeout):
        deadline = time.monotonic() + timeout
        pending = dict(refs)
        while pending and time.monotonic() < deadline:
            ready, _ = ray.wait(list(pending), num_returns=1,
                               timeout=min(1, max(0, deadline - time.monotonic())))
            for ref in ready:
                rank = pending.pop(ref)
                try:
                    result = ray.get(ref)
                    self.report(phase + "_complete", rank=rank, result=result)
                except Exception as exc:
                    self.report(phase + "_error", rank=rank, detail=str(exc))
        for rank in pending.values():
            self.report(phase + "_timeout", rank=rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = Config.load(args.config)
    ips = os.environ["SKYPILOT_NODE_IPS"].split()
    if len(ips) != config.hosts:
        raise SystemExit("SkyPilot host count does not match profile")
    ray.init(address=f"{ips[0]}:{config.ports.ray}", namespace=config.run_id)
    status = RuntimeStatus.options(name="runtime-status", lifetime="detached").remote()
    controller = Controller(config, ips)
    controller.runtime_status = status
    signal.signal(signal.SIGTERM, lambda *_: controller.stopping.set())
    signal.signal(signal.SIGINT, lambda *_: controller.stopping.set())
    code = 1
    try:
        code = controller.run()
    except Exception as exc:
        controller.report("controller_failed", detail=str(exc))
        raise
    finally:
        controller.close()
        ray.get(status.finish.remote(code), timeout=10)
        ray.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
