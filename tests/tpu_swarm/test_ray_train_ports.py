from dataclasses import replace
import socket
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import bootstrap
from tpu.swarm.ray_train.config import Config, Ports


def config():
    return Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json")


def test_defaults_do_not_overlap_skypilot_or_linux():
    bootstrap.validate_port_ranges(bootstrap.workload_ports(config()),
                                   (32768, 60999), [("SkyPilot", (11002, 19999))])


@pytest.mark.parametrize("ports,label", [
    (replace(Ports(), worker_min=42000, worker_max=42999), "Linux ephemeral"),
    (replace(Ports(), ray=19679), "SkyPilot"),
    (replace(Ports(), worker_min=19000, worker_max=20999), "SkyPilot"),
])
def test_rejects_previous_collisions(ports, label):
    cfg = replace(config(), ports=ports)
    with pytest.raises(RuntimeError, match=label):
        bootstrap.validate_port_ranges(bootstrap.workload_ports(cfg),
                                       (32768, 60999), [("SkyPilot", (11002, 19999))])


def test_host_ephemeral_range_is_not_assumed():
    with pytest.raises(RuntimeError, match="Linux ephemeral"):
        bootstrap.validate_port_ranges(bootstrap.workload_ports(config()), (21000, 61000), [])


def test_reads_peer_ray_worker_configuration(monkeypatch):
    import psutil
    raylet = SimpleNamespace(pid=123, info={"cmdline": [
        "/runtime/raylet", "--min_worker_port=21000", "--max_worker_port=23000"]})
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [raylet])
    with pytest.raises(RuntimeError, match="Ray PID 123 workers"):
        bootstrap.check_port_isolation([22000])


def test_bootstrap_checks_worker_range_and_rejects_occupied_socket(monkeypatch, tmp_path):
    # Occupy a worker port, rather than a service port: the old preflight missed it.
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        cfg = replace(config(), ports=replace(Ports(), worker_min=port, worker_max=port))
        monkeypatch.setattr(bootstrap, "check_port_isolation", lambda ports: None)
        original = bootstrap.check_ports_available
        monkeypatch.setattr(bootstrap, "check_ports_available", lambda ports: original(ports, timeout=0))
        with pytest.raises(RuntimeError, match=f"TCP port {port} is unavailable"):
            bootstrap.retire_before_port_check(cfg, "127.0.0.1", tmp_path / "bootstrap.jsonl")
    original([port], timeout=0)


def test_preflight_covers_every_worker_and_tpu_service():
    cfg = config()
    checked = set(bootstrap.workload_ports(cfg))
    assert set(range(cfg.ports.worker_min, cfg.ports.worker_max + 1)) <= checked
    assert {cfg.ports.trainer_jax, cfg.ports.trainer_tpu, cfg.ports.topology_subset} <= checked
