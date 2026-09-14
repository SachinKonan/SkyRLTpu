"""SkyPilot runs this Python entrypoint on every host; no SSH or tmux."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .config import Config


def workload_resources(config, rank):
    if config.arena_grader_rank == rank:
        return {"TPU": 4, "arena_grader": 4, "arena_pregate": 2}
    if rank in config.placement_ranks:
        from tpu.science.placement_slots import chips_from_env, host_resources
        return host_resources(chips_from_env(config.client_env))
    if config.inference_only_ranks is not None and rank not in config.inference_only_ranks:
        # An omitted TPU key enables Ray's automatic hardware detection.
        return {"TPU": 0}
    return {"TPU": 4}


def check_ports_available(ports, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            for port in ports:
                with socket.socket() as sock:
                    # A stopped HTTP server can leave TIME_WAIT sockets. This
                    # does not permit binding over an active listening server.
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("0.0.0.0", port))
            return
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"TCP port {port} is unavailable after {timeout}s") from exc
            time.sleep(1)


def workload_ports(config):
    values = asdict(config.ports)
    lo, hi = values.pop("worker_min"), values.pop("worker_max")
    ports = set(values.values()) | set(range(lo, hi + 1))
    for slot in range(config.engines_per_host):
        ports.update((config.ports.engine + slot, config.ports.inference_tpu + slot))
    return sorted(ports)


def validate_port_ranges(ports, ephemeral, other_worker_ranges):
    for label, (lo, hi) in [("Linux ephemeral", ephemeral), *other_worker_ranges]:
        overlap = next((port for port in ports if lo <= port <= hi), None)
        if overlap is not None:
            raise RuntimeError(f"Workload port {overlap} overlaps {label} range {lo}-{hi}")


def check_port_isolation(ports):
    import psutil
    ephemeral = tuple(map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()))
    ranges = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        args = process.info["cmdline"] or []
        if not args or Path(args[0]).name != "raylet":
            continue
        options = dict(arg[2:].split("=", 1) for arg in args[1:]
                       if arg.startswith("--") and "=" in arg)
        lo = int(options.get("min_worker_port", "0"))
        hi = int(options.get("max_worker_port", "0"))
        if lo and hi:
            ranges.append((f"Ray PID {process.pid} workers", (lo, hi)))
        # Some Ray deployments use a list instead of a contiguous range.
        for port in options.get("worker_port_list", "").split(","):
            if port:
                ranges.append((f"Ray PID {process.pid} worker", (int(port), int(port))))
    validate_port_ranges(ports, ephemeral, ranges)


def stop_ray(ray_tmp):
    import psutil
    selected = {}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        if any(str(ray_tmp) + "/" in arg or arg == str(ray_tmp) for arg in process.info["cmdline"] or []):
            try:
                selected[process.pid] = process
                selected.update({child.pid: child for child in process.children(recursive=True)})
            except psutil.NoSuchProcess:
                pass
    for process in selected.values():
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + 20
    while selected and time.monotonic() < deadline:
        for pid, process in list(selected.items()):
            try:
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    selected.pop(pid)
            except psutil.NoSuchProcess:
                selected.pop(pid)
        time.sleep(0.1)
    for process in selected.values():
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def retire_before_port_check(config, ip, log):
    from .events import emit
    from .retired import retire_workloads
    if config.retired_task_ids or config.retired_processes:
        pids = retire_workloads(config.retired_task_ids, config.retired_processes.get(ip))
        emit(log, "retired_workloads_stopped", ip=ip, pids=pids)
    ports = workload_ports(config)
    check_port_isolation(ports)
    check_ports_available(ports)
    emit(log, "port_preflight_passed", ip=ip, checked=len(ports),
         worker_min=config.ports.worker_min, worker_max=config.ports.worker_max)


def ensure_controller_runtime(root, initial_behavior=False):
    """Reuse or finish an environment while the caller holds owner.lock."""
    runtime = root / "envs/controller"
    python = runtime / "bin/python"
    marker = runtime / ".complete"
    identity = "python3.12-ray2.58-serve-grading-v5-cvxpy-tqdm-sympy-matplotlib"
    # Ray payload graders run in this environment, not the client's venv.
    check = [str(python), "-c",
             "import ray.serve,httpx,google_crc32c,zstandard,numpy,scipy,shapely,numba,sklearn,cvxpy,tqdm,sympy,matplotlib; "
             "solvers = cvxpy.installed_solvers(); "
             "assert solvers, 'CVXPY has no installed solvers'; "
             "print('Grader CVXPY', cvxpy.__version__, 'available solvers:', solvers)"]
    if runtime.resolve() != runtime.absolute():
        raise RuntimeError("controller environment must not redirect outside its owned path")
    if python.exists() and marker.exists() and marker.read_text() == identity:
        subprocess.run(check, check=True, cwd=root)
        return python
    if not python.exists():
        env = dict(os.environ)
        env.pop("UV_VENV_CLEAR", None)
        subprocess.run(["uv", "venv", "--allow-existing", "--python", "3.12", str(runtime)],
                       check=True, cwd=root, env=env)
    # Never clear a warm environment just because an optional dependency was added.
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 12)"],
                   check=True, cwd=root)
    packages = ["ray[serve]==2.58.0", "psutil", "httpx", "jinja2", "google-crc32c",
                "zstandard==0.25.0", "numpy", "scipy", "shapely", "numba", "scikit-learn",
                "cvxpy==1.9.2", "tqdm==4.70.0", "sympy==1.14.0", "matplotlib"]
    subprocess.run(["uv", "pip", "install", "--python", str(python), *packages], check=True, cwd=root)
    subprocess.run(check, check=True, cwd=root)
    marker.write_text(identity)
    return python

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--runtime-ready", action="store_true")
    args = parser.parse_args()
    config = Config.load(args.config)
    ips = os.environ["SKYPILOT_NODE_IPS"].split()
    rank = int(os.environ["SKYPILOT_NODE_RANK"])
    if len(ips) != config.hosts or len(set(ips)) != len(ips) or not 0 <= rank < config.hosts:
        raise SystemExit("invalid SkyPilot host inventory")
    root = Path(config.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.runtime_ready:
        lock_fd = int(os.environ["SKYRL_RAY_LOCK_FD"])
        os.fstat(lock_fd)
    else:
        lock_fd = os.open(root / "owner.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(lock_fd, True)
        os.environ["SKYRL_RAY_LOCK_FD"] = str(lock_fd)
    runtime = root / "envs/controller"
    python = runtime / "bin/python"
    if not args.runtime_ready and config.frozen_benchmark:
        import shutil
        if shutil.disk_usage(root).free < 30 * 1024**3:
            raise RuntimeError("frozen benchmark requires 30 GiB free disk; no automatic cache deletion")
        mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())}
        if mem["MemAvailable"] < (config.cache.inference_gib + config.cache.reserve_gib) * 1024**2:
            raise RuntimeError("insufficient RAM headroom for cache and runtime")
        devices = [str(p) for p in Path("/dev/vfio").glob("*") if p.name != "vfio"]
        devices += [str(p) for p in Path("/dev").glob("accel*")]
        if not devices:
            raise RuntimeError("TPU device inventory is empty")
        audit = subprocess.run(["sudo", "-n", "fuser", *devices], capture_output=True, text=True)
        if audit.returncode != 1 or audit.stdout.strip() or audit.stderr.strip():
            raise RuntimeError("TPU device audit failed or found existing owners: " + audit.stdout.strip() + audit.stderr.strip())
    if not args.runtime_ready:
        python = ensure_controller_runtime(root, bool(config.frozen_benchmark))
        os.execv(str(python), [str(python), "-m", __package__ + ".bootstrap", args.config, "--runtime-ready"])
    import ray
    from .events import emit
    from .process import Process
    p = config.ports
    # Short and stable across job attempts, well under the Unix socket limit.
    ray_tmp = Path.home() / (".rtr-" + hashlib.sha256(str(root).encode()).hexdigest()[:10])
    run = root / "runs" / config.run_id
    run.mkdir(parents=True, exist_ok=True)
    log = run / f"bootstrap-{rank}.jsonl"
    stopped = False

    def signal_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    driver = None
    code = 1
    from .checkpoint_retention import acquire_run_lease
    run_lease = acquire_run_lease(config, root)
    try:
        stop_ray(ray_tmp)
        retire_before_port_check(config, ips[rank], log)
        os.environ.update(RAY_ADDRESS=f"{ips[0]}:{p.ray}", RAY_NAMESPACE=config.run_id,
            RAY_TMPDIR=str(ray_tmp), JAX_PLATFORMS="cpu", TPU_VISIBLE_CHIPS="0,1,2,3",
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAY_USAGE_STATS_ENABLED="0")
        if rank in config.placement_ranks:
            from tpu.science.placement_slots import chips_from_env
            os.environ['TPU_VISIBLE_CHIPS'] = ','.join(map(str, chips_from_env(config.client_env)))
        command = [str(runtime / "bin/ray"), "start", f"--node-ip-address={ips[rank]}",
            "--num-cpus=32", "--resources=" + json.dumps(workload_resources(config, rank)), "--object-store-memory=1073741824",
            f"--object-manager-port={p.object_manager}", f"--node-manager-port={p.node_manager}",
            f"--dashboard-agent-listen-port={p.dashboard_agent}", f"--dashboard-agent-grpc-port={p.dashboard_agent_grpc}",
            f"--runtime-env-agent-port={p.runtime_env}", f"--metrics-export-port={p.metrics}",
            f"--min-worker-port={p.worker_min}", f"--max-worker-port={p.worker_max}", "--disable-usage-stats"]
        if rank == 0:
            command += ["--head", f"--port={p.ray}", f"--dashboard-port={p.dashboard}",
                        f"--ray-client-server-port={p.client}", f"--temp-dir={ray_tmp}"]
        else:
            command += [f"--address={ips[0]}:{p.ray}"]
            deadline = time.monotonic() + config.setup_timeout
            while time.monotonic() < deadline and not stopped:
                try:
                    with socket.create_connection((ips[0], p.ray), timeout=3):
                        break
                except OSError:
                    time.sleep(3)
        subprocess.run(command, check=True, timeout=180)
        ray.init(address=f"{ips[0]}:{p.ray}", namespace=config.run_id)
        emit(log, "ray_node_ready", rank=rank, address=f"{ips[0]}:{p.ray}")
        if rank == 0:
            driver = Process([str(python), "-m", __package__ + ".controller", args.config],
                             run / "driver.log", term_grace=900)
        missing_since = None
        terminal_since = None
        stop_since = None
        while True:
            try:
                status = ray.get_actor("runtime-status", namespace=config.run_id)
                if stopped:
                    stop_since = stop_since or time.monotonic()
                    ray.get(status.request_stop.remote(), timeout=5)
                    if time.monotonic() - stop_since > 900:
                        raise TimeoutError("controller shutdown exceeded 900 seconds")
                state = ray.get(status.read.remote(rank), timeout=10)
                if state["terminal"]:
                    code = state["exit_code"]
                    terminal_since = terminal_since or time.monotonic()
                    if rank != 0 or len(state["acknowledged"]) == config.hosts or time.monotonic() - terminal_since > 60:
                        break
                missing_since = None
                if driver and driver.poll() is not None and not state["terminal"]:
                    raise RuntimeError("controller exited without publishing a terminal state")
            except Exception as exc:
                missing_since = missing_since or time.monotonic()
                emit(log, "controller_poll_retry", rank=rank, detail=str(exc))
                if (stop_since and time.monotonic() - stop_since > 900) or time.monotonic() - missing_since > 180:
                    raise RuntimeError("controller unavailable for 180 seconds") from exc
            time.sleep(5)
        if stopped:
            code = 143
    finally:
        if driver:
            driver.stop()
        if ray.is_initialized():
            ray.shutdown()
        stop_ray(ray_tmp)
        emit(log, "bootstrap_stopped", rank=rank, exit_code=code)
        os.close(run_lease)
        os.close(lock_fd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
