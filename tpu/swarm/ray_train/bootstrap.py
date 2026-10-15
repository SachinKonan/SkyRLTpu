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
    families = config.grading_families
    if config.inference_only and families:
        # Farm hosts: one token per declared grading slot so Ray never packs
        # more candidates onto a host than its partition can isolate.
        tokens = {f"grading_{name}": spec["slots_per_host"] for name, spec in families.items()}
        if "placement" in families:
            tokens["placement_cpu_host"] = families["placement"]["slots_per_host"]
        return {"TPU": config.chips_per_host, **tokens}
    if config.science_task == "placement":
        if config.science_placement_backend == 'cpu':
            return {"TPU": 4, "placement_cpu_host": config.science_placement_slots_per_host}
        # Topology chooses the grading host later. Only its pinned placement
        # group consumes these tokens; trainer and inference reserve their TPUs.
        return {"TPU": 4, "placement_tpu_host": 4}
    if config.arena_grader_rank == rank:
        return {"TPU": 4, "arena_grader": 4, "arena_pregate": 2}
    if rank in config.placement_ranks:
        from tpu.science.placement_slots import chips_from_env, host_resources
        return host_resources(chips_from_env(config.client_env))
    if config.inference_only_ranks is not None and rank not in config.inference_only_ranks:
        # An omitted TPU key enables Ray's automatic hardware detection.
        return {"TPU": 0}
    if "math" in families:
        # Trainer hosts bound local AC2 grading by the same slot token.
        return {"TPU": config.chips_per_host, "grading_math": families["math"]["slots_per_host"]}
    return {"TPU": config.chips_per_host}


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
    # The supervisor already owns its listener; check its range isolation,
    # but do not try to bind that listener again from the child bootstrap.
    coordinator = [config.systemd_port] if getattr(config, "systemd_runtime", False) else []
    check_port_isolation(ports + coordinator)
    check_ports_available(ports)
    emit(log, "port_preflight_passed", ip=ip, checked=len(ports),
         worker_min=config.ports.worker_min, worker_max=config.ports.worker_max)


def bindable(ports):
    """True when every port can be bound on this host right now."""
    try:
        for port in ports:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def farm_ports(config, environ=os.environ, low=26000, high=32768):
    """(http, ray_base) from SKYRL_FARM_PORT / SKYRL_FARM_RAY_PORT_BASE (unset: the profile's ports).

    0 picks a free one: an OS-assigned HTTP port, and a free Ray block in [low, high) below the ephemeral range,
    tried in random order so farms starting together on one node rarely race. The choice is written back to
    ``environ`` so the re-executed bootstrap and its controller use the same ports."""
    import random
    http, ray_base = (int(environ[key]) if environ.get(key) not in (None, "") else None
                      for key in ("SKYRL_FARM_PORT", "SKYRL_FARM_RAY_PORT_BASE"))
    if http == 0:
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", 0))
            http = sock.getsockname()[1]
        environ["SKYRL_FARM_PORT"] = str(http)
    if ray_base == 0:
        block = 11 + config.ports.worker_max - config.ports.worker_min
        bases = list(range(low, high - block, -(-(block + 10) // 50) * 50))
        random.shuffle(bases)
        ray_base = next((b for b in bases if bindable(range(b, b + block))), None)
        if ray_base is None:
            raise RuntimeError("no free Ray port block on this host")
        environ["SKYRL_FARM_RAY_PORT_BASE"] = str(ray_base)
    return http, ray_base


def memory_limit(cgroup_root=Path("/sys/fs/cgroup")):
    """Bytes this job may use: the tightest cgroup memory.max above us (a Slurm job), else the host's RAM."""
    limits = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            path = cgroup_root / line.split(":", 2)[2].lstrip("/")
            for directory in [path, *path.parents]:
                try:
                    value = (directory / "memory.max").read_text().strip()
                except OSError:
                    continue
                if value != "max":
                    limits.append(int(value))
    if limits:
        return min(limits)
    return int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0]) * 1024


def ensure_controller_runtime(root, initial_behavior=False, runtime=None):
    """Reuse or finish an environment while the caller holds owner.lock.

    ``runtime`` is a prebuilt environment (Config.controller_env) for hosts that cannot install one."""
    runtime = Path(runtime) if runtime else root / "envs/controller"
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
    host_sized = config.grading_host_sized
    if host_sized:
        # Grading sized by this host (e.g. a della CPU worker): slots follow its CPUs and job memory.
        config = config.for_host(len(os.sched_getaffinity(0)), memory_limit())
    moved = farm_ports(config)
    if moved != (None, None):
        config = config.with_ports(*moved)  # several farms on one host (e.g. a shared Slurm node)
    ips = os.environ["SKYPILOT_NODE_IPS"].split()
    rank = int(os.environ["SKYPILOT_NODE_RANK"])
    if len(ips) != config.hosts or len(set(ips)) != len(ips) or not 0 <= rank < config.hosts:
        raise SystemExit("invalid SkyPilot host inventory")
    if config.assigned_host_ips and ips != config.assigned_host_ips:
        raise SystemExit("assigned host inventory does not match this partition")
    if config.assigned_host_ips:
        local_ips = subprocess.check_output(["hostname", "-I"], text=True).split()
        if ips[rank] not in local_ips:
            raise SystemExit("assigned host is not a local interface")
    root = Path(config.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    shared_host = bool(config.assigned_host_ips)
    if os.environ.get("SKYRL_SYSTEMD_OWNED") != "1" and not args.runtime_ready:
        # The one start path, for a fresh launch and a preemption recovery alike: stop what earlier tasks left
        # on a pool worker (a shared private host refuses instead) before any port check or Ray start.
        from .lifecycle import prepare
        run_dir = root / "runs" / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        prepare(os.environ.get("SKYPILOT_TASK_ID", ""), shared=shared_host, log=run_dir / f"lifecycle-{rank}.jsonl",
                tpu=bool(config.chips_per_host), scope=os.environ.get("SLURM_JOB_ID"))
    if config.systemd_runtime and os.environ.get("SKYRL_SYSTEMD_OWNED") != "1":
        from .runtime_service import launch
        token = os.environ.get("SKYPILOT_TASK_ID", "")
        # Task identity is shared by all hosts and distinct from unrelated jobs.
        return launch([sys.executable, "-m", __package__ + ".bootstrap", args.config],
                      root / "runs" / config.run_id, ips, rank, token,
                      port=config.systemd_port, peer_timeout=config.systemd_peer_timeout,
                      shutdown_grace=config.systemd_shutdown_grace, setup_timeout=config.setup_timeout,
                      shared=shared_host)

    if args.runtime_ready:
        lock_fd = int(os.environ["SKYRL_RAY_LOCK_FD"])
        os.fstat(lock_fd)
    else:
        lock_fd = os.open(root / "owner.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(lock_fd, True)
        os.environ["SKYRL_RAY_LOCK_FD"] = str(lock_fd)
    runtime = Path(config.controller_env) if config.controller_env else root / "envs/controller"
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
        python = ensure_controller_runtime(root, bool(config.frozen_benchmark), runtime)
        os.execv(str(python), [str(python), "-m", __package__ + ".bootstrap", args.config, "--runtime-ready"])
    import ray
    from .events import emit
    from .process import Process
    p = config.ports
    # Short and stable across job attempts, well under the Unix socket limit.
    ray_tmp = Path.home() / (".rtr-" + hashlib.sha256(str(root).encode()).hexdigest()[:10])
    if len(str(ray_tmp)) > 40:
        # Ray's unix sockets live below this folder (107-byte limit); a long HOME (SkyPilot on Slurm) uses /tmp.
        ray_tmp = Path("/tmp") / ray_tmp.name
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
        from .runtime_service import wait_preflight
        wait_preflight(config, lambda: stopped)
        os.environ.update(RAY_ADDRESS=f"{ips[0]}:{p.ray}", RAY_NAMESPACE=config.run_id,
            RAY_TMPDIR=str(ray_tmp), JAX_PLATFORMS="cpu",
            TPU_VISIBLE_CHIPS=",".join(map(str, range(config.chips_per_host))),
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAY_USAGE_STATS_ENABLED="0",
            SCIENCE_GRADING_ISOLATION=config.grading.isolation)
        if rank in config.placement_ranks:
            from tpu.science.placement_slots import chips_from_env
            os.environ['TPU_VISIBLE_CHIPS'] = ','.join(map(str, chips_from_env(config.client_env)))
        ray_cpus, ray_memory = config.ray_cpus_per_host, []
        if config.grading_pool and not config.ray_cpu_capacity:
            # Publish this host's grading core pool, then give Ray exactly that
            # many CPUs and GiB for grades on top of its service actors.
            from tpu.science import core_pool
            pool = core_pool.install(config.grading_service_cpus, config.grading_memory_gib,
                                     minimum=core_pool.SERVICE_MIN if config.engines_per_host else 1)
            ray_cpus = len(pool['grading']) + config.ray_service_cpus(rank)
            ray_memory = [f"--memory={pool['memory_gib'] * 1024 ** 3}"]
            emit(log, "grading_pool_installed", rank=rank, grading_cpus=len(pool['grading']),
                 service_cpus=pool['service'], memory_gib=pool['memory_gib'], ray_cpus=ray_cpus)
        else:
            # A pool file left by an earlier job on this VM must not steer graders.
            from tpu.science import core_pool
            removed = core_pool.uninstall()
            if removed:
                emit(log, "grading_pool_removed", rank=rank)
        command = [str(runtime / "bin/ray"), "start", f"--node-ip-address={ips[rank]}",
            f"--num-cpus={ray_cpus}", *ray_memory, "--resources=" + json.dumps(workload_resources(config, rank)), "--object-store-memory=1073741824",
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
        # ray.init installs Ray's own SIGTERM handler, which aborts this process; keep the graceful stop.
        signal.signal(signal.SIGTERM, signal_stop)
        signal.signal(signal.SIGINT, signal_stop)
        emit(log, "ray_node_ready", rank=rank, address=f"{ips[0]}:{p.ray}")
        if rank == 0:
            profile = args.config
            if host_sized or moved != (None, None):
                profile = str(run / "profile-resolved.json")  # what this host resolved, for the controller
                Path(profile).write_text(json.dumps(config.to_dict(), indent=1))
            driver = Process([str(python), "-m", __package__ + ".controller", profile],
                             run / "driver.log", term_grace=900)
        missing_since = None
        terminal_since = None
        stop_since = None
        while True:
            try:
                status = ray.get_actor("runtime-status", namespace=config.run_id)
                if stopped:
                    if stop_since is None and driver and driver.poll() is None:
                        driver.process.terminate()  # the controller stops (and drains) now, not at its next poll
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
        try:
            if driver:
                driver.stop()
        finally:
            try:
                if ray.is_initialized():
                    ray.shutdown()
            finally:
                try:
                    stop_ray(ray_tmp)
                    emit(log, "bootstrap_stopped", rank=rank, exit_code=code)
                finally:
                    os.close(run_lease)
                    os.close(lock_fd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
