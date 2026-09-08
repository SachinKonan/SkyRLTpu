"""Host-local lifecycle, invoked by pinned Ray actors instead of SSH/tmux."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import sqlite3
import threading
import time

from .cache import CacheStore, GCS, mount_cache
from .commands import client_environment, trainer_command, trainer_environment
from .config import Config
from .events import emit
from .process import Process

CLIENT_ENV = Path(__file__).with_name("client_env")


def orbax_marker_present(gcs, orbax):
    """True when the orbax tree carries the top-level CHECKPOINT_COMPLETE marker.

    GCS.list() enumerates a prefix (it appends "/**"), so probing the marker's
    own path finds nothing: the marker is an object, not a directory. List the
    checkpoint tree once and look for the marker among its objects instead.
    """
    objects = gcs.list(orbax, allow_empty=True)
    return any(o.relative == "CHECKPOINT_COMPLETE" for o in objects)


class Host:
    def __init__(self, config, rank, ips):
        self.config = Config.from_dict(config)
        self.rank, self.ips = rank, ips
        self.root = Path(self.config.root).expanduser().resolve()
        self.run = self.root / "runs" / self.config.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.source = self.root / "sources" / self.config.base_bundle_sha256
        self.log = self.run / f"host-{rank}.jsonl"
        self.role = None
        self.phase = "created"
        self.processes = {}
        self.lock = threading.RLock()
        self.compile_sync_lock = threading.Lock()
        self.run_sync_lock = threading.Lock()
        self.stopping = threading.Event()
        self.last_heartbeat = time.monotonic()
        self.gcs = GCS(self.run / "transfers", self.config.cache)
        self.sync_gcs = GCS(self.run / "writeback", self.config.cache)
        self.store = None
        self.last_sync = None
        self.sync_error = None
        self.watchdog = threading.Thread(target=self._watch_lease, daemon=True)
        self.watchdog.start()

    def _watch_lease(self):
        while not self.stopping.wait(5):
            if time.monotonic() - self.last_heartbeat > 180:
                emit(self.log, "controller_lease_expired", rank=self.rank)
                self.stop()
                return

    def heartbeat(self):
        self.last_heartbeat = time.monotonic()
        with self.lock:
            return dict(rank=self.rank, role=self.role, phase=self.phase,
                        processes={k: p.poll() for k, p in self.processes.items()},
                        last_cache_sync=self.last_sync, cache_sync_error=self.sync_error,
                        stopped=self.stopping.is_set())

    def start(self, name, command, env=None, cwd=None):
        with self.lock:
            if self.stopping.is_set():
                raise RuntimeError("host is stopping")
            if name in self.processes and self.processes[name].poll() is None:
                raise RuntimeError(f"owned process already active: {name}")
            process = Process(command, self.run / f"{name}.log", env=env, cwd=cwd or self.root)
            self.processes[name] = process
            emit(self.log, "process_started", rank=self.rank, process=name, pid_owned=process.process.pid)
            return process

    def checked(self, name, command, env=None, cwd=None, timeout=3600):
        process = self.start(name, command, env, cwd)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if self.stopping.is_set() or time.monotonic() > deadline:
                    raise TimeoutError(f"{name} interrupted or exceeded timeout")
                time.sleep(0.5)
            if process.poll() != 0:
                raise RuntimeError(f"{name} exited {process.poll()}; see {process.log}")
            with process.log.open("rb") as stream:
                stream.seek(process.log_offset)
                return stream.read().decode(errors="replace")
        finally:
            process.stop()

    def preflight(self):
        if self.config.retired_task_ids or self.config.retired_processes:
            from .retired import retire_workloads
            stopped = retire_workloads(self.config.retired_task_ids,
                                      self.config.retired_processes.get(self.ips[self.rank]))
            emit(self.log, "retired_workloads_stopped", rank=self.rank, pids=stopped,
                 task_ids=self.config.retired_task_ids)
        import psutil
        for process in psutil.process_iter(["pid", "cmdline"]):
            args = process.info["cmdline"] or []
            # Do not disturb detached workloads that the pool calls idle.
            if any(token in args for token in ("skyrl.tinker.api", "skyrl.tinker.engine", "skyrl.backends.rpc")):
                raise RuntimeError(f"host {self.rank} has an existing trainer PID {process.pid}")
            if any(arg.endswith("/vllm_tpu_server.py") or arg.startswith("VLLM::") for arg in args):
                raise RuntimeError(f"host {self.rank} has an existing inference PID {process.pid}")
        self.clear_previous_adapter_exports()
        from .checkpoint_retention import reclaim_checkpoints
        reclaim_checkpoints(self.root, self.config.run_id, self.gcs, self.log,
                            timeout=self.config.checkpoint_cleanup_timeout,
                            stopping=self.stopping.is_set)
        if shutil.disk_usage(self.root).free < 10 * 1024**3:
            raise RuntimeError("need 10 GiB disk for isolated code/envs; refusing unrelated cache deletion")
        self.phase = "preflight_complete"
        return self.heartbeat()

    def clear_previous_adapter_exports(self):
        # No workload is active after preflight. These are derived serving
        # artifacts, not the durable trainer checkpoints or client run state.
        targets = [self.run / name for name in ("loras", "uploads")]
        targets.extend(self.run.glob("*.reload.tar"))
        if any(path.is_symlink() for path in targets):
            raise RuntimeError("refusing adapter cleanup through a symlink")
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        emit(self.log, "previous_adapter_exports_cleared", rank=self.rank)

    def source_ready(self):
        marker = self.source / ".source-complete"
        if marker.exists() and marker.read_text() == self.config.base_bundle_sha256:
            return str(self.source)
        archive = self.root / "base-download.tar.gz"
        self.gcs.transfer(["cp", self.config.base_bundle, str(archive)], "base-code", self.root)
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != self.config.base_bundle_sha256:
            raise RuntimeError("frozen source bundle checksum mismatch")
        staging = self.source.with_name(self.source.name + "-partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="data")
        for path in ("tpu/probe_topology.py", "skyrl/backends/tunix_backend.py",
                     "skyrl/utils/checkpoint_mirror.py", "tpu/vllm_tpu_server.py"):
            if not (staging / path).is_file():
                raise RuntimeError(f"frozen worker source missing {path}")
        if self.source.exists():
            shutil.rmtree(self.source)
        staging.rename(self.source)
        marker.write_text(actual)
        archive.unlink()
        return str(self.source)

    def topology_ready(self):
        self.source_ready()
        folder = self.root / "envs/topology"
        if not (folder / ".complete").exists():
            self.checked("topology-venv", ["uv", "venv", "--python", "3.12", str(folder)])
            self.checked("topology-install", ["uv", "pip", "install", "--python", str(folder / "bin/python"),
                "jax==0.11.1", "jaxlib==0.11.1", "libtpu==0.0.46", "requests==2.32.5"])
            (folder / ".complete").touch()
        return True

    def probe(self, ranks, coordinator_port, subset=False):
        process_id = ranks.index(self.rank)
        env = dict(os.environ, JAX_PLATFORMS="tpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for key in ("TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_PROCESS_ADDRESSES",
                    "TPU_PROCESS_PORT", "CLOUD_TPU_TASK_ID", "TPU_VISIBLE_CHIPS", "JAX_COORDINATOR_ADDRESS"):
            env.pop(key, None)
        if subset:
            env.update(TPU_PROCESS_BOUNDS=self.config.trainer.process_bounds,
                TPU_CHIPS_PER_PROCESS_BOUNDS=self.config.trainer.chip_bounds,
                TPU_PROCESS_ADDRESSES=",".join(f"{self.ips[r]}:{self.config.ports.trainer_tpu}" for r in ranks),
                TPU_PROCESS_PORT=str(self.config.ports.trainer_tpu), CLOUD_TPU_TASK_ID=str(process_id))
        text = self.checked("topology-subset" if subset else "topology-full",
            [str(self.root / "envs/topology/bin/python"), str(self.source / "tpu/probe_topology.py"),
             str(process_id), f"{self.ips[ranks[0]]}:{coordinator_port}", str(len(ranks))], env=env, timeout=240)
        lines = [line.removeprefix("PROBE_RESULT ") for line in text.splitlines() if line.startswith("PROBE_RESULT ")]
        if len(lines) != 1:
            raise RuntimeError("topology probe did not produce one result")
        return json.loads(lines[0])

    # Exact versions read off a live legacy v5p-32 qwen cell (job 340, 2026-09-08):
    # trainer = tpu/tunix_runtime uv.lock + the MaxText fork + these extras;
    # engine = what `vllm-tpu==0.23.0` resolved there. Pinned so the executor's
    # environments are the legacy environments, not what resolves next week.
    TRAINER_PINS = ["aqtp==0.9.0", "pathwaysutils==0.1.11", "tokamax==0.0.13", "tiktoken==0.14.0",
                    "jax==0.11.1", "jaxlib==0.11.1", "libtpu==0.0.46", "transformers==5.8.0"]
    SERVING_PINS = ["vllm-tpu==0.23.0", "jax==0.10.1", "jaxlib==0.10.1", "libtpu==0.0.41",
                    "torch==2.10.0", "torchax==0.0.11", "tokenizers==0.22.2", "numpy==2.3.5",
                    "ray[serve]==2.58.0", "httpx", "psutil"]

    def serving_pins(self):
        pins = list(self.SERVING_PINS)
        version = self.config.inference.transformers_version
        if version:
            pins.append(f"transformers=={version}")
        return pins

    def install_role(self, role):
        folder = self.root / "envs" / ("trainer" if role == "trainer" else "serving")
        identity = self.config.base_bundle_sha256 + (
            self.config.trainer.maxtext_spec + "|" + " ".join(self.TRAINER_PINS + self.config.trainer.extra_pins)
            if role == "trainer"
            else " ".join(self.serving_pins()))
        marker = folder / ".complete"
        if marker.exists() and marker.read_text() == identity:
            if role == "trainer":
                self.checked("trainer-flce-contract", [str(folder / "bin/python"), str(Path(__file__).with_name("patch_maxtext.py"))])
            return
        if folder.exists():
            shutil.rmtree(folder)
        python = str(folder / "bin/python")
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(folder), UV_NO_CONFIG="1")
        if role == "trainer":
            self.checked("trainer-sync", ["uv", "sync", "--project", str(self.source / "tpu/tunix_runtime"),
                                          "--python", "3.12", "--frozen"], env=env)
            self.checked("trainer-source", ["uv", "pip", "install", "--python", python,
                                           "--no-deps", "--editable", str(self.source)])
            self.checked("trainer-maxtext", ["uv", "pip", "install", "--python", python,
                self.config.trainer.maxtext_spec, *self.TRAINER_PINS, *self.config.trainer.extra_pins], cwd=self.root)
            # Import what the backend imports at model creation, so a MaxText
            # fork with an unpinned transitive dependency (job 413: drjax) fails
            # here instead of in the trainer process.
            self.checked("trainer-import", [python, "-c",
                "import jax,skyrl.backends.tunix_backend; from maxtext.utils import model_creation_utils; "
                "assert jax.__version__ == '0.11.1'"],
                         env=dict(os.environ, JAX_PLATFORMS="cpu"))
            self.checked("trainer-flce-contract", [python, str(Path(__file__).with_name("patch_maxtext.py"))])
        else:
            self.checked("serving-venv", ["uv", "venv", "--python", "3.12", str(folder)])
            self.checked("serving-install", ["uv", "pip", "install", "--python", python, *self.serving_pins()])
            self.checked("serving-overlay", ["uv", "pip", "install", "--python", python, "--no-deps",
                                             "--force-reinstall", str(self.source / "third_party/tpu-inference")])
            self.checked("serving-import", [python, "-c",
                "import jax, vllm; from tpu_inference.worker.tpu_worker import TPUWorker; "
                "assert jax.__version__ == '0.10.1'; "
                "missing = [n for n in ('add_lora','remove_lora','list_loras','pin_lora') if not hasattr(TPUWorker, n)]; "
                "assert not missing, missing"], env=dict(os.environ, JAX_PLATFORMS="cpu"))
        marker.write_text(identity)

    def install_client(self):
        folder = self.root / "envs/client"
        marker = folder / ".complete"
        identity = hashlib.sha256(self.config.base_bundle_sha256.encode()
            + (CLIENT_ENV / "pyproject.toml").read_bytes()
            + (CLIENT_ENV / "uv.lock").read_bytes()
            + b"client-frozen-v1").hexdigest()
        python = str(folder / "bin/python")
        verify = [python, str(Path(__file__).with_name("client_dependencies.py")),
                  str(CLIENT_ENV / "uv.lock")]
        if marker.exists() and marker.read_text() == identity:
            self.checked("client-lock-check", verify)
            return
        if folder.exists():
            shutil.rmtree(folder)
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(folder), UV_NO_CONFIG="1")
        self.checked("client-sync", ["uv", "sync", "--project", str(CLIENT_ENV),
                                    "--python", "3.12", "--frozen"], env=env)
        # Build the frozen application source using locked build tools. Neither
        # application dependencies nor isolated build dependencies may resolve.
        self.checked("client-install", ["uv", "pip", "install", "--python", python,
            "--no-deps", "--no-build-isolation", str(self.source / "third_party/discover")])
        self.checked("client-deps-check", ["uv", "pip", "check", "--python", python])
        self.checked("client-lock-check", verify)
        self.checked("client-import", [python, "-c", "import ray,tinker,wandb,torch,ttt_discover; assert torch.version.cuda is None"])
        marker.write_text(identity)

    def prepare(self, role):
        if role not in ("trainer", "inference"):
            raise ValueError("invalid host role")
        self.role, self.phase = role, "cache_setup"
        cap = self.config.cache.trainer_gib if role == "trainer" else self.config.cache.inference_gib
        ram = mount_cache(self.root / "ram", cap, self.config.cache.reserve_gib)
        self.store = CacheStore(ram, self.gcs)
        self.store.reconcile_role(role)
        if role == "trainer":
            orbax = self.config.cache.orbax.rstrip("/") + "/" + self.config.trainer.maxtext_model
            if self.config.trainer.ckpt_require_marker and not orbax_marker_present(self.gcs, orbax):
                # gpt-oss 120B conversion contract (tpu/swarm/prepare_gptoss120b_v6e32.sh):
                # never restore a checkpoint whose upload did not finish.
                raise RuntimeError(f"orbax checkpoint {orbax} has no CHECKPOINT_COMPLETE marker")
            self.store.restore_tree(orbax, "orbax/" + self.config.trainer.maxtext_model)
        self.snapshot = self.store.restore_hf(self.config.cache.hf, self.config.model, weights=role == "inference")
        self.store.scope_compile(self.compile_prefix())
        if role == "inference" and self.config.cache.inference_compile_seed:
            self.store.restore_compile(self.config.cache.inference_compile_seed)
        self.store.restore_compile(self.compile_prefix())
        self.phase = "environment_setup"
        self.install_role(role)
        if self.rank == 0 and not self.config.inference_only:
            self.install_client()
        self.phase = "prepared"
        emit(self.log, "host_prepared", rank=self.rank, role=role, snapshot=str(self.snapshot))
        return dict(rank=self.rank, role=role, source=str(self.source), root=str(self.root), snapshot=str(self.snapshot))

    def compile_prefix(self):
        return self.config.cache.trainer_compile if self.role == "trainer" else self.config.cache.inference_compile

    def sync_compile(self):
        if not self.store:
            return None
        if not self.compile_sync_lock.acquire(timeout=330):
            raise TimeoutError("previous compilation-cache writeback is still running")
        try:
            result = CacheStore(self.root / "ram", self.sync_gcs).publish_compile(self.compile_prefix())
            self.last_sync, self.sync_error = time.time(), None
            return result
        except Exception as exc:
            self.sync_error = str(exc)
            emit(self.log, "cache_writeback_error", rank=self.rank, detail=str(exc))
            raise
        finally:
            self.compile_sync_lock.release()

    def start_trainer(self, train_ranks, inference_ips=None):
        if self.phase != "prepared" or self.role != "trainer" or self.rank not in train_ranks:
            raise RuntimeError("trainer host was not prepared for this role")
        ips = [self.ips[r] for r in train_ranks]
        process_id = train_ranks.index(self.rank)
        self.start("trainer", trainer_command(self.config, self.root, self.source, self.ips[0], ips, process_id,
                                              inference_ips),
                   trainer_environment(self.config, self.root, self.run, ips, process_id), self.source)
        self.phase = "trainer_started"
        return self.heartbeat()

    def start_client(self):
        if self.rank != 0:
            raise RuntimeError("client belongs on the head")
        (self.run / "client").mkdir(exist_ok=True)
        self.start("client", [str(self.root / "envs/client/bin/python"), str(self.source / "tpu/run_ttd_ensemble.py")],
                   client_environment(self.config, self.root, self.ips[0]), self.source)
        self.phase = "client_running"
        return self.heartbeat()

    def trainer_ready(self):
        import re
        if self.rank != 0:
            return self.processes.get("trainer") is not None and self.processes["trainer"].poll() is None
        log = self.run / "trainer.log"
        process = self.processes.get("trainer")
        if not process or process.poll() is not None or not log.exists():
            return False
        with log.open("rb") as stream:
            stream.seek(process.log_offset)
            return re.search(rb"Initialized\s+TinkerEngine\s+with\s+backend=", stream.read()) is not None

    def restore_run(self):
        if self.rank != 0:
            return
        local = self.run / "client"
        if not local.exists() and self.gcs.list(self.config.run_gcs + "/client", allow_empty=True):
            self.gcs.transfer(["cp", "--recursive", self.config.run_gcs + "/client", str(self.run)], "restore-run", self.run)
        db = self.run / "tinker.db"
        if not db.exists():
            listing = self.gcs.metadata("ls", "--json", self.config.run_gcs + "/tinker-backup.db", allow_empty=True)
            if json.loads(listing):
                self.gcs.transfer(["cp", self.config.run_gcs + "/tinker-backup.db", str(db)], "restore-database", self.run)

    def sync_run(self):
        if self.rank != 0:
            return
        if not self.run_sync_lock.acquire(timeout=330):
            raise TimeoutError("previous run-state writeback is still running")
        try:
            return self._sync_run()
        finally:
            self.run_sync_lock.release()

    def _sync_run(self):
        gcs = GCS(self.run / "run-writeback", self.config.cache)
        client = self.run / "client"
        if client.exists() and any(client.iterdir()):
            gcs.transfer(["cp", "--recursive", str(client), self.config.run_gcs + "/"], "client-writeback", timeout=300)
        db = self.run / "tinker.db"
        if db.exists():
            backup = self.run / "tinker-backup.db"
            with sqlite3.connect(db) as source, sqlite3.connect(backup) as destination:
                source.backup(destination)
            gcs.transfer(["cp", str(backup), self.config.run_gcs + "/tinker-backup.db"], "database-writeback", timeout=120)
        logs = [str(path) for pattern in ("*.jsonl", "*.log") for path in self.run.glob(pattern)]
        if logs:
            gcs.transfer(["cp", *logs, self.config.run_gcs + "/logs/"], "logs-writeback", timeout=300)

    def stop(self):
        self.stopping.set()
        with self.lock:
            processes = list(self.processes.values())
        for process in reversed(processes):
            process.stop()
        self.gcs.stop()
        self.phase = "stopped"
        return self.heartbeat()

    def stop_client(self):
        with self.lock:
            process = self.processes.get("client")
        if process:
            process.stop()
