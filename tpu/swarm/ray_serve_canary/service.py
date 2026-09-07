"""Ray Serve routing/recovery plus direct uploads to the existing TPU loader."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import httpx
import ray
from ray import serve
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from common import VersionGate, adapter_name

ROOT = Path(os.environ["CANARY_ROOT"])
HEAD = os.environ["CANARY_HEAD_IP"]
PORT = 18001
BASE = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-27B")


@ray.remote(num_cpus=0)
class Catalog:
    def __init__(self):
        self.replicas = {}
        self.version = None

    def register(self, ip, instance):
        self.replicas[ip] = {"ip": ip, "instance": instance, "registered": time.time()}

    def commit(self, version):
        self.version = version

    def snapshot(self):
        return {"version": self.version, "replicas": list(self.replicas.values())}


@serve.deployment(num_replicas=2, max_ongoing_requests=16,
                  health_check_period_s=10, health_check_timeout_s=10,
                  ray_actor_options={"num_cpus": 8, "resources": {"TPU": 4, "canary_infer": 1}})
class EngineReplica:
    async def __init__(self):
        self.ip = ray.util.get_node_ip_address()
        self.instance = uuid.uuid4().hex
        self.catalog = ray.get_actor("canary_catalog", namespace="ray-serve-canary")
        self.http = httpx.AsyncClient(timeout=1800)
        self.lock = asyncio.Lock()
        self.version = None
        env = dict(os.environ)
        env.update(TPU_PROCESS_BOUNDS="1,1,1", TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1",
                   TPU_PROCESS_ADDRESSES="localhost:18476", TPU_PROCESS_PORT="18476",
                   CLOUD_TPU_TASK_ID="0", TPU_VISIBLE_CHIPS="0,1,2,3",
                   TPU_BACKEND_TYPE="torchax", MODEL_IMPL_TYPE="vllm",
                   SKIP_JAX_PRECOMPILE="1", USE_BATCHED_RPA_KERNEL="1",
                   USE_JAX_RAGGED_CONV1D="1", VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
                   VLLM_WORKER_MULTIPROC_METHOD="spawn", HF_HUB_OFFLINE="1",
                   VLLM_XLA_CACHE_PATH=str(ROOT / "xla"), JAX_PLATFORMS="tpu,cpu",
                   VLLM_PLUGINS="lora_filesystem_resolver",
                   VLLM_LORA_RESOLVER_CACHE_DIR=str(ROOT / "loras"))
        for name in ("TPU_MULTIHOST_BACKEND", "TPU_MULTIPROCESS_DP", "JAX_COORDINATOR_ADDRESS"):
            env.pop(name, None)
        (ROOT / "loras").mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / "source/tpu/vllm_tpu_server.py"),
               str(ROOT / "model"), "--served-model-name", BASE,
               "--skyrl-lora-dir", str(ROOT / "loras"), "--host", "0.0.0.0", "--port", str(PORT),
               "--tensor-parallel-size", "4", "--max-model-len", "22528",
               "--max-num-seqs", "8", "--max-num-batched-tokens", "4096",
               "--enable-chunked-prefill", "--gpu-memory-utilization", "0.85",
               "--limit-mm-per-prompt", '{"image":0,"video":0}',
               "--enable-lora", "--max-loras", "1", "--max-lora-rank", "32"]
        self.log = (ROOT / f"engine-{self.instance}.log").open("ab", buffering=0)
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "code/engine_guard.py"),
                                      str(os.getpid()), *cmd], env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(1800):
            if self.proc.poll() is not None:
                raise RuntimeError(f"engine exited: {self.proc.returncode}; see {self.log.name}")
            try:
                r = await self.http.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=3)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("engine startup timed out")
        state = await self.catalog.snapshot.remote()
        if state["version"]:
            await self.ensure_adapter(state["version"])
        await self.catalog.register.remote(self.ip, self.instance)

    async def ensure_adapter(self, version):
        async with self.lock:
            if self.version == version:
                return
            models = await self.http.get(f"http://127.0.0.1:{PORT}/v1/models")
            models.raise_for_status()
            if version in {m["id"] for m in models.json()["data"]}:
                self.version = version
                return
            # A replacement host can fetch the retained client upload from the
            # head, without making GCS part of the adapter update path.
            archive = ROOT / f"{adapter_name(version)}.tar"
            async with self.http.stream("GET", f"http://{HEAD}:18000/adapters/{version}/archive") as r:
                r.raise_for_status()
                with archive.open("wb") as f:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
            async def chunks():
                with archive.open("rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk
            r = await self.http.post(f"http://127.0.0.1:{PORT}/skyrl/v1/upload_lora_adapter",
                                     params={"lora_name": version}, content=chunks())
            r.raise_for_status()
            archive.unlink()
            self.version = version

    async def generate(self, payload):
        await self.ensure_adapter(payload["model"])
        r = await self.http.post(f"http://127.0.0.1:{PORT}/v1/completions", json=payload)
        r.raise_for_status()
        result = r.json()
        result["canary_replica"] = self.ip
        result["canary_instance"] = self.instance
        return result

    async def fail_engine(self):
        self.proc.send_signal(signal.SIGTERM)
        return {"ip": self.ip, "instance": self.instance}

    async def check_health(self):
        if self.proc.poll() is not None:
            raise RuntimeError("canary engine process exited")
        r = await self.http.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
        r.raise_for_status()

    def __del__(self):
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            self.proc.terminate()


app = FastAPI()


@serve.deployment(num_replicas=1, max_ongoing_requests=128,
                  ray_actor_options={"num_cpus": 1, "resources": {"canary_head": 1}})
@serve.ingress(app)
class Ingress:
    def __init__(self, engines):
        self.engines = engines
        self.catalog = ray.get_actor("canary_catalog", namespace="ray-serve-canary")
        self.gate = VersionGate()
        self.upload_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=1800)
        self.archives = ROOT / "uploads"
        self.archives.mkdir(exist_ok=True)

    @app.get("/status")
    async def status(self):
        state = await self.catalog.snapshot.remote()
        state.update(active=self.gate.active, updating=self.gate.updating,
                     committed=self.gate.version)
        return state

    @app.get("/adapters/{version}/archive")
    def archive(self, version: str):
        try:
            path = self.archives / f"{adapter_name(version)}.tar"
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not path.exists():
            raise HTTPException(404, "unknown adapter")
        return FileResponse(path)

    @app.post("/adapters/{version}")
    async def upload(self, version: str, request: Request):
        try:
            adapter_name(version)
        except ValueError as e:
            raise HTTPException(400, str(e))
        async with self.upload_lock:
            target = self.archives / f"{version}.tar"
            if target.exists():
                raise HTTPException(409, "adapter versions are immutable; use a new name")
            staging = target.with_suffix(".partial")
            digest = hashlib.sha256()
            size = 0
            try:
                with staging.open("wb") as f:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > 2 * 1024**3:
                            raise HTTPException(413, "adapter too large")
                        digest.update(chunk)
                        f.write(chunk)
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
            await self.gate.begin_update()
            state = await self.catalog.snapshot.remote()
            if len(state["replicas"]) != 2:
                raise HTTPException(503, "need two registered replicas; generation remains blocked")
            async def load(replica):
                async def chunks():
                    with target.open("rb") as f:
                        while chunk := f.read(1024 * 1024):
                            yield chunk
                r = await self.http.post(f"http://{replica['ip']}:{PORT}/skyrl/v1/upload_lora_adapter",
                                         params={"lora_name": version}, content=chunks())
                r.raise_for_status()
                return replica["ip"]
            loaded = await asyncio.gather(*(load(r) for r in state["replicas"]))
            await self.catalog.commit.remote(version)
            await self.gate.commit(version)
            return {"version": version, "sha256": digest.hexdigest(), "bytes": size, "loaded": loaded}

    @app.post("/v1/completions")
    async def generate(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(400, "canary uses complete generations, not token streaming")
        try:
            await self.gate.enter(payload.get("model"))
        except ValueError as e:
            raise HTTPException(409, str(e))
        try:
            return await self.engines.generate.remote(payload)
        finally:
            await self.gate.leave()

    @app.post("/canary/fail-engine")
    async def fail_engine(self):
        return await self.engines.fail_engine.remote()


def main():
    ray.init(address=os.environ["RAY_ADDRESS"], namespace="ray-serve-canary")
    Catalog.options(name="canary_catalog", lifetime="detached").remote()
    serve.start(proxy_location="HeadOnly", http_options={"host": "0.0.0.0", "port": 18000})
    serve.run(Ingress.bind(EngineReplica.bind()), name="direct-lora-canary", blocking=True)


if __name__ == "__main__":
    main()
