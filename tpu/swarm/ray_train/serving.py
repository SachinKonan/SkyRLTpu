"""One Ray Serve endpoint over independent, role-fenced TPU engines."""
# Keep concrete Request annotations: Ray rewrites the FastAPI method signatures.

import asyncio
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
import httpx
import ray
from ray import serve

from .commands import inference_command, inference_environment
from .config import Config
from .events import emit
from .process import Process


def adapter_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("invalid adapter name")
    return name


@ray.remote(num_cpus=0)
class Catalog:
    def __init__(self, expected_ips, restart_limit):
        self.expected = set(expected_ips)
        self.replicas = {}
        self.version = None
        self.starts = {}
        self.restart_limit = restart_limit

    def claim(self, ip):
        if ip not in self.expected:
            raise RuntimeError("refusing to initialize inference on a trainer host")
        self.starts[ip] = self.starts.get(ip, 0) + 1
        self.replicas.pop(ip, None)
        if self.starts[ip] > self.restart_limit + 1:
            raise RuntimeError(f"inference restart budget exhausted on {ip}")

    def register(self, ip, instance):
        self.replicas[ip] = dict(ip=ip, instance=instance, registered=time.time())

    def commit(self, version):
        self.version = version

    def snapshot(self):
        return dict(version=self.version, replicas=list(self.replicas.values()), starts=self.starts,
                    exhausted=[ip for ip, count in self.starts.items() if count > self.restart_limit+1])


@serve.deployment(num_replicas=1, max_ongoing_requests=32,
                  health_check_period_s=10, health_check_timeout_s=10,
                  ray_actor_options={"num_cpus": 8, "resources": {"TPU": 4}})
class Engine:
    async def __init__(self, raw_config, prepared, catalog, head):
        self.config = Config.from_dict(raw_config)
        self.ip = ray.util.get_node_ip_address()
        self.catalog = catalog
        # Training actors reserve their four TPU resources before Serve starts.
        # This second check fails closed even if an unexpected placement occurs.
        if self.ip not in prepared or prepared[self.ip]["role"] != "inference":
            raise RuntimeError("inference was scheduled outside its designated role")
        await catalog.claim.remote(self.ip)
        info = prepared[self.ip]
        self.root, self.source = Path(info["root"]), Path(info["source"])
        self.run = self.root / "runs" / self.config.run_id
        (self.run / "loras").mkdir(exist_ok=True)
        self.instance = uuid.uuid4().hex
        self.version = None
        self.lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=self.config.inference.request_timeout)
        self.head = f"http://{head}:{self.config.ports.inference}"
        self.url = f"http://127.0.0.1:{self.config.ports.engine}"
        self.process = Process(inference_command(self.config, self.root, self.source, Path(info["snapshot"]), self.run),
            self.run / f"engine-{self.instance}.log", inference_environment(self.config, self.root, self.run), self.source)
        deadline = time.monotonic() + self.config.ready_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited {self.process.poll()}; see {self.process.log}")
            try:
                result = await self.http.get(self.url + "/v1/models", timeout=5)
                if result.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("vLLM startup deadline exceeded")
        state = await catalog.snapshot.remote()
        if state["version"]:
            await self.ensure_adapter(state["version"])
        await catalog.register.remote(self.ip, self.instance)
        emit(self.run / "inference-events.jsonl", "engine_ready", ip=self.ip, instance=self.instance)

    async def ensure_adapter(self, version):
        if version == self.config.model:
            return
        adapter_name(version)
        async with self.lock:
            if self.version == version:
                return
            models = await self.http.get(self.url + "/v1/models")
            models.raise_for_status()
            if version not in {m["id"] for m in models.json()["data"]}:
                archive = self.run / f"{version}.reload.tar"
                async with self.http.stream("GET", self.head + f"/adapters/{version}/archive") as response:
                    response.raise_for_status()
                    with archive.open("wb") as output:
                        async for chunk in response.aiter_bytes():
                            output.write(chunk)
                async def chunks():
                    with archive.open("rb") as stream:
                        while chunk := stream.read(1024**2):
                            yield chunk
                response = await self.http.post(self.url + "/skyrl/v1/upload_lora_adapter",
                    params={"lora_name": version}, content=chunks())
                response.raise_for_status()
                archive.unlink()
            self.version = version

    async def generate(self, payload):
        await self.ensure_adapter(payload["model"])
        response = await self.http.post(self.url + "/v1/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def check_health(self):
        if self.process.poll() is not None:
            raise RuntimeError("vLLM subprocess died")
        response = await self.http.get(self.url + "/health", timeout=5)
        response.raise_for_status()

    def __del__(self):
        if hasattr(self, "process"):
            self.process.stop()


app = FastAPI()


@serve.deployment(num_replicas=1, max_ongoing_requests=1024, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class Ingress:
    def __init__(self, raw_config, engines, catalog, inference_ips):
        self.config = Config.from_dict(raw_config)
        self.engines, self.catalog = engines, catalog
        self.ips = inference_ips
        self.run = Path(self.config.root).expanduser() / "runs" / self.config.run_id
        self.archives = self.run / "uploads"
        self.archives.mkdir(parents=True, exist_ok=True)
        self.version = None
        self.active = 0
        self.updating = False
        self.condition = asyncio.Condition()
        self.upload_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=self.config.inference.request_timeout)

    @app.get("/status")
    async def status(self):
        result = await self.catalog.snapshot.remote()
        result.update(active=self.active, updating=self.updating, committed=self.version)
        return result

    @app.get("/health")
    async def health(self):
        state = await self.catalog.snapshot.remote()
        if len(state["replicas"]) != len(self.ips) or state["exhausted"]:
            raise HTTPException(503, "inference replicas not ready")
        async def probe(ip):
            result = await self.http.get(f"http://{ip}:{self.config.ports.engine}/health", timeout=5)
            result.raise_for_status()
        results = await asyncio.gather(*(probe(ip) for ip in self.ips), return_exceptions=True)
        if self.updating or any(isinstance(result, Exception) for result in results):
            raise HTTPException(503, "inference replica unavailable or adapter update in progress")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(self):
        names = [self.config.model] + ([self.version] if self.version else [])
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "skyrl"} for n in names]}

    @app.get("/adapters/{version}/archive")
    async def archive(self, version: str):
        try:
            path = self.archives / (adapter_name(version) + ".tar")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not path.exists():
            raise HTTPException(404, "unknown adapter")
        return FileResponse(path)

    @app.post("/skyrl/v1/upload_lora_adapter")
    async def upload(self, request: Request, lora_name: str, previous_lora_name: str | None = None):
        try:
            version = adapter_name(lora_name)
            if previous_lora_name:
                adapter_name(previous_lora_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        async with self.upload_lock:
            target = self.archives / (version + ".tar")
            stage = target.with_suffix(".partial")
            digest = hashlib.sha256()
            size = 0
            try:
                with stage.open("wb") as output:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > 2 * 1024**3:
                            raise HTTPException(413, "adapter exceeds 2 GiB limit")
                        output.write(chunk)
                        digest.update(chunk)
                identity = digest.hexdigest()
                if target.exists():
                    with target.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != identity:
                            raise HTTPException(409, "adapter version is immutable")
                else:
                    stage.replace(target)
            finally:
                stage.unlink(missing_ok=True)
            async with self.condition:
                self.updating = True
                await self.condition.wait_for(lambda: self.active == 0)
            started = time.monotonic()

            async def load(ip):
                deadline = time.monotonic() + self.config.ready_timeout
                while True:
                    async def chunks():
                        with target.open("rb") as stream:
                            while chunk := stream.read(1024**2):
                                yield chunk
                    try:
                        params = {"lora_name": version}
                        if previous_lora_name:
                            params["previous_lora_name"] = previous_lora_name
                        result = await self.http.post(f"http://{ip}:{self.config.ports.engine}/skyrl/v1/upload_lora_adapter",
                                                      params=params, content=chunks(),
                                                      timeout=min(300, max(1, deadline-time.monotonic())))
                        result.raise_for_status()
                        return ip
                    except httpx.HTTPError:
                        if time.monotonic() > deadline:
                            raise
                        await asyncio.sleep(5)

            # Failure leaves admission closed, never a mixed adapter version.
            loaded = await asyncio.gather(*(load(ip) for ip in self.ips), return_exceptions=True)
            if any(isinstance(result, Exception) for result in loaded):
                raise HTTPException(503, "adapter fanout failed; generation remains paused until retry")
            await self.catalog.commit.remote(version)
            async with self.condition:
                self.version, self.updating = version, False
                self.condition.notify_all()
            emit(self.run / "inference-events.jsonl", "adapter_committed", version=version,
                 bytes=size, sha256=identity, hosts=loaded, load_seconds=time.monotonic()-started)
            return {"lora_name": version, "loaded": loaded, "sha256": identity}

    @app.post("/v1/completions")
    async def generate(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(400, "training endpoint expects complete generation responses")
        if payload.get("seed") is not None:
            raise HTTPException(400, "TPU backend does not support per-request seeds")
        async with self.condition:
            if self.updating or payload.get("model") not in (self.version, self.config.model):
                raise HTTPException(409, "adapter update in progress or uncommitted model")
            if not payload.get("model"):
                raise HTTPException(400, "model is required")
            self.active += 1
        try:
            return await self.engines.generate.remote(payload)
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()


def deploy(config, prepared, catalog, head):
    serving = {ip: info for ip, info in prepared.items() if info["role"] == "inference"}
    engines = Engine.options(num_replicas=len(serving)).bind(config.to_dict(), serving, catalog, head)
    ingress = Ingress.options(ray_actor_options={"num_cpus": 1, "resources": {f"node:{head}": 0.01}})
    serve.start(proxy_location="HeadOnly", http_options={"host": "0.0.0.0", "port": config.ports.inference})
    return serve.run_many([serve.RunTarget(
        target=ingress.bind(config.to_dict(), engines, catalog, sorted(serving)), name="skyrl-training")],
        wait_for_ingress_deployment_creation=False, wait_for_applications_running=False)[0]
