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
        self.versions = set()
        self.starts = {}
        self.restart_limit = restart_limit

    def claim(self, ip):
        if ip not in self.expected:
            raise RuntimeError("refusing to initialize inference on a trainer host")
        self.starts[ip] = self.starts.get(ip, 0) + 1
        self.replicas.pop(ip, None)
        if self.starts[ip] > self.restart_limit + 1:
            raise RuntimeError(f"inference restart budget exhausted on {ip}")

    def register(self, ip, instance, versions=None):
        if ip not in self.expected:
            raise RuntimeError('retired inference engine cannot register')
        if versions is not None and set(versions) != self.versions:
            return False
        self.replicas[ip] = dict(ip=ip, instance=instance, registered=time.time())
        return True

    def restrict(self, keys):
        keys = set(keys)
        if not keys or not keys <= self.expected:
            raise ValueError('engine transition must retain a nonempty subset')
        self.expected = keys
        self.replicas = {k: v for k, v in self.replicas.items() if k in keys}
        self.starts = {k: v for k, v in self.starts.items() if k in keys}

    def commit(self, version, previous=None):
        self.version = version
        if previous:
            self.versions.discard(previous)
        self.versions.add(version)

    def snapshot(self):
        return dict(version=self.version, versions=sorted(self.versions), expected=sorted(self.expected), replicas=list(self.replicas.values()), starts=self.starts,
                    exhausted=[ip for ip, count in self.starts.items() if count > self.restart_limit+1])


@serve.deployment(num_replicas=1, max_ongoing_requests=32,
                  health_check_period_s=10, health_check_timeout_s=10,
                  ray_actor_options={"num_cpus": 8, "resources": {"TPU": 4}})
class Engine:
    async def __init__(self, raw_config, prepared, catalog, head, slot=0):
        self.config = Config.from_dict(raw_config)
        self.ip = ray.util.get_node_ip_address()
        self.catalog = catalog
        # Training actors reserve their four TPU resources before Serve starts.
        # This second check fails closed even if an unexpected placement occurs.
        if self.ip not in prepared or prepared[self.ip]["role"] != "inference":
            raise RuntimeError("inference was scheduled outside its designated role")
        self.slot = slot
        self.key = self.ip if self.config.engines_per_host == 1 else f"{self.ip}:{self.config.ports.engine + slot}"
        await catalog.claim.remote(self.key)
        info = prepared[self.ip]
        group = list(info.get("group") or [self.ip])
        if group[0] != self.ip:
            raise RuntimeError("engine replica must run on its group's first host")
        self.group = group
        self.root, self.source = Path(info["root"]), Path(info["source"])
        self.run = self.root / "runs" / self.config.run_id
        self.engine_run = self.run / f"engine-slot-{slot}"
        (self.engine_run / "loras").mkdir(parents=True, exist_ok=True)
        self.instance = uuid.uuid4().hex
        self.version = None
        self.lock = asyncio.Lock()
        self.retiring = False
        self.http = httpx.AsyncClient(timeout=self.config.inference.request_timeout)
        self.head = f"http://{head}:{self.config.ports.inference}"
        self.url = f"http://127.0.0.1:{self.config.ports.engine + slot}"
        command = inference_command(self.config, self.root, self.source, Path(info["snapshot"]), self.engine_run, group=self.group, slot=slot)
        environment = inference_environment(self.config, self.root, self.engine_run, head=head, group=self.group, slot=slot)
        from .launch_contract import write_launch_contract
        write_launch_contract(self.run / f"launch-inference-{slot}.json", command, environment,
                              extra_keys=self.config.inference.engine_env)
        self.process = Process(command, self.run / f"engine-{self.instance}.log", environment, self.source)
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
        while True:
            state = await catalog.snapshot.remote()
            versions = state["versions"]
            for version in versions:
                await self.ensure_adapter(version)
            if await catalog.register.remote(self.key, self.instance, versions):
                break
        emit(self.run / "inference-events.jsonl", "engine_ready", ip=self.ip, slot=slot, tp=self.config.inference.tp, instance=self.instance)

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
                archive = self.engine_run / f"{version}.reload.tar"
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
        if self.retiring:
            raise RuntimeError('engine is retired')
        await self.ensure_adapter(payload["model"])
        try:
            response = await self.http.post(self.url + "/v1/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Preserve the local engine error before Ray wraps the exception or
            # the VM disappears. Never include request headers or prompt text.
            detail = dict(ip=self.ip, instance=self.instance, error=str(exc),
                          engine_returncode=self.process.poll())
            if isinstance(exc, httpx.HTTPStatusError):
                detail.update(status_code=exc.response.status_code,
                              response_body=exc.response.text[:4096])
            try:
                with self.process.log.open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 8192))
                    detail['engine_log_tail'] = stream.read(8192).decode(errors='replace')
            except OSError as log_error:
                detail['engine_log_error'] = str(log_error)
            message = 'Inference generation failed: ' + json.dumps(detail, sort_keys=True)
            try:
                emit(self.run / 'inference-events.jsonl', 'generation_failed', **detail)
            except OSError:
                print(message, flush=True)
            raise RuntimeError(message) from exc
        return response.json()

    async def tokenize(self, payload):
        if payload.get("model") != self.config.model:
            raise ValueError("tokenization model does not match engine")
        response = await self.http.post(self.url + "/tokenize", json=payload)
        response.raise_for_status()
        return response.json()

    async def check_health(self):
        if self.retiring:
            return  # Intentional shutdown must not trigger Serve auto-recovery.
        if self.process.poll() is not None:
            raise RuntimeError("vLLM subprocess died")
        response = await self.http.get(self.url + "/health", timeout=5)
        response.raise_for_status()

    async def retire(self):
        self.retiring = True
        await asyncio.to_thread(self.process.stop)
        if self.process.poll() is None:
            raise RuntimeError('retired inference subprocess remains alive')
        return dict(key=self.key, stopped=True)

    def __del__(self):
        if hasattr(self, "process"):
            self.process.stop()


app = FastAPI()


@serve.deployment(num_replicas=1, max_ongoing_requests=1024, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class Ingress:
    def __init__(self, raw_config, engines, catalog, inference_ips, engine_models=None):
        self.config = Config.from_dict(raw_config)
        self.engines = list(engines) if isinstance(engines, (list, tuple)) else [engines]
        self.next_engine = 0
        self.next_model_engine = {}
        self.engine_models = engine_models
        self.catalog = catalog
        self.ips = inference_ips
        self.engine_urls = [f"http://{s['ip']}:{s['port']}" for s in self.config.engine_slots(inference_ips)]
        self.run = Path(self.config.root).expanduser() / "runs" / self.config.run_id
        self.archives = self.run / "uploads"
        self.archives.mkdir(parents=True, exist_ok=True)
        self.version = None
        self.versions = set()
        self.retired = set()
        self.active = 0
        self.updating = False
        self.condition = asyncio.Condition()
        self.upload_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=self.config.inference.request_timeout)

    @app.get("/status")
    async def status(self):
        result = await self.catalog.snapshot.remote()
        result.update(active=self.active, updating=self.updating, committed=self.version,
                      committed_adapters=sorted(self.versions))
        return result

    @app.get("/health")
    async def health(self):
        state = await self.catalog.snapshot.remote()
        if len(state["replicas"]) != len(self.engine_urls) or state["exhausted"]:
            raise HTTPException(503, "inference replicas not ready")
        async def probe(url):
            result = await self.http.get(url + "/health", timeout=5)
            result.raise_for_status()
        results = await asyncio.gather(*(probe(url) for url in self.engine_urls), return_exceptions=True)
        if self.updating or any(isinstance(result, Exception) for result in results):
            raise HTTPException(503, "inference replica unavailable or adapter update in progress")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(self):
        names = self.config.served_models + sorted(self.versions)
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
        if self.engine_models:
            raise HTTPException(400, "multi-model arena is inference-only")
        try:
            version = adapter_name(lora_name)
            if previous_lora_name:
                adapter_name(previous_lora_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        async with self.upload_lock:
            # Single-adapter clients historically omit the previous name.
            previous = previous_lora_name
            if self.config.inference.max_loras == 1 and version != self.version:
                previous = previous or self.version
            if version in self.retired:
                raise HTTPException(409, "retired adapter version cannot be republished")
            if previous == version:
                raise HTTPException(400, "replacement requires a new version name")
            if previous and previous not in self.versions and version not in self.versions:
                raise HTTPException(409, "previous adapter is not committed")
            if len((self.versions - {previous}) | {version}) > self.config.inference.max_loras:
                raise HTTPException(409, "adapter capacity reached; specify previous_lora_name")
            target = self.archives / (version + ".tar")
            stage = target.with_suffix(".partial")
            digest = hashlib.sha256()
            size = 0
            try:
                with stage.open("wb") as output:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > self.config.inference.max_adapter_upload_bytes:
                            raise HTTPException(413, "adapter exceeds configured upload limit")
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

            async def load(url):
                deadline = time.monotonic() + self.config.ready_timeout
                while True:
                    async def chunks():
                        with target.open("rb") as stream:
                            while chunk := stream.read(1024**2):
                                yield chunk
                    try:
                        params = {"lora_name": version}
                        if previous:
                            params["previous_lora_name"] = previous
                        result = await self.http.post(url + "/skyrl/v1/upload_lora_adapter",
                                                      params=params, content=chunks(),
                                                      timeout=min(300, max(1, deadline-time.monotonic())))
                        result.raise_for_status()
                        return url
                    except httpx.HTTPError:
                        if time.monotonic() > deadline:
                            raise
                        await asyncio.sleep(5)

            # Failure leaves admission closed, never a mixed adapter version.
            loaded = await asyncio.gather(*(load(url) for url in self.engine_urls), return_exceptions=True)
            if any(isinstance(result, Exception) for result in loaded):
                raise HTTPException(503, "adapter fanout failed; generation remains paused until retry")
            if previous:
                await self.catalog.commit.remote(version, previous)
            else:
                await self.catalog.commit.remote(version)
            async with self.condition:
                if previous:
                    self.versions.discard(previous)
                    self.retired.add(previous)
                self.versions.add(version)
                self.version, self.updating = version, False
                self.condition.notify_all()
            emit(self.run / "inference-events.jsonl", "adapter_committed", version=version,
                 bytes=size, sha256=identity, hosts=loaded, load_seconds=time.monotonic()-started)
            return {"lora_name": version, "loaded": loaded, "sha256": identity}

    def select_engine(self, model):
        if self.engine_models:
            if model not in self.engine_models:
                raise HTTPException(400, "unknown base model")
            indices = [i for i, name in enumerate(self.engine_models) if name == model]
            cursor = self.next_model_engine.get(model, 0)
            self.next_model_engine[model] = cursor + 1
            return self.engines[indices[cursor % len(indices)]]
        handle = self.engines[self.next_engine % len(self.engines)]
        self.next_engine += 1
        return handle

    async def restrict_engines(self, remaining_ips):
        """Close admission, drain ALL requests, remove retired targets, then stop them.

        Admission stays closed until the controller redeploys the reduced graph.
        Neither adapter fanout nor tokenization can address a retired host.
        """
        async with self.upload_lock:
            async with self.condition:
                self.updating = True
                await self.condition.wait_for(lambda: self.active == 0)
            slots = self.config.engine_slots(self.ips)
            keep = [i for i, s in enumerate(slots) if s['ip'] in remaining_ips]
            if not keep or len(keep) != len(self.config.engine_slots(remaining_ips)):
                raise ValueError('invalid remaining engine hosts')
            retired = [e for i, e in enumerate(self.engines) if i not in keep]
            await self.catalog.restrict.remote([slots[i]['key'] for i in keep])
            self.engines = [self.engines[i] for i in keep]
            self.engine_urls = [self.engine_urls[i] for i in keep]
            self.ips = list(remaining_ips)
            self.next_engine = 0
            return await asyncio.gather(*(e.retire.remote() for e in retired))

    @app.post("/tokenize")
    async def tokenize(self, request: Request):
        payload = await request.json()
        if payload.get("model") not in self.config.served_models:
            raise HTTPException(400, "unknown base model")
        if not self.config.bootstrap_all_hosts:
            # Preserve legacy tokenization admission when role transitions are
            # disabled. Only expanded bootstrap needs tokenization draining.
            return await self.select_engine(payload["model"]).tokenize.remote(payload)
        async with self.condition:
            if self.updating:
                raise HTTPException(409, 'engine transition or adapter update in progress')
            self.active += 1
        try:
            return await self.select_engine(payload["model"]).tokenize.remote(payload)
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()

    @app.post("/v1/completions")
    async def generate(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(400, "training endpoint expects complete generation responses")
        if payload.get("seed") is not None:
            raise HTTPException(400, "TPU backend does not support per-request seeds")
        async with self.condition:
            if self.updating or payload.get("model") not in self.versions | set(self.config.served_models):
                raise HTTPException(409, "adapter update in progress or uncommitted model")
            if not payload.get("model"):
                raise HTTPException(400, "model is required")
            self.active += 1
        try:
            handle = self.select_engine(payload["model"])
            return await handle.generate.remote(payload)
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()


def engine_version(raw_config, prepared, head, slot=0):
    """Keep unchanged engines alive across a reduced Serve application graph.

    A deployment name alone is insufficient: Serve generates a random code
    version on every run_many() when version is omitted. Include the engine's
    own inputs, not the application's engine list, so retiring its neighbors
    preserves this replica. Changed code or initialization still replaces it.
    """
    inputs = json.dumps([raw_config, prepared, head, slot], sort_keys=True).encode()
    return hashlib.sha256(Path(__file__).read_bytes() + b'\0' + inputs).hexdigest()


def pin_deployment_version(deployment, version):
    # Pinned Ray 2.58 removed options(version=...), but its application client
    # still consumes Deployment._version and otherwise generates a random one.
    # Keep this compatibility access isolated; the real Serve handoff probe
    # verifies that graph rebuilding preserves it through binding/serialization.
    if not hasattr(deployment, '_version'):
        raise RuntimeError('Ray Serve no longer exposes the pinned version contract')
    deployment._version = version
    return deployment


def versioned_engine(*, engine_identity, **options):
    return pin_deployment_version(Engine.options(**options), engine_identity)


def deploy(config, prepared, catalog, head):
    serving = {ip: info for ip, info in prepared.items() if info["role"] == "inference"}
    heads = sorted(ip for ip, info in serving.items() if (info.get("group") or [ip])[0] == ip)
    engine_models = None
    if config.arena_models:
        engine_models = [serving[ip]["config"]["model"] for ip in heads]
        engines = [versioned_engine(name=f"arena-engine-{i}", num_replicas=1,
                    engine_identity=engine_version(serving[ip]["config"], {ip: serving[ip]}, head),
                    ray_actor_options={"num_cpus": 8, "resources": {"TPU": 4, f"node:{ip}": 0.01}})
                   .bind(serving[ip]["config"], {ip: serving[ip]}, catalog, head)
                   for i, ip in enumerate(heads)]
    elif config.inference.hosts_per_engine == 1:
        engines = [versioned_engine(name=(f"engine-{s['ip'].replace('.', '-')}-{s['slot']}"
                                       if config.bootstrap_all_hosts else f"engine-{i}"), num_replicas=1,
                   engine_identity=engine_version(config.to_dict(), {s['ip']: serving[s['ip']]}, head, s['slot']),
                   ray_actor_options={"num_cpus": 8, "resources": {"TPU": config.inference.tp, f"node:{s['ip']}": 0.01}})
                   .bind(config.to_dict(), {s["ip"]: serving[s["ip"]]}, catalog, head, slot=s["slot"])
                   for i, s in enumerate(config.engine_slots(heads))]
    else:
        # One deployment per engine group, pinned to the group's first host and
        # claiming NO TPU: vLLM's Ray executor takes TPU:4 on both hosts of the
        # group through the placement group tpu/vllm_tpu_server.py creates.
        engines = [versioned_engine(name=f"engine-{i}", num_replicas=1,
                                  engine_identity=engine_version(config.to_dict(), serving, head),
                                  ray_actor_options={"num_cpus": 8, "resources": {f"node:{ip}": 0.01}})
                   .bind(config.to_dict(), serving, catalog, head) for i, ip in enumerate(heads)]
    ingress = Ingress.options(ray_actor_options={"num_cpus": 1, "resources": {f"node:{head}": 0.01}})
    serve.start(proxy_location="HeadOnly", http_options={"host": "0.0.0.0", "port": config.ports.inference})
    return serve.run_many([serve.RunTarget(
        target=ingress.bind(config.to_dict(), engines, catalog, heads, engine_models=engine_models), name="skyrl-training")],
        wait_for_ingress_deployment_creation=False, wait_for_applications_running=False)[0]
