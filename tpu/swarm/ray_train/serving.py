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


class GenerationRequestError(Exception):
    """A rejected request, safe to propagate through Ray without killing a farm."""
    def __init__(self, status_code, detail):
        super().__init__(status_code, detail)
        self.status_code, self.detail = status_code, detail


def request_error(exc):
    # RayTaskError carries the original cause even when it cannot subclass it.
    cause = exc.cause if isinstance(exc, ray.exceptions.RayTaskError) else exc
    return cause if isinstance(cause, GenerationRequestError) else None


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
        self.fatal_error = None

    def fail(self, reason):
        if self.restart_limit == 0 and self.fatal_error is None:
            self.fatal_error = reason

    def quarantine(self, reason):
        self.fatal_error = self.fatal_error or reason

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
        return dict(fatal_error=self.fatal_error, version=self.version, versions=sorted(self.versions), expected=sorted(self.expected), replicas=list(self.replicas.values()), starts=self.starts,
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
        self.serving_identity = None
        if self.config.inference.external_pool_attestation:
            from .serving_identity import identity
            self.serving_identity = identity(self.config, self.root, self.source, info['snapshot'])
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
        if self.config.science_routing_evaluator == 'parallel-v2':
            from tpu.science.routing_resources import host_partition
            _, service_cpus = host_partition()
            command = ['taskset', '--cpu-list', ','.join(map(str, service_cpus)), *command]
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

    async def identity(self):
        return self.serving_identity

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
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 404, 413, 422, 429):
                raise GenerationRequestError(exc.response.status_code,
                    'Inference engine rejected the generation request') from exc
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
        self.lease = None
        # Cancellation tombstones live for this ingress incarnation. Expiring
        # them could allow an arbitrarily delayed acquire to grant afterward.
        self.acquire_requests = {}
        self.inflight = set()
        self.quarantined = False
        self.compatibility = None
        self.http = httpx.AsyncClient(timeout=self.config.inference.request_timeout)
        self.borrower = None
        self.borrowing_instance = uuid.uuid4().hex
        if self.config.borrows_inference:
            from .borrowing import Borrower
            from .run_borrowing import RunBorrower
            borrower_type = RunBorrower if self.config.inference.external_pool_lease_scope == 'run' else Borrower
            self.borrower = borrower_type(self.config, self.http,
                lambda event, **fields: emit(self.run / 'inference-events.jsonl', event, **fields))
        self.lease_watchdog = (asyncio.create_task(self.watch_orphaned_lease())
                               if self.config.inference.require_lease else None)
        self.scheduler = None
        if self.config.inference.external_pool_scheduler:
            from .hybrid_scheduler import HybridScheduler
            self.scheduler = HybridScheduler(len(self.engines), self.config.inference.external_pool_max_concurrent_requests,
                lambda: bool(self.borrower and self.borrower.lease and self.borrower._eligible(self.borrower.lease)),
                lambda event, **fields: emit(self.run / 'inference-events.jsonl', event, **fields))

    def require_lease(self, lease_id, *, allow_expired=False):
        if self.quarantined:
            raise HTTPException(503, 'farm quarantined pending owned runtime recycle')
        if not self.config.inference.require_lease:
            return
        lease = self.lease
        if not lease or not lease_id or lease_id != lease["lease_id"]:
            raise HTTPException(409, "missing or stale lease ID")
        if not allow_expired and (lease["releasing"] or time.monotonic() >= lease["deadline"]):
            raise HTTPException(409, "lease is expired or draining")

    async def watch_orphaned_lease(self):
        orphan, started = None, None
        while True:
            await asyncio.sleep(.5)
            lease = self.lease
            expired = lease and (lease['releasing'] or time.monotonic() >= lease['deadline'])
            if not expired or not (self.active or self.inflight or self.upload_lock.locked()):
                orphan, started = None, None
                continue
            if lease is not orphan:
                orphan, started = lease, time.monotonic()
            if time.monotonic() - started >= self.config.inference.farm_drain_timeout:
                # Do not grant another owner while an engine might still execute
                # abandoned work. The controller tears down its owned runtime;
                # the managed farm job's retry starts a fresh lease namespace.
                self.quarantined = True
                lease['releasing'] = True
                await self.catalog.quarantine.remote('expired farm lease failed to drain')
                return

    async def lease_status(self):
        lease = self.lease
        version = lease["adapter_name"] if lease else None
        identity = lease["adapter_sha256"] if lease else None

        async def probe(url):
            try:
                response = await self.http.get(url + "/health", timeout=5)
                response.raise_for_status()
                if identity:
                    response = await self.http.get(url + "/skyrl/v1/adapter_status", timeout=5)
                    response.raise_for_status()
                    return response.json().get("adapters", {}).get(version) == identity
                return False
            except (httpx.HTTPError, ValueError, AttributeError):
                return False

        ready = sum(await asyncio.gather(*(probe(url) for url in self.engine_urls)))
        if lease is not self.lease:
            # Do not report a previous owner's hash as ready after reassignment.
            return {"lease_id": None, "owner_run": None, "adapter_sha256": None,
                    "ready_engines": 0, "state": "transitioning"}
        if self.quarantined:
            state = 'quarantined'
        elif not lease:
            state = "unleased"
        elif lease["releasing"]:
            state = "draining"
        elif time.monotonic() >= lease["deadline"]:
            state = "expired"
        elif self.updating or (version, identity) != (lease["adapter_name"], lease["adapter_sha256"]):
            state = "updating"
        elif not identity:
            state = "awaiting_adapter"
        else:
            state = "ready" if ready == len(self.engine_urls) else "degraded"
        return dict(lease_id=lease["lease_id"] if lease else None,
                    owner_run=lease["owner_run"] if lease else None,
                    adapter_name=version, adapter_sha256=identity,
                    expires_at=lease["expires_at"] if lease else None,
                    ready_engines=ready, expected_engines=len(self.engine_urls), state=state)

    @app.post("/acquire_lease")
    async def acquire_lease(self, request: Request):
        if self.quarantined:
            raise HTTPException(503, 'farm quarantined')
        if not self.config.inference.require_lease:
            raise HTTPException(409, "leases are not enabled on this farm")
        payload = await request.json()
        owner = payload.get("owner_run")
        ttl = payload.get("ttl_seconds", 300)
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 256:
            raise HTTPException(400, "owner_run must be a nonempty string of at most 256 characters")
        if type(ttl) is not int or not 30 <= ttl <= 86400:
            raise HTTPException(400, "ttl_seconds must be an integer between 30 and 86400")
        if payload.get("lease_id"):
            # Renewals must not queue behind a large upload or its first-load
            # compilation; otherwise an active owner could expire mid-upload.
            async with self.condition:
                self.require_lease(payload["lease_id"])
                if owner != self.lease["owner_run"]:
                    raise HTTPException(409, "lease belongs to another run")
                self.lease.update(deadline=time.monotonic() + ttl, expires_at=time.time() + ttl)
            return await self.lease_status()
        acquire_id = payload.get('acquire_id')
        record = None
        if acquire_id is not None:
            self.validate_acquire_identity(payload)
            record = self.acquire_requests.get(acquire_id)
            if record:
                raise HTTPException(410 if record['cancelled'] else 425, 'acquire already submitted')
            record = self.acquire_requests[acquire_id] = dict(owner=owner, cancelled=False)

        def check_cancelled():
            if record and record['cancelled']:
                raise HTTPException(410, 'acquire cancelled')

        try:
            if self.config.inference.external_pool_attestation:
                capabilities = await self.capabilities()
                if payload.get('compatibility_sha256') != capabilities['compatibility_sha256']:
                    raise HTTPException(409, 'compatible serving runtime attestation required')
            async with self.upload_lock:
                async with self.condition:
                    check_cancelled()
                    if self.quarantined:
                        raise HTTPException(503, 'farm quarantined')
                    if self.lease and time.monotonic() < self.lease['deadline']:
                        raise HTTPException(409, 'farm is already leased; provide the lease ID to renew')
                    if self.lease:
                        self.lease['releasing'] = True
                    await self.condition.wait_for(lambda: self.active == 0 or record and record['cancelled'])
                    check_cancelled()
                    self.lease = dict(lease_id=uuid.uuid4().hex, owner_run=owner, acquire_id=acquire_id,
                                      adapter_name=None, adapter_sha256=None, releasing=False,
                                      deadline=time.monotonic() + ttl, expires_at=time.time() + ttl)
                return await self.lease_status()
        except HTTPException:
            if record:
                record['cancelled'] = True
            raise

    def validate_acquire_identity(self, payload):
        acquire_id, owner = payload.get('acquire_id'), payload.get('owner_run')
        if not isinstance(acquire_id, str) or not re.fullmatch(r'[a-f0-9]{32}', acquire_id):
            raise HTTPException(400, 'acquire_id must be a UUID hex string')
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 256:
            raise HTTPException(400, 'invalid owner_run')
        if payload.get('farm_instance') != self.borrowing_instance:
            raise HTTPException(409, 'farm incarnation changed')
        record = self.acquire_requests.get(acquire_id)
        if record and record['owner'] != owner:
            raise HTTPException(409, 'acquire belongs to another run')

    @app.post('/cancel_acquire')
    async def cancel_acquire(self, request: Request):
        if not self.config.inference.require_lease:
            raise HTTPException(409, 'leases are not enabled on this farm')
        payload = await request.json()
        self.validate_acquire_identity(payload)
        acquire_id = payload['acquire_id']
        async with self.condition:
            record = self.acquire_requests.setdefault(acquire_id, dict(owner=payload['owner_run']))
            # A queued coroutine holds this same record.
            record['cancelled'] = True
            self.condition.notify_all()
            lease = self.lease
            matched = lease and lease.get('acquire_id') == acquire_id
            if matched:
                lease['releasing'] = True
        if matched:
            # An acknowledged cancellation means both pending grants and any
            # granted work have drained, including uploads and disconnected HTTP.
            async with self.upload_lock:
                async with self.condition:
                    await self.condition.wait_for(lambda: self.active == 0)
                    if self.lease is lease:
                        self.lease = None
                    self.condition.notify_all()
        return dict(cancelled=True, acquire_id=acquire_id, instance=self.borrowing_instance)

    @app.post("/release_lease")
    async def release_lease(self, request: Request):
        if not self.config.inference.require_lease:
            raise HTTPException(409, "leases are not enabled on this farm")
        payload = await request.json()
        async with self.upload_lock:
            async with self.condition:
                self.require_lease(payload.get("lease_id"), allow_expired=True)
                self.lease["releasing"] = True
                await self.condition.wait_for(lambda: self.active == 0)
                self.lease = None
                self.condition.notify_all()
        return {"state": "unleased", "released": True}

    @app.get("/status")
    async def status(self):
        result = await self.catalog.snapshot.remote()
        result.update(instance=self.borrowing_instance, active=self.active, updating=self.updating, committed=self.version,
                      committed_adapters=sorted(self.versions))
        if self.borrower:
            result['borrowing'] = self.borrower.snapshot()
        if self.scheduler:
            result['scheduling'] = self.scheduler.snapshot()
        if self.config.inference.require_lease:
            result.update(await self.lease_status())
            result['acquire_protocol'] = 1
        if self.config.inference.external_pool_attestation:
            result['capabilities'] = await self.capabilities()
        return result

    async def capabilities(self):
        if self.compatibility is None:
            identities = await asyncio.gather(*(e.identity.remote() for e in self.engines))
            if not identities or any(not item or item != identities[0] for item in identities):
                raise HTTPException(503, 'serving runtime identities disagree')
            self.compatibility = identities[0]
        if self.borrower:
            self.borrower.required_contract = self.compatibility['sha256']
        return dict(compatibility_sha256=self.compatibility['sha256'],
                    contract=self.compatibility['contract'], engines=len(self.engines),
                    max_sequences=self.config.inference.max_sequences,
                    prefix_caching=self.config.inference.prefix_caching,
                    max_loras=self.config.inference.max_loras)

    @app.post('/skyrl/v1/borrowing/reservation')
    async def reservation(self, request: Request):
        if not self.borrower or self.config.inference.external_pool_lease_scope != 'run':
            raise HTTPException(409, 'run reservations are disabled')
        body = await request.json()
        if body.get('run_id') != self.config.run_id or body.get('instance') != self.borrowing_instance:
            raise HTTPException(409, 'stale reservation controller')
        action = body.get('action')
        if action == 'close':
            await self.borrower.close()
        elif action == 'heartbeat':
            self.borrower.touch_run()
        elif action == 'acquire':
            return await self.borrower.reserve()
        else:
            raise HTTPException(400, 'unknown reservation action')
        return self.borrower.snapshot()

    @app.post('/skyrl/v1/borrowing/begin')
    async def begin_borrowing(self, request: Request):
        if not self.borrower:
            return {'enabled': False}
        body = await request.json()
        try:
            phase = adapter_name(body.get('phase_id'))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        expected_n = body.get('expected_n', 1)
        if type(expected_n) is not int or expected_n <= 0:
            raise HTTPException(400, 'expected_n must be a positive integer')
        async with self.upload_lock:
            if self.updating:
                raise HTTPException(409, 'adapter update in progress')
            # Base-model bootstrap needs ownership but no uploaded adapter.
            model = self.config.model if body.get('bootstrap') else self.version
            if not model:
                return {'enabled': True, 'ready': False, 'reason': 'no committed adapter'}
            archive = None if model == self.config.model else self.archives / (adapter_name(model) + '.tar')
            from .borrowing import BorrowingProtocolError
            try:
                return await self.borrower.begin(phase, model, archive, expected_n=expected_n)
            except BorrowingProtocolError as exc:
                raise HTTPException(409, str(exc))

    @app.get('/skyrl/v1/borrowing/services')
    async def borrowing_services(self):
        if self.config.inference.external_pool_attestation:
            await self.capabilities()
        return dict(enabled=self.config.inference.external_pool_updates,
                    model=self.config.model, run_id=self.config.run_id,
                    instance=self.borrowing_instance,
                    lease_scope=self.config.inference.external_pool_lease_scope,
                    compatibility_sha256=self.compatibility['sha256'] if self.compatibility else None,
                    workload=('ac2' if self.config.client_env.get('TTD_PROBLEM_TYPE') == 'ac2'
                              else 'rglru' if self.config.is_recurrent_gemma
                              else 'qubit' if self.config.science_task == 'routing' else 'other'),
                    borrowing=self.borrower.snapshot() if self.borrower else None,
                    urls=list(self.borrower.urls) if self.borrower else [])

    @app.post('/skyrl/v1/borrowing/services')
    async def update_borrowing_services(self, request: Request):
        if not self.borrower or not self.config.inference.external_pool_updates:
            raise HTTPException(409, 'service list updates are disabled')
        body = await request.json()
        if (not isinstance(body, dict) or body.get('run_id') != self.config.run_id
                or body.get('instance') != self.borrowing_instance):
            raise HTTPException(409, 'stale or incorrect target service')
        from .borrowing import BorrowingProtocolError
        try:
            self.borrower.update_urls(body.get('model'), body.get('urls'))
        except (ValueError, BorrowingProtocolError) as exc:
            raise HTTPException(400, str(exc))
        return await self.borrowing_services()

    @app.post('/skyrl/v1/borrowing/heartbeat')
    async def heartbeat_borrowing(self, request: Request):
        if not self.borrower:
            return {'enabled': False}
        body = await request.json()
        phase = body.get('phase_id')
        if not isinstance(phase, str) or not phase:
            raise HTTPException(400, 'phase_id is required')
        return {'renewed': self.borrower.touch(phase)}

    @app.post('/skyrl/v1/borrowing/end')
    async def end_borrowing(self, request: Request):
        if not self.borrower:
            return {'enabled': False}
        body = await request.json()
        phase = body.get('phase_id')
        if not isinstance(phase, str) or not phase:
            raise HTTPException(400, 'phase_id is required')
        return {'ended': await self.borrower.end(phase)}

    @app.get("/health")
    async def health(self):
        if self.quarantined:
            raise HTTPException(503, 'farm quarantined')
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
        # Keep the upload lock until fanout actually finishes, even when the
        # caller disconnects. A successor lease must not race a late load ACK.
        return await self.track_operation(self.upload_adapter(request, lora_name, previous_lora_name))

    async def upload_adapter(self, request, lora_name, previous_lora_name):
        if self.engine_models:
            raise HTTPException(400, "multi-model arena is inference-only")
        try:
            version = adapter_name(lora_name)
            if previous_lora_name:
                adapter_name(previous_lora_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        async with self.upload_lock:
            if self.borrower:
                await self.borrower.end()
            self.require_lease(request.headers.get("x-lease-id"))
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
                expected = request.headers.get("x-adapter-sha256")
                if expected is not None and expected != identity:
                    raise HTTPException(400, "uploaded archive does not match X-Adapter-SHA256")
                if target.exists():
                    with target.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != identity:
                            raise HTTPException(409, "adapter version is immutable")
                else:
                    stage.replace(target)
            finally:
                stage.unlink(missing_ok=True)
            async with self.condition:
                self.require_lease(request.headers.get("x-lease-id"))
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
                        if self.config.inference.require_lease and result.json().get("sha256") != identity:
                            raise ValueError("engine did not acknowledge the expected adapter hash")
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
                if self.config.inference.require_lease:
                    self.lease.update(adapter_name=version, adapter_sha256=identity)
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
            if self.borrower:
                if self.config.inference.external_pool_lease_scope == 'run':
                    await self.borrower.close()
                else:
                    await self.borrower.end()
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
            if self.scheduler:
                self.scheduler.local = [None] * len(self.engines)
            return await asyncio.gather(*(e.retire.remote() for e in retired))

    @app.post("/tokenize")
    async def tokenize(self, request: Request):
        payload = await request.json()
        if payload.get("model") not in self.config.served_models:
            raise HTTPException(400, "unknown base model")
        if not self.config.bootstrap_all_hosts and not self.config.inference.require_lease:
            # Preserve legacy tokenization admission when role transitions are
            # disabled. Only expanded bootstrap needs tokenization draining.
            return await self.select_engine(payload["model"]).tokenize.remote(payload)
        async with self.condition:
            self.require_lease(request.headers.get("x-lease-id"))
            if self.updating:
                raise HTTPException(409, 'engine transition or adapter update in progress')
            self.active += 1
        return await self.finish_admitted("tokenize", payload)

    async def finish_admitted(self, method, payload, *, local_only=False):
        async def local(index):
            try:
                return await getattr(self.engines[index], method).remote(payload)
            except Exception as exc:
                if rejected := request_error(exc):
                    raise HTTPException(rejected.status_code, rejected.detail) from exc
                if method == 'generate' and self.config.inference.restart_limit == 0:
                    await self.catalog.fail.remote('local generation failed')
                raise

        async def remote():
            return await self.borrower.generate(payload, local_active=0,
                                               local_engines=len(self.engines), prefer_remote=True)

        async def run():
            try:
                if method == 'generate' and self.scheduler:
                    return await self.scheduler.run(payload, local, remote, local_only=local_only)
                if method == "generate" and self.borrower and not local_only:
                    external_active = self.borrower.lease.active if self.borrower.lease else 0
                    result = await self.borrower.generate(payload,
                        local_active=max(0, self.active - external_active - 1), local_engines=len(self.engines))
                    if result is not None:
                        return result
                try:
                    return await getattr(self.select_engine(payload["model"]), method).remote(payload)
                except Exception as exc:
                    if rejected := request_error(exc):
                        raise HTTPException(rejected.status_code, rejected.detail) from exc
                    if method == "generate" and self.config.inference.restart_limit == 0:
                        await self.catalog.fail.remote('local generation failed')
                    raise
            finally:
                async with self.condition:
                    self.active -= 1
                    self.condition.notify_all()

        # An HTTP disconnect must not make a still-running engine request
        # disappear from the drain count and allow a lease handoff underneath it.
        return await self.track_operation(run())

    async def track_operation(self, operation):
        task = asyncio.create_task(operation)
        self.inflight.add(task)

        def done(completed):
            self.inflight.discard(completed)
            if not completed.cancelled():
                completed.exception()  # Retrieve errors even if the caller disconnected.

        task.add_done_callback(done)
        return await asyncio.shield(task)

    @app.post("/v1/completions")
    async def generate(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(400, "training endpoint expects complete generation responses")
        if payload.get("seed") is not None:
            raise HTTPException(400, "TPU backend does not support per-request seeds")
        async with self.condition:
            self.require_lease(request.headers.get("x-lease-id"))
            if (self.config.inference.require_lease and payload.get("model") not in self.config.served_models
                    and payload.get("model") != self.lease["adapter_name"]):
                raise HTTPException(409, "adapter has not been verified for this lease")
            if self.updating or payload.get("model") not in self.versions | set(self.config.served_models):
                raise HTTPException(409, "adapter update in progress or uncommitted model")
            if not payload.get("model"):
                raise HTTPException(400, "model is required")
            self.active += 1
        return await self.finish_admitted("generate", payload,
            local_only=request.headers.get('x-skyrl-local-only') == '1')


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
