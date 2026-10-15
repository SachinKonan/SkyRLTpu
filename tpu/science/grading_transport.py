"""Task-tagged candidate grading over local Ray slots and leased farms.

The training client grades candidates through one transport: a local pool
(Ray tasks on the trainer's own cluster, the same executors as before) plus
one pool per farm lease the trainer's ingress currently holds (fetched from
``GET $SKYRL_GRADING_URL/skyrl/v1/grading/farms``). Every request keeps one
``request_id`` across retries so a farm's late result can never enter the
batch twice (first result wins); a retry on a farm submits under an id
derived from the request id and attempt, because the farm keeps a cancelled
id as cancelled. Infrastructure failures retry on another
pool up to ``max_infra_retries`` and then raise ``GradingInfrastructureError``
(``abort_training_step``), exactly like the science graders; candidate
failures are returned as results and stay reward zero.

The transport runs its own asyncio loop in a background thread so the AC2
grader (a worker-thread call) and the science graders (coroutines on the
client's loop) share one scheduler.

Open grading (``SKYRL_OPEN_GRADING=1``): grading does not depend on the served
model, so besides its leased farms the client also grades on up to
``SKYRL_OPEN_GRADING_K`` (4) farms of any model from the farm registry
(``SKYRL_GRADING_REGISTRY``, the discovery's farms.json), without a lease. Each
grade goes to the less loaded of two random healthy farms (by the load the farm
advertises); a farm that answers 429 (full) or runs another grader version is
skipped without counting an infrastructure attempt. Farms built before open
grading refuse lease-free requests and are simply not used.

A farm that refused (429 full, unreachable) is skipped for its ``retry_after``
(x1-2 jitter) and then takes one unacknowledged submission at a time; each ACK
widens that window by one (slow start). Grades waiting for room therefore wait
in the client instead of all resubmitting the moment the farm may be asked again.
"""
import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
import uuid

import httpx

from tpu.swarm.ray_train.request_protocol import Config, Contract, Lost, Refused, State, Ticket, drive

INFRA_MARKERS = ('infrastructure', 'admission timed out', 'before the candidate started', 'without a result file',
                 'WorkerCrashedError', 'RayActorError', 'NodeDiedError', 'OwnerDiedError', 'ObjectLostError',
                 'RaySystemError', 'cpu_scheduler')


# Every grade in flight on a farm holds a long poll. A remote-only client can
# have ~1000 in flight, so a full connection pool must queue a request rather
# than fail it as infrastructure (httpx's default 5 s pool timeout did).
FARM_CONNECTIONS = 256
# Concurrent HTTP requests per leased farm. A local run reaches each farm through
# one SSH tunnel, and circuit's thousands of in-flight grades (one long-poll each)
# made it refuse connections (ConnectTimeout loops). Grades take minutes, so
# polling them through a smaller window costs nothing.
FARM_HTTP_CONCURRENCY = int(os.environ.get('SKYRL_GRADING_FARM_HTTP_CONCURRENCY') or 64)
# Open farms are shared by every run on this host through the same tunnel, so each run polls
# an open farm through a smaller window than its leased ones.
OPEN_FARM_HTTP_CONCURRENCY = int(os.environ.get('SKYRL_OPEN_GRADING_HTTP_CONCURRENCY') or 32)


def wait_for_connection(seconds):
    return httpx.Timeout(seconds, pool=None)


class GradingInfrastructureError(RuntimeError):
    """No pool could run the candidate; the training step must not use a fake zero."""
    abort_training_step = True


class GradingInfrastructureFailed(GradingInfrastructureError):
    """Infrastructure failed past the retry cap: drop this candidate from its step (logged and flagged).

    Never scored 0 and never aborts the step: callers exclude the candidate (``drop_candidate``)."""
    abort_training_step = False
    drop_candidate = True


class _BatchUnsupported(Exception):
    """The farm predates the batch result endpoint: poll ids one by one."""


# Every protocol threshold (poll misses, queue limit, retry cap, jitter, registry staleness): one config.
CONFIG = Config.from_env()
# Each family's own wall-clock cap for one grade: routing outer envelope, placement case envelope.
DEFAULT_TIME_CAPS = {'routing': 2100, 'placement': 510, 'math': 600}


def time_cap(request):
    if request.time_cap_s:
        return float(request.time_cap_s)
    if request.task == 'math' and request.spec.get('eval_timeout_seconds'):
        return float(request.spec['eval_timeout_seconds'])
    return float(DEFAULT_TIME_CAPS.get(request.task, 600))


def local_grader_sha256():
    try:
        from tpu.swarm.ray_train.grading_service import grader_sha256
        return grader_sha256(Path(__file__).resolve().parent)
    except Exception:
        return None


@dataclass
class GradingRequest:
    task: str
    spec: dict
    scope: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    admission_timeout_s: int = 1100
    deadline_s: float | None = None
    # The grade's own wall-clock cap; None = the family default (DEFAULT_TIME_CAPS).
    time_cap_s: float | None = None
    # Original creation time (unix), sent with every (re)submission: farms serve the oldest first.
    created_at: float = field(default_factory=time.time)


def is_infrastructure(exc):
    name = type(exc).__name__
    text = f'{name}: {exc}'
    if isinstance(exc, (httpx.TransportError, asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code in (404, 409)
    return any(marker in text for marker in INFRA_MARKERS)


class Pool:
    kind = 'pool'

    def __init__(self, name, capacity):
        self.name = name
        self.capacity = max(1, int(capacity))
        self.running = 0
        self.queued = 0
        self.mean_seconds = None
        self.unhealthy_until = 0.0
        # A farm that answered 429 is skipped until then (not an infrastructure failure).
        self.full_until = 0.0
        # Submissions awaiting an ACK (``queued``) allowed at once; None until the farm first refuses.
        # A refusal sets 1, each ACK adds 1: one probe at a time while full, back to many once it has room.
        self.window = None
        self.eligible = True
        # The farm rejected our grader contract: never send it another request.
        self.contract_mismatch = False

    def capacity_for(self, task):
        return max(1, int(getattr(self, 'capacities', {}).get(task) or self.capacity))

    def predicted_wait(self, task=None, penalty=0.0):
        mean = self.mean_seconds or 60.0
        capacity = self.capacity_for(task)
        backlog = max(0, self.running + self.queued + 1 - capacity)
        return backlog * mean / capacity + penalty

    def observe(self, seconds):
        self.mean_seconds = seconds if self.mean_seconds is None else .75 * self.mean_seconds + .25 * seconds

    def healthy(self, now):
        return (self.eligible and not self.contract_mismatch and now >= self.unhealthy_until
                and now >= self.full_until and (self.window is None or self.queued < self.window))

    def serves(self, task):
        return True

    def free_slot(self, task):
        """A slot free right now (not just queue room): the only reason to move a queued request here."""
        family = (getattr(self, 'advertised', None) or {}).get('families', {}).get(task)
        if family:
            if family.get('total', 0) - family.get('running', 0) - family.get('queued', 0) <= 0:
                return False
        return self.running + self.queued < self.capacity_for(task)

    async def run(self, request, on_started, attempt=0, alternative=lambda: True, report=None, resume=None):
        raise NotImplementedError

    async def cancel(self, request_id):
        pass


class LocalRayPool(Pool):
    """The trainer's own Ray cluster, dispatching the existing executors."""
    kind = 'local'

    def __init__(self, families, capacity, *, root=None, systemd=True):
        super().__init__('local', capacity)
        self.families = families
        self.root = root
        self.systemd = systemd
        self.refs = {}

    def dispatch(self, request):
        if request.task == 'math':
            from tpu.science.math_grade import grade_math
            family = self.families['math']
            spec = dict(request.spec, systemd=self.systemd, admission_timeout_s=request.admission_timeout_s)
            return grade_math.options(num_cpus=family['cpus'], memory=family['memory_gib'] * 1024 ** 3,
                                     resources={'grading_math': 1}, scheduling_strategy='SPREAD'
                                     ).remote(spec, self.families, self.root)
        if request.task == 'routing':
            from tpu.science.ray_cpu import grade
            spec = request.spec
            options = dict(scheduling_strategy='SPREAD')
            if spec.get('resource_contract'):
                options.update(num_cpus=spec['resource_contract']['program_cpus'],
                               memory=spec['resource_contract']['program_memory_gib'] * 1024 ** 3)
            return grade.options(**options).remote(
                'routing', spec['source'], self.root, admission_timeout_s=request.admission_timeout_s,
                slots_per_host=spec['slots_per_host'], routing_suite=spec.get('routing_suite', 'full'),
                **({'resource_contract': spec['resource_contract']} if spec.get('resource_contract') else {}))
        from tpu.science.placement_ray import grade_cpu_case
        spec = request.spec
        options = dict(scheduling_strategy='SPREAD')
        if spec.get('resource_contract'):
            from tpu.science.placement_resources import ray_options
            options.update(ray_options(spec['resource_contract']))
        return grade_cpu_case.options(**options).remote(
            spec['source'], spec['case'], self.root, admission_timeout_s=request.admission_timeout_s,
            slots_per_host=spec['slots_per_host'], helper=spec.get('helper', 'none'),
            **({'resource_contract': spec['resource_contract']} if spec.get('resource_contract') else {}))

    @staticmethod
    def awaitable(ref):
        return asyncio.wrap_future(ref.future()) if hasattr(ref, 'future') else ref

    async def run(self, request, on_started, attempt=0, alternative=lambda: True, report=None, resume=None):
        ref = self.dispatch(request)
        self.refs[request.request_id] = ref
        on_started(self, None)
        try:
            result = await self.awaitable(ref)
        except Exception as exc:
            raise GradingInfrastructureError(f'local pool: {type(exc).__name__}: {str(exc)[:600]}') from exc
        finally:
            self.refs.pop(request.request_id, None)
        if not isinstance(result, dict):
            raise GradingInfrastructureError('local pool returned a non-dict result')
        return result

    async def cancel(self, request_id):
        ref = self.refs.get(request_id)
        if ref is None:
            return
        if hasattr(ref, 'future'):
            import ray
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass
        elif hasattr(ref, 'cancel'):
            ref.cancel()


class FarmPool(Pool):
    """One leased farm's grading endpoints, fenced by its lease token."""
    kind = 'farm'
    rtt_penalty = 2.0

    open = False

    def __init__(self, farm, http, *, long_poll_seconds=20, poll_seconds=2.0, capacity=None):
        super().__init__(farm['farm_id'], capacity or farm.get('grading_capacity') or 16)
        self.url = farm['url'].rstrip('/')
        self.token = farm['token']
        self.http = http
        self.long_poll_seconds = long_poll_seconds
        self.poll_seconds = poll_seconds
        self.eligible = bool(farm.get('eligible', True))
        # Retired: the lease token changed or the farm left the directory. A
        # brief ineligibility (health pause) leaves in-flight grades running.
        self.retired = False
        self.gate = asyncio.Semaphore(FARM_HTTP_CONCURRENCY)
        self.capacities = {}
        self.ready = None
        self.wire_ids = {}
        # The grader contract travels with every request; a farm with other graders refuses it outright.
        self.grader_sha256 = local_grader_sha256()
        self.advertised = {}
        # Batch long-poll: one request carries every id this client waits on at this farm.
        self.poll_waiters = {}
        self.poll_task = None
        self.batch_supported = True
        # Discovery's own view of this farm (set by the directory): used to tell a path failure from a
        # server failure before rerouting.
        self.directory_healthy = lambda: False

    @property
    def headers(self):
        return {'X-Lease-ID': self.token}

    def predicted_wait(self, task=None, penalty=None):
        return super().predicted_wait(task, self.rtt_penalty if penalty is None else penalty)

    def healthy(self, now):
        return super().healthy(now) and self.ready is not False

    async def probe_capacity(self):
        """Learn per-family slot totals and readiness from the farm itself."""
        try:
            async with self.gate:
                response = await self.http.get(self.url + '/skyrl/v1/grading/capacity', headers=self.headers,
                                               timeout=wait_for_connection(10))
            response.raise_for_status()
            capacity = response.json()
        except Exception:
            return False
        self.capacities = {name: spec.get('total', 0) for name, spec in (capacity.get('families') or {}).items()}
        self.ready = bool(capacity.get('ready', True))
        self.advertised = capacity
        return True

    def wire_id(self, request, attempt):
        # Farms accept only 32-hex ids; every submission gets its own, so a late result is never confused.
        return (request.request_id if not attempt else
                hashlib.sha256(f'{request.request_id}:{attempt}'.encode()).hexdigest()[:32])

    async def _batch_loop(self):
        while self.poll_waiters:
            batch, self.poll_waiters = self.poll_waiters, {}
            try:
                async with self.gate:
                    response = await self.http.post(self.url + '/skyrl/v1/grading/results',
                                                    json=dict(ids=list(batch), wait=self.long_poll_seconds),
                                                    headers=self.headers,
                                                    timeout=wait_for_connection(self.long_poll_seconds + 30))
                if response.status_code in (404, 405):
                    self.batch_supported = False
                    raise _BatchUnsupported()
                response.raise_for_status()
                views = response.json().get('views') or {}
            except BaseException as exc:
                for futures in batch.values():
                    for future in futures:
                        if not future.done():
                            future.set_exception(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
                if not isinstance(exc, Exception):
                    raise
                continue
            for wire_id, futures in batch.items():
                for future in futures:
                    if not future.done():
                        future.set_result(views.get(wire_id))

    async def poll_one(self, wire_id):
        if self.batch_supported:
            future = asyncio.get_running_loop().create_future()
            self.poll_waiters.setdefault(wire_id, []).append(future)
            if self.poll_task is None or self.poll_task.done():
                self.poll_task = asyncio.create_task(self._batch_loop())
            try:
                return await future
            except _BatchUnsupported:
                pass
        async with self.gate:
            response = await self.http.get(f'{self.url}/skyrl/v1/grading/result/{wire_id}',
                                           params={'wait': self.long_poll_seconds}, headers=self.headers,
                                           timeout=wait_for_connection(self.long_poll_seconds + 30))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def confirm_lost(self):
        """True only for a server failure: a direct probe still fails and discovery does not see it healthy."""
        if await self.probe_capacity():
            return False
        return not self.directory_healthy()

    async def run(self, request, on_started, attempt=0, alternative=lambda: True, report=None, resume=None):
        """Submit (or resume), then poll to a verdict (request_protocol.drive): Refused before an ACK, Lost after."""
        wire_id = resume or self.wire_id(request, attempt)
        self.wire_ids[request.request_id] = wire_id
        body = dict(request_id=wire_id, task=request.task, owner_run=os.environ.get('SKYRL_RUN_ID', ''),
                    scope=request.scope, submitted_at=request.created_at,
                    spec=dict(request.spec, admission_timeout_s=request.admission_timeout_s))
        if self.grader_sha256:
            # One contract object with every request; farms still read the bare grader hash too.
            body['contract'] = Contract(kind='grading', grader_sha256=self.grader_sha256).wire()
            body['grader_sha256'] = self.grader_sha256

        async def submit():
            try:
                async with self.gate:
                    response = await self.http.post(self.url + '/skyrl/v1/grading/submit', json=body,
                                                    headers=self.headers, timeout=wait_for_connection(30))
            except httpx.TransportError as exc:
                # No ACK: it may still have been accepted, so cancel that id once and go elsewhere.
                await self.cancel(request.request_id)
                self.unhealthy_until = time.monotonic() + 30
                raise Refused(f'farm {self.name} unreachable: {type(exc).__name__}', 30) from exc
            if response.status_code == 429:
                raise Refused(f'farm {self.name} full', response.json().get('retry_after', 5))
            if response.status_code == 409 and 'grader version' in response.text:
                # Contract mismatch: never retry on this farm.
                self.contract_mismatch = True
                raise Refused(f'farm {self.name}: {response.json().get("detail")}', 3600, banned=True)
            response.raise_for_status()

        ticket = Ticket(request.request_id, self.name, attempt)
        view = await drive(ticket, submit, lambda: self.poll_one(wire_id), lambda: self.cancel(request.request_id),
                           config=CONFIG, queue_limit_s=CONFIG.queue_limit_factor * time_cap(request),
                           alternative=alternative, retired=lambda: self.retired, confirm_lost=self.confirm_lost,
                           resume=bool(resume), on_accept=lambda: on_started(self, wire_id), report=report,
                           poll_interval_s=self.poll_seconds,
                           between_polls_s=0 if self.long_poll_seconds else self.poll_seconds)
        if ticket.state in (State.DONE, State.FAILED_CANDIDATE):
            # A candidate failure is a final verdict, returned like any result (never retried).
            result = view.get('result')
            if not isinstance(result, dict):
                raise GradingInfrastructureError(f'farm {self.name}: done without a result')
            self.wire_ids.pop(request.request_id, None)
            return result
        error = view.get('error') or {}
        raise GradingInfrastructureError(f'farm {self.name}: {ticket.state.value}: {error.get("class")}: '
                                         f'{error.get("detail")}')

    async def cancel(self, request_id):
        request_id = self.wire_ids.pop(request_id, request_id)
        try:
            async with self.gate:
                await self.http.post(f'{self.url}/skyrl/v1/grading/cancel/{request_id}', headers=self.headers,
                                     timeout=wait_for_connection(10))
        except Exception:
            pass


class OpenFarmPool(FarmPool):
    """A farm of any model graded on without a lease (open grading)."""
    open = True

    def __init__(self, farm, http, *, run, token, grader, long_poll_seconds=20, poll_seconds=2.0):
        super().__init__(dict(farm_id=farm['name'], url=farm['url'], token=token, eligible=True), http,
                         long_poll_seconds=long_poll_seconds, poll_seconds=poll_seconds)
        self.run_id = run
        self.grader_sha256 = grader
        self.gate = asyncio.Semaphore(OPEN_FARM_HTTP_CONCURRENCY)

    @property
    def headers(self):
        return {'X-Grading-Run': self.run_id, 'X-Grading-Token': self.token}

    async def probe_capacity(self):
        ok = await super().probe_capacity()
        advertised = self.advertised if ok else {}
        # Farms built before open grading answer 409 (no lease) and are not used.
        self.supported = bool(advertised.get('open'))
        if self.supported and self.grader_sha256 and advertised.get('grader_sha256') not in (None, self.grader_sha256):
            self.supported = False
        if not self.supported:
            self.ready = False
        return self.supported

    def serves(self, task):
        return task in self.capacities

    def reported_backlog(self, task):
        family = (self.advertised.get('families') or {}).get(task) or {}
        return int(family.get('running', 0)) + int(family.get('queued', 0))

    def load_ratio(self):
        families = (self.advertised.get('families') or {}).values()
        return max([(f.get('running', 0) + f.get('queued', 0)) / max(1, f.get('total', 1)) for f in families] or [0])

    def predicted_wait(self, task=None, penalty=None):
        mean = self.mean_seconds or 60.0
        capacity = self.capacity_for(task)
        backlog = max(0, max(self.running + self.queued, self.reported_backlog(task) + self.queued) + 1 - capacity)
        return backlog * mean / capacity + (self.rtt_penalty if penalty is None else penalty)


class OpenFarmDirectory:
    """Up to K open farms from the farm registry, re-ranked by advertised load on every refresh."""

    def __init__(self, registry, http, *, k=4, exclude=lambda: set(), refresh_seconds=10, long_poll_seconds=20,
                 poll_seconds=2.0, report=None):
        self.registry = registry
        self.http = http
        self.k = max(1, int(k))
        self.exclude = exclude
        self.refresh_seconds = refresh_seconds
        self.long_poll_seconds = long_poll_seconds
        self.poll_seconds = poll_seconds
        self.report = report or (lambda *a, **k: None)
        self.run_id = os.environ.get('SKYRL_RUN_ID') or 'open-' + uuid.uuid4().hex[:8]
        self.token = uuid.uuid4().hex
        self.grader = local_grader_sha256()
        self.pools = {}
        self.task = None
        self.last_refresh = 0.0
        self.entries = {}
        self.stale_reported = 0.0

    @staticmethod
    def listable(farm):
        """Only fully ready, attested, healthy farms (fields a registry does not carry are not required)."""
        return (farm.get('ok') and farm.get('tunnel_ok') and farm.get('url')
                and farm.get('health', 'healthy') == 'healthy' and farm.get('ready', True) is not False
                and farm.get('attested', True) is not False)

    async def refresh(self):
        self.last_refresh = time.monotonic()
        try:
            path = Path(self.registry)
            registry = json.loads(path.read_text())
            stamp = registry.get('updated') or registry.get('time') or path.stat().st_mtime
        except (OSError, ValueError) as exc:
            self.report('open_grading_registry_failed', reason=type(exc).__name__)
            return
        farms = registry.get('farms', [])
        age = time.time() - float(stamp)
        stale = age > CONFIG.registry_stale_s
        if stale and time.monotonic() - self.stale_reported > 60:
            self.stale_reported = time.monotonic()
            self.report('grading_registry_stale', age_seconds=round(age), registry=str(self.registry))
        leased = self.exclude()
        # A stale registry is not trusted: every listed farm is probed directly instead.
        listed = {f['name']: f for f in farms if f.get('url') and f['url'].rstrip('/') not in leased
                  and (stale or self.listable(f))}
        self.entries = {name: farm for name, farm in listed.items()}
        for name in list(self.pools):
            pool = self.pools[name]
            if name not in listed or pool.url != listed[name]['url'].rstrip('/'):
                pool.eligible, pool.retired = False, True
                if pool.running == 0 and pool.queued == 0:
                    del self.pools[name]
        for name, farm in listed.items():
            if name not in self.pools:
                self.pools[name] = OpenFarmPool(farm, self.http, run=self.run_id, token=self.token, grader=self.grader,
                                                long_poll_seconds=self.long_poll_seconds, poll_seconds=self.poll_seconds)
                self.pools[name].directory_healthy = (
                    lambda name=name: bool(self.listable(self.entries.get(name) or {})))
        live = [p for p in self.pools.values() if not p.retired]
        await asyncio.gather(*(p.probe_capacity() for p in live), return_exceptions=True)
        usable = sorted((p for p in live if getattr(p, 'supported', False) and p.ready),
                        key=lambda p: (p.load_ratio(), p.name))
        active = {p.name for p in usable[:self.k]}
        before = {p.name for p in live if p.eligible}
        for pool in live:
            # Pools outside the active K keep their in-flight grades; they just take no new ones.
            pool.eligible = pool.name in active
        if active != before:
            self.report('open_grading_farms', active=sorted(active),
                        load={p.name: round(p.load_ratio(), 2) for p in usable})

    async def loop(self):
        while True:
            await self.refresh()
            await asyncio.sleep(self.refresh_seconds)


class FarmDirectory:
    """Tracks the trainer ingress's eligible farm leases as pools."""

    def __init__(self, url, http, *, refresh_seconds=10, long_poll_seconds=20, poll_seconds=2.0, report=None):
        self.url = url.rstrip('/') if url else None
        self.http = http
        self.refresh_seconds = refresh_seconds
        self.long_poll_seconds = long_poll_seconds
        self.poll_seconds = poll_seconds
        self.report = report or (lambda *a, **k: None)
        self.pools = {}
        self.task = None
        self.last_refresh = 0.0

    async def refresh(self):
        if not self.url:
            return
        # Stamp before the request: every waiting grade calls refresh when the
        # stamp is stale, and a failed refresh must not let them all retry at
        # once (1024 bootstrap waiters kept the ingress from ever answering).
        self.last_refresh = time.monotonic()
        try:
            response = await self.http.get(self.url + '/skyrl/v1/grading/farms', timeout=wait_for_connection(10))
            response.raise_for_status()
            farms = response.json().get('farms', [])
        except Exception as exc:
            self.report('grading_farms_refresh_failed', reason=type(exc).__name__)
            return
        seen = set()
        added, removed = [], []
        for farm in farms:
            farm_id = farm['farm_id']
            seen.add(farm_id)
            pool = self.pools.get(farm_id)
            if pool is None or pool.token != farm['token']:
                if pool is not None:
                    pool.eligible = False
                    pool.retired = True
                pool = self.pools[farm_id] = FarmPool(farm, self.http, long_poll_seconds=self.long_poll_seconds,
                                                      poll_seconds=self.poll_seconds)
                pool.directory_healthy = lambda pool=pool: bool(pool.eligible and not pool.retired)
                await pool.probe_capacity()
                added.append(farm_id)
            else:
                pool.eligible = bool(farm.get('eligible', True))
                if pool.ready is not True and pool.eligible:
                    await pool.probe_capacity()
        for farm_id in list(self.pools):
            if farm_id not in seen:
                self.pools[farm_id].eligible = False
                self.pools[farm_id].retired = True
                if self.pools[farm_id].running == 0:
                    del self.pools[farm_id]
                removed.append(farm_id)
        self.last_refresh = time.monotonic()
        if added or removed:
            self.report('grading_farms_changed', added=added, removed=removed)

    async def loop(self):
        while True:
            await self.refresh()
            await asyncio.sleep(self.refresh_seconds)


class GradingScheduler:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, pools, now, task=None, exclude=None):
        candidates = [p for p in pools if p is not exclude and p.healthy(now) and p.serves(task)]
        if not candidates:
            return None
        local = [p for p in candidates if p.kind == 'local']
        # Local grading capacity first: a free local slot always wins. Once every local slot is busy,
        # spread over the farms; queue locally only when no farm can take it.
        if local and local[0].running + local[0].queued < local[0].capacity_for(task):
            return local[0]
        candidates = [p for p in candidates if p.kind != 'local'] or local
        key = lambda p: (p.predicted_wait(task), p.kind != 'local', p.name)
        if any(getattr(p, 'open', False) for p in candidates) and len(candidates) > 2:
            # Power of two choices: the less loaded of two random pools.
            return min(self.rng.sample(candidates, 2), key=key)
        return min(candidates, key=key)


class Journal:
    """Append-only jsonl of in-flight request ids (with farm url and wire id) and finished results.

    A restarted client re-polls journaled ids, keeps finished results, and resubmits only what is missing,
    so a step resumes instead of regrading. Useful when callers give requests stable ids (stable_request_id)."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.inflight, self.results = {}, {}
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            request_id = record.get('request_id')
            if record.get('event') == 'submitted':
                self.inflight[request_id] = record
            elif record.get('event') == 'result':
                self.results.setdefault(request_id, record.get('result'))
                self.inflight.pop(request_id, None)
            elif record.get('event') == 'failed':
                self.inflight.pop(request_id, None)
        # Only recent results matter to a resumed step; older ones are not reloaded.
        for request_id in list(self.results)[:-self.KEEP]:
            del self.results[request_id]

    KEEP = 8192

    def append(self, **record):
        if record.get('event') == 'result' and isinstance(record.get('result'), dict):
            # Bounded journal: long candidate output is cut (the trajectory keeps the full row).
            result = dict(record['result'])
            for key in ('stdout', 'stderr'):
                if isinstance(result.get(key), str) and len(result[key]) > 2000:
                    result[key] = result[key][-2000:]
            record['result'] = result
        record.setdefault('time', time.time())
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as out:
                out.write(json.dumps(record, default=str) + '\n')
        if record.get('event') == 'submitted':
            self.inflight[record['request_id']] = record
        elif record.get('event') in ('result', 'failed'):
            self.inflight.pop(record['request_id'], None)


def stable_request_id(*parts):
    """A 32-hex id from the request's identity, so a restarted client finds it in its journal."""
    return hashlib.sha256('\x00'.join(str(p) for p in parts).encode()).hexdigest()[:32]


class GradingTransport:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, *, families, local_capacity, farm_url=None, root=None, local_systemd=True, max_infra_retries=None,
                 events=None, refresh_seconds=10, long_poll_seconds=20, poll_seconds=2.0, http=None, loop=None,
                 open_registry=None, open_k=4, journal=None, retry_jitter_s=None):
        self.families = families
        self.max_infra_retries = CONFIG.max_infra_retries if max_infra_retries is None else max_infra_retries
        self.retry_jitter_s = CONFIG.retry_jitter_s if retry_jitter_s is None else retry_jitter_s
        self.events = events
        # Exactly once: one grade in flight per request id; concurrent callers share it.
        self.inflight = {}
        # Durable record of in-flight ids and finished results: a restarted client resumes instead of regrading.
        self.journal = Journal(journal) if journal else None
        # First result wins: a request id graded once returns the same result
        # to every later caller instead of running the candidate again.
        self.results = {}
        self.results_limit = 8192
        if self.journal:
            self.results.update(self.journal.results)
        self.local = LocalRayPool(families, local_capacity, root=root, systemd=local_systemd) if local_capacity else None
        self.scheduler = GradingScheduler()
        self.loop = loop
        self.thread = None
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever, name='grading-transport', daemon=True)
            self.thread.start()
        self.http = http
        self.directory = None
        self.farm_url = farm_url
        self.refresh_seconds = refresh_seconds
        self.long_poll_seconds = long_poll_seconds
        self.poll_seconds = poll_seconds
        self.open_registry = open_registry
        self.open_k = open_k
        self.open_directory = None
        self._started = False

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls.from_environment()
            return cls._instance

    @classmethod
    def from_environment(cls):
        env = os.environ
        families = json.loads(env.get('SKYRL_GRADING_FAMILIES') or '{}')
        transport = cls(families=families, local_capacity=int(env.get('SKYRL_GRADING_LOCAL_SLOTS') or 0),
                   farm_url=env.get('SKYRL_GRADING_URL') or None, root=env.get('SCIENCE_WORKER_ROOT') or None,
                   local_systemd=env.get('SKYRL_GRADING_LOCAL_SYSTEMD', '1') == '1',
                   max_infra_retries=int(env.get('SKYRL_GRADING_MAX_INFRA_RETRIES') or CONFIG.max_infra_retries),
                   journal=env.get('SKYRL_GRADING_JOURNAL') or None,
                   events=env.get('SKYRL_GRADING_EVENTS') or None,
                   refresh_seconds=float(env.get('SKYRL_GRADING_FARM_REFRESH_SECONDS') or 10),
                   long_poll_seconds=int(env.get('SKYRL_GRADING_LONG_POLL_SECONDS') or 20),
                   open_registry=(env.get('SKYRL_GRADING_REGISTRY') or None) if env.get('SKYRL_OPEN_GRADING') == '1' else None,
                   open_k=int(env.get('SKYRL_OPEN_GRADING_K') or 4))
        hosts = int(env.get('SKYRL_GRADING_LOCAL_HOSTS') or 0)
        if transport.local and hosts:
            transport.local.capacities = {name: spec['slots_per_host'] * hosts for name, spec in families.items()}
        return transport

    def report(self, event, **fields):
        if not self.events:
            return
        try:
            from tpu.swarm.ray_train.events import emit
            emit(__import__('pathlib').Path(self.events), event, **fields)
        except Exception:
            pass

    async def _ensure_started(self):
        if self._started:
            return
        self._started = True
        if self.http is None:
            self.http = httpx.AsyncClient(limits=httpx.Limits(max_connections=FARM_CONNECTIONS,
                                                              max_keepalive_connections=FARM_CONNECTIONS))
        self.directory = FarmDirectory(self.farm_url, self.http, refresh_seconds=self.refresh_seconds,
                                       long_poll_seconds=self.long_poll_seconds, poll_seconds=self.poll_seconds,
                                       report=self.report)
        if self.farm_url:
            await self.directory.refresh()
            self.directory.task = asyncio.create_task(self.directory.loop())
        if self.open_registry:
            leased = lambda: {p.url for p in self.directory.pools.values()} if self.directory else set()
            self.open_directory = OpenFarmDirectory(self.open_registry, self.http, k=self.open_k, exclude=leased,
                                                    refresh_seconds=self.refresh_seconds,
                                                    long_poll_seconds=self.long_poll_seconds,
                                                    poll_seconds=self.poll_seconds, report=self.report)
            await self.open_directory.refresh()
            self.open_directory.task = asyncio.create_task(self.open_directory.loop())

    def pools(self):
        pools = [self.local] if self.local else []
        if self.directory:
            pools += list(self.directory.pools.values())
        if self.open_directory:
            pools += list(self.open_directory.pools.values())
        return pools

    async def _choose(self, request, attempts=0):
        """Wait until some pool is healthy; there is no deadline while all farms are gone."""
        waited = 0.0
        while True:
            pool = self.scheduler.choose(self.pools(), time.monotonic(), request.task)
            if pool is not None:
                return pool
            if request.deadline_s is not None and waited >= request.deadline_s:
                self.report('grading_infra_failure', request_id=request.request_id, task=request.task,
                            attempts=attempts, last_reason='no grading pool available before the request deadline')
                raise GradingInfrastructureError('no grading pool available before the request deadline')
            if waited == 0 or int(waited) % 60 == 0:
                self.report('grading_no_pool', request_id=request.request_id, task=request.task,
                            waited_seconds=round(waited, 1))
            await asyncio.sleep(1)
            waited += 1
            if self.directory and self.farm_url and time.monotonic() - self.directory.last_refresh > 5:
                await self.directory.refresh()
            if self.open_directory and time.monotonic() - self.open_directory.last_refresh > 5:
                await self.open_directory.refresh()

    def _remember(self, request_id, result):
        if len(self.results) >= self.results_limit:
            self.results.pop(next(iter(self.results)))
        self.results[request_id] = result

    async def grade(self, request):
        """Grade once per request id: concurrent callers share one grade, later callers get its result."""
        await self._ensure_started()
        if request.request_id in self.results:
            self.report('grading_duplicate_request', request_id=request.request_id)
            return self.results[request.request_id]
        shared = self.inflight.get(request.request_id)
        if shared is None:
            shared = self.inflight[request.request_id] = [asyncio.ensure_future(self._grade(request)), 0]
        else:
            self.report('grading_duplicate_request', request_id=request.request_id, inflight=True)
        shared[1] += 1
        try:
            return await asyncio.shield(shared[0])
        except asyncio.CancelledError:
            # The last caller to give up cancels the grade itself (and so its farm request).
            if shared[1] == 1:
                shared[0].cancel()
            raise
        finally:
            shared[1] -= 1
            if shared[0].done() or shared[1] == 0:
                self.inflight.pop(request.request_id, None)

    async def _grade(self, request):
        if request.task not in self.families and self.local:
            raise GradingInfrastructureError(f'grading family {request.task!r} is not configured')
        attempts = 0  # charged infrastructure retries (cap: max_infra_retries)
        tries = 0     # submissions, each under its own wire id
        resume = self.journal.inflight.get(request.request_id) if self.journal else None
        while True:
            if tries and self.retry_jitter_s:
                await asyncio.sleep(random.uniform(0, self.retry_jitter_s))
            pool = None
            if resume:
                pool = next((p for p in self.pools() if getattr(p, 'url', None) == resume.get('url')), None)
                if pool is None:
                    self.report('grading_resume_skipped', request_id=request.request_id, url=resume.get('url'))
                    resume = None
            if pool is None:
                pool = await self._choose(request, attempts)
            submission = tries
            tries += 1
            pool.queued += 1
            started = time.monotonic()
            dispatched = [False]

            def on_started(p, wire_id):
                dispatched[0] = True
                p.queued -= 1
                p.running += 1
                if p.window is not None:
                    p.window += 1
                if self.journal and wire_id:
                    self.journal.append(event='submitted', request_id=request.request_id, wire_id=wire_id,
                                        url=getattr(p, 'url', None), pool=p.name, attempt=attempts,
                                        grader_sha256=getattr(p, 'grader_sha256', None))
            context = dict(request_id=request.request_id, task=request.task, pool=pool.name, attempt=attempts,
                           submission=submission)
            report = lambda event, **fields: self.report(event, **{**context, **fields})
            self.report('grading_dispatched', **context, predicted_wait_s=round(pool.predicted_wait(request.task), 1),
                        resumed=bool(resume))
            alternative = lambda pool=pool: any(
                p is not pool and p.healthy(time.monotonic()) and p.serves(request.task) and p.free_slot(request.task)
                for p in self.pools())
            try:
                result = await pool.run(request, on_started, submission, alternative=alternative, report=report,
                                        resume=(resume or {}).get('wire_id'))
                from .placement_runtime import reject_infrastructure_result
                reject_infrastructure_result(result)
            except asyncio.CancelledError:
                await pool.cancel(request.request_id)
                raise
            except Refused as exc:
                # No ACK (full, contract mismatch, unreachable): another pool, not a charged retry. This farm
                # waits out its retry_after (jittered) and then takes one submission at a time until ACKs resume.
                pool.full_until = time.monotonic() + exc.retry_after * random.uniform(1, 2)
                pool.window = 1
                pool.contract_mismatch = pool.contract_mismatch or exc.banned
                self.report('grading_farm_busy', **context, reason=str(exc)[:200])
                continue
            except Lost as exc:
                # ACKed but must move. Moving a long-queued request or a resumed id the farm forgot is free;
                # a confirmed-dead server or one that left the directory is a charged infrastructure retry.
                if exc.kind == 'poll_miss':
                    pool.unhealthy_until = time.monotonic() + 30
                elif exc.kind == 'queued_too_long':
                    pool.full_until = time.monotonic() + 30
                resume = None
                charged = exc.kind in ('poll_miss', 'retired')
                attempts += charged
                self.report('grading_reroute', **{**context, 'attempt': attempts}, kind=exc.kind, charged=charged,
                            reason=str(exc)[:200])
                if charged and attempts > self.max_infra_retries:
                    return self._give_up(request, attempts, exc)
                continue
            except Exception as exc:
                if not isinstance(exc, GradingInfrastructureError) and not is_infrastructure(exc):
                    raise
                attempts += 1
                resume = None
                pool.unhealthy_until = time.monotonic() + (30 if pool.kind == 'farm' else 5)
                self.report('grading_retry', **{**context, 'attempt': attempts}, from_pool=pool.name,
                            reason=f'{type(exc).__name__}: {exc}'[:300])
                await pool.cancel(request.request_id)
                if attempts > self.max_infra_retries:
                    return self._give_up(request, attempts, exc)
                continue
            finally:
                if dispatched[0]:
                    pool.running -= 1
                else:
                    pool.queued -= 1
            elapsed = time.monotonic() - started
            if request.request_id in self.results:
                # A late duplicate from an abandoned attempt never replaces the published result.
                self.report('grading_late_result_dropped', request_id=request.request_id, pool=pool.name)
                return self.results[request.request_id]
            metrics = result.get('metrics')
            if isinstance(metrics, dict):
                metrics.setdefault('grading_pool', pool.name)
                metrics.setdefault('grading_open', bool(getattr(pool, 'open', False)))
            self._remember(request.request_id, result)
            if self.journal:
                self.journal.append(event='result', request_id=request.request_id, result=result)
            pool.observe(elapsed)
            self.report('grading_completed', **context, seconds=round(elapsed, 2),
                        admission_wait_seconds=(result.get('metrics') or {}).get('admission_wait_seconds'),
                        open=bool(getattr(pool, 'open', False)),
                        cpu_model=(result.get('metrics') or {}).get('grading_cpu_model'))
            return result

    def _give_up(self, request, attempts, exc):
        """Past the retry cap: flag it and drop the candidate from its step (never a silent zero)."""
        self.report('grading_infra_failure', request_id=request.request_id, task=request.task, attempts=attempts,
                    last_reason=str(exc)[:300], dropped=True)
        if self.journal:
            self.journal.append(event='failed', request_id=request.request_id, reason=str(exc)[:300])
        raise GradingInfrastructureFailed(
            f'{request.task} grading failed on every pool after {attempts} infrastructure attempts: {exc}') from exc

    def grade_sync(self, request, timeout=None):
        """Block a worker thread (AC2's SAFE_GRADE_EXECUTOR) on the transport loop."""
        future = asyncio.run_coroutine_threadsafe(self.grade(request), self.loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise

    def submit(self, request):
        """Schedule a grade on the transport loop; returns a concurrent Future."""
        return asyncio.run_coroutine_threadsafe(self.grade(request), self.loop)

    async def close(self):
        for directory in (self.directory, self.open_directory):
            if directory and directory.task:
                directory.task.cancel()
                await asyncio.gather(directory.task, return_exceptions=True)
        if self.http is not None:
            await self.http.aclose()
