"""Task-tagged candidate grading over local Ray slots and leased farms.

The training client grades candidates through one transport: a local pool
(Ray tasks on the trainer's own cluster, the same executors as before) plus
one pool per farm lease the trainer's ingress currently holds (fetched from
``GET $SKYRL_GRADING_URL/skyrl/v1/grading/farms``). Every request keeps one
``request_id`` across retries so a farm's late result can never enter the
batch twice (first result wins). Infrastructure failures retry on another
pool up to ``max_infra_retries`` and then raise ``GradingInfrastructureError``
(``abort_training_step``), exactly like the science graders; candidate
failures are returned as results and stay reward zero.

The transport runs its own asyncio loop in a background thread so the AC2
grader (a worker-thread call) and the science graders (coroutines on the
client's loop) share one scheduler.
"""
import asyncio
from dataclasses import dataclass, field
import json
import os
import threading
import time
import uuid

import httpx

INFRA_MARKERS = ('infrastructure', 'admission timed out', 'before the candidate started', 'without a result file',
                 'WorkerCrashedError', 'RayActorError', 'NodeDiedError', 'OwnerDiedError', 'ObjectLostError',
                 'RaySystemError', 'cpu_scheduler')


class GradingInfrastructureError(RuntimeError):
    """No pool could run the candidate; the training step must not use a fake zero."""
    abort_training_step = True


@dataclass
class GradingRequest:
    task: str
    spec: dict
    scope: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    admission_timeout_s: int = 1100
    deadline_s: float | None = None


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
        self.eligible = True

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
        return self.eligible and now >= self.unhealthy_until

    async def run(self, request, on_started):
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
        if request.task == 'ac2':
            from tpu.science.ac2_grade import grade_ac2
            family = self.families['ac2']
            spec = dict(request.spec, systemd=self.systemd, admission_timeout_s=request.admission_timeout_s)
            return grade_ac2.options(num_cpus=family['cpus'], memory=family['memory_gib'] * 1024 ** 3,
                                     resources={'grading_ac2': 1}, scheduling_strategy='SPREAD'
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
            options['memory'] = 4 * 1024 ** 3
        return grade_cpu_case.options(**options).remote(
            spec['source'], spec['case'], self.root, admission_timeout_s=request.admission_timeout_s,
            slots_per_host=spec['slots_per_host'], helper=spec.get('helper', 'none'),
            **({'resource_contract': spec['resource_contract']} if spec.get('resource_contract') else {}))

    @staticmethod
    def awaitable(ref):
        return asyncio.wrap_future(ref.future()) if hasattr(ref, 'future') else ref

    async def run(self, request, on_started):
        ref = self.dispatch(request)
        self.refs[request.request_id] = ref
        on_started(self)
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

    def __init__(self, farm, http, *, long_poll_seconds=20, poll_seconds=2.0, capacity=None):
        super().__init__(farm['farm_id'], capacity or farm.get('grading_capacity') or 16)
        self.url = farm['url'].rstrip('/')
        self.token = farm['token']
        self.http = http
        self.long_poll_seconds = long_poll_seconds
        self.poll_seconds = poll_seconds
        self.eligible = bool(farm.get('eligible', True))
        self.capacities = {}
        self.ready = None

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
            response = await self.http.get(self.url + '/skyrl/v1/grading/capacity', headers=self.headers, timeout=10)
            response.raise_for_status()
            capacity = response.json()
        except Exception:
            return False
        self.capacities = {name: spec.get('total', 0) for name, spec in (capacity.get('families') or {}).items()}
        self.ready = bool(capacity.get('ready', True))
        return True

    async def run(self, request, on_started):
        body = dict(request_id=request.request_id, task=request.task, owner_run=os.environ.get('SKYRL_RUN_ID', ''),
                    scope=request.scope, spec=dict(request.spec, admission_timeout_s=request.admission_timeout_s))
        while True:
            response = await self.http.post(self.url + '/skyrl/v1/grading/submit', json=body, headers=self.headers,
                                            timeout=30)
            if response.status_code == 429:
                # Farm backpressure is a wait, never a retry: the request keeps
                # its queued slot in this pool's accounting.
                await asyncio.sleep(float(response.json().get('retry_after', 5)))
                continue
            response.raise_for_status()
            break
        on_started(self)
        while True:
            if not self.eligible:
                raise GradingInfrastructureError(f'farm {self.name}: lease no longer eligible')
            response = await self.http.get(f'{self.url}/skyrl/v1/grading/result/{request.request_id}',
                                           params={'wait': self.long_poll_seconds}, headers=self.headers,
                                           timeout=self.long_poll_seconds + 30)
            response.raise_for_status()
            view = response.json()
            state = view.get('state')
            if state == 'done':
                result = view.get('result')
                if not isinstance(result, dict):
                    raise GradingInfrastructureError(f'farm {self.name}: done without a result')
                return result
            if state in ('failed', 'cancelled'):
                error = view.get('error') or {}
                raise GradingInfrastructureError(f'farm {self.name}: {error.get("class")}: {error.get("detail")}')
            await asyncio.sleep(0 if self.long_poll_seconds else self.poll_seconds)

    async def cancel(self, request_id):
        try:
            await self.http.post(f'{self.url}/skyrl/v1/grading/cancel/{request_id}', headers=self.headers, timeout=10)
        except Exception:
            pass


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
        try:
            response = await self.http.get(self.url + '/skyrl/v1/grading/farms', timeout=10)
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
                pool = self.pools[farm_id] = FarmPool(farm, self.http, long_poll_seconds=self.long_poll_seconds,
                                                      poll_seconds=self.poll_seconds)
                await pool.probe_capacity()
                added.append(farm_id)
            else:
                pool.eligible = bool(farm.get('eligible', True))
                if pool.ready is not True and pool.eligible:
                    await pool.probe_capacity()
        for farm_id in list(self.pools):
            if farm_id not in seen:
                self.pools[farm_id].eligible = False
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
    def choose(self, pools, now, task=None):
        candidates = [p for p in pools if p.healthy(now)]
        if not candidates:
            return None
        return min(candidates, key=lambda p: (p.predicted_wait(task), p.kind != 'local', p.name))


class GradingTransport:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, *, families, local_capacity, farm_url=None, root=None, local_systemd=True, max_infra_retries=3,
                 events=None, refresh_seconds=10, long_poll_seconds=20, poll_seconds=2.0, http=None, loop=None):
        self.families = families
        self.max_infra_retries = max_infra_retries
        self.events = events
        # First result wins: a request id graded once returns the same result
        # to every later caller instead of running the candidate again.
        self.results = {}
        self.results_limit = 8192
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
                   max_infra_retries=int(env.get('SKYRL_GRADING_MAX_INFRA_RETRIES') or 3),
                   events=env.get('SKYRL_GRADING_EVENTS') or None,
                   refresh_seconds=float(env.get('SKYRL_GRADING_FARM_REFRESH_SECONDS') or 10),
                   long_poll_seconds=int(env.get('SKYRL_GRADING_LONG_POLL_SECONDS') or 20))
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
            self.http = httpx.AsyncClient()
        self.directory = FarmDirectory(self.farm_url, self.http, refresh_seconds=self.refresh_seconds,
                                       long_poll_seconds=self.long_poll_seconds, poll_seconds=self.poll_seconds,
                                       report=self.report)
        if self.farm_url:
            await self.directory.refresh()
            self.directory.task = asyncio.create_task(self.directory.loop())

    def pools(self):
        pools = [self.local] if self.local else []
        if self.directory:
            pools += list(self.directory.pools.values())
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

    def _remember(self, request_id, result):
        if len(self.results) >= self.results_limit:
            self.results.pop(next(iter(self.results)))
        self.results[request_id] = result

    async def grade(self, request):
        await self._ensure_started()
        if request.request_id in self.results:
            self.report('grading_duplicate_request', request_id=request.request_id)
            return self.results[request.request_id]
        if request.task not in self.families and self.local:
            raise GradingInfrastructureError(f'grading family {request.task!r} is not configured')
        attempts = 0
        while True:
            pool = await self._choose(request, attempts)
            pool.queued += 1
            started = time.monotonic()
            dispatched = [False]

            def on_started(p):
                dispatched[0] = True
                p.queued -= 1
                p.running += 1
            self.report('grading_dispatched', request_id=request.request_id, task=request.task, pool=pool.name,
                        attempt=attempts, predicted_wait_s=round(pool.predicted_wait(request.task), 1))
            try:
                result = await pool.run(request, on_started)
            except asyncio.CancelledError:
                await pool.cancel(request.request_id)
                raise
            except Exception as exc:
                if not isinstance(exc, GradingInfrastructureError) and not is_infrastructure(exc):
                    raise
                attempts += 1
                pool.unhealthy_until = time.monotonic() + (30 if pool.kind == 'farm' else 5)
                self.report('grading_retry', request_id=request.request_id, from_pool=pool.name, attempt=attempts,
                            reason=f'{type(exc).__name__}: {exc}'[:300])
                await pool.cancel(request.request_id)
                if attempts > self.max_infra_retries:
                    self.report('grading_infra_failure', request_id=request.request_id, task=request.task,
                                attempts=attempts, last_reason=str(exc)[:300])
                    raise GradingInfrastructureError(
                        f'{request.task} grading failed on every pool after {attempts} attempts: {exc}') from exc
                continue
            finally:
                if dispatched[0]:
                    pool.running -= 1
                else:
                    pool.queued -= 1
            elapsed = time.monotonic() - started
            if request.request_id in self.results:
                # A late duplicate from an abandoned attempt never replaces the
                # published result.
                self.report('grading_late_result_dropped', request_id=request.request_id, pool=pool.name)
                return self.results[request.request_id]
            self._remember(request.request_id, result)
            pool.observe(elapsed)
            self.report('grading_completed', request_id=request.request_id, pool=pool.name, seconds=round(elapsed, 2),
                        admission_wait_seconds=(result.get('metrics') or {}).get('admission_wait_seconds'))
            return result

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
        if self.directory and self.directory.task:
            self.directory.task.cancel()
            await asyncio.gather(self.directory.task, return_exceptions=True)
        if self.http is not None:
            await self.http.aclose()
