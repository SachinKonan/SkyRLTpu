"""Lease-fenced candidate grading on an inference farm's own Ray cluster.

The ingress owns the lease, so it also owns grading admission: every request
carries the lease token, results are retained briefly for the owning trainer
to fetch, and lease release/expiry cancels everything in flight. The ingress
never executes candidates; it dispatches the same Ray tasks trainer hosts use
(``tpu.science.ac2_grade.grade_ac2``, ``tpu.science.ray_cpu.grade``,
``tpu.science.placement_ray.grade_cpu_case``) with per-family Ray slot tokens
declared by ``bootstrap.workload_resources``.

Failure classes are explicit: ``error.class == 'infrastructure'`` (worker or
node loss, admission timeout, unit died before the candidate started) tells
the trainer to retry elsewhere; a candidate failure is encoded inside the
result and stays a reward-zero grade.
"""
import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import time

REQUEST_ID = re.compile(r'[0-9a-f]{32}')
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
TERMINAL = ('done', 'failed', 'cancelled')
MAX_PROGRAM_BYTES = 1024 * 1024
REFERENCE_AC2 = 'def construct_function():\n    return [1.0] * 1000\n'


@dataclass
class Entry:
    request_id: str
    task: str
    lease_id: str
    owner_run: str
    scope: dict
    state: str = 'queued'
    ref: object = None
    waiter: asyncio.Task | None = field(default=None, repr=False)
    result: dict | None = None
    error: dict | None = None
    host: str | None = None
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    fetched: float | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def view(self):
        metrics = dict((self.result or {}).get('metrics') or {})
        metrics.setdefault('queued_seconds', round((self.started or self.finished or time.monotonic()) - self.created, 3))
        if self.started and self.finished:
            metrics.setdefault('envelope_seconds', round(self.finished - self.started, 3))
        return dict(request_id=self.request_id, task=self.task, state=self.state, result=self.result,
                    error=self.error, host=self.host, metrics=metrics)


def infrastructure_error(exc):
    """Map a Ray-side exception to (class, detail)."""
    try:
        import ray.exceptions as rx
        if isinstance(exc, rx.TaskCancelledError):
            return 'cancelled', 'cancelled'
        if isinstance(exc, rx.RayTaskError):
            cause = getattr(exc, 'cause', None)
            detail = f'{type(cause).__name__ if cause else "RayTaskError"}: {str(cause or exc)[:800]}'
            return 'infrastructure', detail
        if isinstance(exc, (rx.WorkerCrashedError, rx.RayActorError, rx.NodeDiedError, rx.OwnerDiedError,
                            rx.ObjectLostError, rx.RaySystemError)):
            return 'infrastructure', f'{type(exc).__name__}: {str(exc)[:800]}'
    except ImportError:
        pass
    if isinstance(exc, asyncio.CancelledError):
        return 'cancelled', 'cancelled'
    return 'infrastructure', f'{type(exc).__name__}: {str(exc)[:800]}'


class GradingService:
    def __init__(self, config, ips, report=lambda *a, **k: None, *, warmup=True):
        self.config = config
        self.settings = config.grading
        self.families = config.grading_families
        self.ips = list(ips)
        self.report = report
        self.entries = {}
        self.ready = False
        self.readiness = {}
        self.counters = dict(completed=0, infra_failures=0, candidate_failures=0, cancelled=0)
        self.worker_root = os.environ.get('SCIENCE_WORKER_ROOT')
        self.reaper = asyncio.create_task(self._reap_loop())
        self.warmup_task = asyncio.create_task(self._warmup()) if warmup else None

    # -- capacity ---------------------------------------------------------
    def slots_total(self, task):
        return self.families[task]['slots_per_host'] * max(1, len(self.ips))

    def running(self, task=None):
        return sum(1 for e in self.entries.values() if e.state in ('queued', 'running')
                   and (task is None or e.task == task))

    def running_by_host(self):
        result = {}
        for entry in self.entries.values():
            if entry.state == 'running':
                result[entry.host or 'pending'] = result.get(entry.host or 'pending', 0) + 1
        return result

    def capacity(self, lease):
        families = {}
        for name, spec in self.families.items():
            families[name] = dict(spec, total=self.slots_total(name), running=self.running(name),
                                  queued=sum(1 for e in self.entries.values() if e.state == 'queued' and e.task == name))
        return dict(ready=self.ready, hosts=len(self.ips), families=families,
                    lease_id=lease['lease_id'] if lease else None, readiness=self.readiness)

    def snapshot(self):
        return dict(ready=self.ready, families=sorted(self.families), queue_depth=self.running(),
                    running_by_host=self.running_by_host(), retained_results=len(self.entries), **self.counters)

    # -- validation -------------------------------------------------------
    def validate(self, body):
        if not isinstance(body, dict):
            raise ValueError('grading request must be a JSON object')
        request_id, task, spec = body.get('request_id'), body.get('task'), body.get('spec')
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
            raise ValueError('request_id must be a 32-character hex id')
        if task not in self.families:
            raise ValueError(f'grading family {task!r} is not served by this farm')
        if not isinstance(spec, dict):
            raise ValueError('spec must be an object')
        limit = self.settings.stdout_limit_bytes
        if task == 'ac2':
            code, name = spec.get('program_code'), spec.get('function_name')
            if not isinstance(code, str) or not code or len(code.encode()) > MAX_PROGRAM_BYTES:
                raise ValueError('program_code must be a nonempty string of at most 1 MiB')
            if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
                raise ValueError('function_name must be a Python identifier')
            timeout = spec.get('eval_timeout_seconds')
            if type(timeout) is not int or not 1 <= timeout <= 7200:
                raise ValueError('eval_timeout_seconds must be an integer in [1,7200]')
            admission = spec.get('admission_timeout_s', timeout)
            if type(admission) is not int or admission < 1:
                raise ValueError('admission_timeout_s must be a positive integer')
            return dict(program_code=code, function_name=name, eval_timeout_seconds=timeout,
                        admission_timeout_s=admission,
                        stdout_limit_bytes=min(limit, int(spec.get('stdout_limit_bytes') or limit)),
                        systemd=bool(self.settings.local_systemd))
        source = spec.get('source')
        if not isinstance(source, str) or not source or len(source.encode()) > MAX_PROGRAM_BYTES:
            raise ValueError('source must be a nonempty string of at most 1 MiB')
        admission = spec.get('admission_timeout_s', 2400)
        if type(admission) is not int or admission < 1:
            raise ValueError('admission_timeout_s must be a positive integer')
        slots = spec.get('slots_per_host', self.families[task]['slots_per_host'])
        if type(slots) is not int or slots < 1:
            raise ValueError('slots_per_host must be a positive integer')
        contract = spec.get('resource_contract')
        if contract is not None and not isinstance(contract, dict):
            raise ValueError('resource_contract must be an object or null')
        if task == 'routing':
            suite = spec.get('routing_suite', 'full')
            if suite not in ('full', 'q20'):
                raise ValueError('routing_suite must be full or q20')
            return dict(source=source, routing_suite=suite, resource_contract=contract,
                        slots_per_host=slots, admission_timeout_s=admission)
        case = spec.get('case')
        if not isinstance(case, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', case):
            raise ValueError('case must be a short identifier')
        helper = spec.get('helper', 'none')
        if helper not in ('none', 'fast_proxy_v1'):
            raise ValueError('unknown placement helper')
        return dict(source=source, case=case, resource_contract=contract, slots_per_host=slots,
                    admission_timeout_s=admission, helper=helper)

    # -- dispatch (monkeypatchable seam) ---------------------------------
    def dispatch(self, task, spec):
        """Return a Ray ObjectRef (or any awaitable) for one candidate."""
        root = self.worker_root
        if task == 'ac2':
            from tpu.science.ac2_grade import grade_ac2
            family = self.families['ac2']
            return grade_ac2.options(num_cpus=family['cpus'], memory=family['memory_gib'] * 1024 ** 3,
                                     resources={'grading_ac2': 1}, scheduling_strategy='SPREAD'
                                     ).remote(spec, self.families, root)
        if task == 'routing':
            from tpu.science.ray_cpu import grade
            family = self.families['routing']
            return grade.options(num_cpus=family['cpus'], memory=family['memory_gib'] * 1024 ** 3,
                                 resources={'grading_routing': 1}, scheduling_strategy='SPREAD'
                                 ).remote('routing', spec['source'], root, admission_timeout_s=spec['admission_timeout_s'],
                                          slots_per_host=spec['slots_per_host'], routing_suite=spec['routing_suite'],
                                          **({'resource_contract': spec['resource_contract']} if spec['resource_contract'] else {}))
        from tpu.science.placement_ray import grade_cpu_case
        family = self.families['placement']
        return grade_cpu_case.options(memory=family['memory_gib'] * 1024 ** 3,
                                      resources={'placement_cpu_host': 1, 'grading_placement': 1},
                                      scheduling_strategy='SPREAD'
                                      ).remote(spec['source'], spec['case'], root, admission_timeout_s=spec['admission_timeout_s'],
                                               slots_per_host=spec['slots_per_host'], helper=spec['helper'],
                                               **({'resource_contract': spec['resource_contract']} if spec['resource_contract'] else {}))

    def cancel_ref(self, ref):
        if hasattr(ref, 'future'):
            import ray
            ray.cancel(ref, force=True)
        elif hasattr(ref, 'cancel'):
            ref.cancel()

    @staticmethod
    def awaitable(ref):
        return asyncio.wrap_future(ref.future()) if hasattr(ref, 'future') else ref

    # -- lifecycle --------------------------------------------------------
    async def submit(self, lease, body):
        """Idempotent by request_id; returns (http_status, view)."""
        try:
            spec = self.validate(body)
        except ValueError as exc:
            return 400, dict(detail=str(exc))
        request_id, task = body['request_id'], body['task']
        existing = self.entries.get(request_id)
        if existing is not None:
            if existing.lease_id != lease['lease_id']:
                return 409, dict(detail='request belongs to another lease')
            return 200, existing.view()
        if len(self.entries) >= self.settings.max_requests:
            return 429, dict(detail='grading registry full', retry_after=5)
        if self.running(task) >= self.slots_total(task) * self.settings.queue_factor:
            return 429, dict(detail='grading queue full', retry_after=5)
        entry = Entry(request_id=request_id, task=task, lease_id=lease['lease_id'],
                      owner_run=lease['owner_run'], scope=body.get('scope') or {})
        self.entries[request_id] = entry
        entry.waiter = asyncio.create_task(self._run(entry, spec))
        self._event('grading_submitted', request_id=request_id, task=task, owner_run=entry.owner_run)
        return 202, entry.view()

    async def _run(self, entry, spec):
        try:
            entry.ref = self.dispatch(entry.task, spec)
            entry.state = 'running'
            entry.started = time.monotonic()
            result = await self.awaitable(entry.ref)
            if not isinstance(result, dict):
                raise RuntimeError('executor returned a non-dict result')
            entry.result = result
            entry.host = (result.get('metrics') or {}).get('host')
            entry.state = 'done'
            if result.get('error') or result.get('correctness') == 0:
                self.counters['candidate_failures'] += 1
            else:
                self.counters['completed'] += 1
        except asyncio.CancelledError:
            entry.state = 'cancelled'
            entry.error = dict(**{'class': 'cancelled'}, detail='cancelled')
            self.counters['cancelled'] += 1
        except Exception as exc:
            kind, detail = infrastructure_error(exc)
            entry.state = 'cancelled' if kind == 'cancelled' else 'failed'
            entry.error = dict(**{'class': kind}, detail=detail)
            self.counters['cancelled' if kind == 'cancelled' else 'infra_failures'] += 1
        finally:
            entry.finished = time.monotonic()
            entry.done.set()
            self._event('grading_finished', request_id=entry.request_id, task=entry.task, state=entry.state,
                        error=(entry.error or {}).get('class'))

    async def result(self, lease, request_id, wait=0):
        entry = self.entries.get(request_id)
        if entry is None:
            return None
        if entry.lease_id != lease['lease_id']:
            return None
        if wait > 0 and not entry.done.is_set():
            try:
                await asyncio.wait_for(entry.done.wait(), timeout=min(wait, self.settings.long_poll_seconds))
            except (TimeoutError, asyncio.TimeoutError):
                pass
        if entry.state in TERMINAL and entry.fetched is None:
            entry.fetched = time.monotonic()
        return entry.view()

    async def cancel(self, request_id, reason):
        entry = self.entries.get(request_id)
        if entry is None:
            return None
        if entry.state not in TERMINAL:
            if entry.ref is not None:
                try:
                    self.cancel_ref(entry.ref)
                except Exception:
                    pass
            if entry.waiter is not None and not entry.waiter.done():
                entry.waiter.cancel()
                await asyncio.gather(entry.waiter, return_exceptions=True)
            self._event('grading_cancelled', request_id=request_id, reason=reason)
        return entry.view()

    async def cancel_all(self, reason):
        active = [e.request_id for e in self.entries.values() if e.state not in TERMINAL]
        for request_id in active:
            await self.cancel(request_id, reason)
        if active:
            self._event('grading_cancel_all', reason=reason, requests=len(active))
        return len(active)

    async def drain(self, timeout):
        deadline = time.monotonic() + timeout
        while self.running() and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        return self.running() == 0

    def _event(self, event, **fields):
        try:
            self.report(event, **fields)
        except Exception:
            pass

    def reap(self, now=None):
        """Drop terminal results after the retention window (or its ceiling)."""
        now = time.monotonic() if now is None else now
        retention, ceiling = self.settings.result_retention_seconds, 4 * self.settings.result_retention_seconds
        removed = 0
        for request_id, entry in list(self.entries.items()):
            if entry.state not in TERMINAL:
                continue
            expired = (entry.fetched is not None and now - entry.fetched >= retention) or \
                      (entry.finished is not None and now - entry.finished >= ceiling)
            if expired:
                del self.entries[request_id]
                removed += 1
        return removed

    async def _reap_loop(self):
        while True:
            await asyncio.sleep(5)
            self.reap()

    async def _warmup(self):
        """One reference grade per family per host marks the farm ready."""
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            if not ray.is_initialized():
                self.readiness = dict(error='ray not initialized')
                return
            nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
            for name in self.families:
                for ip in self.ips:
                    node = nodes.get(ip)
                    if node is None:
                        self.readiness[f'{name}@{ip}'] = 'node missing'
                        continue
                    strategy = NodeAffinitySchedulingStrategy(node, soft=False)
                    if name == 'ac2':
                        from tpu.science.ac2_grade import grade_ac2
                        family = self.families['ac2']
                        spec = dict(program_code=REFERENCE_AC2, function_name='construct_function',
                                    eval_timeout_seconds=60, admission_timeout_s=600, stdout_limit_bytes=1024,
                                    systemd=bool(self.settings.local_systemd))
                        ref = grade_ac2.options(num_cpus=family['cpus'], memory=family['memory_gib'] * 1024 ** 3,
                                                resources={'grading_ac2': 1}, scheduling_strategy=strategy
                                                ).remote(spec, self.families, self.worker_root)
                        result = await self.awaitable(ref)
                        ok = isinstance(result, dict) and result.get('result') == [1.0] * 1000
                        self.readiness[f'{name}@{ip}'] = 'ok' if ok else f'reference mismatch: {str(result)[:200]}'
                    else:
                        root = self.worker_root

                        def prepared(root=root):
                            return bool(root) and Path(root, '.science/ready.json').is_file()
                        ref = ray.remote(num_cpus=0.1, scheduling_strategy=strategy)(prepared).remote()
                        ok = await self.awaitable(ref)
                        self.readiness[f'{name}@{ip}'] = 'ok' if ok else 'science worker not prepared'
            self.ready = all(v == 'ok' for v in self.readiness.values()) and bool(self.readiness)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.readiness = dict(error=f'{type(exc).__name__}: {exc}'[:400])
        self._event('grading_readiness', ready=self.ready, readiness=self.readiness)

    async def close(self):
        await self.cancel_all('service closing')
        for task in (self.reaper, self.warmup_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
