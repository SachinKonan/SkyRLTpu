"""Della grader gateway: one grading endpoint on della-vis1 for GCP trainers, reached over one reverse tunnel.

GCP hosts cannot open connections to della (the campus firewall drops all inbound), but this controller host
reaches both: the della grading farms directly (discovery's `--slurm-pool` probe) and the trainers over SSH.
With `discovery --grader-gateway --push`:

- The gateway (`Broker`, an asyncio Starlette app on 127.0.0.1:--gateway-port in a thread of the discovery
  process) speaks the farm grading API (`/skyrl/v1/grading/{capacity,submit,results,result/<id>,cancel/<id>}`,
  `/status`, `/health`). A trainer grades on it exactly as on an open farm (`grading_transport.OpenFarmPool`).
- For every trainer rank-0 host (trainer pools, job ids, private trainers) discovery keeps ONE
  `ssh -N -R 127.0.0.1:<remote port>:127.0.0.1:<gateway port> <host>` and, once per tick over one more SSH call,
  writes the open-grading registry on that host (`--gateway-registry-path`) with one entry, the tunnel's URL,
  only if the tunnel answered `/status` *on the trainer host* with this gateway's instance id.
- The trainer's profile sets SKYRL_OPEN_GRADING=1, SKYRL_GRADING_REGISTRY=<that path>, SKYRL_RUN_ID=<run id>
  (client_env); no trainer code changes.

Routing. Each submit is queued here per (family, run) and dispatched round-robin across runs to the della farm
with the most free slots for that family and the request's grader hash. Free slots = the farm's slots for the
family minus its load from other clients (its advertised `running` less ours, refreshed every few seconds)
minus our grades on it: the gateway never sends a farm more than its slots. The submit is forwarded with the
trainer's own X-Grading-Run / X-Grading-Token, so the farm's per-run fairness (and its 1.5x open admission)
still applies across these trainers and the RQ4 della clients. Polls are batched per (farm, run, token).

Admission and fairness. The capacity a run sees is the sum of the compatible farms' slots, so its load
balancer sends many grades at once. A family admits up to 1.5x those slots outstanding; past that a run is
still admitted up to its fair share (limit / runs with work), anything else gets 429 (retry_after), which the
trainer treats as "full" (no infrastructure retry charged).

Priority. Optional, from `--gateway-priority PREFIX=N` (lower N first; runs matching no prefix get
DEFAULT_PRIORITY). With no entries the gateway behaves exactly as above. With entries:
- dispatch serves the best priority class first, round-robin within a class;
- a run is refused (429) while a run of better priority has grades queued here, and admission counts only the
  grades of runs at its own or better priority, so a lower class's backlog never blocks a better one;
- grades of runs below the best configured class are dispatched only while more than `--gateway-reserve`
  (a fraction of the compatible slots) stay free. There is no preemption: a dispatched grade holds its slot up
  to its time cap, so this reserve is what keeps slots open for the better class.

Loss. A farm that forgets a grade (restart, TTL), drains, or stops answering for FARM_LOST_S moves its grades
back to the queue here (at their original age), up to MAX_LOSSES times, then reports an infrastructure
failure, which the trainer retries elsewhere. The gateway keeps no state across a discovery restart: in-flight
grades are unknown to the new process, the trainer's polls fail, and its request protocol declares them LOST
(poll_miss) and resubmits them (one charged infrastructure retry of 5 per grade); the abandoned farm grades
expire on the farm after its request TTL (300 s). Nothing is lost silently.
"""
import asyncio
from collections import Counter
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import uuid

import httpx

DEFAULT_REGISTRY = '/home/gcpuser/.cache/skyrl/della-graders.json'
DEFAULT_PORT = 28650          # gateway listen port on della-vis1 (127.0.0.1 only)
DEFAULT_REMOTE_PORT = 26000   # trainer-host port; fallback slot +1000
ENTRY_NAME = 'della-gateway'
GATEWAY_RUN = 'della-gateway'  # the gateway's own identity for capacity probes

QUEUE_FACTOR = 1.5            # outstanding grades admitted per family slot (as an open farm)
TTL_S = 300.0                 # a grade its trainer stopped polling for this long is cancelled
RETENTION_S = 600.0           # terminal grades kept for their trainer
FETCHED_RETENTION_S = 60.0
LONG_POLL_S = 20
PROBE_S = 5.0                 # farm capacity refresh
FARM_LOST_S = 60.0            # a farm failing every probe/poll this long loses its grades back to the queue
UNKNOWN_POLLS_LOST = 2        # consecutive polls in which the farm does not know the id
MAX_LOSSES = 3
DEFAULT_PRIORITY = 1          # runs matching no --gateway-priority prefix
RESERVE_FRACTION = 0.25       # slots kept free from runs below the best configured priority
NO_FARM_RETRY_S = 60
FORGET_UNLISTED_S = 600.0
REQUEST_ID = re.compile(r'[0-9a-f]{32}')
TERMINAL = ('done', 'failed', 'cancelled')
ON_FARM = ('dispatching', 'dispatched')

FAILED_STARTS_BEFORE_SLOT_SWITCH = 2
MISSED_CHECKS_BEFORE_RESTART = 3
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ExitOnForwardFailure=yes',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4']


# ---------------------------------------------------------------------------------------------- the gateway
class Farm:
    def __init__(self, url, name):
        self.url, self.name = url.rstrip('/'), name
        self.listed, self.unlisted_since = True, None
        self.capacity = None          # the farm's last /capacity answer
        self.instance = None
        self.others = {}              # family -> load from other clients at the last probe
        self.ours = Counter()         # family -> our grades on it (dispatching or dispatched)
        self.full_until = {}          # family -> monotonic
        self.unhealthy_until = 0.0
        self.failing_since = None

    def families(self):
        return (self.capacity or {}).get('families') or {}

    def total(self, task):
        return int((self.families().get(task) or {}).get('total') or 0)

    def grader(self):
        return (self.capacity or {}).get('grader_sha256')

    def compatible(self, grader):
        return grader is None or self.grader() == grader

    def usable(self, now):
        capacity = self.capacity or {}
        return bool(self.listed and capacity.get('ready') and not capacity.get('draining')
                    and self.failing_since is None and now >= self.unhealthy_until)

    def free(self, task):
        return self.total(task) - self.others.get(task, 0) - self.ours[task]


@dataclass(eq=False)
class Grade:
    wire_id: str
    run: str
    token: str
    task: str
    body: dict = field(repr=False)
    grader: str | None
    age: float
    state: str = 'queued'             # queued | dispatching | dispatched | done | failed | cancelled
    farm: Farm | None = None
    view: dict | None = field(default=None, repr=False)
    created: float = field(default_factory=time.monotonic)
    expires: float = 0.0
    losses: int = 0
    attempt: int = 0                  # farm-side submissions that may have been accepted
    unknown: int = 0
    finished: float | None = None
    fetched: float | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def terminal(self):
        return self.state in TERMINAL

    @property
    def headers(self):
        return {'X-Grading-Run': self.run, 'X-Grading-Token': self.token}

    @property
    def farm_id(self):
        """The id on the farm: a farm keeps a cancelled id as cancelled, so each new attempt gets its own."""
        return (self.wire_id if not self.attempt else
                hashlib.sha256(f'{self.wire_id}:gateway:{self.attempt}'.encode()).hexdigest()[:32])

    def public_view(self, now):
        if self.view is not None and (self.terminal or self.state == 'dispatched'):
            return dict(self.view, request_id=self.wire_id)
        return dict(request_id=self.wire_id, task=self.task, state='queued', result=None, error=None, host=None,
                    metrics=dict(queued_seconds=round(now - self.created, 3)))


def cancelled_view(grade, detail, kind='cancelled'):
    return dict(request_id=grade.wire_id, task=grade.task, state='cancelled', result=None,
                error={'class': kind, 'detail': detail}, host=None, metrics={})


def failed_view(grade, detail):
    return dict(request_id=grade.wire_id, task=grade.task, state='failed', result=None,
                error={'class': 'infrastructure', 'detail': detail[:800]}, host=None, metrics={})


def payload_of(response):
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_priorities(specs):
    """['rl28k-=0', 'codex-bench-=2'] -> [('rl28k-', 0), ('codex-bench-', 2)]; raises ValueError."""
    parsed = []
    for spec in specs or ():
        prefix, sep, value = str(spec).rpartition('=')
        if not sep or not prefix:
            raise ValueError(f'--gateway-priority wants PREFIX=N, got {spec!r}')
        parsed.append((prefix, int(value)))
    return parsed


def parse_queue_deadlines(specs):
    """['rl28k-=900'] -> [('rl28k-', 900.0)]; raises ValueError.

    Only runs matching a prefix get a deadline, and that is deliberate: a trainer on a TPU can grade
    locally if the gateway gives its grade back, so expiring is strictly better than waiting. A client
    with no local grading (the della-hosted benchmark runs) has nowhere else to go, so expiring would
    just destroy its work -- those runs must match no prefix and keep waiting.
    """
    parsed = []
    for spec in specs or ():
        prefix, sep, value = str(spec).rpartition('=')
        if not sep or not prefix:
            raise ValueError(f'--gateway-queue-deadline wants PREFIX=SECONDS, got {spec!r}')
        seconds = float(value)
        # nan/inf pass `> 0` but make the expiry comparison never true, so a typo would silently
        # disable the deadline for that run instead of being rejected at startup.
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f'--gateway-queue-deadline wants a finite SECONDS > 0, got {spec!r}')
        parsed.append((prefix, seconds))
    return sorted(parsed, key=lambda p: -len(p[0]))


class Broker:
    """All gateway state; runs on one event loop (no locks)."""

    def __init__(self, http=None, *, instance=None, now=time.monotonic, probe_seconds=PROBE_S,
                 long_poll_seconds=LONG_POLL_S, retry_seconds=2.0, emit=lambda **k: None, priorities=(),
                 reserve=RESERVE_FRACTION, queue_deadlines=()):
        self.http = http or httpx.AsyncClient(limits=httpx.Limits(max_connections=512, max_keepalive_connections=128))
        self.instance = instance or uuid.uuid4().hex
        self.token = uuid.uuid4().hex
        self.now, self.emit = now, emit
        self.probe_seconds, self.long_poll_seconds, self.retry_seconds = probe_seconds, long_poll_seconds, retry_seconds
        self.queue_deadlines = list(queue_deadlines)
        self.farms = {}               # url -> Farm
        self.grades = {}              # wire id -> Grade
        self.queues = {}              # family -> {run: [Grade] oldest first}
        self.turn = {}                # family -> run served last
        self.run_grader = {}          # run -> grader hash it submits with
        self.groups = {}              # (farm url, run, token) -> {farm id: Grade}
        self.pollers = {}
        self.spawned = set()
        self.background = []
        self.counters = Counter()
        self.priorities = sorted(priorities, key=lambda entry: -len(entry[0]))  # longest prefix wins
        self.best = min((n for _, n in self.priorities), default=None)
        self.reserve = float(reserve)

    def priority(self, run):
        return next((n for prefix, n in self.priorities if run.startswith(prefix)), DEFAULT_PRIORITY)

    def queue_deadline(self, run):
        """Seconds a queued grade of this run may wait while NO farm can serve it, or None to wait forever."""
        return next((d for prefix, d in self.queue_deadlines if run.startswith(prefix)), None)

    def _expire_unservable(self):
        """Hand back grades that no farm can serve and whose run has a deadline, as an infrastructure
        failure so the client can grade them locally. Only fires while slots(task) == 0: if a farm exists
        but is busy, queueing is correct and nothing expires."""
        if not self.queue_deadlines:
            return
        now = self.now()
        for task, runs in list(self.queues.items()):
            for run, queue in list(runs.items()):
                deadline = self.queue_deadline(run)
                if deadline is None or not queue:
                    continue
                # Per grade, not per queue: one run's queue can hold grades submitted with different
                # grader hashes, and slots() is grader-specific, so a shared check could expire a grade
                # whose own grader does have a compatible farm.
                stale = [g for g in queue
                         if g.state == 'queued' and now - g.created >= deadline
                         and self.slots(task, g.grader) == 0]
                if not stale:
                    continue
                # Leave the queue consistent BEFORE finishing anything: _finish re-enters _pump, and
                # _pump assumes every run key holds a non-empty list. self.turn is deliberately left
                # alone -- it only records the run served last, a stale name sorts harmlessly, and
                # clearing it would restart round-robin and favour the alphabetically-first run.
                for grade in stale:
                    queue.remove(grade)
                if not queue:
                    runs.pop(run, None)
                for grade in stale:
                    # Finish FIRST, then count, then log. The emitter is a plain print() in production,
                    # so a BrokenPipeError between removing a grade from the queue and finishing it would
                    # leave it detached: owned by nobody, never completed, with its waiter still polling.
                    waited = now - grade.created
                    self._finish(grade, failed_view(
                        grade, f'no della farm could serve it for {waited:.0f}s '
                               f'(deadline {deadline:.0f}s); grade it locally'))
                    self.counters['expired_unservable'] = self.counters.get('expired_unservable', 0) + 1
                    try:
                        self.emit(event='gateway_grade_expired', request_id=grade.wire_id, run=run,
                                  task=task, queued_seconds=round(waited, 1), deadline_seconds=deadline)
                    except Exception:
                        pass

    def reserved(self, run):
        """True if this run's grades must leave the reserve free (it is below the best configured class)."""
        return self.best is not None and self.priority(run) > self.best

    def reserve_open(self, task, grader):
        """A reserved run may take a slot only while more than the reserve stays free."""
        farms = self.compatible_farms(task, grader)
        free = sum(max(0, f.free(task)) for f in farms)
        return free > math.ceil(sum(f.total(task) for f in farms) * self.reserve)

    def start(self):
        self.background = [asyncio.create_task(self._probe_loop()), asyncio.create_task(self._reap_loop()),
                           asyncio.create_task(self._expire_loop())]

    async def close(self):
        tasks = self.background + list(self.pollers.values()) + list(self.spawned)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.http.aclose()

    def _spawn(self, coroutine):
        task = asyncio.ensure_future(coroutine)
        self.spawned.add(task)
        task.add_done_callback(self.spawned.discard)
        return task

    # -- farms (from discovery) ------------------------------------------------------------------------------
    def set_farms(self, records):
        """Ready della grading farms from discovery: [{'url', 'name'}]. An unlisted farm takes no new grades
        but keeps polling the ones it holds (one missed discovery probe must not move running grades)."""
        now = self.now()
        listed = {r['url'].rstrip('/'): r for r in records}
        for url, record in listed.items():
            farm = self.farms.get(url)
            if farm is None:
                farm = self.farms[url] = Farm(url, record['name'])
                self._spawn(self._probe(farm))
                self.emit(event='gateway_farm_added', farm=farm.name, url=url)
            farm.listed, farm.unlisted_since, farm.name = True, None, record['name']
        for url, farm in list(self.farms.items()):
            if url in listed:
                continue
            if farm.listed:
                farm.listed, farm.unlisted_since = False, now
                self.emit(event='gateway_farm_unlisted', farm=farm.name, ours=sum(farm.ours.values()))
            elif now - farm.unlisted_since > FORGET_UNLISTED_S and not sum(farm.ours.values()):
                del self.farms[url]
        self._pump_all()

    async def _probe(self, farm):
        try:
            response = await self.http.get(farm.url + '/skyrl/v1/grading/capacity', timeout=httpx.Timeout(10, pool=None),
                                           headers={'X-Grading-Run': GATEWAY_RUN, 'X-Grading-Token': self.token})
            response.raise_for_status()
            capacity = response.json()
        except Exception as exc:
            self._farm_failed(farm, f'capacity probe: {type(exc).__name__}')
            return
        instance = capacity.get('instance')
        if farm.instance and instance and instance != farm.instance:
            self._lose_all(farm, 'farm restarted (new instance)', cancel=False)
        farm.capacity, farm.instance, farm.failing_since = capacity, instance or farm.instance, None
        farm.others = {name: max(0, int(spec.get('running') or 0) - farm.ours[name])
                       for name, spec in (capacity.get('families') or {}).items()}
        self._pump_all()

    def _farm_failed(self, farm, reason):
        now = self.now()
        farm.failing_since = farm.failing_since or now
        if now - farm.failing_since >= FARM_LOST_S and sum(farm.ours.values()):
            self._lose_all(farm, f'farm unreachable for {now - farm.failing_since:.0f}s ({reason})')

    async def _probe_loop(self):
        while True:
            await asyncio.gather(*(self._probe(f) for f in list(self.farms.values()) if f.listed or sum(f.ours.values())),
                                 return_exceptions=True)
            await asyncio.sleep(self.probe_seconds)

    async def _expire_loop(self):
        """Own task on purpose: _probe_loop awaits every farm probe, and one hung probe would otherwise
        hold off every queue deadline -- exactly when farms are sick is when expiry matters most."""
        while True:
            try:
                self._expire_unservable()
            except Exception as exc:
                self.counters['expire_errors'] = self.counters.get('expire_errors', 0) + 1
                try:
                    self.emit(event='gateway_expire_error', detail=repr(exc)[:300])
                except Exception:
                    pass
            await asyncio.sleep(self.probe_seconds)

    # -- accounting and routing ------------------------------------------------------------------------------
    def compatible_farms(self, task, grader):
        now = self.now()
        return [f for f in self.farms.values() if f.usable(now) and f.compatible(grader) and f.total(task) > 0]

    def slots(self, task, grader):
        return sum(f.total(task) for f in self.compatible_farms(task, grader))

    def pick(self, task, grader):
        now = self.now()
        best = None
        for farm in self.compatible_farms(task, grader):
            if now < farm.full_until.get(task, 0) or farm.free(task) <= 0:
                continue
            if best is None or (farm.free(task), best.name) > (best.free(task), farm.name):
                best = farm
        return best

    def _assign(self, grade, farm):
        grade.farm, grade.state = farm, 'dispatching'
        farm.ours[grade.task] += 1

    def _release(self, grade):
        farm = grade.farm
        if farm is not None and grade.state in ON_FARM:
            farm.ours[grade.task] -= 1
            group = self.groups.get((farm.url, grade.run, grade.token))
            if group is not None:
                group.pop(grade.farm_id, None)
        grade.farm = None

    def _enqueue(self, grade):
        queue = self.queues.setdefault(grade.task, {}).setdefault(grade.run, [])
        position = next((i for i, other in enumerate(queue) if other.age > grade.age), len(queue))
        queue.insert(position, grade)

    def _unqueue(self, grade):
        runs = self.queues.get(grade.task) or {}
        queue = runs.get(grade.run)
        if queue and grade in queue:
            queue.remove(grade)
            if not queue:
                del runs[grade.run]

    def _pump_all(self):
        for task in list(self.queues):
            self._pump(task)

    def _pump(self, task):
        """Dispatch queued grades round-robin across runs while some compatible farm has a free slot."""
        runs = self.queues.get(task) or {}
        while runs:
            last = self.turn.get(task)
            order = sorted(runs, key=lambda r: (self.priority(r), last is not None and r <= last, r))
            for run in order:
                grade = runs[run][0]
                if self.reserved(run) and not self.reserve_open(task, grade.grader):
                    continue
                farm = self.pick(task, grade.grader)
                if farm is None:
                    continue
                runs[run].pop(0)
                if not runs[run]:
                    del runs[run]
                self.turn[task] = run
                self._assign(grade, farm)
                self._spawn(self._dispatch(grade, farm))
                break
            else:
                return

    async def _dispatch(self, grade, farm):
        farm_id = grade.farm_id
        try:
            response = await self.http.post(farm.url + '/skyrl/v1/grading/submit', json=dict(grade.body, request_id=farm_id),
                                            headers=grade.headers, timeout=httpx.Timeout(30, pool=None))
        except httpx.HTTPError as exc:
            # No ACK: it may still have been accepted there, so cancel that id once and route it under a new one.
            farm.unhealthy_until = self.now() + 30
            self._spawn(self._cancel_at(farm, grade, farm_id))
            if grade.state == 'dispatching' and grade.farm is farm:
                grade.attempt += 1
                self._back(grade, f'submit to {farm.name} failed: {type(exc).__name__}')
            return
        status, payload = response.status_code, payload_of(response)
        if grade.state != 'dispatching' or grade.farm is not farm:
            # Cancelled or expired while the submit was in flight.
            if status in (200, 202):
                self._spawn(self._cancel_at(farm, grade, farm_id))
            return
        if status in (200, 202):
            grade.state, grade.view = 'dispatched', payload
            self.counters['dispatched'] += 1
            self._watch(grade, farm)
            return
        if status == 429:
            farm.full_until[grade.task] = self.now() + float(payload.get('retry_after') or 5)
            if payload.get('draining') and farm.capacity is not None:
                farm.capacity = dict(farm.capacity, draining=True)
            self._back(grade, f'{farm.name} full')
            return
        if status == 409 and 'grader version' in str(payload.get('detail')):
            if farm.capacity is not None:
                farm.capacity = dict(farm.capacity, grader_sha256=payload.get('grader_sha256'))
            self._back(grade, f'{farm.name} runs another grader')
            return
        if status in (400, 409):
            # The farm will never take it (invalid, or the id belongs to another client there): the trainer
            # retries it elsewhere under a new id.
            self._finish(grade, failed_view(grade, f'della farm {farm.name} refused it ({status}): '
                                                   f'{payload.get("detail")}'))
            return
        farm.unhealthy_until = self.now() + 30
        self._back(grade, f'{farm.name} answered {status}')

    def _back(self, grade, reason, lost=False):
        """Move a grade off its farm and back into the queue (lost: the farm had accepted it)."""
        self._release(grade)
        grade.view, grade.unknown = None, 0
        if lost:
            grade.losses += 1
            grade.attempt += 1
            self.counters['lost'] += 1
            self.emit(event='gateway_grade_lost', request_id=grade.wire_id, run=grade.run, task=grade.task,
                      losses=grade.losses, reason=reason[:200])
            if grade.losses > MAX_LOSSES:
                self._finish(grade, failed_view(grade, f'lost on della farms {grade.losses} times: {reason}'))
                return
        grade.state = 'queued'
        self._enqueue(grade)
        self._pump(grade.task)

    def _lose_all(self, farm, reason, cancel=True):
        for grade in [g for g in self.grades.values() if g.farm is farm and g.state in ON_FARM]:
            if cancel and grade.state == 'dispatched':
                self._spawn(self._cancel_at(farm, grade, grade.farm_id))
            self._back(grade, reason, lost=True)

    def _finish(self, grade, view):
        self._release(grade)
        self._unqueue(grade)
        grade.view, grade.state = view, view.get('state') if view.get('state') in TERMINAL else 'failed'
        grade.finished = self.now()
        grade.done.set()
        self.counters[grade.state] += 1
        self._pump(grade.task)

    async def _cancel_at(self, farm, grade, farm_id):
        try:
            await self.http.post(f'{farm.url}/skyrl/v1/grading/cancel/{farm_id}', headers=grade.headers,
                                 timeout=httpx.Timeout(10, pool=None))
        except Exception:
            pass

    # -- result polling (one batch long-poll per farm and client identity) ------------------------------------
    def _watch(self, grade, farm):
        key = (farm.url, grade.run, grade.token)
        self.groups.setdefault(key, {})[grade.farm_id] = grade
        poller = self.pollers.get(key)
        if poller is None or poller.done():
            self.pollers[key] = asyncio.create_task(self._poll(key, farm))

    async def _poll(self, key, farm):
        url, run, token = key
        headers = {'X-Grading-Run': run, 'X-Grading-Token': token}
        try:
            while self.groups.get(key):
                ids = sorted(self.groups[key])
                started = self.now()
                try:
                    response = await self.http.post(url + '/skyrl/v1/grading/results',
                                                    json=dict(ids=ids, wait=self.long_poll_seconds), headers=headers,
                                                    timeout=httpx.Timeout(self.long_poll_seconds + 30, pool=None))
                    response.raise_for_status()
                    views = response.json().get('views') or {}
                except Exception as exc:
                    self._farm_failed(farm, f'results poll: {type(exc).__name__}')
                    await asyncio.sleep(self.retry_seconds)
                    continue
                for farm_id in ids:
                    grade = self.groups[key].get(farm_id)
                    if grade is None or grade.farm is not farm or grade.state != 'dispatched' or grade.farm_id != farm_id:
                        self.groups[key].pop(farm_id, None)
                        continue
                    self._observe(grade, views.get(farm_id))
                if (self.groups.get(key) and self.now() - started < 1
                        and not any(isinstance(v, dict) and v.get('state') in TERMINAL for v in views.values())):
                    # Answered at once with nothing finished (an id it does not know): do not spin.
                    await asyncio.sleep(self.retry_seconds)
        finally:
            if self.pollers.get(key) is asyncio.current_task():
                del self.pollers[key]
            if not self.groups.get(key):
                self.groups.pop(key, None)

    def _observe(self, grade, view):
        if not isinstance(view, dict):
            grade.unknown += 1
            if grade.unknown >= UNKNOWN_POLLS_LOST:
                self._back(grade, f'{grade.farm.name} no longer knows the request', lost=True)
            return
        grade.unknown = 0
        state = view.get('state')
        if state in ('done', 'failed'):
            self._finish(grade, view)
        elif state == 'cancelled':
            # Only the farm cancels here (an owner cancel finishes the grade first): draining, or its TTL.
            self._back(grade, f'{grade.farm.name} cancelled it: {(view.get("error") or {}).get("class")}', lost=True)
        else:
            grade.view = view

    # -- trainer-facing API ----------------------------------------------------------------------------------
    def _owned(self, identity, wire_id):
        grade = self.grades.get(wire_id)
        if grade is None or identity is None or grade.token != identity[1]:
            return None
        grade.expires = self.now() + TTL_S
        return grade

    def capacity(self, identity):
        run = identity[0]
        grader = self.run_grader.get(run)
        families = {}
        for farm in self.farms.values():
            if not (farm.usable(self.now()) and farm.compatible(grader)):
                continue
            for name, spec in farm.families().items():
                total = int(spec.get('total') or 0)
                if total <= 0:
                    continue
                family = families.setdefault(name, dict(total=0, running=0, queued=0))
                family['total'] += total
                family['running'] += min(total, farm.others.get(name, 0) + farm.ours[name])
        for name, family in families.items():
            family['queued'] = sum(len(q) for q in (self.queues.get(name) or {}).values())
            family['open_limit'] = math.ceil(family['total'] * QUEUE_FACTOR)
        return dict(open=True, ready=bool(families), draining=False, instance=self.instance, grader_sha256=grader,
                    gateway=True, farms=sum(f.usable(self.now()) for f in self.farms.values()), families=families)

    def load(self, task, grader):
        outstanding = [g for g in self.grades.values() if g.task == task and not g.terminal]
        return dict(task=task, total=self.slots(task, grader), queued=sum(g.state == 'queued' for g in outstanding),
                    running=sum(g.state in ON_FARM for g in outstanding))

    def submit(self, identity, body):
        """(http status, payload), like a farm's open submit."""
        if identity is None:
            return 409, dict(detail='open grading needs X-Grading-Run and X-Grading-Token')
        run, token = identity
        if not isinstance(body, dict):
            return 400, dict(detail='grading request must be a JSON object')
        wire_id, task = body.get('request_id'), body.get('task')
        if task == 'ac2':
            task = 'math'
        if not isinstance(wire_id, str) or not REQUEST_ID.fullmatch(wire_id):
            return 400, dict(detail='request_id must be a 32-character hex id')
        if not isinstance(task, str) or not isinstance(body.get('spec'), dict):
            return 400, dict(detail='task and spec are required')
        existing = self.grades.get(wire_id)
        if existing is not None:
            if existing.token != token:
                return 409, dict(detail='request belongs to another lease or client')
            existing.expires = self.now() + TTL_S
            return 200, existing.public_view(self.now())
        grader = body.get('grader_sha256') or (body.get('contract') or {}).get('grader_sha256')
        if grader:
            self.run_grader[run] = grader
        total = self.slots(task, grader)
        if total <= 0:
            return 429, dict(detail=f'no della grader farm serves {task} with this grader', retry_after=NO_FARM_RETRY_S,
                             **self.load(task, grader))
        limit = math.ceil(total * QUEUE_FACTOR)
        mine = self.priority(run)
        if any(self.priority(other) < mine for other in (self.queues.get(task) or {})):
            self.counters['refused_priority'] += 1
            return 429, dict(detail='higher-priority runs are queued', retry_after=5, **self.load(task, grader))
        # Admission counts only runs at this run's priority or better: a lower class's backlog never blocks it.
        outstanding = Counter(g.run for g in self.grades.values()
                              if g.task == task and not g.terminal and self.priority(g.run) <= mine)
        share = math.ceil(limit / len(set(outstanding) | {run}))
        if sum(outstanding.values()) >= limit and outstanding[run] >= share:
            self.counters['refused'] += 1
            return 429, dict(detail='della graders full', retry_after=5, **self.load(task, grader))
        submitted = body.get('submitted_at')
        age = float(submitted) if isinstance(submitted, (int, float)) and 0 < submitted <= time.time() + 60 else time.time()
        grade = Grade(wire_id=wire_id, run=run, token=token, task=task, body=dict(body, task=task), grader=grader,
                      age=age, created=self.now(), expires=self.now() + TTL_S)
        self.grades[wire_id] = grade
        self.counters['submitted'] += 1
        self._enqueue(grade)
        self._pump(task)
        return 202, grade.public_view(self.now())

    async def results(self, identity, ids, wait=0):
        owned = {}
        for wire_id in ids:
            grade = self._owned(identity, wire_id)
            if grade is not None:
                owned[wire_id] = grade
        pending = [g for g in owned.values() if not g.terminal]
        if wait > 0 and pending and len(pending) == len(owned):
            waiters = [asyncio.ensure_future(g.done.wait()) for g in pending]
            try:
                await asyncio.wait(waiters, timeout=min(wait, self.long_poll_seconds), return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiter in waiters:
                    waiter.cancel()
        now = self.now()
        views = {}
        for wire_id in ids:
            grade = owned.get(wire_id)
            if grade is not None and grade.terminal and grade.fetched is None:
                grade.fetched = now
            views[wire_id] = grade.public_view(now) if grade is not None else None
        return views

    def cancel(self, identity, wire_id, detail='cancelled by owner'):
        grade = self._owned(identity, wire_id)
        if grade is None:
            return None
        if not grade.terminal:
            if grade.farm is not None:
                self._spawn(self._cancel_at(grade.farm, grade, grade.farm_id))
            self._finish(grade, cancelled_view(grade, detail))
        return grade.public_view(self.now())

    async def _reap_loop(self):
        while True:
            await asyncio.sleep(10)
            self.reap()

    def reap(self):
        now = self.now()
        for wire_id, grade in list(self.grades.items()):
            if not grade.terminal and grade.expires < now:
                # Its trainer stopped polling (gone, or moved the grade elsewhere without a cancel).
                if grade.farm is not None and grade.state == 'dispatched':
                    self._spawn(self._cancel_at(grade.farm, grade, grade.farm_id))
                self._finish(grade, cancelled_view(grade, 'request TTL expired: its client stopped polling'))
                self.counters['expired'] += 1
            if grade.terminal and ((grade.fetched and now - grade.fetched > FETCHED_RETENTION_S)
                                   or now - grade.finished > RETENTION_S):
                del self.grades[wire_id]

    def status(self):
        now = self.now()
        return dict(instance=self.instance, gateway=True, counters=dict(self.counters),
                    priorities=dict(self.priorities), reserve=self.reserve if self.best is not None else None,
                    grades=dict(Counter(g.state for g in self.grades.values())),
                    runs=dict(Counter(g.run for g in self.grades.values() if not g.terminal)),
                    farms=[dict(name=f.name, url=f.url, listed=f.listed, usable=f.usable(now), grader=f.grader(),
                                totals={n: f.total(n) for n in f.families()}, ours=dict(f.ours), others=f.others)
                           for f in self.farms.values()])


def identity_of(request):
    run, token = request.headers.get('x-grading-run'), request.headers.get('x-grading-token')
    return (run, token) if run and token else None


def create_app(broker):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(request):
        return JSONResponse(dict(status='ok'))

    async def status(request):
        return JSONResponse(broker.status())

    async def capacity(request):
        identity = identity_of(request)
        if identity is None:
            return JSONResponse(dict(detail='open grading needs X-Grading-Run and X-Grading-Token'), status_code=409)
        return JSONResponse(broker.capacity(identity))

    async def submit(request):
        try:
            body = await request.json()
        except ValueError:
            body = None
        code, payload = broker.submit(identity_of(request), body)
        return JSONResponse(payload, status_code=code)

    async def results(request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        ids = [i for i in (body.get('ids') or []) if isinstance(i, str)][:4096] if isinstance(body, dict) else []
        wait = max(0, int((body or {}).get('wait') or 0)) if isinstance(body, dict) else 0
        return JSONResponse(dict(views=await broker.results(identity_of(request), ids, wait=wait)))

    async def result(request):
        wire_id = request.path_params['request_id']
        wait = max(0, int(request.query_params.get('wait') or 0))
        view = (await broker.results(identity_of(request), [wire_id], wait=wait))[wire_id]
        if view is None:
            return JSONResponse(dict(detail='unknown or expired grading request'), status_code=404)
        return JSONResponse(view)

    async def cancel(request):
        view = broker.cancel(identity_of(request), request.path_params['request_id'])
        if view is None:
            return JSONResponse(dict(detail='unknown or expired grading request'), status_code=404)
        return JSONResponse(view)

    return Starlette(routes=[
        Route('/health', health), Route('/status', status),
        Route('/skyrl/v1/grading/capacity', capacity),
        Route('/skyrl/v1/grading/submit', submit, methods=['POST']),
        Route('/skyrl/v1/grading/results', results, methods=['POST']),
        Route('/skyrl/v1/grading/result/{request_id}', result),
        Route('/skyrl/v1/grading/cancel/{request_id}', cancel, methods=['POST'])])


class GatewayServer:
    """The broker and its HTTP server on their own event loop thread inside the discovery process."""

    def __init__(self, port, host='127.0.0.1', emit=lambda **k: None, priorities=(),
                 reserve=RESERVE_FRACTION, queue_deadlines=()):
        # Bind here, in the caller's thread: a taken port fails discovery's start instead of a silent thread.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(2048)
        self.port = self.sock.getsockname()[1]
        self.instance = uuid.uuid4().hex
        self.emit = emit
        self.priorities, self.reserve = list(priorities), reserve
        self.queue_deadlines = list(queue_deadlines)
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()
        self.broker = self.server = None
        self.thread = threading.Thread(target=self._run, name='grader-gateway', daemon=True)
        self.thread.start()
        if not self.started.wait(30):
            raise RuntimeError('grader gateway did not start')

    def _run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        finally:
            self.loop.close()

    async def _serve(self):
        import uvicorn
        self.broker = Broker(instance=self.instance, emit=self.emit, priorities=self.priorities,
                             reserve=self.reserve, queue_deadlines=self.queue_deadlines)
        self.broker.start()
        config = uvicorn.Config(create_app(self.broker), log_config=None, log_level='warning', access_log=False,
                                lifespan='off', timeout_keep_alive=75, timeout_graceful_shutdown=5)
        self.server = uvicorn.Server(config)
        self.started.set()
        try:
            await self.server.serve(sockets=[self.sock])
        finally:
            await self.broker.close()

    def alive(self):
        return self.thread.is_alive()

    def set_farms(self, farms):
        self.loop.call_soon_threadsafe(self.broker.set_farms, farms)

    def stats(self):
        async def snapshot():
            status = self.broker.status()
            return dict(grades=status['grades'], counters=status['counters'], runs=status['runs'],
                        usable_farms=sum(f['usable'] for f in status['farms']))
        return asyncio.run_coroutine_threadsafe(snapshot(), self.loop).result(10)

    def close(self):
        if self.server is not None and self.thread.is_alive():
            self.loop.call_soon_threadsafe(setattr, self.server, 'should_exit', True)
            self.thread.join(15)


# ------------------------------------------------------------------------------ tunnels to the trainer hosts
# Runs on the trainer host: check the tunnel end to end, then publish the registry atomically.
REMOTE = r'''
import json, os, time, urllib.request
def check(farm):
    try:
        with urllib.request.urlopen(farm['url'] + '/status', timeout=5) as response:
            return json.load(response).get('instance') == farm['instance']
    except Exception:
        return False
farms = PARAMS['farms']
verified = [check(f) for f in farms]
now = time.time()
listed = [dict(f, ok=True, tunnel_ok=True, health='healthy') for f, ok in zip(farms, verified) if ok]
document = dict(time=now, updated=now, contract='ready-only-v1', source='grader-gateway', farms=listed,
                unready=[dict(name=f['name'], url=f['url'], error='tunnel_unverified')
                         for f, ok in zip(farms, verified) if not ok])
path = PARAMS['path']
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path + '.partial', 'w') as handle:
    json.dump(document, handle, indent=1, sort_keys=True)
os.replace(path + '.partial', path)
print(json.dumps(dict(ok=True, verified=[f['name'] for f, ok in zip(farms, verified) if ok])))
'''


def ssh_host_name(config_path, alias):
    """HostName of `Host <alias>` in a SkyPilot ssh config (the head, i.e. rank 0), or None."""
    current = None
    try:
        lines = Path(config_path).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        words = line.split()
        if len(words) >= 2 and words[0].lower() == 'host':
            current = words[1:]
        elif len(words) >= 2 and words[0].lower() == 'hostname' and current and alias in current:
            return words[1]
    return None


def trainer_hosts(rows, args, private_trainers, ssh_argv):
    """{address: ssh argv prefix + destination} for every trainer rank-0 host, one per address.

    The od v4-64 pool job and an SSH-launched partition can share worker 0: one tunnel serves both runs."""
    hosts = {}
    wanted = set(args.trainer_job_id)
    for row in rows:
        if row['status'] != 'RUNNING' or not row.get('cluster'):
            continue
        if row.get('pool') not in args.trainer_pool and row['job_id'] not in wanted:
            continue
        cluster = row['cluster']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]+', cluster):
            continue
        config = Path(args.ssh_config_dir) / cluster
        address = ssh_host_name(config, cluster)
        if address is None:
            continue
        hosts.setdefault(address, dict(argv=['ssh', '-F', str(config)], destination=cluster,
                                       run=row.get('run_id'), job_id=row['job_id']))
    for entry in private_trainers:
        if entry.get('grader_gateway') is False:
            continue
        argv = ssh_argv(entry)
        hosts.setdefault(entry['external_ip'], dict(argv=argv[:-1], destination=argv[-1], run=entry['run_id'],
                                                    job_id=None))
    return hosts


def della_farms(published):
    """Ready della grading farms: direct (no cloud metadata), no engines, grading, with a known incarnation."""
    return [dict(url=r['internal_url'], name='%s@%s' % (r['name'], r['job_id']))
            for r in published
            if r.get('direct') and r.get('engines') == 0 and r.get('grading') and r.get('internal_url')
            and r.get('incarnation')]


class ReverseTunnel:
    """`ssh -N -R 127.0.0.1:<remote port>:127.0.0.1:<gateway port>` to one trainer host."""

    def __init__(self, host, gateway_port, remote_port):
        self.host, self.gateway_port, self.base = host, gateway_port, remote_port
        self.slot = 0
        self.process = None
        self.starts = 0
        self.failed_starts = 0  # consecutive starts that died before the tunnel was ever verified
        self.verified = False
        self.missed = 0

    @property
    def port(self):
        return self.base + 1000 * self.slot

    @property
    def url(self):
        return 'http://127.0.0.1:%d' % self.port

    def argv(self):
        return (self.host['argv'] + SSH_OPTIONS
                + ['-N', '-R', '127.0.0.1:%d:127.0.0.1:%d' % (self.port, self.gateway_port), self.host['destination']])

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def ensure(self, popen=subprocess.Popen):
        """Start (or restart) the forward; returns True if a new ssh was started."""
        if self.alive():
            return False
        if self.process is not None and not self.verified:
            self.failed_starts += 1
            if self.failed_starts >= FAILED_STARTS_BEFORE_SLOT_SWITCH:
                # The port is taken on the trainer host (or a dead session's listener lingers): other slot.
                self.slot, self.failed_starts = 1 - self.slot, 0
        self.verified, self.missed = False, 0
        self.starts += 1
        self.process = popen(self.argv(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        return True

    def observe(self, ok):
        """The trainer host's own check of this forward."""
        if ok:
            self.verified, self.missed, self.failed_starts = True, 0, 0
            return
        self.missed += 1
        if self.alive() and self.missed >= MISSED_CHECKS_BEFORE_RESTART:
            self.close()  # a forward that stopped answering: the next tick starts a fresh one

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def push_registry(host, path, farms, run=subprocess.run):
    params = dict(path=path, farms=farms)
    try:
        proc = run(host['argv'] + ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host['destination'],
                                   'python3 -'],
                   input='PARAMS = %r\n' % params + REMOTE, text=True, capture_output=True, timeout=60)
        if proc.returncode:
            return {'ok': False, 'error': 'ssh_failed', 'exit_code': proc.returncode}
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {'ok': False, 'error': type(exc).__name__}


class GraderGateway:
    """Discovery's side: feed the broker its farms, keep one tunnel per trainer host, publish the registry."""

    def __init__(self, args, private_trainers, ssh_argv, emit=lambda **k: None, popen=subprocess.Popen,
                 run=subprocess.run, server=None):
        self.args, self.private_trainers, self.ssh_argv = args, private_trainers, ssh_argv
        self.emit, self.popen, self.run = emit, popen, run
        self.path = getattr(args, 'gateway_registry_path', None) or DEFAULT_REGISTRY
        self.remote_port = getattr(args, 'gateway_remote_port', None) or DEFAULT_REMOTE_PORT
        self.server = server
        if self.server is None and args.push:
            reserve = getattr(args, 'gateway_reserve', None)
            self.server = GatewayServer(getattr(args, 'gateway_port', None) or DEFAULT_PORT, emit=emit,
                                        priorities=parse_priorities(getattr(args, 'gateway_priority', None)),
                                        queue_deadlines=parse_queue_deadlines(
                                            getattr(args, 'gateway_queue_deadline', None)),
                                        reserve=RESERVE_FRACTION if reserve is None else reserve)
        self.tunnels = {}  # trainer host address -> ReverseTunnel

    def tick(self, rows, published):
        hosts = trainer_hosts(rows, self.args, self.private_trainers, self.ssh_argv)
        farms = della_farms(published)
        for address in [a for a in self.tunnels if a not in hosts]:
            self.tunnels.pop(address).close()
            self.emit(event='gateway_tunnel_closed', host=address)
        if not self.args.push:
            return dict(shadow=True, hosts=sorted(hosts), farms=[f['name'] for f in farms])
        if not self.server.alive():
            raise RuntimeError('grader gateway thread died')
        self.server.set_farms(farms)

        def serve(item):
            address, host = item
            tunnel = self.tunnels.get(address)
            if tunnel is None:
                tunnel = self.tunnels[address] = ReverseTunnel(host, self.server.port, self.remote_port)
            if tunnel.ensure(self.popen):
                self.emit(event='gateway_tunnel_started', host=address, port=tunnel.port, starts=tunnel.starts)
            entry = dict(name=ENTRY_NAME, url=tunnel.url, instance=self.server.instance, engines=0, grading=True,
                         source='della-gateway')
            result = push_registry(host, self.path, [entry], run=self.run)
            verified = (ENTRY_NAME in (result.get('verified') or [])) if result.get('ok') else None
            if verified is not None:
                tunnel.observe(verified)
            return dict(host=address, run=host['run'], destination=host['destination'], port=tunnel.port,
                        verified=verified, error=None if result.get('ok') else result.get('error'))
        with ThreadPoolExecutor(max_workers=8) as workers:
            reports = list(workers.map(serve, sorted(hosts.items())))
        return dict(shadow=False, port=self.server.port, farms=[f['name'] for f in farms], hosts=reports,
                    **self.server.stats())

    def close(self):
        for tunnel in self.tunnels.values():
            tunnel.close()
        if self.server is not None:
            self.server.close()
