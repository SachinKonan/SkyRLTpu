"""Bound backend queues and dispatch whole groups by predicted finish time.

Two modes share one queue:

* legacy: local engines plus ONE remote pool gated by ``ready()``; a failed
  remote attempt is retried locally (``remote_call()`` takes no argument);
* multi-farm (``farms`` given): local engines (possibly none) plus one pool per
  eligible farm from ``farms() -> [(key, capacity)]``; a failed remote attempt
  re-queues the whole group at the head and reports the farm through
  ``on_remote_failure(key)`` (``remote_call(key)`` names the farm). Nothing is
  ever retried locally when there are no local engines: the ticket waits, and
  ``remote_queue_waiting`` is reported at most every ``alert_seconds``.
"""
import asyncio
from collections import deque
import time


class HybridScheduler:
    def __init__(self, local_engines, remote_limit, ready, report=lambda *a, **k: None, *,
                 farms=None, alert_seconds=60, on_remote_failure=None):
        self.local = [None] * local_engines
        self.remote = [None] * remote_limit
        self.ready = ready
        self.report = report
        self.farms = farms
        self.alert_seconds = alert_seconds
        self.on_remote_failure = on_remote_failure
        self.pools = {}
        self.rates = {'local': None, 'remote': None}
        self.completed = {'local': 0, 'remote': 0}
        self.condition = asyncio.Condition()
        self.waiters = deque()
        self.last_alert = 0
        self.retries = 0

    def snapshot(self):
        result = dict(queued=len(self.waiters), local_active=sum(x is not None for x in self.local),
                      remote_active=sum(x is not None for x in self.remote), rates=dict(self.rates),
                      completed=dict(self.completed))
        if self.farms is not None:
            result['farms'] = {key: sum(x is not None for x in slots) for key, slots in self.pools.items()}
            result['remote_active'] = sum(result['farms'].values())
        return result

    def _seconds(self, route, work):
        # Equal weights until evidence exists; never infer speed from chip names.
        rate = self.rates.get(route) or self.rates['local'] or next(
            (r for k, r in self.rates.items() if k != 'local' and r), None) or 1
        return work / rate

    def _remote_pools(self):
        """Eligible remote pools as (key, slots), synchronised with ``farms()``."""
        if self.farms is None:
            return [('remote', self.remote)] if self.ready() else []
        eligible = {}
        for key, capacity in self.farms():
            slots = self.pools.get(key)
            if slots is None:
                slots = self.pools[key] = []
            busy = [x for x in slots if x is not None]
            if len(slots) != max(capacity, len(busy)):
                # Never drop an assigned slot; grow/shrink the free tail only.
                slots[:] = busy + [None] * max(0, capacity - len(busy))
            eligible[key] = slots
        for key in list(self.pools):
            if key not in eligible and not any(x is not None for x in self.pools[key]):
                del self.pools[key]
        return list(eligible.items())

    def _choose(self, work, local_only):
        now = time.monotonic()
        duration = self._seconds('local', work)
        # One dispatched group per local engine. Remaining groups stay here.
        local_candidates = [(duration, i) for i, job in enumerate(self.local) if job is None]
        local_finish = min((max(0, start + self._seconds('local', assigned_work) - now) + duration
                            for job in self.local if job is not None
                            for start, assigned_work in [job]), default=float('inf'))
        candidates = [('local', 'local', i, seconds) for seconds, i in local_candidates]
        if not local_only:
            bound = min([local_finish] + [d for d, _ in local_candidates])
            for key, slots in self._remote_pools():
                remote_seconds = self._seconds(key, work)
                # Avoid a slow remote tail if a busy local engine can finish sooner.
                if remote_seconds <= bound:
                    candidates += [('remote', key, i, remote_seconds)
                                   for i, job in enumerate(slots) if job is None]
        return min(candidates, key=lambda x: x[3]) if candidates else None

    def _pool(self, route, key):
        if route == 'local':
            return self.local
        return self.remote if self.farms is None else self.pools[key]

    def _alert(self, ticket_started, retries):
        now = time.monotonic()
        if now - self.last_alert < self.alert_seconds:
            return
        self.last_alert = now
        try:
            self.report('remote_queue_waiting', queued=len(self.waiters),
                        farms=[key for key, _ in self._remote_pools()],
                        waited_seconds=round(now - ticket_started, 1), retries=retries)
        except Exception:
            pass

    async def run(self, payload, local_call, remote_call, *, local_only=False):
        prompt = payload.get('prompt', [])
        n = payload.get('n', 1)
        work = max(1, n * (len(prompt) + payload.get('max_tokens', 1)))
        ticket = object()
        force_local = local_only
        started_waiting = time.monotonic()
        retries = 0
        self.waiters.append(ticket)
        try:
            while True:
                async with self.condition:
                    while True:
                        choice = self._choose(work, force_local) if self.waiters[0] is ticket else None
                        if choice:
                            route, key, slot, estimate = choice
                            pool = self._pool(route, key)
                            job = pool[slot] = (time.monotonic(), work)
                            self.waiters.popleft()
                            self.condition.notify_all()
                            break
                        if self.waiters[0] is ticket and self.farms is not None and not self.local:
                            self._alert(started_waiting, retries)
                        # External publication and lease state change independently.
                        try:
                            await asyncio.wait_for(self.condition.wait(), timeout=1)
                        except TimeoutError:
                            pass
                started = time.monotonic()
                result = None
                try:
                    if route == 'local':
                        result = await local_call(slot)
                    else:
                        result = await (remote_call() if self.farms is None else remote_call(key))
                    if result is not None:
                        self.completed[key] = self.completed.get(key, 0) + 1
                        return result
                finally:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    if result is not None:
                        choices = result.get('choices', [])
                        output = sum(len(c.get('token_ids') or []) for c in choices)
                        completed_work = n * len(prompt) + output
                        if completed_work > 0:
                            rate = completed_work / elapsed
                            old = self.rates.get(key)
                            self.rates[key] = rate if old is None else .75 * old + .25 * rate
                        # Telemetry must not discard a valid group or prevent
                        # its capacity slot from being released.
                        try:
                            self.report('hybrid_generated', route=route if self.farms is None else key,
                                        seconds=elapsed, n=n, output_tokens=output)
                        except Exception:
                            pass
                    async with self.condition:
                        # A farm pool may have been compacted while the request
                        # ran; release our job by identity, not by index.
                        for i, assigned in enumerate(pool):
                            if assigned is job:
                                pool[i] = None
                                break
                        self.condition.notify_all()
                # An unsuccessful remote result must never enter the batch.
                if self.local or self.farms is None:
                    force_local = True
                else:
                    retries += 1
                    self.retries += 1
                    if self.on_remote_failure is not None:
                        try:
                            self.on_remote_failure(key)
                        except Exception:
                            pass
                    try:
                        self.report('remote_group_requeued', farm=key, retries=retries)
                    except Exception:
                        pass
                self.waiters.appendleft(ticket)
        finally:
            async with self.condition:
                if ticket in self.waiters:
                    self.waiters.remove(ticket)
                self.condition.notify_all()
