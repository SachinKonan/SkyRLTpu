"""Bound backend queues and dispatch whole groups by predicted finish time."""
import asyncio
from collections import deque
import time


class HybridScheduler:
    def __init__(self, local_engines, remote_limit, ready, report=lambda *a, **k: None):
        self.local = [None] * local_engines
        self.remote = [None] * remote_limit
        self.ready = ready
        self.report = report
        self.rates = {'local': None, 'remote': None}
        self.completed = {'local': 0, 'remote': 0}
        self.condition = asyncio.Condition()
        self.waiters = deque()

    def snapshot(self):
        return dict(queued=len(self.waiters), local_active=sum(x is not None for x in self.local),
                    remote_active=sum(x is not None for x in self.remote), rates=dict(self.rates),
                    completed=dict(self.completed))

    def _seconds(self, route, work):
        # Equal weights until evidence exists; never infer speed from chip names.
        rate = self.rates[route] or self.rates['local'] or self.rates['remote'] or 1
        return work / rate

    def _choose(self, work, local_only):
        now = time.monotonic()
        duration = self._seconds('local', work)
        # One dispatched group per local engine. Remaining groups stay here.
        local_candidates = [(duration, i) for i, job in enumerate(self.local) if job is None]
        local_finish = min((max(0, start + self._seconds('local', assigned_work) - now) + duration
                            for job in self.local if job is not None
                            for start, assigned_work in [job]), default=float('inf'))
        candidates = [('local', i, seconds) for seconds, i in local_candidates]
        if not local_only and self.ready():
            remote_seconds = self._seconds('remote', work)
            # Avoid a slow remote tail if a busy local engine can finish sooner.
            if remote_seconds <= min([local_finish] + [d for d, _ in local_candidates]):
                candidates += [('remote', i, remote_seconds)
                               for i, job in enumerate(self.remote) if job is None]
        return min(candidates, key=lambda x: x[2]) if candidates else None

    async def run(self, payload, local_call, remote_call, *, local_only=False):
        prompt = payload.get('prompt', [])
        n = payload.get('n', 1)
        work = max(1, n * (len(prompt) + payload.get('max_tokens', 1)))
        ticket = object()
        force_local = local_only
        self.waiters.append(ticket)
        try:
            while True:
                async with self.condition:
                    while True:
                        choice = self._choose(work, force_local) if self.waiters[0] is ticket else None
                        if choice:
                            route, slot, estimate = choice
                            pool = self.local if route == 'local' else self.remote
                            pool[slot] = (time.monotonic(), work)
                            self.waiters.popleft()
                            self.condition.notify_all()
                            break
                        # External publication and lease state change independently.
                        try:
                            await asyncio.wait_for(self.condition.wait(), timeout=1)
                        except TimeoutError:
                            pass
                started = time.monotonic()
                result = None
                try:
                    result = await (local_call(slot) if route == 'local' else remote_call())
                    if result is not None:
                        self.completed[route] += 1
                        return result
                finally:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    if result is not None:
                        choices = result.get('choices', [])
                        output = sum(len(c.get('token_ids') or []) for c in choices)
                        completed_work = n * len(prompt) + output
                        if completed_work > 0:
                            rate = completed_work / elapsed
                            old = self.rates[route]
                            self.rates[route] = rate if old is None else .75 * old + .25 * rate
                        # Telemetry must not discard a valid group or prevent
                        # its capacity slot from being released.
                        try:
                            self.report('hybrid_generated', route=route, seconds=elapsed,
                                        n=n, output_tokens=output)
                        except Exception:
                            pass
                    async with self.condition:
                        pool[slot] = None
                        self.condition.notify_all()
                # An unsuccessful remote result must never enter the batch.
                force_local = True
                self.waiters.appendleft(ticket)
        finally:
            async with self.condition:
                if ticket in self.waiters:
                    self.waiters.remove(ticket)
                self.condition.notify_all()
