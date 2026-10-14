"""Farm-side generation requests acknowledged at submit and polled for their result.

``POST /skyrl/v1/generation/submit`` admits one group under the caller's lease and
returns an acknowledgement (202) with the request id; the client then polls
``GET /skyrl/v1/generation/result/{id}`` and may ``POST .../cancel/{id}`` once.
The group runs exactly as a synchronous ``/v1/completions`` request would
(``Ingress.finish_admitted``), so admission, engine choice and lease fencing are
unchanged; only the transport is split so a client can tell a slow farm from a
dead one and reroute without waiting for an HTTP timeout.

States: ``queued`` (acknowledged, waiting for a farm admission slot), ``running``
(dispatched to an engine), then ``done`` / ``failed`` / ``cancelled``. Ids are
the client's; a repeated submit of a known id returns the existing view, so a
lost acknowledgement never runs a group twice. Finished views are kept for
``retention`` seconds or until fetched; a fetch of a done result removes it, so a
late poll after a reroute finds nothing (the client drops it by id anyway).
"""
import asyncio
import time

STATES = ('queued', 'running', 'done', 'failed', 'cancelled')
FINAL = ('done', 'failed', 'cancelled')


class GenerationJobs:
    def __init__(self, *, retention=600, clock=time.monotonic):
        self.entries = {}
        self.retention = retention
        self.clock = clock
        self.changed = asyncio.Condition()

    def view(self, entry):
        now = self.clock()
        result = dict(request_id=entry['request_id'], state=entry['state'],
                      queued_seconds=round((entry['started'] or now) - entry['submitted'], 3))
        if entry['started'] is not None:
            result['running_seconds'] = round((entry['finished'] or now) - entry['started'], 3)
        if entry['state'] == 'done':
            result['result'] = entry['result']
        elif entry['state'] in ('failed', 'cancelled'):
            result['error'] = entry['error']
            if entry.get('status') is not None:
                result['status'] = entry['status']
        return result

    def expire(self):
        now = self.clock()
        for request_id, entry in list(self.entries.items()):
            if entry['state'] in FINAL and now - entry['finished'] > self.retention:
                del self.entries[request_id]

    def owns(self, entry, lease_id):
        return entry['lease_id'] == lease_id

    def submit(self, request_id, lease_id, run):
        """Admit ``run(mark_running)``; return (http status, view). ``run`` is a coroutine factory."""
        self.expire()
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
            return 400, dict(error='request_id must be a nonempty string of at most 128 characters')
        entry = self.entries.get(request_id)
        if entry is not None:
            if not self.owns(entry, lease_id):
                return 409, dict(error='request id belongs to another lease')
            return 202, self.view(entry)
        entry = self.entries[request_id] = dict(request_id=request_id, lease_id=lease_id, state='queued',
                                                submitted=self.clock(), started=None, finished=None,
                                                result=None, error=None, status=None, task=None)

        def mark_running():
            if entry['state'] == 'queued':
                entry['state'] = 'running'
                entry['started'] = self.clock()

        async def body():
            try:
                result = await run(mark_running)
                self._finish(entry, 'done', result=result)
            except asyncio.CancelledError:
                self._finish(entry, 'cancelled', error=entry.get('cancel_reason') or 'cancelled')
                raise
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                self._finish(entry, 'failed', error=f'{type(exc).__name__}: {getattr(exc, "detail", exc)}'[:512],
                             status=status)

        entry['task'] = asyncio.create_task(body())
        return 202, self.view(entry)

    def _finish(self, entry, state, *, result=None, error=None, status=None):
        if entry['state'] in FINAL:
            return
        entry.update(state=state, result=result, error=error, status=status, finished=self.clock())
        if entry['started'] is None:
            entry['started'] = entry['finished']
        asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self):
        async with self.changed:
            self.changed.notify_all()

    async def result(self, request_id, lease_id, *, wait=0):
        entry = self.entries.get(request_id)
        if entry is None or not self.owns(entry, lease_id):
            return None
        if wait and entry['state'] not in FINAL:
            try:
                async with self.changed:
                    await asyncio.wait_for(self.changed.wait_for(lambda: entry['state'] in FINAL),
                                           timeout=min(wait, 60))
            except (TimeoutError, asyncio.TimeoutError):
                pass
        view = self.view(entry)
        if entry['state'] in FINAL:
            # Handed over once: a later poll (e.g. after the client rerouted) finds nothing.
            self.entries.pop(request_id, None)
        return view

    def cancel(self, request_id, lease_id, reason='cancelled by owner'):
        entry = self.entries.get(request_id)
        if entry is None or not self.owns(entry, lease_id):
            return None
        if entry['state'] not in FINAL and entry['task'] is not None:
            entry['cancel_reason'] = reason
            entry['task'].cancel()
        return dict(request_id=request_id, state='cancelling' if entry['state'] not in FINAL else entry['state'])

    def cancel_lease(self, lease_id, reason):
        """Cancel every unfinished group admitted under ``lease_id`` (lease released or lost)."""
        count = 0
        for entry in list(self.entries.values()):
            if (entry['lease_id'] == lease_id and entry['state'] not in FINAL and entry['task'] is not None
                    and not entry.get('cancel_reason')):
                entry['cancel_reason'] = reason
                entry['task'].cancel()
                count += 1
        return count

    def snapshot(self):
        counts = {state: 0 for state in STATES}
        for entry in self.entries.values():
            counts[entry['state']] += 1
        return counts
