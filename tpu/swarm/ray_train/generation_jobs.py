"""Farm-side generation requests acknowledged at submit and polled for their result.

``POST /skyrl/v1/generation/submit`` admits one group under the caller's lease and
returns an acknowledgement (202) with the request id; the client then polls
``GET /skyrl/v1/generation/result/{id}`` and may ``POST .../cancel/{id}`` once.
The group runs exactly as a synchronous ``/v1/completions`` request would
(``Ingress.finish_admitted``), so admission, engine choice and lease fencing are
unchanged; only the transport is split so a client can tell a slow farm from a
dead one and reroute without waiting for an HTTP timeout.

States: ``queued`` (acknowledged, waiting for a farm admission slot), ``running``
(dispatched to an engine), then ``done`` / ``failed`` / ``cancelled``.

* Ids are the client's. A repeated submit of a known id returns the existing view
  (a lost acknowledgement never runs a group twice) and is reported as a duplicate.
* Ownership: the lease that admitted the group, or the run identity (``run`` +
  ``token``) sent with it. A client restarted under a new lease can therefore
  re-poll (``reclaim``) a group its previous incarnation submitted, and take a
  result that finished while it was down instead of regenerating the group.
* Every request has a server-side TTL renewed by each poll: an unfinished group
  nobody polls for ``ttl`` seconds is cancelled (its client is gone).
* Releasing or losing the lease cancels its unfinished groups; finished results
  stay claimable by run identity for ``retention`` seconds, then are dropped.
  A fetched result is handed over once.
"""
import asyncio
import time

STATES = ('queued', 'running', 'done', 'failed', 'cancelled')
FINAL = ('done', 'failed', 'cancelled')


class GenerationJobs:
    def __init__(self, *, retention=600, ttl=180, clock=time.monotonic):
        self.entries = {}
        self.retention = retention
        self.ttl = ttl
        self.clock = clock
        self.changed = asyncio.Condition()
        self.duplicates = 0

    def view(self, entry):
        now = self.clock()
        result = dict(request_id=entry['request_id'], state=entry['state'],
                      queued_seconds=round((entry['started'] or now) - entry['submitted'], 3),
                      contract=entry.get('contract'))
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
        """Drop old finished results; cancel unfinished groups whose client stopped polling."""
        now = self.clock()
        for request_id, entry in list(self.entries.items()):
            if entry['state'] in FINAL:
                if now - entry['finished'] > self.retention:
                    del self.entries[request_id]
            elif now > entry['expires'] and entry['task'] is not None and not entry.get('cancel_reason'):
                entry['cancel_reason'] = f'request TTL {self.ttl}s lapsed without a poll'
                entry['task'].cancel()

    @staticmethod
    def owns(entry, identity):
        if identity.get('lease_id') and entry['lease_id'] == identity['lease_id']:
            return True
        return bool(identity.get('run') and identity.get('token')
                    and (entry['run'], entry['token']) == (identity['run'], identity['token']))

    def submit(self, request_id, identity, run, *, contract=None):
        """Admit ``run(mark_running)``; return (http status, view). ``run`` is a coroutine factory."""
        self.expire()
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
            return 400, dict(error='request_id must be a nonempty string of at most 128 characters')
        entry = self.entries.get(request_id)
        if entry is not None:
            if not self.owns(entry, identity):
                return 409, dict(error='request id belongs to another lease or run')
            self.duplicates += 1
            entry['expires'] = self.clock() + self.ttl
            return 202, dict(self.view(entry), duplicate=True)
        entry = self.entries[request_id] = dict(
            request_id=request_id, lease_id=identity.get('lease_id'), run=identity.get('run'),
            token=identity.get('token'), contract=contract, state='queued', submitted=self.clock(),
            expires=self.clock() + self.ttl, started=None, finished=None, result=None, error=None,
            status=None, task=None)

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

    async def result(self, request_id, identity, *, wait=0):
        self.expire()
        entry = self.entries.get(request_id)
        if entry is None or not self.owns(entry, identity):
            return None
        entry['expires'] = self.clock() + self.ttl  # a poll renews the request
        if wait and entry['state'] not in FINAL:
            try:
                async with self.changed:
                    await asyncio.wait_for(self.changed.wait_for(lambda: entry['state'] in FINAL),
                                           timeout=min(wait, 60))
            except (TimeoutError, asyncio.TimeoutError):
                pass
            entry['expires'] = self.clock() + self.ttl
        view = self.view(entry)
        if entry['state'] in FINAL:
            # Handed over once: a later poll (e.g. after the client rerouted) finds nothing.
            self.entries.pop(request_id, None)
        return view

    def cancel(self, request_id, identity, reason='cancelled by owner'):
        entry = self.entries.get(request_id)
        if entry is None or not self.owns(entry, identity):
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
        counts['duplicates'] = self.duplicates
        return counts
