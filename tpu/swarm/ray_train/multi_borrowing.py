"""Run-scoped leases on several farms at once for remote-only trainers.

One ``FarmBorrower`` (a ``RunBorrower`` pinned to a single URL) per candidate
farm keeps the existing acquire, heartbeat, adapter publication, health-grace
and release logic per lease. ``MultiRunBorrower`` presents the ``RunBorrower``
surface to the ingress and reconciles the set of held leases toward
``external_pool_target_leases`` as the supervisor changes the candidate list.

Invariants:
  * one member per URL; at most ``target`` leases held; acquisition is
    sequential, so at most one acquire is in flight at a time;
  * a farm serves only after the current adapter is attested on it
    (``RunBorrower._eligible``); a failed request never loses a lease
    (``request_failure_policy = 'probe'``);
  * a member holding a live lease survives candidate-list updates;
  * ``close()`` releases every lease; tokens are never serialized.
"""
import asyncio
import time

from .borrowing import BorrowingProtocolError, RemoteRequestRejected
from .run_borrowing import RunBorrower


class FarmBorrower(RunBorrower):
    request_failure_policy = 'probe'

    def __init__(self, parent, url):
        super().__init__(parent.config, parent.http, parent.report)
        self.parent = parent
        self.url = url
        self.urls = [url]
        self.owner = parent.owner
        self.offered_contract = None

    def update_urls(self, model, urls):
        raise BorrowingProtocolError('farm members are pinned to one URL')

    def _contract_accepted(self, sha):
        if self.parent.accepts(sha):
            self.offered_contract = sha
            return True
        return False

    def _contract_claim(self, sha):
        return sha

    def _eligible(self, lease):
        # A fresh base reservation must never serve an adapter phase.
        if self.phase_archive is not None and lease.digest is None:
            return False
        return super()._eligible(lease)

    def needs_publication(self):
        lease = self.lease
        return bool(self.phase and self.phase_archive is not None and lease and lease.digest is None
                    and not lease.lost.is_set() and self.preparing is None)

    def retry_preparation(self):
        # Only the parent's reconcile loop acquires; members retry publication
        # on a lease they already hold.
        if not self.parent.may_prepare(self):
            return
        if self.needs_publication():
            self.preparing = asyncio.create_task(self._prepare(self.phase, self.model, self.phase_archive))
            return
        super().retry_preparation()

    def prepare_for_phase(self):
        """Acquire (if needed) and publish the phase adapter; awaited by reconcile."""
        if self.closed or not self.phase or self.preparing is not None:
            return None
        self.preparing = asyncio.create_task(self._prepare(self.phase, self.model, self.phase_archive))
        return self.preparing

    def _lose(self, lease, reason):
        was_lost = lease.lost.is_set()
        super()._lose(lease, reason)
        if not was_lost:
            self.parent.wake.set()

    def _event(self, event, **fields):
        super()._event(event, **fields)
        if event == 'ready':
            self.parent._on_ready(self)


class MultiRunBorrower:
    request_failure_policy = 'probe'

    def __init__(self, config, http, report=lambda *a, **kw: None):
        self.config = config
        self.settings = config.inference
        self.http = http
        self.report = report
        self.owner = config.run_id + ':' + __import__('uuid').uuid4().hex
        self.target = self.settings.external_pool_target_leases
        self.accepted = set(self.settings.external_pool_required_compatibility)
        self.learning = not self.accepted
        self.order = list(self.settings.external_pool_urls.get(config.model, []))
        self.members = {}
        for url in self.order:
            self.members[url] = FarmBorrower(self, url)
        self.phase = None
        self.model = None
        self.phase_archive = None
        self.phase_expected_n = 1
        self.closed = False
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.reconcile_task = None
        self.run_deadline = 0
        self.run_watchdog = None
        self.acquiring = None

    # -- surface shared with RunBorrower ----------------------------------
    @property
    def urls(self):
        return list(self.order)

    @property
    def required_contract(self):
        return next(iter(self.accepted)) if len(self.accepted) == 1 else None

    @required_contract.setter
    def required_contract(self, value):
        # The ingress learns the local serving identity from its engines; a
        # remote-only ingress has none, so only configured/learned hashes count.
        if value and not self.accepted:
            self.accepted.add(value)

    @property
    def lease(self):
        for member in self.members.values():
            if member.lease and member._eligible(member.lease):
                return member.lease
        for member in self.held():
            return member.lease
        return None

    @property
    def preparing(self):
        return next((m.preparing for m in self.members.values() if m.preparing is not None), None)

    def _eligible(self, lease):
        return any(m.lease is lease and m._eligible(lease) for m in self.members.values())

    def accepts(self, sha):
        return sha in self.accepted or (self.learning and not self.accepted)

    def held(self):
        return [m for m in self.members.values() if m.lease and not m.lease.lost.is_set()
                and time.monotonic() < m._deadline(m.lease)]

    def eligible_members(self):
        return [m for m in self.members.values() if m.lease and m._eligible(m.lease)]

    def eligible_pools(self):
        return [(m.url, m.lease.max_requests) for m in self.eligible_members()]

    def farm_leases(self):
        """Per-farm lease view for the grading transport (tokens included)."""
        result = []
        for member in self.members.values():
            lease = member.lease
            if not lease or lease.lost.is_set():
                continue
            result.append(dict(farm_id=member.url, url=member.url, token=lease.token,
                               eligible=member._eligible(lease), capacity=lease.max_requests))
        return result

    def may_prepare(self, member):
        return member in self.held()

    def _event(self, event, **fields):
        try:
            self.report('borrow_' + event, **fields)
        except Exception:
            pass

    def _on_ready(self, member):
        sha = member.offered_contract
        if not sha:
            return
        if self.learning and not self.accepted:
            self.accepted.add(sha)
            self._event('contract_learned', service=member.url, compatibility_sha256=sha)
        elif sha not in self.accepted and member.lease:
            member._lose(member.lease, 'incompatible contract learned')

    def snapshot(self):
        held = self.held()
        eligible = self.eligible_members()
        first = eligible[0].lease if eligible else (held[0].lease if held else None)
        leases = []
        for member in self.members.values():
            lease = member.lease
            if not lease:
                continue
            leases.append(dict(url=member.url, ready=lease.ready, attested=lease.attested,
                               preparing=member.preparing is not None, lost=lease.lost.is_set(),
                               draining=lease.draining, active=lease.active,
                               valid_for=max(0, round(member._deadline(lease) - time.monotonic(), 1))))
        return dict(phase=self.phase, ready=bool(eligible), service=first.url if first else None,
                    adapter_sha256=first.digest if first else None,
                    engines=sum(m.lease.engines for m in eligible), active=sum(m.lease.active for m in held),
                    scope='run', owner_run=self.owner, reserved=bool(held),
                    preparing=any(m.preparing is not None for m in self.members.values()),
                    remote_only=True, target=self.target, held=[m.url for m in held],
                    candidates=list(self.order), accepted_compatibility=sorted(self.accepted),
                    leases=leases)

    def update_urls(self, model, urls):
        if not self.settings.external_pool_updates:
            raise BorrowingProtocolError('service list updates are disabled')
        if model != self.config.model:
            raise BorrowingProtocolError('service list model mismatch')
        from dataclasses import replace
        settings = replace(self.settings, external_pool_urls={model: urls})
        replace(self.config, inference=settings).validate()
        self.order = list(urls)
        for url in urls:
            if url not in self.members:
                self.members[url] = FarmBorrower(self, url)
        for url in list(self.members):
            member = self.members[url]
            if url in self.order or member in self.held() or member.preparing is not None:
                continue
            del self.members[url]
        self.wake.set()

    def touch(self, phase):
        return any([m.touch(phase) for m in self.members.values()]) or phase == self.phase

    def touch_run(self):
        if self.closed:
            return False
        self.run_deadline = time.monotonic() + self.settings.external_pool_lease_seconds
        for member in self.members.values():
            member.touch_run()
        if self.run_watchdog is None:
            self.run_watchdog = asyncio.create_task(self._watch_run())
        if self.reconcile_task is None:
            self.reconcile_task = asyncio.create_task(self._reconcile_loop())
        return True

    async def _watch_run(self):
        try:
            while not self.closed:
                remaining = self.run_deadline - time.monotonic()
                if remaining <= 0:
                    self._event('run_expired')
                    await self.close()
                    return
                await asyncio.sleep(min(remaining, self.settings.external_pool_heartbeat_seconds))
        finally:
            self.run_watchdog = None

    # -- reconciliation ---------------------------------------------------
    async def _reconcile_loop(self):
        try:
            while not self.closed:
                self.wake.clear()
                try:
                    await self._reconcile_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._event('reconcile_error', reason=type(exc).__name__)
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=self.settings.external_pool_heartbeat_seconds)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
        finally:
            self.reconcile_task = None

    async def _reconcile_once(self):
        if self.closed:
            return
        for member in self.held():
            member.touch_run()
            if self.phase is not None and member.phase == self.phase:
                member.retry_preparation()
        for url in list(self.order):
            if self.closed or len(self.held()) >= self.target:
                break
            member = self.members.get(url)
            if (member is None or member in self.held() or member.preparing is not None
                    or member.closed or time.monotonic() < member.uncertain_until):
                continue
            self.acquiring = url
            try:
                if self.phase is not None and member.phase == self.phase:
                    # Lost mid-phase: re-acquire and republish through _prepare,
                    # exactly as a single RunBorrower recovers.
                    pending = member.prepare_for_phase()
                    if pending is not None:
                        await asyncio.gather(pending, return_exceptions=True)
                else:
                    await member.reserve()
            finally:
                self.acquiring = None
            if member.lease and not member.lease.lost.is_set():
                self._event('member_acquired', service=url, held=len(self.held()), target=self.target)
                if self.phase is not None and not self.closed and member.phase != self.phase:
                    try:
                        await member.begin(self.phase, self.model, self.phase_archive,
                                           expected_n=self.phase_expected_n)
                    except BorrowingProtocolError as exc:
                        self._event('member_begin_failed', service=url, reason=str(exc))
        self._event('reconcile', held=[m.url for m in self.held()], target=self.target,
                    candidates=len(self.order), eligible=[m.url for m in self.eligible_members()],
                    preparing=[m.url for m in self.members.values() if m.preparing is not None])

    async def reserve(self):
        async with self.lock:
            if self.closed:
                return self.snapshot()
            self.touch_run()
        await self._reconcile_once()
        return self.snapshot()

    # -- sampling phases ----------------------------------------------------
    async def begin(self, phase, model, archive=None, *, expected_n=1):
        async with self.lock:
            if self.closed:
                raise BorrowingProtocolError('reservation controller heartbeat expired')
            if self.phase:
                if self.phase != phase or self.model != model:
                    raise BorrowingProtocolError('another sampling phase is active')
                return self.snapshot()
            self.phase, self.model = phase, model
            self.phase_archive, self.phase_expected_n = archive, expected_n
            for member in self.held():
                try:
                    await member.begin(phase, model, archive, expected_n=expected_n)
                except BorrowingProtocolError as exc:
                    self._event('member_begin_failed', service=member.url, reason=str(exc))
            self.wake.set()
            return self.snapshot()

    async def end(self, phase=None):
        if phase is not None and phase != self.phase:
            return False
        await asyncio.gather(*(m.end(phase) for m in list(self.members.values())), return_exceptions=True)
        async with self.lock:
            if phase is not None and phase != self.phase:
                return False
            self.phase = self.model = None
            self.phase_archive = None
            return True

    async def close(self):
        self.closed = True
        for task in (self.reconcile_task, self.run_watchdog):
            if task and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(m.close() for m in list(self.members.values())), return_exceptions=True)
        self._event('closed', released=[m.url for m in self.members.values()])

    # -- generation ---------------------------------------------------------
    def _choose(self, farm):
        if farm is not None:
            member = self.members.get(farm)
            return member if member and member.lease and member._eligible(member.lease) else None
        eligible = self.eligible_members()
        return min(eligible, key=lambda m: m.lease.active / max(1, m.lease.max_requests)) if eligible else None

    async def generate(self, payload, local_active=0, local_engines=0, *, prefer_remote=True, farm=None):
        member = self._choose(farm)
        if member is None:
            return None
        try:
            return await member.generate(payload, 0, 0, prefer_remote=True)
        except RemoteRequestRejected:
            raise
