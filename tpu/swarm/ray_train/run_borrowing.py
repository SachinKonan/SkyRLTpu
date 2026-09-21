"""Run-scoped ownership with independently committed sampling versions.

The controller heartbeat owns the reservation. Sampling-phase completion only
drains generation; it does not hand the farm to a different experiment.
"""
import asyncio
import time

from .borrowing import Borrower, BorrowingProtocolError, archive_digest


class RunBorrower(Borrower):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_deadline = 0
        self.preparing = None
        self.run_watchdog = None
        self.closed = False

    def touch_run(self):
        if self.closed:
            return False
        self.run_deadline = time.monotonic() + self.settings.external_pool_lease_seconds
        if self.run_watchdog is None:
            self.run_watchdog = asyncio.create_task(self._watch_run())
        return True

    def _deadline(self, lease):
        return min(lease.valid_until, lease.health_deadline, self.run_deadline)

    def _eligible(self, lease):
        return (self.phase is not None and self.preparing is None
                and super()._eligible(lease))

    def snapshot(self):
        result = super().snapshot()
        result.update(scope='run', owner_run=self.owner,
                      reserved=bool(self.lease and not self.lease.lost.is_set()
                                    and time.monotonic() < self._deadline(self.lease)),
                      preparing=self.preparing is not None)
        return result

    async def _watch_run(self):
        try:
            while not self.closed:
                remaining = self.run_deadline - time.monotonic()
                if remaining <= 0:
                    await self.close()
                    return
                await asyncio.sleep(min(remaining, self.settings.external_pool_heartbeat_seconds))
        finally:
            self.run_watchdog = None

    async def reserve(self):
        """Called before trainers start, and again at a later phase after loss."""
        async with self.lock:
            if self.closed:
                return self.snapshot()
            self.touch_run()
            if self.lease and not self.lease.lost.is_set() and time.monotonic() < self._deadline(self.lease):
                return self.snapshot()
            if self.lease:
                await self._release(self.lease)
                self.lease = None
            if time.monotonic() < self.uncertain_until or not self.urls:
                return self.snapshot()
            try:
                async with asyncio.timeout(self.settings.external_pool_prepare_timeout):
                    await self._acquire('reservation', None, None)
            except (Exception, asyncio.CancelledError) as exc:
                self._event('unavailable', reason=type(exc).__name__)
                if self.lease:
                    await self._release(self.lease)
                    self.lease = None
                if isinstance(exc, asyncio.CancelledError):
                    raise
            return self.snapshot()

    async def begin(self, phase, model, archive=None, *, expected_n=1):
        async with self.lock:
            if self.closed:
                raise BorrowingProtocolError('reservation controller heartbeat expired')
            if self.phase:
                if self.phase != phase or self.model != model:
                    raise BorrowingProtocolError('another sampling phase is active')
                return self.snapshot()
            self.phase, self.model = phase, model
            self.touch(phase)
            if expected_n <= self.settings.external_pool_max_n:
                # Local requests need not wait for cross-region upload/compile.
                self.preparing = asyncio.create_task(self._prepare(phase, model, archive))
            return self.snapshot()

    async def _prepare(self, phase, model, archive):
        try:
            await self.reserve()
            lease = self.lease
            if not lease or lease.lost.is_set() or self.phase != phase:
                return
            async with asyncio.timeout(self.settings.external_pool_prepare_timeout):
                if archive:
                    digest = await asyncio.to_thread(archive_digest, archive)
                    if lease.digest == digest and lease.attested:
                        return
                    previous = lease.adapter if lease.digest is not None else None
                    # Suppress heartbeat attestation during a version transition,
                    # while continuing to renew ownership through compilation.
                    lease.ready = lease.attested = False
                    lease.digest = digest
                    lease.adapter = 'borrow-' + lease.lease_id + '-' + digest
                    params = {'lora_name': lease.adapter}
                    if previous:
                        params['previous_lora_name'] = previous

                    async def chunks():
                        with open(archive, 'rb') as stream:
                            while chunk := await asyncio.to_thread(stream.read, 1024**2):
                                yield chunk

                    response = await self.http.post(lease.url + '/skyrl/v1/upload_lora_adapter',
                        params=params, headers={**lease.headers, 'X-Adapter-SHA256': digest},
                        content=chunks(), timeout=self.settings.external_pool_prepare_timeout)
                    response.raise_for_status()
                    receipt = response.json()
                    if (receipt.get('sha256') != digest or receipt.get('lora_name') != lease.adapter
                            or len(set(receipt.get('loaded', []))) != self.settings.external_pool_engines):
                        raise BorrowingProtocolError('incomplete adapter upload acknowledgement')
                elif lease.digest is not None:
                    # Never treat an already-adapted reservation as a fresh base
                    # bootstrap lease. Recovery must reconstruct the correct phase.
                    raise BorrowingProtocolError('base bootstrap after adapter publication')
                response = await self.http.get(lease.url + '/status', headers=lease.headers,
                                               timeout=self.settings.external_pool_rpc_timeout)
                response.raise_for_status()
                self._verify(lease, response.json())
                if self.phase != phase or lease.lost.is_set() or time.monotonic() >= self._deadline(lease):
                    raise BorrowingProtocolError('reservation lost during publication')
                lease.attested = True
                self._resume(lease)
                self._event('ready', service=lease.url, phase=phase, sha256=lease.digest,
                            engines=lease.engines)
        except asyncio.CancelledError:
            if self.lease:
                self._lose(self.lease, 'publication cancelled')
            raise
        except Exception as exc:
            if self.lease:
                self._lose(self.lease, 'publication failed')
            self._event('unavailable', reason=type(exc).__name__)
        finally:
            self.preparing = None

    async def end(self, phase=None):
        """Stop this phase, retaining exclusive ownership across optimizer steps."""
        if phase is not None and phase != self.phase:
            return False
        # Cancel outside the lock: reserve() in the preparation task uses it.
        pending = self.preparing
        if pending and pending is not asyncio.current_task():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        async with self.lock:
            if phase is not None and phase != self.phase:
                return False
            self.phase = self.model = None  # Closes new remote admission first.
            lease = self.lease
            if lease:
                async with self.changed:
                    await self.changed.wait_for(lambda: lease.active == 0)
            return True

    async def close(self):
        self.closed = True
        if self.lease:
            self._lose(self.lease, 'run closed')
        await self.end()
        async with self.lock:
            if self.lease:
                await self._release(self.lease)
                self.lease = None
        if self.run_watchdog and self.run_watchdog is not asyncio.current_task():
            self.run_watchdog.cancel()
            await asyncio.gather(self.run_watchdog, return_exceptions=True)
