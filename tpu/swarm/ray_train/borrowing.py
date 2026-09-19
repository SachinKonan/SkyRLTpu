"""Optional, exclusive HTTP inference borrowing. No dependency on a remote Ray cluster.

Farm contract: INFERENCE_BORROWING.md (multi-LoRA farm lease API). An unmodified serving farm is deliberately
ineligible: ordinary /v1/models is not evidence of ownership or adapter identity.
"""
import asyncio
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
import uuid

import httpx


class BorrowingProtocolError(RuntimeError):
    pass


@dataclass
class Lease:
    url: str
    lease_id: str
    token: str = field(repr=False)
    adapter: str = ''
    digest: str | None = None
    valid_until: float = 0
    ready: bool = False
    engines: int = 0
    max_requests: int = 0
    max_n: int = 0
    active: int = 0
    heartbeat: asyncio.Task | None = field(default=None, repr=False)
    lost: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def headers(self):
        return {'X-Lease-ID': self.token}

    @property
    def path(self):
        return self.url


def archive_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Borrower:
    def __init__(self, config, http, report=lambda *a, **kw: None):
        self.config = config
        self.settings = config.inference
        self.urls = self.settings.external_pool_urls.get(config.model, [])
        self.http = http
        self.report = report
        self.owner = config.run_id + ':' + uuid.uuid4().hex
        self.phase = None
        self.model = None
        self.lease = None
        self.lock = asyncio.Lock()
        self.changed = asyncio.Condition()
        self.uncertain_until = 0
        self.next_url = 0
        self.phase_deadline = 0
        self.phase_watchdog = None

    def _event(self, event, **fields):
        # Never serialize a Lease or transport exception: both can expose tokens.
        try:
            self.report('borrow_' + event, **fields)
        except Exception:
            pass  # Optional telemetry must not break the local fallback path.

    def snapshot(self):
        lease = self.lease
        return dict(phase=self.phase, ready=bool(lease and self._eligible(lease)),
                    service=lease.url if lease else None,
                    adapter_sha256=lease.digest if lease else None,
                    engines=lease.engines if lease else 0,
                    active=lease.active if lease else 0)

    def _eligible(self, lease):
        return lease.ready and not lease.lost.is_set() and time.monotonic() < lease.valid_until

    def _expiry(self, response, sent):
        # Use the request's send time and requested TTL, avoiding clock skew
        # against the farm's informational wall-clock expires_at field.
        expires = response.get('expires_at')
        if type(expires) not in (int, float) or not 0 < expires < float('inf'):
            raise BorrowingProtocolError('invalid lease expiration acknowledgement')
        return sent + self.settings.external_pool_lease_seconds - 1

    def _identity(self, lease, body):
        if body.get('lease_id') != lease.lease_id or body.get('owner_run') != self.owner:
            raise BorrowingProtocolError('lease identity mismatch')

    def _verify(self, lease, body):
        self._identity(lease, body)
        if body.get('expected_engines') != self.settings.external_pool_engines:
            raise BorrowingProtocolError('unexpected farm size')
        if lease.digest is not None:
            if (body.get('state') != 'ready' or body.get('adapter_sha256') != lease.digest
                    or body.get('adapter_name') != lease.adapter
                    or body.get('ready_engines') != self.settings.external_pool_engines):
                raise BorrowingProtocolError('adapter not verified on every engine')
        elif body.get('state') != 'awaiting_adapter':
            raise BorrowingProtocolError('base bootstrap lease is not available')
        # The current farm API does not advertise validated sampling limits.
        # These are explicit client-side acceptance settings, not inferred from
        # max_num_sequences or from having a single resident adapter.
        lease.engines = self.settings.external_pool_engines
        lease.max_requests = self.settings.external_pool_max_concurrent_requests
        lease.max_n = self.settings.external_pool_max_n

    def touch(self, phase):
        if phase != self.phase:
            return False
        self.phase_deadline = time.monotonic() + self.settings.external_pool_lease_seconds
        return True

    async def _watch_phase(self, phase):
        while self.phase == phase:
            remaining = self.phase_deadline - time.monotonic()
            if remaining <= 0:
                if self.lease:
                    self.lease.ready = False
                    self.lease.lost.set()
                await self.end(phase)
                return
            await asyncio.sleep(min(remaining, self.settings.external_pool_heartbeat_seconds))

    async def begin(self, phase, model, archive=None, *, expected_n=1):
        async with self.lock:
            if self.phase:
                if self.phase != phase:
                    raise BorrowingProtocolError('another sampling phase is active')
                return self.snapshot()
            self.phase, self.model = phase, model
            self.touch(phase)
            self.phase_watchdog = asyncio.create_task(self._watch_phase(phase))
            if (not self.urls or time.monotonic() < self.uncertain_until
                    or expected_n > self.settings.external_pool_max_n):
                return self.snapshot()
            # Bound all remote preparation, including streamed upload/status, as a
            # single operation. Local publication never waits on external hosts.
            try:
                async with asyncio.timeout(self.settings.external_pool_prepare_timeout):
                    digest = await asyncio.to_thread(archive_digest, archive) if archive else None
                    await self._acquire(phase, digest, archive)
            except (Exception, asyncio.CancelledError) as exc:
                self._event('unavailable', reason=type(exc).__name__)
                if self.lease:
                    await self._release(self.lease)
                    self.lease = None
                if isinstance(exc, asyncio.CancelledError):
                    raise
            return self.snapshot()

    async def _acquire(self, phase, digest, archive):
        count = len(self.urls)
        urls = [self.urls[(self.next_url + i) % count] for i in range(count)]
        self.next_url = (self.next_url + 1) % count
        for url in urls:
            try:
                response = await self.http.get(url + '/health', timeout=self.settings.external_pool_rpc_timeout)
                response.raise_for_status()
                response = await self.http.get(url + '/v1/models', timeout=self.settings.external_pool_rpc_timeout)
                response.raise_for_status()
                if self.config.model not in {m['id'] for m in response.json()['data']}:
                    continue
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
            # Once an acquire is sent, ambiguous failure must not claim a second
            # machine. A timed-out acquire could still be queued server-side,
            # so its eventual lease TTL cannot be inferred from our send time.
            sent = time.monotonic()
            self.uncertain_until = float('inf')
            response = await self.http.post(url + '/acquire_lease', json=dict(
                owner_run=self.owner, ttl_seconds=self.settings.external_pool_lease_seconds),
                timeout=self.settings.external_pool_rpc_timeout)
            if response.status_code in (409, 423, 404):
                # Explicit refusal/no lease API is safe to try on another URL.
                self.uncertain_until = 0
                continue
            response.raise_for_status()
            body = response.json()
            lease_id = body.get('lease_id')
            token = lease_id
            if (not isinstance(lease_id, str) or not lease_id or len(lease_id) > 128
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in lease_id)
                    or not isinstance(token, str) or not token):
                raise BorrowingProtocolError('malformed reservation acknowledgement')
            lease = Lease(url, lease_id, token, digest=digest)
            self.lease = lease
            self._identity(lease, body)
            lease.valid_until = self._expiry(body, sent)
            self.uncertain_until = 0
            lease.adapter = ('borrow-' + lease_id) if archive else self.config.model
            lease.heartbeat = asyncio.create_task(self._heartbeat(lease))
            if archive:
                async def chunks():
                    with Path(archive).open('rb') as stream:
                        while chunk := await asyncio.to_thread(stream.read, 1024**2):
                            yield chunk
                response = await self.http.post(lease.path + '/skyrl/v1/upload_lora_adapter', headers=dict(lease.headers, **{'X-Adapter-SHA256': digest}),
                    params={'lora_name': lease.adapter}, content=chunks(),
                    timeout=self.settings.external_pool_prepare_timeout)
                response.raise_for_status()
                receipt = response.json()
                if (receipt.get('sha256') != digest or receipt.get('lora_name') != lease.adapter
                        or len(set(receipt.get('loaded', []))) != self.settings.external_pool_engines):
                    raise BorrowingProtocolError('incomplete or mismatched upload acknowledgement')
            # Status must attest every engine's actual loaded content, not merely
            # repeat an adapter name supplied in our acquire request.
            while time.monotonic() < lease.valid_until:
                response = await self.http.get(lease.path + '/status', headers=lease.headers,
                                              timeout=self.settings.external_pool_rpc_timeout)
                response.raise_for_status()
                status = response.json()
                self._identity(lease, status)
                if lease.lost.is_set():
                    raise BorrowingProtocolError('lease lost during preparation')
                if status.get('state') not in ('updating', 'transitioning'):
                    self._verify(lease, status)
                    lease.ready = True
                    self._event('ready', service=url, phase=phase, sha256=digest, engines=lease.engines)
                    return
                await asyncio.sleep(.2)
            raise BorrowingProtocolError('lease expired during loading')

    async def _heartbeat(self, lease):
        try:
            while True:
                await asyncio.sleep(self.settings.external_pool_heartbeat_seconds)
                if time.monotonic() >= self.phase_deadline or lease.lost.is_set():
                    lease.ready = False
                    lease.lost.set()
                    return
                sent = time.monotonic()
                response = await self.http.post(lease.path + '/acquire_lease', headers=lease.headers,
                    json={'lease_id': lease.lease_id, 'owner_run': self.owner,
                          'ttl_seconds': self.settings.external_pool_lease_seconds},
                    timeout=self.settings.external_pool_rpc_timeout)
                response.raise_for_status()
                body = response.json()
                self._identity(lease, body)
                if lease.ready:
                    self._verify(lease, body)
                lease.valid_until = self._expiry(body, sent)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            lease.ready = False
            lease.lost.set()
            self._event('heartbeat_failed', service=lease.url, reason=type(exc).__name__)

    async def _wait_lost(self, lease):
        while not lease.lost.is_set():
            remaining = lease.valid_until - time.monotonic()
            if remaining <= 0:
                lease.lost.set()
                break
            try:
                await asyncio.wait_for(lease.lost.wait(), timeout=remaining)
            except TimeoutError:
                pass  # A heartbeat may have extended the deadline meanwhile.

    async def generate(self, payload, local_active, local_engines):
        lease = self.lease
        if (not lease or not self._eligible(lease) or payload.get('model') != self.model
                or type(payload.get('n', 1)) is not int or not 1 <= payload.get('n', 1) <= lease.max_n
                or lease.active >= lease.max_requests
                or lease.active / lease.engines > local_active / max(1, local_engines)):
            return None  # Caller uses its unchanged local path.
        lease.active += 1
        request_id = uuid.uuid4().hex
        request = watcher = None
        try:
            request = asyncio.create_task(self.http.post(lease.url + '/v1/completions',
                json=dict(payload, model=lease.adapter), headers=lease.headers,
                timeout=self.settings.request_timeout))
            watcher = asyncio.create_task(self._wait_lost(lease))
            done, _ = await asyncio.wait((request, watcher), return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                raise BorrowingProtocolError('borrowed service lost or lease expired')
            response = await request
            response.raise_for_status()
            result = response.json()
            # Admission is token-fenced by the farm and weights cannot change
            # during an admitted request. Preserve the native response verbatim.
            if (not isinstance(result, dict) or not isinstance(result.get('choices'), list)
                    or len(result['choices']) != payload.get('n', 1)):
                raise BorrowingProtocolError('completion shape mismatch')
            self._event('generated', service=lease.url, phase=self.phase, request_id=request_id)
            return result
        except asyncio.CancelledError:
            lease.ready = False
            lease.lost.set()
            raise
        except Exception as exc:
            lease.ready = False
            lease.lost.set()
            self._event('generation_failed', service=lease.url, reason=type(exc).__name__)
            return None
        finally:
            for pending in (request, watcher):
                if pending is not None and not pending.done():
                    pending.cancel()
            await asyncio.gather(*(t for t in (request, watcher) if t is not None), return_exceptions=True)
            async with self.changed:
                lease.active -= 1
                self.changed.notify_all()

    async def _release(self, lease):
        lease.ready = False
        acknowledged = False
        try:
            async with asyncio.timeout(self.settings.external_pool_release_timeout):
                async with self.changed:
                    await self.changed.wait_for(lambda: lease.active == 0)
                response = await self.http.post(lease.path + '/release_lease', headers=lease.headers,
                                               json={'lease_id': lease.lease_id}, timeout=self.settings.external_pool_release_timeout)
                response.raise_for_status()
                body = response.json()
                if body.get('state') != 'unleased' or body.get('released') is not True:
                    raise BorrowingProtocolError('release not acknowledged')
                acknowledged = True
        except Exception as exc:
            self._event('release_pending', service=lease.url, reason=type(exc).__name__)
        finally:
            if lease.heartbeat:
                lease.heartbeat.cancel()
                await asyncio.gather(lease.heartbeat, return_exceptions=True)
            if not acknowledged:
                self.uncertain_until = max(self.uncertain_until,
                    time.monotonic() + self.settings.external_pool_lease_seconds + self.settings.external_pool_rpc_timeout)
            self._event('released', service=lease.url, acknowledged=acknowledged)

    async def end(self, phase=None):
        async with self.lock:
            if phase is not None and self.phase != phase:
                return False  # Delayed cleanup must not end a newer phase.
            if self.lease:
                await self._release(self.lease)
            if self.phase_watchdog and self.phase_watchdog is not asyncio.current_task():
                self.phase_watchdog.cancel()
                await asyncio.gather(self.phase_watchdog, return_exceptions=True)
            self.phase_watchdog = None
            self.phase = self.model = self.lease = None
            return True
