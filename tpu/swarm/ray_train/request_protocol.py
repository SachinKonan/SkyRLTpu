"""One request protocol for work a client sends to a farm (grading now, generation next).

Each submission of a request is a ``Ticket`` with one state, changed only by ``transition``, which checks the
move is allowed and logs it (request id, server, attempt, from, to, reason):

    SUBMITTED --ACK--> ACCEPTED --started--> RUNNING --+--> DONE              (result; final)
        |                 |                     |      +--> FAILED_CANDIDATE  (the candidate failed; final)
        |                 |                     |      +--> FAILED_INFRA      (server says infrastructure)
        |                 +---------------------+--------> LOST              (resubmit elsewhere)
        |                 +----------------------------> CANCELLED         (server cancelled it)
        +--no ACK (full / contract mismatch / unreachable)--> REFUSED       (try another server)

Rules (all of them live in ``drive`` below; callers only react to the final state):
1. A server either ACKs (ACCEPTED, pollable by id) or refuses (REFUSED, 429 with its load). Contract
   mismatch is a refusal that also bans that server for this client.
2. Polls renew the request's server-side TTL; a request nobody polls is cancelled by the server itself.
3. ``Config.poll_miss_limit`` consecutive failed polls ask ``confirm_lost``: if the server answers a direct
   probe, or discovery still lists it healthy (it may be reconnecting the tunnel), poll one more round.
   Only a confirmed server failure is LOST(poll_miss).
4. ACCEPTED (queued, never RUNNING) longer than ``queue_limit_s`` moves only if another server has a free
   slot now: send one cancel (not awaited), LOST(queued_too_long). A RUNNING request is never cancelled to
   move it. The resubmission keeps the request's original age, and farms serve the oldest first.
5. The server left the directory: LOST(retired). A resumed id the server no longer knows: LOST(resume_unknown).
The caller's retry policy (grading_transport): REFUSED and LOST(queued_too_long / resume_unknown) are free;
LOST(poll_miss / retired), FAILED_INFRA and CANCELLED are infrastructure retries, capped at
``Config.max_infra_retries``, after which the candidate is dropped (flagged), never scored 0.
"""
import asyncio
from dataclasses import asdict, dataclass, field
import enum
import hashlib
import json
import os
import time


class State(str, enum.Enum):
    SUBMITTED = 'submitted'
    ACCEPTED = 'accepted'
    RUNNING = 'running'
    DONE = 'done'
    FAILED_CANDIDATE = 'failed_candidate'
    FAILED_INFRA = 'failed_infra'
    LOST = 'lost'
    CANCELLED = 'cancelled'
    REFUSED = 'refused'


FINAL = {State.DONE, State.FAILED_CANDIDATE, State.FAILED_INFRA, State.LOST, State.CANCELLED, State.REFUSED}
ALLOWED = {
    State.SUBMITTED: {State.ACCEPTED, State.REFUSED},
    State.ACCEPTED: {State.RUNNING, State.DONE, State.FAILED_CANDIDATE, State.FAILED_INFRA, State.LOST,
                     State.CANCELLED},
    State.RUNNING: {State.DONE, State.FAILED_CANDIDATE, State.FAILED_INFRA, State.LOST, State.CANCELLED},
}


def env_number(name, default, cast=float):
    value = os.environ.get(name)
    return cast(value) if value not in (None, '') else default


@dataclass(frozen=True)
class Config:
    """Every protocol threshold, in one place (env SKYRL_REQ_*)."""
    poll_miss_limit: int = 3
    queue_limit_factor: float = 2.0  # x the request's own time cap
    max_infra_retries: int = 5
    retry_jitter_s: float = 1.0
    registry_stale_s: float = 120.0
    request_ttl_s: float = 300.0     # server side: cancel a request nobody polled for this long

    @classmethod
    def from_env(cls):
        return cls(poll_miss_limit=env_number('SKYRL_REQ_POLL_MISS_LIMIT', cls.poll_miss_limit, int),
                   queue_limit_factor=env_number('SKYRL_REQ_QUEUE_LIMIT_FACTOR', cls.queue_limit_factor),
                   max_infra_retries=env_number('SKYRL_REQ_MAX_INFRA_RETRIES', cls.max_infra_retries, int),
                   retry_jitter_s=env_number('SKYRL_REQ_RETRY_JITTER_S', cls.retry_jitter_s),
                   registry_stale_s=env_number('SKYRL_REQ_REGISTRY_STALE_S', cls.registry_stale_s),
                   request_ttl_s=env_number('SKYRL_REQ_TTL_S', cls.request_ttl_s))


@dataclass(frozen=True)
class Contract:
    """What a server must match to take a request; sent with every request and rejected outright on mismatch.

    Grading fills ``grader_sha256``; generation adds the serving hash (incl. SKYRL_SWA_KV), the LoRA id and
    the renderer date in ``extra``."""
    kind: str
    grader_sha256: str | None = None
    extra: tuple = ()

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()

    def wire(self):
        return dict(asdict(self), extra=dict(self.extra), sha256=self.digest())


class Refused(Exception):
    """No ACK: try another server. ``retry_after`` says when this one may be asked again."""

    def __init__(self, reason, retry_after=5.0, banned=False):
        super().__init__(reason)
        self.retry_after = float(retry_after)
        self.banned = banned  # contract mismatch: never ask this server again


class Lost(Exception):
    """An ACKed request must be resubmitted elsewhere."""

    def __init__(self, reason, kind):
        super().__init__(reason)
        self.kind = kind  # 'poll_miss' | 'retired' | 'queued_too_long' | 'resume_unknown'


@dataclass
class Ticket:
    """One submission of one request to one server."""
    request_id: str
    server: str
    attempt: int = 0
    state: State = State.SUBMITTED
    since: float = field(default_factory=time.monotonic)
    reason: str = ''


def transition(ticket, new, reason='', report=None, now=time.monotonic):
    """The only way a ticket changes state: checks the move and logs it."""
    new = State(new)
    if new != ticket.state:
        if new not in ALLOWED.get(ticket.state, ()):
            raise RuntimeError(f'illegal request transition {ticket.state.value} -> {new.value} ({reason})')
        old, ticket.state, ticket.since, ticket.reason = ticket.state, new, now(), str(reason)
        if report:
            report('request_state', request_id=ticket.request_id, server=ticket.server, attempt=ticket.attempt,
                   **{'from': old.value, 'to': new.value}, reason=str(reason)[:300])
    return ticket


# Fire-and-forget cancels, kept referenced until they finish.
_CANCELS = set()
UNKNOWN = 'request unknown to the server'


async def drive(ticket, submit, poll, cancel, *, config, queue_limit_s=None, alternative=lambda: False,
                retired=lambda: False, confirm_lost=None, resume=False, on_accept=None, report=None,
                poll_interval_s=2.0, between_polls_s=0.0, now=time.monotonic, sleep=asyncio.sleep):
    """Take a ticket from SUBMITTED to a final state; returns the server's final view.

    ``submit()`` returns on ACK or raises Refused; ``poll()`` returns the server's view of the id (None if it
    does not know it) or raises when it cannot be reached; ``cancel()`` asks the server to drop it.
    Raises Refused (REFUSED) or Lost (LOST); returns the view for DONE / FAILED_* / CANCELLED."""
    def to(state, why=''):
        transition(ticket, state, why, report, now)

    if resume:
        to(State.ACCEPTED, 'resumed from the client journal')
    else:
        try:
            await submit()
        except Refused as exc:
            to(State.REFUSED, exc)
            raise
        to(State.ACCEPTED, 'ACK')
    if on_accept:
        on_accept()
    accepted, misses, extended, first = now(), 0, False, True
    while True:
        if retired():
            to(State.LOST, 'server left the directory')
            raise Lost('server left the directory', 'retired')
        try:
            view = await poll()
            error = UNKNOWN if view is None else None
        except Exception as exc:
            view, error = None, f'{type(exc).__name__}: {exc}'[:200]
        if resume and first and error == UNKNOWN:
            to(State.LOST, error)
            raise Lost(error, 'resume_unknown')
        first = False
        if view is None:
            misses += 1
            if misses >= config.poll_miss_limit:
                if confirm_lost is not None and not extended and not await confirm_lost():
                    # A path failure, not a server failure: one more round before giving up on it.
                    extended, misses = True, 0
                    if report:
                        report('request_path_recovering', request_id=ticket.request_id, server=ticket.server,
                               reason=error)
                    continue
                to(State.LOST, f'{misses} consecutive polls failed: {error}')
                raise Lost(f'{misses} consecutive polls failed: {error}', 'poll_miss')
            await sleep(poll_interval_s)
            continue
        misses = 0
        state = view.get('state')
        if state == 'running' and ticket.state == State.ACCEPTED:
            to(State.RUNNING, 'started')
        elif state == 'done':
            failed = bool((view.get('result') or {}).get('error'))
            to(State.FAILED_CANDIDATE if failed else State.DONE, 'result')
            return view
        elif state == 'failed':
            to(State.FAILED_INFRA, (view.get('error') or {}).get('detail') or 'failed')
            return view
        elif state == 'cancelled':
            to(State.CANCELLED, (view.get('error') or {}).get('detail') or 'cancelled by the server')
            return view
        elif (state == 'queued' and queue_limit_s is not None and now() - accepted > queue_limit_s
              and alternative()):
            task = asyncio.ensure_future(cancel())
            _CANCELS.add(task)
            task.add_done_callback(_CANCELS.discard)
            to(State.LOST, f'queued {now() - accepted:.0f}s > {queue_limit_s:.0f}s; a free slot elsewhere')
            raise Lost('queued too long', 'queued_too_long')
        if between_polls_s:
            await sleep(between_polls_s)
