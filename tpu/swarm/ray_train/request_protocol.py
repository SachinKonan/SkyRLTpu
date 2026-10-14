"""One request protocol for work sent to a farm (grading now, generation next).

A client submits a request with an idempotent id. The server either ACKs it (it is queued or running there
and can be polled by id) or refuses it (full: 429 with its load). After an ACK the client polls:

* ``done`` / ``failed`` / ``cancelled``: the server's verdict, returned to the caller.
* ``poll_miss_limit`` consecutive polls that fail (no answer, transport error, 5xx, or the request is gone)
  mean the server is dead for this request: ``Lost`` -> the caller resubmits elsewhere. This is an
  infrastructure reroute, never a candidate failure, and is not charged as a failed attempt.
* Still ``queued`` (ACKed but not started) after ``queue_limit_s``: if another server could take it, send
  one cancel without waiting for its answer and raise ``Lost`` -> resubmit elsewhere. If nothing else
  could take it, keep it where it is and keep polling.
* The server dropped out of the directory (``retired()``): ``Lost`` -> resubmit elsewhere.

``drive`` knows nothing about HTTP or grading; callers pass ``submit``, ``poll`` and ``cancel`` coroutines.
"""
import asyncio
from dataclasses import dataclass
import time


class Refused(Exception):
    """The server did not ACK the request (full, wrong version, or unreachable): try another server."""

    def __init__(self, reason, retry_after=5.0):
        super().__init__(reason)
        self.retry_after = float(retry_after)


class Lost(Exception):
    """An ACKed request must be resubmitted elsewhere: server dead, gone, or queued too long."""

    def __init__(self, reason, kind):
        super().__init__(reason)
        self.kind = kind  # 'poll_miss' | 'retired' | 'queued_too_long'


TERMINAL = ('done', 'failed', 'cancelled')
# Fire-and-forget cancels, kept referenced until they finish.
_CANCELS = set()


@dataclass
class Limits:
    poll_miss_limit: int = 3
    # None: never move a queued request.
    queue_limit_s: float | None = None
    # Pause after a failed poll, and after a non-terminal one (0 when polls long-poll on the server).
    poll_interval_s: float = 2.0
    between_polls_s: float = 0.0


async def drive(submit, poll, cancel, *, limits, alternative=lambda: True, retired=lambda: False, on_ack=None,
                now=time.monotonic, sleep=asyncio.sleep, report=None):
    """Submit once, then poll to a terminal view. Raises Refused before an ACK and Lost after one."""
    report = report or (lambda *a, **k: None)
    await submit()
    acked = now()
    if on_ack:
        on_ack()
    misses = 0
    while True:
        if retired():
            raise Lost('server left the directory', 'retired')
        try:
            view = await poll()
        except Exception as exc:
            view, error = None, f'{type(exc).__name__}: {exc}'[:200]
        else:
            error = 'request unknown to the server' if view is None else None
        if view is None:
            misses += 1
            report('request_poll_miss', misses=misses, reason=error)
            if misses >= limits.poll_miss_limit:
                raise Lost(f'{misses} consecutive polls failed: {error}', 'poll_miss')
            await sleep(limits.poll_interval_s)
            continue
        misses = 0
        state = view.get('state')
        if state in TERMINAL:
            return view
        if (state == 'queued' and limits.queue_limit_s is not None and now() - acked > limits.queue_limit_s
                and alternative()):
            # Once, without waiting: the server may already be gone or busy.
            task = asyncio.ensure_future(cancel())
            _CANCELS.add(task)
            task.add_done_callback(_CANCELS.discard)
            raise Lost(f'queued {now() - acked:.0f}s > {limits.queue_limit_s:.0f}s', 'queued_too_long')
        if limits.between_polls_s:
            await sleep(limits.between_polls_s)
