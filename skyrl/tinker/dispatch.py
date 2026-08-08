"""Dispatch + recovery for EXTERNAL (externally-served) sampling requests.

Background
----------
``RequestType.EXTERNAL`` futures are the one request type the engine loop does
NOT process: ``TinkerEngine.find_single_requests`` explicitly filters them out
(``engine.py``, ``.where(FutureDB.request_type != types.RequestType.EXTERNAL)``).
They are instead served by a fire-and-forget asyncio task created in the API
process (``api.py::asample``), which calls the external vLLM and writes the
result back into the ``futures`` row.

That made a stalled request unrecoverable: nothing ever re-scans PENDING
EXTERNAL rows, so a row is stuck until (and unless) its own task writes to it.
Measured on the live sweep slices (see the write-up in
``tpu/results/tinker-external-watchdog/``), in order of how much damage they did:

1. **A wedged/overloaded vLLM pins the task for the full read timeout.**
   ``external_inference_timeout_sec`` defaults to 7200s, eight times the
   client's 900s progress timeout, so the client gives up, restarts, and the row
   stays PENDING for up to two more hours with a task still uselessly waiting on
   it. On dbtest-e this produced 733 ``httpx.ReadTimeout`` in 12 hours and 248
   simultaneously-PENDING rows, the oldest exactly 2h00m old.
2. **Unprotected completion write.** Both inference clients ran the final
   ``AsyncSession`` write *outside* their ``try/except``, using a
   read-modify-write on the ORM object. If that write raised, the task died with
   the result already in hand. Confirmed on the live DB: 733 "External engine
   error" log lines but only 709 FAILED rows — 24 results evaporated.
3. **Untracked tasks.** ``asyncio.create_task`` results were discarded, so the
   event loop held only a weak reference (a task may be garbage collected
   mid-flight) and no exception was ever retrieved or logged.
4. **Event-loop shutdown.** In-flight tasks are cancelled on API restart;
   ``CancelledError`` is a ``BaseException`` and was not caught, so rows stayed
   PENDING and the next API process had no idea they existed.

This module closes all four:

* :class:`ExternalDispatcher` owns a strong reference to every in-flight task.
* :func:`complete_external_future` makes the terminal write **conditional on the
  row still being PENDING**, so a request can never be completed twice or
  produce duplicate results, no matter how many times it is re-dispatched.
* The watchdog re-dispatches orphaned rows (and cancels + re-dispatches tasks
  that blew through a hard ceiling), failing them explicitly after a bounded
  number of attempts so the client gets an error instead of hanging.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlmodel import select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger

# --- Environment knobs ------------------------------------------------------
# All are read once at dispatcher construction (i.e. at API startup).
ENV_ENABLED = "SKYRL_EXTERNAL_WATCHDOG_ENABLED"
ENV_POLL_SEC = "SKYRL_EXTERNAL_WATCHDOG_POLL_SEC"
ENV_STALE_SEC = "SKYRL_EXTERNAL_WATCHDOG_STALE_SEC"
ENV_INFLIGHT_SEC = "SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC"
ENV_MAX_REDISPATCH = "SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH"
ENV_ABANDON_SEC = "SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC"

# How often the watchdog sweeps the futures table.
DEFAULT_POLL_SEC = 30.0
# How old a PENDING EXTERNAL row with NO live task must be before we consider it
# dropped. Only orphans are gated by this, so it can safely be well under the
# client's progress timeout (the discover client uses 900s by default).
DEFAULT_STALE_SEC = 300.0
# Hard ceiling for a row that DOES have a live task. Past this the worker is
# assumed wedged (accepted the connection, never answered): cancel and retry on
# the next engine in the round-robin.
#
# Deliberately generous. Measured legitimate end-to-end latency on a loaded
# v5p-32 slice (32 concurrent group requests across 3 colocated vLLM workers)
# ranges from 18 to 55 minutes, so anything under ~1h would cancel healthy work.
# 3600s sits above that and well below the 7200s httpx read timeout, which is
# what actually bounds stranding today. Operators who would rather burn a
# duplicate generation than let a row outlive the client's 900s progress timeout
# should set this to ~900. Set <= 0 to never cancel a live task.
DEFAULT_INFLIGHT_SEC = 3600.0
# Re-dispatch attempts before the row is failed so the client can retry.
DEFAULT_MAX_REDISPATCH = 2
# Past this age a row is failed outright rather than re-dispatched: no client is
# still waiting (the SDK's own ceiling is 7200s, the discover client's is 900s),
# so regenerating it would burn real vLLM time on a result nobody reads. This is
# what stops an API restart against a populated DB from kicking off a storm of
# useless generations. Set <= 0 to always re-dispatch regardless of age.
DEFAULT_ABANDON_SEC = 7200.0

# Retries for the terminal write itself (pool timeouts / 'database is locked').
_COMPLETE_WRITE_ATTEMPTS = 3
_COMPLETE_WRITE_BACKOFF_SEC = 0.5


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def format_exception(exc: BaseException) -> str:
    """Render an exception for a FAILED row's ``error`` field.

    ``str(httpx.ReadTimeout())`` is the empty string, so the live slices wrote
    709 rows of ``{"error": "", "status": "failed"}`` and ``retrieve_future``
    handed the client a 400 with a blank detail — an unattributable failure.
    Always keep the type name.
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _as_utc(value: datetime) -> datetime:
    """SQLite has no timezone type: ``DateTime(timezone=True)`` round-trips as a
    naive datetime. Everything we write is UTC, so re-attach it."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# --- Idempotent terminal write ---------------------------------------------


async def complete_external_future(
    db_engine,
    request_id: int,
    result_data: dict,
    status: RequestStatus,
) -> bool:
    """Write a terminal result for ``request_id``, exactly once.

    The UPDATE is conditional on the row still being PENDING, which is what makes
    re-dispatch safe: if the original (assumed-lost) attempt eventually finishes
    after a retry already landed, its write matches zero rows and is discarded.
    The client therefore sees exactly one result, never a duplicate and never a
    value that flips after it was read.

    Returns True if this call is the one that completed the row.
    """
    stmt = (
        update(FutureDB)
        .where(FutureDB.request_id == request_id)
        .where(FutureDB.status == RequestStatus.PENDING)
        .values(
            result_data=result_data,
            status=status,
            completed_at=datetime.now(timezone.utc),
        )
    )

    last_exc: Exception | None = None
    for attempt in range(_COMPLETE_WRITE_ATTEMPTS):
        try:
            # Core-level single-statement transaction rather than an ORM session:
            # one round trip, no identity map, and no read-modify-write race.
            async with db_engine.begin() as conn:
                result = await conn.execute(stmt)
            won = bool(result.rowcount)
            if not won:
                # Either already terminal (a retry beat us) or the row is gone.
                logger.warning(
                    "External future %s was already terminal when the result arrived — "
                    "discarding this result to avoid double-completion",
                    request_id,
                )
            return won
        except Exception as e:  # pool timeout, 'database is locked', ...
            last_exc = e
            if attempt + 1 < _COMPLETE_WRITE_ATTEMPTS:
                await asyncio.sleep(_COMPLETE_WRITE_BACKOFF_SEC * (2**attempt))

    # Leaving the row PENDING is deliberate: the watchdog will see an orphan and
    # re-dispatch it. Failing it here would strand a client that could recover.
    logger.error(
        "Could not write result for external future %s after %d attempts (%s: %s); "
        "leaving it PENDING for the watchdog to re-dispatch",
        request_id,
        _COMPLETE_WRITE_ATTEMPTS,
        type(last_exc).__name__,
        last_exc,
    )
    return False


# --- Replaying a stored request --------------------------------------------
#
# The inference clients consume the API-layer ``SampleRequest`` shape, but the
# futures row only stores ``types.SampleInput``. These shims re-present the
# stored payload in the shape the clients duck-type against, without importing
# api.py (which imports this module).


@dataclass(frozen=True)
class ReplayPrompt:
    model_input: types.ModelInput

    def to_types(self) -> types.ModelInput:
        return self.model_input


@dataclass(frozen=True)
class ReplaySamplingParams:
    max_tokens: int | None
    seed: int | None
    stop: list[int] | list[str] | None
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class ReplaySampleRequest:
    """Duck-typed stand-in for ``api.SampleRequest`` built from a stored future.

    Must expose exactly the attributes the inference clients read:
    ``prompt.to_types()``, ``sampling_params.{max_tokens,seed,stop,temperature,
    top_k,top_p}``, ``num_samples``, ``prompt_logprobs``, ``sampling_session_id``
    and ``seq_id``.
    """

    num_samples: int
    prompt: ReplayPrompt
    sampling_params: ReplaySamplingParams
    prompt_logprobs: bool
    sampling_session_id: str | None
    seq_id: int | None


def replay_request_from_sample_input(sample_input: types.SampleInput) -> ReplaySampleRequest:
    sp = sample_input.sampling_params
    stop: list[int] | list[str] | None = None
    if sp.stop_tokens:
        stop = list(sp.stop_tokens)
    elif sp.stop_strings:
        stop = list(sp.stop_strings)
    return ReplaySampleRequest(
        num_samples=sample_input.num_samples,
        prompt=ReplayPrompt(model_input=sample_input.prompt),
        sampling_params=ReplaySamplingParams(
            max_tokens=sp.max_tokens,
            seed=sp.seed,
            stop=stop,
            temperature=sp.temperature,
            top_k=sp.top_k,
            top_p=sp.top_p,
        ),
        prompt_logprobs=bool(sample_input.prompt_logprobs),
        sampling_session_id=sample_input.sampling_session_id,
        seq_id=sample_input.seq_id,
    )


class _InferenceClient(Protocol):
    async def call_and_store_result(
        self,
        request_id: int,
        sample_req: Any,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ) -> None: ...


@dataclass
class SweepReport:
    """Result of one watchdog pass — returned for logging and for tests."""

    pending: int = 0
    inflight: int = 0
    redispatched: list[int] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        return bool(self.redispatched or self.cancelled or self.failed)


class ExternalDispatcher:
    """Owns the lifecycle of EXTERNAL sampling tasks and recovers dropped ones."""

    def __init__(
        self,
        client: _InferenceClient,
        db_engine,
        *,
        enabled: bool | None = None,
        poll_interval_sec: float | None = None,
        stale_after_sec: float | None = None,
        inflight_timeout_sec: float | None = None,
        max_redispatch: int | None = None,
        abandon_after_sec: float | None = None,
    ):
        self.client = client
        self.db_engine = db_engine
        self.enabled = _env_bool(ENV_ENABLED, True) if enabled is None else enabled
        self.poll_interval_sec = (
            _env_float(ENV_POLL_SEC, DEFAULT_POLL_SEC) if poll_interval_sec is None else poll_interval_sec
        )
        self.stale_after_sec = (
            _env_float(ENV_STALE_SEC, DEFAULT_STALE_SEC) if stale_after_sec is None else stale_after_sec
        )
        self.inflight_timeout_sec = (
            _env_float(ENV_INFLIGHT_SEC, DEFAULT_INFLIGHT_SEC) if inflight_timeout_sec is None else inflight_timeout_sec
        )
        self.max_redispatch = (
            _env_int(ENV_MAX_REDISPATCH, DEFAULT_MAX_REDISPATCH) if max_redispatch is None else max_redispatch
        )
        self.abandon_after_sec = (
            _env_float(ENV_ABANDON_SEC, DEFAULT_ABANDON_SEC) if abandon_after_sec is None else abandon_after_sec
        )

        # Strong references: without these the event loop only weakly references
        # a task and it can vanish mid-execution.
        self._inflight: dict[int, asyncio.Task] = {}
        # When the CURRENT attempt started (monotonic). The in-flight ceiling has
        # to be measured from here, not from the row's created_at: a re-dispatch
        # inherits an already-old row, and measuring from created_at would cancel
        # the fresh attempt on the very next sweep and burn the retry budget in
        # three sweeps flat.
        self._started: dict[int, float] = {}
        self._attempts: dict[int, int] = {}
        self._watchdog: asyncio.Task | None = None

    # -- dispatch ----------------------------------------------------------

    def dispatch(
        self,
        request_id: int,
        sample_req: Any,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ) -> asyncio.Task:
        """Start (and track) the task that serves one EXTERNAL request."""
        task = asyncio.ensure_future(self._run(request_id, sample_req, model_id, checkpoint_id, base_model=base_model))
        self._inflight[request_id] = task
        self._started[request_id] = time.monotonic()
        task.add_done_callback(lambda t, rid=request_id: self._on_task_done(rid, t))
        return task

    def _on_task_done(self, request_id: int, task: asyncio.Task) -> None:
        # Only drop the entry if it still points at *this* task: the watchdog may
        # have cancelled it and already registered a replacement.
        if self._inflight.get(request_id) is task:
            self._inflight.pop(request_id, None)
            self._started.pop(request_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Previously this exception was never retrieved and never logged.
            logger.error(
                "External sampling task for request %s died with %s: %s — " "leaving the row PENDING for the watchdog",
                request_id,
                type(exc).__name__,
                exc,
                exc_info=exc,
            )

    async def _run(
        self,
        request_id: int,
        sample_req: Any,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None,
    ) -> None:
        started = time.monotonic()
        try:
            await self.client.call_and_store_result(
                request_id, sample_req, model_id, checkpoint_id, base_model=base_model
            )
        except asyncio.CancelledError:
            logger.warning(
                "External sampling task for request %s cancelled after %.1fs",
                request_id,
                time.monotonic() - started,
            )
            raise

    # -- watchdog ----------------------------------------------------------

    def start_watchdog(self) -> asyncio.Task | None:
        if not self.enabled:
            logger.info("External-request watchdog disabled via %s", ENV_ENABLED)
            return None
        if self._watchdog is not None and not self._watchdog.done():
            return self._watchdog
        logger.info(
            "External-request watchdog: poll=%.0fs stale_after=%.0fs inflight_ceiling=%s max_redispatch=%d",
            self.poll_interval_sec,
            self.stale_after_sec,
            f"{self.inflight_timeout_sec:.0f}s" if self.inflight_timeout_sec > 0 else "disabled",
            self.max_redispatch,
        )
        self._watchdog = asyncio.ensure_future(self._watchdog_loop())
        return self._watchdog

    async def _watchdog_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.poll_interval_sec)
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The watchdog is the last line of defence; it must never die.
                logger.exception("External-request watchdog sweep failed; continuing")

    async def _pending_rows(self) -> list[tuple[int, datetime]]:
        async with AsyncSession(self.db_engine) as session:
            stmt = (
                select(FutureDB.request_id, FutureDB.created_at)
                .where(FutureDB.request_type == types.RequestType.EXTERNAL)
                .where(FutureDB.status == RequestStatus.PENDING)
                .order_by(FutureDB.request_id)
            )
            rows = (await session.exec(stmt)).all()
        return [(int(rid), created_at) for rid, created_at in rows]

    async def _load_request(self, request_id: int) -> tuple[str, types.SampleInput] | None:
        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            if future is None or future.status != RequestStatus.PENDING:
                return None
            if not future.request_data:
                return None
            return (future.model_id or "", types.SampleInput.model_validate(future.request_data))

    async def sweep_once(self) -> SweepReport:
        """One watchdog pass. Split out from the loop so tests can drive it."""
        report = SweepReport()
        rows = await self._pending_rows()
        report.pending = len(rows)
        now = datetime.now(timezone.utc)

        monotonic_now = time.monotonic()

        for request_id, created_at in rows:
            age = (now - _as_utc(created_at)).total_seconds()
            task = self._inflight.get(request_id)

            if task is not None and not task.done():
                report.inflight += 1
                # Measured from when THIS attempt started, so a re-dispatch gets a
                # full fresh budget instead of inheriting the row's age.
                inflight_age = monotonic_now - self._started.get(request_id, monotonic_now)
                if self.inflight_timeout_sec > 0 and inflight_age > self.inflight_timeout_sec:
                    logger.error(
                        "WATCHDOG: external request %s has been in flight for %.0fs "
                        "(row age %.0fs, ceiling %.0fs) — the worker is wedged; "
                        "cancelling and re-dispatching",
                        request_id,
                        inflight_age,
                        age,
                        self.inflight_timeout_sec,
                    )
                    task.cancel()
                    self._inflight.pop(request_id, None)
                    self._started.pop(request_id, None)
                    report.cancelled.append(request_id)
                    if await self._redispatch(request_id, age, report):
                        continue
                # Slow but alive: leave it alone. This is the guard that keeps a
                # legitimately long generation from being restarted underneath us.
                continue

            # No live task: this row is orphaned (task died, was cancelled at
            # shutdown, or belonged to a previous API process).
            if age < self.stale_after_sec:
                continue
            await self._redispatch(request_id, age, report)

        if report.acted:
            logger.warning(
                "WATCHDOG summary: pending=%d inflight=%d redispatched=%s cancelled=%s failed=%s",
                report.pending,
                report.inflight,
                report.redispatched,
                report.cancelled,
                report.failed,
            )
        return report

    async def _fail(self, request_id: int, reason: str, report: SweepReport) -> None:
        await complete_external_future(
            self.db_engine,
            request_id,
            {"error": reason, "status": "failed"},
            RequestStatus.FAILED,
        )
        self._attempts.pop(request_id, None)
        report.failed.append(request_id)

    async def _redispatch(self, request_id: int, age: float, report: SweepReport) -> bool:
        if self.abandon_after_sec > 0 and age > self.abandon_after_sec:
            # Nobody is waiting on a row this old, so regenerating it would burn
            # vLLM time for nothing. Fail it instead — this is also what keeps an
            # API restart against a populated DB from starting a re-dispatch storm.
            logger.error(
                "WATCHDOG: external request %s is %.0fs old (abandon threshold %.0fs) — "
                "no client is still waiting; failing it instead of re-dispatching",
                request_id,
                age,
                self.abandon_after_sec,
            )
            await self._fail(
                request_id,
                f"external sampling request {request_id} was abandoned after {age:.0f}s "
                f"(abandon threshold {self.abandon_after_sec:.0f}s); it was never served",
                report,
            )
            return False

        attempts = self._attempts.get(request_id, 0)
        if attempts >= self.max_redispatch:
            logger.error(
                "WATCHDOG: external request %s still unserved after %d re-dispatches "
                "(age %.0fs) — failing it so the client can retry",
                request_id,
                attempts,
                age,
            )
            await self._fail(
                request_id,
                f"external sampling request {request_id} was dropped and could not be "
                f"recovered after {attempts} re-dispatch attempts (age {age:.0f}s)",
                report,
            )
            return False

        loaded = await self._load_request(request_id)
        if loaded is None:
            # Completed or vanished between the scan and now — nothing to do.
            self._attempts.pop(request_id, None)
            return False
        model_id, sample_input = loaded

        self._attempts[request_id] = attempts + 1
        logger.error(
            "WATCHDOG: re-dispatching dropped external request %s (age %.0fs, attempt %d/%d, "
            "model=%s checkpoint=%s)",
            request_id,
            age,
            attempts + 1,
            self.max_redispatch,
            model_id or "<base>",
            sample_input.checkpoint_id or "<base>",
        )
        self.dispatch(
            request_id,
            replay_request_from_sample_input(sample_input),
            model_id,
            sample_input.checkpoint_id,
            base_model=sample_input.base_model,
        )
        report.redispatched.append(request_id)
        return True

    # -- shutdown ----------------------------------------------------------

    async def aclose(self) -> None:
        """Stop the watchdog and drop in-flight tasks.

        In-flight rows are deliberately left PENDING rather than failed: the next
        API process's watchdog re-dispatches them, which is strictly better for a
        client that is still waiting.
        """
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog = None

        if self._inflight:
            logger.warning(
                "Shutting down with %d external sampling request(s) in flight (%s); "
                "they stay PENDING and will be re-dispatched after restart",
                len(self._inflight),
                sorted(self._inflight),
            )
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        self._started.clear()
        self._attempts.clear()
