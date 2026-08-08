"""Tests for EXTERNAL sampling-request dispatch and the dropped-request watchdog.

These reproduce the production failure that stranded 104 PENDING EXTERNAL rows on
a live v5p-32 slice while every vLLM worker sat idle: the fire-and-forget task
that serves an EXTERNAL future died before writing its result, and because
``TinkerEngine.find_single_requests`` deliberately excludes EXTERNAL requests,
nothing ever re-scanned the row. The client then blocked until its progress
timeout.

Everything here is CPU-only: a temp-file SQLite DB and a fake inference client.
No engine, no backend, no model.
"""

from __future__ import annotations

import asyncio
import gc
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.tinker.dispatch import (
    ExternalDispatcher,
    complete_external_future,
    replay_request_from_sample_input,
)

pytestmark = pytest.mark.asyncio


# --- helpers ---------------------------------------------------------------


def make_sample_input(tokens=(1, 2, 3), num_samples=2, checkpoint_id="ckpt0") -> types.SampleInput:
    return types.SampleInput(
        base_model=None,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=list(tokens))]),
        sampling_params=types.SamplingParams(
            temperature=0.7,
            max_tokens=16,
            seed=1234,
            stop_tokens=[7],
            top_k=-1,
            top_p=0.95,
        ),
        num_samples=num_samples,
        checkpoint_id=checkpoint_id,
        prompt_logprobs=False,
        seq_id=3,
        sampling_session_id="sampling_abcd",
    )


def sample_output(text_marker: int) -> dict:
    return types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[text_marker], logprobs=[-0.1])],
        prompt_logprobs=[],
    ).model_dump()


@pytest_asyncio.fixture
async def db_engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker-test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


async def insert_external_future(db_engine, sample_input=None, *, age_sec: float = 0.0) -> int:
    """Insert a PENDING EXTERNAL row, optionally backdated by ``age_sec``."""
    sample_input = sample_input or make_sample_input()
    async with AsyncSession(db_engine) as session:
        row = FutureDB(
            request_type=types.RequestType.EXTERNAL,
            model_id="model_test",
            request_data=sample_input.model_dump(),
            status=RequestStatus.PENDING,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=age_sec),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return int(row.request_id)


async def read_future(db_engine, request_id: int) -> FutureDB:
    async with AsyncSession(db_engine) as session:
        return await session.get(FutureDB, request_id)


async def poll_for_terminal(db_engine, request_id: int, timeout_sec: float) -> RequestStatus | None:
    """Stand-in for the tinker SDK's ``retrieve_future`` long poll.

    Returns the terminal status, or None if the client would have timed out —
    which in production is the "No progress made" hang.
    """
    deadline = asyncio.get_running_loop().time() + timeout_sec
    while asyncio.get_running_loop().time() < deadline:
        future = await read_future(db_engine, request_id)
        if future.status in (RequestStatus.COMPLETED, RequestStatus.FAILED):
            return future.status
        await asyncio.sleep(0.01)
    return None


class FakeInferenceClient:
    """Stands in for ExternalInferenceClient / SkyRLTrainInferenceForwardingClient.

    ``behavior`` is consulted per call, so a client can be flipped from broken to
    healthy between the original dispatch and the watchdog's re-dispatch.
    """

    def __init__(self, db_engine, behavior: str = "ok", marker: int = 42):
        self.db_engine = db_engine
        self.behavior = behavior
        self.marker = marker
        self.calls: list[int] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def call_and_store_result(self, request_id, sample_req, model_id, checkpoint_id, *, base_model=None):
        self.calls.append(request_id)
        self.started.set()

        if self.behavior == "hang":
            # The wedged-worker case: connection accepted, answer never arrives.
            await asyncio.Event().wait()

        if self.behavior == "gated":
            await self.release.wait()

        if self.behavior == "write_fails":
            # The result was computed; the terminal DB write is what blows up
            # (pool timeout / 'database is locked'). Pre-fix this killed the
            # task with the row still PENDING and nobody watching.
            raise RuntimeError("QueuePool limit of size 20 overflow 60 reached, connection timed out")

        await complete_external_future(self.db_engine, request_id, sample_output(self.marker), RequestStatus.COMPLETED)


# --- reproduction: a dropped request strands the row -----------------------


async def test_dropped_request_blocks_client_forever_without_watchdog(db_engine):
    """Reproduction: the completion write fails, the task dies, the row is stuck.

    This is the exact production signature -- PENDING row, zero in-flight work,
    worker idle -- and it never resolves on its own.
    """
    client = FakeInferenceClient(db_engine, behavior="write_fails")
    dispatcher = ExternalDispatcher(client, db_engine, enabled=False, stale_after_sec=0.0)

    request_id = await insert_external_future(db_engine)
    task = dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    with pytest.raises(RuntimeError):
        await task

    # The client polls and gets nothing: this is "No progress made" forever.
    assert await poll_for_terminal(db_engine, request_id, timeout_sec=0.3) is None
    future = await read_future(db_engine, request_id)
    assert future.status == RequestStatus.PENDING
    assert future.result_data is None
    # And nothing is in flight — the machine is idle while the client waits.
    assert dispatcher._inflight == {}


async def test_watchdog_recovers_a_dropped_request(db_engine):
    """Same failure, but the watchdog re-dispatches and the client gets a result."""
    client = FakeInferenceClient(db_engine, behavior="write_fails")
    dispatcher = ExternalDispatcher(db_engine=db_engine, client=client, stale_after_sec=0.0, max_redispatch=3)

    request_id = await insert_external_future(db_engine)
    task = dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    with pytest.raises(RuntimeError):
        await task
    assert (await read_future(db_engine, request_id)).status == RequestStatus.PENDING

    # The worker recovers (or a different one picks it up).
    client.behavior = "ok"
    report = await dispatcher.sweep_once()

    assert report.redispatched == [request_id]
    await asyncio.gather(*dispatcher._inflight.values())
    assert await poll_for_terminal(db_engine, request_id, timeout_sec=1.0) == RequestStatus.COMPLETED
    assert client.calls == [request_id, request_id]

    future = await read_future(db_engine, request_id)
    assert future.result_data["sequences"][0]["tokens"] == [client.marker]
    assert future.completed_at is not None


async def test_orphan_from_a_previous_api_process_is_redispatched(db_engine):
    """A row left PENDING by an API restart has no task at all; age alone recovers it."""
    client = FakeInferenceClient(db_engine, behavior="ok")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=60.0)

    fresh = await insert_external_future(db_engine, age_sec=5.0)
    orphan = await insert_external_future(db_engine, age_sec=600.0)

    report = await dispatcher.sweep_once()

    # The 5s-old row is younger than stale_after and is left alone.
    assert report.redispatched == [orphan]
    await asyncio.gather(*dispatcher._inflight.values())
    assert (await read_future(db_engine, orphan)).status == RequestStatus.COMPLETED
    assert (await read_future(db_engine, fresh)).status == RequestStatus.PENDING


async def test_hung_worker_is_cancelled_and_redispatched(db_engine):
    """A worker that accepts and never answers is detected by the in-flight ceiling."""
    client = FakeInferenceClient(db_engine, behavior="hang")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0, inflight_timeout_sec=100.0)

    request_id = await insert_external_future(db_engine, age_sec=1.0)
    task = dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    await client.started.wait()

    # Under the ceiling: left alone even though it has made no progress.
    report = await dispatcher.sweep_once()
    assert report.inflight == 1
    assert report.cancelled == [] and report.redispatched == []
    assert not task.done()

    # Push this attempt's start past the ceiling and let a healthy worker take over.
    dispatcher._started[request_id] = dispatcher._started[request_id] - 500.0
    client.behavior = "ok"

    report = await dispatcher.sweep_once()
    assert report.cancelled == [request_id]
    assert report.redispatched == [request_id]
    await asyncio.gather(*dispatcher._inflight.values())
    assert (await read_future(db_engine, request_id)).status == RequestStatus.COMPLETED

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_redispatched_attempt_gets_a_fresh_inflight_clock(db_engine):
    """The ceiling is measured per attempt, not from the row's created_at.

    Measuring from created_at would cancel the replacement on the very next sweep
    (the row only gets older) and burn the whole retry budget in three sweeps.
    """
    client = FakeInferenceClient(db_engine, behavior="hang")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0, inflight_timeout_sec=1.0)

    # Row is an hour old and the first attempt has been hanging past the ceiling.
    request_id = await insert_external_future(db_engine, age_sec=3600.0)
    first = dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    await client.started.wait()
    dispatcher._started[request_id] = dispatcher._started[request_id] - 10.0

    report = await dispatcher.sweep_once()
    assert report.cancelled == [request_id] and report.redispatched == [request_id]
    with pytest.raises(asyncio.CancelledError):
        await first
    second = dispatcher._inflight[request_id]

    # The replacement is brand new: the next sweep must leave it alone even though
    # the row itself is now well over an hour old.
    report = await dispatcher.sweep_once()
    assert report.inflight == 1
    assert report.cancelled == [] and report.redispatched == [] and report.failed == []
    assert not second.done()
    assert dispatcher._attempts[request_id] == 1

    await dispatcher.aclose()


# --- the "do not touch a healthy request" guard ----------------------------


async def test_slow_but_alive_request_is_not_redispatched(db_engine):
    """A live task is never re-dispatched, no matter how old the row is.

    stale_after is 0 and the row is backdated an hour, so the only thing keeping
    the watchdog off it is the in-flight registry.
    """
    client = FakeInferenceClient(db_engine, behavior="gated")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0, inflight_timeout_sec=0.0)

    request_id = await insert_external_future(db_engine, age_sec=3600.0)
    task = dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    await client.started.wait()

    for _ in range(5):
        report = await dispatcher.sweep_once()
        assert report.inflight == 1
        assert report.redispatched == [] and report.cancelled == [] and report.failed == []
        await asyncio.sleep(0.01)

    # Exactly one call was ever made, and it still completes normally.
    assert client.calls == [request_id]
    client.release.set()
    await task
    assert (await read_future(db_engine, request_id)).status == RequestStatus.COMPLETED
    assert client.calls == [request_id]


# --- idempotency -----------------------------------------------------------


async def test_complete_external_future_is_exactly_once(db_engine):
    request_id = await insert_external_future(db_engine)

    assert await complete_external_future(db_engine, request_id, sample_output(1), RequestStatus.COMPLETED) is True
    # A late writer (the "lost" original attempt finishing after a retry landed).
    assert await complete_external_future(db_engine, request_id, sample_output(2), RequestStatus.COMPLETED) is False
    assert (
        await complete_external_future(
            db_engine, request_id, {"error": "boom", "status": "failed"}, RequestStatus.FAILED
        )
        is False
    )

    future = await read_future(db_engine, request_id)
    assert future.status == RequestStatus.COMPLETED
    # The first result stands; nothing is merged, duplicated, or overwritten.
    assert future.result_data["sequences"][0]["tokens"] == [1]


async def test_redispatched_request_cannot_be_completed_twice(db_engine):
    """End-to-end idempotency: a re-dispatch wins, the resurrected original loses."""
    client = FakeInferenceClient(db_engine, behavior="write_fails", marker=7)
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0)

    request_id = await insert_external_future(db_engine)
    with pytest.raises(RuntimeError):
        await dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")

    client.behavior = "ok"
    await dispatcher.sweep_once()
    await asyncio.gather(*dispatcher._inflight.values())
    assert (await read_future(db_engine, request_id)).status == RequestStatus.COMPLETED

    # Now the original attempt "wakes up" and tries to store its stale result.
    assert await complete_external_future(db_engine, request_id, sample_output(99), RequestStatus.COMPLETED) is False

    future = await read_future(db_engine, request_id)
    assert future.result_data["sequences"][0]["tokens"] == [7]
    assert len(future.result_data["sequences"]) == 1


async def test_watchdog_does_not_redispatch_an_already_completed_row(db_engine):
    """A stale read must not resurrect work for a row that finished meanwhile."""
    client = FakeInferenceClient(db_engine, behavior="ok")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0)

    request_id = await insert_external_future(db_engine, age_sec=600.0)
    await complete_external_future(db_engine, request_id, sample_output(5), RequestStatus.COMPLETED)

    report = await dispatcher.sweep_once()
    assert report.pending == 0
    assert report.redispatched == []
    assert client.calls == []


# --- bounded retries -------------------------------------------------------


async def test_request_is_failed_after_max_redispatch(db_engine):
    """After the retry budget the row is failed explicitly so the client can retry."""
    client = FakeInferenceClient(db_engine, behavior="write_fails")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0, max_redispatch=2)

    request_id = await insert_external_future(db_engine)
    with pytest.raises(RuntimeError):
        await dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")

    for expected_attempt in range(2):
        report = await dispatcher.sweep_once()
        assert report.redispatched == [request_id], f"attempt {expected_attempt}"
        with pytest.raises(RuntimeError):
            await asyncio.gather(*dispatcher._inflight.values())

    report = await dispatcher.sweep_once()
    assert report.failed == [request_id]
    assert report.redispatched == []

    future = await read_future(db_engine, request_id)
    assert future.status == RequestStatus.FAILED
    assert "could not be recovered" in future.result_data["error"]
    # Total calls: original + 2 re-dispatches, then it gives up.
    assert client.calls == [request_id] * 3


async def test_abandoned_rows_are_failed_not_regenerated(db_engine):
    """An API restart against a populated DB must not start a re-dispatch storm.

    Rows far past any client deadline are failed outright: regenerating them would
    burn real vLLM time on results nobody will ever read.
    """
    client = FakeInferenceClient(db_engine, behavior="ok")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=300.0, abandon_after_sec=7200.0)

    recoverable = await insert_external_future(db_engine, age_sec=600.0)
    abandoned = [await insert_external_future(db_engine, age_sec=20000.0) for _ in range(3)]

    report = await dispatcher.sweep_once()

    assert report.redispatched == [recoverable]
    assert report.failed == abandoned
    assert client.calls == [recoverable]  # zero wasted generations

    for request_id in abandoned:
        future = await read_future(db_engine, request_id)
        assert future.status == RequestStatus.FAILED
        assert "abandoned" in future.result_data["error"]


# --- task lifetime ---------------------------------------------------------


async def test_dispatcher_holds_a_strong_reference_to_in_flight_tasks(db_engine):
    """asyncio only weakly references tasks; an untracked one can vanish mid-flight."""
    client = FakeInferenceClient(db_engine, behavior="gated")
    dispatcher = ExternalDispatcher(client, db_engine, enabled=False)

    request_id = await insert_external_future(db_engine)
    dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    await client.started.wait()

    # Drop every local reference the test holds and force a collection.
    gc.collect()
    assert request_id in dispatcher._inflight

    client.release.set()
    await asyncio.gather(*dispatcher._inflight.values())
    assert (await read_future(db_engine, request_id)).status == RequestStatus.COMPLETED
    # Completed tasks are unregistered so the map cannot grow without bound.
    assert dispatcher._inflight == {}


async def test_aclose_leaves_inflight_rows_pending_for_the_next_process(db_engine):
    client = FakeInferenceClient(db_engine, behavior="hang")
    dispatcher = ExternalDispatcher(client, db_engine, enabled=False)

    request_id = await insert_external_future(db_engine)
    task = dispatcher.dispatch(request_id, object(), "model_test", "ckpt0")
    await client.started.wait()

    await dispatcher.aclose()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Deliberately PENDING, not FAILED: the next API process re-dispatches it.
    assert (await read_future(db_engine, request_id)).status == RequestStatus.PENDING
    assert dispatcher._inflight == {}


async def test_watchdog_loop_survives_a_failing_sweep(db_engine, monkeypatch):
    client = FakeInferenceClient(db_engine, behavior="ok")
    dispatcher = ExternalDispatcher(client, db_engine, poll_interval_sec=0.01)

    calls = {"n": 0}

    async def flaky_sweep():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient DB error")
        return None

    monkeypatch.setattr(dispatcher, "sweep_once", flaky_sweep)
    dispatcher.start_watchdog()
    await asyncio.sleep(0.1)
    watchdog = dispatcher._watchdog
    assert watchdog is not None and not watchdog.done()
    assert calls["n"] >= 2
    await dispatcher.aclose()


# --- replaying a stored request --------------------------------------------


async def test_replay_request_matches_the_client_contract():
    """The shim must expose exactly what the inference clients read off a request."""
    sample_input = make_sample_input(tokens=(5, 6, 7), num_samples=4)
    replay = replay_request_from_sample_input(sample_input)

    assert replay.prompt.to_types() == sample_input.prompt
    assert replay.prompt.to_types().chunks[0].tokens == [5, 6, 7]
    assert replay.num_samples == 4
    assert replay.sampling_params.max_tokens == 16
    assert replay.sampling_params.temperature == 0.7
    assert replay.sampling_params.top_p == 0.95
    assert replay.sampling_params.top_k == -1
    assert replay.sampling_params.seed == 1234
    assert replay.sampling_params.stop == [7]  # stop_tokens -> int stop list
    assert replay.prompt_logprobs is False
    assert replay.sampling_session_id == "sampling_abcd"
    assert replay.seq_id == 3


async def test_replay_request_prefers_string_stops_when_no_stop_tokens():
    sample_input = make_sample_input()
    sample_input.sampling_params.stop_tokens = None
    sample_input.sampling_params.stop_strings = ["</answer>"]
    replay = replay_request_from_sample_input(sample_input)
    assert replay.sampling_params.stop == ["</answer>"]

    sample_input.sampling_params.stop_strings = None
    assert replay_request_from_sample_input(sample_input).sampling_params.stop is None


async def test_redispatch_replays_the_stored_payload(db_engine):
    """An orphan is re-dispatched with the original prompt/params, not a stub."""
    seen: list[tuple] = []

    class RecordingClient(FakeInferenceClient):
        async def call_and_store_result(self, request_id, sample_req, model_id, checkpoint_id, *, base_model=None):
            seen.append((sample_req, model_id, checkpoint_id, base_model))
            await super().call_and_store_result(request_id, sample_req, model_id, checkpoint_id, base_model=base_model)

    client = RecordingClient(db_engine, behavior="ok")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0)

    sample_input = make_sample_input(tokens=(11, 12), num_samples=3, checkpoint_id="ckpt_replay")
    await insert_external_future(db_engine, sample_input, age_sec=600.0)

    await dispatcher.sweep_once()
    await asyncio.gather(*dispatcher._inflight.values())

    assert len(seen) == 1
    replayed, model_id, checkpoint_id, base_model = seen[0]
    assert model_id == "model_test"
    assert checkpoint_id == "ckpt_replay"
    assert base_model is None
    assert replayed.prompt.to_types().chunks[0].tokens == [11, 12]
    assert replayed.num_samples == 3


# --- env configuration -----------------------------------------------------


async def test_watchdog_reads_env_configuration(db_engine, monkeypatch):
    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_POLL_SEC", "7")
    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_STALE_SEC", "111")
    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC", "2222")
    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH", "5")
    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC", "3333")
    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_ENABLED", "0")

    dispatcher = ExternalDispatcher(FakeInferenceClient(db_engine), db_engine)
    assert dispatcher.poll_interval_sec == 7
    assert dispatcher.stale_after_sec == 111
    assert dispatcher.inflight_timeout_sec == 2222
    assert dispatcher.max_redispatch == 5
    assert dispatcher.abandon_after_sec == 3333
    assert dispatcher.enabled is False
    assert dispatcher.start_watchdog() is None

    monkeypatch.setenv("SKYRL_EXTERNAL_WATCHDOG_STALE_SEC", "not-a-number")
    assert ExternalDispatcher(FakeInferenceClient(db_engine), db_engine).stale_after_sec == 300.0


async def test_create_session_starts_the_staleness_clock(db_engine):
    """The engine's stale-session sweep is the only thing that frees an adapter
    slot, and it cannot see a NULL heartbeat. create_session must not leave one."""
    from skyrl.tinker.api import CreateSessionRequest, create_session
    from skyrl.tinker.db_models import SessionDB

    async with AsyncSession(db_engine) as session:
        response = await create_session(CreateSessionRequest(tags=[], user_metadata=None, sdk_version="test"), session)

    async with AsyncSession(db_engine) as session:
        row = await session.get(SessionDB, response.session_id)

    assert row.last_heartbeat_at is not None


async def test_format_exception_keeps_the_type_for_blank_messages():
    """httpx timeouts stringify to "", which is how 709 live rows got a blank error."""
    import httpx

    from skyrl.tinker.dispatch import format_exception

    assert str(httpx.ReadTimeout("")) == ""
    assert format_exception(httpx.ReadTimeout("")) == "ReadTimeout"
    assert format_exception(RuntimeError("boom")) == "RuntimeError: boom"


async def test_failed_sample_records_an_attributable_error(db_engine):
    """A failing forward must not produce {"error": ""} in the futures row."""
    import httpx

    from skyrl.tinker.extra import ExternalInferenceClient

    client = ExternalInferenceClient.__new__(ExternalInferenceClient)
    client.base_urls = ["http://127.0.0.1:1/v1"]
    client.base_url = client.base_urls[0]
    client._rr_counter = 0
    client.api_key = "EMPTY"
    client.request_timeout_sec = 0.01
    client.db_engine = db_engine

    async def boom(*args, **kwargs):
        raise httpx.ReadTimeout("")

    client._forward_to_engine = boom

    request_id = await insert_external_future(db_engine)
    await client.call_and_store_result(request_id, object(), "model_test", "ckpt0")

    future = await read_future(db_engine, request_id)
    assert future.status == RequestStatus.FAILED
    assert future.result_data["error"] == "ReadTimeout"


async def test_non_external_pending_rows_are_ignored(db_engine):
    """The watchdog must never touch requests the engine loop owns."""
    client = FakeInferenceClient(db_engine, behavior="ok")
    dispatcher = ExternalDispatcher(client, db_engine, stale_after_sec=0.0)

    async with AsyncSession(db_engine) as session:
        row = FutureDB(
            request_type=types.RequestType.SAMPLE,
            model_id="model_test",
            request_data=make_sample_input().model_dump(),
            status=RequestStatus.PENDING,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=99999),
        )
        session.add(row)
        await session.commit()

    report = await dispatcher.sweep_once()
    assert report.pending == 0
    assert report.redispatched == []
    assert client.calls == []
