import asyncio
import signal
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from skyrl.tinker import api


def _request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_preempt_routes_are_registered():
    route_paths = {getattr(route, "path", None) for route in api.app.routes}
    assert "/api/v1/preempt" in route_paths
    assert "/preempt" in route_paths


def test_preempt_endpoint_disabled_returns_404(monkeypatch):
    scheduled = []
    monkeypatch.delenv("SKYRL_ENABLE_PREEMPT_ENDPOINT", raising=False)
    monkeypatch.setattr(
        api, "_schedule_preempt_signal", lambda *args: scheduled.append(args)
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            api.preempt(
                _request("127.0.0.1"),
                api.PreemptRequest(confirm="preempt", delay_seconds=0),
            )
        )

    assert exc_info.value.status_code == 404
    assert scheduled == []


def test_preempt_endpoint_rejects_remote_client_by_default(monkeypatch):
    scheduled = []
    monkeypatch.setenv("SKYRL_ENABLE_PREEMPT_ENDPOINT", "true")
    monkeypatch.delenv("SKYRL_PREEMPT_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(
        api, "_schedule_preempt_signal", lambda *args: scheduled.append(args)
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            api.preempt(
                _request("10.0.0.2"),
                api.PreemptRequest(confirm="preempt", delay_seconds=0),
            )
        )

    assert exc_info.value.status_code == 403
    assert scheduled == []


def test_preempt_endpoint_schedules_sigterm_with_hard_exit_fallback_for_local_client(
    monkeypatch,
):
    scheduled = []
    monkeypatch.setenv("SKYRL_ENABLE_PREEMPT_ENDPOINT", "true")
    monkeypatch.delenv("SKYRL_PREEMPT_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(
        api,
        "_schedule_preempt_signal",
        lambda signum, delay_seconds, hard_exit_grace_seconds=5.0: scheduled.append(
            (signum, delay_seconds, hard_exit_grace_seconds)
        ),
    )

    response = asyncio.run(
        api.preempt(
            _request("127.0.0.1"),
            api.PreemptRequest(confirm="preempt", delay_seconds=0.01),
        )
    )

    assert scheduled == [(signal.SIGTERM, 0.01, 5.0)]
    assert response.status == "scheduled"
    assert response.signum == signal.SIGTERM
    assert response.signal_name == "SIGTERM"
    assert response.delay_seconds == 0.01


def test_preempt_request_requires_exact_confirmation():
    with pytest.raises(ValidationError):
        api.PreemptRequest(confirm="yes", delay_seconds=0)
