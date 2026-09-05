from pathlib import Path

import httpx
import pytest
from cloudpathlib import AnyPath

from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.extra import external_inference
from skyrl.tinker.extra.external_inference import ExternalInferenceClient, _is_retryable_external_error


@pytest.mark.asyncio
async def test_http_pushed_adapter_does_not_read_trainer_local_checkpoint(tmp_path):
    model_name = "model_a_ckpt_1"

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": model_name}]})

    client = ExternalInferenceClient.__new__(ExternalInferenceClient)
    client._available_adapters = set()
    client._adapter_locks = {}
    missing_checkpoint = AnyPath(tmp_path / "trainer-only" / "ckpt_1.tar.gz")

    async with httpx.AsyncClient(
        base_url="http://inference.example/v1",
        transport=httpx.MockTransport(handle),
    ) as http_client:
        await client._ensure_adapter_available(
            http_client,
            model_name,
            missing_checkpoint,
            Path(tmp_path / "loras" / model_name),
        )

    assert ("http://inference.example/v1/", model_name) in client._available_adapters


@pytest.mark.parametrize("status_code", [404, 408, 409, 425, 429, 500, 503])
def test_retryable_external_http_statuses(status_code):
    request = httpx.Request("POST", "http://inference.example/v1/completions")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("engine request failed", request=request, response=response)

    assert _is_retryable_external_error(error)


def test_bad_external_request_is_not_retryable():
    request = httpx.Request("POST", "http://inference.example/v1/completions")
    response = httpx.Response(400, request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=response)

    assert not _is_retryable_external_error(error)


@pytest.mark.asyncio
async def test_transport_error_leaves_request_pending_and_forgets_engine_adapters(monkeypatch):
    failed_url = "http://inference-a.example/v1"
    healthy_url = "http://inference-b.example/v1"
    client = ExternalInferenceClient.__new__(ExternalInferenceClient)
    client.base_urls = [failed_url, healthy_url]
    client._rr_counter = 0
    client.api_key = "test"
    client.request_timeout_sec = 1
    client.db_engine = object()
    client._available_adapters = {
        (f"{failed_url}/", "model_a_ckpt_1"),
        (f"{healthy_url}/", "model_a_ckpt_1"),
    }

    async def fail_request(*args, **kwargs):
        request = httpx.Request("POST", f"{failed_url}/completions")
        raise httpx.ConnectError("engine restarted", request=request)

    terminal_writes = []

    async def record_terminal_write(*args):
        terminal_writes.append(args)

    client._forward_to_engine = fail_request
    monkeypatch.setattr(external_inference, "complete_external_future", record_terminal_write)

    with pytest.raises(httpx.ConnectError, match="engine restarted"):
        await client.call_and_store_result(7, object(), "model_a", "ckpt_1")

    assert terminal_writes == []
    assert (f"{failed_url}/", "model_a_ckpt_1") not in client._available_adapters
    assert (f"{healthy_url}/", "model_a_ckpt_1") in client._available_adapters


@pytest.mark.asyncio
async def test_permanent_engine_error_is_written_as_failed(monkeypatch):
    base_url = "http://inference.example/v1"
    client = ExternalInferenceClient.__new__(ExternalInferenceClient)
    client.base_urls = [base_url]
    client._rr_counter = 0
    client.api_key = "test"
    client.request_timeout_sec = 1
    client.db_engine = object()
    client._available_adapters = set()

    async def reject_request(*args, **kwargs):
        request = httpx.Request("POST", f"{base_url}/completions")
        response = httpx.Response(400, request=request)
        raise httpx.HTTPStatusError("bad request", request=request, response=response)

    terminal_writes = []

    async def record_terminal_write(*args):
        terminal_writes.append(args)

    client._forward_to_engine = reject_request
    monkeypatch.setattr(external_inference, "complete_external_future", record_terminal_write)

    await client.call_and_store_result(8, object(), "model_a", "ckpt_1")

    assert len(terminal_writes) == 1
    assert terminal_writes[0][1] == 8
    assert terminal_writes[0][3] == RequestStatus.FAILED
