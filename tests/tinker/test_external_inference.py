from pathlib import Path

import httpx
import pytest
from cloudpathlib import AnyPath

from skyrl.tinker.extra.external_inference import ExternalInferenceClient


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
