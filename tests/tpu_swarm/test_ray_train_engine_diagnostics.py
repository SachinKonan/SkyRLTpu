"""Preserve engine failures even when the failing VM cannot upload its logs."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip('ray.serve')
from tpu.swarm.ray_train import serving


@pytest.mark.parametrize('failure', ['http', 'transport', 'missing_log'])
def test_generation_failure_preserves_bounded_diagnostics(tmp_path, failure):
    async def run():
        log = tmp_path / 'engine.log'
        if failure != 'missing_log':
            log.write_text('old output\n' * 10000 + 'ROOT_CAUSE: allocator failure\n')

        async def transport(request):
            if failure == 'transport':
                raise httpx.ConnectError('engine disconnected', request=request)
            return httpx.Response(500, text='engine error body: ' + 'x' * 20000)

        async def ensure_adapter(model):
            pass

        cls = serving.Engine.func_or_class
        engine = cls.__new__(cls)
        engine.retiring, engine.run = False, tmp_path
        engine.ip, engine.instance = '10.0.0.3', 'failed-instance'
        engine.process = SimpleNamespace(log=log, poll=lambda: 1, stop=lambda: None)
        engine.url = 'http://engine:19801'
        engine.ensure_adapter = ensure_adapter
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        try:
            with pytest.raises(RuntimeError, match='Inference generation failed') as error:
                await engine.generate({'model': 'base', 'prompt': 'PRIVATE_PROMPT'})
            assert 'PRIVATE_PROMPT' not in str(error.value)
            event = json.loads((tmp_path / 'inference-events.jsonl').read_text())
            assert event['ip'] == '10.0.0.3'
            assert event['instance'] == 'failed-instance'
            assert event['engine_returncode'] == 1
            if failure == 'missing_log':
                assert 'engine_log_error' in event
            else:
                assert event['engine_log_tail'].endswith('ROOT_CAUSE: allocator failure\n')
                assert len(event['engine_log_tail']) <= 8192
            if failure != 'transport':
                assert event['status_code'] == 500
                assert event['response_body'].startswith('engine error body:')
                assert len(event['response_body']) == 4096
            else:
                assert 'engine disconnected' in event['error']
        finally:
            await engine.http.aclose()

    asyncio.run(run())
