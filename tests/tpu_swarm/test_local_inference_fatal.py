"""A strict local endpoint failure must terminate the owning run."""
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from tpu.swarm.ray_train import controller, serving
from tpu.swarm.ray_train.config import Config


@pytest.mark.parametrize('state,error', [
    ({'instance': 'original', 'updating': True}, None),
    ({'instance': 'replacement'}, 'local ingress was replaced'),
    ({'instance': 'original', 'fatal_error': 'local generation failed'}, 'local inference failed'),
    ({'instance': 'original', 'exhausted': ['host']}, 'local inference failed'),
    (None, 'ConnectError'),
])
def test_local_monitor_stops_on_failure_but_allows_adapter_updates(monkeypatch, state, error):
    cfg = Config.load('tpu/swarm/ray_train/profiles/science-circuit-v6e-qwen-borrow-supervised-20260920.json')
    owner = controller.Controller(cfg, ['127.0.0.1'])
    owner.local_inference_instance = 'original'
    events = []
    owner.report = lambda event, **fields: events.append(event)
    original_client = httpx.Client
    def transport(request):
        assert request.url.path == '/status'
        if state is None:
            raise httpx.ConnectError('endpoint died', request=request)
        return httpx.Response(200, json=state)
    monkeypatch.setattr(controller.httpx, 'Client', lambda **kw: original_client(
        transport=httpx.MockTransport(transport), **kw))
    owner.check_local_inference()
    if error:
        assert error in owner.failure
        assert events == ['local_inference_failed']
        with pytest.raises(RuntimeError, match='local inference fatal'):
            owner.checked_get(['pending'], 1)
    else:
        assert owner.failure is None and not events


def test_monitor_is_disarmed_before_readiness_and_during_intentional_transition(monkeypatch):
    cfg = Config.load('tpu/swarm/ray_train/profiles/science-circuit-v6e-qwen-borrow-supervised-20260920.json')
    owner = controller.Controller(cfg, ['127.0.0.1'])
    monkeypatch.setattr(controller.httpx, 'Client', lambda **kw: pytest.fail('unexpected probe'))
    owner.check_local_inference()
    owner.local_inference_instance = 'original'
    owner.transitioning = True
    owner.check_local_inference()


def test_fatal_catalog_error_is_sticky_only_for_strict_runs():
    cls = serving.Catalog.__ray_metadata__.modified_class
    strict = cls(['host'], 0)
    strict.fail('first local failure')
    strict.fail('later error')
    assert strict.snapshot()['fatal_error'] == 'first local failure'
    legacy = cls(['host'], 3)
    legacy.fail('first local failure')
    assert legacy.snapshot()['fatal_error'] is None


def test_local_generation_failure_is_reported_without_retry(tmp_path, monkeypatch):
    import asyncio
    import inspect
    class Remote:
        def __init__(self, fn): self.fn = fn
        async def remote(self, *args): return self.fn(*args)
    async def run():
        cfg = Config.load('tpu/swarm/ray_train/profiles/science-circuit-v6e-qwen-borrow-supervised-20260920.json')
        cfg = replace(cfg, root=str(tmp_path), inference=replace(cfg.inference, external_pool_updates=False))
        errors, attempts = [], []
        def fail(payload):
            attempts.append(payload)
            raise RuntimeError('local engine died')
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(),
            [SimpleNamespace(generate=Remote(fail))], SimpleNamespace(fail=Remote(errors.append)), ['127.0.0.1'])
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
                with pytest.raises(RuntimeError, match='local engine died'):
                    await client.post('/v1/completions', json={'model': cfg.model, 'prompt': [1], 'n': 1})
            assert errors == ['local generation failed']
            assert len(attempts) == 1 and gateway.active == 0
        finally:
            await gateway.http.aclose()
    asyncio.run(run())
