"""Exercise the deployed forwarding methods and real persisted output schema.

Only database setup and model execution are replaced. HTTP request construction,
response validation, sequence conversion, and JSON persistence use real code.
"""
import ast
import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


types = load('external_test_types', 'skyrl/tinker/types.py')
validator = load('external_test_validator', 'skyrl/backends/native_completion.py')


def forwarding_method(route):
    name, method = (
        ('external_inference.py', '_forward_to_engine') if route == 'external'
        else ('skyrl_train_inference_forwarding.py', '_forward')
    )
    path = ROOT / 'skyrl/tinker/extra' / name
    tree = ast.parse(path.read_text())
    node = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == method)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    namespace = dict(types=types, httpx=httpx, validate_choice=validator.validate_choice,
                     render_model_input=lambda inputs: [SimpleNamespace(prompt_ids=[100, 101])],
                     logger=SimpleNamespace(warning=lambda *a: None))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[method]


async def roundtrip(route, choices, budget, max_tokens=100, stop=None):
    params = SimpleNamespace(thinking_token_budget=budget, max_tokens=max_tokens,
                             temperature=.8, top_p=1., top_k=-1, seed=None, stop=stop)
    request = SimpleNamespace(prompt=SimpleNamespace(to_types=lambda: None), sampling_params=params,
                              num_samples=len(choices), prompt_logprobs=False,
                              sampling_session_id='session', seq_id=7)
    def engine(req):
        payload = json.loads(req.content)
        assert payload['prompt'] == [100, 101]
        assert payload['n'] == len(choices)
        assert payload['max_tokens'] == max_tokens
        assert payload.get('thinking_token_budget') == budget
        assert ('thinking_token_budget' in payload) == (budget is not None)
        if stop:
            assert payload['stop_token_ids' if isinstance(stop[0], int) else 'stop'] == stop
        assert req.headers['X-Session-ID'] == types.make_routing_session_id('session', 7)
        return httpx.Response(200, json={'choices': copy.deepcopy(choices)})
    async with httpx.AsyncClient(base_url='http://engine/v1', transport=httpx.MockTransport(engine)) as client:
        owner = SimpleNamespace(allow_prompt_logprobs=False, _http_client=client)
        method = forwarding_method(route)
        if route == 'external':
            result = await method(owner, request, 'model', 'checkpoint', client, base_model='base-model')
        else:
            result = await method(owner, 'http://engine', request, 'model', base_model='base-model')
    # Same schema serialization used by call_and_store_result and FutureDB.
    return types.SampleOutput.model_validate_json(result.model_dump_json())


def choice(forced=True):
    return dict(token_ids=[5, 6, 7], finish_reason='stop',
                logprobs={'token_logprobs': [-.4, -.5, -.6]},
                loss_mask=[1., 0., 1.] if forced else [1., 1., 1.],
                forced_token_positions=[1] if forced else [],
                thinking_budget=dict(enforced=True, forced=forced, budget_basis='phase1_generated_tokens', counted_phase1_tokens=1))


@pytest.mark.parametrize('route', ['external', 'forwarding'])
@pytest.mark.parametrize('stop', [None, [99], ['STOP']])
def test_mixed_group_preserves_masks_audit_and_sampled_logprobs(route, stop):
    result = asyncio.run(roundtrip(route, [choice(True), choice(False)], 1, stop=stop))
    assert result.sequences[0].loss_mask == [1., 0., 1.]
    assert result.sequences[0].logprobs == [-.4, 0., -.6]
    assert result.sequences[1].logprobs == [-.4, -.5, -.6]
    assert result.sequences[1].thinking_budget['forced'] is False


@pytest.mark.parametrize('route', ['external', 'forwarding'])
@pytest.mark.parametrize('missing', ['loss_mask', 'thinking_budget', 'forced_token_positions', 'logprobs'])
def test_unannotated_native_response_fails_in_proxy(route, missing):
    c = choice(); c.pop(missing)
    with pytest.raises(ValueError):
        asyncio.run(roundtrip(route, [c], 1))


@pytest.mark.parametrize('route', ['external', 'forwarding'])
def test_legacy_request_remains_unchanged(route):
    c = dict(token_ids=[5], logprobs={'token_logprobs': [-.4]}, finish_reason='stop')
    result = asyncio.run(roundtrip(route, [c], None))
    assert result.sequences[0].tokens == [5]
    assert result.sequences[0].loss_mask is None


@pytest.mark.parametrize('route', ['external', 'forwarding'])
def test_zero_budget_is_forwarded(route):
    c = choice(); c['thinking_budget']['counted_phase1_tokens'] = 0
    asyncio.run(roundtrip(route, [c], 0))
