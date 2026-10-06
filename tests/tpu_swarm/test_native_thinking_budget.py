"""Run on CPU; load pure control modules without importing a TPU runtime."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tpu/swarm/ray_train/thinking_budget"
namespace = ModuleType("thinking_test")
namespace.__path__ = [str(PACKAGE)]
sys.modules[namespace.__name__] = namespace


def load(name):
    spec = importlib.util.spec_from_file_location("thinking_test." + name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


budget = load("thinking_budget")
api = load("thinking_budget_api")


def spec(limit=3, transition=(11, 12, 13)):
    return budget.BudgetSpec.parse(dict(budget=limit, start_id=10, end_id=11, transition=list(transition)), 100)


@pytest.mark.parametrize("limit", [0, 1, 3, 17])
def test_exact_budget_and_multi_token_transition(limit):
    settings = spec(limit)
    state = budget.initial_state(settings, [8, 10, 9])
    tokens, positions = [], []
    for i in range(limit+6):
        token, state, forced = budget.reference_step(settings, state, 7)
        tokens.append(token)
        if forced:
            positions.append(i)
    assert tokens == [7]*limit + [11, 12, 13] + [7]*3
    assert positions == [limit, limit+1, limit+2]
    assert budget.audit_output(settings, [10], tokens)[0] == positions


def test_natural_close_remains_sampled_and_no_reopening():
    settings = spec()
    output = [7, 11, 10, 7, 7, 7, 7]
    positions, state = budget.audit_output(settings, [10], output)
    assert positions == []
    assert state[0] == budget.ANSWERING


def test_prompt_without_open_think_waits_for_generated_marker():
    settings = spec(1)
    output = [8, 8, 10, 7, 11, 12, 13, 8]
    assert budget.audit_output(settings, [8], output)[0] == [4, 5, 6]
    assert budget.audit_output(settings, [10, 11], [8]*50)[0] == []


@pytest.mark.parametrize("bad", [-1, True, 1.5, "3", 2**31])
def test_invalid_budget(bad):
    with pytest.raises(ValueError):
        spec(bad)


@pytest.mark.parametrize("transition", [(), (12,), (11, -1), (11, 101), (11,)*129])
def test_invalid_transition(transition):
    with pytest.raises(ValueError):
        spec(transition=transition)


def test_output_audit_rejects_unenforced_budget():
    with pytest.raises(RuntimeError, match="not enforced"):
        budget.audit_output(spec(1), [10], [7, 7])


def test_jax_matches_cpu_oracle_randomized():
    rng = np.random.default_rng(123)
    settings = [spec(int(n)) for n in rng.integers(0, 20, size=16)]
    states = [budget.initial_state(s, [10] if i % 2 else [8]) for i, s in enumerate(settings)]
    device_states = jnp.array(states, dtype=jnp.int32)
    transitions = np.zeros((16, budget.MAX_TRANSITION), np.int32)
    transitions[:, :3] = [11, 12, 13]
    args = [jnp.array(x) for x in ([s.budget for s in settings], [10]*16, [11]*16, transitions, [3]*16)]
    step = jax.jit(budget.apply_budget)
    for _ in range(80):
        sampled = rng.integers(7, 15, size=16, dtype=np.int32)
        valid = rng.integers(0, 2, size=16).astype(bool)
        expected_tokens, expected_states, expected_forced = [], [], []
        for s, state, token, enabled in zip(settings, states, sampled, valid):
            result = budget.reference_step(s, state, int(token)) if enabled else (int(token), state, False)
            expected_tokens.append(result[0]); expected_states.append(result[1]); expected_forced.append(result[2])
        actual, device_states, forced = step(device_states, jnp.array(sampled), *args, jnp.array(valid))
        np.testing.assert_array_equal(actual, expected_tokens)
        np.testing.assert_array_equal(device_states, expected_states)
        np.testing.assert_array_equal(forced, expected_forced)
        states = expected_states
    # Budget values do not become JIT cache keys.
    args[0] = jnp.array([999]*16, dtype=jnp.int32)
    step(device_states, jnp.zeros(16, jnp.int32), *args, jnp.ones(16, bool))
    assert step._cache_size() == 1


def test_request_reordering_pause_padding_and_async_placeholders():
    def request(limit):
        return SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: spec(limit).serialize()}),
            prompt_token_ids=[10], output_token_ids=[], num_computed_tokens=0, num_tokens=1)
    a, b = request(1), request(3)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    runner = SimpleNamespace(requests={"a": a, "b": b},
        input_batch=SimpleNamespace(req_ids=["a", "b"], vocab_size=100),
        mesh=mesh, speculative_config=None, enable_continue_decode=False)
    bank = budget.BudgetRunner()

    def step(ids, valid=True):
        runner.input_batch.req_ids = ids
        for req_id in ids:
            request = runner.requests[req_id]
            request.num_tokens = len(request.prompt_token_ids) + len(request.output_token_ids)
            request.num_computed_tokens = request.num_tokens - 1
        schedule = SimpleNamespace(num_scheduled_tokens={i: int(valid) for i in ids})
        with jax.set_mesh(mesh):
            result = np.asarray(bank.apply(runner, schedule, jnp.array([7, 7, 7, 7]), {0: ids}))
        if valid:
            for req_id in ids:
                runner.requests[req_id].output_token_ids.append(0)
        return result

    assert list(step(["a", "b"])) == [7, 7, 7, 7]
    # CPU output placeholders are intentionally wrong; device state is authoritative.
    a.output_token_ids = [0]
    b.output_token_ids = [0]
    assert list(step(["b", "a"])) == [7, 11, 7, 7]
    assert list(step(["a"], valid=False)) == [7, 7, 7, 7]
    assert list(step(["a"])) == [12, 7, 7, 7]
    assert list(step(["b", "a"])) == [7, 13, 7, 7]
    assert list(step(["a", "b"])) == [7, 11, 7, 7]
    # Scheduler discards an in-flight closing token; repeat it, not the next
    # transition token. The CPU placeholder contains no usable token identity.
    b.output_token_ids.pop()
    assert list(step(["b"])) == [11, 7, 7, 7]
    del runner.requests["a"]
    step(["b"])
    assert "a" not in bank.states
    # A new request may reuse a finished ID, with a different budget.
    runner.requests["b"] = request(0)
    assert list(step(["b"])) == [11, 7, 7, 7]


@pytest.mark.parametrize("mode", ["speculative_config", "enable_continue_decode"])
def test_unsupported_backend_rejected_before_fast_path(mode):
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: spec().serialize()}))
    runner = SimpleNamespace(requests={"a": request}, input_batch=SimpleNamespace(req_ids=["a"]),
                             speculative_config=None, enable_continue_decode=False)
    setattr(runner, mode, True)
    with pytest.raises(ValueError, match="single-step"):
        budget.validate_backend(runner)
    request.sampling_params.extra_args = {}
    budget.validate_backend(runner)


def test_disabled_budget_returns_original_tensor_without_compilation():
    bank = budget.BudgetRunner()
    tokens = jnp.array([1, 2])
    runner = SimpleNamespace(requests={"a": SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))},
                             input_batch=SimpleNamespace(req_ids=["a"]))
    compiled_before = bank.step._cache_size()
    assert bank.apply(runner, None, tokens, None) is tokens
    assert bank.step._cache_size() == compiled_before


def test_dp_group_padding_and_partial_prefill():
    def request(computed):
        return SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: spec(0).serialize()}),
            prompt_token_ids=[8, 10], output_token_ids=[], num_computed_tokens=computed, num_tokens=2)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    runner = SimpleNamespace(requests={"a": request(0), "b": request(1)},
        input_batch=SimpleNamespace(req_ids=["b", "a"], vocab_size=100),
        mesh=mesh, speculative_config=None, enable_continue_decode=False)
    scheduler = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})
    with jax.set_mesh(mesh):
        result = budget.BudgetRunner().apply(runner, scheduler, jnp.full(8, 7), {0: ["a"], 1: ["b"]})
    np.testing.assert_array_equal(result, [7, 7, 7, 7, 11, 7, 7, 7])


def test_deep_rollback_replays_committed_history():
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: spec(3).serialize()}),
        prompt_token_ids=[10], output_token_ids=[], num_computed_tokens=0, num_tokens=1)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    runner = SimpleNamespace(requests={"a": request},
        input_batch=SimpleNamespace(req_ids=["a"], vocab_size=100),
        mesh=mesh, speculative_config=None, enable_continue_decode=False)
    bank = budget.BudgetRunner()
    def step():
        request.num_computed_tokens = len(request.output_token_ids)
        request.num_tokens = 1 + len(request.output_token_ids)
        with jax.set_mesh(mesh):
            result = bank.apply(runner, SimpleNamespace(num_scheduled_tokens={"a": 1}), jnp.array([7]), {0: ["a"]})
        token = int(result[0])
        request.output_token_ids.append(token)
        return token
    for _ in range(10):
        step()
    assert len(bank.history["a"]) == 4
    request.output_token_ids = request.output_token_ids[:1]
    assert [step() for _ in range(5)] == [7, 7, 11, 12, 13]


@pytest.mark.parametrize('completer', [False, True])
def test_stable_decode_has_no_control_transfers_or_eager_device_operations(monkeypatch, completer):
    settings = api.prepare_request(payload(), Tokenizer(), 64)[0] if completer else spec(2)
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: settings.serialize()}),
        prompt_token_ids=[10], output_token_ids=[], num_computed_tokens=0, num_tokens=1)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    runner = SimpleNamespace(requests={"a": request},
        input_batch=SimpleNamespace(req_ids=["a"], vocab_size=100),
        mesh=mesh, speculative_config=None, enable_continue_decode=False)
    bank = budget.BudgetRunner()
    schedule = SimpleNamespace(num_scheduled_tokens={"a": 1})
    tokens = jnp.array([7, 7, 7, 7])
    with jax.set_mesh(mesh):
        bank.apply(runner, schedule, tokens, {0: ["a"]}).block_until_ready()
        request.output_token_ids.append(0)
        original_arrays = bank.batch_arrays
        def forbidden(*args, **kwargs):
            raise AssertionError("hot-path control initiated a device operation")
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_put", forbidden)
            patch.setattr(jnp, "array", forbidden)
            patch.setattr(jnp, "stack", forbidden)
            for i in range(1, 15):
                request.num_computed_tokens, request.num_tokens = i, i+1
                prepared = bank.prepare(runner, schedule, 4, {0: ["a"]})
                if i >= 5:
                    assert prepared is None  # Fully emitted transition: ordinary sampler.
                    request.output_token_ids.append(0)
                    continue
                assert prepared[0] is bank.bank
                assert bank.batch_arrays is original_arrays
                selected, device_bank = bank.step(tokens, *prepared)
                bank.commit(device_bank)
                request.output_token_ids.append(0)
                expected = [7, 7, 11, 12, 13][i] if i < 5 else 7
                assert int(selected[0]) == expected


def test_completed_control_bypasses_device_and_reactivates_on_rollback():
    mesh = Mesh(np.array(jax.devices()), ("x",))
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: spec(1).serialize()}),
        prompt_token_ids=[10], output_token_ids=[], num_computed_tokens=0, num_tokens=1)
    runner = SimpleNamespace(requests={"a": request}, input_batch=SimpleNamespace(req_ids=["a"], vocab_size=100),
                             mesh=mesh, speculative_config=None, enable_continue_decode=False)
    bank = budget.BudgetRunner()
    schedule = SimpleNamespace(num_scheduled_tokens={"a": 1})
    tokens = jnp.array([7])
    def step():
        request.num_computed_tokens = len(request.output_token_ids)
        request.num_tokens = request.num_computed_tokens + 1
        with jax.set_mesh(mesh):
            out = bank.apply(runner, schedule, tokens, {0: ["a"]})
        request.output_token_ids.append(int(out[0]))
        return out
    assert [int(step()[0]) for _ in range(4)] == [7, 11, 12, 13]
    for _ in range(10):
        assert step() is tokens
    request.output_token_ids = request.output_token_ids[:3]
    assert int(step()[0]) == 13
    assert step() is tokens
    # An already-closed prompt never initializes a device bank.
    request.prompt_token_ids = [10, 11]
    request.output_token_ids = []
    closed = budget.BudgetRunner()
    assert closed.apply(runner, schedule, tokens, {0: ["a"]}) is tokens
    assert closed.bank is None


def test_device_bank_growth_preserves_paused_requests_and_mixed_rows():
    mesh = Mesh(np.array(jax.devices()), ("x",))
    runner = SimpleNamespace(requests={}, input_batch=SimpleNamespace(req_ids=[], vocab_size=100),
        mesh=mesh, speculative_config=None, enable_continue_decode=False)
    bank = budget.BudgetRunner()
    def step(ids):
        runner.input_batch.req_ids = ids
        for key in ids:
            request = runner.requests[key]
            request.num_computed_tokens = len(request.output_token_ids)
            request.num_tokens = 1 + len(request.output_token_ids)
        with jax.set_mesh(mesh):
            out = bank.apply(runner, SimpleNamespace(num_scheduled_tokens={k: 1 for k in ids}),
                             jnp.full(4, 7), {0: ids})
        for key in ids:
            runner.requests[key].output_token_ids.append(0)
        return np.asarray(out)
    for i in range(20):
        runner.requests[str(i)] = SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args={budget.KEY: spec(1).serialize()}),
            prompt_token_ids=[10], output_token_ids=[])
        assert step([str(i)])[0] == 7
    assert bank.capacity == 32
    runner.requests["plain"] = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}),
                                               prompt_token_ids=[10], output_token_ids=[])
    np.testing.assert_array_equal(step(["19", "plain", "0"]), [11, 7, 11, 7])
    np.testing.assert_array_equal(step(["0", "19", "plain"]), [12, 12, 7, 7])


@pytest.mark.parametrize("sampling", [False, True])
@pytest.mark.parametrize("completer", [False, True])
def test_actual_fused_sampler_preserves_other_rows_and_processed_logits(monkeypatch, sampling, completer, patched_core):
    import ast
    from dataclasses import dataclass
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec

    # Execute the actual sampler functions without importing TPU-only vLLM
    # platform/bootstrap modules into a CPU test process.
    source = patched_core / "tpu_inference/layers/jax/sample/sampling.py"
    tree = ast.parse(source.read_text())
    names = {"sample", "sample_with_budget", "_apply_sampling_transforms"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    binary_path = patched_core / "tpu_inference/layers/common/binary_search.py"
    module_spec = importlib.util.spec_from_file_location("thinking_binary_test", binary_path)
    binary = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(binary)
    @partial(jax.tree_util.register_dataclass,
             data_fields=["temperature", "top_k", "top_p", "_cache_collision_dummy"],
             meta_fields=["do_sampling"])
    @dataclass
    class Metadata:
        temperature: object
        top_k: object
        top_p: object
        _cache_collision_dummy: object = None
        do_sampling: bool = False
    scope = dict(jax=jax, jnp=jnp, Mesh=Mesh, NamedSharding=NamedSharding, P=PartitionSpec,
                 TPUSupportedSamplingMetadata=Metadata, _SAMPLING_EPS=1e-5,
                 ShardingAxisName=SimpleNamespace(ATTN_DATA="x"),
                 topk_mask=binary.topk_mask, topp_mask=binary.topp_mask)
    monkeypatch.setitem(sys.modules, "tpu_inference.runner.thinking_budget_device",
                        sys.modules["thinking_test.thinking_budget_device"])
    exec(compile(tree, str(source), "exec"), scope)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    settings = api.prepare_request(payload() | {'thinking_token_budget': 0}, Tokenizer(), 64)[0] if completer else spec(0)
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY: settings.serialize()}),
        prompt_token_ids=[10], output_token_ids=[], num_computed_tokens=0, num_tokens=1)
    runner = SimpleNamespace(requests={"a": request}, input_batch=SimpleNamespace(req_ids=["a"], vocab_size=100),
                             mesh=mesh, speculative_config=None, enable_continue_decode=False)
    metadata = Metadata(jnp.full(4, 0.7), jnp.full(4, 20), jnp.full(4, 0.95), do_sampling=sampling)
    logits = jnp.asarray(np.random.default_rng(42).normal(size=(4, 100)), jnp.float32)
    rng = jax.random.key(17)
    bank = budget.BudgetRunner()
    with jax.set_mesh(mesh):
        prepared = bank.prepare(runner, SimpleNamespace(num_scheduled_tokens={"a": 1}), 4, {0: ["a"]})
        baseline, original_logits = scope["sample"](rng, mesh, logits, metadata)
        result, processed, state = scope["sample_with_budget"](rng, mesh, logits, metadata, *prepared)
    assert int(result[0]) == 11
    np.testing.assert_array_equal(result[1:], baseline[1:])
    np.testing.assert_array_equal(processed, original_logits)
    assert int(state.positions[0]) == 1


def test_worker_flag_is_explicit_and_standard_sampler_unchanged(patched_core):
    import ast
    runner = ast.parse((patched_core / "tpu_inference/runner/tpu_runner.py").read_text())
    cls = next(n for n in runner.body if isinstance(n, ast.ClassDef) and n.name == "TPUModelRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "self._thinking_budget_runner")
    code = compile(ast.Expression(assignment.value), "<opt-in>", "eval")
    for env, enabled in (({}, False), ({"SKYRL_TPU_THINKING_BUDGET": "0"}, False),
                         ({"SKYRL_TPU_THINKING_BUDGET": "1"}, True)):
        marker = object()
        actual = eval(code, {"os": SimpleNamespace(environ=env), "BudgetRunner": lambda: marker})
        assert actual is (marker if enabled else None)


class Tokenizer:
    eos_token_id = 99
    def get_vocab(self):
        return {str(i): i for i in range(100)}
    def batch_decode(self, batch, **kwargs):
        pieces = {10: "<think>", 11: "</think>", 12: "cue", 13: "answer"}
        return ["".join(pieces.get(i, "x") for i in ids) for ids in batch]
    def __len__(self):
        return 100
    def encode(self, text, **kwargs):
        return {"<think>": [10], "</think>": [11], "close and answer": [11, 12, 13],
                "bad eos": [11, 99], "\n</think>\n\n": [11, 12]}[text]


def payload():
    return dict(prompt=[8, 10], thinking_token_budget=2,
                forced_thinking_transition="close and answer", max_tokens=10)


def test_vocab_size_is_not_materialized_per_prompt_token():
    class CountedTokenizer(Tokenizer):
        calls = 0
        def __len__(self):
            self.calls += 1
            return super().__len__()
    tokenizer = CountedTokenizer()
    request = payload() | {"prompt": [8] * 8192 + [10]}
    api.prepare_request(request, tokenizer, 16384)
    assert tokenizer.calls == 1
    app = api.ThinkingBudgetMiddleware(None, tokenizer, 16384)
    assert tokenizer.calls == 2
    for _ in range(3):
        api.prepare_request(payload() | {"prompt": [8] * 8192 + [10]},
                            tokenizer, 16384, app.vocab_size)
    assert tokenizer.calls == 2
    with pytest.raises(ValueError, match="invalid token"):
        api.prepare_request(payload() | {"prompt": [100]}, tokenizer, 16384, app.vocab_size)


def test_api_preserves_forced_mask_and_sampled_logprobs():
    request = payload()
    settings, prompt = api.prepare_request(request, Tokenizer(), 64)
    assert request["thinking_token_budget"] is None
    assert budget.BudgetSpec.parse(request["vllm_xargs"][budget.KEY]) == settings
    response = dict(choices=[dict(token_ids=[7, 7, 11, 12, 13, 8], logprobs={"token_logprobs": [-1.0]*6})])
    result = api.annotate_response(response, settings, prompt)["choices"][0]
    assert result["forced_token_positions"] == [2, 3, 4]
    assert result["loss_mask"] == [1, 1, 0, 0, 0, 1]
    assert result["logprobs"]["token_logprobs"] == [-1.0]*6


@pytest.mark.parametrize("changes", [dict(stream=True), dict(echo=True), dict(prompt="hello"),
    dict(max_tokens=3), dict(max_tokens=100), dict(forced_thinking_transition="bad eos"),
    dict(stop_token_ids=[11]), dict(stop="answer"), dict(thinking_token_budget=True)])
def test_api_rejects_unsafe_combinations(changes):
    request = payload() | changes
    with pytest.raises(ValueError):
        api.prepare_request(request, Tokenizer(), 64)


def test_asgi_request_and_response_round_trip():
    async def run():
        calls, output = [], []
        async def app(scope, receive, send):
            data = json.loads((await receive())["body"])
            calls.append(data)
            response = dict(choices=[dict(token_ids=[7, 7, 11, 12, 13, 8])])
            body = json.dumps(response).encode()
            await send(dict(type="http.response.start", status=200, headers=[]))
            await send(dict(type="http.response.body", body=body[:5], more_body=True))
            await send(dict(type="http.response.body", body=body[5:], more_body=False))
        async def receive():
            return dict(type="http.request", body=json.dumps(payload()).encode())
        async def send(message):
            output.append(message)
        middleware = api.ThinkingBudgetMiddleware(app, Tokenizer(), 64)
        await middleware(dict(type="http", path="/v1/completions", method="POST", headers=[]), receive, send)
        assert calls[0]["return_token_ids"]
        assert output[0]["status"] == 200
        assert json.loads(output[1]["body"])["choices"][0]["loss_mask"] == [1, 1, 0, 0, 0, 1]
    asyncio.run(run())


@pytest.mark.parametrize("changes,supported,status", [
    ({"thinking_token_budget": -1.0}, True, 400),
    ({"thinking_token_budget": 1.5}, True, 400),
    ({"vllm_xargs": "invalid"}, True, 400),
    ({"vllm_xargs": {budget.KEY: "reserved"}}, True, 400),
    ({}, False, 400),
    ({}, True, 500),
])
def test_middleware_fails_closed(changes, supported, status):
    async def run():
        output, calls = [], []
        async def app(scope, receive, send):
            calls.append(True)
            # Simulate an engine which silently ignores the control.
            await send(dict(type="http.response.start", status=200, headers=[]))
            await send(dict(type="http.response.body", body=b'{"choices":[{"token_ids":[7,7,7]}]}'))
        async def receive():
            return dict(type="http.request", body=json.dumps(payload() | changes).encode())
        async def send(message):
            output.append(message)
        await api.ThinkingBudgetMiddleware(app, Tokenizer(), 64, supported)(
            dict(type="http", path="/v1/completions", method="POST"), receive, send)
        assert output[0]["status"] == status
        assert bool(calls) == (status == 500)
    asyncio.run(run())


def test_wrapper_handles_prebuilt_stack_and_preserves_routes_state_lifespan():
    from contextlib import asynccontextmanager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.testclient import TestClient

    lifecycle = []
    @asynccontextmanager
    async def lifespan(app):
        lifecycle.append("start")
        yield
        lifecycle.append("stop")

    app = Starlette(lifespan=lifespan)
    app.middleware_stack = app.build_middleware_stack()
    wrapper = api.ThinkingBudgetApp(app, Tokenizer(), 64)
    wrapper.state.marker = "shared"
    assert app.state.marker == "shared"
    async def endpoint(request):
        return JSONResponse({"marker": request.app.state.marker})
    wrapper.add_route("/health", endpoint)
    with TestClient(wrapper) as client:
        assert client.get("/health").json() == {"marker": "shared"}
    assert lifecycle == ["start", "stop"]



def test_real_qwen_tokenizer_when_fixture_available():
    import os
    location = os.environ.get("THINKING_TEST_TOKENIZER")
    if not location:
        pytest.skip("set THINKING_TEST_TOKENIZER to the pinned Qwen tokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    prompt = tokenizer.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    request = dict(prompt=prompt, thinking_token_budget=64, max_tokens=160,
                   forced_thinking_transition="\n</think>\n\nHere is the final complete program:\n\n```python\n")
    settings, _ = api.prepare_request(request, tokenizer, 22528)
    assert budget.initial_state(settings, prompt)[0] == budget.THINKING


@pytest.fixture
def patched_core(tmp_path):
    import tarfile
    from tpu.swarm.ray_train.thinking_budget.install import install
    with tarfile.open(ROOT / "tests/tpu_swarm/fixtures/thinking_budget_core.tar.gz") as archive:
        archive.extractall(tmp_path, filter="data")
    (tmp_path / "tpu").mkdir()
    # The installer now verifies the packaged client contract before touching
    # the serving source. Include that actual client alongside the core fixture.
    import shutil
    client = Path("third_party/discover/ttt_discover/tinker_utils/completers.py")
    (tmp_path / client).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / client, tmp_path / client)
    install(tmp_path)
    return tmp_path / "third_party/tpu-inference"


@pytest.mark.parametrize("start,end,transition", [
    ([10], [11], [11, 12]),
    ([10, 12, 13], [11], [11]),
    ([328, 19669, 200023], [328, 76976, 200023], [200007, 200022, 140680, 328, 76976, 200023]),
])
def test_multimodel_twenty_thousand_cap_on_device(start, end, transition):
    settings = budget.BudgetSpec.parse(dict(budget=20000, start_id=start[0], end_id=end[0],
        start_sequence=start, end_sequence=end, transition=transition), 250000)
    padded = lambda ids, length: jnp.array([ids + [-1]*(length-len(ids))], dtype=jnp.int32)
    trans = jnp.array([transition + [0]*(budget.MAX_TRANSITION-len(transition))])
    init = jnp.array([budget.initial_state(settings, start)])
    def step(state, sampled):
        tokens, state, forced = budget.apply_budget(state, sampled[None], jnp.array([20000]),
            padded(start, budget.MAX_MARKER), padded(end, budget.MAX_MARKER), trans,
            jnp.array([len(transition)]), jnp.array([True]))
        return state, (tokens[0], forced[0])
    state, (tokens, forced) = jax.jit(lambda: jax.lax.scan(step, init, jnp.full(20000+len(transition)+5, 7)))()
    output = np.asarray(tokens).tolist()
    assert output == [7]*20000 + transition + [7]*5
    assert int(np.sum(forced)) == len(transition)
    positions, audit = budget.audit_output(settings, start, output)
    assert positions == list(range(20000, 20000+len(transition)))
    np.testing.assert_array_equal(np.asarray(state)[0], audit)


def test_muse_natural_close_and_partial_transition_at_cap():
    start, end, transition = [328, 19669, 200023], [328, 76976, 200023], [200007, 200022, 140680, 328, 76976, 200023]
    def settings(cap):
        return budget.BudgetSpec.parse(dict(budget=cap, start_id=start[0], end_id=end[0],
            start_sequence=start, end_sequence=end, transition=transition))
    # A second self message does not release the cap; only the user channel does.
    natural = start + [7, 200007, 200022, 140680] + start + [8, 200007, 200022, 140680] + end + [9]
    assert budget.audit_output(settings(100), [], natural)[0] == []
    assert budget.audit_output(settings(100), [], natural)[1][0] == budget.ANSWERING
    # Cap expires after the first token of the natural user header: complete it once.
    partial = [7, 328, 76976, 200023, 9]
    positions, state = budget.audit_output(settings(2), start, partial)
    assert positions == [2, 3]
    assert state[0] == budget.ANSWERING
    # Markers may be split between an assistant-prefix prompt and its output.
    assert budget.initial_state(settings(2), [99, 328]) == (budget.WAITING, 0, 1)


def test_patch_is_against_pinned_core_and_keeps_other_code(patched_core):
    runner = (patched_core / 'tpu_inference/runner/tpu_runner.py').read_text()
    assert 'SERIALIZE_MODEL_AND_SAMPLING' in runner
    assert 'sample_with_budget(' in runner
    assert (patched_core / 'tpu_inference/runner/thinking_budget_api.py').is_file()


@pytest.mark.parametrize('model', ['qwen3.5-27b', 'gemma4-31b', 'muse-glimmer-30b'])
def test_real_tokenizer_markers_api_masks_and_one_request(model):
    fixture = json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    class FrozenTokenizer:
        eos_token_id = None
        def __len__(self):
            return fixture['vocab_size']
        def encode(self, text, **kwargs):
            return {fixture['text'][k]: fixture[k] for k in ('start', 'end', 'transition')}[text]
        def get_vocab(self):
            return {str(i): int(i) for i in fixture['pieces']}
        def batch_decode(self, batch, **kwargs):
            return [''.join(fixture['pieces'][str(i)] for i in ids) for ids in batch]
    async def run():
        calls, output = [], []
        async def backend(scope, receive, send):
            request = json.loads((await receive())['body'])
            calls.append(request)
            settings = budget.BudgetSpec.parse(request['vllm_xargs'][budget.KEY])
            state = budget.initial_state(settings, request['prompt'])
            tokens = []
            for _ in range(2 + len(settings.transition) + 3):
                token, state, _ = budget.reference_step(settings, state, 7)
                tokens.append(token)
            body = json.dumps({'choices': [{'token_ids': tokens}]}).encode()
            await send(dict(type='http.response.start', status=200, headers=[]))
            await send(dict(type='http.response.body', body=body))
        async def receive():
            return dict(type='http.request', body=json.dumps(dict(prompt=fixture['start'],
                thinking_token_budget=2, max_tokens=64)).encode())
        async def send(message):
            output.append(message)
        await api.ThinkingBudgetMiddleware(backend, FrozenTokenizer(), 40960, thinking_format=model)(
            dict(type='http', path='/v1/completions', method='POST'), receive, send)
        assert len(calls) == 1
        assert output[0]['status'] == 200
        choice = json.loads(output[1]['body'])['choices'][0]
        assert choice['thinking_budget'] == dict(enforced=True, forced=True, thinking_tokens=2, answered=True,
            budget_basis='phase1_generated_tokens', counted_phase1_tokens=2)
        assert choice['loss_mask'] == [1,1]+[0]*len(fixture['transition'])+[1]*3
    asyncio.run(run())


def test_multitoken_marker_device_matches_oracle_with_partial_close_and_padding():
    settings = budget.BudgetSpec.parse(dict(budget=5, start_id=10, end_id=10,
        start_sequence=[10, 12, 13], end_sequence=[10, 14, 13], transition=[20,10,14,13]))
    streams = [
        [10,12,13,7,7,10,14,13,8,8,8,8,8,8,8],  # natural
        [10,12,13,7,7,7,7,10,14,13,8,8,8,8,8],  # cap mid-marker
        [10,12,13,7,7,7,7,7,7,7,7,7,7,7,7],   # forced full transition
        [10,12,13,10,99,7,7,7,7,7,7,7,7,7,7], # mismatched marker prefix
    ]
    states = [budget.initial_state(settings, []) for _ in streams]
    arrays = jnp.array(states)
    transitions = jnp.array([list(settings.transition)+[0]*(budget.MAX_TRANSITION-len(settings.transition))]*4)
    starts, ends = jnp.array([settings.start]*4), jnp.array([settings.end]*4)
    step = jax.jit(budget.apply_budget)
    for sampled in zip(*streams):
        expected = [budget.reference_step(settings, s, t) for s,t in zip(states,sampled)]
        out, arrays, forced = step(arrays,jnp.array(sampled),jnp.full(4,5),starts,ends,transitions,jnp.full(4,4),jnp.ones(4,bool))
        states = [item[1] for item in expected]
        np.testing.assert_array_equal(out,[item[0] for item in expected])
        np.testing.assert_array_equal(arrays,states)
        np.testing.assert_array_equal(forced,[item[2] for item in expected])


@pytest.fixture(scope='module')
def existing_completers():
    """Execute the actual client methods with only transport/types stubbed."""
    import ast
    path = ROOT/'third_party/discover/ttt_discover/tinker_utils/completers.py'
    names = {'NativeCompletionError', 'QwenTwoPhaseTokenCompleter', 'GemmaTwoPhaseTokenCompleter', 'MuseTwoPhaseTokenCompleter'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    namespace = dict(TokenCompleter=object, TokensWithLogprobs=SimpleNamespace,
        tinker=SimpleNamespace(ModelInput=lambda **kw: SimpleNamespace(**kw),
            SamplingParams=lambda **kw: SimpleNamespace(**kw),
            types=SimpleNamespace(EncodedTextChunk=lambda **kw: SimpleNamespace(**kw))),
        logger=SimpleNamespace(info=lambda *a, **kw: None))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return dict(zip(('qwen3.5-27b','gemma4-31b','muse-glimmer-30b'),
                    (namespace[n] for n in ('QwenTwoPhaseTokenCompleter','GemmaTwoPhaseTokenCompleter','MuseTwoPhaseTokenCompleter'))))


def compat_settings(fixture, cap):
    return budget.BudgetSpec.parse(dict(budget=cap, start_id=fixture['start'][0], end_id=fixture['end'][0],
        start_sequence=fixture['start'], end_sequence=fixture['end'], transition=fixture['transition'],
        completer_detector=fixture['completer_detector']), fixture['vocab_size'])


@pytest.mark.parametrize('marker', ['</think>', '<channel|>', 'to=user'])
def test_compiled_text_detector_matches_decoded_substring(marker):
    # Use multiple tokenizations, tokens containing a complete marker, and
    # mismatched/overlapping prefixes; compare with Python's actual substring
    # predicate on the accumulated decoded output rather than a second FSM.
    pieces = list(dict.fromkeys(['X',' ',*marker,marker,'prefix '+marker+' suffix',
                  *[marker[:i] for i in range(1,len(marker))],
                  *[marker[i:] for i in range(1,len(marker))]]))
    class Pieces:
        def get_vocab(self):return dict(zip(pieces,range(len(pieces))))
        def batch_decode(self,batch,**kw):return [''.join(pieces[i] for i in ids) for ids in batch]
    classes, table = budget.text_detector(budget.build_text_detector(Pieces(),marker,len(pieces)))
    for stream in np.random.default_rng(8).integers(0,len(pieces),size=(500,16)):
        text,state='',0
        for token in stream:
            text+=pieces[token]
            state=table[state][classes[token]]
            assert (state==len(table)-1)==(marker in text)


@pytest.mark.parametrize('model', ['qwen3.5-27b','gemma4-31b','muse-glimmer-30b'])
@pytest.mark.parametrize('grouped', [False, True])
def test_native_matches_actual_two_phase_continuations(existing_completers, monkeypatch, model, grouped):
    monkeypatch.setenv('TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE', '0')
    f = json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    cls = existing_completers[model]
    # Fail immediately if the existing completer ever changes its protocol.
    assert api.FORMATS[model][2] == cls.THINK_CLOSE + cls.ANSWER_CUE
    assert api.CLOSE_MARKERS[model] == cls.THINK_CLOSE_MARKER
    for example in f['compat_examples'].values():
        first, tail = example['tokens'], f['answer_tail']
        policy = cls()
        policy.phase1_max_tokens = 100 + len(first)
        policy.context_window = policy.phase1_max_tokens + len(f['transition']) + len(tail) + 50
        policy.temperature, policy.context_buffer, policy.min_think_tokens = 0.8, 50, 0
        class RecordedTokenizer:
            def decode(self, ids):
                assert ids == first
                return example['decoded']
            def encode(self, text, **kw):
                assert text == cls.THINK_CLOSE + cls.ANSWER_CUE
                return f['transition']
        policy.tokenizer = RecordedTokenizer()
        calls = []
        async def sample(chunks, stop, max_tokens):
            calls.append((chunks, max_tokens))
            tokens = tail if grouped or len(calls) > 1 else first
            return tokens, [-1.0]*len(tokens)
        async def sample_async(**kw):
            assert kw['sampling_params'].max_tokens == len(first)
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=first,logprobs=[-1.0]*len(first))])
        policy._sample = sample
        policy.sampling_client = SimpleNamespace(sample_async=sample_async)
        prompt = SimpleNamespace(length=100, chunks=[])
        result = (asyncio.run(policy.sample_group(prompt, ['STOP'], 1))[0] if grouped
                  else asyncio.run(policy(prompt, ['STOP'])))
        expected_forced = [] if cls.THINK_CLOSE_MARKER in example['decoded'] else f['transition']
        assert result.tokens == first + expected_forced + tail
        settings = compat_settings(f, len(first))
        # Markers in the prompt do not affect the existing completer's check.
        positions, state = budget.audit_output(settings, f['end'], result.tokens)
        assert positions == [i for i, weight in enumerate(result.maybe_mask) if weight == 0]
        assert state[0] == budget.ANSWERING
        assert calls[-1][1] == len(tail) + (0 if expected_forced else len(f['transition']))


@pytest.mark.parametrize('model', ['qwen3.5-27b','gemma4-31b','muse-glimmer-30b'])
def test_completer_detection_device_batch_and_rollback(model):
    f = json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    marker = api.CLOSE_MARKERS[model]
    examples = list(f['compat_examples'].values())
    settings = [compat_settings(f, len(e['tokens'])) for e in examples]
    streams = [e['tokens'] + ([] if marker in e['decoded'] else f['transition']) + f['answer_tail'] for e in examples]
    n, length = len(streams), max(map(len, streams))
    padded = lambda ids: ids + [-1]*(budget.MAX_MARKER-len(ids))
    starts, ends = jnp.array([padded(f['start'])]*n), jnp.array([padded(f['end'])]*n)
    transitions = jnp.array([f['transition']+[0]*(budget.MAX_TRANSITION-len(f['transition']))]*n)
    detector = tuple(jnp.asarray(x,jnp.int32) for x in budget.text_detector(f['completer_detector']))
    tokens = np.full((length,n), 7,np.int32)
    valid = np.zeros((length,n),bool)
    for i, stream in enumerate(streams):
        tokens[:len(stream),i], valid[:len(stream),i] = stream, True
    def step(states, inputs):
        sampled, active = inputs
        out, states, forced = budget.apply_budget(states,sampled,jnp.array([s.budget for s in settings]),
            starts,ends,transitions,jnp.full(n,len(f['transition'])),active,detector)
        return states,(out,forced)
    final,(actual,forced) = jax.jit(lambda: jax.lax.scan(step,jnp.array([budget.initial_state(s,[]) for s in settings]),
                                                     (jnp.array(tokens),jnp.array(valid))))()
    np.testing.assert_array_equal(actual,tokens)
    for i,(s,stream) in enumerate(zip(settings,streams)):
        positions,state=budget.audit_output(s,[],stream)
        assert np.flatnonzero(np.asarray(forced)[:,i]).tolist()==positions
        np.testing.assert_array_equal(final[i],state)
    # Also exercise the production bank path, including a rollback at the cap.
    s = settings[0]
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={budget.KEY:s.serialize()}),
        prompt_token_ids=f['end'],output_token_ids=[],num_computed_tokens=0,num_tokens=len(f['end']))
    mesh=Mesh(np.array(jax.devices()),('x',))
    runner=SimpleNamespace(requests={'r':request},input_batch=SimpleNamespace(req_ids=['r'],vocab_size=f['vocab_size']+256),
                           mesh=mesh,speculative_config=None,enable_continue_decode=False)
    bank=budget.BudgetRunner()
    schedule=SimpleNamespace(num_scheduled_tokens={'r':1})
    def run_token(token):
        request.num_tokens=len(request.prompt_token_ids)+len(request.output_token_ids)
        request.num_computed_tokens=request.num_tokens-1
        with jax.set_mesh(mesh):out=bank.apply(runner,schedule,jnp.array([token]),None)
        request.output_token_ids.append(int(out[0]));return int(out[0])
    for token in streams[0]:assert run_token(token)==token
    request.output_token_ids=request.output_token_ids[:s.budget]
    assert [run_token(7) for _ in f['transition']]==f['transition']


@pytest.mark.parametrize('model', ['qwen3.5-27b','gemma4-31b','muse-glimmer-30b'])
def test_detector_close_completing_exactly_on_cap_does_not_force(model):
    f = json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    first = [7] + f['end']
    settings = compat_settings(f, len(first))
    state = budget.initial_state(settings, [])
    for token in first:
        out, state, forced = budget.reference_step(settings, state, token)
        assert out == token and not forced
    assert state[0] == budget.ANSWERING
    out, state, forced = budget.reference_step(settings, state, 7)
    assert out == 7 and not forced
