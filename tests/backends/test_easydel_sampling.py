from types import SimpleNamespace

import pytest

from skyrl.backends.easydel_sampling import batch_esurge_generate


class FakeEngine:
    _sampling_params_callback = None

    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, request_id, use_tqdm):
        assert use_tqdm is False
        observed = []
        outputs = []
        for prompt, req_id in zip(prompts, request_id, strict=True):
            params = self._sampling_params_callback(
                sampling_params,
                {"request_id": req_id, "prompt": prompt, "engine": self},
            )
            observed.append((prompt, params.seed, params.max_tokens))
            outputs.append(SimpleNamespace(request_id=req_id, outputs=[params.seed]))
        self.calls.append(observed)
        return list(reversed(outputs))


def test_batch_preserves_params_order_and_restores_callback():
    engine = FakeEngine()
    params = {
        "a": SimpleNamespace(seed=11, max_tokens=101),
        "b": SimpleNamespace(seed=22, max_tokens=202),
        "c": SimpleNamespace(seed=33, max_tokens=303),
    }

    outputs = batch_esurge_generate(
        engine=engine,
        prompts=["one", "two", "three"],
        params_by_request=params,
        request_ids=["a", "b", "c"],
    )

    assert engine.calls == [[("one", 11, 101), ("two", 22, 202), ("three", 33, 303)]]
    assert [outputs[key].outputs[0] for key in ("a", "b", "c")] == [11, 22, 33]
    assert engine._sampling_params_callback is None


def test_callback_is_restored_after_generation_failure():
    original = object()
    engine = FakeEngine()
    engine._sampling_params_callback = original
    engine.generate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("boom")
    )

    with pytest.raises(RuntimeError, match="boom"):
        batch_esurge_generate(
            engine=engine,
            prompts=["one"],
            params_by_request={"a": SimpleNamespace(seed=1, max_tokens=2)},
            request_ids=["a"],
        )

    assert engine._sampling_params_callback is original
