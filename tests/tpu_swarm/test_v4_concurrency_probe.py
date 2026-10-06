import pytest

from tpu.diagnostics.v4_inference.concurrency_probe import request_payload, validate_payload


def test_replays_recorded_request_without_changing_sampling_parameters():
    payload = dict(model="base", prompt=[100, 101], max_tokens=64, logprobs=1, temperature=0)
    assert validate_payload({"request": payload}, "base") is payload


@pytest.mark.parametrize("change", [dict(model="adapter"), dict(n=8), dict(seed=1),
                                     dict(stream=True), dict(max_tokens=0), dict(prompt=[])])
def test_rejects_payloads_outside_the_control(change):
    payload = dict(model="base", prompt=[100], max_tokens=64)
    payload.update(change)
    with pytest.raises(ValueError):
        validate_payload(payload, "base")


def test_distinct_prompts_preserve_length_and_do_not_mutate_shared_payload():
    payload = dict(model="base", prompt=[100, 101, 102], max_tokens=64)
    requests = [request_payload(payload, i, True) for i in range(8)]
    assert len({p["prompt"][0] for p in requests}) == 8
    assert all(p["prompt"][1:] == [101, 102] for p in requests)
    assert payload["prompt"] == [100, 101, 102]
    assert request_payload(payload, 0, False) is payload
