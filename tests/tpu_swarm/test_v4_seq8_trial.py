import pytest

from tpu.diagnostics.v4_inference.seq8_trial import check_idle, file_sha256, metrics, payload_from_fixture


def fixture():
    return dict(prompt=dict(chunks=[dict(type="encoded_text", tokens=[1, 2]),
                                    dict(type="encoded_text", tokens=[3])]),
                sampling_params=dict(temperature=1, top_p=1, top_k=-1, seed=99,
                                     stop_tokens=[4], stop_strings=None))


def test_preserves_rl_tokens_and_sampling_without_unsupported_seed():
    p = payload_from_fixture(fixture(), "adapter", 13196)
    assert p["prompt"] == [1, 2, 3]
    assert p["n"] == 8 and p["max_tokens"] == 13196
    assert p["stop_token_ids"] == [4]
    assert p["logprobs"] and p["return_token_ids"]
    assert "seed" not in p and "prompt_logprobs" not in p


@pytest.mark.parametrize("tokens,samples", [(0, 8), (22526, 8), (64, 0), (64, 33)])
def test_rejects_invalid_budget(tokens, samples):
    with pytest.raises(ValueError):
        payload_from_fixture(fixture(), "base", tokens, samples)


def test_idle_guard_checks_both_queues_and_adapter_identity():
    row = dict(status=dict(active=0, updating=False, exhausted=[], version=None,
                           committed=None, replicas=[dict(ip="host")]),
               metrics={"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 0})
    check_idle(row, "host", None)
    with pytest.raises(RuntimeError):
        check_idle(row, "host", "adapter")
    row["metrics"]["vllm:num_requests_waiting"] = 1
    with pytest.raises(RuntimeError):
        check_idle(row, "host", None)


def test_metrics_include_cache_pressure_and_preemptions():
    assert metrics('# comment\nvllm:kv_cache_usage_perc{engine="0"} 0.9\n'
                   'vllm:num_preemptions_total{engine="0"} 2\n') == {
        "vllm:kv_cache_usage_perc": 0.9, "vllm:num_preemptions_total": 2}


def test_streaming_hash_works_without_python311_file_digest(tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
