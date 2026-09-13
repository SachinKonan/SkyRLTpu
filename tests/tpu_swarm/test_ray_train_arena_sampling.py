import json
from pathlib import Path
import tarfile

import pytest

from tpu.swarm.ray_train.arena_sampling import extract_code, judge_env, generation_payload
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config

PROFILE = Path(__file__).resolve().parents[2] / "tpu/swarm/ray_train/profiles/qwen35_rglru_sampling_v5p_32.json"


def test_judge_head_is_not_available_to_inference():
    config = Config.load(PROFILE)
    assert config.inference_hosts == 3
    assert workload_resources(config, 0) == {"TPU": 0}
    assert all(workload_resources(config, r) == {"TPU": 4} for r in (1, 2, 3))
    bad = config.to_dict()
    bad["inference_only_ranks"] = [0, 1, 2]
    with pytest.raises(ValueError, match="judge host"):
        Config.from_dict(bad)


def test_bundle_contains_actual_judge_and_seed(tmp_path):
    archive, _, _ = build(PROFILE, tmp_path)
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
        for name in ("judge/ray_pool.py", "judge/worker.py", "judge/queue.py", "rl/seed_rglru.py"):
            assert "tpu/pallas_arena/" + name in names


def test_judge_cannot_inherit_serving_ray_or_all_chip_pin(monkeypatch):
    for key in ("RAY_ADDRESS", "TPU_VISIBLE_CHIPS", "TPU_PROCESS_ADDRESSES"):
        monkeypatch.setenv(key, "inherited")
    env = judge_env(None)
    assert not any(k in env for k in ("RAY_ADDRESS", "TPU_VISIBLE_CHIPS", "TPU_PROCESS_ADDRESSES"))
    assert env["ARENA_CHILD_JAX_PLATFORMS"] == "cpu"


def test_only_extract_complete_answer_not_thinking():
    assert extract_code("```python\nwrong\n```</think>```python\ndef kernel(): pass\n```") == "def kernel(): pass\n"
    with pytest.raises(ValueError):
        extract_code("</think>```python\ndef kernel():")


def test_sampling_request_respects_tpu_ingress_contract():
    payload = generation_payload("Qwen/Qwen3.5-27B", "prompt")
    assert "seed" not in payload
    assert not payload.get("stream")
    assert payload["temperature"] > 0 and 0 < payload["top_p"] <= 1


MULTI_PROFILE = PROFILE.with_name("rglru_three_models_v4_32.json")


def test_each_host_loads_its_model_and_cache_without_preset_leakage():
    config = Config.load(MULTI_PROFILE)
    children = [config.for_inference_rank(r) for r in (1, 2, 3)]
    assert [c.model for c in children] == config.served_models
    assert len({c.cache.hf for c in children}) == 3
    assert len({c.cache.inference_compile for c in children}) == 3
    assert children[1].inference.max_model_length == 16384
    assert "--disable-chunked-mm-input" in children[1].inference.extra_args
    assert children[2].inference.ragged_conv1d
    assert children[2].inference.unset_plugins
    assert children[0].inference.ragged_conv1d
    assert not children[1].inference.ragged_conv1d
    assert all(c.inference.memory_utilization == 0.8 for c in children)
    assert config.for_inference_rank(0) is config
    bad = config.to_dict()
    bad["arena_models"][2] = bad["arena_models"][0]
    with pytest.raises(ValueError, match="distinct"):
        Config.from_dict(bad)


def test_v4_qwen_uses_compatible_convolution_without_changing_other_flags(tmp_path):
    from tpu.swarm.ray_train.commands import inference_environment
    qwen = Config.load(MULTI_PROFILE).for_inference_rank(1)
    original = qwen.to_dict()
    original["inference"]["ragged_conv1d"] = False
    before = inference_environment(Config.from_dict(original), tmp_path, tmp_path / "run")
    after = inference_environment(qwen, tmp_path, tmp_path / "run")
    assert {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)} == {
        "USE_JAX_RAGGED_CONV1D"}
    assert after["USE_JAX_RAGGED_CONV1D"] == "1"
    assert qwen.inference.chunk_tokens == 4096
    assert qwen.inference.memory_utilization == 0.8
    assert "jaxconv" in qwen.cache.inference_compile_seed
    assert "jaxconv" in qwen.cache.inference_compile
    assert not Config.load(PROFILE).inference.ragged_conv1d


def test_muse_comparison_host_allows_native_registration(monkeypatch, tmp_path):
    from tpu.swarm.ray_train.commands import inference_environment
    monkeypatch.setenv("VLLM_PLUGINS", "stale_restriction")
    config = Config.load(MULTI_PROFILE)
    for rank in (1, 2, 3):
        child = Config.from_dict(config.for_inference_rank(rank).to_dict())
        env = inference_environment(child, tmp_path, tmp_path / "run")
        if child.model_preset == "muse-glimmer-30b":
            assert "VLLM_PLUGINS" not in env
            assert env["TPU_BACKEND_TYPE"] == "jax"
            assert env["MODEL_IMPL_TYPE"] == "vllm"
            assert env["VLLM_LORA_RESOLVER_CACHE_DIR"] == str(tmp_path / "run/loras")
        else:
            assert env["VLLM_PLUGINS"] == "lora_filesystem_resolver"


@pytest.mark.parametrize("preset,bos,separator", [
    ("qwen3.5-27b", "<|im_start|>", "</think>"),
    ("gemma4-31b", "<bos>", "<channel|>"),
    ("muse-glimmer-30b", "<|begin_of_text|>", "to=user<|message|>"),
])
def test_model_answer_channels_and_prompt_framing(preset, bos, separator):
    from tpu.swarm.ray_train.arena_sampling import render_prompt
    prompt, stops = render_prompt(preset, "the same task")
    assert prompt.startswith(bos) and "the same task" in prompt
    assert "<|eom|>" not in stops
    text = "```python\nwrong\n```" + separator + "```python\ncorrect\n```"
    assert extract_code(text, preset) == "correct\n"
    with pytest.raises(ValueError, match="answer channel"):
        extract_code("```python\nonly thinking\n```", preset)


def test_model_routing_is_independent_of_host_order(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from fastapi import HTTPException
    from tpu.swarm.ray_train import serving
    async def check():
        config = Config.load(MULTI_PROFILE).to_dict()
        config["root"] = str(tmp_path)
        names = list(reversed(Config.from_dict(config).served_models))
        handles = [object() for _ in names]
        cls = serving.Ingress.func_or_class.__mro__[1]
        gateway = cls(config, handles, SimpleNamespace(), ["a", "b", "c"], names)
        try:
            for name, handle in zip(names, handles):
                assert gateway.select_engine(name) is handle
                assert gateway.select_engine(name) is handle
            with pytest.raises(HTTPException):
                gateway.select_engine("unknown")
            with pytest.raises(HTTPException, match="inference-only"):
                await gateway.upload(None, "adapter")
        finally:
            await gateway.http.aclose()
    asyncio.run(check())


def test_summary_keeps_invalid_and_infrastructure_outcomes_in_denominator():
    from tpu.swarm.ray_train.arena_sampling import summarize_model
    records = [dict(model="m", kind="sample", verdict={"passed": True}, translated={"raw_score": 2}),
               dict(model="m", kind="sample", verdict={"passed": False}),
               dict(model="m", kind="sample", infrastructure_error="HTTP 400"),
               dict(model="m", kind="sample", format_error="truncated", finish_reason="length")]
    result = summarize_model("m", records, 10)
    assert result["samples"] == 4 and result["passed"] == 1
    assert result["infrastructure_errors"] == 1 and result["truncated"] == 1
    assert result["median_valid_speedup"] == 2


def test_native_budget_profile_preserves_baseline_and_leaves_room_for_prompt(tmp_path):
    from tpu.swarm.ray_train.commands import inference_command, inference_environment
    from tpu.swarm.ray_train.thinking_budget.install import identity
    config = Config.load(MULTI_PROFILE.with_name('rglru_three_models_v4_32_thinking.json'))
    baseline = Config.load(MULTI_PROFILE)
    assert (config.arena_max_tokens, config.arena_thinking_tokens) == (32768, 20000)
    assert baseline.arena_thinking_tokens is None
    assert config.arena_samples == baseline.arena_samples == 32
    assert config.arena_concurrency == baseline.arena_concurrency == 8
    assert len(identity()) == 64
    for rank in (1, 2, 3):
        child, original = config.for_inference_rank(rank), baseline.for_inference_rank(rank)
        assert child.inference.max_model_length >= 4388 + 32768
        assert child.inference.memory_utilization == original.inference.memory_utilization
        assert child.inference.chunk_tokens == original.inference.chunk_tokens
        assert child.cache.hf == original.cache.hf
        assert child.cache.inference_compile_seed == original.cache.inference_compile
        command = inference_command(child, tmp_path, tmp_path, tmp_path/'model', tmp_path/'run')
        assert command[1] == str(tmp_path/'tpu/thinking_budget/server.py')
        env = inference_environment(child, tmp_path, tmp_path/'run')
        assert env['SKYRL_THINKING_FORMAT'] == child.model_preset
    archive, _, _ = build(MULTI_PROFILE.with_name('rglru_three_models_v4_32_thinking.json'), tmp_path)
    with tarfile.open(archive) as bundle:
        assert 'tpu/swarm/ray_train/thinking_budget/runner.patch' in bundle.getnames()


@pytest.mark.parametrize('cap', [-1, True, 32768, 32700])
def test_native_budget_rejects_invalid_profile_cap(cap):
    config = Config.load(MULTI_PROFILE.with_name('rglru_three_models_v4_32_thinking.json')).to_dict()
    config['arena_thinking_tokens'] = cap
    with pytest.raises(ValueError, match='thinking cap'):
        Config.from_dict(config)


def test_v4_64_replicas_do_not_multiply_sample_cohorts():
    from collections import Counter
    from tpu.swarm.ray_train.arena_sampling import model_configs
    config = Config.load(MULTI_PROFILE.with_name('rglru_three_models_v4_64_thinking.json'))
    children = [config.for_inference_rank(rank) for rank in config.inference_only_ranks]
    assert Counter(c.model for c in children) == {m: 2 for m in config.served_models}
    assert len(model_configs(config)) == 3
    assert sum(c.arena_samples for c in model_configs(config)) == 96
    assert all(c.inference.tp == 4 for c in children)
    assert all(workload_resources(config, r) == {'TPU': 0} for r in (0, 7))
    assert all(workload_resources(config, r) == {'TPU': 4} for r in range(1,7))


def test_multi_model_routing_balances_each_models_replicas(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from tpu.swarm.ray_train import serving
    async def check():
        config = Config.load(MULTI_PROFILE.with_name('rglru_three_models_v4_64_thinking.json'))
        raw = config.to_dict() | {'root': str(tmp_path)}
        a,b,c = config.served_models
        names = [a,b,c,c,a,b]
        handles = [object() for _ in names]
        cls = serving.Ingress.func_or_class.__mro__[1]
        gateway = cls(raw, handles, SimpleNamespace(), ['a']*6, names)
        try:
            for i in range(8):
                assert gateway.select_engine(a) is handles[[0,4][i%2]]
                assert gateway.select_engine(c) is handles[[2,3][i%2]]
            assert gateway.select_engine(b) is handles[1]
            assert gateway.select_engine(b) is handles[5]
        finally:
            await gateway.http.aclose()
    asyncio.run(check())


def test_speed_summary_separates_forced_tokens_and_uses_cohort_wall_time():
    from tpu.swarm.ray_train.arena_sampling import summarize_model
    rows = [dict(model='m',kind='sample',generation_s=t,response={
        'usage': {'completion_tokens': 100},'choices':[{'forced_token_positions':[2,3]}]}) for t in (10,20)]
    result = summarize_model('m',rows,25,replicas=2)
    assert result['output_tokens_per_s'] == 8
    assert result['sampled_tokens_per_s'] == 196/25
    assert result['forced_tokens'] == 4
    assert result['output_tokens_per_s_per_engine'] == 4
    assert result['request_latency_p50_s'] == 15
    assert result['request_latency_p95_s'] == 19.5
