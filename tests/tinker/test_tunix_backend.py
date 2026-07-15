"""Unit tests for the tunix backend (CPU, tiny qwen3).

Integration coverage (real server + real tinker SDK) comes from the
backend-parametrized ``api_server`` fixture in test_api.py.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from skyrl.tinker import types

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture(scope="module")
def backend():
    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig

    return TunixBackend(BASE_MODEL, TunixBackendConfig())


@pytest.fixture()
def model_pair(backend):
    """Two models on the same rank template; deleted after the test."""
    names = ["unit_a", "unit_b"]
    for i, name in enumerate(names):
        backend.create_model(name, types.LoraConfig(rank=8, alpha=16.0, seed=42 + i))
    yield names
    for name in names:
        if backend.has_model(name):
            backend.delete_model(name)


def make_model_pass_batch(model_ids, loss_fn="cross_entropy", tokens=TOKENS):
    n = len(model_ids)
    return types.PreparedModelPassBatch(
        all_model_inputs=[types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)]) for _ in range(n)],
        all_targets=[[t + 1 for t in tokens]] * n,
        all_token_weights=[[1.0] * len(tokens)] * n,
        all_sampling_logprobs=[[0.0] * len(tokens)] * n,
        all_advantages=[[1.0] * len(tokens)] * n,
        all_values=[[]] * n,
        all_returns=[[]] * n,
        all_model_ids=list(model_ids),
        all_loss_fns=[loss_fn] * n,
        all_loss_fn_configs=[None] * n,
        request_batch_slices=[(str(i), mid, i, i + 1) for i, mid in enumerate(model_ids)],
    )


def make_sample_batch(model_id, checkpoint_id, checkpoint_path, seeds, tokens=TOKENS, **param_kw):
    param_kw.setdefault("temperature", 0.0)
    param_kw.setdefault("max_tokens", 10)
    n = len(seeds)
    params = [types.SamplingParams(seed=s, **param_kw) for s in seeds]
    return types.PreparedSampleBatch(
        all_model_inputs=[types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)])] * n,
        all_sampling_params=params,
        all_model_ids=[model_id] * n,
        all_checkpoint_ids=[checkpoint_id] * n,
        all_checkpoint_paths=[checkpoint_path] * n,
        all_session_ids=[None] * n,
        needs_prompt_logprobs=False,
        request_batch_slices=[("r0", model_id, 0, n, False)],
    )


ADAM = types.AdamParams(learning_rate=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)


def mean_loss(output, request_id="0"):
    return float(np.mean(output[request_id].loss_fn_outputs[0]["elementwise_loss"]["data"]))


def test_forward_backward_accumulates_per_model(backend, model_pair):
    a, b = model_pair
    out = backend.forward_backward(make_model_pass_batch([a, b, a]))
    assert set(out.keys()) == {"0", "1", "2"}
    assert backend.models[a].accum_count == 2
    assert backend.models[b].accum_count == 1
    # Fresh adapters have zero delta: losses identical across models
    assert mean_loss(out, "0") == pytest.approx(mean_loss(out, "1"), abs=1e-5)
    # Output tensors cover each token
    lens = [len(o.loss_fn_outputs[0]["logprobs"]["data"]) for o in out.values()]
    assert lens == [len(TOKENS)] * 3


def test_optim_step_isolates_models(backend, model_pair):
    a, b = model_pair
    out0 = backend.forward_backward(make_model_pass_batch([a, b]))
    res = backend.optim_step(a, types.OptimStepInput(adam_params=ADAM))
    assert res.metrics["skyrl.ai/grad_norm"] > 0
    assert backend.models[a].accum_count == 0
    assert backend.models[b].accum_count == 1  # b untouched

    out1 = backend.forward_backward(make_model_pass_batch([a, b]))
    assert mean_loss(out1, "0") < mean_loss(out0, "0"), "training model a must reduce its loss"
    assert mean_loss(out1, "1") == pytest.approx(mean_loss(out0, "1"), abs=1e-5), "model b must be unaffected"


def test_forward_does_not_accumulate(backend, model_pair):
    a, _ = model_pair
    backend.forward(make_model_pass_batch([a]))
    assert backend.models[a].accum_count == 0


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo"])
def test_loss_fns_run(backend, model_pair, loss_fn):
    a, _ = model_pair
    out = backend.forward(make_model_pass_batch([a], loss_fn=loss_fn))
    data = out["0"].loss_fn_outputs[0]["elementwise_loss"]["data"]
    assert len(data) == len(TOKENS)
    assert np.isfinite(data).all()


def test_micro_batching_matches_full_batch(backend, model_pair):
    a, _ = model_pair
    out_full = backend.forward(make_model_pass_batch([a] * 4))
    original = backend.config.train_micro_batch_size
    try:
        backend.config.train_micro_batch_size = 2
        out_micro = backend.forward(make_model_pass_batch([a] * 4))
    finally:
        backend.config.train_micro_batch_size = original
    for rid in ("0", "3"):
        np.testing.assert_allclose(
            out_full[rid].loss_fn_outputs[0]["logprobs"]["data"],
            out_micro[rid].loss_fn_outputs[0]["logprobs"]["data"],
            atol=1e-5,
        )


def test_checkpoint_round_trip(backend, model_pair):
    a, _ = model_pair
    backend.forward_backward(make_model_pass_batch([a]))
    backend.optim_step(a, types.OptimStepInput(adam_params=ADAM))
    state_before = backend._flat_numpy(backend.models[a].lora_state)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.tar.gz"
        backend.save_checkpoint(ckpt, a)

        backend.forward_backward(make_model_pass_batch([a]))
        backend.optim_step(a, types.OptimStepInput(adam_params=ADAM))
        state_moved = backend._flat_numpy(backend.models[a].lora_state)
        assert max(np.abs(state_before[k] - state_moved[k]).max() for k in state_before) > 0

        backend.load_checkpoint(ckpt, a)
        state_after = backend._flat_numpy(backend.models[a].lora_state)
        assert all(np.array_equal(state_before[k], state_after[k]) for k in state_before)


def test_checkpoint_rank_mismatch_rejected(backend, model_pair):
    a, _ = model_pair
    backend.create_model("unit_rank4", types.LoraConfig(rank=4, alpha=8.0, seed=0))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt.tar.gz"
            backend.save_checkpoint(ckpt, "unit_rank4")
            with pytest.raises(ValueError, match="[Rr]ank"):
                backend.load_checkpoint(ckpt, a)
    finally:
        backend.delete_model("unit_rank4")


def test_sample_greedy_and_stops(backend):
    out = backend.sample(make_sample_batch("", "", "", seeds=[42, 43]))
    seqs = out["r0"].sequences
    assert [len(s.tokens) for s in seqs] == [10, 10]
    assert all(s.stop_reason == "length" for s in seqs)
    assert seqs[0].tokens == seqs[1].tokens, "greedy must ignore seed"
    assert len(seqs[0].logprobs) == 10

    stop_tok = seqs[0].tokens[4]
    out_stop = backend.sample(make_sample_batch("", "", "", seeds=[42], max_tokens=50, stop_tokens=[stop_tok]))
    st = out_stop["r0"].sequences[0]
    assert st.stop_reason == "stop"
    assert st.tokens[-1] == stop_tok
    assert len(st.tokens) == 5


def test_sample_seed_semantics(backend):
    batch = lambda seeds: make_sample_batch("", "", "", seeds=seeds, temperature=1.0)
    toks_a = [s.tokens for s in backend.sample(batch([42, 43, 44]))["r0"].sequences]
    toks_b = [s.tokens for s in backend.sample(batch([42, 43, 44]))["r0"].sequences]
    assert toks_a == toks_b, "same seeds must reproduce"
    assert len({tuple(t) for t in toks_a}) > 1, "distinct seeds must diversify"

    k1 = [s.tokens for s in backend.sample(make_sample_batch("", "", "", seeds=[1, 2], temperature=1.0, top_k=1))["r0"].sequences]
    assert k1[0] == k1[1], "top_k=1 must be deterministic"


def test_sampler_checkpoint_memory_and_disk(backend, model_pair):
    a, _ = model_pair
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "s1.tar.gz"
        backend.save_sampler_checkpoint(ckpt, a, persist=True)
        out_mem = backend.sample(make_sample_batch(a, "s1", str(ckpt), seeds=[42]))
        backend.models[a].sampler_lora_states.clear()
        out_disk = backend.sample(make_sample_batch(a, "s1", str(ckpt), seeds=[42]))
        assert out_mem["r0"].sequences[0].tokens == out_disk["r0"].sequences[0].tokens


def test_ephemeral_sampler_checkpoint_marker(backend, model_pair):
    a, _ = model_pair
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "eph1.tar.gz"
        backend.save_sampler_checkpoint(ckpt, a, persist=False)
        # In-memory snapshot serves sampling
        out = backend.sample(make_sample_batch(a, "eph1", str(ckpt), seeds=[42]))
        assert len(out["r0"].sequences[0].tokens) == 10
        # But the disk artifact is only a marker: once memory is gone, loading fails clearly
        backend.models[a].sampler_lora_states.clear()
        with pytest.raises(ValueError, match="ephemeral"):
            backend.sample(make_sample_batch(a, "eph1", str(ckpt), seeds=[42]))


def test_prompt_logprobs(backend):
    pb = make_sample_batch("", "", "", seeds=[42])
    pb = pb.model_copy(update={"needs_prompt_logprobs": True, "request_batch_slices": [("r0", "", 0, 1, True)]})
    plp = backend.sample(pb)["r0"].prompt_logprobs
    assert plp is not None and len(plp) == len(TOKENS)
    assert plp[0] == 0.0
    assert all(np.isfinite(plp[1:]))


def test_peft_export_layout(backend, model_pair):
    a, _ = model_pair
    from safetensors.numpy import load_file

    with tempfile.TemporaryDirectory() as tmp:
        backend._export_peft_adapter(backend.models[a], Path(tmp))
        tensors = load_file(Path(tmp) / "adapter_model.safetensors")
        cfg = json.loads((Path(tmp) / "adapter_config.json").read_text())

    a_keys = sorted(k for k in tensors if k.endswith("lora_A.weight"))
    b_keys = sorted(k for k in tensors if k.endswith("lora_B.weight"))
    assert len(a_keys) == len(b_keys) > 0
    # 2 layers x 7 projections for the tiny model
    assert len(a_keys) == 14
    assert a_keys[0].startswith("base_model.model.model.layers.")
    assert cfg["r"] == 8 and cfg["peft_type"] == "LORA"
    assert set(cfg["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for k in a_keys:
        assert tensors[k].shape[0] == 8, f"lora_A must be (r, in): {k} {tensors[k].shape}"
    for k in b_keys:
        assert tensors[k].shape[1] == 8, f"lora_B must be (out, r): {k} {tensors[k].shape}"
    # A@B delta must match the qwix delta for a mapped module (transpose conventions)
    q_a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    q_b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    import jax

    lora_map = {jax.tree_util.keystr(p): np.asarray(v, dtype=np.float32) for p, v in
                jax.tree.flatten_with_path(backend.models[a].lora_state)[0]}
    qwix_a = lora_map["['layers'][0]['attn']['q_proj']['w_lora_a'].value"]
    qwix_b = lora_map["['layers'][0]['attn']['q_proj']['w_lora_b'].value"].reshape(8, -1)
    np.testing.assert_allclose(q_b @ q_a, (qwix_a @ qwix_b).T, atol=1e-5)


def test_create_model_validation(backend):
    with pytest.raises(ValueError, match="rank"):
        backend.create_model("unit_bad", types.LoraConfig(rank=64, alpha=16.0, seed=0))
    with pytest.raises(ValueError, match="model_role"):
        backend.create_model("unit_bad", types.LoraConfig(rank=8, alpha=16.0, seed=0), model_role="critic")
    with pytest.raises(ValueError, match="train_unembed"):
        backend.create_model("unit_bad", types.LoraConfig(rank=8, alpha=16.0, seed=0, train_unembed=True))
    assert not backend.has_model("unit_bad")


def test_parity_with_jax_backend(backend, model_pair):
    """Fresh LoRA (zero delta) => logprobs/losses must match the JaxBackend on identical inputs."""
    pytest.importorskip("peft")
    from skyrl.backends.jax import JaxBackendImpl, JaxBackendConfig

    jax_backend = JaxBackendImpl(
        BASE_MODEL,
        JaxBackendConfig(max_lora_adapters=4, max_lora_rank=8, loss_chunk_size=0),
        process_id=0,
    )
    jax_backend.create_model("parity", types.LoraConfig(rank=8, alpha=16.0, seed=42))
    a, _ = model_pair

    # The two backends agree up to compute-dtype drift: the tx backend runs with
    # the checkpoint's bfloat16 weights while the tunix backend loads float32.
    # A semantic bug (e.g. token-shift misalignment) shows up as O(1) mismatches,
    # not the ~0.1% relative drift allowed here.
    for loss_fn in ("cross_entropy", "ppo"):
        batch_tunix = make_model_pass_batch([a], loss_fn=loss_fn)
        batch_jax = make_model_pass_batch(["parity"], loss_fn=loss_fn)
        out_tunix = backend.forward(batch_tunix)["0"].loss_fn_outputs[0]
        out_jax = jax_backend.forward(batch_jax)["0"].loss_fn_outputs[0]
        np.testing.assert_allclose(
            out_tunix["logprobs"]["data"], out_jax["logprobs"]["data"], rtol=5e-3, atol=2e-2,
            err_msg=f"logprob mismatch for {loss_fn}",
        )
        np.testing.assert_allclose(
            out_tunix["elementwise_loss"]["data"], out_jax["elementwise_loss"]["data"], rtol=5e-3, atol=2e-2,
            err_msg=f"loss mismatch for {loss_fn}",
        )
