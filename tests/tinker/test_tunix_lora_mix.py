"""Learnable carried/fresh LoRA mix (skyrl.backends.lora_mix) on the tunix backend.

Pure-math tests need only jax; the backend tests use the tiny CPU qwen3 from
test_tunix_backend.py and exercise create_model / load_checkpoint (carry) /
forward_backward / optim_step / checkpoints / PEFT export with the mix on.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from skyrl.backends import lora_mix
from skyrl.tinker import types

from tests.tinker.test_tunix_backend import ADAM, BASE_MODEL, make_model_pass_batch, mean_loss

jax = pytest.importorskip("jax")
jnp = jax.numpy


def _patch_public_tunix_qwen3():
    """The backend's native path calls ``model(..., skip_lm_head=True)`` and
    ``model.compute_final_logits`` (a tunix fork API). Public google-tunix
    0.1.7 has neither; accept the kwarg and return logits so the CPU tests
    run on the released package. Production uses model_source="maxtext"."""
    import inspect

    from tunix.models.qwen3 import model as qwen3_model

    cls = qwen3_model.Qwen3
    if "skip_lm_head" in inspect.signature(cls.__call__).parameters:
        return
    original = cls.__call__

    def __call__(self, input_tokens, positions, cache, attention_mask, output_hidden_states=False,
                 segment_ids=None, skip_lm_head=False):
        return original(self, input_tokens, positions, cache, attention_mask,
                        output_hidden_states=output_hidden_states, segment_ids=segment_ids)

    cls.__call__ = __call__
    if not hasattr(cls, "compute_final_logits"):
        cls.compute_final_logits = lambda self, x: x.astype(jnp.float32)


_patch_public_tunix_qwen3()


# ----------------------------------------------------------------------------- pure math


def _fake_state(rank, layers=2, d_in=6, d_out=5, seed=0):
    """A qwix-shaped pytree: dict keys carry 'lora_a' / 'lora_b' like real keystrs."""
    rng = np.random.default_rng(seed)
    state = {}
    for layer in range(layers):
        state[f"layers.{layer}.q_proj.w_lora_a"] = jnp.asarray(rng.normal(size=(d_in, rank)), jnp.float32)
        state[f"layers.{layer}.q_proj.w_lora_b"] = jnp.asarray(rng.normal(size=(rank, d_out)), jnp.float32)
    return state


def _deltas(state, rank):
    """(new, old) delta matrices per layer from a stacked rank-2r state."""
    out = {}
    for key in state:
        if lora_mix.is_lora_a(key):
            a = np.asarray(state[key])
            b = np.asarray(state[key.replace("lora_a", "lora_b")])
            out[key] = (a[:, :rank] @ b[:rank], a[:, rank:] @ b[rank:])
    return out


def test_effective_state_is_convex_mix_of_deltas():
    r = 3
    state = _fake_state(2 * r)
    gamma = lora_mix.init_gamma(state, 0.3)
    assert len(gamma) == 2 and all(v.shape == (1, 1) for v in gamma.values())
    eff = lora_mix.effective_state(state, gamma, r)
    for key, (d_new, d_old) in _deltas(state, r).items():
        a = np.asarray(eff[key])
        b = np.asarray(eff[key.replace("lora_a", "lora_b")])
        np.testing.assert_allclose(a @ b, 0.3 * d_new + 0.7 * d_old, atol=1e-5)
        # A is untouched; only B is scaled
        np.testing.assert_array_equal(a, np.asarray(state[key]))


def test_split_gradients_matches_autodiff():
    r = 3
    state = _fake_state(2 * r, seed=1)
    gamma = lora_mix.init_gamma(state, 0.6)
    rng = np.random.default_rng(2)
    probe = {k: jnp.asarray(rng.normal(size=v.shape), jnp.float32) for k, v in state.items()}

    def loss(state_, gamma_):
        eff = lora_mix.effective_state(state_, gamma_, r)
        return sum(jnp.sum(eff[k] * probe[k]) for k in eff)

    g_state, g_gamma = jax.grad(loss, argnums=(0, 1))(state, gamma)
    # dL/d(effective) is the probe itself
    lora_grads, gamma_grads = lora_mix.split_gradients(probe, state, gamma, r)
    for k in state:
        got = np.asarray(lora_grads[k])
        want = np.asarray(g_state[k])
        if lora_mix.is_lora_b(k):
            np.testing.assert_allclose(got[:r], want[:r], atol=1e-5)
            assert np.all(got[r:] == 0), "old half must never receive a gradient"
        else:
            np.testing.assert_allclose(got[:, :r], want[:, :r], atol=1e-5)
            assert np.all(got[:, r:] == 0)
    for k in gamma:
        np.testing.assert_allclose(np.asarray(gamma_grads[k]), np.asarray(g_gamma[k]), atol=1e-4)


def test_split_gradients_maxtext_layout_reduces_per_layer():
    """MaxText stacks layers on axis 1: lora_a (in, L, 2r), lora_b (2r, L, out) -> gamma (1, L, 1)."""
    r, L = 2, 3
    rng = np.random.default_rng(3)
    state = {
        "decoder.layers.query.kernel_lora_a": jnp.asarray(rng.normal(size=(4, L, 2 * r)), jnp.float32),
        "decoder.layers.query.kernel_lora_b": jnp.asarray(rng.normal(size=(2 * r, L, 5)), jnp.float32),
    }
    gamma = lora_mix.init_gamma(state, 0.5)
    (k,) = gamma  # keyed by jax keystr of the lora_b leaf
    assert "kernel_lora_b" in k and gamma[k].shape == (1, L, 1)
    probe = {k2: jnp.asarray(rng.normal(size=v.shape), jnp.float32) for k2, v in state.items()}

    def loss(state_, gamma_):
        eff = lora_mix.effective_state(state_, gamma_, r)
        return sum(jnp.sum(eff[k2] * probe[k2]) for k2 in eff)

    _, g_gamma = jax.grad(loss, argnums=(0, 1))(state, gamma)
    _, gamma_grads = lora_mix.split_gradients(probe, state, gamma, r)
    assert gamma_grads[k].shape == (1, L, 1)
    np.testing.assert_allclose(np.asarray(gamma_grads[k]), np.asarray(g_gamma[k]), atol=1e-4)


def test_merge_plain_checkpoint_fills_old_half_only():
    r = 2
    template = {k: np.asarray(v) for k, v in _fake_state(2 * r, seed=4).items()}
    plain = {k: np.asarray(v) for k, v in _fake_state(r, seed=5).items()}
    merged = lora_mix.merge_plain_checkpoint(template, plain, r)
    for k in template:
        if lora_mix.is_lora_a(k):
            np.testing.assert_array_equal(merged[k][:, :r], template[k][:, :r])  # fresh A kept
            np.testing.assert_array_equal(merged[k][:, r:], plain[k])
        else:
            assert np.all(merged[k][:r] == 0), "fresh B starts at zero"
            np.testing.assert_array_equal(merged[k][r:], plain[k])
    with pytest.raises(ValueError, match="rank layout"):
        lora_mix.merge_plain_checkpoint(template, {k: v for k, v in template.items()}, r)


def test_restore_old_half_and_gamma_clamp():
    r = 2
    before = _fake_state(2 * r, seed=6)
    after = {k: v + 1.0 for k, v in before.items()}
    restored = lora_mix.restore_old_half(after, before, r)
    for k in before:
        if lora_mix.is_lora_a(k):
            np.testing.assert_array_equal(restored[k][:, r:], before[k][:, r:])
            np.testing.assert_array_equal(restored[k][:, :r], after[k][:, :r])
        else:
            np.testing.assert_array_equal(restored[k][r:], before[k][r:])
            np.testing.assert_array_equal(restored[k][:r], after[k][:r])
    mix = lora_mix.MixState(rank=r, gamma=lora_mix.init_gamma(before, 0.99), gamma_lr=0.5, gamma_init=0.99)
    lora_mix.gamma_step(mix, {k: -jnp.ones_like(v) for k, v in mix.gamma.items()})  # pushes gamma up
    assert all(float(v.max()) <= 1.0 for v in mix.gamma.values())
    assert mix.metrics()["lora_mix/gamma_mean"] > 0.99


def test_mix_settings_from_env():
    assert lora_mix.mix_settings_from_env({}) is None
    assert lora_mix.mix_settings_from_env({"TUNIX_LORA_MIX_GAMMA": "0.9"}) == (0.9, lora_mix.DEFAULT_GAMMA_LR)
    assert lora_mix.mix_settings_from_env({"TUNIX_LORA_MIX_GAMMA": "0.5", "TUNIX_LORA_MIX_GAMMA_LR": "0.1"}) == (0.5, 0.1)
    with pytest.raises(ValueError):
        lora_mix.mix_settings_from_env({"TUNIX_LORA_MIX_GAMMA": "1.5"})


# ----------------------------------------------------------------------------- backend integration

RANK, ALPHA = 8, 16.0
CFG = types.LoraConfig(rank=RANK, alpha=ALPHA, seed=7)


@pytest.fixture(scope="module")
def backend():
    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig

    return TunixBackend(BASE_MODEL, TunixBackendConfig(max_lora_rank=32))


@pytest.fixture(scope="module")
def carried_checkpoint(backend, tmp_path_factory):
    """A trained PLAIN rank-8 model and its checkpoint: the 'carried' generation."""
    backend.create_model("carry_src", CFG)
    for _ in range(2):
        backend.forward_backward(make_model_pass_batch(["carry_src"]))
        backend.optim_step("carry_src", types.OptimStepInput(adam_params=ADAM))
    path = tmp_path_factory.mktemp("carry") / "000002.tar.gz"
    backend.save_checkpoint(path, "carry_src")
    flat = backend._flat_numpy(backend.models["carry_src"].lora_state)
    assert max(np.abs(v).max() for k, v in flat.items() if lora_mix.is_lora_b(k)) > 0
    return path, flat


def _mix_model(backend, monkeypatch, name, gamma, gamma_lr="0.05"):
    monkeypatch.setenv(lora_mix.GAMMA_ENV, str(gamma))
    monkeypatch.setenv(lora_mix.GAMMA_LR_ENV, gamma_lr)
    backend.create_model(name, CFG)
    return backend.models[name]


def test_create_model_builds_rank_doubled_template(backend, monkeypatch):
    slot = _mix_model(backend, monkeypatch, "mix_shape", 0.5)
    try:
        assert slot.mix is not None and slot.mix.rank == RANK
        assert slot.lora_config.rank == RANK
        assert slot.export_lora_config.rank == 2 * RANK and slot.export_lora_config.alpha == 2 * ALPHA
        flat = backend._flat_numpy(slot.lora_state)
        b_keys = [k for k in flat if lora_mix.is_lora_b(k)]
        assert len(slot.mix.gamma) == len(b_keys) == 14  # 2 layers x 7 projections
        for k in flat:
            assert flat[k].shape[lora_mix.rank_axis(k)] == 2 * RANK
        assert np.allclose(slot.mix.gamma_values(), 0.5)
    finally:
        backend.delete_model("mix_shape")


def test_rank_limit_covers_doubled_rank(backend, monkeypatch):
    monkeypatch.setenv(lora_mix.GAMMA_ENV, "0.5")
    with pytest.raises(ValueError, match="max_lora_rank"):
        backend.create_model("mix_toobig", types.LoraConfig(rank=32, alpha=32.0, seed=0))
    assert not backend.has_model("mix_toobig")


def test_carry_load_and_forward_parity(backend, monkeypatch, carried_checkpoint):
    """Loading a plain checkpoint fills the old half; at gamma the forward equals a
    plain adapter whose B is scaled by (1 - gamma)."""
    path, carried = carried_checkpoint
    gamma = 0.25
    slot = _mix_model(backend, monkeypatch, "mix_carry", gamma)
    try:
        backend.load_checkpoint(path, "mix_carry")
        assert slot.mix.old_loaded
        flat = backend._flat_numpy(slot.lora_state)
        for k, v in flat.items():
            if lora_mix.is_lora_b(k):
                assert np.all(v[:RANK] == 0)
                np.testing.assert_allclose(v[RANK:], carried[k], atol=1e-6)
            else:
                np.testing.assert_allclose(v[..., RANK:], carried[k], atol=1e-6)

        # reference: plain model with B *= (1 - gamma)
        monkeypatch.delenv(lora_mix.GAMMA_ENV)
        backend.create_model("mix_ref", CFG)
        ref = backend.models["mix_ref"]
        scaled = {k: (v * (1 - gamma) if lora_mix.is_lora_b(k) else v) for k, v in carried.items()}
        ref.lora_state = backend._state_from_flat(ref.lora_state, scaled, None)
        out = backend.forward(make_model_pass_batch(["mix_carry", "mix_ref"]))
        assert abs(mean_loss(out, "0") - mean_loss(out, "1")) < 1e-4
    finally:
        for name in ("mix_carry", "mix_ref"):
            if backend.has_model(name):
                backend.delete_model(name)


def test_train_step_moves_gamma_and_new_half_only(backend, monkeypatch, carried_checkpoint):
    path, carried = carried_checkpoint
    slot = _mix_model(backend, monkeypatch, "mix_train", 0.5, gamma_lr="0.05")
    try:
        backend.load_checkpoint(path, "mix_train")
        before = backend._flat_numpy(slot.lora_state)
        gamma_before = slot.mix.gamma_values().copy()
        backend.forward_backward(make_model_pass_batch(["mix_train"]))
        out = backend.optim_step("mix_train", types.OptimStepInput(adam_params=ADAM))
        assert "lora_mix/gamma_mean" in out.metrics
        after = backend._flat_numpy(slot.lora_state)
        gamma_after = slot.mix.gamma_values()
        assert np.abs(gamma_after - gamma_before).max() > 1e-4, "gamma must learn"
        assert np.all((gamma_after >= 0) & (gamma_after <= 1))
        moved_new_b = 0
        for k in before:
            if lora_mix.is_lora_b(k):
                np.testing.assert_array_equal(after[k][RANK:], before[k][RANK:])  # frozen old half
                moved_new_b += float(np.abs(after[k][:RANK] - before[k][:RANK]).max() > 0)
            else:
                np.testing.assert_array_equal(after[k][:, RANK:], before[k][:, RANK:])
        assert moved_new_b > 0, "the fresh half must train"
    finally:
        backend.delete_model("mix_train")


def test_mix_checkpoint_round_trip(backend, monkeypatch, carried_checkpoint):
    path, _ = carried_checkpoint
    slot = _mix_model(backend, monkeypatch, "mix_ckpt", 0.7)
    try:
        backend.load_checkpoint(path, "mix_ckpt")
        backend.forward_backward(make_model_pass_batch(["mix_ckpt"]))
        backend.optim_step("mix_ckpt", types.OptimStepInput(adam_params=ADAM))
        state_before = backend._flat_numpy(slot.lora_state)
        gamma_before = slot.mix.gamma_values().copy()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mix.tar.gz"
            backend.save_checkpoint(ckpt, "mix_ckpt")
            backend.forward_backward(make_model_pass_batch(["mix_ckpt"]))
            backend.optim_step("mix_ckpt", types.OptimStepInput(adam_params=ADAM))
            assert np.abs(slot.mix.gamma_values() - gamma_before).max() > 0
            backend.load_checkpoint(ckpt, "mix_ckpt")
            payload = backend._read_checkpoint_archive(ckpt)
            assert payload["lora_mix"]["rank"] == RANK and payload["lora_mix"]["old_loaded"]
        state_after = backend._flat_numpy(slot.lora_state)
        assert all(np.array_equal(state_before[k], state_after[k]) for k in state_before)
        np.testing.assert_allclose(slot.mix.gamma_values(), gamma_before)
        # a plain model must refuse a mix checkpoint
        monkeypatch.delenv(lora_mix.GAMMA_ENV)
        backend.create_model("mix_plain", CFG)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mix2.tar.gz"
            backend.save_checkpoint(ckpt, "mix_ckpt")
            with pytest.raises(ValueError, match="LoRA mix"):
                backend.load_checkpoint(ckpt, "mix_plain")
    finally:
        for name in ("mix_ckpt", "mix_plain"):
            if backend.has_model(name):
                backend.delete_model(name)


def test_peft_export_is_effective_rank_2r_adapter(backend, monkeypatch, carried_checkpoint):
    from safetensors.numpy import load_file

    path, _ = carried_checkpoint
    gamma = 0.4
    slot = _mix_model(backend, monkeypatch, "mix_peft", gamma)
    try:
        backend.load_checkpoint(path, "mix_peft")
        backend.forward_backward(make_model_pass_batch(["mix_peft"]))
        backend.optim_step("mix_peft", types.OptimStepInput(adam_params=ADAM))
        with tempfile.TemporaryDirectory() as tmp:
            sampler = Path(tmp) / "s.tar.gz"
            backend.save_sampler_checkpoint(sampler, "mix_peft", persist=True)
            meta = backend._read_checkpoint_archive(sampler)
            assert meta["lora_config"]["rank"] == 2 * RANK
            out = Path(tmp) / "peft"
            out.mkdir()
            backend._export_peft_adapter(slot, out)
            tensors = load_file(out / "adapter_model.safetensors")
            cfg = json.loads((out / "adapter_config.json").read_text())
        assert cfg["r"] == 2 * RANK and cfg["lora_alpha"] == 2 * ALPHA
        q_a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]  # (2r, in)
        q_b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]  # (out, 2r)
        a_key = "['layers'][0]['attn']['q_proj']['w_lora_a'].value"
        b_key = "['layers'][0]['attn']['q_proj']['w_lora_b'].value"
        raw = backend._flat_numpy(slot.lora_state)
        g = np.asarray(slot.mix.gamma[b_key])  # (1, *middle, 1): one gamma per adapter slice
        b_raw = raw[b_key]
        b_mixed = np.concatenate([g * b_raw[:RANK], (1 - g) * b_raw[RANK:]], axis=0)
        want = raw[a_key] @ b_mixed.reshape(2 * RANK, -1)
        np.testing.assert_allclose(q_b @ q_a, want.T, atol=1e-5)
        # and the exported B really is the gamma-mixed one, not the raw halves
        eff = backend._flat_numpy(backend._pass_lora_state(slot))
        np.testing.assert_allclose(eff[b_key], b_mixed, atol=1e-6)
        assert np.abs(eff[b_key] - b_raw).max() > 0
        # exported scale alpha'/r' equals the client's alpha/r
        assert cfg["lora_alpha"] / cfg["r"] == ALPHA / RANK
    finally:
        backend.delete_model("mix_peft")
