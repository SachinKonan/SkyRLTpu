"""qwix LoRA install under nnx.jit must match the eager install (job 416).

The eager tracing forward through MaxText's layer scan holds three whole-model
copies plus a scoped fourth; the jitted path emits one copy. Same seed must
give the same adapters and the same forward output.
"""
import numpy as np
import pytest

pytest.importorskip("maxtext")

from flax import nnx
import jax
import jax.numpy as jnp

from skyrl.tinker import types

BASE_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def backend():
    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig

    return TunixBackend(
        BASE_MODEL,
        TunixBackendConfig(model_source="maxtext", maxtext_max_target_length=64,
                           free_base_state_after_template=False, qwix_jit_init=False),
    )


def _forward(model):
    inputs = model.get_model_input()
    out = model(**inputs)
    hidden = out[0] if isinstance(out, tuple) else out
    return np.asarray(jnp.asarray(hidden, jnp.float32))


def test_jit_install_matches_eager(backend):
    cfg = types.LoraConfig(rank=8, alpha=16.0, seed=3)
    backend.config.qwix_jit_init = False
    eager = backend._wrap_with_lora(cfg, seed=3)
    eager_lora = nnx.to_flat_state(nnx.state(eager, nnx.LoRAParam))
    eager_out = _forward(eager)

    backend.config.qwix_jit_init = True
    jitted = backend._wrap_with_lora(cfg, seed=3)
    jit_lora = nnx.to_flat_state(nnx.state(jitted, nnx.LoRAParam))
    jit_out = _forward(jitted)

    assert [p for p, _ in eager_lora] == [p for p, _ in jit_lora]
    assert len(jit_lora) > 0
    for (path, a), (_, b) in zip(eager_lora, jit_lora):
        av, bv = np.asarray(a.value), np.asarray(b.value)
        assert av.shape == bv.shape and av.dtype == bv.dtype, path
        np.testing.assert_allclose(av, bv, rtol=0, atol=0, err_msg=str(path))
        # Scan-aware logical axes survive the jitted path.
        assert a.get_metadata().get("out_sharding") == b.get_metadata().get("out_sharding"), path
        assert isinstance(b.value.sharding, jax.sharding.Sharding)
    np.testing.assert_allclose(eager_out, jit_out, rtol=1e-5, atol=1e-5)


def test_jit_install_is_default_for_maxtext():
    from skyrl.backends.tunix_backend import TunixBackendConfig

    assert TunixBackendConfig(model_source="maxtext").qwix_jit_init is True
