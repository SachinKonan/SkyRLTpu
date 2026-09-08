"""qwix LoRA install under abstract evaluation must match the eager install (jobs 414-417).

The eager tracing forward through MaxText's layer scan holds three whole-model
copies plus a scoped fourth, and even under jit XLA wanted 104 GiB of
temporaries for gpt-oss-120b on 8 v5p chips. The abstract path never copies
the base weights: same structure, same base arrays, zero B factors, and hence
the same forward output as the eager install.
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
                           free_base_state_after_template=False, qwix_init_mode="eager"),
    )


def _forward(model):
    inputs = model.get_model_input()
    out = model(**inputs)
    hidden = out[0] if isinstance(out, tuple) else out
    return np.asarray(jnp.asarray(hidden, jnp.float32))


def test_abstract_install_matches_eager(backend):
    cfg = types.LoraConfig(rank=8, alpha=16.0, seed=3)
    backend.config.qwix_init_mode = "eager"
    eager = backend._wrap_with_lora(cfg, seed=3)
    eager_lora = nnx.to_flat_state(nnx.state(eager, nnx.LoRAParam))
    eager_base = nnx.to_flat_state(nnx.state(eager, nnx.Param))
    eager_out = _forward(eager)

    backend.config.qwix_init_mode = "abstract"
    abstract = backend._wrap_with_lora(cfg, seed=3)
    abs_lora = nnx.to_flat_state(nnx.state(abstract, nnx.LoRAParam))
    abs_base = nnx.to_flat_state(nnx.state(abstract, nnx.Param))
    abs_out = _forward(abstract)

    # Same adapter structure, dtypes and scan-aware logical axes.
    assert [p for p, _ in eager_lora] == [p for p, _ in abs_lora]
    assert len(abs_lora) > 0
    for (path, a), (_, b) in zip(eager_lora, abs_lora):
        assert a.value.shape == b.value.shape and a.value.dtype == b.value.dtype, path
        assert a.get_metadata().get("out_sharding") == b.get_metadata().get("out_sharding"), path
        bv = np.asarray(b.value, dtype=np.float32)
        if str(path[-1]).endswith("_lora_b"):
            assert not bv.any(), path
        else:
            assert bv.any(), path
            assert np.isfinite(bv).all(), path
    # Base weights are the very same arrays as the pristine base state, not copies.
    def _base_only(flat):
        return [(tuple(p), l) for p, l in flat if not str(p[-1]).endswith(("_lora_a", "_lora_b"))]
    eager_base, abs_base = _base_only(eager_base), _base_only(abs_base)
    assert [p for p, _ in eager_base] == [p for p, _ in abs_base]
    pristine = {tuple(p): l.value for p, l in nnx.to_flat_state(backend.base_state)}
    shared = 0
    for path, leaf in abs_base:
        src = pristine[path]
        if leaf.value is src:
            shared += 1
        else:
            assert np.array_equal(np.asarray(leaf.value), np.asarray(src)), path
    assert shared == len(abs_base), f"abstract path copied {len(abs_base) - shared} base arrays"
    # B == 0 on both paths, so the forward output is the base model's on both.
    np.testing.assert_allclose(eager_out, abs_out, rtol=1e-5, atol=1e-5)


def test_abstract_install_is_default_for_maxtext():
    from skyrl.backends.tunix_backend import TunixBackendConfig

    assert TunixBackendConfig(model_source="maxtext").qwix_init_mode == "abstract"
