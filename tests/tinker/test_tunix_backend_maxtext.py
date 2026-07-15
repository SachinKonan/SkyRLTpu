"""MaxText-path tests for the tunix backend (CPU, Qwen3-0.6B).

Requires maxtext installed (skipped otherwise). Uses the smallest
MaxText-supported qwen3; the tiny test checkpoint used elsewhere has no
MaxText config.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maxtext")

from skyrl.tinker import types
from tests.tinker.test_tunix_backend import ADAM, make_model_pass_batch, make_sample_batch, mean_loss

BASE_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def mt_backend():
    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig

    backend = TunixBackend(
        BASE_MODEL,
        TunixBackendConfig(
            model_source="maxtext",
            maxtext_max_target_length=64,
        ),
    )
    backend.create_model("mt_model", types.LoraConfig(rank=8, alpha=16.0, seed=42))
    return backend


def test_maxtext_lora_training_lifecycle(mt_backend):
    out0 = mt_backend.forward_backward(make_model_pass_batch(["mt_model"]))
    assert mt_backend.models["mt_model"].accum_count == 1
    logprobs = out0["0"].loss_fn_outputs[0]["logprobs"]["data"]
    assert len(logprobs) == 8 and np.isfinite(logprobs).all()

    res = mt_backend.optim_step("mt_model", types.OptimStepInput(adam_params=ADAM))
    assert res.metrics["skyrl.ai/grad_norm"] > 0
    assert np.isfinite(res.metrics["skyrl.ai/grad_norm"])

    out1 = mt_backend.forward_backward(make_model_pass_batch(["mt_model"]))
    assert mean_loss(out1, "0") < mean_loss(out0, "0"), "loss must decrease after optim_step"


def test_maxtext_sampling_and_sampler_checkpoint(mt_backend):
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "mt1.tar.gz"
        mt_backend.save_sampler_checkpoint(ckpt, "mt_model", persist=True)

        out = mt_backend.sample(make_sample_batch("mt_model", "mt1", str(ckpt), seeds=[42, 43], max_tokens=5))
        seqs = out["r0"].sequences
        assert [len(s.tokens) for s in seqs] == [5, 5]
        assert seqs[0].tokens == seqs[1].tokens, "greedy must ignore seed"
        assert all(np.isfinite(s.logprobs).all() for s in seqs)

        # base-model sampling should differ from the (trained) adapter's
        out_base = mt_backend.sample(make_sample_batch("", "", "", seeds=[42], max_tokens=5))
        assert len(out_base["r0"].sequences[0].tokens) == 5


def test_maxtext_peft_export_layout(mt_backend):
    from safetensors.numpy import load_file

    with tempfile.TemporaryDirectory() as tmp:
        mt_backend._export_peft_adapter(mt_backend.models["mt_model"], Path(tmp))
        tensors = load_file(Path(tmp) / "adapter_model.safetensors")

    a_keys = sorted(k for k in tensors if k.endswith("lora_A.weight"))
    b_keys = sorted(k for k in tensors if k.endswith("lora_B.weight"))
    # Qwen3-0.6B: 28 layers x 7 projections
    assert len(a_keys) == len(b_keys) == 28 * 7
    assert a_keys[0].startswith("base_model.model.model.layers.")
    modules = {k.split(".")[-3] for k in a_keys}
    assert modules == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for k in a_keys:
        assert tensors[k].shape[0] == 8, f"lora_A must be (r, in): {k} {tensors[k].shape}"
    for k in b_keys:
        assert tensors[k].shape[1] == 8, f"lora_B must be (out, r): {k} {tensors[k].shape}"
    # q_proj lora_B out dim = num_heads * head_dim = 16 * 128
    q_b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert q_b.shape[0] == 2048


@pytest.mark.slow
def test_maxtext_parity_with_native(mt_backend):
    """Same HF checkpoint through MaxText and native tunix must agree on logprobs.

    Catches wrong weight loading, acausal attention (the adapter ignores the
    attention_mask input and must mask internally), and position handling.
    """
    from skyrl.backends.tunix_backend import TunixBackend, TunixBackendConfig

    native = TunixBackend(BASE_MODEL, TunixBackendConfig())
    native.create_model("native_model", types.LoraConfig(rank=8, alpha=16.0, seed=7))

    batch_mt = make_model_pass_batch(["mt_parity"], tokens=[151644, 872, 198, 9707, 11, 1246, 525, 498])
    batch_nat = make_model_pass_batch(["native_model"], tokens=[151644, 872, 198, 9707, 11, 1246, 525, 498])

    mt_backend.create_model("mt_parity", types.LoraConfig(rank=8, alpha=16.0, seed=7))
    out_mt = mt_backend.forward(batch_mt)["0"].loss_fn_outputs[0]["logprobs"]["data"]
    out_nat = native.forward(batch_nat)["0"].loss_fn_outputs[0]["logprobs"]["data"]

    np.testing.assert_allclose(out_mt, out_nat, rtol=5e-3, atol=2e-2)
