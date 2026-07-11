import pytest
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from skyrl_agent.integrations.tinker.tinker_train import compute_kl_sample_train


def _datum(mask_key: str) -> types.Datum:
    inputs = {
        "logprobs": TensorData.from_torch(torch.tensor([0.0, -2.0, -4.0])),
        mask_key: TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
    }
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs=inputs,
    )


@pytest.mark.parametrize("mask_key", ["weights", "mask"])
def test_compute_kl_accepts_current_and_legacy_action_mask_keys(mask_key):
    metrics = compute_kl_sample_train(
        [_datum(mask_key)],
        [torch.tensor([0.0, -1.5, -3.0])],
    )

    assert metrics["optim/kl_sample_train_v1"] == pytest.approx(-0.75)
    assert metrics["optim/kl_sample_train_v2"] == pytest.approx(0.3125)
    assert metrics["optim/entropy"] == pytest.approx(3.0)
