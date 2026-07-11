import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[2] / "skyrl-agent/skyrl_agent/integrations/tinker/tinker_rl_utils.py"
_SPEC = importlib.util.spec_from_file_location("tinker_rl_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_rl_training_datums = _MODULE.build_rl_training_datums
compute_advantages_grpo = _MODULE.compute_advantages_grpo


def test_grpo_uses_prompt_group_sample_std():
    actual = compute_advantages_grpo([0.0, 1.0, 5.0, 5.0], group_size=2, normalize_by_std=True)
    np.testing.assert_allclose(actual[:2], [-1 / np.sqrt(2), 1 / np.sqrt(2)], rtol=1e-5)
    assert actual[2:] == [0.0, 0.0]


def test_grpo_rejects_partial_prompt_group():
    with pytest.raises(ValueError, match="not divisible"):
        compute_advantages_grpo([0.0, 1.0, 2.0], group_size=2)


def test_rl_datums_mask_prompts_and_form_global_token_mean():
    datums, stats = build_rl_training_datums(
        prompt_token_ids=[[10, 11], [20]],
        response_ids=[[12, 13], [21, 22, 23]],
        loss_masks=[[1.0, 0.0], [1.0, 1.0, 0.0]],
        sampled_logprobs=[[-0.1, -0.2], [-0.3, -0.4, -0.5]],
        step_advantages=[1.0, -1.0],
    )

    assert stats == {"num_episodes": 2.0, "action_tokens": 3.0, "target_weight_sum": 2.0}
    weights = [datum.loss_fn_inputs["weights"].to_torch() for datum in datums]
    assert sum(tensor.sum().item() for tensor in weights) == pytest.approx(2.0)
    assert weights[0][0].item() == 0.0
    assert datums[0].loss_fn_inputs["rollout_logprobs"].to_torch().tolist() == pytest.approx([0.0, -0.1, -0.2])


def test_rl_datums_reject_all_masked_batch():
    with pytest.raises(ValueError, match="zero action-token weight"):
        build_rl_training_datums([[1]], [[2]], [[0.0]], [[-0.1]], [1.0])
