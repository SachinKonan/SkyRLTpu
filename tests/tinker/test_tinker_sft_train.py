import json
import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).parents[2] / "skyrl-agent/skyrl_agent/integrations/tinker/tinker_sft_train.py"
)
_SPEC = importlib.util.spec_from_file_location("tinker_sft_train", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SFTExample = _MODULE.SFTExample
crossed_example_checkpoints = _MODULE.crossed_example_checkpoints
find_resume_checkpoint = _MODULE.find_resume_checkpoint
make_tinker_datums = _MODULE.make_tinker_datums
parse_sft_row = _MODULE.parse_sft_row


def _example(example_id: str, response_len: int, supervised: int) -> SFTExample:
    return SFTExample(
        example_id=example_id,
        prompt_token_ids=[10, 11],
        response_ids=list(range(20, 20 + response_len)),
        loss_mask=[1.0] * supervised + [0.0] * (response_len - supervised),
        supervised_tokens=supervised,
        sequence_length=2 + response_len,
    )


def test_parse_sft_row_rejects_overlong_sequence():
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        parse_sft_row(
            {"id": "long", "prompt_token_ids": [1, 2], "response_ids": [3, 4], "loss_mask": [1, 1]},
            max_sequence_length=3,
        )


def test_make_tinker_datums_is_global_batch_token_mean():
    datums, stats = make_tinker_datums([_example("a", 3, 1), _example("b", 4, 3)])
    weights = [datum.loss_fn_inputs["weights"].to_torch() for datum in datums]

    assert stats["supervised_tokens"] == 4
    assert stats["target_weight_sum"] == pytest.approx(2.0)
    assert weights[0].sum().item() == pytest.approx(0.5)
    assert weights[1].sum().item() == pytest.approx(1.5)


def test_crossed_example_checkpoints_handles_resume_and_crossing():
    assert crossed_example_checkpoints(1998, 2000, 2000) == [2000]
    assert crossed_example_checkpoints(2000, 2002, 2000) == []
    assert crossed_example_checkpoints(1999, 4001, 2000) == [2000, 4000]
    assert crossed_example_checkpoints(0, 10, 0) == []


def test_find_resume_checkpoint_uses_examples_then_step(tmp_path):
    records = [
        {"sft_step": 10, "examples_seen": 20, "supervised_tokens_seen": 100, "state_path": "state-a"},
        {"sft_step": 9, "examples_seen": 22, "supervised_tokens_seen": 110, "state_path": "state-b"},
    ]
    (tmp_path / "checkpoints.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    assert find_resume_checkpoint(str(tmp_path)) == (9, 22, 110, "state-b")
