"""Engine LoRA slot count stays an explicit, validated CLI knob (legacy: 8 slots)."""
from dataclasses import replace
from pathlib import Path

import pytest

from tpu.swarm.ray_train.config import Config

PROFILE = "tpu/swarm/ray_train/profiles/qwen_v5p_32.json"


def test_lora_slot_ablation_changes_only_the_requested_cli_argument():
    from tpu.swarm.ray_train.commands import inference_command
    config = Config.load(PROFILE)
    paths = [Path("/root"), Path("/source"), Path("/snapshot"), Path("/run")]
    original = inference_command(config, *paths)
    changed = replace(config, inference=replace(config.inference, max_loras=2))
    changed.validate()
    command = inference_command(changed, *paths)
    index = original.index("--max-loras") + 1
    assert original[index] == "8" and command[index] == "2"
    assert original[:index] == command[:index] and original[index+1:] == command[index+1:]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_lora_slot_count_must_be_a_positive_integer(value):
    config = Config.load(PROFILE)
    with pytest.raises(ValueError, match="max_loras"):
        replace(config, inference=replace(config.inference, max_loras=value)).validate()
