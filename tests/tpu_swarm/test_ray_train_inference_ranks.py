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


def test_population_serving_does_not_require_a_backward_output_mode():
    config = Config.load(
        "tpu/swarm/ray_train/profiles/qwen35-v432-native-multi-lora-inference-20260919.json")
    # Fleet profiles run one adapter; retain population-mode coverage here.
    config = replace(config, adapter_count=2,
                     inference=replace(config.inference, max_loras=2))
    config.validate()
    assert config.inference_only and config.trainer.hosts == 0
    assert config.inference_only_ranks == [0, 1, 2, 3]
    assert len(config.engine_slots(["host0", "host1", "host2", "host3"])) == 4
    assert config.inference.max_loras == config.adapter_count == 2
    assert not config.sequential_probe and not config.stacked_probe
    # vLLM's default can enable prefix caching; the CLI must explicitly
    # turn it off for this farm rather than omit the positive flag.
    from tpu.swarm.ray_train.commands import inference_command
    command = inference_command(config, Path('/root'), Path('/source'),
                                Path('/snapshot'), Path('/run'))
    assert not config.inference.prefix_caching
    assert '--enable-prefix-caching' not in command
    assert '--no-enable-prefix-caching' in command
    # The same defaults must still fail closed if a trainer is introduced.
    training = replace(config, inference_only=False, inference_only_ranks=None,
                       trainer=replace(config.trainer, hosts=1))
    with pytest.raises(ValueError, match="full backward logprobs"):
        training.validate()
