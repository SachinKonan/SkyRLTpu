from __future__ import annotations

import numpy as np
import torch
from tinker import types
from tinker.types.tensor_data import TensorData


def compute_advantages_grpo(
    rewards: list[float],
    group_size: int | None = None,
    normalize_by_std: bool = True,
) -> list[float]:
    rewards_array = np.asarray(rewards, dtype=np.float64)
    if group_size is None:
        group_size = len(rewards_array)
    if group_size <= 0 or len(rewards_array) % group_size:
        raise ValueError(f"Reward count {len(rewards_array)} is not divisible by group_size={group_size}")

    advantages: list[float] = []
    for start in range(0, len(rewards_array), group_size):
        group = rewards_array[start : start + group_size]
        centered = group - group.mean()
        if normalize_by_std:
            std = group.std(ddof=1) if len(group) > 1 else 0.0
            centered = np.zeros_like(centered) if std < 1e-8 else centered / (std + 1e-6)
        advantages.extend(centered.tolist())
    return advantages


def build_rl_training_datums(
    prompt_token_ids: list[list[int]],
    response_ids: list[list[int]],
    loss_masks: list[list[float]],
    sampled_logprobs: list[list[float]],
    step_advantages: list[float],
) -> tuple[list[types.Datum], dict[str, float]]:
    num_episodes = len(response_ids)
    if not (
        len(prompt_token_ids)
        == len(loss_masks)
        == len(sampled_logprobs)
        == len(step_advantages)
        == num_episodes
    ):
        raise ValueError("RL rollout fields must have one entry per episode")

    total_action_weight = float(sum(sum(float(value) for value in mask) for mask in loss_masks))
    if total_action_weight <= 0.0:
        raise ValueError("Cannot train on a rollout batch with zero action-token weight")

    datums: list[types.Datum] = []
    for prompt, response, response_mask, response_logprobs, advantage_value in zip(
        prompt_token_ids,
        response_ids,
        loss_masks,
        sampled_logprobs,
        step_advantages,
        strict=True,
    ):
        if not (len(response) == len(response_mask) == len(response_logprobs)):
            raise ValueError("Response ids, loss mask, and rollout logprobs must have equal lengths")
        full_sequence = prompt + response
        prompt_len = len(prompt)
        target_tokens = full_sequence[1:]
        rollout_logprobs = ([0.0] * prompt_len + response_logprobs)[1:]
        raw_mask = ([0.0] * prompt_len + response_mask)[1:]
        weights = [num_episodes * float(value) / total_action_weight for value in raw_mask]
        advantages = [float(advantage_value) if value > 0.0 else 0.0 for value in raw_mask]

        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                    "logprobs": TensorData.from_torch(torch.tensor(rollout_logprobs, dtype=torch.float32)),
                    "rollout_logprobs": TensorData.from_torch(
                        torch.tensor(rollout_logprobs, dtype=torch.float32)
                    ),
                    "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
                    "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32)),
                },
            )
        )

    return datums, {
        "num_episodes": float(num_episodes),
        "action_tokens": total_action_weight,
        "target_weight_sum": float(num_episodes),
    }
