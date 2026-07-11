"""Exercise a nonzero Qwen3.5 RL update and checkpoint resume via Tinker."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _load_tokenizer(path: str):
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise
        return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)


def _datum(tokens: list[int], advantage: float) -> types.Datum:
    if len(tokens) < 3:
        raise ValueError("learner smoke needs at least three tokens")
    target_tokens = tokens[1:]
    response_start = max(len(target_tokens) // 2, 1)
    weights = [0.0] * response_start + [1.0] * (len(target_tokens) - response_start)
    advantages = [0.0 if weight == 0.0 else advantage for weight in weights]
    # The objective uses current-policy logprobs as the CISPO old policy. The
    # supplied behavior values only exercise token-level TIS and are capped.
    behavior_logprobs = [-5.0] * len(target_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "logprobs": TensorData.from_torch(torch.tensor(behavior_logprobs, dtype=torch.float32)),
            "rollout_logprobs": TensorData.from_torch(
                torch.tensor(behavior_logprobs, dtype=torch.float32)
            ),
        },
    )


def _update(training_client, datums: list[types.Datum], learning_rate: float) -> dict[str, float]:
    forward = training_client.forward_backward(
        datums,
        loss_fn="cispo",
        loss_fn_config={
            "clip_low_threshold": 1.0,
            "clip_high_threshold": 6.0,
            "tis_imp_ratio_cap": 2.0,
            "old_logprobs_from_target": 1.0,
            "token_mean": 1.0,
        },
    )
    optimizer = training_client.optim_step(
        types.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
    )
    forward_result = forward.result()
    optimizer_result = optimizer.result()
    metrics = {key: float(value) for key, value in (optimizer_result.metrics or {}).items()}
    grad_norm = metrics.get("skyrl.ai/grad_norm", 0.0)
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise AssertionError(f"expected finite nonzero gradient norm, got {grad_norm}")
    if not forward_result.loss_fn_outputs:
        raise AssertionError("CISPO forward/backward returned no loss outputs")
    for output in forward_result.loss_fn_outputs:
        logprobs = output["logprobs"].data
        if not logprobs or not all(math.isfinite(float(value)) for value in logprobs):
            raise AssertionError("CISPO returned missing or non-finite logprobs")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="tml-dummy")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--verify-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer_path)
    seed_texts = [
        "Research evidence should be retrieved, checked, and cited with exact paper identifiers.",
        "A policy update should increase useful citation behavior while preserving exploration.",
    ]
    datums = []
    for text, advantage in zip(seed_texts, (1.0, -0.25), strict=True):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        datums.append(_datum(tokens, advantage))

    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    client = service.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    first_metrics = _update(client, datums, args.learning_rate)
    first_state = client.save_state(name="qwen35-rl-smoke-step1").result().path
    sampler_state = client.save_weights_for_sampler(name="qwen35-rl-smoke-step1").result().path

    post_update_sample = None
    if args.verify_sampler:
        sampler = service.create_sampling_client(model_path=sampler_state)
        prompt = types.ModelInput.from_ints(
            tokenizer.encode("A useful research citation is", add_special_tokens=False)
        )
        sample_result = sampler.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(temperature=0.0, max_tokens=4, seed=7),
            num_samples=1,
        ).result()
        sequence = sample_result.sequences[0]
        if not sequence.tokens or len(sequence.tokens) != len(sequence.logprobs):
            raise AssertionError("post-update sampler returned missing tokens or logprobs")
        if not all(math.isfinite(float(value)) for value in sequence.logprobs):
            raise AssertionError("post-update sampler returned non-finite logprobs")
        post_update_sample = {
            "tokens": sequence.tokens,
            "logprobs": sequence.logprobs,
            "text": tokenizer.decode(sequence.tokens),
        }

    resumed = service.create_training_client_from_state(first_state)
    second_metrics = _update(resumed, datums, args.learning_rate)
    resumed_state = resumed.save_state(name="qwen35-rl-smoke-resumed-step2").result().path

    result = {
        "status": "passed",
        "base_model": args.base_model,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "num_datums": len(datums),
        "first_metrics": first_metrics,
        "second_metrics": second_metrics,
        "first_state_path": first_state,
        "sampler_state_path": sampler_state,
        "resumed_state_path": resumed_state,
        "post_update_sample": post_update_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
