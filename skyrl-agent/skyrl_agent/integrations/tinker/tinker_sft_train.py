import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any

import chz
import numpy as np
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B"
    tinker_base_url: str | None = None
    tinker_api_key: str | None = None
    tinker_project_id: str | None = None

    dataset_file: str = ""
    output_dir: str = "./tinker_output"
    run_name: str | None = None
    resume_exp_name: str | None = None

    batch_size: int = 1
    max_batch_tokens: int = 65536
    max_sequence_length: int | None = None
    max_examples: int | None = None
    max_steps: int = 1
    batch_order: str = "bucket_shuffle"
    bucket_size: int = 128
    seed: int = 0

    learning_rate: float = 5e-7
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0

    lora_rank: int = 32
    lora_pass_module_flags: bool = False
    lora_train_mlp: bool = True
    lora_train_attn: bool = True
    lora_train_unembed: bool = False

    save_every: int = 1
    save_every_examples: int = 0
    save_final: bool = True
    checkpoint_kind: str = "state"
    final_checkpoint_kind: str = "both"
    dry_run_pack_only: bool = False
    log_every: int = 1


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    prompt_token_ids: list[int]
    response_ids: list[int]
    loss_mask: list[float]
    supervised_tokens: int
    sequence_length: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"record {row.get('id', '<unknown>')} missing list field {key!r}")
    return value


def parse_sft_row(row: dict[str, Any], *, max_sequence_length: int | None) -> SFTExample:
    prompt_token_ids = [int(x) for x in _get_list(row, "prompt_token_ids")]
    response_ids = [int(x) for x in _get_list(row, "response_ids")]
    loss_mask = [float(x) for x in _get_list(row, "loss_mask")]

    if len(response_ids) != len(loss_mask):
        raise ValueError(
            f"record {row.get('id', '<unknown>')} has response/loss_mask length mismatch: "
            f"{len(response_ids)} vs {len(loss_mask)}"
        )
    if not prompt_token_ids:
        raise ValueError(f"record {row.get('id', '<unknown>')} has empty prompt_token_ids")
    if not response_ids:
        raise ValueError(f"record {row.get('id', '<unknown>')} has empty response_ids")

    sequence_length = len(prompt_token_ids) + len(response_ids)
    if max_sequence_length is not None and sequence_length > max_sequence_length:
        raise ValueError(f"sequence_length {sequence_length} exceeds max_sequence_length={max_sequence_length}")

    supervised_tokens = int(sum(1 for x in loss_mask if x > 0.0))
    if supervised_tokens <= 0:
        raise ValueError(f"record {row.get('id', '<unknown>')} has no supervised tokens")

    return SFTExample(
        example_id=str(row.get("id") or row.get("checkpoint_id") or row.get("paper_id") or "unknown"),
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        supervised_tokens=supervised_tokens,
        sequence_length=sequence_length,
    )


def load_sft_examples(path: str, *, max_examples: int | None, max_sequence_length: int | None) -> tuple[list[SFTExample], dict[str, int]]:
    stats = {
        "rows_seen": 0,
        "rows_kept": 0,
        "dropped_parse_error": 0,
        "dropped_overlong": 0,
        "dropped_empty_supervision": 0,
    }
    examples: list[SFTExample] = []
    with open(path) as f:
        for line in f:
            if max_examples is not None and stats["rows_seen"] >= max_examples:
                break
            stats["rows_seen"] += 1
            try:
                row = json.loads(line)
                example = parse_sft_row(row, max_sequence_length=max_sequence_length)
            except Exception as exc:
                message = str(exc)
                if "exceeds max_sequence_length" in message:
                    stats["dropped_overlong"] += 1
                elif "no supervised tokens" in message:
                    stats["dropped_empty_supervision"] += 1
                else:
                    stats["dropped_parse_error"] += 1
                logger.warning("Dropping SFT row %s: %s", stats["rows_seen"], exc)
                continue
            examples.append(example)
            stats["rows_kept"] += 1
    if not examples:
        raise ValueError(f"No usable SFT examples loaded from {path}")
    return examples, stats


def order_examples(examples: list[SFTExample], config: Config) -> list[SFTExample]:
    ordered = list(examples)
    rng = random.Random(config.seed)
    if config.batch_order == "length_sorted":
        return sorted(ordered, key=lambda item: item.sequence_length)
    if config.batch_order == "random":
        rng.shuffle(ordered)
        return ordered
    if config.batch_order == "bucket_shuffle":
        bucket_size = max(config.bucket_size, config.batch_size)
        ordered = sorted(ordered, key=lambda item: item.sequence_length)
        buckets = [ordered[i : i + bucket_size] for i in range(0, len(ordered), bucket_size)]
        for bucket in buckets:
            rng.shuffle(bucket)
        rng.shuffle(buckets)
        return [item for bucket in buckets for item in bucket]
    raise ValueError(f"Unsupported batch_order={config.batch_order}")


def build_batches(examples: list[SFTExample], config: Config) -> list[list[SFTExample]]:
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.max_batch_tokens <= 0:
        raise ValueError(f"max_batch_tokens must be positive, got {config.max_batch_tokens}")

    batches: list[list[SFTExample]] = []
    current: list[SFTExample] = []
    current_tokens = 0
    for example in order_examples(examples, config):
        would_overflow_count = len(current) >= config.batch_size
        would_overflow_tokens = current and current_tokens + example.sequence_length > config.max_batch_tokens
        if would_overflow_count or would_overflow_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        if example.sequence_length > config.max_batch_tokens:
            logger.warning(
                "Keeping single over-budget example %s with sequence_length=%s > max_batch_tokens=%s",
                example.example_id,
                example.sequence_length,
                config.max_batch_tokens,
            )
        current.append(example)
        current_tokens += example.sequence_length
    if current:
        batches.append(current)
    return batches


def make_tinker_datums(batch: list[SFTExample]) -> tuple[list[types.Datum], dict[str, float]]:
    target_masks: list[list[float]] = []
    raw_items: list[tuple[SFTExample, list[int], list[int], list[float]]] = []

    for example in batch:
        full_sequence = example.prompt_token_ids + example.response_ids
        target_tokens = full_sequence[1:]
        raw_mask = [0.0] * len(example.prompt_token_ids) + example.loss_mask
        shifted_mask = raw_mask[1:]
        if len(target_tokens) != len(shifted_mask):
            raise ValueError(
                f"target/mask length mismatch for {example.example_id}: {len(target_tokens)} vs {len(shifted_mask)}"
            )
        target_masks.append(shifted_mask)
        raw_items.append((example, full_sequence[:-1], target_tokens, shifted_mask))

    supervised_target_tokens = float(sum(sum(mask) for mask in target_masks))
    if supervised_target_tokens <= 0.0:
        raise ValueError("Cannot build SFT batch with zero supervised target-token weight")

    datums: list[types.Datum] = []
    max_weight_sum_error = 0.0
    for _example, model_input_tokens, target_tokens, shifted_mask in raw_items:
        # EasyDeL averages accumulated gradients over examples. Scaling the
        # batch weights to N therefore recovers one global token mean.
        weights = [len(batch) * float(mask) / supervised_target_tokens for mask in shifted_mask]
        max_weight_sum_error = max(
            max_weight_sum_error,
            abs(sum(weights) - len(batch) * sum(shifted_mask) / supervised_target_tokens),
        )
        zero_float = torch.zeros(len(target_tokens), dtype=torch.float32)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=model_input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32)),
                "logprobs": TensorData.from_torch(zero_float),
                "advantages": TensorData.from_torch(zero_float),
            },
        )
        datums.append(datum)

    total_weight = 0.0
    for datum in datums:
        total_weight += float(datum.loss_fn_inputs["weights"].to_torch().sum().item())

    return datums, {
        "batch_size": float(len(batch)),
        "sequence_tokens": float(sum(item.sequence_length for item in batch)),
        "supervised_tokens": supervised_target_tokens,
        "target_weight_sum": total_weight,
        "target_weight_sum_error": abs(total_weight - len(batch)),
        "max_internal_weight_sum_error": max_weight_sum_error,
    }


def summarize_examples(examples: list[SFTExample]) -> dict[str, float]:
    seq_lens = [item.sequence_length for item in examples]
    supervised = [item.supervised_tokens for item in examples]
    return {
        "num_examples": float(len(examples)),
        "sequence_length_min": float(min(seq_lens)),
        "sequence_length_mean": float(mean(seq_lens)),
        "sequence_length_max": float(max(seq_lens)),
        "supervised_tokens_min": float(min(supervised)),
        "supervised_tokens_mean": float(mean(supervised)),
        "supervised_tokens_max": float(max(supervised)),
    }


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    *,
    kind: str = "both",
) -> dict[str, str]:
    futures: dict[str, Any] = {}
    if kind in {"state", "both"}:
        futures["state"] = await training_client.save_state_async(name)
    if kind in {"sampler", "both"}:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)
    results = {key: await future.result_async() for key, future in futures.items()}
    paths = {f"{key}_path": value.path for key, value in results.items()}
    record = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info("Saved checkpoint %s: %s", name, paths)
    return paths


def find_resume_checkpoint(save_path: str) -> tuple[int, int, int, str | None]:
    checkpoint_path = os.path.join(save_path, "checkpoints.jsonl")
    if not os.path.exists(checkpoint_path):
        return 0, 0, 0, None
    records = []
    with open(checkpoint_path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "state_path" in record:
                records.append(record)
    if not records:
        return 0, 0, 0, None
    latest = max(
        records,
        key=lambda item: (
            int(item.get("examples_seen", 0)),
            int(item.get("sft_step", item.get("policy_iteration_step", 0))),
        ),
    )
    return (
        int(latest.get("sft_step", latest.get("policy_iteration_step", 0))),
        int(latest.get("examples_seen", 0)),
        int(latest.get("supervised_tokens_seen", 0)),
        latest["state_path"],
    )


def crossed_example_checkpoints(previous: int, current: int, interval: int) -> list[int]:
    if interval <= 0 or current <= previous:
        return []
    first = ((previous // interval) + 1) * interval
    return list(range(first, current + 1, interval))


def extract_loss(fwd_bwd_result: Any) -> float:
    losses: list[float] = []
    for output in fwd_bwd_result.loss_fn_outputs:
        elementwise = output.get("elementwise_loss")
        if elementwise is None:
            continue
        tensor = elementwise.to_torch()
        losses.append(float(tensor.sum().item()))
    if not losses:
        return float("nan")
    return float(sum(losses))


async def main(config: Config) -> None:
    if not config.dataset_file:
        raise ValueError("dataset_file must point to tokenized citation SFT JSONL")
    if config.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {config.max_steps}")
    for field_name in ("checkpoint_kind", "final_checkpoint_kind"):
        value = getattr(config, field_name)
        if value not in {"state", "sampler", "both"}:
            raise ValueError(f"{field_name} must be state, sampler, or both; got {value!r}")

    set_seed(config.seed)
    examples, load_stats = load_sft_examples(
        config.dataset_file,
        max_examples=config.max_examples,
        max_sequence_length=config.max_sequence_length,
    )
    batches = build_batches(examples, config)
    if not batches:
        raise ValueError("No SFT batches could be built")

    dataset_summary = summarize_examples(examples)
    first_datums, first_batch_stats = make_tinker_datums(batches[0])
    if abs(first_batch_stats["target_weight_sum"] - len(first_datums)) > 1e-5:
        raise RuntimeError(f"First SFT batch is not token-mean normalized: {first_batch_stats}")
    logger.info("Loaded SFT data from %s", config.dataset_file)
    logger.info("Load stats: %s", json.dumps(load_stats, sort_keys=True))
    logger.info("Dataset summary: %s", json.dumps(dataset_summary, sort_keys=True))
    logger.info("First batch stats: %s", json.dumps(first_batch_stats, sort_keys=True))
    logger.info("First batch datum count: %s", len(first_datums))

    if config.dry_run_pack_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "dataset_file": config.dataset_file,
                    "load_stats": load_stats,
                    "dataset_summary": dataset_summary,
                    "num_batches": len(batches),
                    "first_batch_stats": first_batch_stats,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    run_name = config.resume_exp_name or config.run_name or f"sft-{config.model_name.split('/')[-1]}-{datetime.now():%Y%m%d-%H%M%S}"
    save_path = os.path.join(config.output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    resume_from_step, resume_examples_seen, resume_supervised_tokens_seen, load_state_path = find_resume_checkpoint(
        save_path
    )
    if load_state_path:
        logger.info("Resuming SFT from step %s at %s", resume_from_step, load_state_path)

    service_kwargs: dict[str, Any] = {}
    if config.tinker_base_url:
        service_kwargs["base_url"] = config.tinker_base_url
    if config.tinker_api_key:
        service_kwargs["api_key"] = config.tinker_api_key
    if config.tinker_project_id:
        service_kwargs["project_id"] = config.tinker_project_id

    service_client = tinker.ServiceClient(**service_kwargs)
    lora_kwargs: dict[str, Any] = {"base_model": config.model_name, "rank": config.lora_rank}
    if config.lora_pass_module_flags:
        lora_kwargs.update(
            train_mlp=config.lora_train_mlp,
            train_attn=config.lora_train_attn,
            train_unembed=config.lora_train_unembed,
        )
    logger.info("Creating LoRA training client: %s", lora_kwargs)
    training_client = await service_client.create_lora_training_client_async(**lora_kwargs)
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        await future.result_async()
        logger.info("Loaded SFT state from %s", load_state_path)

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    start_time = time.time()
    recent_losses: list[float] = []
    examples_seen = resume_examples_seen
    supervised_tokens_seen = resume_supervised_tokens_seen

    total_steps = min(config.max_steps, len(batches))
    for local_step in range(resume_from_step, total_steps):
        batch = batches[local_step % len(batches)]
        datums, batch_stats = make_tinker_datums(batch)
        if abs(batch_stats["target_weight_sum"] - len(datums)) > 1e-5:
            raise RuntimeError(f"SFT batch is not token-mean normalized: {batch_stats}")

        step_start = time.time()
        fwd_bwd_future = training_client.forward_backward(
            datums,
            loss_fn="cross_entropy",
            loss_fn_config={"token_mean": 1.0},
        )
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        step = local_step + 1
        loss = extract_loss(fwd_bwd_result)
        recent_losses.append(loss)
        if len(recent_losses) > config.log_every:
            recent_losses.pop(0)
        previous_examples_seen = examples_seen
        examples_seen += len(batch)
        supervised_tokens_seen += int(batch_stats["supervised_tokens"])

        metrics = {
            "sft_step": step,
            "loss_token_mean": loss,
            "loss_recent_mean": float(mean(recent_losses)),
            "examples_seen": examples_seen,
            "supervised_tokens_seen": supervised_tokens_seen,
            "batch_size": int(batch_stats["batch_size"]),
            "batch_sequence_tokens": int(batch_stats["sequence_tokens"]),
            "batch_supervised_tokens": int(batch_stats["supervised_tokens"]),
            "target_weight_sum": batch_stats["target_weight_sum"],
            "time_step_sec": time.time() - step_start,
            "time_total_sec": time.time() - start_time,
        }
        if step == 1 or step % config.log_every == 0:
            logger.info("SFT metrics: %s", json.dumps(metrics, sort_keys=True))

        if config.save_every > 0 and step % config.save_every == 0:
            await save_checkpoint_async(
                training_client,
                f"{step:06d}",
                log_path=save_path,
                kind=config.checkpoint_kind,
                loop_state={**metrics, "policy_iteration_step": step},
            )

        for threshold in crossed_example_checkpoints(
            previous_examples_seen,
            examples_seen,
            config.save_every_examples,
        ):
            await save_checkpoint_async(
                training_client,
                f"examples_{threshold:06d}",
                log_path=save_path,
                kind=config.checkpoint_kind,
                loop_state={**metrics, "policy_iteration_step": step, "checkpoint_example_threshold": threshold},
            )

    if config.save_final:
        await save_checkpoint_async(
            training_client,
            f"final_{examples_seen:06d}",
            log_path=save_path,
            kind=config.final_checkpoint_kind,
            loop_state={
                "sft_step": total_steps,
                "policy_iteration_step": total_steps,
                "examples_seen": examples_seen,
                "supervised_tokens_seen": supervised_tokens_seen,
            },
        )
    logger.info("Tinker SFT completed successfully: save_path=%s", save_path)


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
