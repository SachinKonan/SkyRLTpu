"""Resumable eval-only runner for Tinker-backed SkyRL agent checkpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import chz
import numpy as np
import tinker
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from skyrl_agent import AutoAgentRunner

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3.5-9B"
    tokenizer_name_or_path: str | None = None
    tinker_base_url: str | None = None
    tinker_api_key: str | None = None
    tinker_project_id: str | None = None
    lora_rank: int = 32
    state_paths: str = ""
    checkpoint_labels: str = ""
    skyrl_agent_task_yaml: str = ""
    eval_dataset_file: str = ""
    output_dir: str = "./tinker_eval_output"
    max_examples: int = 100
    batch_size: int = 8
    max_parallel: int = 4
    max_prompt_length: int = 131072
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 20
    max_tokens: int = 4096


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_runtime_task(config: Config, output_dir: Path) -> Path:
    task = OmegaConf.load(config.skyrl_agent_task_yaml)
    task.data.instance_id_key = "eval_index"
    task.generator.max_prompt_length = int(config.max_prompt_length)
    task.generator.val_config.num_trajectories = 1
    task.generator.val_config.sampling_params.temperature = float(config.temperature)
    task.generator.val_config.sampling_params.top_p = float(config.top_p)
    task.generator.val_config.sampling_params.top_k = int(config.top_k)
    task.generator.val_config.sampling_params.max_tokens = int(config.max_tokens)
    task.dispatcher.max_eval_parallel_agents = int(config.max_parallel)
    task.dispatcher.val_config = {
        "max_parallel_agents": int(config.max_parallel),
        "max_eval_parallel_agents": int(config.max_parallel),
    }
    path = output_dir / "runtime_task.yaml"
    OmegaConf.save(task, path)
    return path


def _load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            completed[int(record["eval_index"])] = record
    return completed


def _write_summary(label: str, checkpoint: str, records: list[dict[str, Any]], path: Path) -> None:
    rewards = [float(record["reward"]) for record in records]
    reasons = Counter(str(record.get("finish_reason") or "unknown") for record in records)
    summary = {
        "checkpoint_label": label,
        "checkpoint_path": checkpoint,
        "num_examples": len(records),
        "recall": float(np.mean(rewards)) if rewards else 0.0,
        "pass_at_1": float(np.mean([reward > 0 for reward in rewards])) if rewards else 0.0,
        "max_single_prompt_recall": max(rewards, default=0.0),
        "finish_reasons": dict(sorted(reasons.items())),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(f"[tinker-eval] summary: {json.dumps(summary, sort_keys=True)}", flush=True)


async def main(config: Config) -> None:
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = _csv(config.state_paths)
    labels = _csv(config.checkpoint_labels)
    if not checkpoints or len(checkpoints) != len(labels):
        raise ValueError("state_paths and checkpoint_labels must be non-empty CSV lists of equal length")

    service_kwargs: dict[str, Any] = {}
    if config.tinker_base_url:
        service_kwargs["base_url"] = config.tinker_base_url
    if config.tinker_api_key:
        service_kwargs["api_key"] = config.tinker_api_key
    if config.tinker_project_id:
        service_kwargs["project_id"] = config.tinker_project_id
    service_client = tinker.ServiceClient(**service_kwargs)
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    tokenizer_source = config.tokenizer_name_or_path or config.model_name
    local_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_source, local_files_only=local_only)

    dataset = load_dataset("parquet", data_files=config.eval_dataset_file, split="train")
    limit = min(int(config.max_examples), len(dataset))
    examples = [dict(dataset[index], eval_index=index) for index in range(limit)]
    runtime_task = _write_runtime_task(config, output_root)

    for checkpoint, label in zip(checkpoints, labels, strict=True):
        checkpoint_dir = output_root / label
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        records_path = checkpoint_dir / "trajectories.jsonl"
        summary_path = checkpoint_dir / "aggregate_first100.json"
        completed = _load_completed(records_path)
        if len(completed) >= limit:
            _write_summary(label, checkpoint, [completed[index] for index in sorted(completed)], summary_path)
            continue

        print(f"[tinker-eval] loading {label}: {checkpoint}", flush=True)
        load_future = await training_client.load_state_async(checkpoint)
        await load_future.result_async()
        sampler_future = await training_client.save_weights_for_sampler_async(f"eval_{label}")
        sampler_result = await sampler_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=sampler_result.path)
        runner = AutoAgentRunner.from_task(str(runtime_task), infer_engine=sampling_client, tokenizer=tokenizer)

        pending = [example for example in examples if int(example["eval_index"]) not in completed]
        for offset in range(0, len(pending), int(config.batch_size)):
            batch = pending[offset : offset + int(config.batch_size)]
            rollouts = await runner.run(batch, val_mode=True)
            trajectory_results = rollouts["trajectory_results"]
            if len(trajectory_results) != len(batch):
                raise RuntimeError(f"expected {len(batch)} trajectories, got {len(trajectory_results)}")
            with records_path.open("a") as handle:
                for example, result in zip(batch, trajectory_results, strict=True):
                    record = {
                        "eval_index": int(example["eval_index"]),
                        "data_source": example.get("data_source"),
                        "reward": float(result.get("reward", 0.0)),
                        "finish_reason": result.get("finish_reason"),
                        "eval_error": result.get("eval_error"),
                        "result": result.get("result"),
                        "messages": result.get("messages", []),
                        "state": result.get("state", {}),
                    }
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    handle.flush()
                    completed[record["eval_index"]] = record
            ordered = [completed[index] for index in sorted(completed)]
            _write_summary(label, checkpoint, ordered, summary_path)
            print(f"[tinker-eval] {label}: {len(completed)}/{limit}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(chz.entrypoint(Config)))
