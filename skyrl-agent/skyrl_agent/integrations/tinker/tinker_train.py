import asyncio
import json
import logging
import os
import pprint
import random
import time
from datetime import datetime
from typing import Any, Literal, List, Dict, cast
from contextlib import contextmanager

import chz
import numpy as np
import tinker
import torch
import wandb
from tinker import types
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from skyrl_agent import AutoAgentRunner
from skyrl_agent.integrations.tinker.tinker_rl_utils import build_rl_training_datums, compute_advantages_grpo

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    logger.info(f"Starting {key}")
    tstart = time.time()
    yield
    logger.info(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))


def compute_kl_sample_train(data_D: List[tinker.Datum], training_logprobs_D: List[torch.Tensor]) -> Dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs."""
    all_diffs: list[torch.Tensor] = []
    all_sampling_logprobs: list[torch.Tensor] = []

    for datum, training_logprobs in safezip(data_D, training_logprobs_D):
        # Get logprobs from sampling
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        mask_tensor = datum.loss_fn_inputs.get("weights") or datum.loss_fn_inputs.get("mask")
        if mask_tensor is None:
            raise ValueError("RL datum is missing both 'weights' and legacy 'mask'")
        action_mask = mask_tensor.to_torch() > 0
        # Extract only action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)

    assert all_diffs
    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()
    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
    }


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-32B"
    tokenizer_name_or_path: str | None = None
    tinker_base_url: str | None = None
    tinker_api_key: str | None = None
    tinker_project_id: str | None = None
    batch_size: int = 64
    eval_batch_size: int = 1024
    learning_rate: float = 4e-5
    lora_rank: int = 16
    lora_pass_module_flags: bool = False
    lora_train_mlp: bool = True
    lora_train_attn: bool = True
    lora_train_unembed: bool = True
    seed: int = 0
    max_steps: int = 200
    save_every: int = 2
    eval_every: int = 10
    resume_exp_name: str = None
    initial_state_path: str | None = None

    skyrl_agent_task_yaml: str = None
    dataset_file: str = None  # Path to the training dataset parquet file
    eval_dataset_file: str = None  # Path to the evaluation dataset parquet file

    # Loss function configuration
    loss_fn: Literal["importance_sampling", "ppo", "cispo"] = "cispo"
    # Options:
    #   "ppo" or "importance_sampling": Use Tinker's built-in loss (forward_backward)

    # GRPO (Group Relative Policy Optimization) settings
    group_size: int = 8  # Trajectories per prompt group (None = auto-infer from task yaml)
    grpo_norm_by_std: bool = True
    cispo_clip_low_threshold: float = 1.0
    cispo_clip_high_threshold: float = 6.0
    tis_imp_ratio_cap: float = 2.0
    token_mean: bool = True

    wandb_project: str | None = None
    wandb_name: str | None = None
    log_dir: str | None = None
    output_dir: str = "./tinker_output"
    agent_max_parallel: int | None = None
    agent_max_prompt_length: int | None = None


def prepare_agent_task_yaml(config: Config, save_path: str) -> str:
    """Write a runtime task YAML whose trajectory fanout matches this run."""
    task_config = OmegaConf.load(config.skyrl_agent_task_yaml)

    if config.group_size is not None:
        task_config.generator.num_trajectories = int(config.group_size)

    if config.agent_max_prompt_length is not None:
        task_config.generator.max_prompt_length = int(config.agent_max_prompt_length)

    if config.agent_max_parallel is not None:
        max_parallel = int(config.agent_max_parallel)
    else:
        max_parallel = int(config.batch_size) * int(task_config.generator.num_trajectories)

    # Keep smoke tests and small TPU runs from launching hidden concurrent samples.
    task_config.dispatcher.max_parallel_agents = min(int(task_config.dispatcher.max_parallel_agents), max_parallel)
    task_config.dispatcher.max_eval_parallel_agents = min(
        int(task_config.dispatcher.max_eval_parallel_agents),
        max_parallel,
    )

    runtime_task_yaml = os.path.join(save_path, "runtime_task.yaml")
    OmegaConf.save(task_config, runtime_task_yaml)
    print(
        "[tinker-train] runtime task config: "
        f"num_trajectories={task_config.generator.num_trajectories}, "
        f"max_prompt_length={task_config.generator.max_prompt_length}, "
        f"max_parallel_agents={task_config.dispatcher.max_parallel_agents}, "
        f"path={runtime_task_yaml}",
        flush=True,
    )
    return runtime_task_yaml


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


def collate_fn(batch):
    """Custom collate function that returns batch as-is without tensor collation.

    This is needed because the agent runner expects to handle the raw batch data
    through build_generator_input, rather than having PyTorch stack tensors.
    """
    return batch


async def main(config: Config):
    print("[tinker-train] main: starting", flush=True)
    # Set random seed for reproducibility
    set_seed(config.seed)

    # Setup logging
    if config.resume_exp_name:
        wandb_name = config.resume_exp_name
    else:
        wandb_name = config.wandb_name or config.model_name.split("/")[-1]
        wandb_name += "_" + datetime.now().strftime("%m%dT%H:%M:%S")
    save_path = os.path.join(config.output_dir, wandb_name)
    os.makedirs(save_path, exist_ok=True)

    # read the most recent checkpoint
    checkpoint_path = os.path.join(save_path, "checkpoints.jsonl")
    load_state_path = None  # Initialize to None
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoints = [json.loads(line) for line in f]
        most_recent_checkpoint = max(checkpoints, key=lambda x: x["policy_iteration_step"])
        resume_from_step = most_recent_checkpoint["policy_iteration_step"]
        load_state_path = most_recent_checkpoint["state_path"]
        print(f"Resuming training from step {resume_from_step}")
    else:
        resume_from_step = 0
        print("Starting training from scratch")

    wandb.init(
        project=config.wandb_project,
        config=chz.asdict(config),
        dir=str(config.log_dir) if config.log_dir else None,
        name=wandb_name,
    )
    print("[tinker-train] wandb initialized", flush=True)

    # dataset and dataloader
    print(f"[tinker-train] loading train dataset: {config.dataset_file}", flush=True)
    train_dataset = load_dataset("parquet", data_files=config.dataset_file)["train"]
    print(f"[tinker-train] loading eval dataset: {config.eval_dataset_file}", flush=True)
    eval_dataset = load_dataset("parquet", data_files=config.eval_dataset_file)["train"]

    # Calculate steps per epoch for tracking
    steps_per_epoch = (len(train_dataset) + config.batch_size - 1) // config.batch_size
    logger.info(f"Dataset size: {len(train_dataset)}, Steps per epoch: {steps_per_epoch}")

    # Create function to get dataloader for a specific epoch
    def create_train_dataloader(epoch: int):
        """Create dataloader with epoch-specific seed for different shuffle orders."""
        return DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(config.seed + epoch),  # Different shuffle per epoch
        )

    # Initialize iterator state for resuming
    current_epoch = resume_from_step // steps_per_epoch
    batch_offset_in_epoch = resume_from_step % steps_per_epoch

    train_dataloader = create_train_dataloader(current_epoch)
    train_iterator = iter(train_dataloader)

    # Skip batches within the current epoch if resuming mid-epoch
    if batch_offset_in_epoch > 0:
        logger.info(f"Resuming from epoch {current_epoch}, batch {batch_offset_in_epoch}/{steps_per_epoch}")
        for _ in range(batch_offset_in_epoch):
            next(train_iterator)

    # Setup agent (tinker training client)
    service_kwargs: dict[str, Any] = {}
    if config.tinker_base_url:
        service_kwargs["base_url"] = config.tinker_base_url
    if config.tinker_api_key:
        service_kwargs["api_key"] = config.tinker_api_key
    if config.tinker_project_id:
        service_kwargs["project_id"] = config.tinker_project_id
    print(f"[tinker-train] creating Tinker service client: {service_kwargs}", flush=True)
    service_client = tinker.ServiceClient(**service_kwargs)
    lora_kwargs: dict[str, Any] = {"base_model": config.model_name, "rank": config.lora_rank}
    if config.lora_pass_module_flags:
        lora_kwargs.update(
            train_mlp=config.lora_train_mlp,
            train_attn=config.lora_train_attn,
            train_unembed=config.lora_train_unembed,
        )
    print(f"[tinker-train] creating LoRA training client: {lora_kwargs}", flush=True)
    training_client = await service_client.create_lora_training_client_async(**lora_kwargs)
    print("[tinker-train] LoRA training client ready", flush=True)
    state_to_load = load_state_path or config.initial_state_path
    if state_to_load:
        future = await training_client.load_state_async(state_to_load)
        _ = await future.result_async()
        print(
            f"[tinker-train] loaded {'resume' if load_state_path else 'initial'} state: {state_to_load}",
            flush=True,
        )
        logger.info(
            "Loaded %s state from %s",
            "resume" if load_state_path else "initial",
            state_to_load,
        )

    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    skyrl_agent_task_yaml_path = prepare_agent_task_yaml(config, save_path)
    tokenizer_local_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    print(
        f"[tinker-train] loading tokenizer: {config.model_name} local_files_only={tokenizer_local_only}",
        flush=True,
    )
    tokenizer_source = config.tokenizer_name_or_path or config.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=tokenizer_local_only)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise
        logger.warning("Falling back to PreTrainedTokenizerFast for converted EasyDeL tokenizer: %s", exc)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_source,
            local_files_only=tokenizer_local_only,
        )
    print("[tinker-train] tokenizer ready", flush=True)

    # training loop
    for policy_iteration_step in range(resume_from_step, config.max_steps):
        print("=" * 10 + f" Step {policy_iteration_step} " + "=" * 10, flush=True)
        metrics = {
            "step": policy_iteration_step,
            "epoch": policy_iteration_step // steps_per_epoch,
            "batch_in_epoch": policy_iteration_step % steps_per_epoch,
        }

        # save model
        if config.save_every > 0 and policy_iteration_step > 0 and policy_iteration_step % config.save_every == 0:
            await save_checkpoint_async(
                training_client,
                f"{policy_iteration_step:06d}",
                log_path=save_path,
                kind="state",
                loop_state={"policy_iteration_step": policy_iteration_step},
            )

        print(f"[tinker-train] saving sampler weights for step {policy_iteration_step}", flush=True)
        sampling_path = training_client.save_weights_for_sampler(name=f"{policy_iteration_step:06d}").result().path
        print(f"[tinker-train] sampler weights ready: {sampling_path}", flush=True)
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        print("[tinker-train] creating agent generator", flush=True)
        agent_generator = AutoAgentRunner.from_task(
            skyrl_agent_task_yaml_path, infer_engine=sampling_client, tokenizer=tokenizer
        )
        print("[tinker-train] agent generator ready", flush=True)

        if config.eval_every > 0 and policy_iteration_step % config.eval_every == 0:
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn
            )
            data_source_rewards = {}
            for batch in eval_dataloader:
                input_batch = batch
                rollouts = await agent_generator.run(input_batch, val_mode=True)
                traj_rewards_list = rollouts["traj_rewards"]
                for data, reward in zip(input_batch, traj_rewards_list):
                    data_source = data["data_source"]
                    if data_source not in data_source_rewards:
                        data_source_rewards[data_source] = []
                    data_source_rewards[data_source].append(reward)
            # get avg reward per data source
            for data_source, rewards in data_source_rewards.items():
                metrics[f"eval/reward/mean/{data_source}"] = np.mean(rewards)

        # Collect rollouts using AgentRunner
        print(f"🎲 Start collecting episodes at step {policy_iteration_step}")
        st = time.time()

        # Get next batch, handling epoch transitions
        try:
            input_batch = next(train_iterator)
        except StopIteration:
            # Start new epoch with different shuffle order
            current_epoch += 1
            logger.info(f"Starting epoch {current_epoch} with new shuffle order")
            train_dataloader = create_train_dataloader(current_epoch)
            train_iterator = iter(train_dataloader)
            input_batch = next(train_iterator)

        rollouts = await agent_generator.run(input_batch, val_mode=False)
        metrics["time/sample"] = time.time() - st
        # rollout time
        print(f"Rollout time: {metrics['time/sample']}")

        # Write rollout_metrics to wandb
        rollout_metrics = rollouts.get("rollout_metrics", {})
        wandb.log({f"rollout/{k}": v for k, v in rollout_metrics.items()}, step=policy_iteration_step)

        # Extract rollout data
        prompt_token_ids = rollouts["prompt_token_ids"]  # List of prompt token IDs
        response_ids = rollouts["response_ids"]  # List of response token IDs
        traj_rewards_list = rollouts["traj_rewards"]  # List of rewards (binary: 0 or 1)
        loss_masks = rollouts["loss_masks"]  # List of loss masks
        sampled_logprobs = rollouts["rollout_logprobs"]  # List of sampled logprobs
        num_steps_per_trajectory = rollouts["episode_nums"]  # List of number of steps per trajectory

        actual_batch_size = len(response_ids)
        logger.info(f"Processing {actual_batch_size} rollouts for training")

        # Compute advantages using GRPO (Group Relative Policy Optimization)
        all_returns = [float(r) for r in traj_rewards_list]

        # Determine group size for GRPO
        group_size = config.group_size
        if group_size is None:
            task_config = OmegaConf.load(skyrl_agent_task_yaml_path)
            group_size = task_config.generator.get("num_trajectories", 1)
            logger.info(f"Auto-inferred group_size={group_size} from task config")

        # Compute GRPO advantages
        logger.info(f"Computing GRPO advantages: group_size={group_size}, norm_by_std={config.grpo_norm_by_std}")
        all_advantages = compute_advantages_grpo(
            all_returns, group_size=group_size, normalize_by_std=config.grpo_norm_by_std
        )
        # broadcast advantages to num_steps per trajectory
        step_advantages = []
        for idx, num_steps in enumerate(num_steps_per_trajectory):
            step_advantages.extend([all_advantages[idx]] * num_steps)

        metrics["reward/mean"] = np.mean(all_returns)
        metrics["reward/max"] = np.max(all_returns)
        metrics["reward/min"] = np.min(all_returns)
        metrics["advantage/mean"] = np.mean(all_advantages)
        metrics["advantage/std"] = np.std(all_advantages)

        # Prepare training datums compatible with Tinker API
        # For each trajectory, we need to provide:
        # - model_input: the full sequence (prompt + response)
        # - loss_fn_inputs: target_tokens, advantages, logprobs (if available), mask
        training_datums, datum_stats = build_rl_training_datums(
            prompt_token_ids,
            response_ids,
            loss_masks,
            sampled_logprobs,
            step_advantages,
        )
        metrics.update({f"train_batch/{key}": value for key, value in datum_stats.items()})

        # Training step
        print(f"🎈 Start training at step {policy_iteration_step}")
        st = time.time()

        # Use Tinker's built-in loss function ("ppo" or "importance_sampling")
        loss_fn_config = {
            "clip_low_threshold": config.cispo_clip_low_threshold,
            "clip_high_threshold": config.cispo_clip_high_threshold,
            "tis_imp_ratio_cap": config.tis_imp_ratio_cap,
            "old_logprobs_from_target": 1.0,
            "token_mean": float(config.token_mean),
        }
        fwd_bwd_future = training_client.forward_backward(
            training_datums,
            loss_fn=config.loss_fn,
            loss_fn_config=loss_fn_config,
        )
        # Optimize
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()

        # Extract training logprobs from loss_fn_outputs
        training_logprobs_D: list[torch.Tensor] = []
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs = output["logprobs"].to_torch()
            training_logprobs_D.append(training_logprobs)
        with timed("compute_kl_sample_train", metrics):
            kl_sample_train_metrics = compute_kl_sample_train(training_datums, training_logprobs_D)
            metrics.update(kl_sample_train_metrics)

        optim_result = optim_step_future.result()
        for key, value in (optim_result.metrics or {}).items():
            metrics[f"optim/{key}"] = float(value)
        metrics["time/train"] = time.time() - st

        pprint.pprint(metrics)
        wandb.log(metrics, step=policy_iteration_step)

    # Save final checkpoint
    if config.save_every > 0:
        await save_checkpoint_async(
            training_client,
            "final",
            log_path=save_path,
            kind="both",
            loop_state={"policy_iteration_step": config.max_steps},
        )

    wandb.finish()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
