"""End-to-end EasyDeL Tinker backend smoke for a multi-host TPU slice."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import jax
from cloudpathlib import AnyPath

from skyrl.backends.easydel import EasyDeLBackend, EasyDeLBackendConfig
from skyrl.tinker import types
from skyrl.tinker.engine import prepare_model_pass_batch, prepare_sample_batch


def _rl_datum(
    prompt_tokens: list[int],
    sequence: types.GeneratedSequence,
    advantage: float,
) -> types.Datum:
    full_sequence = prompt_tokens + sequence.tokens
    response_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(sequence.tokens)
    old_logprobs = [0.0] * (len(prompt_tokens) - 1) + sequence.logprobs
    advantages = [0.0] * (len(prompt_tokens) - 1) + [advantage] * len(sequence.tokens)
    return types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=full_sequence[:-1])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=full_sequence[1:]),
            weights=types.TensorData(data=response_mask),
            advantages=types.TensorData(data=advantages),
            logprobs=types.TensorData(data=old_logprobs),
        ),
    )


def _sample_request(
    base_model: str,
    prompt_tokens: list[int],
    *,
    seed: int,
    checkpoint_id: str = "",
) -> types.SampleInput:
    return types.SampleInput(
        base_model=base_model,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=prompt_tokens)]),
        sampling_params=types.SamplingParams(
            temperature=1.0,
            max_tokens=4,
            seed=seed,
            top_k=32,
            top_p=0.95,
        ),
        num_samples=2,
        checkpoint_id=checkpoint_id,
        prompt_logprobs=False,
    )


def _long_context_rl_step(
    backend: EasyDeLBackend,
    model_id: str,
    sequence_length: int,
) -> dict[str, object]:
    """Run PPO with exact behavior logprobs over a long synthetic sequence."""
    if sequence_length < 2:
        raise ValueError("long-context sequence length must be at least 2")
    seed_tokens = backend.tokenizer.encode(
        "Training long-context language models requires stable distributed optimization. ",
        add_special_tokens=False,
    )
    if not seed_tokens:
        raise AssertionError("Tokenizer produced no tokens for the long-context seed text")
    full_tokens = (seed_tokens * math.ceil((sequence_length + 1) / len(seed_tokens)))[: sequence_length + 1]
    model_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=full_tokens[:-1])])
    targets = full_tokens[1:]
    scoring = types.ForwardBackwardInput(
        data=[
            types.Datum(
                model_input=model_input,
                loss_fn_inputs=types.LossFnInputs(
                    target_tokens=types.TensorData(data=targets),
                    weights=types.TensorData(data=[1.0] * sequence_length),
                    advantages=types.TensorData(data=[]),
                    logprobs=types.TensorData(data=[]),
                ),
            )
        ],
        loss_fn="cross_entropy",
    )
    optimizer_input = types.OptimStepInput(
        adam_params=types.AdamParams(
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
        )
    )

    def run_pass(label: str) -> dict[str, float]:
        score_started = time.perf_counter()
        scored = backend.forward(prepare_model_pass_batch({f"long-score-{label}": (model_id, scoring)}))[
            f"long-score-{label}"
        ]
        score_seconds = time.perf_counter() - score_started
        behavior_logprobs = scored.loss_fn_outputs[0]["logprobs"]["data"]
        if len(behavior_logprobs) != sequence_length or not all(math.isfinite(v) for v in behavior_logprobs):
            raise AssertionError("Long-context behavior logprobs are missing or non-finite")
        ppo = types.ForwardBackwardInput(
            data=[
                types.Datum(
                    model_input=model_input,
                    loss_fn_inputs=types.LossFnInputs(
                        target_tokens=types.TensorData(data=targets),
                        weights=types.TensorData(data=[1.0] * sequence_length),
                        advantages=types.TensorData(data=[1.0] * sequence_length),
                        logprobs=types.TensorData(data=behavior_logprobs),
                    ),
                )
            ],
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        )
        ppo_started = time.perf_counter()
        updated = backend.forward_backward(prepare_model_pass_batch({f"long-ppo-{label}": (model_id, ppo)}))[
            f"long-ppo-{label}"
        ]
        ppo_seconds = time.perf_counter() - ppo_started
        if not all(math.isfinite(value) for value in updated.loss_fn_outputs[0]["logprobs"]["data"]):
            raise AssertionError("Long-context PPO logprobs are non-finite")
        optim_started = time.perf_counter()
        step = backend.optim_step(model_id, optimizer_input)
        optim_seconds = time.perf_counter() - optim_started
        grad_norm = float((step.metrics or {}).get("skyrl.ai/grad_norm", 0.0))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise AssertionError(f"Expected a finite nonzero long-context gradient norm, got {grad_norm}")
        return {
            "score_seconds": score_seconds,
            "ppo_forward_backward_seconds": ppo_seconds,
            "optimizer_seconds": optim_seconds,
            "grad_norm": grad_norm,
        }

    memory_before = [device.memory_stats() for device in jax.local_devices()]
    compile_and_first_step = run_pass("compile")
    warmed_step = run_pass("warm")
    return {
        "sequence_length": sequence_length,
        "compile_and_first_step": compile_and_first_step,
        "warmed_step": warmed_step,
        "estimated_compile_overhead_seconds": {
            key: max(compile_and_first_step[key] - warmed_step[key], 0.0)
            for key in ("score_seconds", "ppo_forward_backward_seconds", "optimizer_seconds")
        },
        "memory_before": memory_before,
        "memory_after": [device.memory_stats() for device in jax.local_devices()],
    }


_ACTIVE_BACKEND: EasyDeLBackend | None = None


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_progress(path: Path, payload: object, progress_uri: str | None) -> None:
    _write_json(path, payload)
    if progress_uri:
        subprocess.run(
            ["gcloud", "storage", "cp", str(path), progress_uri],
            check=True,
        )


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _run_smoke() -> None:
    global _ACTIVE_BACKEND
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="trl-internal-testing/tiny-Qwen3ForCausalLM")
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--coordinator-address")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--distributed-service-name")
    parser.add_argument("--distributed-host", action="append", default=[])
    parser.add_argument("--tp", type=int, default=-1)
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-torch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-scan-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--long-context-lengths", default="")
    parser.add_argument("--long-context-only", action="store_true")
    parser.add_argument("--progress-uri")
    parser.add_argument("--resume-learner-path", type=Path)
    parser.add_argument("--resume-sampler-path", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = EasyDeLBackend(
        args.base_model,
        EasyDeLBackendConfig(
            max_lora_adapters=4,
            max_lora_rank=8,
            model_name_or_path=args.model_path,
            tokenizer_name_or_path=args.tokenizer_path,
            from_torch=args.from_torch,
            tensor_parallel_size=args.tp,
            sequence_parallel_size=args.sp,
            train_micro_batch_size=1,
            use_scan_mlp=args.use_scan_mlp,
            lmhead_token_chunk_size=128,
            lmhead_vocab_chunk_size=32768,
            sample_max_num_sequences=2,
            sample_max_model_len=32,
            sample_hbm_utilization=0.05,
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            sample_distributed_service_name=args.distributed_service_name,
            sample_distributed_hosts=args.distributed_host or None,
        ),
    )
    _ACTIVE_BACKEND = backend

    model_id = "tpu-smoke-policy"
    backend.create_model(model_id, types.LoraConfig(rank=2, alpha=4, seed=7))
    if args.long_context_only and args.resume_learner_path is not None:
        backend.load_checkpoint(AnyPath(args.resume_learner_path), model_id)
    elif args.resume_sampler_path is not None:
        # A persisted sampler archive contains the learner and optimizer state
        # as well as the sampler payload. Avoid loading the same state twice,
        # which transiently duplicates restored device arrays.
        backend.load_sampler_checkpoint(model_id, "resume-sampler", str(args.resume_sampler_path))
    elif args.resume_learner_path is not None:
        backend.load_checkpoint(AnyPath(args.resume_learner_path), model_id)
    restored_step = int(jax.device_get(backend._runtimes[model_id].state.step))
    prompt_tokens = backend.tokenizer.encode("Continue this sequence:", add_special_tokens=False)
    rollout_lengths: list[int] = []
    refreshed_lengths: list[int] = []
    grad_norm: float | None = None
    if not args.long_context_only:
        rollout = backend.sample(
            prepare_sample_batch({"rollout": (model_id, _sample_request(args.base_model, prompt_tokens, seed=17))})
        )["rollout"]
        if len(rollout.sequences) != 2 or not all(sequence.tokens for sequence in rollout.sequences):
            raise AssertionError("eSurge did not return two non-empty sequences")
        if not all(
            len(sequence.tokens) == len(sequence.logprobs) and all(math.isfinite(value) for value in sequence.logprobs)
            for sequence in rollout.sequences
        ):
            raise AssertionError("eSurge behavior logprobs are missing or non-finite")
        rollout_lengths = [len(sequence.tokens) for sequence in rollout.sequences]

        request = types.ForwardBackwardInput(
            data=[
                _rl_datum(prompt_tokens, rollout.sequences[0], 1.0),
                # Keep this deliberately asymmetric: a short smoke can sample
                # two identical sequences, in which case gradients cancel.
                _rl_datum(prompt_tokens, rollout.sequences[1], -0.25),
            ],
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        )
        forward = backend.forward_backward(prepare_model_pass_batch({"ppo": (model_id, request)}))["ppo"]
        if len(forward.loss_fn_outputs) != 2:
            raise AssertionError("PPO forward/backward did not return both sequences")
        step = backend.optim_step(
            model_id,
            types.OptimStepInput(
                adam_params=types.AdamParams(
                    learning_rate=1e-4,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                    weight_decay=0.0,
                )
            ),
        )
        grad_norm = float((step.metrics or {}).get("skyrl.ai/grad_norm", 0.0))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise AssertionError(f"Expected a finite nonzero gradient norm, got {grad_norm}")

    long_context_lengths = [int(value) for value in args.long_context_lengths.split(",") if value]
    if args.long_context_only and not long_context_lengths:
        raise ValueError("--long-context-only requires --long-context-lengths")
    long_context_results: list[dict[str, object]] = []
    progress_path = args.output_dir / "long-context-progress.json"
    for sequence_length in long_context_lengths:
        started = time.perf_counter()
        try:
            measurement = _long_context_rl_step(backend, model_id, sequence_length)
        except Exception as error:
            _write_progress(
                progress_path,
                {
                    "status": "failed",
                    "failed_sequence_length": sequence_length,
                    "error": f"{type(error).__name__}: {error}",
                    "completed": long_context_results,
                },
                args.progress_uri,
            )
            raise
        long_context_results.append(measurement)
        _write_progress(
            progress_path,
            {
                "status": "running",
                "last_completed_sequence_length": sequence_length,
                "last_wall_seconds": time.perf_counter() - started,
                "completed": long_context_results,
            },
            args.progress_uri,
        )

    learner_checkpoint = AnyPath(args.output_dir / "learner-checkpoint.pkl.gz")
    backend.save_checkpoint(learner_checkpoint, model_id)
    learner_path = Path(str(learner_checkpoint))
    learner_checkpoint_bytes = _path_bytes(learner_path)
    if learner_checkpoint_bytes <= 0:
        raise AssertionError("Learner checkpoint was not written on process zero")
    backend.load_checkpoint(learner_checkpoint, model_id)

    sampler_path: Path | None = None
    if not args.long_context_only:
        sampler_checkpoint = AnyPath(args.output_dir / "sampler-smoke.tar.gz")
        backend.save_sampler_checkpoint(sampler_checkpoint, model_id, persist=True)
        sampler_path = Path(str(sampler_checkpoint))
        if not sampler_path.is_file() or sampler_path.stat().st_size <= 0:
            raise AssertionError("Sampler checkpoint archive is missing or empty")

        refreshed = backend.sample(
            prepare_sample_batch({"refreshed": (model_id, _sample_request(args.base_model, prompt_tokens, seed=23))})
        )["refreshed"]
        if len(refreshed.sequences) != 2 or not all(sequence.tokens for sequence in refreshed.sequences):
            raise AssertionError("Refreshed eSurge sampler did not return two sequences")
        refreshed_lengths = [len(sequence.tokens) for sequence in refreshed.sequences]

    result = {
        "status": "passed",
        "mode": "long-context-only" if args.long_context_only else "lifecycle",
        "base_model": args.base_model,
        "process_count": args.num_processes or 1,
        "tp": args.tp,
        "sp": args.sp,
        "rollout_lengths": rollout_lengths,
        "refreshed_lengths": refreshed_lengths,
        "grad_norm": grad_norm,
        "restored_step": restored_step,
        "final_step": int(jax.device_get(backend._runtimes[model_id].state.step)),
        "long_context_results": long_context_results,
        "mesh": {name: int(size) for name, size in backend.mesh.shape.items()},
        "lmhead_token_chunk_size": backend.config.lmhead_token_chunk_size,
        "lmhead_vocab_chunk_size": backend.config.lmhead_vocab_chunk_size,
        "learner_checkpoint_bytes": learner_checkpoint_bytes,
        "sampler_checkpoint_bytes": sampler_path.stat().st_size if sampler_path is not None else 0,
    }
    _write_json(args.output_dir / "result.json", result)
    if long_context_lengths:
        _write_progress(
            progress_path,
            {"status": "passed", "completed": long_context_results},
            args.progress_uri,
        )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    try:
        _run_smoke()
    finally:
        if _ACTIVE_BACKEND is not None:
            _ACTIVE_BACKEND.shutdown()


if __name__ == "__main__":
    main()
