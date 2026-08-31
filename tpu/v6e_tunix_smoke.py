"""One real four-row Tunix training transaction through the Tinker API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tinker
from tinker import types


def logprob_vector(output: types.ForwardBackwardOutput) -> np.ndarray:
    """Flatten the per-row target logprobs into a stable state fingerprint."""
    rows = []
    for row, loss_output in enumerate(output.loss_fn_outputs):
        if "logprobs" not in loss_output:
            raise RuntimeError(f"forward output row {row} has no logprobs")
        rows.append(np.asarray(loss_output["logprobs"].data, dtype=np.float32))
    if not rows:
        raise RuntimeError("forward output is empty")
    values = np.concatenate(rows)
    if values.size == 0:
        raise RuntimeError("forward output logprobs are empty")
    return values


def vector_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def make_datum(tokenizer, row: int, sequence_length: int = 0) -> types.Datum:
    prompt = f"Question {row}: compute {row + 2}+{row + 3}.\nAnswer:"
    completion = f" {2 * row + 5}\n"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    if sequence_length:
        if sequence_length <= len(prompt_tokens):
            raise ValueError(
                f"sequence length {sequence_length} must exceed prompt length {len(prompt_tokens)}"
            )
        # Exercise a genuinely dense long-context mask instead of a short
        # example padded to the compiled shape. Cycling valid completion token
        # IDs keeps the request deterministic and tokenizer-independent.
        if not completion_tokens:
            raise RuntimeError("tokenizer produced no completion tokens")
        completion_length = sequence_length - len(prompt_tokens)
        repeats = (completion_length + len(completion_tokens) - 1) // len(completion_tokens)
        completion_tokens = (completion_tokens * repeats)[:completion_length]
    tokens = prompt_tokens + completion_tokens
    targets = tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * max(0, len(prompt_tokens) - 1) + [1.0] * (len(targets) - max(0, len(prompt_tokens) - 1))
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": targets,
            "weights": weights,
            "logprobs": [0.0] * len(targets),
            "advantages": [1.0] * len(targets),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, default=Path.home() / "v6e-tunix-smoke.json")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--rows", type=int, default=4, help="number of deterministic dense rows")
    parser.add_argument(
        "--replays",
        type=int,
        default=1,
        help="number of checkpoint restore + identical update replays",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="construct each row with exactly this many valid tokens (zero keeps the short smoke input)",
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if args.replays < 1:
        parser.error("--replays must be at least 1")

    service = tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")
    trainer = service.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    tokenizer = trainer.get_tokenizer()
    batch = [make_datum(tokenizer, row, args.sequence_length) for row in range(args.rows)]

    before = trainer.save_state(name="v6e-smoke-before").result()
    baseline = logprob_vector(trainer.forward(batch, "importance_sampling").result())
    fb_future = trainer.forward_backward(batch, "importance_sampling")
    optim_future = trainer.optim_step(types.AdamParams(learning_rate=1e-2))
    fb = fb_future.result()
    fb_values = logprob_vector(fb)
    optim = optim_future.result()
    updated = logprob_vector(trainer.forward(batch, "importance_sampling").result())
    after = trainer.save_state(name="v6e-smoke-after").result()

    grad_norm = float(optim.metrics.get("skyrl.ai/grad_norm", float("nan")))
    update_delta = float(np.max(np.abs(updated - baseline)))
    replay_records = []
    restored_checkpoint = None
    for replay_index in range(1, args.replays + 1):
        trainer.load_state_with_optimizer(before.path).result()
        restored_values = logprob_vector(trainer.forward(batch, "importance_sampling").result())
        if replay_index == 1:
            restored_checkpoint = trainer.save_state(name="v6e-smoke-restored").result()

        replay_fb_future = trainer.forward_backward(batch, "importance_sampling")
        replay_optim_future = trainer.optim_step(types.AdamParams(learning_rate=1e-2))
        replay_fb = replay_fb_future.result()
        replay_fb_values = logprob_vector(replay_fb)
        replay_optim = replay_optim_future.result()
        replay_updated = logprob_vector(trainer.forward(batch, "importance_sampling").result())
        replay_records.append(
            {
                "index": replay_index,
                "grad_norm": float(replay_optim.metrics.get("skyrl.ai/grad_norm", float("nan"))),
                "restore_logprob_max_abs_delta": float(np.max(np.abs(restored_values - baseline))),
                "updated_logprob_max_abs_delta": float(np.max(np.abs(replay_updated - updated))),
                "fb_logprob_max_abs_delta": float(np.max(np.abs(replay_fb_values - fb_values))),
                "restore_logprob_sha256": vector_hash(restored_values),
                "updated_logprob_sha256": vector_hash(replay_updated),
                "fb_logprob_sha256": vector_hash(replay_fb_values),
                "output_rows": len(replay_fb.loss_fn_outputs),
            }
        )

    replay_grad_norm = replay_records[0]["grad_norm"]
    restore_delta = replay_records[0]["restore_logprob_max_abs_delta"]
    replay_delta = replay_records[0]["updated_logprob_max_abs_delta"]
    grad_valid = bool(np.isfinite(grad_norm) and grad_norm > 0)
    replay_grad_valid = all(np.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 for r in replay_records)
    output_rows_valid = len(fb.loss_fn_outputs) == args.rows and all(
        r["output_rows"] == args.rows for r in replay_records
    )
    update_changed = update_delta > 1e-7
    restore_matches = all(r["restore_logprob_max_abs_delta"] <= 1e-5 for r in replay_records)
    replay_update_matches = all(r["updated_logprob_max_abs_delta"] <= 1e-5 for r in replay_records)
    replay_grad_matches = all(
        np.isclose(r["grad_norm"], grad_norm, rtol=1e-5, atol=1e-5) for r in replay_records
    )
    capacity_pass = grad_valid and replay_grad_valid and output_rows_valid and update_changed
    acceptance_pass = capacity_pass and restore_matches and replay_update_matches and replay_grad_matches

    result = {
        "base_model": args.base_model,
        "lora_rank": args.rank,
        "sequence_length": args.sequence_length or len(batch[0].model_input.to_ints()),
        "rows": len(batch),
        "requested_replays": args.replays,
        "grad_norm": grad_norm,
        "replay_grad_norm": replay_grad_norm,
        "grad_norms": [grad_norm] + [r["grad_norm"] for r in replay_records],
        "replays": replay_records,
        "update_logprob_max_abs_delta": update_delta,
        "restore_logprob_max_abs_delta": restore_delta,
        "replay_logprob_max_abs_delta": replay_delta,
        "capacity_pass": capacity_pass,
        "checkpoint_restore_pass": restore_matches,
        "replay_update_pass": replay_update_matches,
        "replay_grad_norm_pass": replay_grad_matches,
        "acceptance_pass": acceptance_pass,
        "baseline_logprob_sha256": vector_hash(baseline),
        "fb_logprob_sha256": vector_hash(fb_values),
        "updated_logprob_sha256": vector_hash(updated),
        "restored_logprob_sha256": replay_records[0]["restore_logprob_sha256"],
        "replay_updated_logprob_sha256": replay_records[0]["updated_logprob_sha256"],
        "before_checkpoint": before.path,
        "after_checkpoint": after.path,
        "restored_checkpoint": restored_checkpoint.path,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)

    # Write the diagnostic record before retaining the strict correctness
    # gates. Long-context fused kernels can therefore prove capacity while a
    # replay mismatch remains a visible failure rather than losing all data.
    if not grad_valid:
        raise RuntimeError(f"invalid grad norm from distributed step: {grad_norm}")
    if not replay_grad_valid:
        raise RuntimeError(f"invalid grad norm from replayed distributed step: {replay_grad_norm}")
    if len(fb.loss_fn_outputs) != args.rows:
        raise RuntimeError(f"expected {args.rows} forward/backward outputs, got {len(fb.loss_fn_outputs)}")
    bad_replay_rows = [r for r in replay_records if r["output_rows"] != args.rows]
    if bad_replay_rows:
        raise RuntimeError(f"replay forward/backward output-count mismatch: {bad_replay_rows}")
    if not update_changed:
        raise RuntimeError(f"optimizer step did not observably change target logprobs: max delta={update_delta}")
    if not restore_matches:
        raise RuntimeError(f"checkpoint restore did not reproduce baseline target logprobs: {replay_records}")
    if not replay_update_matches:
        raise RuntimeError(f"checkpoint replay did not reproduce the first optimizer update: {replay_records}")
    if not replay_grad_matches:
        raise RuntimeError(f"checkpoint replay grad norm mismatch: first={grad_norm}, replays={replay_records}")


if __name__ == "__main__":
    main()
