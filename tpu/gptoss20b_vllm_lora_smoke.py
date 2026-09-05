#!/usr/bin/env python3
"""Live GPT-OSS 20B MXFP4 + LoRA acceptance client for a vLLM TPU server.

The harness creates deterministic synthetic adapters in the exact split format
emitted by the Tunix backend: ordinary PEFT tensors for the BF16 router and a
``moe_lora.safetensors`` sidecar for sparse expert factors.  It proves:

* a zero adapter is numerically identical to the native MXFP4 base;
* nonzero expert factors alter inference;
* base, expert A, and expert B can execute concurrently without crosstalk;
* moving to a router-only adapter clears only its assigned expert slot;
* ordinary vLLM router LoRA alters inference; and
* clearing the adapter restores the initial base response exactly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tarfile
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file


MODEL = "openai/gpt-oss-20b"
LAYERS = 24
EXPERTS = 32
HIDDEN = 2880
INTERMEDIATE = 2880


def _adapter_config(rank: int, alpha: float) -> dict[str, Any]:
    return {
        "base_model_name_or_path": MODEL,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": ["router"],
        "task_type": "CAUSAL_LM",
    }


def _random(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return np.ascontiguousarray(rng.normal(0.0, scale, shape).astype(np.float32))


def _write_adapter(
    root: Path,
    name: str,
    *,
    rank: int,
    alpha: float,
    router_seed: int | None,
    expert_seed: int | None,
) -> Path:
    target = root / name
    target.mkdir(parents=True)

    router: dict[str, np.ndarray] = {}
    router_rng = np.random.default_rng(router_seed or 0)
    for layer in range(LAYERS):
        prefix = f"base_model.model.model.layers.{layer}.mlp.router"
        if router_seed is None:
            a = np.zeros((rank, HIDDEN), np.float32)
            b = np.zeros((EXPERTS, rank), np.float32)
        else:
            # Router-only validation uses a deliberately visible but bounded
            # update; expert adapters keep these tensors exactly zero.
            a = _random(router_rng, (rank, HIDDEN), 0.08)
            b = _random(router_rng, (EXPERTS, rank), 0.08)
        router[f"{prefix}.lora_A.weight"] = a
        router[f"{prefix}.lora_B.weight"] = b
    save_file(router, str(target / "adapter_model.safetensors"))
    (target / "adapter_config.json").write_text(
        json.dumps(_adapter_config(rank, alpha), indent=2) + "\n"
    )

    if expert_seed is not None:
        expert: dict[str, np.ndarray] = {}
        expert_rng = np.random.default_rng(expert_seed)
        is_zero = expert_seed == 0

        def factor(shape: tuple[int, ...]) -> np.ndarray:
            if is_zero:
                return np.zeros(shape, np.float32)
            return _random(expert_rng, shape, 0.035)

        for layer in range(LAYERS):
            for component in ("wi_0", "wi_1"):
                expert[f"layers.{layer}.{component}.lora_a"] = factor((HIDDEN, rank))
                expert[f"layers.{layer}.{component}.lora_b"] = factor(
                    (rank, EXPERTS, INTERMEDIATE)
                )
            expert[f"layers.{layer}.wo.lora_a"] = factor(
                (EXPERTS, INTERMEDIATE, rank)
            )
            expert[f"layers.{layer}.wo.lora_b"] = factor((rank, HIDDEN))
        save_file(expert, str(target / "moe_lora.safetensors"))
        (target / "moe_lora.json").write_text(
            json.dumps(
                {
                    "format": "gptoss-moe-lora/v1",
                    "rank": rank,
                    "alpha": alpha,
                    "scale": alpha / rank,
                    "num_layer_groups": 2,
                    "fixture": name,
                },
                indent=2,
            )
            + "\n"
        )
    return target


def _tar_adapter(adapter: Path) -> Path:
    tar_path = adapter.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as archive:
        for child in sorted(adapter.iterdir()):
            archive.add(child, arcname=child.name)
    return tar_path


def _json_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _upload(
    base_url: str,
    tar_path: Path,
    name: str,
    previous: str | None,
    timeout: float,
) -> dict[str, Any]:
    query = {"lora_name": name}
    if previous:
        query["previous_lora_name"] = previous
    url = f"{base_url}/skyrl/v1/upload_lora_adapter?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        data=tar_path.read_bytes(),
        headers={"Content-Type": "application/x-tar"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} uploading {name}: {body}") from exc


PROMPTS = (
    "The capital of France is",
    "A concise proof that 2 + 2 = 4 begins:",
)


def _sample(base_url: str, model: str, timeout: float) -> list[dict[str, Any]]:
    samples = []
    for prompt in PROMPTS:
        response = _json_request(
            f"{base_url}/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 4,
                "temperature": 0,
                "seed": 7524,
                "logprobs": 20,
            },
            timeout,
        )
        choice = response["choices"][0]
        logprobs = choice.get("logprobs") or {}
        samples.append(
            {
                "prompt": prompt,
                "text": choice.get("text"),
                "tokens": logprobs.get("tokens"),
                "token_logprobs": logprobs.get("token_logprobs"),
                "top_logprobs": logprobs.get("top_logprobs"),
            }
        )
    return samples


def _sample_concurrently(base_url: str, models: tuple[str, ...],
                         timeout: float) -> dict[str, list[dict[str, Any]]]:
    """Issue different-model requests together so vLLM can mix LoRA slots."""

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(models)) as executor:
        futures = {
            model: executor.submit(_sample, base_url, model, timeout)
            for model in models
        }
        return {model: futures[model].result() for model in models}


def _fingerprint(samples: list[dict[str, Any]]) -> str:
    raw = json.dumps(samples, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _distance(left: Any, right: Any) -> float:
    """Maximum numeric delta, or a finite sentinel for discrete differences."""
    discrete_difference = 1e300
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return discrete_difference
        return max((_distance(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return discrete_difference
        return max((_distance(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else discrete_difference


def _updates(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"layers", "cleared", "base_weights_mutated"} <= set(value):
            found.append(value)
        for child in value.values():
            found.extend(_updates(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_updates(child))
    return found


def _assert_update(response: dict[str, Any], *, cleared: bool) -> None:
    matches = _updates(response.get("moe_update"))
    if not matches:
        raise AssertionError(f"upload returned no MXFP4 worker update: {response}")
    for update in matches:
        if int(update["layers"]) != LAYERS:
            raise AssertionError(f"expected {LAYERS} updated layers: {update}")
        if bool(update["cleared"]) is not cleared:
            raise AssertionError(f"unexpected clear result: {update}")
        if bool(update["base_weights_mutated"]):
            raise AssertionError(f"MXFP4 base mutation reported: {update}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "schema": "gptoss20b-vllm-mxfp4-multilora-smoke/v2",
        "acceptance_pass": False,
        "model": MODEL,
        "accelerator": "v6e-8",
        "zone": args.zone,
        "repo_commit": args.repo_commit,
        "tpu_inference_commit": args.tpu_inference_commit,
        "rank": args.rank,
        "alpha": args.alpha,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="gptoss20b-lora-") as tmp:
            root = Path(tmp)
            specs = {
                "zero": dict(router_seed=None, expert_seed=0),
                "expert_a": dict(router_seed=None, expert_seed=7524),
                "expert_b": dict(router_seed=None, expert_seed=9752),
                "router": dict(router_seed=4242, expert_seed=None),
                "clear": dict(router_seed=None, expert_seed=None),
            }
            archives = {
                name: _tar_adapter(
                    _write_adapter(
                        root,
                        name,
                        rank=args.rank,
                        alpha=args.alpha,
                        **spec,
                    )
                )
                for name, spec in specs.items()
            }

            samples: dict[str, list[dict[str, Any]]] = {}
            uploads: dict[str, dict[str, Any]] = {}
            samples["base_initial"] = _sample(args.base_url, MODEL, args.request_timeout)

            # Keep zero/A/B resident together. Passing no previous adapter is
            # the contract exercised by multi-tenant SkyRL sampling.
            for name in ("zero", "expert_a", "expert_b"):
                uploads[name] = _upload(
                    args.base_url,
                    archives[name],
                    name,
                    None,
                    args.upload_timeout,
                )
                _assert_update(uploads[name], cleared=False)
                samples[name] = _sample(args.base_url, name,
                                        args.request_timeout)

            mixed = _sample_concurrently(
                args.base_url,
                (MODEL, "expert_a", "expert_b"),
                args.request_timeout,
            )
            samples["mixed_base"] = mixed[MODEL]
            samples["mixed_expert_a"] = mixed["expert_a"]
            samples["mixed_expert_b"] = mixed["expert_b"]

            # Fill the fourth resident slot with an ordinary router adapter,
            # then force an LRU slot reuse with an all-zero adapter. The new
            # occupant must not inherit expert factors from the evicted slot.
            for name in ("router", "clear"):
                uploads[name] = _upload(
                    args.base_url,
                    archives[name],
                    name,
                    None,
                    args.upload_timeout,
                )
                _assert_update(uploads[name], cleared=True)
                samples[name] = _sample(args.base_url, name,
                                        args.request_timeout)

            samples["base_final"] = _sample(args.base_url, MODEL, args.request_timeout)

            slots = {
                name: {
                    int(update["slot"])
                    for update in _updates(response.get("moe_update"))
                }
                for name, response in uploads.items()
            }

            distances = {
                "base_zero": _distance(samples["base_initial"], samples["zero"]),
                "base_expert_a": _distance(samples["base_initial"], samples["expert_a"]),
                "expert_a_b": _distance(samples["expert_a"], samples["expert_b"]),
                "isolated_mixed_base": _distance(
                    samples["base_initial"], samples["mixed_base"]),
                "isolated_mixed_expert_a": _distance(
                    samples["expert_a"], samples["mixed_expert_a"]),
                "isolated_mixed_expert_b": _distance(
                    samples["expert_b"], samples["mixed_expert_b"]),
                "base_router": _distance(samples["base_initial"], samples["router"]),
                "base_clear": _distance(samples["base_initial"], samples["clear"]),
                "base_final": _distance(samples["base_initial"], samples["base_final"]),
            }
            checks = {
                "zero_adapter_parity": distances["base_zero"] <= args.parity_atol,
                "expert_adapter_changes_output": distances["base_expert_a"] > args.effect_atol,
                "expert_replacement_changes_output": distances["expert_a_b"] > args.effect_atol,
                "resident_experts_use_distinct_slots": slots["expert_a"] != slots["expert_b"],
                "mixed_base_parity": distances["isolated_mixed_base"] <= args.parity_atol,
                "mixed_expert_a_parity": distances["isolated_mixed_expert_a"] <= args.parity_atol,
                "mixed_expert_b_parity": distances["isolated_mixed_expert_b"] <= args.parity_atol,
                "router_lora_changes_output": distances["base_router"] > args.effect_atol,
                "clear_adapter_parity": distances["base_clear"] <= args.parity_atol,
                "base_immutable_after_swaps": distances["base_final"] <= args.parity_atol,
            }
            result.update(
                {
                    "uploads": uploads,
                    "slots": {key: sorted(value) for key, value in slots.items()},
                    "fingerprints": {key: _fingerprint(value) for key, value in samples.items()},
                    "distances": distances,
                    "checks": checks,
                    "samples": samples,
                    "acceptance_pass": all(checks.values()),
                }
            )
            if not result["acceptance_pass"]:
                raise AssertionError(f"acceptance checks failed: {checks}")
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.time() - started
        Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--result", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--tpu-inference-commit", required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--parity-atol", type=float, default=1e-6)
    parser.add_argument("--effect-atol", type=float, default=1e-5)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--upload-timeout", type=float, default=3600)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps({key: result[key] for key in ("acceptance_pass", "zone", "distances", "checks")}, indent=2))


if __name__ == "__main__":
    main()
