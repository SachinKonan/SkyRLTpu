#!/usr/bin/env python3
"""Compare classic Tinker/JAX sampling against direct vLLM sampling."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

from transformers import AutoTokenizer


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} failed: {e.code} {e.reason}: {body}") from e
    return json.loads(body) if body else {}


def sample_tinker(
    base_url: str,
    model_name: str,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    future = post_json(
        base_url,
        "/api/v1/asample",
        {
            "base_model": model_name,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_ids}]},
            "num_samples": 1,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            },
        },
    )
    return post_json(base_url, "/api/v1/retrieve_future", {"request_id": future["request_id"]})


def sample_vllm(
    base_url: str,
    model_name: str,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    return post_json(
        base_url,
        "/v1/completions",
        {
            "model": model_name,
            "prompt": prompt_ids,
            "n": 1,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "return_token_ids": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--classic-url", default="http://localhost:8000")
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--prompt", default="Solve this arithmetic problem. What is 17 + 25?")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)

    started = time.perf_counter()
    classic = sample_tinker(
        args.classic_url,
        args.model,
        prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    classic_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    vllm = sample_vllm(
        args.vllm_url,
        args.model,
        prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    vllm_elapsed = time.perf_counter() - started

    classic_tokens = classic["sequences"][0]["tokens"]
    vllm_tokens = vllm["choices"][0]["token_ids"]
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": prompt_ids,
        "classic": {
            "elapsed_sec": round(classic_elapsed, 3),
            "tokens": classic_tokens,
            "text": tokenizer.decode(classic_tokens, skip_special_tokens=True),
            "stop_reason": classic["sequences"][0].get("stop_reason"),
        },
        "vllm": {
            "elapsed_sec": round(vllm_elapsed, 3),
            "tokens": vllm_tokens,
            "text": tokenizer.decode(vllm_tokens, skip_special_tokens=True),
            "finish_reason": vllm["choices"][0].get("finish_reason"),
        },
        "exact_token_match": classic_tokens == vllm_tokens,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
