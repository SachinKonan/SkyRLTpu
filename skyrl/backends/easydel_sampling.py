"""Small, dependency-free helpers for EasyDeL eSurge request batching."""

from __future__ import annotations

from typing import Any


def batch_esurge_generate(
    *,
    engine: Any,
    prompts: list[str],
    params_by_request: dict[str, Any],
    request_ids: list[str],
) -> dict[str, Any]:
    """Generate a heterogeneous batch while preserving per-request parameters."""
    if len(prompts) != len(request_ids) or len(request_ids) != len(params_by_request):
        raise ValueError(
            "prompts, request_ids, and params_by_request must have matching lengths"
        )
    if not request_ids:
        return {}

    original_callback = getattr(engine, "_sampling_params_callback", None)

    def request_params_callback(template, metadata):
        del template
        return params_by_request[metadata["request_id"]]

    engine._sampling_params_callback = request_params_callback
    try:
        outputs = engine.generate(
            prompts,
            sampling_params=params_by_request[request_ids[0]],
            request_id=request_ids,
            use_tqdm=False,
        )
    finally:
        engine._sampling_params_callback = original_callback

    outputs_by_request = {output.request_id: output for output in outputs}
    missing = set(request_ids).difference(outputs_by_request)
    if missing:
        raise RuntimeError(f"eSurge omitted batched requests: {sorted(missing)}")
    return outputs_by_request
