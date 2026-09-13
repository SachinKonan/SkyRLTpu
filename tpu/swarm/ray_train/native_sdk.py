"""Opt-in fields for pinned Tinker 0.22.7 JSON sampling; no shared SDK changes."""
from pathlib import Path
import hashlib
import importlib.metadata
import json

PATCHES = {'types/_pydantic_types/sampling_params.py': {'before': '3f23917fd7d37cd2ae8a788e6b2c4eb841e0bf4de97e0c222971f4b2f47d9815', 'after': '5dd7b971f5564b8e0bc92d09c13360d2df78c69a3cc5ec891df615fcd8fa1da8', 'old': 'class SamplingParams(BaseModel):', 'new': 'class SamplingParams(BaseModel):\n    thinking_token_budget: Optional[int] = None'}, 'types/_pydantic_types/sampled_sequence.py': {'before': '2b061f92be0897cc662f9b566dfd025ee1e9afdc6db9018625c9f9fc8da1ff8f', 'after': '3ad5ddd1623751da862a4e5dff37b0e67350bd8334f378aa02b1c4c53c2dff88', 'old': 'class SampledSequence(BaseModel):', 'new': 'class SampledSequence(BaseModel):\n    loss_mask: Optional[List[float]] = None\n    thinking_budget: Optional[dict] = None'}, 'types/sampled_sequence.py': {'before': '152039982be568cbad7dd09302c3542af8a6cea01772d3230195ed18c2d41efa', 'after': '59de9bfa1ff1733906be72d69b83b2bc2ac69bad1e4282f9e893c808984d9d07', 'old': '    stop_reason: StopReason', 'new': '    loss_mask: Optional[List[float]] = field(default=None, kw_only=True)\n    thinking_budget: Optional[dict] = field(default=None, kw_only=True)\n\n    stop_reason: StopReason'}, 'lib/_pydantic_conv.py': {'before': 'cadf3bab128c0fdf4703b161ead21e444a225564d1f68cf3f790f1f97c7e9cdd', 'after': '362d4b4722c1f3ba966642ef372a545e6ed07e54d6218635a4403d93b4e4924d', 'old': '                _logprobs_list=s.logprobs,', 'new': '                _logprobs_list=s.logprobs,\n                loss_mask=s.loss_mask,\n                thinking_budget=s.thinking_budget,'}}


def install(root=None):
    if root is None:
        import tinker
        if importlib.metadata.version("tinker") != "0.22.7":
            raise RuntimeError("native SDK extension requires pinned tinker 0.22.7")
        root = Path(tinker.__file__).parent
    root = Path(root)
    pending = []
    for name, patch in PATCHES.items():
        path = root / name
        text = path.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest == patch["after"]:
            continue
        if digest != patch["before"] or text.count(patch["old"]) != 1:
            raise RuntimeError("unexpected native SDK source: " + name)
        pending.append((path, text.replace(patch["old"], patch["new"])))
    for path, text in pending:
        temp = path.with_suffix(".native.tmp")
        temp.write_text(text)
        temp.replace(path)
    print("native SDK request fields and loss-mask transport verified", flush=True)


if __name__ == "__main__":
    install()
