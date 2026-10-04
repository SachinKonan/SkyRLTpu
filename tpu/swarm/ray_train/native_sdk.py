"""Opt-in fields for pinned Tinker 0.22.7 JSON sampling; no shared SDK changes."""
from pathlib import Path
import hashlib
import importlib.metadata
import json

PATCHES = {'types/_pydantic_types/sampling_params.py': {'before': '3f23917fd7d37cd2ae8a788e6b2c4eb841e0bf4de97e0c222971f4b2f47d9815', 'after': '5dd7b971f5564b8e0bc92d09c13360d2df78c69a3cc5ec891df615fcd8fa1da8', 'old': 'class SamplingParams(BaseModel):', 'new': 'class SamplingParams(BaseModel):\n    thinking_token_budget: Optional[int] = None'}, 'types/_pydantic_types/sampled_sequence.py': {'before': '2b061f92be0897cc662f9b566dfd025ee1e9afdc6db9018625c9f9fc8da1ff8f', 'after': '268fc8d25387daa8b16f9381ce628ee345289f597bbaa7be252ba39b953fc11a', 'old': 'class SampledSequence(BaseModel):', 'new': 'class SampledSequence(BaseModel):\n    loss_mask: Optional[List[float]] = None\n    thinking_budget: Optional[dict] = None\n    served_by: Optional[dict] = None'}, 'types/sampled_sequence.py': {'before': '152039982be568cbad7dd09302c3542af8a6cea01772d3230195ed18c2d41efa', 'after': '92eb26f3ff225c9cd2e695f03b5209bbfba0160f613dbac97b095ce4d7556c9c', 'old': '    stop_reason: StopReason', 'new': '    loss_mask: Optional[List[float]] = field(default=None, kw_only=True)\n    thinking_budget: Optional[dict] = field(default=None, kw_only=True)\n    served_by: Optional[dict] = field(default=None, kw_only=True)\n\n    stop_reason: StopReason'}, 'lib/_pydantic_conv.py': {'before': 'cadf3bab128c0fdf4703b161ead21e444a225564d1f68cf3f790f1f97c7e9cdd', 'after': '631af191839c6ca33789f4ef0f8dbb2654e03172786a1b295a4458f0ffd1fcd1', 'old': '                _logprobs_list=s.logprobs,', 'new': '                _logprobs_list=s.logprobs,\n                loss_mask=s.loss_mask,\n                thinking_budget=s.thinking_budget,\n                served_by=s.served_by,'}}


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
    print("native SDK request fields and loss-mask/served-by transport verified", flush=True)


if __name__ == "__main__":
    install()
