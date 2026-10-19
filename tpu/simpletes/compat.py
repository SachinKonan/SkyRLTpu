"""Import SimpleTES inside the locked RL client environment without changing it.

The client lock (tpu/swarm/ray_train/client_env/uv.lock) has rich and pandas but
not litellm or matplotlib. SimpleTES imports litellm at package import time only
to build its default LLM client, which backend.py replaces; matplotlib is used only
for checkpoint plots. Both are stubbed/skipped here so the vendored tree stays
byte-identical to upstream.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import types
from pathlib import Path

VENDORED = Path(__file__).resolve().parents[2] / "third_party" / "simpletes"


def _stub_litellm():
    if "litellm" in sys.modules or importlib.util.find_spec("litellm") is not None:
        return   # already stubbed (idempotent) or really installed
    module = types.ModuleType("litellm")

    def unavailable(*args, **kwargs):
        raise RuntimeError("litellm is stubbed: SimpleTES must use tpu.simpletes.backend")

    for name in ("completion", "acompletion", "get_supported_openai_params", "token_counter"):
        setattr(module, name, unavailable)
    sys.modules["litellm"] = module


def _csv_only_score_statistics(nodes, output_dir, completed_evaluations, best_score):
    """CSV half of simpletes.node.save_score_statistics, without the matplotlib plots."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"scores_{completed_evaluations:06d}.csv")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "score", "created_at", "parent_ids", "gen_id", "chain_idx", "error"])
        for n in nodes:
            writer.writerow([n["id"], n["score"], n["created_at"], ";".join(n.get("parent_ids", [])),
                             n["gen_id"], n["chain_idx"], n.get("metrics", {}).get("error", "")])


def import_simpletes():
    """Put the vendored package on sys.path and return (simpletes, engine modules)."""
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    _stub_litellm()
    import simpletes  # noqa: F401
    import simpletes.engine.checkpoint as checkpoint
    if importlib.util.find_spec("matplotlib") is None:
        checkpoint.save_score_statistics = _csv_only_score_statistics
    return simpletes
