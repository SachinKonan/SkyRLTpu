import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _block(source: str, start_pat: str, end_pat: str) -> str:
    start = source.index(start_pat)
    return source[start:source.index(end_pat, start)]


def test_every_model_serves_at_090_memory_utilization():
    # User decision 2026-09-06: 0.90 everywhere (qwen already was; gemma and
    # muse were at 0.85 and gemma's KV cache was the binding constraint).
    source = (REPO / "tpu/jobman/cell_worker.sh").read_text()
    utils = re.findall(r"--gpu-memory-utilization (0\.\d+)", source)
    assert utils and set(utils) == {"0.90"}, utils


def test_gemma_max_num_seqs_matches_its_kv_budget():
    source = (REPO / "tpu/jobman/cell_worker.sh").read_text()
    gemma = _block(source, "  g-*)\n    MODEL_NAME=google/gemma-4-31B-it", "  m-*)")
    assert "MAX_NUM_SEQS=32" in gemma
    # qwen keeps the 128 default (4.3M-token KV cache), muse keeps 64
    assert "MAX_NUM_SEQS=128\n" in source
    muse = _block(source, "  m-*)", "  *)")
    assert "MAX_NUM_SEQS=64" in muse
