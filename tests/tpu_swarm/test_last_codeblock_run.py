"""The grader entry point must survive a trailing usage snippet (gpt-oss job 578)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/discover"))
from ttt_discover.tinker_utils.dataset_builder import last_codeblock_postprocess  # noqa: E402

PROGRAM = "import numpy as np\n\ndef helper(x):\n    return x\n\ndef run(seed=42, budget_s=1000, **kwargs):\n    return [0.5], 0.4, 2"
SNIPPET = "h, c5, n = run(seed=1)\nprint(c5)"


def test_prefers_last_block_defining_run():
    text = f"Here is the program:\n```python\n{PROGRAM}\n```\nUsage:\n```python\n{SNIPPET}\n```\n"
    out = last_codeblock_postprocess(text, keep_separators=False)
    assert out == PROGRAM


def test_falls_back_to_last_block_without_run():
    text = f"```python\nx = 1\n```\n```python\n{SNIPPET}\n```"
    assert last_codeblock_postprocess(text, keep_separators=False) == SNIPPET
