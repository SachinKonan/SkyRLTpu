"""The grader entry point must survive a trailing usage snippet (gpt-oss job 578).

The function is loaded from source with `ast` so the test does not import the
ttt_discover package (its import chain needs the client venv)."""
import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "third_party/discover/ttt_discover/tinker_utils/dataset_builder.py"


def _load():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "last_codeblock_postprocess")
    ns = {"re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["last_codeblock_postprocess"]


PROGRAM = "import numpy as np\n\ndef helper(x):\n    return x\n\ndef run(seed=42, budget_s=1000, **kwargs):\n    return [0.5], 0.4, 2"
SNIPPET = "h, c5, n = run(seed=1)\nprint(c5)"


def test_prefers_last_block_defining_run():
    f = _load()
    text = f"Here is the program:\n```python\n{PROGRAM}\n```\nUsage:\n```python\n{SNIPPET}\n```\n"
    assert f(text, keep_separators=False) == PROGRAM


def test_falls_back_to_last_block_without_run():
    f = _load()
    text = f"```python\nx = 1\n```\n```python\n{SNIPPET}\n```"
    assert f(text, keep_separators=False) == SNIPPET
