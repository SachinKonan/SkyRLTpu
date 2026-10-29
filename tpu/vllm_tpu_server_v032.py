"""vllm_tpu_server.py (or the native thinking-budget server) on the vllm-tpu 0.32 runtime (serving_runtime v0.32).

vLLM 0.32 moved make_arg_parser from vllm.entrypoints.openai.cli_args to
vllm.entrypoints.launchers.cli_args. Alias the old module, then run the unchanged
server, so v0.23 farms keep their server files and compatibility hash. Engines with
the native thinking budget (SKYRL_THINKING_FORMAT set) run tpu/thinking_budget/server.py.
"""
import os
import runpy
import sys
from pathlib import Path

from vllm.entrypoints.launchers import cli_args

sys.modules.setdefault("vllm.entrypoints.openai.cli_args", cli_args)
server = "thinking_budget/server.py" if os.environ.get("SKYRL_THINKING_FORMAT") else "vllm_tpu_server.py"
runpy.run_path(str(Path(__file__).parent / server), run_name="__main__")
