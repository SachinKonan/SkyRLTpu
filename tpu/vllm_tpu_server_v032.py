"""vllm_tpu_server.py on the stock vllm-tpu 0.32 runtime (inference.serving_runtime v0.32).

vLLM 0.32 moved make_arg_parser from vllm.entrypoints.openai.cli_args to
vllm.entrypoints.launchers.cli_args. Alias the old module, then run the unchanged
server, so v0.23 farms keep their server file and compatibility hash.
"""
import runpy
import sys
from pathlib import Path

from vllm.entrypoints.launchers import cli_args

sys.modules.setdefault("vllm.entrypoints.openai.cli_args", cli_args)
runpy.run_path(str(Path(__file__).with_name("vllm_tpu_server.py")), run_name="__main__")
