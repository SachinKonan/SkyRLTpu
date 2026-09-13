"""Opt-in TPU server; leaves existing training launches unchanged."""
import os
import sys
from pathlib import Path

# Direct file execution must import the same isolated source, not the submitter.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# This entrypoint is explicitly experimental. Standard launches do not opt in.
os.environ["SKYRL_TPU_THINKING_BUDGET"] = "1"

from transformers import AutoTokenizer

from tpu import vllm_tpu_server as server
from tpu_inference.runner.thinking_budget_api import ThinkingBudgetApp

original_build_app = server.build_app


def build_app(args):
    app = original_build_app(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    additional = args.additional_config or {}
    supported = not args.speculative_config and not additional.get("enable_continue_decode", False)
    return ThinkingBudgetApp(app, tokenizer=tokenizer,
                             max_model_len=args.max_model_len, supported=supported,
                             thinking_format=os.environ["SKYRL_THINKING_FORMAT"])


if __name__ == "__main__":
    server.build_app = build_app
    server.main()
