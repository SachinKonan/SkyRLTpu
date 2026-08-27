"""One way to find the ANSWER in a completion: ttt_discover's RosettaStone.

The arena had grown its own extraction heuristics because it consumes the
vLLM /v1/chat/completions endpoint, which returns only the generated
continuation. Rosetta expects a full assistant turn, so it silently matched
nothing here (measured 2026-08-27: 0/32 qwen, 0/10 gemma) while working
correctly everywhere else in the codebase -- the renderer path
(renderers.py parse_response) and the muse pipeline (rs_generate) both hand
it a locally decoded token stream with every marker intact.

Two DIFFERENT reasons the markers went missing, both confirmed from the
model files rather than inferred:

  qwen3.5  <think>/</think> are added tokens with special=False, so they are
           never stripped -- but the chat template PREFILLS '<think>\\n' into
           the prompt (chat_template.jinja:152), so the completion carries
           only the closing tag. Re-attach the opener and Rosetta matches.

  gemma-4  '<|channel>' (id 100) and '<channel|>' (id 101) are added tokens
           with special=True, so skip_special_tokens strips them and leaves
           the bare word 'thought'. The sampler must ask for them to be kept;
           nothing can be reconstructed after the fact.

This module owns ONLY the arena-side reconstitution. The splitting itself
stays in rosetta_stone so the arena and the training paths can never
disagree about where reasoning ends.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_RS_PATH = (pathlib.Path(__file__).resolve().parents[3]
            / "third_party/discover/ttt_discover/tinker_utils/rosetta_stone.py")

_rs = None


def _rosetta():
    """Load rosetta_stone by PATH -- importing ttt_discover pulls in chz."""
    global _rs
    if _rs is None:
        spec = importlib.util.spec_from_file_location("rosetta_stone", _RS_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rosetta_stone"] = mod      # dataclasses needs it registered
        spec.loader.exec_module(mod)
        _rs = mod
    return _rs


def family_of(model: str) -> str:
    """Rosetta family for a served model name."""
    m = (model or "").lower()
    if "qwen" in m:
        return "qwen3"
    if "gemma" in m:
        return "gemma4"
    if "gpt-oss" in m or "gpt_oss" in m:
        return "gpt_oss"
    if "muse" in m or "glimmer" in m:
        return "muse_glimmer"
    return "plain"


def reconstitute(text: str, family: str) -> str:
    """Put back what the serving path removed, so Rosetta sees a full turn."""
    if family == "qwen3" and "</think>" in text and "<think>" not in text:
        return "<think>\n" + text          # the opener lives in the prompt
    return text


def answer(text: str, model_or_family: str) -> tuple[str, bool]:
    """(answer_text, split_happened).

    Returns the post-reasoning content when Rosetta can find the boundary,
    else the untouched text so callers keep their existing fallback. Never
    raises: a parse failure must not lose a candidate.
    """
    fam = model_or_family if model_or_family in (
        "qwen3", "gemma4", "gpt_oss", "muse_glimmer", "plain") else family_of(model_or_family)
    if fam == "plain":
        return text, False
    try:
        parsed = _rosetta().parse(reconstitute(text, fam), fam)
    except Exception:                       # noqa: BLE001 -- never lose a candidate
        return text, False
    if parsed.thinking is None:
        return text, False
    return parsed.content, True
