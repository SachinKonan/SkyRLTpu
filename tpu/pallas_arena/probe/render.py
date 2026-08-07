"""Chat rendering for the two probe models, plus a server-side check that it
is right.

We render TEXT ourselves and post it to ``/v1/completions`` with
``add_special_tokens: false`` rather than letting ``/v1/chat/completions``
apply the model's template, for one measured reason: the repo's
``Gemma4Renderer`` emits ``<bos>`` explicitly because the gemma tokenizer's
``add_special_tokens=True`` inserts nothing, and a BOS-less gemma prompt
"silently collapses generation quality on hard prompts (measured: 1/32 vs 7/8
compiling programs on the same prompt)". A probe that silently lost that
would be measuring the harness, not the model.

``verify_against_server`` proves the rendering is right without needing a
local tokenizer: vLLM's ``/tokenize`` endpoint will apply the model's OWN
chat template to a message list, and will separately tokenize our rendered
string. If the two token id lists agree, our text is byte-equivalent to the
template the model was trained with.
"""

from __future__ import annotations

import json
import urllib.request

# --------------------------------------------------------------------------
# Qwen3 (ttt_discover.tinker_utils.renderers.Qwen3Renderer)
#   prefix  = f"{'' if idx == 0 else chr(10)}<|im_start|>{role}\n"
#   content = msg + "<|im_end|>"
#   an assistant turn with no <think> gets "<think>\n" appended to its PREFIX,
#   so the generation prompt forces the thinking channel open.
QWEN3_STOP = "<|im_end|>"


def render_qwen3(user_prompt: str) -> str:
    return "<|im_start|>user\n" + user_prompt + "<|im_end|>" + "\n<|im_start|>assistant\n<think>\n"


# --------------------------------------------------------------------------
# Gemma 4 (ttt_discover.tinker_utils.renderers.Gemma4Renderer)
#   <bos> is explicit; the thinking-mode system turn is injected even when the
#   conversation has no system message; the generation prompt is a bare
#   "<|turn>model\n" and the MODEL opens <|channel>thought itself.
GEMMA4_STOP = "<turn|>"
_GEMMA4_THINK_SYSTEM_HEADER = "<|turn>system\n<|think|>\n"


def render_gemma4(user_prompt: str) -> str:
    return (
        "<bos>"
        + _GEMMA4_THINK_SYSTEM_HEADER
        + "<turn|>\n"
        + "<|turn>user\n"
        + user_prompt
        + "<turn|>\n"
        + "<|turn>model\n"
    )


RENDERERS = {"qwen3": render_qwen3, "gemma4": render_gemma4}
STOPS = {"qwen3": QWEN3_STOP, "gemma4": GEMMA4_STOP}


def render(renderer: str, user_prompt: str) -> str:
    return RENDERERS[renderer](user_prompt)


# ------------------------------------------------------------- verification
def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def verify_against_server(base_url: str, model: str, renderer: str, probe_text: str = "PROBE-RENDER-CHECK") -> dict:
    """Tokenize (a) our rendered prompt and (b) the same conversation through
    the server's own chat template, and compare the ids.

    Returns a report dict; never raises. `agree` False is not automatically
    fatal (gemma's <bos> is a deliberate, measured divergence from the naive
    template path) — it is reported, with the diff, so the run says which
    rendering it used.
    """
    base = base_url.rstrip("/")
    ours = render(renderer, probe_text)
    out: dict = {"model": model, "renderer": renderer, "rendered_prefix": ours[:200], "rendered_len": len(ours)}
    try:
        a = _post(f"{base}/tokenize", {"model": model, "prompt": ours, "add_special_tokens": False})
        out["ours_tokens"] = a.get("count")
        out["ours_ids"] = a.get("tokens")
    except Exception as e:
        out["error_ours"] = repr(e)
        return out
    try:
        b = _post(
            f"{base}/tokenize",
            {
                "model": model,
                "messages": [{"role": "user", "content": probe_text}],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        out["template_tokens"] = b.get("count")
        out["template_ids"] = b.get("tokens")
    except Exception as e:
        out["error_template"] = repr(e)
    ours_ids, tmpl_ids = out.get("ours_ids"), out.get("template_ids")
    if isinstance(ours_ids, list) and isinstance(tmpl_ids, list):
        out["agree"] = ours_ids == tmpl_ids
        if not out["agree"]:
            n = min(len(ours_ids), len(tmpl_ids))
            first = next((i for i in range(n) if ours_ids[i] != tmpl_ids[i]), n)
            out["first_divergence_at"] = first
            out["ours_head"] = ours_ids[: first + 4]
            out["template_head"] = tmpl_ids[: first + 4]
    # The single fact that must hold no matter what: the special tokens the
    # renderer relies on must each be ONE token.
    singles = {}
    for tok in (["<|im_start|>", "<|im_end|>", "<think>"] if renderer == "qwen3"
                else ["<bos>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>"]):
        try:
            r = _post(f"{base}/tokenize", {"model": model, "prompt": tok, "add_special_tokens": False})
            singles[tok] = r.get("count")
        except Exception as e:
            singles[tok] = repr(e)
    out["special_token_counts"] = singles
    out["special_tokens_all_single"] = all(v == 1 for v in singles.values())
    return out
