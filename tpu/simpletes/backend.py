"""SimpleTES LLM backend on our native-thinking ingress.

Implements SimpleTES's ``LLMBackend`` protocol (generate / generate_batch / close)
with the RQ4 frozen client's generation stack instead of LiteLLM:

* prompt tokens from the RL renderer (``ttt_discover.tinker_utils.renderers``),
  so chat framing, Qwen's ``<think>`` prefill and Muse's system block are the RL
  bytes (the Muse date is pinned per run so prefix caching holds across days);
* the request shape of the frozen client (temperature 1, top_p 1, top_k off,
  native ``thinking_token_budget``) and ``frozen_benchmark.check_choice``;
* the RQ4 caps, relative to each prompt exactly as the native completer computes
  them: thinking = phase1_max_tokens - prompt, total = context_window - prompt - 50;
* the RQ4 answer rule (``answer_only_code``): the last ```python block of the answer
  channel is the candidate, so a SimpleTES grade and an RQ4 grade see the same code.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

SEPARATORS = {"qwen": "</think>", "gemma": "<channel|>"}
# 409/429: the remote-only ingress is waiting for a farm (the frozen client retries these too).
RETRYABLE = (409, 429, 500, 502, 503, 504)
MIN_AUX_THINKING = 1024
# The SimpleTES batch (gen_id) a generate call serves; set by the scaffold arm, logged with each call.
GEN_ID: contextvars.ContextVar = contextvars.ContextVar("simpletes_gen_id", default=None)
# Thinking budget 0 + this transition: the server closes the thinking block at once without the
# "final complete program" cue, so an auxiliary call (reflection) answers in prose.
CLOSE_THINKING = {"qwen": "\n</think>\n\n", "gemma": "<channel|>",
                  "muse": "<|eom|><|start|>assistant to=user<|message|>"}
# An engine restart outlasts minutes; a batch that gives up is charged K failed candidates.
# If the engines are gone for good the controller fails the job and the resume re-dispatches.
DEFAULT_RETRIES, DEFAULT_RETRY_SECONDS = 480, 30.0   # ~4 h


@dataclass(frozen=True)
class Caps:
    """RQ4 generation limits; the per-request caps depend on the prompt length."""
    context_window: int       # client_context_window
    phase1_max_tokens: int    # client_phase1_max_tokens (counts the prompt)
    context_buffer: int = 50  # NativeCompletionCompleter.context_buffer
    fixed_thinking: int | None = None  # TTD_FIXED_THINK_TOKENS
    fixed_total: int | None = None     # TTD_FIXED_TOTAL_TOKENS

    def max_prompt_tokens(self, min_thinking_tokens: int) -> int:
        if self.fixed_thinking:
            return self.context_window - self.fixed_total - self.context_buffer
        return self.phase1_max_tokens - min_thinking_tokens

    def budget(self, prompt_tokens: int, transition_tokens: int = 0) -> tuple[int, int] | None:
        """(thinking_token_budget, max_tokens) for this prompt, or None if it leaves no room.

        Same arithmetic as ``completers._think_budget`` / ``_output_limit``: with a fixed budget,
        thinking and total ignore the prompt, which must leave room for the total; otherwise
        thinking = phase1 - prompt, total = context - prompt - buffer, and the forced transition
        plus one answer token must fit after a full thinking budget.
        """
        if self.fixed_thinking:
            if prompt_tokens + self.fixed_total + self.context_buffer > self.context_window:
                return None
            return self.fixed_thinking, self.fixed_total
        thinking = self.phase1_max_tokens - prompt_tokens
        total = self.context_window - prompt_tokens - self.context_buffer
        if thinking <= 0 or total < thinking + transition_tokens + 1:
            return None
        return thinking, total


def family_of(renderer_name: str) -> str:
    if renderer_name.startswith("qwen"):
        return "qwen"
    if renderer_name.startswith("gemma"):
        return "gemma"
    if renderer_name.startswith("muse"):
        return "muse"
    raise ValueError(f"unsupported renderer for SimpleTES: {renderer_name}")


def make_renderer(renderer_name: str, tokenizer, renderer_date: str | None):
    from ttt_discover.tinker_utils import renderers
    if renderer_name == "muse_glimmer_high_reasoning":
        return renderers.MuseGlimmerRenderer(tokenizer, reasoning_strength="high", current_date=renderer_date)
    return renderers.get_renderer(renderer_name, tokenizer)


def split_answer(message: dict, family: str) -> str:
    """Answer channel only, so SimpleTES never extracts a draft from the reasoning."""
    content = message.get("content") or ""
    separator = SEPARATORS.get(family)
    if separator and separator in content:
        return content.rsplit(separator, 1)[-1]
    return content


def answer_code(message: dict, family: str) -> str:
    """The RQ4 candidate: ``answer_only_code`` without fences (science envs drop separators)."""
    from ttt_discover.tinker_utils.dataset_builder import answer_only_code
    return answer_only_code(message, family, keep_separators=False)


def as_evolve_block(code: str) -> str:
    """Wrap the RQ4 candidate as SimpleTES's whole EVOLVE block (tasks.init_program has no fixed parts).

    Marker lines the model wrote inside its block are dropped, so the program SimpleTES
    reconstructs is START + code + END, and the evaluator grades exactly ``code``.
    """
    from .tasks import EVOLVE_END, EVOLVE_START
    body = "\n".join(line for line in code.splitlines() if line.strip() not in (EVOLVE_START, EVOLVE_END))
    return f"```python\n{EVOLVE_START}\n{body.strip()}\n{EVOLVE_END}\n```"


def raw_text(message: dict) -> str:
    if message.get("thinking"):
        return f"<thinking>{message['thinking']}</thinking>\n{message.get('content') or ''}"
    return message.get("content") or ""


class NativeThinkingBackend:
    def __init__(self, *, base_url: str, model: str, renderer, family: str, caps: Caps,
                 log_path: Path, request_timeout: float = 21600, retries: int = DEFAULT_RETRIES,
                 retry_seconds: float = DEFAULT_RETRY_SECONDS, check_choice=None, http=None,
                 transition_tokens: int = 0):
        import httpx
        from .compat import import_simpletes
        import_simpletes()   # vendored package first: `tpu/` on sys.path would shadow `simpletes`
        from simpletes.llm.types import LLMCallError, LLMResult
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.renderer = renderer
        self.family = family
        self.caps = caps
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.retry_seconds = retry_seconds
        self.transition_tokens = transition_tokens
        self._check = check_choice
        self._http = http or httpx.AsyncClient(timeout=request_timeout)
        self._result, self._error = LLMResult, LLMCallError
        self.stop_ids = list(renderer.get_stop_sequences())

    # ---- request construction -------------------------------------------------
    def prompt_ids(self, prompt: str) -> list[int]:
        rendered = self.renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        return rendered.to_ints()

    def request(self, ids: list[int], n: int, *, thinking_tokens: int, max_tokens: int,
                temperature: float = 1.0, transition: str | None = None) -> dict:
        request = dict(model=self.model, prompt=ids, n=n, max_tokens=max_tokens,
                       thinking_token_budget=thinking_tokens,
                       temperature=temperature, top_p=1.0, top_k=-1, stop_token_ids=self.stop_ids,
                       skip_special_tokens=False, add_special_tokens=False, logprobs=1,
                       stream=False, return_token_ids=True)
        if transition is not None:
            request["forced_thinking_transition"] = transition
        return request

    async def _post(self, request: dict) -> dict:
        import httpx
        last = None
        for attempt in range(self.retries + 1):
            try:
                response = await self._http.post(self.base_url + "/v1/completions", json=request)
                if response.status_code in RETRYABLE:
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                else:
                    response.raise_for_status()
                    return response.json()
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                await asyncio.sleep(self.retry_seconds)
        raise self._error(model=self.model, api_base=self.base_url, error_type="ingress_unavailable",
                          message=str(last))

    async def _complete(self, ids: list[int], n: int, thinking_tokens: int, max_tokens: int,
                        temperature: float, transition: str | None = None) -> tuple[dict | None, list[dict]]:
        """One sampling call; returns (request for check_choice, choices sorted by index)."""
        request = self.request(ids, n, thinking_tokens=thinking_tokens, max_tokens=max_tokens,
                               temperature=temperature, transition=transition)
        data = await self._post(request)
        return request, sorted(data.get("choices", []), key=lambda c: c.get("index", 0))

    def _log(self, **record):
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    # ---- SimpleTES LLMBackend protocol --------------------------------------------
    async def generate(self, prompt: str, instance_id: str = "", track_io: bool = False):
        return (await self.generate_batch(prompt, 1, instance_id=instance_id, track_io=track_io))[0]

    async def generate_batch(self, prompt: str, n: int, instance_id: str = "", track_io: bool = False):
        ids = self.prompt_ids(prompt)
        start = time.time()
        budget = self.caps.budget(len(ids), self.transition_tokens)
        if budget is None:
            self._log(kind="generate", time=start, n=n, gen_id=GEN_ID.get(), prompt_tokens=len(ids), overflow=True)
            return [self._result(text="", prompt=prompt if track_io else None,
                                 token_usage=dict(prompt_tokens=len(ids), completion_tokens=0, total_tokens=len(ids)),
                                 error_reason="prompt_overflow") for _ in range(n)]
        thinking, total = budget
        request, choices = await self._complete(ids, n, thinking, total, 1.0)
        if len(choices) != n:
            raise self._error(model=self.model, api_base=self.base_url, error_type="short_batch",
                              message=f"expected {n} choices, got {len(choices)}")
        results, usage_log = [], []
        for choice in choices:
            try:
                if self._check is not None:
                    self._check(choice, request)
                message, _ = self.renderer.parse_response(choice["token_ids"])
            except Exception as exc:
                # One malformed choice fails that candidate only, not its K-1 siblings.
                usage_log.append(dict(prompt_tokens=len(ids), completion_tokens=len(choice.get("token_ids") or []),
                                      bad_choice=f"{type(exc).__name__}: {exc}"[:300]))
                results.append(self._result(text="", token_usage=usage_log[-1],
                                            error_reason=f"bad_choice: {type(exc).__name__}"))
                continue
            code = answer_code(message, self.family)
            audit = choice.get("thinking_budget") or {}
            usage = dict(prompt_tokens=len(ids), completion_tokens=len(choice["token_ids"]),
                         total_tokens=len(ids) + len(choice["token_ids"]),
                         thinking_budget=thinking, max_tokens=total,
                         reasoning_tokens=audit.get("counted_phase1_tokens"),
                         forced=bool(choice.get("forced_token_positions")),
                         finish_reason=choice.get("finish_reason"), no_code=not code.strip())
            usage_log.append(usage)
            # No python block in the answer: RQ4 grades empty code (reward 0); SimpleTES gets an
            # answer without markers and records a failed candidate.
            text = as_evolve_block(code) if code.strip() else split_answer(message, self.family)
            results.append(self._result(text=text,
                                        prompt=prompt if track_io else None,
                                        raw_output=raw_text(message) if track_io else None,
                                        token_usage=usage))
        self._log(kind="generate", time=start, seconds=time.time() - start, n=n, gen_id=GEN_ID.get(),
                  prompt_tokens=len(ids), thinking_budget=thinking, max_tokens=total, choices=usage_log)
        return results

    async def generate_text(self, prompt: str, *, thinking_tokens: int, answer_tokens: int,
                            temperature: float, kind: str) -> str:
        """Single auxiliary call (SimpleTES reflection); logged separately, never graded.

        ``thinking_tokens`` 0: no thinking. The server closes the thinking block at once
        (``CLOSE_THINKING``, no program cue), the answer gets ``answer_tokens``, and the prompt
        only needs room for that answer. Otherwise the generation caps, with the thinking budget
        lowered to ``thinking_tokens`` when that is smaller and the total cut to thinking + answer.
        """
        ids = self.prompt_ids(prompt)
        transition = None
        if thinking_tokens == 0:
            transition = CLOSE_THINKING[self.family]
            thinking, total = 0, answer_tokens
            if len(ids) + total + self.caps.context_buffer > self.caps.context_window:
                self._log(kind=kind, time=time.time(), prompt_tokens=len(ids), overflow=True)
                return ""
        else:
            budget = self.caps.budget(len(ids), self.transition_tokens)
            if budget is None:
                self._log(kind=kind, time=time.time(), prompt_tokens=len(ids), overflow=True)
                return ""
            thinking = min(thinking_tokens, budget[0])
            total = min(budget[1], thinking + answer_tokens)
        start = time.time()
        request, choices = await self._complete(ids, 1, thinking, total, temperature, transition)
        choice = choices[0]
        if self._check is not None:
            self._check(choice, request)
        message, _ = self.renderer.parse_response(choice["token_ids"])
        self._log(kind=kind, time=start, seconds=time.time() - start, prompt_tokens=len(ids),
                  thinking_budget=thinking, max_tokens=total, completion_tokens=len(choice["token_ids"]),
                  finish_reason=choice.get("finish_reason"))
        return split_answer(message, self.family)

    async def aclose(self):
        await self._http.aclose()

    def close(self) -> None:
        pass
