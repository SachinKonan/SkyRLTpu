"""Minimal vLLM ``/v1/completions`` sampler for the probe.

Deliberately not ``skyrl.backends.vllm_sampling``: that client is coupled to
the training stack (LoRA push/pull, tinker session ids) and the probe stands
up no training stack at all. What is reused is its two load-bearing lessons:
no per-request ``seed`` (the tpu-inference backend rejects it with "JAX does
not support per-request seed"), and grouped ``n`` in ONE request so the
shared prompt prefix is prefilled once.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_program(text: str) -> tuple[str, str]:
    """(code, how). Take the LAST fenced python block — models routinely
    quote the reference implementation first and give their answer last.
    Falls back to an unterminated final fence, then to the raw text if it
    looks like a program at all.
    """
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return blocks[-1].strip(), "fenced"
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S)
    if m and "def " in m.group(1):
        return m.group(1).strip(), "unterminated-fence"
    if re.search(r"^\s*(import |from |def kernel)", text, re.M):
        return text.strip(), "bare"
    return "", "none"


class VllmSampler:
    def __init__(self, base_url: str, model: str, *, timeout_s: float = 1800.0):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as r:
            return json.loads(r.read())

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/v1/models", timeout=15) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_ready(self, deadline_s: float, poll_s: float = 15.0) -> bool:
        end = time.time() + deadline_s
        while time.time() < end:
            if self.ready():
                return True
            time.sleep(poll_s)
        return False

    def sample_group(
        self,
        prompt: str,
        n: int,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        stop: str | None = None,
        retries: int = 2,
    ) -> list[dict]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            # our text already carries every special token the renderer emits
            "add_special_tokens": False,
        }
        if stop:
            payload["stop"] = [stop]
        last = None
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                resp = self._post("/v1/completions", payload)
                dt = time.time() - t0
                out = []
                for ch in resp.get("choices", []):
                    text = ch.get("text") or ""
                    code, how = extract_program(text)
                    out.append(
                        {
                            "text": text,
                            "code": code,
                            "extraction": how,
                            "finish_reason": ch.get("finish_reason"),
                            "gen_s": dt / max(1, len(resp.get("choices", []))),
                        }
                    )
                return out
            except Exception as e:  # a 500 from a busy engine is survivable
                last = e
                time.sleep(5 * (attempt + 1))
        return [{"text": "", "code": "", "extraction": "error", "finish_reason": f"error:{last!r}", "gen_s": 0.0}] * n
