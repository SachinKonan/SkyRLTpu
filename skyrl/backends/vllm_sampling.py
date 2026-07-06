"""vLLM sampling client used by the JAX Tinker backend."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cloudpathlib import AnyPath

from skyrl.tinker import types
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack


def _normalize_vllm_urls(base_url: str) -> tuple[str, str]:
    """Return (server_base_url, openai_base_url)."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")], url
    return url, f"{url}/v1"


def _checkpoint_id_from_path(path: AnyPath) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    return Path(name).stem


def _sanitize_lora_name(model_id: str, checkpoint_id: str) -> str:
    return f"{model_id}_{checkpoint_id}".replace("/", "_").replace(":", "_")


@dataclass
class VllmSamplingClient:
    """Small synchronous client for vLLM's OpenAI-compatible server.

    The JAX backend is synchronous, so this intentionally uses blocking HTTP
    calls. Batch concurrency is provided with a thread pool so vLLM can still
    continuously batch independent requests.
    """

    base_url: str
    model_name: str
    api_key: str = "EMPTY"
    lora_base_dir: Path = Path("/tmp/skyrl_jax_vllm_loras")
    lora_load_endpoint: str = "/v1/load_lora_adapter"
    request_timeout_sec: float = 300.0
    max_concurrent_requests: int = 64
    _loaded_loras: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.server_base_url, self.openai_base_url = _normalize_vllm_urls(self.base_url)
        if not self.lora_load_endpoint.startswith("/"):
            self.lora_load_endpoint = f"/{self.lora_load_endpoint}"
        self.max_concurrent_requests = max(1, self.max_concurrent_requests)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={**self._headers(), **(headers or {})}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                content_type = resp.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed: {e.code} {e.reason}: {body}") from e

        if "application/json" in content_type:
            return json.loads(body) if body else {}
        return body

    def _extract_lora_checkpoint(self, checkpoint_path: AnyPath, target_dir: Path) -> None:
        if target_dir.exists():
            return

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = target_dir.with_name(f".{target_dir.name}.tmp-{os.getpid()}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        try:
            with download_and_unpack(checkpoint_path) as extracted_path:
                shutil.copytree(extracted_path, tmp_dir)
            os.replace(tmp_dir, target_dir)
        except FileExistsError:
            pass
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def ensure_lora_loaded(self, model_id: str, checkpoint_path: AnyPath, checkpoint_id: str | None = None) -> str:
        """Extract a JAX LoRA sampler checkpoint and load it into vLLM.

        Returns the vLLM model name to use for requests targeting this adapter.
        """
        checkpoint_id = checkpoint_id or _checkpoint_id_from_path(checkpoint_path)
        lora_name = _sanitize_lora_name(model_id, checkpoint_id)
        target_dir = self.lora_base_dir / lora_name
        self._extract_lora_checkpoint(checkpoint_path, target_dir)

        target_path = str(target_dir)
        if self._loaded_loras.get(lora_name) == target_path:
            return lora_name

        payload = {"lora_name": lora_name, "lora_path": target_path}
        self._post_json(f"{self.server_base_url}{self.lora_load_endpoint}", payload)
        self._loaded_loras[lora_name] = target_path
        logger.info("Loaded LoRA adapter %s into vLLM from %s", lora_name, target_path)
        return lora_name

    @staticmethod
    def _prompt_logprobs_from_response(result: dict[str, Any], prompt_ids: list[int]) -> list[float] | None:
        raw = result.get("prompt_logprobs")
        if raw is None and result.get("choices"):
            raw = result["choices"][0].get("prompt_logprobs")
        if raw is None:
            return None

        values: list[float] = []
        for token_id, pos in zip(prompt_ids, raw):
            if pos is None:
                values.append(0.0)
            elif isinstance(pos, dict):
                entry = pos.get(str(token_id)) or pos.get(token_id)
                if isinstance(entry, dict):
                    values.append(float(entry.get("logprob", 0.0)))
                elif isinstance(entry, (int, float)):
                    values.append(float(entry))
                else:
                    values.append(0.0)
            elif isinstance(pos, (int, float)):
                values.append(float(pos))
            else:
                values.append(0.0)
        return values

    @staticmethod
    def _sequence_from_choice(choice: dict[str, Any]) -> types.GeneratedSequence:
        tokens = choice.get("token_ids") or []
        lp = choice.get("logprobs") or {}
        logprobs = lp.get("token_logprobs") or []
        if not logprobs and tokens:
            content = lp.get("content") or []
            logprobs = [float(item.get("logprob", 0.0)) for item in content]
        if len(logprobs) != len(tokens):
            logprobs = [0.0] * len(tokens)

        finish_reason = choice.get("finish_reason")
        stop_reason = "stop" if finish_reason in ("stop", "stop_token") else "length"
        return types.GeneratedSequence(tokens=tokens, logprobs=logprobs, stop_reason=stop_reason)

    def sample_one(
        self,
        prompt_ids: list[int],
        sampling_params: types.SamplingParams,
        *,
        model_name: str,
        session_id: str | None = None,
        prompt_logprobs: bool = False,
    ) -> tuple[types.GeneratedSequence, list[float] | None]:
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt_ids,
            "n": 1,
            "seed": sampling_params.seed,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "logprobs": 1,
            "stream": False,
            "return_token_ids": True,
        }
        if sampling_params.stop_tokens:
            payload["stop_token_ids"] = sampling_params.stop_tokens
        if sampling_params.stop_strings:
            payload["stop"] = sampling_params.stop_strings
        if prompt_logprobs:
            payload["prompt_logprobs"] = 1

        headers = {"X-Session-ID": session_id} if session_id else None
        result = self._post_json(f"{self.openai_base_url}/completions", payload, headers=headers)
        choices = result.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError(f"Expected one vLLM completion choice, got {len(choices)}")

        prompt_lps = self._prompt_logprobs_from_response(result, prompt_ids) if prompt_logprobs else None
        return self._sequence_from_choice(choices[0]), prompt_lps

    def sample_many(
        self,
        prompt_ids: list[list[int]],
        sampling_params: list[types.SamplingParams],
        model_names: list[str],
        session_ids: list[str | None],
        *,
        prompt_logprobs: bool = False,
    ) -> tuple[list[types.GeneratedSequence], list[list[float] | None]]:
        if not (len(prompt_ids) == len(sampling_params) == len(model_names) == len(session_ids)):
            raise ValueError("Mismatched vLLM sample batch input lengths")

        def run_one(i: int) -> tuple[types.GeneratedSequence, list[float] | None]:
            return self.sample_one(
                prompt_ids[i],
                sampling_params[i],
                model_name=model_names[i],
                session_id=session_ids[i],
                prompt_logprobs=prompt_logprobs,
            )

        workers = min(self.max_concurrent_requests, len(prompt_ids))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(run_one, range(len(prompt_ids))))

        sequences, prompt_lps = zip(*outputs) if outputs else ([], [])
        return list(sequences), list(prompt_lps)
