"""vLLM sampling client used by the JAX Tinker backend."""

from __future__ import annotations

import json
import os
import shutil
import time
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


class VllmRequestError(RuntimeError):
    """HTTP/transport error returned by a vLLM server."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _normalize_vllm_urls(base_url: str) -> tuple[str, str]:
    """Return (server_base_url, openai_base_url)."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")], url
    return url, f"{url}/v1"


def _normalize_vllm_url_list(base_url: str) -> tuple[tuple[str, str], ...]:
    """Return normalized URL pairs for one or more comma-separated servers."""
    urls = [url.strip() for url in base_url.split(",") if url.strip()]
    if not urls:
        raise ValueError("At least one vLLM base URL is required")
    return tuple(_normalize_vllm_urls(url) for url in urls)


def _checkpoint_id_from_path(path: AnyPath) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    return Path(name).stem


def _sanitize_lora_name(model_id: str, checkpoint_id: str) -> str:
    return f"{model_id}_{checkpoint_id}".replace("/", "_").replace(":", "_")


@dataclass
class GroupedCompletion:
    """A single grouped vLLM completion request: ``n`` samples for one prompt.

    Every sample in the group shares the same prompt, sampling params, model,
    and routing session, so vLLM can serve all ``n`` completions from a single
    request instead of ``n`` separate ones.
    """

    prompt_ids: list[int]
    sampling_params: types.SamplingParams
    model_name: str
    n: int
    session_id: str | None = None


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
    lora_unload_endpoint: str = "/v1/unload_lora_adapter"
    # When set, ephemeral adapters are pushed over HTTP to this endpoint
    # (tpu/vllm_tpu_server.py) instead of round-tripping shared storage.
    lora_upload_endpoint: str = ""
    lora_load_retries: int = 3
    lora_load_retry_sleep_sec: float = 2.0
    request_timeout_sec: float = 300.0
    max_concurrent_requests: int = 64
    client_side_round_robin: bool = False
    _loaded_loras: dict[str, dict[str, str]] = field(default_factory=dict)
    _latest_lora_by_model: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        url_pairs = _normalize_vllm_url_list(self.base_url)
        self.server_base_urls = tuple(pair[0] for pair in url_pairs)
        self.openai_base_urls = tuple(pair[1] for pair in url_pairs)
        self.server_base_url = self.server_base_urls[0]
        self.openai_base_url = self.openai_base_urls[0]
        if not self.lora_load_endpoint.startswith("/"):
            self.lora_load_endpoint = f"/{self.lora_load_endpoint}"
        if self.lora_unload_endpoint and not self.lora_unload_endpoint.startswith("/"):
            self.lora_unload_endpoint = f"/{self.lora_unload_endpoint}"
        self.lora_load_retries = max(1, self.lora_load_retries)
        self.lora_load_retry_sleep_sec = max(0.0, self.lora_load_retry_sleep_sec)
        self.max_concurrent_requests = max(1, self.max_concurrent_requests)

    def _completion_url_for_index(self, index: int) -> str:
        if self.client_side_round_robin:
            return self.openai_base_urls[index % len(self.openai_base_urls)]
        return self.openai_base_url

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
            raise VllmRequestError(f"vLLM request failed: {e.code} {e.reason}: {body}", e.code) from e
        except urllib.error.URLError as e:
            raise VllmRequestError(f"vLLM request failed: {e}") from e

        if "application/json" in content_type:
            return json.loads(body) if body else {}
        return body

    def _post_json_with_retries(self, url: str, payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.lora_load_retries + 1):
            try:
                return self._post_json(url, payload)
            except Exception as e:  # noqa: BLE001 - surface the final vLLM error after bounded retries.
                last_error = e
                if attempt == self.lora_load_retries:
                    break
                logger.warning(
                    "vLLM LoRA load failed on attempt %s/%s for %s: %s; retrying in %.1fs",
                    attempt,
                    self.lora_load_retries,
                    url,
                    e,
                    self.lora_load_retry_sleep_sec,
                )
                time.sleep(self.lora_load_retry_sleep_sec)
        assert last_error is not None
        raise last_error

    def _unload_lora(self, lora_name: str) -> None:
        if not self.lora_unload_endpoint:
            return
        payload = {"lora_name": lora_name}
        for server_url in self.server_base_urls:
            try:
                self._post_json(f"{server_url}{self.lora_unload_endpoint}", payload)
            except VllmRequestError as e:
                if e.status_code in (400, 404):
                    logger.info("LoRA adapter %s was not loaded on vLLM server %s", lora_name, server_url)
                    continue
                logger.warning("Failed to unload LoRA adapter %s from vLLM server %s: %s", lora_name, server_url, e)
            except Exception as e:  # noqa: BLE001 - unload should not block loading the replacement adapter.
                logger.warning("Failed to unload LoRA adapter %s from vLLM server %s: %s", lora_name, server_url, e)
        self._loaded_loras.pop(lora_name, None)

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
        loaded_servers = self._loaded_loras.setdefault(lora_name, {})
        if all(loaded_servers.get(server_url) == target_path for server_url in self.server_base_urls):
            return lora_name

        previous_lora = self._latest_lora_by_model.get(model_id)
        if previous_lora and previous_lora != lora_name:
            self._unload_lora(previous_lora)

        payload = {"lora_name": lora_name, "lora_path": target_path}
        for server_url in self.server_base_urls:
            if loaded_servers.get(server_url) == target_path:
                continue
            self._post_json_with_retries(f"{server_url}{self.lora_load_endpoint}", payload)
            loaded_servers[server_url] = target_path
            logger.info("Loaded LoRA adapter %s into vLLM server %s from %s", lora_name, server_url, target_path)
        self._latest_lora_by_model[model_id] = lora_name
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

    def _completion_request(
        self,
        prompt_ids: list[int],
        sampling_params: types.SamplingParams,
        *,
        n: int,
        model_name: str,
        session_id: str | None = None,
        prompt_logprobs: bool = False,
        openai_base_url: str | None = None,
    ) -> tuple[list[types.GeneratedSequence], list[float] | None]:
        """Issue one vLLM completion request that returns ``n`` sequences.

        With ``n > 1`` this asks vLLM for a whole group of completions sharing a
        single prompt in one request (grouped sampling), which is much cheaper
        than ``n`` separate ``n=1`` requests. The prompt is shared, so a single
        prompt-logprobs list is returned for the group.
        """
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt_ids,
            "n": n,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            # No per-request seed: the vLLM TPU (tpu-inference) backend rejects
            # it with "JAX does not support per-request seed".
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
        result = self._post_json(f"{openai_base_url or self.openai_base_url}/completions", payload, headers=headers)
        choices = result.get("choices") or []
        if len(choices) != n:
            raise RuntimeError(f"Expected {n} vLLM completion choice(s), got {len(choices)}")
        # vLLM returns one choice per requested sample; keep them in index order.
        choices = sorted(choices, key=lambda choice: choice.get("index", 0))

        prompt_lps = self._prompt_logprobs_from_response(result, prompt_ids) if prompt_logprobs else None
        return [self._sequence_from_choice(choice) for choice in choices], prompt_lps

    def sample_one(
        self,
        prompt_ids: list[int],
        sampling_params: types.SamplingParams,
        *,
        model_name: str,
        session_id: str | None = None,
        prompt_logprobs: bool = False,
        openai_base_url: str | None = None,
    ) -> tuple[types.GeneratedSequence, list[float] | None]:
        sequences, prompt_lps = self._completion_request(
            prompt_ids,
            sampling_params,
            n=1,
            model_name=model_name,
            session_id=session_id,
            prompt_logprobs=prompt_logprobs,
            openai_base_url=openai_base_url,
        )
        return sequences[0], prompt_lps

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
        if not prompt_ids:
            return [], []

        def run_one(i: int) -> tuple[types.GeneratedSequence, list[float] | None]:
            return self.sample_one(
                prompt_ids[i],
                sampling_params[i],
                model_name=model_names[i],
                session_id=session_ids[i],
                prompt_logprobs=prompt_logprobs,
                openai_base_url=self._completion_url_for_index(i),
            )

        workers = min(self.max_concurrent_requests, len(prompt_ids))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(run_one, range(len(prompt_ids))))

        sequences, prompt_lps = zip(*outputs) if outputs else ([], [])
        return list(sequences), list(prompt_lps)

    def push_adapter(self, model_id: str, checkpoint_id: str, tar_bytes: bytes) -> str:
        """Push an adapter tarball directly to the server's upload endpoint.

        Returns the served adapter name. The server extracts to its local disk,
        unloads the previous version for this model, and hot-loads the new one —
        no shared filesystem involved.
        """
        if not self.lora_upload_endpoint:
            raise RuntimeError("push_adapter requires lora_upload_endpoint to be configured")
        lora_name = _sanitize_lora_name(model_id, checkpoint_id)
        previous = self._latest_lora_by_model.get(model_id)
        query = f"lora_name={lora_name}"
        if previous and previous != lora_name:
            query += f"&previous_lora_name={previous}"

        last_error: Exception | None = None
        for server_url in self.server_base_urls:
            url = f"{server_url}{self.lora_upload_endpoint}?{query}"
            for attempt in range(1, self.lora_load_retries + 1):
                req = urllib.request.Request(
                    url,
                    data=tar_bytes,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/x-tar"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                        resp.read()
                    last_error = None
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    if attempt < self.lora_load_retries:
                        time.sleep(self.lora_load_retry_sleep_sec)
            if last_error is not None:
                raise VllmRequestError(f"adapter upload to {url} failed: {last_error}")

        self._latest_lora_by_model[model_id] = lora_name
        return lora_name

    def sample_groups(
        self,
        groups: list[GroupedCompletion],
        *,
        prompt_logprobs: bool = False,
    ) -> tuple[list[list[types.GeneratedSequence]], list[list[float] | None]]:
        """Sample each group with one ``n=group.n`` vLLM completion request.

        Returns one list of generated sequences per group (each of length
        ``group.n``) plus one shared prompt-logprobs list per group, aligned to
        the input ``groups`` order.
        """
        if not groups:
            return [], []

        def run_one(i: int) -> tuple[list[types.GeneratedSequence], list[float] | None]:
            group = groups[i]
            return self._completion_request(
                group.prompt_ids,
                group.sampling_params,
                n=group.n,
                model_name=group.model_name,
                session_id=group.session_id,
                prompt_logprobs=prompt_logprobs,
                openai_base_url=self._completion_url_for_index(i),
            )

        workers = min(self.max_concurrent_requests, len(groups))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(run_one, range(len(groups))))

        sequences = [seqs for seqs, _ in outputs]
        prompt_lps = [lps for _, lps in outputs]
        return sequences, prompt_lps
