import asyncio
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from cloudpathlib import AnyPath
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack

if TYPE_CHECKING:
    from skyrl.tinker.api import SampleRequest


def _extract_checkpoint_sync(checkpoint_path: AnyPath, target_dir: Path) -> None:
    """Extract a LoRA checkpoint to disk for vLLM to load.

    This is a blocking operation (filesystem/network I/O) and should be called
    via asyncio.to_thread() to avoid blocking the event loop.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        return

    # The unpack tempdir usually lives on a different filesystem than the
    # lora dir (e.g. /tmp vs a GCS FUSE mount), so a direct rename raises
    # EXDEV. Copy into a same-directory staging path, then rename in place.
    staging = target_dir.parent / f".{target_dir.name}.staging-{os.getpid()}-{os.urandom(4).hex()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        with download_and_unpack(checkpoint_path) as extracted_path:
            shutil.copytree(extracted_path, staging)
        try:
            staging.rename(target_dir)
        except (FileExistsError, OSError):
            # Another process won the race and created target_dir.
            if not target_dir.exists():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.base_url = f"{engine_config.external_inference_url}/v1"
        self.api_key = engine_config.external_inference_api_key
        self.request_timeout_sec = engine_config.external_inference_timeout_sec
        self.allow_prompt_logprobs = engine_config.external_inference_prompt_logprobs
        self.checkpoints_base = engine_config.checkpoints_base
        self.lora_base_dir = engine_config.external_inference_lora_base
        self.db_engine = db_engine
        # Adapter names already extracted this process — a sampling burst is
        # hundreds of concurrent requests for the same checkpoint, and each
        # gcsfuse stat/unpack is expensive. The per-name lock makes extraction
        # single-flight: exactly one request unpacks, the rest wait on it.
        self._extracted_adapters: set[str] = set()
        self._extract_locks: dict[str, asyncio.Lock] = {}

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Background task to call external engine and store result in database."""
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(self.request_timeout_sec, connect=10.0),
            ) as http_client:
                result = await self._forward_to_engine(
                    sample_req, model_id, checkpoint_id, http_client, base_model=base_model
                )
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_to_engine(
        self,
        request: "SampleRequest",
        model_id: str,
        checkpoint_id: str,
        http_client: httpx.AsyncClient,
        *,
        base_model: str | None = None,
    ) -> types.SampleOutput:
        """Forward request to vLLM with dynamic LoRA loading.

        Extracts the checkpoint to the configured external_inference_lora_base and references it by a model name
        that vLLM can dynamically load via the lora_filesystem_resolver plugin.

        For base model sampling (no LoRA), the request is sent directly using the base model name.
        """
        model_input = request.prompt.to_types()
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        if base_model:
            # Base model sampling: use the model name directly, no LoRA checkpoint needed
            model_name = base_model
        else:
            # LoRA sampling: extract checkpoint and reference it by name for dynamic loading
            model_name = f"{model_id}_{checkpoint_id}"
            checkpoint_path = self.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
            target_dir = self.lora_base_dir / model_name

            if model_name not in self._extracted_adapters:
                async with self._extract_locks.setdefault(model_name, asyncio.Lock()):
                    if model_name not in self._extracted_adapters:
                        await asyncio.to_thread(_extract_checkpoint_sync, checkpoint_path, target_dir)
                        self._extracted_adapters.add(model_name)

        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": request.num_samples,
            # No per-request seed: the vLLM TPU (tpu-inference) backend rejects
            # it with "JAX does not support per-request seed".
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "top_p": request.sampling_params.top_p,
            "top_k": request.sampling_params.top_k,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
        }
        # Forward stop conditions (previously dropped: clients only stopped on
        # the model's EOS). Ints are token ids, strings are text stops.
        if request.sampling_params.stop:
            stops = list(request.sampling_params.stop)
            if all(isinstance(s, int) for s in stops):
                payload["stop_token_ids"] = stops
            else:
                payload["stop"] = stops
        # See EngineConfig.external_inference_prompt_logprobs: the vLLM TPU
        # backend dies on this parameter, so it is opt-in.
        want_prompt_logprobs = bool(request.prompt_logprobs) and self.allow_prompt_logprobs
        if want_prompt_logprobs:
            payload["prompt_logprobs"] = 1

        # Pass X-Session-ID for deterministic routing
        headers = {}
        session_id = types.make_routing_session_id(request.sampling_session_id, request.seq_id)
        if session_id is not None:
            headers["X-Session-ID"] = session_id

        response = await http_client.post("/completions", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        sequences = []
        for choice in result["choices"]:
            lp = choice["logprobs"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=choice["token_ids"],
                    logprobs=lp["token_logprobs"],
                    stop_reason=choice["finish_reason"],
                )
            )

        prompt_lps: list[float] | None = None
        if want_prompt_logprobs:
            from skyrl.backends.vllm_sampling import VllmSamplingClient

            prompt_lps = VllmSamplingClient._prompt_logprobs_from_response(result, prompt_tokens)
        return types.SampleOutput(sequences=sequences, prompt_logprobs=prompt_lps if prompt_lps is not None else [])
