"""vLLM TPU server with the SkyRL adapter-upload endpoint.

Drop-in replacement for ``vllm serve`` on TPU workers. Adds
``POST /skyrl/v1/upload_lora_adapter?lora_name=...[&previous_lora_name=...]``:
the request body is an uncompressed tar of an HF-PEFT adapter dir, streamed to
local disk (``--skyrl-lora-dir``), extracted, and hot-loaded — the previous
adapter version is unloaded first. This replaces the shared-filesystem
(GCS FUSE) round-trip for ephemeral RL weight syncs: the trainer POSTs the
adapter over the VPC instead of writing it to a bucket.

Self-contained on purpose: it is scp'd to the vLLM worker and run inside the
vllm-tpu venv, which does not have the skyrl package installed.

Compatible with vllm==0.23 (mirrors skyrl's GPU vllm_server_actor pattern).
"""

import argparse
import asyncio
import inspect
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import uvicorn
import vllm.envs as envs
from fastapi import HTTPException, Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit

logger = logging.getLogger(__name__)


def _add_upload_endpoint(app, lora_dir: Path, engine) -> None:
    lora_dir.mkdir(parents=True, exist_ok=True)

    @app.post("/skyrl/v1/upload_lora_adapter")
    async def _upload_lora_adapter(request: Request):
        lora_name = request.query_params.get("lora_name")
        previous = request.query_params.get("previous_lora_name")
        if not lora_name or "/" in lora_name or lora_name.startswith("."):
            raise HTTPException(status_code=400, detail="valid 'lora_name' query param required")

        target = lora_dir / lora_name
        if not target.exists():
            # Stream the tar body to disk (payloads are up to ~GBs of f32
            # LoRA factors; never buffer fully in RAM).
            with tempfile.NamedTemporaryFile(dir=lora_dir, suffix=".tar", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                async for chunk in request.stream():
                    tmp.write(chunk)
            staging = lora_dir / f".{lora_name}.staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            try:
                with tarfile.open(tmp_path, "r:") as tar:
                    tar.extractall(staging, filter="data")
            except tarfile.ReadError as exc:
                # 0-byte or truncated body (health probes POST with no body).
                # A 400 keeps the log clean; a raw ReadError is a 500 traceback
                # that reads like an export bug — it cost a day of muse triage.
                shutil.rmtree(staging, ignore_errors=True)
                raise HTTPException(status_code=400, detail=f"unreadable tar body: {exc}") from exc
            finally:
                tmp_path.unlink(missing_ok=True)
            if not (staging / "adapter_config.json").exists():
                shutil.rmtree(staging, ignore_errors=True)
                raise HTTPException(status_code=400, detail="tar does not contain adapter_config.json at its root")
            staging.replace(target)
        else:
            # The adapter is already extracted (a retry after a lost ACK), but
            # the body must still be drained: responding with megabytes of
            # request unread makes uvicorn close the socket mid-upload, the
            # client sees ECONNRESET instead of our 200, and every retry
            # repeats the cycle — the trainer never learns the push succeeded.
            async for _ in request.stream():
                pass

        # GPT-OSS expert sidecar: replace the fixed-shape BF16 factor buffers
        # evaluated beside the immutable MXFP4 base GMMs. This must run on
        # every upload call, including retries whose directory already exists.
        # The BF16 router is part of adapter_model.safetensors and follows
        # vLLM's ordinary ReplicatedLinear LoRA path below.
        moe_update = None
        moe_path = target / "moe_lora.safetensors"
        if moe_path.exists():
            import json as _json

            from safetensors import safe_open as _safe_open

            meta = _json.loads((target / "moe_lora.json").read_text())
            with _safe_open(str(moe_path), framework="numpy") as _f:
                n_tensors = len(list(_f.keys()))

            async def _rpc(rpc_factors, rpc_meta=meta):
                # vllm 0.23: AsyncLLMEngine is v1 AsyncLLM
                # (vllm/engine/async_llm_engine.py aliases it), whose
                # `collective_rpc(method, timeout=None, args=(), kwargs=None)`
                # is an async method fanning out to the TPU workers (the
                # isawaitable branch also covers a sync variant).
                res = engine.collective_rpc(
                    "set_moe_lora_factors", args=(rpc_factors, rpc_meta)
                )
                if inspect.isawaitable(res):
                    res = await res
                return res

            # Wire format: the frontend->EngineCore collective_rpc hop
            # serializes args with vllm's MsgpackEncoder, whose enc_hook
            # turns every ndarray into a (dtype_str, shape,
            # inline-bytes-or-aux-buffer-index) triple that untyped RPC
            # targets can never reconstruct (big arrays are an index into
            # aux buffers the untyped path cannot access). So do NOT send
            # ndarrays: send the safetensors PATH (frontend and EngineCore
            # share this host) and let the worker load it locally. If a
            # deployment ever splits the processes across hosts, fall back
            # to nested Python lists, which survive msgpack exactly
            # (~845MB f32 -> ~2GB message for a rank-32 20B push).
            try:
                try:
                    result = await _rpc(str(moe_path))
                except Exception:
                    logger.exception(
                        "MoE LoRA factor update: path-based RPC for %s failed; "
                        "retrying with inline nested-list factors", lora_name,
                    )
                    from safetensors.numpy import load_file as _st_load
                    result = await _rpc({
                        k: v.tolist() for k, v in _st_load(str(moe_path)).items()
                    })
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"set_moe_lora_factors RPC failed: {exc!r}",
                ) from exc
            logger.info(
                "MXFP4 expert LoRA factor update for %s: %d tensors -> %s",
                lora_name, n_tensors, result,
            )
            moe_update = result
        elif previous and (lora_dir / previous / "moe_lora.safetensors").exists():
            # Moving from a full GPT-OSS adapter to attention/router-only must
            # not leave the previous expert factors globally active.
            res = engine.collective_rpc("set_moe_lora_factors", args=(None, {"scale": 0.0}))
            if inspect.isawaitable(res):
                res = await res
            logger.info("Cleared MXFP4 expert LoRA factors for %s", previous)
            moe_update = res

        models = request.app.state.openai_serving_models

        if previous and previous != lora_name:
            resp = await models.unload_lora_adapter(UnloadLoRAAdapterRequest(lora_name=previous))
            # Not-loaded is fine (server restart, first sync); surface other errors.
            shutil.rmtree(lora_dir / previous, ignore_errors=True)

        resp = await models.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name=lora_name, lora_path=str(target))
        )
        if not isinstance(resp, str):  # vllm returns ErrorResponse objects on failure
            detail = getattr(resp, "message", None) or str(resp)
            if "already been loaded" not in detail:
                raise HTTPException(status_code=400, detail=f"load_lora_adapter failed: {detail}")

        # Returning the worker result makes the live acceptance gate capable
        # of proving that all MXFP4 expert buffers were updated (or cleared),
        # without scraping a human-oriented server log. Existing clients only
        # consume status/lora_name, so this is backward compatible.
        return {
            "status": "ok",
            "lora_name": lora_name,
            "moe_update": moe_update,
        }


async def _serve(args) -> None:
    # This module logs under "__main__"; without an explicit level its INFO
    # records (notably the "MoE LoRA merge-on-load ... -> {...}" success
    # line, the observable proof that an adapter push merged) are dropped at
    # the root logger's default WARNING threshold.
    logger.setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    set_ulimit()
    app = build_app(args)

    engine = AsyncLLMEngine.from_engine_args(
        engine_args=AsyncEngineArgs.from_cli_args(args),
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    _add_upload_endpoint(app, Path(args.skyrl_lora_dir), engine)
    await init_app_state(engine, app.state, args)

    config = uvicorn.Config(
        app,
        host=args.host or "0.0.0.0",
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
    )
    await uvicorn.Server(config).serve()


def main() -> None:
    parser = FlexibleArgumentParser(description="vLLM TPU server with SkyRL adapter upload")
    parser.add_argument(
        "--skyrl-lora-dir",
        type=str,
        default=str(Path.home() / "skyrl-local-loras"),
        help="Local directory where uploaded adapters are extracted.",
    )
    parser = make_arg_parser(parser)
    # Newer vllm defines the `model_tag` positional itself; adding a second
    # positional with the same dest makes the optional one overwrite the
    # parsed value with None.
    if not any(a.dest == "model_tag" for a in parser._actions if not a.option_strings):
        parser.add_argument("model_tag", type=str, help="model name or path (positional, like `vllm serve`)")
    args = parser.parse_args()
    if getattr(args, "model_tag", None):
        args.model = args.model_tag

    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
