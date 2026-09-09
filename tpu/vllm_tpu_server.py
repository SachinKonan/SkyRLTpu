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
import sys
import json
import os
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


def _add_upload_endpoint(app, lora_dir: Path, engine, partner_hosts: list[str] | None = None) -> None:
    lora_dir.mkdir(parents=True, exist_ok=True)
    partner_hosts = list(partner_hosts or [])

    @app.post("/skyrl/v1/upload_lora_adapter")
    async def _upload_lora_adapter(request: Request):
        lora_name = request.query_params.get("lora_name")
        previous = request.query_params.get("previous_lora_name")
        if not lora_name or "/" in lora_name or lora_name.startswith("."):
            raise HTTPException(status_code=400, detail="valid 'lora_name' query param required")

        # The trainer-side client is recreated when only the trainer restarts,
        # while this vLLM process may stay alive. Persist the last upload here
        # so a fresh client cannot leak old adapters by forgetting their name.
        latest_marker = lora_dir / ".skyrl-latest-lora"
        if not previous:
            try:
                candidate = latest_marker.read_text().strip()
                if candidate and "/" not in candidate and not candidate.startswith("."):
                    previous = candidate
            except FileNotFoundError:
                pass

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
            if partner_hosts:
                await asyncio.get_running_loop().run_in_executor(
                    None, _replicate_adapter, target, partner_hosts)
        else:
            # The adapter is already extracted (a retry after a lost ACK), but
            # the body must still be drained: responding with megabytes of
            # request unread makes uvicorn close the socket mid-upload, the
            # client sees ECONNRESET instead of our 200, and every retry
            # repeats the cycle — the trainer never learns the push succeeded.
            async for _ in request.stream():
                pass

        # Load the ordinary PEFT half first. vLLM assigns the adapter a
        # physical Punica slot here; the expert sidecar must be installed in
        # that exact slot so mixed base/A/B requests select matching router,
        # attention, and expert factors.
        models = request.app.state.openai_serving_models

        if previous and previous != lora_name:
            await models.unload_lora_adapter(
                UnloadLoRAAdapterRequest(lora_name=previous))
            # Not-loaded is fine (server restart, first sync).
            shutil.rmtree(lora_dir / previous, ignore_errors=True)
            if partner_hosts:
                await asyncio.get_running_loop().run_in_executor(
                    None, _remove_on_hosts, lora_dir / previous, partner_hosts)

        was_loaded = lora_name in models.lora_requests
        resp = await models.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name=lora_name,
                                   lora_path=str(target),
                                   load_inplace=was_loaded))
        if not isinstance(resp, str):  # vllm returns ErrorResponse objects on failure
            detail = getattr(resp, "message", None) or str(resp)
            raise HTTPException(
                status_code=400,
                detail=f"load_lora_adapter failed: {detail}")
        lora_request = models.lora_requests.get(lora_name)
        if lora_request is None:
            raise HTTPException(
                status_code=500,
                detail=f"loaded adapter {lora_name!r} has no vLLM request id")
        lora_int_id = int(lora_request.lora_int_id)

        async def _rollback_new_adapter() -> None:
            if not was_loaded:
                await models.unload_lora_adapter(
                    UnloadLoRAAdapterRequest(lora_name=lora_name))

        def _validate_moe_update(result, *, cleared: bool):
            updates = result if isinstance(result, list) else [result]
            if not updates:
                raise RuntimeError(
                    "expert LoRA update returned no worker results")
            for item in updates:
                if not isinstance(item, dict):
                    raise RuntimeError(
                        f"unexpected expert LoRA result: {item!r}")
                if item.get("base_weights_mutated") is not False:
                    raise RuntimeError(
                        "worker did not prove immutable MXFP4 base: "
                        f"{item!r}")
                if item.get("cleared") is not cleared:
                    raise RuntimeError(
                        f"worker returned the wrong clear state: {item!r}")
                if item.get("lora_id") != lora_int_id:
                    raise RuntimeError(
                        f"worker updated the wrong LoRA id: {item!r}")
            return result

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
                    "set_moe_lora_factors",
                    args=(rpc_factors, rpc_meta, lora_int_id),
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
                result = _validate_moe_update(result, cleared=False)
            except Exception as exc:
                await _rollback_new_adapter()
                raise HTTPException(
                    status_code=500,
                    detail=f"set_moe_lora_factors RPC failed: {exc!r}",
                ) from exc
            logger.info(
                "MXFP4 expert LoRA factor update for %s: %d tensors -> %s",
                lora_name, n_tensors, result,
            )
            moe_update = result
        else:
            # An attention/router-only adapter still owns a physical expert
            # slot. Explicitly zero it in case vLLM reused a formerly active
            # expert slot.
            try:
                res = engine.collective_rpc(
                    "set_moe_lora_factors",
                    args=(None, {"scale": 0.0}, lora_int_id),
                )
                if inspect.isawaitable(res):
                    res = await res
                res = _validate_moe_update(res, cleared=True)
            except Exception as exc:
                await _rollback_new_adapter()
                raise HTTPException(
                    status_code=500,
                    detail=f"clear expert LoRA slot failed: {exc!r}",
                ) from exc
            logger.info("Cleared MXFP4 expert LoRA factors for %s", lora_name)
            moe_update = res

        # Remember the adapter that is now live so a restarted trainer client can
        # name it as `previous_lora_name` instead of leaking it (see top of handler).
        marker_tmp = latest_marker.with_suffix(".tmp")
        marker_tmp.write_text(f"{lora_name}\n")
        marker_tmp.replace(latest_marker)

        # Returning the worker result makes the live acceptance gate capable
        # of proving that all MXFP4 expert buffers were updated (or cleared),
        # without scraping a human-oriented server log. Existing clients only
        # consume status/lora_name, so this is backward compatible.
        return {
            "status": "ok",
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "moe_update": moe_update,
        }


# Environment the second host's vLLM worker needs verbatim (vLLM itself only
# copies VLLM_* and the TPU platform's additional_env_vars to Ray workers).
_WORKER_ENV_KEYS = (
    "HF_HOME", "HF_HUB_OFFLINE", "JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "VLLM_XLA_CACHE_PATH", "TPU_BACKEND_TYPE",
    "MODEL_IMPL_TYPE", "TPU_MULTIHOST_BACKEND", "SKIP_JAX_PRECOMPILE", "USE_BATCHED_RPA_KERNEL",
    "USE_JAX_RAGGED_CONV1D", "USE_MOE_EP_KERNEL", "MOE_REQUANTIZE_WEIGHT_DTYPE",
    "MOE_REQUANTIZE_BLOCK_SIZE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    # The lora_filesystem_resolver plugin loads in every worker and refuses to
    # start without its cache dir (job 532); vLLM's own VLLM_* copy missed it.
    "VLLM_LORA_RESOLVER_CACHE_DIR", "VLLM_PLUGINS", "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
)


def _replicate_adapter(target: Path, hosts: list[str]) -> None:
    """Copy an extracted adapter directory to the other hosts of a
    pipeline-parallel engine, at the same path.

    Every vLLM worker loads the PEFT adapter from `lora_path` on its own disk
    and the expert-factor RPC reads a local safetensors path too, so a pair's
    second host must hold the files before load_lora_adapter runs. Shipped
    through the executor's Ray cluster (tasks pinned to each host) so no ssh
    assumptions are made.
    """
    if not hosts:
        return
    import ray

    @ray.remote(num_cpus=0)
    def _write(root: str, files: dict) -> int:
        dest = Path(root)
        staging = dest.with_name(dest.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for rel, data in files.items():
            out = staging / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        shutil.rmtree(dest, ignore_errors=True)
        staging.replace(dest)
        return len(files)

    files = {str(f.relative_to(target)): f.read_bytes() for f in sorted(target.rglob("*")) if f.is_file()}
    payload = ray.put(files)
    del files
    written = ray.get([_write.options(resources={f"node:{host}": 0.001}).remote(str(target), payload)
                       for host in hosts], timeout=900)
    logger.info("replicated adapter %s (%d files) to %s", target.name, written[0], hosts)


def _remove_on_hosts(path: Path, hosts: list[str]) -> None:
    if not hosts:
        return
    import ray

    @ray.remote(num_cpus=0)
    def _rm(root: str) -> None:
        shutil.rmtree(root, ignore_errors=True)

    try:
        ray.get([_rm.options(resources={f"node:{host}": 0.001}).remote(str(path)) for host in hosts], timeout=120)
    except Exception as exc:  # disk hygiene only; never fail an upload over it
        logger.warning("could not remove %s on %s: %r", path.name, hosts, exc)


def _mkdir_on_hosts(path: Path, hosts: list[str]) -> None:
    """Create `path` on the given hosts through the executor's Ray cluster.

    The lora_filesystem_resolver plugin loads in every vLLM worker and
    requires its cache dir to exist locally; only the engine head created it
    (job 550: partner workers died on "must be set to a valid directory").
    """
    if not hosts:
        return
    import ray

    @ray.remote(num_cpus=0)
    def _mk(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        return root

    ray.get([_mk.options(resources={f"node:{host}": 0.001}).remote(str(path)) for host in hosts], timeout=120)


def _engine_on_existing_ray(engine_args, hosts: list[str], lora_dir: Path | None = None):
    """Pipeline-parallel engine over `hosts` on an already-running Ray cluster.

    Joins the cluster named by RAY_ADDRESS as a driver whose job runtime_env
    runs every actor under THIS interpreter (the serving venv; the cluster's
    raylets were started from another venv), reserves TPU:4 on exactly the
    given hosts with a placement group, and hands that group to vLLM so its
    Ray executor does not try to span every TPU node in the cluster. One
    worker per host: pipeline stage i on hosts[i], tensor-parallel over that
    host's own chips (tpu-inference's Ray multihost mode).
    """
    import ray
    from ray.util.placement_group import placement_group

    from ray.runtime_env import RuntimeEnv

    env_vars = {k: os.environ[k] for k in _WORKER_ENV_KEYS if k in os.environ}
    runtime_env = {"py_executable": sys.executable, "env_vars": env_vars}
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True,
             runtime_env=runtime_env)
    # vLLM forks its engine-core subprocess by default and only forces spawn
    # inside a Ray actor; forking this driver after ray.init deadlocks the
    # child (job 523: EngineCore asleep on a futex with three threads, the
    # placement group reserved but no worker ever created). Spawn a fresh
    # interpreter instead; it reconnects to Ray on RAY_ADDRESS with the same
    # runtime env through parallel_config.ray_runtime_env.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # vLLM's parallel_config.ray_runtime_env does not survive into the
    # spawned core (job 527: its Ray job had no py_executable and its workers
    # died on `No module named tpu_inference`). Ray's job-config variable is
    # read by every ray.init in a child process, so hand the runtime env over
    # that way; RAY_OVERRIDE_JOB_RUNTIME_ENV lets it merge with whatever
    # vLLM passes instead of raising on overlap.
    os.environ["RAY_JOB_CONFIG_JSON_ENV_VAR"] = json.dumps({"runtime_env": runtime_env})
    os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
    chips = int(engine_args.tensor_parallel_size)
    bundles = [{"TPU": chips, f"node:{host}": 0.001} for host in hosts]
    group = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(group.ready(), timeout=300)
    logger.info("placement group ready on %s (%d TPU each)", hosts, chips)
    if lora_dir is not None:
        _mkdir_on_hosts(lora_dir, hosts[1:])
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    vllm_config.parallel_config.placement_group = group
    vllm_config.parallel_config.ray_runtime_env = RuntimeEnv(**runtime_env)
    return AsyncLLMEngine.from_vllm_config(vllm_config, usage_context=UsageContext.OPENAI_API_SERVER)


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

    engine_args = AsyncEngineArgs.from_cli_args(args)
    placement_hosts = [h for h in (args.skyrl_ray_placement_hosts or "").split(",") if h]
    if placement_hosts:
        engine = _engine_on_existing_ray(engine_args, placement_hosts, Path(args.skyrl_lora_dir))
    else:
        engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

    _add_upload_endpoint(app, Path(args.skyrl_lora_dir), engine, partner_hosts=placement_hosts[1:])
    await init_app_state(engine, app.state, args)

    config = uvicorn.Config(
        app,
        host=args.host or "0.0.0.0",
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
    )
    server = uvicorn.Server(config)
    # vLLM request error handlers call through this attribute.
    app.state.server = server
    await server.serve()


def main() -> None:
    parser = FlexibleArgumentParser(description="vLLM TPU server with SkyRL adapter upload")
    parser.add_argument(
        "--skyrl-lora-dir",
        type=str,
        default=str(Path.home() / "skyrl-local-loras"),
        help="Local directory where uploaded adapters are extracted.",
    )
    parser.add_argument(
        "--skyrl-ray-placement-hosts",
        type=str,
        default="",
        help="Comma-separated host IPs of a pipeline-parallel engine on an existing Ray "
             "cluster (RAY_ADDRESS); one TPU worker per host, this host first.",
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
