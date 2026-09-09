import asyncio
import inspect
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import httpx
import pytest

from tpu.swarm.ray_train import serving
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.commands import client_environment, inference_command, trainer_backend_config, trainer_environment
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.overlay import install, identity


class Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args):
        value = self.fn(*args)
        return await value if inspect.isawaitable(value) else value


@pytest.mark.parametrize("max_loras", [1, 2])
@pytest.mark.parametrize("expert_lora_slots", [False, True])
def test_engine_upload_replaces_only_the_selected_adapter(tmp_path, max_loras, expert_lora_slots):
    # Execute the production endpoint without importing the TPU-only vLLM
    # engine. Fake only its load/unload API; exercise real HTTP and tar uploads.
    import ast
    import io
    import logging
    import shutil
    import tempfile
    from fastapi import FastAPI, HTTPException, Request

    source = Path("tpu/vllm_tpu_server.py").read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_add_upload_endpoint")
    scope = dict(Path=Path, HTTPException=HTTPException, Request=Request,
                 tempfile=tempfile, shutil=shutil, tarfile=tarfile, inspect=inspect,
                 logger=logging.getLogger(__name__),
                 LoadLoRAAdapterRequest=SimpleNamespace, UnloadLoRAAdapterRequest=SimpleNamespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "tpu/vllm_tpu_server.py", "exec"), scope)
    loaded = set()

    async def load(request):
        loaded.add(request.lora_name)
        models.lora_requests[request.lora_name] = SimpleNamespace(lora_int_id=len(loaded))
        return "ok"

    async def unload(request):
        loaded.discard(request.lora_name)
        models.lora_requests.pop(request.lora_name, None)
        return "ok"

    app = FastAPI()
    models = SimpleNamespace(load_lora_adapter=load, unload_lora_adapter=unload, lora_requests={})
    app.state.openai_serving_models = models
    cleared_ids = []
    async def rpc(method, args):
        assert expert_lora_slots, "Dense Qwen adapters must not call the GPT-OSS expert RPC"
        assert method == "set_moe_lora_factors"
        assert args[0] is None
        cleared_ids.append(args[2])
        # Two workers in the paired engine must acknowledge the same LoRA ID.
        return [dict(base_weights_mutated=False, cleared=True, lora_id=args[2]) for _ in range(2)]
    scope["_add_upload_endpoint"](app, tmp_path, SimpleNamespace(collective_rpc=rpc),
                                  max_loras=max_loras, expert_lora_slots=expert_lora_slots)
    # A leftover single-adapter marker must not affect a multi-adapter server.
    (tmp_path / ".skyrl-latest-lora").write_text("stale-adapter\n")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as archive:
        info = tarfile.TarInfo("adapter_config.json")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://engine") as client:
            async def upload(name, previous=None):
                params = {"lora_name": name}
                if previous:
                    params["previous_lora_name"] = previous
                response = await client.post("/skyrl/v1/upload_lora_adapter", params=params, content=body.getvalue())
                assert response.status_code == 200, response.text
                assert (response.json()["moe_update"] is not None) == expert_lora_slots

            await upload("A-0")
            await upload("B-0")
            assert loaded == ({"A-0", "B-0"} if max_loras == 2 else {"B-0"})
            await upload("B-1", "B-0")
            await upload("B-1", "B-0")  # Retry after a lost acknowledgement.
            assert loaded == ({"A-0", "B-1"} if max_loras == 2 else {"B-1"})
            expected_ids = ([1, 2, 2, 2] if max_loras == 2 else [1, 1, 1, 1]) if expert_lora_slots else []
            assert cleared_ids == expected_ids
            assert not (tmp_path / "B-0").exists()
            assert (tmp_path / "A-0").exists() == (max_loras == 2)

    asyncio.run(run())


def test_multi_lora_profile_and_frozen_overlay(tmp_path):
    profile = "tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_multi_lora.json"
    cfg = Config.load(profile)
    env = client_environment(cfg, tmp_path, "10.0.0.1")
    assert cfg.inference.max_adapter_upload_bytes == 8 * 1024**3
    assert env["TTD_ENSEMBLE_MODELS"].count("openai/gpt-oss-120b") == 2
    assert trainer_environment(cfg, tmp_path, tmp_path, ["head"], 0)["TUNIX_MINIMAL_FB_OUTPUT"] == "0"
    assert env["GROUP_SIZE"] == "32" and env["TTD_LOSS_FN"] == "cispo"
    assert env["TTD_M0_LORA_SEED"] != env["TTD_M1_LORA_SEED"]
    assert env["TTD_M0_BASE_URL"] == env["TTD_M1_BASE_URL"]
    cmd = inference_command(cfg, tmp_path, tmp_path, tmp_path, tmp_path, group=["a", "b"])
    assert cmd[cmd.index("--pipeline-parallel-size")+1] == "2"
    assert env["TTD_M1_PHASE1_MAX_TOKENS"] == "6656"
    assert env["TTD_M1_TRAIN_MAX_SEQ"] == "10240"
    assert "gpt_oss_high_reasoning" in env["TTD_ENSEMBLE_MODELS"]
    assert cmd[cmd.index("--max-loras")+1] == "2"
    assert trainer_backend_config(cfg, tmp_path, "head", ["head"])["independent_lora_init"]
    archive, _, task = build(profile, tmp_path / "build")
    assert "us-east5-b" in task.read_text()
    with tarfile.open(archive) as bundle:
        bundle.extractall(tmp_path / "unpack", filter="data")
    overlay = tmp_path / "unpack/tpu/swarm/ray_train/source_overlay"
    assert len(identity(overlay)) == 64
    install(overlay, tmp_path / "runtime")
    assert "max_loras == 1" in (tmp_path / "runtime/tpu/vllm_tpu_server.py").read_text()
    assert "sampling_phase" in (tmp_path / "runtime/third_party/discover/ttt_discover/rl/multi_lora.py").read_text()
    (overlay / "skyrl/backends/lora_init.py").write_text("tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        install(overlay, tmp_path / "runtime2")










@pytest.mark.parametrize("changes", [{"adapter_count": 3}, {"importance_cap": float("nan")},
                                      {"inference": {"max_loras": 1}},
                                      {"trainer": {"minimal_fb_output": True}}])
def test_invalid_population_config(changes):
    raw = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_multi_lora.json").to_dict()
    raw.update(changes)
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_http_two_adapters_replace_one_preserve_other_and_retry_failed_fanout(tmp_path, monkeypatch):
    async def run():
        raw = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_multi_lora.json").to_dict()
        raw["root"] = str(tmp_path)
        raw["ready_timeout"] = 1
        ips = ["10.0.0.1", "10.0.0.2"]
        catalog = serving.Catalog.__ray_metadata__.modified_class(ips, 3)
        for ip in ips:
            catalog.register(ip, ip)
        engines = {ip: set() for ip in ips}
        fail = set()

        async def transport(request):
            ip = request.url.host
            if ip in fail:
                raise httpx.ReadTimeout("injected fanout failure", request=request)
            name = request.url.params["lora_name"]
            previous = request.url.params.get("previous_lora_name")
            engines[ip].discard(previous)
            engines[ip].add(name)
            return httpx.Response(200, json={})

        original = serving.Ingress.func_or_class.__mro__[1]
        routed = []
        def generate(pair, payload):
            routed.append((pair, payload['model']))
            return payload
        handles = [SimpleNamespace(generate=Remote(lambda p, pair=pair: generate(pair, p)))
                   for pair in range(2)]
        gateway = original(raw, handles,
            SimpleNamespace(commit=Remote(catalog.commit), snapshot=Remote(catalog.snapshot)), ips)
        await gateway.http.aclose()
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        monkeypatch.setattr(serving.serve, "get_replica_context", lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals["frozen_app_or_func"]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ingress") as client:
                async def upload(name, previous=None):
                    params = {"lora_name": name}
                    if previous:
                        params["previous_lora_name"] = previous
                    return await client.post("/skyrl/v1/upload_lora_adapter", params=params, content=name.encode())

                from dataclasses import replace
                original_config = gateway.config
                gateway.config = replace(original_config, inference=replace(original_config.inference, max_adapter_upload_bytes=2))
                assert (await upload("too-large")).status_code == 413
                assert catalog.snapshot()["versions"] == []
                gateway.config = original_config
                assert (await upload("A-0")).status_code == 200
                assert (await upload("B-0")).status_code == 200
                assert set(catalog.snapshot()["versions"]) == {"A-0", "B-0"}
                assert (await upload("C-0")).status_code == 409
                for name in ("A-0", "B-0"):
                    assert (await client.post("/v1/completions", json={"model": name})).status_code == 200
                assert routed == [(0, "A-0"), (1, "B-0")]
                fail.add(ips[-1])
                assert (await upload("A-1", "A-0")).status_code == 503
                assert gateway.updating
                assert (await client.post("/v1/completions", json={"model": "B-0"})).status_code == 409
                assert set(catalog.snapshot()["versions"]) == {"A-0", "B-0"}
                fail.clear()
                assert (await upload("A-1", "A-0")).status_code == 200
                assert all(names == {"A-1", "B-0"} for names in engines.values())
                assert set(catalog.snapshot()["versions"]) == {"A-1", "B-0"}
                assert (await upload("A-1", "A-0")).status_code == 200
                assert (await upload("A-0")).status_code == 409
                assert (await client.post("/v1/completions", json={"model": "A-0"})).status_code == 409
                assert (await client.post("/v1/completions", json={"model": "B-0"})).status_code == 200
                assert not catalog.register(ips[0], "stale-replacement", ["A-0", "B-0"])
                assert catalog.register(ips[0], "fresh-replacement", ["A-1", "B-0"])
        finally:
            await gateway.http.aclose()
    asyncio.run(run())
