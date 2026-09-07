import asyncio
import inspect
import json
from pathlib import Path
import subprocess
import socket
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("ray.serve")
import httpx
import psutil
from fastapi import HTTPException

from tpu.swarm.ray_train import serving
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.patch_maxtext import OLD, NEW, CONDITION, patch
from tpu.swarm.ray_train.host import Host
from tpu.swarm.ray_train.process import Process
from tpu.swarm.ray_train.bootstrap import check_ports_available


def test_maxtext_patch_is_idempotent_and_rejects_unknown_source(tmp_path):
    path = tmp_path / "models.py"
    path.write_text(OLD)
    assert patch(path)
    assert path.read_text().count(CONDITION) == 1
    assert not patch(path)
    path.write_text("unknown upstream layout")
    with pytest.raises(RuntimeError, match="contract"):
        patch(path)


@pytest.mark.parametrize("initial", [OLD + "\n" + OLD, NEW + "\n" + OLD])
def test_maxtext_patch_handles_both_linen_and_nnx_wrappers(tmp_path, initial):
    path = tmp_path / "models.py"
    path.write_text(initial)
    assert patch(path)
    assert path.read_text().count(CONDITION) == 2
    assert not patch(path)


def test_trainer_ready_handles_wrapped_logs_but_not_previous_process(tmp_path):
    log = tmp_path / "trainer.log"
    message = b"INFO skyrl: Initialized TinkerEngine with           \nbackend=DistributedTunixBackend\n"
    log.write_bytes(message)
    process = SimpleNamespace(poll=lambda: None, log_offset=0)
    host = SimpleNamespace(rank=0, run=tmp_path, processes={"trainer": process})
    assert Host.trainer_ready(host)
    process.log_offset = len(message)
    assert not Host.trainer_ready(host)
    process.log_offset = 0
    process.poll = lambda: 1
    assert not Host.trainer_ready(host)


def test_port_probe_accepts_time_wait_but_rejects_live_listener():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)
        with pytest.raises(RuntimeError, match=f"TCP port {port}"):
            check_ports_available([port], timeout=0)
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = server.accept()
            connection.close()
            assert client.recv(1) == b""
    check_ports_available([port], timeout=0)


def test_previous_adapter_cleanup_preserves_checkpoints_client_and_caches(tmp_path):
    run = tmp_path / "runs/test"
    for name in ("loras", "uploads", "checkpoints", "client"):
        (run / name).mkdir(parents=True)
        (run / name / "payload").write_bytes(b"keep unless derived adapter")
    cache = tmp_path / "ram/hf"
    cache.mkdir(parents=True)
    (cache / "weights").write_bytes(b"warm cache")
    (run / "tinker.db").write_bytes(b"database")
    host = SimpleNamespace(run=run, rank=0, log=run / "host.jsonl")
    Host.clear_previous_adapter_exports(host)
    assert not (run / "loras").exists() and not (run / "uploads").exists()
    assert (run / "checkpoints/payload").exists() and (run / "client/payload").exists()
    assert (run / "tinker.db").read_bytes() == b"database"
    assert (cache / "weights").read_bytes() == b"warm cache"
    (run / "loras").symlink_to(cache)
    with pytest.raises(RuntimeError, match="symlink"):
        Host.clear_previous_adapter_exports(host)
    assert (cache / "weights").exists()


def test_owned_process_cleanup_keeps_unrelated_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    owned = Process([sys.executable, "-u", "-c",
        "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(p.pid,flush=True); time.sleep(120)"], tmp_path / "owned.log")
    try:
        deadline = time.monotonic() + 10
        while not owned.log.read_text().strip() and time.monotonic() < deadline:
            time.sleep(0.1)
        child_pid = int(owned.log.read_text().strip())
        owned.stop()
        assert owned.poll() is not None
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        owned.stop()
        unrelated.terminate()
        unrelated.wait(timeout=5)


class Remote:
    def __init__(self, fn):
        self.remote = fn


class Request:
    def __init__(self, data=b"adapter bytes", payload=None):
        self.data, self.payload = data, payload

    async def stream(self):
        yield self.data

    async def json(self):
        return self.payload


def ingress(tmp_path, transport):
    raw = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64.json").to_dict()
    raw["root"] = str(tmp_path)
    commits = []

    async def commit(version):
        commits.append(version)

    async def snapshot():
        return dict(version=commits[-1] if commits else None, replicas=[{}]*4, exhausted=[])

    async def generate(payload):
        return payload

    original = serving.Ingress.func_or_class.__mro__[1]
    gateway = original(raw, SimpleNamespace(generate=Remote(generate)),
        SimpleNamespace(commit=Remote(commit), snapshot=Remote(snapshot)), [f"10.0.0.{r}" for r in range(4)])
    previous = gateway.http
    gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return gateway, commits, previous


def test_adapter_commit_waits_for_every_engine_and_is_immutable(tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def transport(request):
            if request.url.host == "10.0.0.3":
                entered.set()
                await release.wait()
            return httpx.Response(200, json={})

        gateway, commits, previous = ingress(tmp_path, transport)
        try:
            task = asyncio.create_task(gateway.upload(Request(), "adapter-1"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert gateway.updating and not commits
            with pytest.raises(HTTPException) as exc:
                await gateway.generate(Request(payload={"model": gateway.config.model}))
            assert exc.value.status_code == 409
            release.set()
            result = await asyncio.wait_for(task, timeout=5)
            assert len(result["loaded"]) == 4
            assert commits == ["adapter-1"] and gateway.version == "adapter-1"
            assert not gateway.updating
            await gateway.upload(Request(), "adapter-1")
            with pytest.raises(HTTPException) as exc:
                await gateway.upload(Request(b"different weights"), "adapter-1")
            assert exc.value.status_code == 409
            assert (gateway.archives / "adapter-1.tar").read_bytes() == b"adapter bytes"
            assert not list(gateway.archives.glob("*.partial"))
            assert (await gateway.generate(Request(payload={"model": "adapter-1"})))["model"] == "adapter-1"
            assert gateway.active == 0
        finally:
            release.set()
            await gateway.http.aclose()
            await previous.aclose()
    asyncio.run(run())


def test_ray_ingress_http_upload_and_generation(tmp_path, monkeypatch):
    async def run():
        uploads = []

        async def transport(request):
            uploads.append(await request.aread())
            return httpx.Response(200, json={})

        gateway, commits, previous = ingress(tmp_path, transport)
        monkeypatch.setattr(serving.serve, "get_replica_context",
                            lambda: SimpleNamespace(servable_object=gateway))
        wrapper = serving.Ingress.func_or_class
        app = inspect.getclosurevars(wrapper.__init__).nonlocals["frozen_app_or_func"]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://ingress") as client:
                response = await client.post("/skyrl/v1/upload_lora_adapter",
                    params={"lora_name": "adapter-http"}, content=b"direct trainer weights")
                assert response.status_code == 200, response.text
                assert commits == ["adapter-http"]
                assert uploads == [b"direct trainer weights"] * 4
                response = await client.post("/v1/completions",
                    json={"model": "adapter-http", "prompt": "test", "max_tokens": 1})
                assert response.status_code == 200, response.text
                assert response.json()["model"] == "adapter-http"
        finally:
            await gateway.http.aclose()
            await previous.aclose()
    asyncio.run(run())


def test_failed_adapter_fanout_stays_closed_and_retry_recovers(tmp_path):
    async def run():
        fail = True

        async def transport(request):
            if fail and request.url.host == "10.0.0.3":
                raise RuntimeError("engine failed during load")
            return httpx.Response(200, json={})

        gateway, commits, previous = ingress(tmp_path, transport)
        try:
            with pytest.raises(HTTPException) as exc:
                await gateway.upload(Request(), "adapter-1")
            assert exc.value.status_code == 503
            assert gateway.updating and not commits
            fail = False
            await gateway.upload(Request(), "adapter-1")
            assert not gateway.updating and commits == ["adapter-1"]
            with pytest.raises(HTTPException) as exc:
                await gateway.generate(Request(payload={"model": "adapter-1", "seed": 1}))
            assert exc.value.status_code == 400
        finally:
            await gateway.http.aclose()
            await previous.aclose()
    asyncio.run(run())


def test_replacement_engine_restores_direct_adapter_before_generation(tmp_path):
    async def run():
        loaded = []

        async def transport(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": []})
            if request.url.path.endswith("/archive"):
                return httpx.Response(200, content=b"trainer-to-client-adapter")
            if request.url.path.endswith("upload_lora_adapter"):
                loaded.append(await request.aread())
                return httpx.Response(200, json={})
            assert loaded and request.url.path == "/v1/completions"
            return httpx.Response(200, json={"choices": ["generated"]})

        cls = serving.Engine.func_or_class
        engine = cls.__new__(cls)
        engine.config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64.json")
        engine.version, engine.lock, engine.run = None, asyncio.Lock(), tmp_path
        engine.url, engine.head = "http://engine:19801", "http://head:19800"
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        try:
            assert await engine.generate({"model": "adapter-1"}) == {"choices": ["generated"]}
            assert loaded == [b"trainer-to-client-adapter"]
            assert engine.version == "adapter-1"
            assert not list(tmp_path.glob("*.reload.tar"))
        finally:
            await engine.http.aclose()
    asyncio.run(run())
