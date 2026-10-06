import asyncio
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "tpu/swarm/ray_serve_canary"
spec = importlib.util.spec_from_file_location("canary_common", ROOT / "common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
cleanup_spec = importlib.util.spec_from_file_location("canary_cleanup", ROOT / "stop_ray.py")
cleanup = importlib.util.module_from_spec(cleanup_spec)
cleanup_spec.loader.exec_module(cleanup)


def test_generation_requests_do_not_send_unsupported_seed():
    tree = ast.parse((ROOT / "client.py").read_text())
    requests = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                and any(isinstance(key, ast.Constant) and key.value == "prompt"
                        for key in node.keys)]
    assert requests
    for request in requests:
        assert "seed" not in [key.value for key in request.keys if isinstance(key, ast.Constant)]


def test_rerun_uses_unique_immutable_adapter_versions():
    client = (ROOT / "client.py").read_text()
    assert "run_id = uuid.uuid4().hex" in client
    assert 'f"canary_{run_id}_a"' in client
    assert 'f"canary_{run_id}_b"' in client
    assert 'upload("canary_step_' not in client


def test_cleanup_without_pidfd_wait(monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(cleanup.psutil, "wait_procs", unsupported)
    clock = [0.0]
    monkeypatch.setattr(cleanup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cleanup.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class Process:
        pid = 123
        running = True
        killed = False

        def terminate(self):
            pass

        def kill(self):
            self.killed = True
            self.running = False

        def is_running(self):
            return self.running

        def status(self):
            return cleanup.psutil.STATUS_RUNNING

    process = Process()
    cleanup.stop_owned([process])
    assert process.killed
    assert 15 <= clock[0] <= 20


def test_cleanup_does_not_kill_zombies_or_reused_pids():
    class Process:
        def __init__(self, running, status):
            self.running, self.state = running, status

        def is_running(self):
            return self.running

        def status(self):
            return self.state

    assert not cleanup.wait_owned([
        Process(False, cleanup.psutil.STATUS_RUNNING),
        Process(True, cleanup.psutil.STATUS_ZOMBIE),
    ], timeout=0)


@pytest.mark.parametrize("name", ["../x", "/x", "", "a/b", ".hidden", "x" * 129, None])
def test_reject_unsafe_version(name):
    with pytest.raises(ValueError):
        common.adapter_name(name)


def test_density_grade_matches_constant_construction():
    result = common.grade_construction('{"h_values": [0.5,0.5,0.5,0.5]}')
    assert result["valid"]
    assert result["c5"] == pytest.approx(0.5)


@pytest.mark.parametrize("text", ["not JSON", '{"h_values": [0,0,0,0]}',
                                  '{"h_values": [2,0,0,0]}', '{"h_values": [NaN,0,0,0]}'])
def test_invalid_constructions(text):
    assert not common.grade_construction(text)["valid"]


def test_update_waits_for_inflight_and_failure_stays_closed():
    async def check():
        gate = common.VersionGate()
        with pytest.raises(ValueError):
            await gate.enter(None)
        await gate.commit("a")
        await gate.enter("a")
        task = asyncio.create_task(gate.begin_update())
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(ValueError):
            await gate.enter("a")
        await gate.leave()
        await task
        with pytest.raises(ValueError):
            await gate.enter("b")
        await gate.commit("b")
        with pytest.raises(ValueError):
            await gate.enter("a")
        await gate.enter("b")
        await gate.leave()
    asyncio.run(check())


def test_no_production_mutations_or_global_ray_stop():
    shell = (ROOT / "run.sh").read_text()
    assert "ray stop" not in shell
    assert "pkill" not in shell
    assert "6380" not in shell
    assert '"TPU":0' in shell
    assert "canary_infer" in shell
    infer = shell.split('elif [[ "$SKYPILOT_NODE_RANK" == 1')[1].split('\nfi\n')[0]
    assert "export TPU_VISIBLE_CHIPS=0,1,2,3" in infer
    service = (ROOT / "service.py").read_text()
    assert "TPU_MULTIHOST_BACKEND" in service
    assert "VersionGate" in service
    assert "data-parallel-size" not in service


def test_v4_64_host_count_and_isolated_root():
    import yaml
    config = yaml.safe_load((ROOT / "canary_v4_64.yaml").read_text())
    assert config["resources"]["accelerators"] == "tpu-v4-64"
    assert config["envs"]["CANARY_HOST_COUNT"] == "8"
    assert 'export CANARY_ROOT="$HOME/ray-serve-canary-v4-64-v3"' in config["run"]
    shell = (ROOT / "run.sh").read_text()
    assert "int(os.environ['CANARY_HOST_COUNT'])" in shell
    assert '${CANARY_ROOT:-$HOME/ray-serve-canary-v1}' in shell
    client = (ROOT / "client.py").read_text()
    assert 'int(os.environ.get("CANARY_HOST_COUNT", "4"))' in client
    assert 'len({g["host"] for g in grades}) == len(nodes)' in client


def test_ray_socket_path_is_short_and_cleanup_uses_actual_path():
    import yaml
    config = yaml.safe_load((ROOT / "canary_v4_64.yaml").read_text())
    tmp = config["envs"]["CANARY_RAY_TMPDIR"]
    socket = tmp + "/session_2026-09-06_21-59-45_798689_9999999/sockets/plasma_store"
    assert len(socket.encode()) <= 107
    assert '${CANARY_RAY_TMPDIR:-$CANARY_ROOT/ray-tmp}' in (ROOT / "run.sh").read_text()
    assert 'os.environ["RAY_TMPDIR"]' in (ROOT / "stop_ray.py").read_text()
