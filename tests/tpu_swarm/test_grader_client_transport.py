"""Real Ray lookup from a fresh process with the frozen client's package layout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tpu.swarm.ray_train.grader_actor import Grader
from tpu.swarm.ray_train.overlay import ARENA_FILES


def test_frozen_client_named_actor_transport(tmp_path):
    import ray

    class TestGrader(Grader):
        async def _child(self, mode, payload, tag, chip=None, case=None):
            (self.run / tag).mkdir(exist_ok=True)
            assert mode == "pregate"
            return {"passed": False, "violations": ["synthetic non-Pallas candidate"]}

    source = tmp_path / "frozen"
    missing = {"tpu/swarm/ray_train/grader_actor.py",
               "tpu/pallas_arena/judge/collect.py", "tpu/pallas_arena/judge/timing.py"}
    for name in ARENA_FILES:
        if name in missing:
            continue
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)

    # Use the production evaluator method with only its unrelated Discover
    # base class omitted. Ray lookup, serialization, RPC, verdict translation,
    # and disk evidence are real, with a separate process for each attempt.
    client = r'''
import ast, json, os, sys, uuid, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
source = Path(os.environ['GRADER_SOURCE'])
sys.path[:0] = [str(source / 'tpu'), str(source)]
from pallas_arena.rl.task import public_contract, translate_verdict, ArenaInfrastructureError
tree = ast.parse((source / 'tpu/pallas_arena/rl/env.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RecurrentGemmaRewardEvaluator')
ns = dict(globals(), BaseRewardEvaluator=object, State=object)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'evaluator', 'exec'), ns)
e = ns['RecurrentGemmaRewardEvaluator']('rg_lru', source / 'evidence', eval_timeout=30)
try:
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: e.get_reward('def kernel(x,a,reset): return x', None), range(32)))
    assert all(result['correctness'] == 0 and result['reward'] == 0 for result in results)
    print('CLIENT_TRANSPORT_PASS', flush=True)
finally:
    import ray
    ray.shutdown()
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               GRADER_SOURCE=str(source), ARENA_RAY_ACTOR="rglru-grader",
               RAY_NAMESPACE="grader-client-regression", ARENA_WAIT_TIMEOUT="20")
    ray.init(num_cpus=2, resources={"TPU": 4}, include_dashboard=False,
             _node_ip_address="127.0.0.1", namespace=env["RAY_NAMESPACE"],
             object_store_memory=100 * 1024**2)
    try:
        actor = ray.remote(num_cpus=1, resources={"TPU": 4}, max_concurrency=16)(TestGrader).options(
            name=env["ARENA_RAY_ACTOR"]).remote(str(tmp_path), "actor")
        ray.get(actor.status.remote(), timeout=45)
        env["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        broken = subprocess.run([sys.executable, "-c", client], cwd=source, env=env,
                                capture_output=True, text=True, timeout=60)
        assert broken.returncode != 0
        assert "too many positional arguments" in broken.stderr, broken.stderr
        for name in missing:
            shutil.copyfile(ROOT / name, source / name)
        fixed = subprocess.run([sys.executable, "-c", client], cwd=source, env=env,
                               capture_output=True, text=True, timeout=60)
        assert fixed.returncode == 0, fixed.stdout + fixed.stderr
        assert "CLIENT_TRANSPORT_PASS" in fixed.stdout
        evidence = list((source / "evidence/arena").glob("*.json"))
        assert len(evidence) == 32
        assert json.loads(evidence[0].read_text())["result"]["gate"] == "pregate"
        assert ray.get(actor.status.remote())["completed"] == 32
        ray.get(actor.close.remote(), timeout=10)
        ray.kill(actor, no_restart=True)
    finally:
        ray.shutdown()
