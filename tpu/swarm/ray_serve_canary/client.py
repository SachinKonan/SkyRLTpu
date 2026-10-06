"""Exercise upload -> generation -> immediate CPU grading -> replica recovery."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import httpx
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from common import grade_construction

ROOT = Path(os.environ["CANARY_ROOT"])
URL = "http://" + os.environ["CANARY_HEAD_IP"] + ":18000"


def event(kind, **data):
    record = {"event": kind, "time": time.time(), **data}
    print(json.dumps(record), flush=True)
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


@ray.remote(num_cpus=1, resources={"canary_grade": 1}, max_calls=1)
def grade(text, request_id):
    start = time.time()
    result = grade_construction(text)
    return dict(result, request_id=request_id, grade_start=start, grade_end=time.time(),
                host=ray.util.get_node_ip_address(), pid=os.getpid(),
                cluster_id=ray.get_runtime_context().get_job_id(),
                tpu_resources=ray.get_runtime_context().get_accelerator_ids())


async def main():
    ray.init(address=os.environ["RAY_ADDRESS"], namespace="ray-serve-canary")
    nodes = [n for n in ray.nodes() if n["Alive"]]
    assert len(nodes) == int(os.environ.get("CANARY_HOST_COUNT", "4"))
    event("cluster", address=os.environ["RAY_ADDRESS"], nodes=[
        {"ip": n["NodeManagerAddress"], "resources": n["Resources"]} for n in nodes])
    http = httpx.AsyncClient(timeout=1800)
    async def status():
        r = await http.get(URL + "/status", timeout=10)
        r.raise_for_status()
        return r.json()
    try:
        for i in range(720):
            try:
                state = await status()
                if len(state["replicas"]) == 2:
                    event("service_ready", **state)
                    break
            except httpx.HTTPError:
                pass
            if i % 6 == 0:
                event("waiting_service")
            await asyncio.sleep(10)
        else:
            raise RuntimeError("Serve did not become ready in two hours")
        fixture = ROOT / "fixture.tar"
        for i in range(720):
            if fixture.is_file():
                break
            if i % 6 == 0:
                event("waiting_direct_adapter_fixture", path=str(fixture))
            await asyncio.sleep(10)
        else:
            raise RuntimeError("direct-upload adapter fixture was not delivered")

        async def upload(version):
            async def chunks():
                with fixture.open("rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk
            start = time.time()
            r = await http.post(URL + f"/adapters/{version}", content=chunks())
            r.raise_for_status()
            event("adapter_committed", duration=time.time()-start, **r.json())

        counter = 0
        generations = []
        grades = []
        async def request_and_grade(version):
            nonlocal counter
            index = counter
            counter += 1
            prompt = (
                "Complete the JSON object with exactly 16 numeric density values between 0 and 1. "
                "Their sum must equal 8. Output JSON only, no reasoning or code. "
                f"Construction candidate {index}.\n{{\"h_values\": ["
            )
            start = time.time()
            for attempt in range(4):
                r = await http.post(URL + "/v1/completions", json={
                    "model": version, "prompt": prompt, "max_tokens": 256,
                    "temperature": 0.7, "stop": ["}\n"]})
                if r.status_code == 200:
                    break
                event("generation_retry", request_id=index, status=r.status_code,
                      detail=r.text[:300], attempt=attempt)
                await asyncio.sleep(5)
            r.raise_for_status()
            data = r.json()
            finished = time.time()
            text = '{"h_values": [' + data["choices"][0]["text"]
            # stop strings are excluded from the returned completion.
            if not text.rstrip().endswith("}"):
                text += "}"
            event("generation_done", request_id=index, version=version,
                  host=data["canary_replica"], instance=data["canary_instance"],
                  seconds=finished-start, usage=data.get("usage"), text=text)
            generations.append({"time": finished, "host": data["canary_replica"]})
            node = nodes[index % len(nodes)]
            ref = grade.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False)).remote(text, index)
            submitted = time.time()
            event("grade_submitted", request_id=index, delay=submitted-finished,
                  host=node["NodeManagerAddress"])
            result = await ref
            grades.append(dict(result, submitted=submitted))
            event("grade_done", **result)

        run_id = uuid.uuid4().hex
        version_a, version_b = f"canary_{run_id}_a", f"canary_{run_id}_b"
        event("test_run", run_id=run_id, version_a=version_a, version_b=version_b)
        await upload(version_a)
        await asyncio.gather(*(request_and_grade(version_a) for _ in range(8)))
        # Two immutable version IDs exercise update coordination. This fixture
        # does not pretend to be two different optimizer steps.
        await upload(version_b)
        event("fixture_note", distinct_adapter_names=True, distinct_weight_values=False)
        await asyncio.gather(*(request_and_grade(version_b) for _ in range(8)))
        before = await status()
        r = await http.post(URL + "/canary/fail-engine")
        r.raise_for_status()
        victim = r.json()
        killed_at = time.time()
        event("engine_failure_injected", **victim)
        await asyncio.sleep(15)
        await asyncio.gather(*(request_and_grade(version_b) for _ in range(8)))
        recovered_at = None
        for _ in range(360):
            state = await status()
            replacement = next((r for r in state["replicas"] if r["ip"] == victim["ip"]
                                and r["instance"] != victim["instance"]), None)
            if replacement:
                recovered_at = time.time()
                event("replica_recovered", elapsed=recovered_at-killed_at, **replacement)
                break
            await asyncio.sleep(10)
        if recovered_at is None:
            raise RuntimeError("replica did not recover in one hour")
        await asyncio.gather(*(request_and_grade(version_b) for _ in range(8)))
        peers_unchanged = all(any(r["instance"] == old["instance"] for r in state["replicas"])
                              for old in before["replicas"] if old["ip"] != victim["ip"])
        overlap = min(g["submitted"] for g in grades) < max(g["time"] for g in generations[:8])
        peer_during_recovery = any(killed_at < g["time"] < recovered_at and g["host"] != victim["ip"]
                                   for g in generations)
        checks = {"peer_instance_preserved": peers_unchanged, "grading_overlaps_generation": overlap,
                  "healthy_peer_served_during_recovery": peer_during_recovery,
                  "grading_used_all_hosts": len({g["host"] for g in grades}) == len(nodes),
                  "some_valid_constructions": any(g["valid"] for g in grades),
                  "inference_used_both_hosts": len({g["host"] for g in generations}) == 2}
        event("result", passed=all(checks.values()), checks=checks,
              generations=len(generations), grades=len(grades))
        if not all(checks.values()):
            raise RuntimeError(f"failed canary checks: {checks}")
    finally:
        await http.aclose()
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
