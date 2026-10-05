"""Change only the verified-idle job396 diagnostic Serve application."""
import json
from pathlib import Path
import time
import urllib.request

from seq8_trial import check_idle, sample


def main():
    run_id = "qwen-ray-v4-64-concurrency-001"
    ip = "10.130.0.105"
    root = Path.home() / ".cache/skyrl-ray"
    run = root / "runs" / run_id
    output = run / "arm-max-seqs-8.json"
    if output.exists():
        raise RuntimeError("arm already configured; inspect it before any repeat")
    initial = sample()
    check_idle(initial, ip, None)
    if initial["status"]["replicas"][0]["instance"] != "dea3e4713f6d4261bdac270fe48337df":
        raise RuntimeError("unexpected diagnostic engine instance")
    models = json.load(urllib.request.urlopen("http://127.0.0.1:19801/v1/models", timeout=5))
    if [m["id"] for m in models["data"]] != ["Qwen/Qwen3.5-27B"]:
        raise RuntimeError("diagnostic engine has an unexpected adapter")
    metadata = json.loads((run / "arm-max-loras-2.json").read_text())
    code = metadata["ray_code"]
    import ray
    from ray import serve
    from tpu.swarm.ray_train.config import Config
    from tpu.swarm.ray_train.serving import Engine, Ingress

    raw = metadata["config"]
    if raw["run_id"] != run_id or not raw["inference_only"] or raw["trainer"]["hosts"] != 0:
        raise RuntimeError("not the expected inference-only job")
    raw["inference"].update(max_sequences=8, max_model_length=22528, tp=4,
                            memory_utilization=0.8, max_loras=1)
    config = Config.from_dict(raw)
    snapshot = models["data"][0]["root"]
    if not Path(snapshot).is_dir():
        raise RuntimeError("cached model snapshot missing")
    runtime = {"env_vars": {"PYTHONPATH": code}}
    ray.init(address="127.0.0.1:19679", namespace=run_id, runtime_env=runtime)
    catalog = ray.get_actor("inference-catalog", namespace=run_id)
    state = ray.get(catalog.snapshot.remote())
    if state["starts"] != initial["status"]["starts"]:
        raise RuntimeError("engine changed during preflight")
    prepared = {ip: dict(rank=0, role="inference", root=str(root),
                         source=str(root / "sources" / config.base_bundle_sha256), snapshot=snapshot)}
    engines = Engine.options(num_replicas=1, ray_actor_options={
        "num_cpus": 8, "resources": {"TPU": 4, f"node:{ip}": 0.01},
        "runtime_env": runtime}).bind(config.to_dict(), prepared, catalog, ip)
    ingress = Ingress.options(ray_actor_options={"num_cpus": 1,
        "resources": {f"node:{ip}": 0.01}, "runtime_env": runtime})
    output.write_text(json.dumps(dict(time=time.time(), prior=initial, config=config.to_dict(),
                                     ray_code=code, intentional_reconfiguration=True), indent=2))
    serve.run_many([serve.RunTarget(target=ingress.bind(config.to_dict(), engines, catalog, [ip]),
                                   name="skyrl-training")],
                   wait_for_ingress_deployment_creation=False, wait_for_applications_running=False)
    print(json.dumps(dict(submitted=True, run=run_id, inference=raw["inference"])), flush=True)


if __name__ == "__main__":
    main()
