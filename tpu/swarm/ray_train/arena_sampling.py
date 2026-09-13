"""Bounded multi-model sampling with an owned, single-host RG-LRU Ray judge."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time


def judge_ray_tmp(root):
    return Path.home() / (".rar-" + hashlib.sha256(str(root).encode()).hexdigest()[:10])


def judge_env(host):
    code = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=f"{code / 'tpu'}:{code}",
               JAX_PLATFORMS="cpu", ARENA_CHILD_JAX_PLATFORMS="cpu",
               ARENA_BASELINE="all", PALLAS_INTERPRET="0", ARENA_RLIMIT_GB="64",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAY_USAGE_STATS_ENABLED="0")
    for key in ("RAY_ADDRESS", "RAY_NAMESPACE", "RAY_TMPDIR", "TPU_VISIBLE_CHIPS",
                "TPU_PROCESS_ADDRESSES", "TPU_PROCESS_PORT", "TPU_PROCESS_BOUNDS",
                "TPU_CHIPS_PER_PROCESS_BOUNDS", "CLOUD_TPU_TASK_ID", "JAX_COORDINATOR_ADDRESS"):
        env.pop(key, None)
    return env


def prepare_host(host):
    if host.rank != 0 or not host.config.arena_samples:
        raise ValueError("Arena judge must own the omitted inference head")
    from .bootstrap import check_ports_available, stop_ray
    stop_ray(judge_ray_tmp(host.root))
    check_ports_available([8791])
    folder = host.root / "envs/arena"
    python = str(folder / "bin/python")
    marker = folder / ".complete"
    identity = "jax0.10.2-recurrentgemma1.0.1-jax-extra-ray2.58-v1"
    env = judge_env(host)
    if not marker.exists() or marker.read_text() != identity:
        # A failed attempt may leave an environment without its completion
        # marker. Recreate only this run root's owned judge environment.
        host.checked("arena-venv", ["uv", "venv", "--clear", "--python", "3.12", str(folder)])
        host.checked("arena-install", ["uv", "pip", "install", "--python", python,
            "jax[tpu]==0.10.2", "recurrentgemma[jax]==1.0.1", "ray[default]==2.58.0",
            "numpy", "fastapi", "uvicorn", "pydantic", "flatbuffers", "google-cloud-storage",
            "flax", "einops", "sentencepiece", "einshape", "xprof", "httpx"], env=env)
        host.checked("arena-import", [python, "-c",
            "import jax,ray; from recurrentgemma.jax.pallas import lru_pallas_scan; "
            "from pallas_arena.judge.worker import PersistentWorker; "
            "assert jax.__version__=='0.10.2'; assert ray.__version__=='2.58.0'"], env=env)
        marker.write_text(identity)
    config_file = host.run / "arena-config.json"
    config_file.write_text(json.dumps(host.config.to_dict()))
    (host.run / "arena-pool-ready.json").unlink(missing_ok=True)
    host.start("arena-queue", [python, "-m", "pallas_arena.judge.queue",
        "--host", "0.0.0.0" if host.config.arena_service_only else "127.0.0.1", "--port", "8791", "--lease-timeout", "1800"], env=env)
    host.start("arena-pool", [python, "-m", "tpu.swarm.ray_train.arena_sampling",
        "pool", str(config_file)], env=env)
    import httpx
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if any(host.processes[n].poll() is not None for n in ("arena-queue", "arena-pool")):
            raise RuntimeError("Arena service exited during startup")
        try:
            response = httpx.get("http://127.0.0.1:8791/status", timeout=2)
            response.raise_for_status()
            if (host.run / "arena-pool-ready.json").exists():
                host.phase = "judge_ready"
                return host.heartbeat()
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("Arena queue/pool readiness deadline")


def start_sampling(host):
    python = str(host.root / "envs/arena/bin/python")
    host.start("client", [python, "-m", "tpu.swarm.ray_train.arena_sampling",
        "sample", str(host.run / "arena-config.json")], env=judge_env(host))
    return host.heartbeat()


def run_pool(config):
    import ray
    from pallas_arena.judge.ray_pool import run_pool as grade, cpu_per_task
    from pallas_arena.rl.task import public_contract
    root = Path(config.root).expanduser().resolve()
    run = root / "runs" / config.run_id
    cache = root / f"arena-cache/{config.accelerator}/jax-0.10.2"
    # Head advertises TPU:0 to workload Ray; this private local runtime alone
    # owns its four chips. Never connect judge workers to the serving runtime.
    ray.init(address="local", _node_ip_address="127.0.0.1", num_cpus=max(32, 4 * cpu_per_task()),
        resources={"TPU": 4}, object_store_memory=1024**3, include_dashboard=False,
        _temp_dir=str(judge_ray_tmp(root)))
    try:
        (run / "arena-pool-ready.json").write_text(json.dumps({"chips": 4, "cases": public_contract()[1]}))
        grade("http://127.0.0.1:8791", ["rg_lru"], chips=4, max_tp_width=1, cfg={
            "cases_by_problem": {"rg_lru": [n for n, _ in public_contract()[1]]},
            "baseline": "all", "cache": None,
            "compile_cache_dir": str(cache), "timing_pairs": 20,
            "compile_budget_s": 180, "grade_budget_s": 900, "ray_address": "auto"})
    finally:
        ray.shutdown()


def render_prompt(preset, question):
    """Text-only equivalents of the repository's Qwen/Gemma/Muse renderers.

    Explicit BOS is essential for Gemma and Muse. Tokenize with
    add_special_tokens=False so the server cannot prepend a second BOS.
    """
    system = "You write correct, efficient TPU Pallas kernels."
    if preset == "qwen3.5-27b":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"), ["<|im_end|>", "<|endoftext|>"]
    if preset == "gemma4-31b":
        return (f"<bos><|turn>system\n<|think|>\n{system}<turn|>\n"
                f"<|turn>user\n{question}<turn|>\n<|turn>model\n"), ["<turn|>"]
    if preset == "muse-glimmer-30b":
        return (f"<|begin_of_text|><|start|>system<|message|>{system}"
                '\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|>'
                f"<|start|>user<|message|>{question}<|eot|><|start|>assistant"), ["<|eot|>", "<|end_of_text|>"]
    raise ValueError(f"unsupported arena renderer: {preset}")


def extract_code(text, preset="qwen3.5-27b"):
    separator = {"qwen3.5-27b": "</think>", "gemma4-31b": "<channel|>",
                 "muse-glimmer-30b": "to=user<|message|>"}[preset]
    if separator not in text:
        raise ValueError("model did not reach its answer channel")
    answer = text.rsplit(separator, 1)[-1]
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", answer, re.DOTALL)
    if not blocks:
        raise ValueError("no complete Python code block in model answer")
    return blocks[-1].strip() + "\n"


def generation_payload(model, prompt, *, max_tokens=8192, stops=None):
    # TPU ingress rejects per-request seeds; use the engine's stochastic RNG.
    return {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.8, "top_p": 0.95, "skip_special_tokens": False,
            "add_special_tokens": False,
            "stop": stops or ["<|im_end|>", "<|endoftext|>"]}


def summarize_model(model, records, generation_s, replicas=1):
    import statistics
    samples = [r for r in records if r.get("model") == model and r["kind"] == "sample"]
    scores = [r["translated"]["raw_score"] for r in samples
              if r.get("verdict", {}).get("passed") is True and "translated" in r]
    tokens = sum(r.get("response", {}).get("usage", {}).get("completion_tokens", 0) for r in samples)
    forced = sum(len(r.get("response", {}).get("choices", [{}])[0].get("forced_token_positions", [])) for r in samples)
    latencies = [r["generation_s"] for r in samples if "response" in r and "generation_s" in r]
    p95 = statistics.quantiles(latencies, n=20, method="inclusive")[-1] if len(latencies) > 1 else (latencies[0] if latencies else None)
    return dict(model=model, samples=len(samples), passed=len(scores),
        completed_requests=len(latencies), inference_replicas=replicas,
        forced_tokens=forced, sampled_tokens=tokens-forced,
        sampled_tokens_per_s=(tokens-forced)/generation_s if generation_s else None,
        output_tokens_per_s_per_engine=tokens/generation_s/replicas if generation_s else None,
        request_latency_p50_s=statistics.median(latencies) if latencies else None,
        request_latency_p95_s=p95,
        infrastructure_errors=sum("infrastructure_error" in r for r in samples),
        format_errors=sum("format_error" in r for r in samples),
        truncated=sum(r.get("finish_reason") == "length" for r in samples),
        best_speedup=max(scores) if scores else None,
        median_valid_speedup=statistics.median(scores) if scores else None,
        generation_s=generation_s, output_tokens=tokens,
        output_tokens_per_s=tokens/generation_s if generation_s else None)


def model_configs(config):
    """One sampling cohort per model, independent of its serving replica count."""
    configs = ([config.for_inference_rank(r) for r in config.inference_only_ranks]
               if config.arena_models else [config])
    return list({child.model: child for child in configs}.values())


async def sample(config):
    import httpx
    from pallas_arena.judge.client import ArenaQueueClient
    from pallas_arena.rl.task import build_prompt, public_contract, translate_verdict, ARENA
    root = Path(config.root).expanduser().resolve()
    output = root / "runs" / config.run_id / "client/arena"
    output.mkdir(parents=True, exist_ok=True)
    seed = (ARENA / "rl/seed_rglru.py").read_text()
    question = (build_prompt() + "\nStarting implementation:\n```python\n" + seed + "\n```\n"
                + f"Optimize this implementation for {config.accelerator}. "
                  "Preserve forward and backward correctness. Return a complete program.\n")
    (output / "question.txt").write_text(question)
    (output / "seed.py").write_text(seed)
    cases = [n for n, _ in public_contract()[1]]
    records, pending = [], []
    queue = ArenaQueueClient("http://127.0.0.1:8791")
    started = time.monotonic()

    def save(record):
        (output / (record["id"] + ".json")).write_text(json.dumps(record, indent=2))

    async def submit(record, code):
        wid = await asyncio.to_thread(queue.submit, "rg_lru", code, mode="full",
            smoke=False, cases=cases, enforce_pallas=True, tag=record["id"])
        record["work_id"] = wid
        (output / (record["id"] + ".py")).write_text(code)
        pending.append((record, wid))
        save(record)

    async def drain(timeout=7200):
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            done = await asyncio.to_thread(queue.poll_bulk, [wid for _, wid in pending])
            for record, wid in list(pending):
                if wid not in done:
                    continue
                result = done[wid]["result"]
                record["verdict"] = result
                try:
                    record["translated"] = translate_verdict(result)
                except Exception as exc:
                    record["infrastructure_error"] = str(exc)
                save(record)
                print(json.dumps({"event": "graded", "id": record["id"],
                    "passed": result.get("passed"), "gate": result.get("gate"),
                    "translated": record.get("translated")}), flush=True)
                pending.remove((record, wid))
            if pending:
                await asyncio.sleep(2)
        for record, wid in pending:
            record["infrastructure_error"] = f"grading timeout: {wid}"
            save(record)
        pending.clear()

    # Fail closed before generating candidates if the fixed control does not
    # pass the entire forward/backward contract on the actual target hardware.
    control = {"id": "seed", "kind": "seed"}
    records.append(control)
    await submit(control, seed)
    await drain(1800)
    if not control.get("translated", {}).get("correctness"):
        (output / "summary.json").write_text(json.dumps({"phase": "seed_preflight_failed", "seed": control}, indent=2))
        raise RuntimeError("seed failed target-hardware preflight; see seed.json")
    print(json.dumps({"event": "seed_preflight_passed", "translated": control["translated"]}), flush=True)

    configs = model_configs(config)
    base = f"http://127.0.0.1:{config.ports.inference}"
    prompts, warmups, durations = {}, {}, {}
    async with httpx.AsyncClient(timeout=config.inference.request_timeout) as http:
        async def prepare(model_config):
            model = model_config.model
            prompt, stops = render_prompt(model_config.model_preset, question)
            response = await http.post(base + "/tokenize", json={"model": model,
                "prompt": prompt, "add_special_tokens": False})
            response.raise_for_status()
            tokens = response.json()["tokens"]
            if len(tokens) + config.arena_max_tokens > model_config.inference.max_model_length:
                raise RuntimeError(f"{model}: prompt {len(tokens)} plus output exceeds context")
            prompts[model] = generation_payload(model, tokens, max_tokens=config.arena_max_tokens, stops=stops)
            if config.arena_thinking_tokens is not None:
                prompts[model]["thinking_token_budget"] = config.arena_thinking_tokens
            (output / (model_config.model_preset + "-prompt.json")).write_text(json.dumps(
                {"text": prompt, "tokens": tokens, "stops": stops, "model": model}, indent=2))
            begin = time.monotonic()
            async def warmup(index):
                response = await http.post(base + "/v1/completions", json={**prompts[model], "max_tokens": 64,
                    **({"thinking_token_budget": 8} if config.arena_thinking_tokens is not None else {})})
                response.raise_for_status()
                data = response.json()
                if not data.get("choices") or not data["choices"][0].get("text"):
                    raise RuntimeError(f"{model}: warmup returned no text")
                return data
            results = await asyncio.gather(*(warmup(i) for i in range(config.arena_concurrency)))
            warmups[model] = {"seconds": time.monotonic()-begin, "responses": results}
            (output / (model_config.model_preset + "-warmup.json")).write_text(json.dumps(warmups[model], indent=2))
            print(json.dumps({"event": "warmup_complete", "model": model, "prompt_tokens": len(tokens),
                              "seconds": warmups[model]["seconds"]}), flush=True)
        await asyncio.gather(*(prepare(c) for c in configs))

        async def model_samples(model_config):
            sem = asyncio.Semaphore(config.arena_concurrency)
            begin_model = time.monotonic()
            async def generate(index):
                record = {"id": f"{model_config.model_preset}-{index:03d}", "kind": "sample",
                          "model": model_config.model, "model_preset": model_config.model_preset}
                records.append(record)
                async with sem:
                    begin = time.monotonic()
                    try:
                        response = await http.post(base + "/v1/completions", json=prompts[model_config.model])
                        response.raise_for_status()
                        data = response.json()
                        if config.arena_thinking_tokens is not None:
                            audit = data["choices"][0].get("thinking_budget", {})
                            if not audit.get("enforced") or audit.get("thinking_tokens", float("inf")) > config.arena_thinking_tokens:
                                raise RuntimeError("server did not verify the requested thinking cap")
                            record["thinking_budget"] = audit
                        record.update(response=data, generation_s=time.monotonic()-begin,
                                      finish_reason=data["choices"][0].get("finish_reason"))
                        code = extract_code(data["choices"][0]["text"], model_config.model_preset)
                        await submit(record, code)
                    except httpx.HTTPStatusError as exc:
                        record.update(infrastructure_error=str(exc), server_error=exc.response.text[:4000])
                    except ValueError as exc:
                        record["format_error"] = str(exc)
                    except Exception as exc:
                        record["infrastructure_error"] = str(exc)
                    save(record)
                    print(json.dumps({k:v for k,v in record.items() if k != "response"}), flush=True)
            await asyncio.gather(*(generate(i) for i in range(config.arena_samples)))
            durations[model_config.model] = time.monotonic()-begin_model
            metrics = summarize_model(model_config.model, records, durations[model_config.model],
                                      config.arena_replicas_per_model)
            # Publish speed as soon as generation ends, without waiting for grading.
            for key in ("passed", "best_speedup", "median_valid_speedup"):
                metrics.pop(key)
            speed = {"event": "generation_speed", **metrics}
            (output / (model_config.model_preset + "-speed.json")).write_text(json.dumps(speed, indent=2))
            print(json.dumps(speed), flush=True)
        await asyncio.gather(*(model_samples(c) for c in configs))
    await drain()

    # Reward caching is disabled in this bounded experiment: the selected
    # winner gets genuinely new timing pairs, even when its source is identical.
    for model_config in configs:
        valid = [r for r in records if r.get("model") == model_config.model and r["kind"] == "sample"
                 and r.get("translated", {}).get("correctness")]
        if valid:
            best = max(valid, key=lambda r: r["translated"]["raw_score"])
            record = {"id": model_config.model_preset + "-retest", "kind": "retest",
                      "model": model_config.model, "original_id": best["id"]}
            records.append(record)
            await submit(record, (output / (best["id"] + ".py")).read_text())
    await drain(1800)
    summary = {"accelerator": config.accelerator, "samples_per_model": config.arena_samples,
        "concurrency_per_model": config.arena_concurrency, "max_tokens": config.arena_max_tokens,
        "thinking_token_budget": config.arena_thinking_tokens,
        "inference_replicas_per_model": config.arena_replicas_per_model,
        "elapsed_s": time.monotonic()-started,
        "models": [summarize_model(c.model, records, durations[c.model], config.arena_replicas_per_model) for c in configs],
        "infrastructure_errors": sum("infrastructure_error" in r for r in records),
        "results": [{k:v for k,v in r.items() if k not in ("response", "verdict")} |
                    {"passed": r.get("verdict",{}).get("passed"), "gate": r.get("verdict",{}).get("gate")}
                    for r in records]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return int(summary["infrastructure_errors"] > 0)


if __name__ == "__main__":
    import sys
    from .config import Config
    config = Config.load(sys.argv[2])
    if sys.argv[1] == "pool":
        run_pool(config)
    elif sys.argv[1] == "sample":
        raise SystemExit(asyncio.run(sample(config)))
    else:
        raise SystemExit("expected pool or sample")
