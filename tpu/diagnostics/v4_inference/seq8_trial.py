"""Bounded job396 serving control with real RL tokens and memory telemetry."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, payload=None, timeout=10):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def payload_from_fixture(fixture, model, max_tokens, samples=8):
    chunks = fixture["prompt"]["chunks"]
    if any(c["type"] != "encoded_text" for c in chunks):
        raise ValueError("only pre-rendered token chunks are supported")
    tokens = [t for c in chunks for t in c["tokens"]]
    if (not tokens or any(type(t) is not int or t < 0 for t in tokens)
            or not 1 <= samples <= 32 or not 1 <= max_tokens <= 22528 - len(tokens)):
        raise ValueError("invalid token or sample budget")
    p = fixture["sampling_params"]
    result = dict(model=model, prompt=tokens, n=samples, max_tokens=max_tokens,
                  temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"],
                  logprobs=True, stream=False, return_token_ids=True)
    if p.get("stop_tokens"):
        result["stop_token_ids"] = p["stop_tokens"]
    if p.get("stop_strings"):
        result["stop"] = p["stop_strings"]
    return result


def metrics(raw):
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split()[0]
        if name in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                    "vllm:generation_tokens_total", "vllm:kv_cache_usage_perc",
                    "vllm:gpu_cache_usage_perc", "vllm:num_preemptions_total"):
            values[name] = values.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return values


def sample():
    row = dict(time=time.time(), memory={})
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.split(":")[0] in ("MemTotal", "MemAvailable", "Shmem", "SwapFree"):
            row["memory"][line.split(":")[0] + "_kib"] = int(line.split()[1])
    row["engine_memory"] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "cmdline").read_bytes().split(b"\0")[0] != b"VLLM::EngineCore":
                continue
            fields = (proc / "status").read_text().splitlines()
            row["engine_memory"].append(dict(pid=int(proc.name), fields=[line for line in fields
                if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:", "Threads:"))]))
        except OSError:
            continue
    run = Path.home() / ".cache/skyrl-ray/runs/qwen-ray-v4-64-concurrency-001"
    logs = sorted(run.glob("engine-*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        path = logs[-1]
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16384))
            lines = stream.read().decode(errors="replace").splitlines()
        row["engine_log"] = path.name
        row["memory_or_error_lines"] = [line[:1000] for line in lines if any(word in line.lower()
            for word in ("memory", "hbm", "vmem", "resource_exhausted", "allocationfailure", "fatal", "enginedead"))][-12:]
    for key, path in (("status", "19800/status"), ("metrics", "19801/metrics")):
        try:
            raw = fetch("http://127.0.0.1:" + path)
            row[key] = json.loads(raw) if key == "status" else metrics(raw)
        except Exception as exc:
            row[key + "_error"] = str(exc)
    return row


def check_idle(row, ip, version):
    s = row["status"]
    if (s.get("active") != 0 or s.get("updating") is not False or s.get("exhausted")
            or s.get("version") != version or s.get("committed") != version
            or len(s["replicas"]) != 1 or s["replicas"][0]["ip"] != ip
            or any(row["metrics"].get(k) != 0 for k in
                   ("vllm:num_requests_running", "vllm:num_requests_waiting"))):
        raise RuntimeError("diagnostic endpoint is not exclusively idle as expected")


def trial(payload, output, expected_version, ip, timeout):
    initial = sample()
    check_idle(initial, ip, expected_version)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(payload))
    (output / "initial.json").write_text(json.dumps(initial))
    stop = threading.Event()
    def watch():
        with (output / "progress.jsonl").open("w") as stream:
            while not stop.is_set():
                stream.write(json.dumps(sample()) + "\n")
                stream.flush()
                stop.wait(5)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    start = time.monotonic()
    result = {}
    try:
        raw = fetch("http://127.0.0.1:19800/v1/completions", payload, timeout)
        (output / "response.json").write_text(raw)
        data = json.loads(raw)
        result.update(choices=len(data.get("choices", [])), usage=data.get("usage"),
                      finish_reasons=[c.get("finish_reason") for c in data.get("choices", [])])
    except Exception as exc:
        result["error"] = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            (output / "error.txt").write_bytes(exc.read())
    finally:
        stop.set()
        thread.join(timeout=30)
    final = sample()
    result.update(seconds=time.monotonic()-start, final=final)
    result["restarted"] = final.get("status", {}).get("starts") != initial["status"]["starts"]
    result["passed"] = not result.get("error") and not result["restarted"] and result.get("choices") == payload["n"]
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--version")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 7200:
        parser.error("timeout must be 1..7200 seconds")
    payload = payload_from_fixture(json.loads(args.fixture.read_text()), args.model, args.tokens, args.samples)
    raise SystemExit(0 if trial(payload, args.output, args.version, args.ip, args.timeout) else 1)


if __name__ == "__main__":
    main()
