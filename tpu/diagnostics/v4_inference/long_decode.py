"""One bounded base-model decode control against an explicitly idle Ray endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
import urllib.request


def fetch(url, payload=None, timeout=10):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def counters(raw):
    values = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key = line.split("{")[0].split()[0]
        if key in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:generation_tokens_total"):
            values[key] = values.get(key, 0) + float(line.rsplit(" ", 1)[1])
    return values


def require_idle(status, expected_ips, metrics):
    if (status.get("active") != 0 or status.get("updating") is not False
            or status.get("version") is not None or status.get("committed") is not None
            or status.get("exhausted")
            or {r["ip"] for r in status["replicas"]} != set(expected_ips)):
        raise RuntimeError("refusing non-idle, adapter-bearing or unexpected endpoint")
    for ip in expected_ips:
        if any(metrics[ip].get(key) != 0 for key in
               ("vllm:num_requests_running", "vllm:num_requests_waiting")):
            raise RuntimeError("engine has outstanding or unknown requests")


def run(args):
    initial = json.loads(fetch(args.endpoint + "/status"))
    ips = args.expected_ips.split(",")
    metrics = {ip: counters(fetch(f"http://{ip}:19801/metrics")) for ip in ips}
    require_idle(initial, ips, metrics)
    model = json.loads(fetch(args.endpoint + "/v1/models"))["data"][0]["id"]
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(time=time.time(), initial_status=initial, metrics=metrics, model=model,
                requests=args.requests, tokens=args.tokens, timeout=args.timeout, expected_ips=ips,
                temperature=0.7, ignore_eos=True, adapter=None, logprobs=None,
                workload="short prompt, long forced decode; no training or grading")
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    stopped = threading.Event()

    def watch():
        with (args.output / "progress.jsonl").open("w") as output:
            while not stopped.is_set():
                sample = dict(time=time.time(), engines={})
                try:
                    sample["status"] = json.loads(fetch(args.endpoint + "/status"))
                except Exception as exc:
                    sample["error"] = str(exc)
                for ip in ips:
                    try:
                        sample["engines"][ip] = counters(fetch(f"http://{ip}:19801/metrics"))
                    except Exception as exc:
                        sample["engines"][ip] = {"error": str(exc)}
                output.write(json.dumps(sample) + "\n")
                output.flush()
                stopped.wait(30)

    def generate(index):
        payload = dict(model=model, prompt=f"Problem {index:03d}: Explain a proof that there are infinitely many prime numbers, with detailed mathematical reasoning.\nSolution:",
                       max_tokens=args.tokens, temperature=0.7, ignore_eos=True, stream=False)
        start = time.monotonic()
        try:
            response = json.loads(fetch(args.endpoint + "/v1/completions", payload, args.timeout))
            if response.get("error") or not response.get("choices") or "usage" not in response:
                raise ValueError("invalid completion response")
            return dict(index=index, seconds=time.monotonic()-start, payload=payload, response=response)
        except Exception as exc:
            return dict(index=index, seconds=time.monotonic()-start, payload=payload, error=str(exc))

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    start = time.monotonic()
    records = []
    try:
        with (args.output / "responses.jsonl").open("w") as output:
            with ThreadPoolExecutor(max_workers=args.requests) as pool:
                for future in as_completed([pool.submit(generate, i) for i in range(args.requests)]):
                    record = future.result()
                    records.append(record)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(json.dumps(dict(completed=len(records), seconds=time.monotonic()-start,
                                          error=record.get("error"))), flush=True)
    finally:
        stopped.set()
        monitor.join(timeout=60)
    elapsed = time.monotonic()-start
    failures = sum("error" in record for record in records)
    tokens = sum(record.get("response", {}).get("usage", {}).get("completion_tokens", 0) for record in records)
    summary = dict(seconds=elapsed, failures=failures, successes=len(records)-failures,
                   completion_tokens=tokens, tokens_per_second=tokens/elapsed,
                   requested_tokens=args.requests*args.tokens)
    try:
        final = json.loads(fetch(args.endpoint + "/status"))
        summary["final_status"] = final
        summary["engine_restarted"] = final.get("starts") != initial.get("starts")
    except Exception as exc:
        summary["final_status_error"] = str(exc)
    summary["passed"] = (failures == 0 and tokens == args.requests*args.tokens
                         and summary.get("engine_restarted") is False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--expected-ips", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.requests <= 64 or not 1 <= args.tokens <= 16384 or not 1 <= args.timeout <= 7200:
        parser.error("requests 1..64, tokens 1..16384, timeout 1..7200 required")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
