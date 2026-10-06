"""Replay a saved request at concurrency 1 and 8 on one idle base-model engine."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
import urllib.error

from .long_decode import counters, fetch, require_idle


def validate_payload(record, model):
    payload = record.get("request", record)
    if (payload.get("model") != model or payload.get("seed") is not None
            or payload.get("n", 1) != 1 or payload.get("stream", False)
            or not 1 <= payload.get("max_tokens", 0) <= 18432
            or not payload.get("prompt")):
        raise ValueError("expected bounded, non-streaming n=1 base-model payload without seed")
    return payload


def sample(endpoint, ip):
    return dict(time=time.time(), status=json.loads(fetch(endpoint + "/status")),
                metrics=counters(fetch(f"http://{ip}:19801/metrics")))


def request_payload(payload, index, distinct_prefix):
    if not distinct_prefix:
        return payload
    prompt = payload["prompt"]
    if not isinstance(prompt, list) or not all(type(token) is int for token in prompt):
        raise ValueError("distinct-prefix diagnostic requires a token-ID prompt")
    return dict(payload, prompt=[1000 + index] + prompt[1:])


def trial(endpoint, ip, payload, count, output, timeout, distinct_prefix=False):
    initial = sample(endpoint, ip)
    require_idle(initial["status"], [ip], {ip: initial["metrics"]})
    if len(initial["status"]["replicas"]) != 1:
        raise RuntimeError("expected exactly one engine")
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(payload, indent=2))
    (output / "initial.json").write_text(json.dumps(initial, indent=2))
    stopped = threading.Event()
    samples = []
    barrier = threading.Barrier(count)

    def watch():
        with (output / "progress.jsonl").open("w") as stream:
            while not stopped.is_set():
                try:
                    row = sample(endpoint, ip)
                except Exception as exc:
                    row = dict(time=time.time(), error=str(exc))
                samples.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stopped.wait(1)

    def generate(index):
        request = request_payload(payload, index, distinct_prefix)
        barrier.wait(timeout=30)
        started = time.monotonic()
        try:
            response = json.loads(fetch(endpoint + "/v1/completions", request, timeout))
            if response.get("error") or not response.get("choices") or "usage" not in response:
                raise ValueError("invalid response: " + str(response)[:2000])
            return dict(index=index, seconds=time.monotonic()-started, request=request, response=response)
        except Exception as exc:
            body = exc.read().decode(errors="replace") if isinstance(exc, urllib.error.HTTPError) else None
            return dict(index=index, seconds=time.monotonic()-started, request=request, error=str(exc), body=body)

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    started = time.monotonic()
    records = []
    try:
        with (output / "responses.jsonl").open("w") as stream:
            with ThreadPoolExecutor(max_workers=count) as executor:
                for future in as_completed([executor.submit(generate, i) for i in range(count)]):
                    row = future.result()
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(dict(concurrency=count, completed=len(records),
                                          seconds=row["seconds"], error=row.get("error"))), flush=True)
    finally:
        stopped.set()
        monitor.join(timeout=30)
    elapsed = time.monotonic()-started
    summary = dict(concurrency=count, distinct_prefix=distinct_prefix, seconds=elapsed, successes=sum("error" not in r for r in records),
                   peak_running=max((r.get("metrics", {}).get("vllm:num_requests_running", 0) for r in samples), default=0),
                   completion_tokens=sum(r.get("response", {}).get("usage", {}).get("completion_tokens", 0) for r in records))
    try:
        final = sample(endpoint, ip)
        summary["final"] = final
        summary["restarted"] = (final["status"]["starts"] != initial["status"]["starts"]
                                or final["status"]["replicas"] != initial["status"]["replicas"])
    except Exception as exc:
        summary["final_error"] = str(exc)
    summary["passed"] = summary["successes"] == count and summary.get("restarted") is False
    summary["batch_concurrency_observed"] = summary["peak_running"] >= count
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.ready_timeout <= 3600 or not 1 <= args.request_timeout <= 3600:
        parser.error("timeouts must be 1..3600 seconds")
    deadline = time.monotonic() + args.ready_timeout
    while True:
        try:
            fetch(args.endpoint + "/health")
            break
        except Exception as exc:
            if time.monotonic() > deadline:
                raise TimeoutError("engine readiness deadline exceeded") from exc
            print(json.dumps(dict(waiting_ready=True, detail=str(exc))), flush=True)
            time.sleep(30)
    model = json.loads(fetch(args.endpoint + "/v1/models"))["data"][0]["id"]
    payload = validate_payload(json.loads(args.case.read_text()), model)
    args.output.mkdir(parents=True, exist_ok=False)
    for count in (1, 8):
        summary = trial(args.endpoint, args.ip, payload, count, args.output / f"c{count}", args.request_timeout)
        if not summary["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
