"""Bounded inference-only timing probe; saves responses and engine counters."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


def fetch(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if min(args.requests, args.tokens, args.timeout) < 1:
        parser.error("requests, tokens and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    model = json.loads(fetch(args.endpoint + "/v1/models"))["data"][0]["id"]
    initial = json.loads(fetch(args.endpoint + "/status"))
    if initial.get("active") or initial.get("updating"):
        raise SystemExit("Endpoint must be idle before measuring a new benchmark")
    ips = [item["ip"] for item in initial["replicas"]]
    metadata = dict(model=model, requests=args.requests, output_tokens=args.tokens,
                    endpoint=args.endpoint, start_time=time.time(), initial_status=initial,
                    temperature=0.7, ignore_eos=True, adapter=None, timeout=args.timeout)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def snapshot(label):
        for ip in ips:
            for route in ("health", "metrics"):
                try:
                    value = fetch(f"http://{ip}:19801/{route}")
                except Exception as exc:
                    value = repr(exc)
                (args.output / f"{label}-{ip}-{route}.txt").write_text(value)
        state = json.loads(fetch(args.endpoint + "/status"))
        (args.output / f"{label}-status.json").write_text(json.dumps(state, indent=2))
        return state

    def request_one(index):
        payload = dict(model=model, prompt=f"Problem {index:03d}: Explain a proof that there are infinitely many prime numbers, with detailed mathematical reasoning.\nSolution:",
                       max_tokens=args.tokens, temperature=0.7, ignore_eos=True, stream=False)
        started = time.perf_counter()
        try:
            response = json.loads(fetch(args.endpoint + "/v1/completions", payload, timeout=args.timeout))
            if response.get("error") or "usage" not in response or not response.get("choices"):
                raise ValueError(f"invalid completion: {response}")
            return dict(index=index, seconds=time.perf_counter()-started, payload=payload, response=response)
        except Exception as exc:
            return dict(index=index, seconds=time.perf_counter()-started, payload=payload, error=repr(exc))

    snapshot("before")
    summaries = []
    for label in ("first", "repeat"):
        started = time.perf_counter()
        records = []
        print(json.dumps(dict(event="round_started", round=label, time=time.time())), flush=True)
        with (args.output / f"{label}-responses.jsonl").open("w") as output:
            with ThreadPoolExecutor(max_workers=args.requests) as executor:
                futures = [executor.submit(request_one, i) for i in range(args.requests)]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    if len(records) % 8 == 0 or "error" in record:
                        print(json.dumps(dict(event="progress", round=label, completed=len(records),
                                              seconds=time.perf_counter()-started, error=record.get("error"))), flush=True)
        elapsed = time.perf_counter()-started
        successful = [r for r in records if "response" in r]
        latencies = sorted(r["seconds"] for r in successful)
        tokens = sum(r["response"]["usage"]["completion_tokens"] for r in successful)
        summary = dict(round=label, seconds=elapsed, successes=len(successful), failures=len(records)-len(successful),
                       output_tokens=tokens, output_tokens_per_second=tokens/elapsed,
                       prompt_tokens=sum(r["response"]["usage"]["prompt_tokens"] for r in successful),
                       latency_median=statistics.median(latencies) if latencies else None,
                       latency_p95=latencies[math.ceil(.95*len(latencies))-1] if latencies else None)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
        state = snapshot(label)
        if summary["failures"] or state["starts"] != initial["starts"]:
            raise SystemExit("Stopping benchmark after request failure or engine restart")


if __name__ == "__main__":
    main()
