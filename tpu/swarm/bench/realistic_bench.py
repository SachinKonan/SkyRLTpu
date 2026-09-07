#!/usr/bin/env python3
"""vLLM serving benchmark for the v5p-32 cell shape (3 engine hosts, 1 driver).

Two modes:

* ``realistic`` (default): replay ONE production sampling step against the
  engines -- every state in states.json becomes a group of ``--group-size``
  rollouts, all groups in flight at once, with the real two-phase completion
  (phase-1 reasoning up to the cap, then answer injection), temperature 1.0,
  ``logprobs=1`` and prefix caching, exactly as the RL client drives them.
  Groups are round-robined over ``--bases`` (one URL per engine).
* ``capacity``: the decode-capacity probe (method of tpu/muse_glimmer/
  mg_bench22k.py): long UNIQUE prompts (no prefix-cache help), ``ignore_eos``,
  an untimed warm pass per concurrency, then a short and a long generation
  whose DIFFERENCE gives the steady-state decode tok/s -- swept over a
  concurrency ladder to find the max_num_seqs that still pays.

Stdlib + transformers (tokenizer) only; runs inside the vLLM venv on an engine
host.  Every phase records per-request wall times so p50/p90/p95/max latencies
come out per sequence (``--request-size 1``) or per request (``n>1``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

METRIC_COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
)
METRIC_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)


@dataclass(frozen=True)
class RenderSpec:
    prompt: list[int]
    stop_ids: list[int]
    think_marker: str
    think_close: str


class LiveCounters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.http_active = 0
        self.http_started = 0
        self.http_completed = 0
        self.groups_completed = 0
        self.tokens_completed = 0

    def http_enter(self) -> None:
        with self.lock:
            self.http_active += 1
            self.http_started += 1

    def http_leave(self) -> None:
        with self.lock:
            self.http_active -= 1
            self.http_completed += 1

    def group_done(self, tokens: int) -> None:
        with self.lock:
            self.groups_completed += 1
            self.tokens_completed += tokens

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "http_active": self.http_active,
                "http_started": self.http_started,
                "http_completed": self.http_completed,
                "groups_completed": self.groups_completed,
                "tokens_completed": self.tokens_completed,
            }


# ----------------------------------------------------------------------------
# prompt rendering (byte-identical to the RL client's renderers)
# ----------------------------------------------------------------------------
def single_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {ids}")
    return ids[0]


def render_prompt(tokenizer, renderer: str, question: str, current_date: str) -> RenderSpec:
    if renderer == "qwen3":
        text = "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
        return RenderSpec(tokenizer.encode(text, add_special_tokens=False),
                          [single_token(tokenizer, "<|im_end|>")], "</think>", "\n</think>\n\n")
    if renderer == "gemma4":
        text = ("<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n" + question
                + "<turn|>\n<|turn>model\n")
        return RenderSpec(tokenizer.encode(text, add_special_tokens=False),
                          [single_token(tokenizer, "<turn|>")], "<channel|>", "<channel|>")
    if renderer == "muse_glimmer_high_reasoning":
        text = ("<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n"
                "Knowledge cutoff: 2026-01-04.\n"
                f"Current date: {current_date}.\n\nReasoning strength: high.\n\n"
                '# Valid recipients: "self", "user".<|eot|>'
                "<|start|>user<|message|>" + question + "<|eot|><|start|>assistant")
        return RenderSpec(tokenizer.encode(text, add_special_tokens=False),
                          [single_token(tokenizer, "<|eot|>"), single_token(tokenizer, "<|end_of_text|>")],
                          "to=user", "<|eom|><|start|>assistant to=user<|message|>")
    raise ValueError(f"unsupported renderer: {renderer}")


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def post_json(url: str, payload: dict, timeout: int, counters: LiveCounters | None = None) -> tuple[dict, float]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    if counters:
        counters.http_enter()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode()), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:4000]
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from None
    finally:
        if counters:
            counters.http_leave()


def get_text(url: str, timeout: int = 20) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode(errors="replace")


def scrape_metrics(base: str) -> dict:
    """Parse the engine's Prometheus text for the counters/gauges we care about
    (summed over label sets). Best effort: an engine without /metrics yields {}."""
    try:
        text = get_text(base.rstrip("/") + "/metrics")
    except Exception as error:  # noqa: BLE001 - best effort by design
        return {"_error": str(error)[:200]}
    out: dict[str, float] = {}
    wanted = set(METRIC_COUNTERS) | set(METRIC_GAUGES)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if not match or match.group(1) not in wanted:
            continue
        try:
            out[match.group(1)] = out.get(match.group(1), 0.0) + float(match.group(3))
        except ValueError:
            continue
    return out


class GaugeSampler(threading.Thread):
    """Samples engine gauges every `interval` seconds while the run is live."""

    def __init__(self, bases: list[str], interval: float) -> None:
        super().__init__(daemon=True)
        self.bases = bases
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            stamp = time.time()
            for base in self.bases:
                metrics = scrape_metrics(base)
                self.samples.append({"t": stamp, "base": base,
                                     **{k: metrics.get(k) for k in METRIC_GAUGES if k in metrics}})
            self.stop_event.wait(self.interval)

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for base in self.bases:
            rows = [s for s in self.samples if s["base"] == base]
            entry: dict[str, dict] = {}
            for gauge in METRIC_GAUGES:
                values = [r[gauge] for r in rows if r.get(gauge) is not None]
                if values:
                    entry[gauge] = {"max": max(values), "mean": statistics.mean(values), "samples": len(values)}
            out[base] = entry
        return out


def metric_deltas(before: dict[str, dict], after: dict[str, dict]) -> dict:
    out: dict[str, dict] = {}
    for base, post in after.items():
        pre = before.get(base, {})
        entry = {}
        for counter in METRIC_COUNTERS:
            if counter in post and counter in pre:
                entry[counter] = post[counter] - pre[counter]
        if entry.get("vllm:prefix_cache_queries_total"):
            entry["prefix_cache_hit_rate"] = (entry.get("vllm:prefix_cache_hits_total", 0.0)
                                              / entry["vllm:prefix_cache_queries_total"])
        out[base] = entry
    return out


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        return ordered[index]

    return {"n": len(ordered), "p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99),
            "max": ordered[-1], "mean": statistics.mean(ordered)}


# ----------------------------------------------------------------------------
# realistic mode
# ----------------------------------------------------------------------------
def choices_from_response(response: dict, tokenizer, expected: int) -> list[dict]:
    choices = sorted(response.get("choices") or [], key=lambda item: item.get("index", 0))
    if len(choices) != expected:
        raise RuntimeError(f"expected {expected} choices, got {len(choices)}")
    output = []
    for choice in choices:
        ids = choice.get("token_ids") or []
        if not ids and choice.get("text"):
            ids = tokenizer.encode(choice["text"], add_special_tokens=False)
        if not ids:
            raise RuntimeError("vLLM response did not include token_ids or text")
        output.append({"ids": list(ids), "text": choice.get("text") or tokenizer.decode(ids),
                       "finish_reason": choice.get("finish_reason") or ""})
    return output


def completion_payload(model: str, prompt: list[int], n: int, max_tokens: int, temperature: float,
                       stop_ids: list[int], *, ignore_eos: bool = False, logprobs: int | None = 1) -> dict:
    payload = {"model": model, "prompt": prompt, "n": n, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": 1.0, "top_k": -1, "stream": False,
               "return_token_ids": True}
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if stop_ids:
        payload["stop_token_ids"] = stop_ids
    if ignore_eos:
        payload["ignore_eos"] = True
    return payload


def request_sizes(group_size: int, request_size: int) -> list[int]:
    sizes, remaining = [], group_size
    while remaining:
        size = min(request_size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def run_group(item: dict, model: str, endpoint: str, tokenizer, phase1_max_tokens: int, context_window: int,
              context_buffer: int, temperature: float, group_size: int, request_size: int, timeout: int,
              counters: LiveCounters, phase1_pool, phase2_pool) -> dict:
    started = time.perf_counter()
    render = item["render"]
    prompt = render.prompt
    phase1_budget = phase1_max_tokens - len(prompt)
    if phase1_budget <= 0:
        raise RuntimeError(f"{item['problem']} state {item['state_id']} prompt length {len(prompt)} "
                           f"exceeds phase-1 cap {phase1_max_tokens}")
    sizes = request_sizes(group_size, request_size)
    phase1_jobs = [(size, phase1_pool.submit(post_json, endpoint,
                                             completion_payload(model, prompt, size, phase1_budget, temperature,
                                                                render.stop_ids), timeout, counters))
                   for size in sizes]
    choices, phase1_request_walls = [], []
    for size, future in phase1_jobs:
        response, wall = future.result()
        phase1_request_walls.append({"n": size, "wall_seconds": wall})
        choices.extend(choices_from_response(response, tokenizer, size))
    phase1_done = time.perf_counter()

    phase2_tokens = [0] * group_size
    phase2_reasons = [""] * group_size
    phase2_used = [False] * group_size
    phase2_walls = [0.0] * group_size
    language = item["code_language"]
    answer_cue = f"Here is the final complete program:\n\n```{language}\n"

    def continue_choice(index: int, generated: dict, cue_ids: list[int], answer_max: int):
        followup, wall = post_json(endpoint, completion_payload(model, prompt + generated["ids"] + cue_ids, 1,
                                                                answer_max, temperature, render.stop_ids),
                                   timeout, counters)
        result = choices_from_response(followup, tokenizer, 1)[0]
        return index, len(result["ids"]), result["finish_reason"], wall

    phase2_jobs = []
    for index, generated in enumerate(choices):
        if len(generated["ids"]) < phase1_budget:
            continue
        cue = "" if render.think_marker in generated["text"] else render.think_close + answer_cue
        cue_ids = tokenizer.encode(cue, add_special_tokens=False)
        answer_max = context_window - len(prompt) - len(generated["ids"]) - len(cue_ids) - context_buffer
        if answer_max > 0:
            phase2_used[index] = True
            phase2_jobs.append(phase2_pool.submit(continue_choice, index, generated, cue_ids, answer_max))
    for future in concurrent.futures.as_completed(phase2_jobs):
        index, token_count, finish_reason, wall = future.result()
        phase2_tokens[index] = token_count
        phase2_reasons[index] = finish_reason
        phase2_walls[index] = wall

    finished = time.perf_counter()
    phase1_lengths = [len(c["ids"]) for c in choices]
    total_lengths = [phase1_lengths[i] + phase2_tokens[i] for i in range(group_size)]
    # Per-sequence wall: the phase-1 request that carried it (exact when
    # request_size == 1, the shared request wall otherwise) plus its phase-2 call.
    seq_walls, cursor = [], 0
    for req in phase1_request_walls:
        for _ in range(req["n"]):
            seq_walls.append(req["wall_seconds"] + phase2_walls[cursor])
            cursor += 1
    total_tokens = sum(total_lengths)
    counters.group_done(total_tokens)
    return {
        "problem": item["problem"], "state_id": item["state_id"], "timestep": item["timestep"],
        "value": item["value"], "base": endpoint.rsplit("/v1/", 1)[0], "prompt_tokens": len(prompt),
        "phase1_budget": phase1_budget, "group_size": group_size, "request_size": request_size,
        "phase1_requests": len(sizes), "phase1_wall_seconds": phase1_done - started,
        "wall_seconds": finished - started, "phase1_tokens": sum(phase1_lengths),
        "phase2_tokens": sum(phase2_tokens), "total_completion_tokens": total_tokens,
        "completion_lengths": total_lengths, "sequence_wall_seconds": seq_walls,
        "phase1_request_walls": phase1_request_walls, "phase2_wall_seconds": phase2_walls,
        "phase1_finish_reasons": [c["finish_reason"] for c in choices],
        "phase2_finish_reasons": phase2_reasons, "phase2_sequences": sum(phase2_used),
    }


def summarize_problem(problem: str, groups: list[dict], elapsed: float) -> dict:
    selected = [g for g in groups if g["problem"] == problem]
    lengths = [n for g in selected for n in g["completion_lengths"]]
    walls = [w for g in selected for w in g["sequence_wall_seconds"]]
    tokens = sum(lengths)
    return {
        "groups": len(selected), "sequences": len(lengths),
        "unique_prompt_tokens": sum(g["prompt_tokens"] for g in selected),
        "completion_tokens": tokens, "completion_tokens_per_second_over_cycle": tokens / elapsed,
        "mean_completion_tokens": statistics.mean(lengths), "median_completion_tokens": statistics.median(lengths),
        "min_completion_tokens": min(lengths), "max_completion_tokens": max(lengths),
        "phase2_sequences": sum(g["phase2_sequences"] for g in selected),
        "group_wall_seconds": percentiles([g["wall_seconds"] for g in selected]),
        "sequence_wall_seconds": percentiles(walls),
    }


def run_realistic(args, tokenizer, bases: list[str], engine_info: dict) -> dict:
    source = json.loads(Path(args.states).read_text())
    items = []
    for problem in ("erdos", "jssp", "ac1"):
        spec = source["problems"][problem]
        for state in spec["states"]:
            item = {"problem": problem, "code_language": spec["code_language"],
                    **{k: state[k] for k in ("state_id", "timestep", "value")}}
            item["render"] = render_prompt(tokenizer, args.renderer, state["question"], args.current_date)
            items.append(item)
    prompt_lengths = [len(i["render"].prompt) for i in items]
    print(f"WORKLOAD model={args.model} renderer={args.renderer} groups={len(items)} group_size={args.group_size} "
          f"sequences={len(items) * args.group_size} request_size={args.request_size} engines={len(bases)} "
          f"prompt_tokens=min:{min(prompt_lengths)} median:{statistics.median(prompt_lengths)} "
          f"max:{max(prompt_lengths)} phase1_cap={args.phase1_max_tokens} context={args.context_window}", flush=True)
    if max(prompt_lengths) >= args.phase1_max_tokens:
        raise RuntimeError("at least one rendered prompt exceeds the phase-1 cap")

    endpoints = [b.rstrip("/") + "/v1/completions" for b in bases]
    counters = LiveCounters()
    warmup = None
    if not args.skip_warmup:
        dummy_id = tokenizer.encode(" token", add_special_tokens=False)[0]

        def warm(index: int, item: dict, size: int) -> dict:
            length = len(item["render"].prompt)
            response, wall = post_json(endpoints[index % len(endpoints)], completion_payload(
                args.model, [dummy_id] * length, size, 8, args.temperature, [], ignore_eos=True),
                args.http_timeout, counters)
            choices = choices_from_response(response, tokenizer, size)
            return {"prompt_tokens": length, "completion_tokens": sum(len(c["ids"]) for c in choices),
                    "wall_seconds": wall}

        print("WARMUP shape-matched prompts (compiles the prefill/decode buckets on every engine)", flush=True)
        started = time.perf_counter()
        specs = [(gi, item, size) for gi, item in enumerate(items) for size in request_sizes(args.group_size, args.request_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(256, len(specs))) as pool:
            rows = list(pool.map(lambda s: warm(*s), specs))
        warmup = {"wall_seconds": time.perf_counter() - started, "groups": len(items), "requests": len(rows),
                  "completion_tokens": sum(r["completion_tokens"] for r in rows),
                  "max_request_wall_seconds": max(r["wall_seconds"] for r in rows)}
        print(f"WARMUP complete {json.dumps(warmup, sort_keys=True)}", flush=True)

    metrics_before = {b: scrape_metrics(b) for b in bases}
    sampler = GaugeSampler(bases, args.gauge_interval)
    sampler.start()
    per_request = len(request_sizes(args.group_size, args.request_size))
    print(f"MEASURE {len(items)} concurrent groups over {len(items) * per_request} phase-1 requests, "
          f"round-robin over {len(bases)} engines", flush=True)
    cycle_started = time.perf_counter()
    phase2_workers = min(512, len(items) * args.group_size)
    phase1_workers = min(512, len(items) * per_request)
    with concurrent.futures.ThreadPoolExecutor(max_workers=phase1_workers) as phase1_pool, \
            concurrent.futures.ThreadPoolExecutor(max_workers=phase2_workers) as phase2_pool, \
            concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as group_pool:
        futures = [group_pool.submit(run_group, item, args.model, endpoints[gi % len(endpoints)], tokenizer,
                                     args.phase1_max_tokens, args.context_window, args.context_buffer,
                                     args.temperature, args.group_size, args.request_size, args.http_timeout,
                                     counters, phase1_pool, phase2_pool)
                   for gi, item in enumerate(items)]
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=args.heartbeat_seconds,
                                                    return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            print(f"HEARTBEAT elapsed={time.perf_counter() - cycle_started:.0f}s pending_groups={len(pending)} "
                  f"{json.dumps(counters.snapshot(), sort_keys=True)}", flush=True)
        groups = [f.result() for f in futures]
    cycle_seconds = time.perf_counter() - cycle_started
    sampler.stop_event.set()
    sampler.join(timeout=60)
    metrics_after = {b: scrape_metrics(b) for b in bases}

    total_tokens = sum(g["total_completion_tokens"] for g in groups)
    lengths = [n for g in groups for n in g["completion_lengths"]]
    seq_walls = [w for g in groups for w in g["sequence_wall_seconds"]]
    result = {
        "schema_version": 2, "mode": "realistic", "model": args.model, "renderer": args.renderer,
        "bases": bases, "engines": len(bases), "engines_per_host": args.engines_per_host,
        "tp_size": args.tp_size, "max_num_seqs": args.max_num_seqs, "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.context_window, "phase1_max_tokens": args.phase1_max_tokens,
        "context_buffer": args.context_buffer, "temperature": args.temperature, "prefix_caching": True,
        "logprobs": 1, "group_size": args.group_size, "request_size": args.request_size,
        "groups": len(groups), "sequences": len(lengths), "warmup": warmup, "cycle_seconds": cycle_seconds,
        "unique_prompt_tokens": sum(g["prompt_tokens"] for g in groups), "completion_tokens": total_tokens,
        "completion_tokens_per_second": total_tokens / cycle_seconds,
        "completion_tokens_per_second_per_engine": total_tokens / cycle_seconds / len(bases),
        "mean_completion_tokens": statistics.mean(lengths), "median_completion_tokens": statistics.median(lengths),
        "min_completion_tokens": min(lengths), "max_completion_tokens": max(lengths),
        "phase2_sequences": sum(g["phase2_sequences"] for g in groups),
        "latency": {"sequence_wall_seconds": percentiles(seq_walls),
                    "group_wall_seconds": percentiles([g["wall_seconds"] for g in groups]),
                    "phase1_group_wall_seconds": percentiles([g["phase1_wall_seconds"] for g in groups]),
                    "note": "per-sequence wall = its phase-1 request wall (+ phase-2 call); exact only with --request-size 1"},
        "per_problem": {p: summarize_problem(p, groups, cycle_seconds) for p in ("erdos", "jssp", "ac1")},
        "prompt_lengths": {"min": min(prompt_lengths), "median": statistics.median(prompt_lengths),
                           "max": max(prompt_lengths),
                           "per_group": [{"problem": i["problem"], "state_id": i["state_id"],
                                          "tokens": len(i["render"].prompt)} for i in items]},
        "engine_info": engine_info, "engine_metrics_before": metrics_before, "engine_metrics_after": metrics_after,
        "engine_metric_deltas": metric_deltas(metrics_before, metrics_after), "engine_gauges": sampler.summary(),
        "group_results": groups, "source_manifest": source,
    }
    return result


# ----------------------------------------------------------------------------
# capacity mode
# ----------------------------------------------------------------------------
def fire(bases: list[str], model: str, prompts: list[list[int]], max_tokens: int, timeout: int) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        futures = [pool.submit(post_json, bases[i % len(bases)].rstrip("/") + "/v1/completions",
                               completion_payload(model, p, 1, max_tokens, 0.0, [], ignore_eos=True, logprobs=None),
                               timeout) for i, p in enumerate(prompts)]
        outs = [f.result() for f in futures]
    wall = time.perf_counter() - started
    completion = sum(o.get("usage", {}).get("completion_tokens", 0) for o, _ in outs)
    per_base = [0] * len(bases)
    for i, (o, _) in enumerate(outs):
        per_base[i % len(bases)] += o.get("usage", {}).get("completion_tokens", 0)
    return {"wall": wall, "completion_tokens": completion, "per_base": per_base,
            "request_wall_seconds": percentiles([w for _, w in outs])}


def run_capacity(args, tokenizer, bases: list[str], engine_info: dict) -> dict:
    vocab = getattr(tokenizer, "vocab_size", 32000) or 32000
    rng = random.Random(args.seed)
    lo, hi = 1000, max(2000, vocab - 1000)
    max_c = max(int(x) for x in args.concurrencies.split(",") if x.strip())
    # Unique prompts (random ids, deterministic) so prefix caching cannot hide
    # prefill cost or share KV; length = the production request shape.
    prompts = [[rng.randrange(lo, hi) for _ in range(args.capacity_prompt_tokens)] for _ in range(max_c)]
    result = {"schema_version": 2, "mode": "capacity", "model": args.model, "bases": bases, "engines": len(bases),
              "engines_per_host": args.engines_per_host, "tp_size": args.tp_size, "max_num_seqs": args.max_num_seqs,
              "gpu_memory_utilization": args.gpu_memory_utilization, "prompt_tokens": args.capacity_prompt_tokens,
              "gen_short": args.gen_short, "gen_long": args.gen_long, "warmed_before_timing": True,
              "engine_info": engine_info, "buckets": []}
    for base in bases:
        _, wall = post_json(base.rstrip("/") + "/v1/completions",
                            completion_payload(args.model, prompts[0][:16], 1, 2, 0.0, [], ignore_eos=True, logprobs=None),
                            args.http_timeout)
        print(f"  liveness {base}: {wall:.1f}s", flush=True)
    for c in [int(x) for x in args.concurrencies.split(",") if x.strip()]:
        if args.max_concurrency and c > args.max_concurrency:
            print(f"  skip c={c} > engine max concurrency {args.max_concurrency}", flush=True)
            continue
        batch = prompts[:c]
        t0 = time.perf_counter()
        fire(bases, args.model, batch, args.warm_tokens, args.http_timeout)  # warm, untimed
        warm_wall = time.perf_counter() - t0
        metrics_before = {b: scrape_metrics(b) for b in bases}
        short = fire(bases, args.model, batch, args.gen_short, args.http_timeout)
        long_ = fire(bases, args.model, batch, args.gen_long, args.http_timeout)
        metrics_after = {b: scrape_metrics(b) for b in bases}
        dtok = long_["completion_tokens"] - short["completion_tokens"]
        dwall = long_["wall"] - short["wall"]
        steady = dtok / dwall if dwall > 0 else float("nan")
        raw = long_["completion_tokens"] / long_["wall"]
        row = {"concurrency": c, "per_engine_concurrency": c / len(bases), "warm_wall_untimed": warm_wall,
               "short": short, "long": long_, "steady_decode_tok_s": steady, "raw_long_tok_s": raw,
               "steady_decode_tok_s_per_engine": steady / len(bases),
               "engine_metric_deltas": metric_deltas(metrics_before, metrics_after)}
        result["buckets"].append(row)
        print(f"  CAPACITY c={c:4d} steady={steady:8.1f} tok/s ({steady / len(bases):7.1f}/engine) raw={raw:8.1f} "
              f"walls short/long {short['wall']:.1f}/{long_['wall']:.1f}s per-base {long_['per_base']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["realistic", "capacity"], default="realistic")
    parser.add_argument("--states", help="states.json from build_states.py (realistic mode)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bases", default="http://127.0.0.1:8001",
                        help="comma-separated engine base URLs; groups/requests are round-robined over them")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True, help="HF id or path for the tokenizer (local files only)")
    parser.add_argument("--renderer", choices=["qwen3", "gemma4", "muse_glimmer_high_reasoning"])
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--engines-per-host", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--engine-info-json", help="boot-log facts collected by the runner (KV cache size, max concurrency)")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--request-size", type=int, default=32)
    parser.add_argument("--phase1-max-tokens", type=int, default=13824)
    parser.add_argument("--context-window", type=int, default=22528)
    parser.add_argument("--context-buffer", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--current-date", default="2026-09-05")
    parser.add_argument("--http-timeout", type=int, default=7200)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--gauge-interval", type=float, default=30.0)
    parser.add_argument("--skip-warmup", action="store_true")
    # capacity mode
    parser.add_argument("--concurrencies", default="16,32,64,96,128,192,256,384")
    parser.add_argument("--max-concurrency", type=int, default=0, help="cap the ladder (e.g. engines x boot-log max)")
    parser.add_argument("--capacity-prompt-tokens", type=int, default=20480)
    parser.add_argument("--gen-short", type=int, default=128)
    parser.add_argument("--gen-long", type=int, default=384)
    parser.add_argument("--warm-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.mode == "realistic":
        if not args.states or not args.renderer:
            parser.error("realistic mode needs --states and --renderer")
        if not 1 <= args.request_size <= args.group_size:
            parser.error("--request-size must be between 1 and --group-size")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, local_files_only=True)
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    engine_info = json.loads(Path(args.engine_info_json).read_text()) if args.engine_info_json else {}
    result = (run_realistic if args.mode == "realistic" else run_capacity)(args, tokenizer, bases, engine_info)
    Path(args.out).write_text(json.dumps(result, indent=2))
    if args.mode == "realistic":
        keys = ("model", "engines", "engines_per_host", "tp_size", "max_num_seqs", "gpu_memory_utilization",
                "sequences", "cycle_seconds", "completion_tokens", "completion_tokens_per_second",
                "completion_tokens_per_second_per_engine", "mean_completion_tokens", "phase2_sequences")
        headline = {k: result[k] for k in keys}
        headline["sequence_wall_p50_p95_max"] = [result["latency"]["sequence_wall_seconds"].get(k)
                                                  for k in ("p50", "p95", "max")]
    else:
        headline = {"model": result["model"], "engines": result["engines"],
                    "best_steady_tok_s": max((b["steady_decode_tok_s"] for b in result["buckets"]), default=None),
                    "buckets": [(b["concurrency"], round(b["steady_decode_tok_s"], 1)) for b in result["buckets"]]}
    print(f"BENCHMARK_RESULT {json.dumps(headline, sort_keys=True, default=str)}", flush=True)


if __name__ == "__main__":
    main()
