#!/usr/bin/env python3
"""Emit the v5p-32 vLLM serving-sweep job YAMLs: one per model x setting.

Settings (all at --gpu-memory-utilization BENCH_GPU_UTIL, default 0.92):
  S1  1 engine/host, TP4, max_num_seqs = 128        (production qwen/gemma shape)
  S2  1 engine/host, TP4, max_num_seqs = MAX        (MAX from capacity-<model>)
  S3  2 engines/host, TP2, max_num_seqs = 64/engine (production muse shape)
  S4  2 engines/host, TP2, max_num_seqs = MAX/2 per engine
plus capacity-<model>: the decode-capacity ladder that determines MAX.

    python3 tpu/swarm/examples/v5p32-bench/gen_bench_yamls.py [--max-num-seqs qwen35=256,...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKET = "gs://sk7524-tinker-tpu-us-east5"
BUNDLE_URL = f"{BUCKET}/code-bundles/tpuswarm-skyrl-v5p32-bench-v2.tar.gz"
STATES_URL = f"{BUCKET}/v5p-bench/states-v1.json"
RESULT_PREFIX = f"{BUCKET}/v5p-bench-results"
MODELS = ("qwen35", "gemma4", "muse")
SETTINGS = {
    # name: (engines_per_host, max_num_seqs expression)
    "S1": (1, "128"),
    "S2": (1, "MAX"),
    "S3": (2, "64"),
    "S4": (2, "MAX/2"),
}

TEMPLATE = """name: {name}

# v5p-32 vLLM serving benchmark: {comment}
# Production cell shape on one pool worker: rank 0 = idle trainer/driver host,
# ranks 1-3 = vLLM engines (no trainer is started).  Setting: {setting_desc}.

resources:
  cloud: gcp
  region: us-east5
  zone: us-east5-a
  accelerators: tpu-v5p-32
  accelerator_args:
    gcp_queued_resource: true
    runtime_version: v2-alpha-tpuv5
  use_spot: true
  disk_size: 150
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 1

envs:
  PROJECT: vision-mix
  ZONE: us-east5-a
  REMOTE_USER: gcpuser
  SKYRL_REPO_DIR: /home/gcpuser/SkyRLTpu-tpuswarm
  TPUSWARM_SKYRL_BUNDLE_URL: {bundle_url}
  TPUSWARM_BUNDLE_ID: bench-{model}-{setting}

  BENCH_NAME: {name}
  BENCH_MODE: {mode}
  BENCH_MODEL: {model}
  BENCH_ENGINES_PER_HOST: "{engines}"
  BENCH_MAX_NUM_SEQS: "{mns}"
  BENCH_GPU_UTIL: "{util}"
  BENCH_REQUEST_SIZE: "{request_size}"
  BENCH_STATES_URL: {states_url}
  BENCH_RESULT_GCS_PREFIX: {result_prefix}

run: |
  set -euo pipefail
  export PATH="$HOME/.local/bin:$PATH"
  generation=$(gcloud storage objects describe "$TPUSWARM_SKYRL_BUNDLE_URL" \\
    --format='value(generation)')
  test -n "$generation"
  bundles="$HOME/.cache/tpuswarm/bundles"
  destination="$bundles/$generation"
  mkdir -p "$bundles"
  if [ ! -d "$destination" ]; then
    archive=$(mktemp /tmp/tpuswarm-skyrl.XXXXXX.tar.gz)
    staging=$(mktemp -d "$bundles/.extract.XXXXXX")
    trap 'rm -f -- "$archive"; rm -rf -- "$staging"' EXIT
    gcloud storage cp "$TPUSWARM_SKYRL_BUNDLE_URL" "$archive"
    tar -xzf "$archive" -C "$staging"
    test -r "$staging/tpu/swarm/bench/run_v5p32_bench.sh"
    test -r "$staging/tpu/swarm/bench/realistic_bench.py"
    test -r "$staging/tpu/start_vllm_tpu.sh"
    mv "$staging" "$destination"
    rm -f -- "$archive"
    trap - EXIT
  fi
  ln -sfn "$destination" "$SKYRL_REPO_DIR"
  exec bash "$SKYRL_REPO_DIR/tpu/swarm/bench/run_v5p32_bench.sh"
"""


def resolve_mns(expr: str, model_max: int) -> int:
    if expr == "MAX":
        return model_max
    if expr == "MAX/2":
        return max(1, model_max // 2)
    return int(expr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--bundle-url", default=BUNDLE_URL)
    parser.add_argument("--states-url", default=STATES_URL)
    parser.add_argument("--result-prefix", default=RESULT_PREFIX)
    parser.add_argument("--gpu-util", default="0.92")
    parser.add_argument("--request-size", default="32",
                        help="phase-1 n per HTTP call; 1 gives exact per-sequence latency at more HTTP overhead")
    parser.add_argument("--max-num-seqs", default="qwen35=256,gemma4=256,muse=256",
                        help="per-model MAX from the capacity probe, e.g. qwen35=320,gemma4=384,muse=192")
    args = parser.parse_args()

    model_max = {}
    for pair in args.max_num_seqs.split(","):
        model, value = pair.split("=")
        model_max[model.strip()] = int(value)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in MODELS:
        for setting, (engines, expr) in SETTINGS.items():
            mns = resolve_mns(expr, model_max.get(model, 256))
            tp = 4 // engines
            name = f"bench-{model}-{setting}"
            text = TEMPLATE.format(
                name=name, comment=f"{model} {setting}", model=model, setting=setting, mode="realistic",
                setting_desc=f"{engines}x TP{tp} per host, max_num_seqs={mns}/engine, gpu util {args.gpu_util}",
                bundle_url=args.bundle_url, engines=engines, mns=mns, util=args.gpu_util,
                request_size=args.request_size, states_url=args.states_url, result_prefix=args.result_prefix,
            )
            (out_dir / f"{name}.yaml").write_text(text)
            written.append(name)
        name = f"capacity-{model}"
        text = TEMPLATE.format(
            name=name, comment=f"{model} decode-capacity ladder (finds MAX for S2/S4)", model=model,
            setting="capacity", mode="capacity",
            setting_desc=f"1x TP4 per host at max_num_seqs={model_max.get(model, 256)}, synthetic 20k prompts, "
                         f"differenced steady-state decode over a concurrency ladder",
            bundle_url=args.bundle_url, engines=1, mns=model_max.get(model, 256), util=args.gpu_util,
            request_size=args.request_size, states_url=args.states_url, result_prefix=args.result_prefix,
        )
        (out_dir / f"{name}.yaml").write_text(text)
        written.append(name)
    for name in written:
        print(name)


if __name__ == "__main__":
    main()
