# v5p-32 inference benchmarks

**Status: INCOMPLETE / experimental.** This commit preserves the benchmark
harness and candidate configurations, not a completed or validated comparison
across all models and settings. Do not treat the checked-in concurrency limits
as measured production recommendations.

The runner uses one v5p-32 slice: host 0 drives the benchmark and hosts 1-3
serve independent vLLM engines. No trainer is started. The suite includes
Qwen3.5, Gemma 4, and Muse configurations, capacity probes, realistic grouped
generation, and one- or two-engine-per-host variants.

- `build_states.py`: prepare replay inputs.
- `run_v5p32_bench.sh`: prepare hosts, launch engines, run the benchmark, and
  publish artifacts.
- `realistic_bench.py`: capacity and realistic generation measurements.
- `../examples/v5p32-bench/`: candidate YAMLs and their generator.
- `../../../tests/tpu_swarm/test_v5p32_bench.py`: local regression checks.

Before declaring the study complete, reconcile each submitted job with its
saved result, record failures and missing configurations, verify warmup and
measurement boundaries, and publish a comparison backed by raw artifacts.
Local tests check harness contracts; they do not establish TPU throughput,
capacity, or successful completion of the full benchmark matrix.
