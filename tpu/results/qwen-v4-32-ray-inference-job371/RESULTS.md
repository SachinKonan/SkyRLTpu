# Qwen v4-32 Ray inference timing, job 371

Measured 2026-09-07 on worker 94 of `tpuswarm-v4-32-central2-smoke`.
Four independent Qwen/Qwen3.5-27B TP4 engines, one per host; shared Ray Serve
endpoint `http://10.130.0.102:19800`. No training or grading.

## Workload

- 64 concurrent HTTP completions through the shared endpoint per round.
- Each request: 25 prompt tokens, 512 requested output tokens, temperature 0.7,
  `ignore_eos=true`, no per-request seed, no loaded LoRA adapter.
- Short mathematical proof prompts, identical across rounds. Prefix caching can
  benefit repeated prompts. This is not the long-context Erdos workload.
- Engine configuration: TP4, max sequences 16, memory utilization 0.80,
  max model length 22528, prefill chunk 4096, LoRA support enabled at rank 32.
- Wall time includes gateway routing, engine queueing, compilation, prefill,
  decoding, and returning the complete JSON response. It excludes provisioning
  and initial engine startup. Requests are non-streaming; TTFT was not measured.

## Results

| Round | Successful requests | Wall seconds | Returned output tokens | Output tokens/s | Median request seconds | p95 request seconds |
|---|---:|---:|---:|---:|---:|---:|
| Initial first-use attempt | 55/64 | 600.18 | 28160 | 46.92 | 557.49 | 589.99 |
| Subsequent round 1 | 64/64 | 499.35 | 32768 | 65.62 | 248.69 | 499.28 |
| Subsequent round 2 | 64/64 | 279.40 | 32768 | 117.28 | 55.94 | 243.71 |

The initial benchmark's 600-second client timeout was too short for compilation:
nine requests timed out. Its latency percentiles include successful requests only;
it is not a valid full-batch throughput result. Before resubmitting, both the
gateway and all engine queues were checked idle. Later rounds used a 1800-second
client timeout and completed without request errors.

**No engine restarts occurred.** All four catalog instance IDs stayed unchanged.
Final gateway health was HTTP 200, active requests zero, and all host compile-cache
sync error fields were clear. Periodic cache publication had transient errors
during compilation that subsequently recovered.

## Interpretation

Initial engine health did not mean generation was compiled. TPU driver logs
showed multi-minute XLA compilation of `jit_step_fun_impl`, including additional
compilation in subsequent rounds. The final round's first 32 requests finished
in about 51-56 seconds, but the full batch took 279 seconds. Do not extrapolate
fast-engine latency into a four-engine throughput claim.

The measured final aggregate rate is **117.28 output tokens/s**, not an established
steady-state ceiling. Further shape warmup and routing analysis are needed to
separate compilation overhead from sustained decoding performance. These runs
do not validate long-context generation, grading, LoRA updates, or training.

## Artifacts

- `benchmark-20260907-short512/`: initial attempt, raw responses, timing summary,
  per-engine metrics and catalog snapshots.
- `benchmark-20260907-short512-warm/`: the two subsequent rounds with the same
  artifacts. Despite the directory name, these rounds were not fully warmed.
- Probe: `tpu/swarm/ray_train/benchmark_inference.py`.

Job 371 and its four inference engines were left running and idle. Other jobs,
including the v5p trainer, were not modified.
