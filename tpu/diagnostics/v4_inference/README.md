# v4 Inference Failure Investigation

Status: IN PROGRESS, started 2026-09-07. No general root cause established.

This investigation separates fatal TPU runtime failures from setup problems,
host/HBM allocation failures, client timeouts, and numerical audit failures.
An engine restart is a recovery event, not proof of the initiating cause.
Healthy/READY provider state is not proof of sustained inference health.

## Evidence Collected

Raw snapshots and the initial inventory are in
`tpu/results/v4-inference-diagnosis-20260907/`.

- Twelve hosts inspected: four serving hosts on the current v4-64 slice, four
  v4-32 inference controls, and the v5p-32 trainer plus three serving hosts.
- Initial inventory: **22 engine instances with PID-matched TPU-fatal logs**
  across Qwen v4-64 runs `001`, `002`, and `budget-003`. Counts by physical
  serving host: w1=5, w2=6, w5=5, w6=6. These are all the SAME allocated
  v4-64 slice, not 22 independent machines or evidence of fleet-wide failure.
- The inspected v4-32 and v5p Ray engine logs have no fatal TPU matches.
  This is bounded evidence from retained logs, not an exhaustive fleet history.
- All inspected serving environments report JAX/JAXLIB 0.10.1, libtpu 0.0.41,
  vllm-tpu 0.23.0, torchax 0.0.11, torch 2.10.0, transformers 5.8.0,
  and Ray 2.58.0. Installed tpu-inference Python trees on one representative
  host per group have identical SHA256
  `c3b5e1b31887b1a05fff63cb3a10aa5b2594cbb40c7c33f1315da06a155e4ecf`.
  Package versions are snapshot-time values, not proof of historical pins.
- Current v4-64 and v4-32 serving flags match: TP4, s16, u0.80, context22528,
  chunk4096, batched RPA and JAX ragged conv enabled, subprocess method spawn.
  Current v5p differs in s32/u0.90. Runtime namespaces differ as intended.
- Kernel `VM_DONTCOPY` mapping warnings also occur on all four healthy v4-32
  control hosts. The warning alone does not prove a fork occurred or caused a
  crash. Do not attribute these failures to grader Ray from this warning.
- GCP describes the affected v4-64 node as READY/HEALTHY at collection time.
  Its node is `tpuswarm-v4-64-centra-ea9i-7bfcb694-head-13ls8qz5-tpu`.

For job382, physical w5's EngineCore PID167335 logged its fatal TPU error at
15:51:15 UTC and vLLM noticed its exit at 15:51:48. Physical w6's PID173203
logged a fatal interrupt at 15:51:58 and vLLM noticed its exit at 15:52:47.
Both runtimes reported a chip error, failed their internal SliceBuilder health
check, collected dumps, and explicitly terminated the process. SliceBuilder
is libtpu's controller, not SkyPilot or our Ray head. The driver labels the
underlying error Unknown. This establishes the failure chain but does not
distinguish a hardware defect from a compiler/runtime/kernel-triggered fault.

Two historical instances also contain `RuntimeProgramAllocationFailure` after
TPU-fatal evidence. The inventory records both facts; do not automatically
relabel those as ordinary HBM exhaustion or assume the allocation error came first.

## Successful Controls

- `tpu/results/qwen-v4-32-ray-inference-job371/RESULTS.md`: two complete
  64-request, 512-output-token rounds without engine restarts. This does not
  validate long decoding or LoRA.
- `tpu/results/v4-32-inference-sampler-ablation/RESULTS.md`: prior completed
  multi-hour Qwen, Muse, and Gemma replays, including Qwen's 2,917,573 output
  tokens over 7,660 seconds. The Qwen raw GCS JSON was retrieved and saved as
  `qwen-success-control.json` for comparison.
- The successful Qwen object's name includes `chunk1024-default-rpa-sync-v11`.
  This is a lead for recovering the old launch recipe, NOT proof of the exact
  deployed kernel/scheduler flags. Current Ray logs explicitly enable async
  scheduling and use chunk4096/experimental batched RPA.
- Old Gemma attempts197/198 were setup/unsupported-kernel failures, whereas
  corrected199/207 completed. Keep these separate from TPU-fatal runtime exits.

## Controlled Matrix

Hold model snapshot, installed code hash, packages, request corpus, TP4, s16,
u0.80, max context22528 and sampling settings fixed unless the arm explicitly
changes one. Reuse complete model caches. Use separate compilation namespaces
for changed software/kernel shapes; never clear active training caches.

| Arm | Change from its paired control | Question | Status |
| --- | --- | --- | --- |
| A0 | Existing job371, 512 generated tokens, no adapter | Short-decode baseline | Prior successful evidence |
| A1 | Same engines/payloads, 8192 generated tokens | Is long decode alone sufficient? | Passed: 64/64, no restarts |
| A2 | A1 plus one fixed production-format rank32 adapter | Is LoRA necessary to reproduce? | Pending A1 |
| A3 | A2 plus adapter replacement between drained rounds | Does updating add a failure mode? | Pending A2 |
| B1 | Repeat concurrency control on an independent v4-64, inference only | Physical slice/topology versus workload? | Job396: short 1/8-request controls passed; not full failing workload |
| B2 | Same B1 slice/config plus idle initialized trainer, then actual training | Does colocated trainer runtime matter? | Not launched |
| C1 | Failing arm, async scheduling disabled only | Scheduler interaction? | Not launched |
| C2 | Failing arm, default RPA only, after verifying v4 support | Experimental attention kernel? | Not launched |
| C3 | Failing arm, JAX ragged conv disabled only | Qwen hybrid/GDN kernel? | Not launched |
| C4 | Failing arm, chunk1024 only | Prefill/batch-shape transition? | Not launched |
| D1 | Same workload on v5p, normalized s16/u0.80 | Accelerator-specific behavior? | Not launched |
| E1 | Failing arm, coherent alternative JAX/JAXLIB/libtpu stack | Runtime-version regression? | Not launched |

If A1 passes, it does NOT exonerate long decoding with LoRA or real 3-6K token
prompts. After isolating a minimal failure, replay identical realistic prompts
with logprobs and two phases. Require repeats on two independently allocated
v4 slices before attributing the problem to software rather than this slice.
Then validate Muse and Gemma separately; do not assume Qwen's hybrid-attention
result applies to them. Finally validate generation plus grading, optimizer
steps, durable checkpoints, and in-place recovery as separate acceptance gates.

## Current Canary

This is an HTTP client using the already allocated, verified-idle **job371**,
worker94, head35.186.64.248. It is not a new managed job or a new TPU request.
Jobs382 and384 are untouched. No engine restart, dependency change, model-cache
deletion, or LoRA load was performed to start this arm.

- 64 concurrent requests, one round, max8192 output tokens each, EOS ignored.
- Same short proof prompts and temperature0.7 as the recorded short baseline.
- No per-request seed, logprobs, adapter, grading or trainer.
- Expected total: 524,288 output tokens. HTTP timeout: 3600 seconds.
- Remote artifacts: `~/.cache/skyrl-ray/diagnostics/v4-long-base-20260907-01/`.
- Launch receipt: `long-base-launch.json`; client PID409549 at launch.
- `progress.jsonl` saves counters and catalog state every30 seconds;
  `responses.jsonl` saves individual results; `summary.json` marks completion.
- Pass requires every requested token returned, zero HTTP errors, and unchanged
  engine start counts. Inspect driver logs afterward even on a pass. Timeout
  alone is not classified as a TPU crash. No automatic second arm or retry.

The existing five-minute workload monitor still watches jobs371/382/384.
The long-decode control finished: 64/64 successful, 524,288 output tokens in
1885.82 seconds (278.02 tokens/s), unchanged engine start counts. Saved summary:
`tpu/results/v4-inference-diagnosis-20260907/v4-32-long-base-summary.json`.

## Fresh v4-64 Concurrency Control

Job396 is a new managed inference-only job on idle worker60, not a direct-SSH
engine launch. All eight hosts were checked for stray workload processes before
submission. Only logical rank0 hosts a TP4 engine; the other seven hosts join the
private CPU Ray runtime. No trainer or adapter is loaded. Existing jobs382/384/389
were not cancelled, modified, or restarted.

Results, startup caveats and immutable bundle identities are in
`tpu/results/v4-inference-diagnosis-20260907/v4-64-concurrency-job396/RESULTS.md`.
The fresh-worker baseline passed both 1 and 8 concurrent copies of job389's
recorded base request. A further 8-request round with distinct first token IDs
also passed. Engine-side metrics observed eight sequences running together.
This rules out a universal eight-request crash, not the original longer RL case.

`concurrency_probe.py` saves the payload, raw responses, per-second engine metrics,
catalog state and elapsed time. It refuses non-idle, adapter-bearing or unexpected
endpoints. The distinct-prefix arm changes only the first token ID and preserves
prompt length. These are synthetic token-ID prompts, not completed RL rollouts.

## Tools

`collect.py` uses only stdlib metadata reads, local logs, `/proc` and read-only
kernel journal queries. It never imports JAX, initializes a TPU, signals a
process, reads credential files, or dumps arbitrary process environments.
Deploy/run the same script over authenticated SSH and save stdout as JSON.
Raw logs are bounded excerpts, with exact paths/PIDs and omission markers.

```bash
python3 tpu/diagnostics/v4_inference/collect.py --run qwen-ray-v4-64-budget-003
python3 -m tpu.diagnostics.v4_inference.summarize \
  tpu/results/v4-inference-diagnosis-20260907/v4-64-w[1256].json
```

The first command runs ON a TPU host, not this repository's controller node.
The second consumes saved files locally. Do not include later `*-provenance`
snapshots in instance counts or they duplicate the initial inventory.

`long_decode.py` refuses busy endpoints, any committed adapter, unexpected host
inventories, and non-idle/unknown engine counters. It only submits requests;
it does not own or kill the serving processes. Before another arm, independently
verify the managed job's ownership and drain ALL engine queues, not just ingress.

CPU regression tests:

```bash
srun -p cpu --cpus-per-task=2 --mem=4G --time=00:10:00 \
  /scratch/gpfs/ZHUANGL/sk7524/.cache/skyrl-ray-tests/venv/bin/python \
  -m pytest tests/tpu_swarm/test_v4_inference_diagnostics.py -q
```
