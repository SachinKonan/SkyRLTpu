# Qwen circuit training with borrowed inference — September 19

Round outcome: all **512 completions returned in 45m27s**, but the four borrowed
groups were cancelled by the strict heartbeat-readiness check and regenerated
locally. Borrowed full-length response delivery was not established. The
consumer now has a tested, bounded health grace period; job 1270's running bundle
has not been changed.

Earlier live snapshot: job 1270 acquired farm 173 and both farms generated tokens.
The eight-engine snapshot records **4,032.8 generation tokens/s**, including
**1,599.9 tokens/s** from the borrowed farm. This is **65.8% additional observed
throughput**, not a measured full-round speedup. As of September 20, 00:40 UTC,
no full borrowed group or optimizer step had been verified.
See the [architecture and measurements write-up](../../../../docs/borrowed-inference-qwen-2026-09-19.md),
[raw engine snapshot](fix1/throughput-snapshot.json), and
[live events](fix1/live-events.json).

Replacement trial submitted as **job 1270** to
`tpuswarm-v6e32-east5b-qwen35`, assigned to worker **5143**.
The original job **1262** was cancelled after a packaging failure before any
training or borrowed generation. Existing jobs 1259–1261 remain untouched.

Replacement profile: `science-circuit-v6e-qwen-borrow173-grpo-20260919-fix1.json`.
It derives from the existing v6e Qwen direct-training circuit profile, with a
fresh run/root, isolated compilation-cache write destinations, and external
inference borrowing enabled. Model caches and compatible v6e compilation caches
are read-only seeds. No prior optimizer, model adapter, or discovered programs
are imported into this run; recovery may resume this run's own checkpoints.

| Setting | Value |
| --- | --- |
| Algorithm | GRPO: mean baseline, existing importance-sampling loss |
| Trainer | 4 v6e hosts; TP8 / FSDP2; LoRA rank 32; token budget 45,056 |
| Local inference | 4 independent TP4 engines, 16 sequences each, memory fraction 0.8 |
| Borrowed farm | **Qwen job 1251, worker 173**, `http://10.130.0.171:24800` |
| External capacity | 4 TP4 engines; at most 4 concurrent requests, each at most `n=32` |
| Rollouts | 16 groups × 32; no group splitting |
| Context | 22,528 total; 16,384 prompt-plus-thinking allowance |
| Training | 15 epochs; learning rate 4e-5; adapter initialization seed 1 |
| Circuit grader | All 17 IBM cases, Xplace initialization and fast C proxy helper |
| CPU grading | 16 concurrent submissions/host, 4 CPUs and 8 GiB each |
| Bootstrap | None; direct training from fresh initialization |
| External preparation | 300s cap; 10s control RPC timeout; 30s heartbeat; 300s lease |
| Failure behavior | Unfinished borrowed requests retry on existing local inference |

Only this Qwen service URL is configured; Gemma/Muse farms are not candidates.
The v6e head successfully reached the Qwen farm's private `/health`, `/status`
and `/v1/models` before submission. Gcloud and ADC both matched the approved
compute service account; TPU inventory and storage read probes succeeded.
The local and remote engines use different TPU generations, and prefix caching
remains enabled locally but disabled on the serving farm. This is a borrowing
integration experiment, not an assertion of bitwise-equivalent inference.

## Live farm preflight

`probe_farm.py` ran on worker 173 using the existing trained Qwen step-12 fixture
recorded in `fixture.json`. This adapter was for the preflight only, not an
initialization for training. It claimed a test lease, verified the archive hash
and 4/4 engine readiness, then submitted **four concurrent native-budget n=32
requests** through port 24800. Every request used a 64-token output cap and an
8-token thinking cap, so this is not a full-length generation benchmark.

All **128 completions passed** token/logprob/mask length and finiteness checks,
native thinking-budget audits, and forced-token masking checks. Four group
latencies were 16.99, 17.07, 17.25 and 17.47 seconds. The lease was released and
the farm returned to `unleased` before the training submission.

The initial harness attempt supplied a text prompt, which native-budget
inference correctly rejected: `thinking-budget completions require one token-ID
prompt`. The harness now tokenizes through the leased `/tokenize` endpoint;
the production client already supplies token IDs. That initial attempt also
released its lease. No farm restart or serving-code change was needed.

`batch-probe-results.json` preserves the passing summary. No new claim of
concurrent-versus-sequential numerical parity is made.

## Deployment evidence

`submission.json` records the job ID, immutable archive URI and checksums.
The bundle was uploaded with create-only semantics and downloaded again to
verify its SHA-256 before submission. `submit.py` writes an attempt receipt
before dispatch and refuses to resubmit automatically.

At submission, startup/generation/training progress is not yet validated.
Inspect the new run's `driver.log`, `client.log`, and
`inference-events.jsonl` on its assigned worker. `borrow_ready` must show the
specified farm and verified adapter hash; `borrow_generated` confirms actual
external generations. A `RUNNING` SkyPilot row alone does not prove these.

## Packaging correction and replacement

The initial submission used the generic executor builder, omitting the top-level
science modules, native case inputs and science host setup. It failed importing
`tpu.science.training_setup`. This was a packaging error, not an inference farm
failure. Job 1262 was cancelled rather than allowing more automatic retries.

The public generic builder now automatically delegates science profiles to the
science packager. Its internal executor-only mode is used by that packager.
A regression test extracts the returned archive and imports controller, grader,
seed and borrowing modules in an isolated subprocess outside the checkout.
The focused packaging, borrowing and command suites passed **75 tests**.

`fix1/artifact-check.json` records checks on the actual replacement circuit
archive: isolated imports, all 17 Xplace/native input hash checks, and successful
compilation of the supplied C helper. The submission reads the archive URI and
checksum from the generated task, verifies the local artifact-check evidence,
and verifies the uploaded bytes before dispatch. `fix1/submission.json` records
job 1270 and its new output/cache namespace. The original submission receipt is
preserved; model, training and sampling settings are unchanged.

At replacement submission, farm 1251 / worker 173 was healthy, unleased and had
no active requests or exhausted engines. Full training and borrowed generation
remain to be verified from live events, not from the scheduler's RUNNING label.
