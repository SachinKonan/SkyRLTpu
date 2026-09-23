# Inference farm versus local inference — September 22, 2026

**Correction from the subsequent engine audit:** Muse's local engines use default RPA (`USE_BATCHED_RPA_KERNEL=0`), while all four farm engines use experimental batched RPA (`1`), with KV block sizes 16 versus 256. The original configuration comparison below omitted this material difference. Timings remain valid, but batch size alone must not be treated as the explanation. The full corrected comparison, including Gemma, is in [MUSE_GEMMA_LOCAL_FARM_DIFFERENCES_2026-09-22.md](MUSE_GEMMA_LOCAL_FARM_DIFFERENCES_2026-09-22.md).

The three models behave differently. Qwen's v4 farm is broadly comparable to its local v4 engines and contributes useful parallel capacity. Muse's v4 farm is approximately 3.5 times slower on completed groups and is now being excluded by the scheduler while its lease is retained. Gemma's previous v5p farm was approximately 3.5 times faster than local v4 on completed groups; its replacement v4 farm is instead constrained by KV-cache capacity and is currently slower than local v4.

This is a read-only operational analysis of qubit-routing PWC jobs 1482 (Muse), 1483 (Qwen), and 1493 (Gemma). No jobs, farm configurations, scheduler settings, leases, or TPU resources were changed. The cancelled Gemma GRPO job 1496 is excluded. Measurements were collected around 10:58–11:08 a.m. EDT; the current-Gemma engine comparison ends at 11:01 a.m. EDT. Times in the JSON evidence are UTC.

## What is measured

Two distinct measurements are used:

1. **Completed group latency:** from scheduler dispatch until a complete 32-rollout response is returned. It includes backend queueing, prefill, decoding, response transfer, and any work performed within that call. It excludes time waiting in the training worker's scheduler before dispatch, adapter preparation before dispatch, grading, and training. Failed requests and unfinished requests are not represented as successful completions.
2. **Engine decode throughput:** vLLM's generated-token throughput samples, taken from all four engines on each side during overlapping observation windows. This separates engine generation speed from network/HTTP completion time. Full-occupancy and all-active samples are distinguished because Gemma rarely reaches its configured sequence limit.

For completed-request comparisons, I restricted the analysis to complete 16-group phases with both local and remote results. Groups are associated with the same committed adapter version using dispatch time and adapter-publication events. This controls policy version and batch, but **does not pair identical prompts or random seeds**. The scheduler chooses routes non-randomly. These are observational comparisons, not randomized benchmarks or statistically established causal effects.

The surviving current worker attempts cover different periods because Qwen and Gemma recovered onto replacement workers. Counts below are those selected phases, not lifetime rollout totals. Engine throughput samples are correlated observations, not independent experimental repetitions.

## Completed 32-rollout groups, within the same adapter phases

| Model | Local / farm groups | Farm hardware | Local median group time | Farm median group time | Median output tokens per rollout, local / farm | Output tokens per request-second, local / farm |
|---|---:|---|---:|---:|---:|---:|
| Muse | 12 / 4 | v4-32, job 1490 | 13.37 min | 47.83 min | 10,134 / 10,248 | 398.8 / 115.1 |
| Qwen | 24 / 24 | v4-32, job 1484 | 17.55 min | 19.87 min | 6,995 / 10,024 | 207.2 / 269.4 |
| Gemma | 4 / 12 | **v5p-32, former job 1507** | 32.08 min | 9.27 min | 6,333 / 6,520 | 108.3 / 381.3 |

“Output tokens per request-second” is total output tokens divided by summed durations of requests on that route. It is not wall-clock throughput of the entire four-engine fleet: requests overlap. The observed farm/local ratios for this measure are **0.289x Muse, 1.300x Qwen, and 3.521x Gemma-v5p**.

The median times have these 10th–90th percentile ranges: Muse local 12.92–13.89 min, farm 47.19–48.16 min; Qwen local 12.46–18.87 min, farm 18.96–20.94 min; Gemma local 25.99–33.45 min, v5p farm 8.68–9.56 min. Gemma's local sample is only four groups.

Observation windows and versions:

- Muse: 01:59:50–02:48:06 EDT, adapter `model_bd1b930e_000002`.
- Qwen: 06:14:54–09:49:20 EDT, three complete phases: `model_885e53f2_ss0_seq2`, `model_885e53f2_000005`, and `model_885e53f2_000006`.
- Gemma-v5p: 06:09:08–06:43:00 EDT, adapter `model_401e7237_ss0_seq2`. Discovery history identifies endpoint `10.202.0.204:24800` as job 1507 in the v5p32-east5a pool. That job is now cancelled. Its performance must not be attributed to current v4 farm job 1485 at `10.130.1.31:24800`.

## Qwen: useful farm capacity, little engine-speed penalty

At 16 simultaneously running sequences, the four local engines' median decode rates were **339.1–340.8 tokens/s per engine**, versus **352.0–353.6 tokens/s** on the farm. The farm is approximately 4% faster at this occupancy, not dramatically slower.

Its group latency is nevertheless 13% longer because the farm groups returned much more output: the median output per rollout is approximately **43% longer**. Local groups also spend more time with partially occupied engines: median running sequences by local engine were 9–13, versus 16 on each farm engine. The farm's higher completed-request output rate therefore must not be interpreted as a pure 30% kernel speed improvement.

Each of the three complete phases split **8 local groups / 8 farm groups**. The observed generation span from first dispatch to last completion was 40.96, 40.87, and 40.51 minutes. These are generation spans, not full optimizer-step times or measured speedups against an otherwise identical local-only run.

The local and farm configuration both use TP4, max 16 sequences, 22,528 context, 1,024 batched tokens, and memory utilization 0.8. Differences include **prefix caching on locally versus off on the farm**, and max-LoRA slots **8 locally versus 1 on the farm**. Runtime compatibility hashes match, but the attestation deliberately does not establish equality of every performance setting or generated trajectory.

KV-cache usage remained below 47% in the compared samples. There is no indication that KV capacity is limiting this Qwen comparison. No borrower generation/health failure events appeared in the selected current-attempt event file.

**Interpretation:** retain farm assistance. Investigate the output-length difference with identical prompt/adapter replay before treating the two routes as fully equivalent. Different routed parents and stochastic generations are possible explanations; the observations alone do not establish a correctness bug.

## Muse: smaller batches plus lower decode speed, followed by scheduler exclusion

During its mixed phase, the four local engines' median full-occupancy decode rates were **462.3–463.9 tokens/s**, versus **122.8–131.2 tokens/s** on the farm. Output lengths were similar, so the approximately 3.6x latency gap is not explained by longer farm answers.

The farm's actual launch command sets **max 8 sequences**, while local inference uses **16**. A group of 32 consequently starts with **8 running / 24 waiting** on the farm versus **16 running / 16 waiting** locally. Even after accounting for occupancy, decode rate is approximately 15–16 tokens/s per active farm sequence versus 29 locally. A controlled configuration replay is required to determine how much of that residual comes from batch-shape efficiency versus other settings.

Other measured settings: memory utilization **0.70 farm / 0.75 local**, max-LoRA slots **1 / 8**. Both use TP4, the same Muse model revision, 22,528 context, 1,024 batched tokens, prefix caching enabled, JAX serving, and matching runtime compatibility hashes. All three currently inspected Muse farm deployments advertise max 8 sequences; a job name containing “prefix16” is not evidence of a 16-sequence engine limit.

All four farm engines decoded continuously during the slow requests. The window had no logged compile pauses, preemption warnings, or errors. Farm KV-cache usage peaked at 22.9–28.4% across the four engines, versus 47.3–50.6% locally. The measured request slowdown is therefore consistent with sustained engine throughput, not an HTTP stall, large compilation pause, or observed KV exhaustion. Low KV occupancy alone does not prove enough overall TPU memory headroom to raise concurrency safely.

The scheduler then learned a low remote rate from those four groups and stopped dispatching to the farm. A live snapshot showed **4 local requests, 8 queued groups, 0 remote requests, and a ready four-engine lease**. A later snapshot still showed the same unused lease. Estimates only update after successful completions, so remote exclusion prevents the farm from receiving work that could refresh its estimate. Lease renewal is independent of use and keeps reserving the idle farm.

There were four earlier failed remote generation events associated with the previous Muse farm failure; the slow successful phase is a separate observation. It is not valid to count failed attempts as successful slow requests or infer the current farm is broken merely from its lack of requests.

**Interpretation:** the farm is slower under its current settings, and the scheduler compounds the problem by permanently avoiding it while retaining ownership. Test one isolated farm engine against the local 16-sequence configuration, then refresh routing estimates from the result. Do not assume a concurrency increase alone resolves the full gap.

## Gemma: distinguish the fast v5p farm from the constrained v4 replacement

The former v5p farm completed 12 groups in approximately 9.3 minutes each while four local groups took approximately 32.1 minutes. Output lengths were similar. This is a useful observed benefit, but it combines different TPU generations and memory capacities. Live engine logs for the retired v5p worker were not available in this collection; the completed-response timing and endpoint/hardware identity are verified.

The **current v4 farm is not reproducing that performance**. Across the overlapping 10:27:41–11:01:23 EDT window:

| Metric, per engine | Local v4 engines | Current v4 farm |
|---|---:|---:|
| Configured max sequences | 16 | 16 |
| Median simultaneously running sequences | 5–6 | 2–3 |
| Median decode throughput while active | 95.5–108.2 tokens/s | 41.6–60.6 tokens/s |
| Peak KV-cache utilization | 99.9–100% | 99.1–100% |
| KV capacity, two inspected engines on each side | 32,512 tokens | 42,496 tokens |
| Prefix caching | Enabled | Disabled |
| Max-LoRA slots | 8 | 1 |
| Memory utilization setting | 0.8 | 0.8 |

None of the 203 samples per current farm engine reached 16 simultaneously running sequences. The farm had no completed groups in the head snapshot; these unfinished groups must not disappear into a success-only latency comparison. Its requests had already been producing tokens for over 30 minutes by the end of the engine window.

Both local and farm engines are KV-constrained. vLLM reports maximum concurrency for full 22,528-token sequences of **1.44x local / 1.89x farm**. A max-sequence flag of 16 does not guarantee that 16 long sequences fit. Local prefix-cache hit rates were around 97–98%; current farm rates were zero. Shared prompt prefixes plausibly allow more local concurrency despite the smaller total KV cache. This is strongly supported by the observed capacity/occupancy difference, but a controlled prefix-cache experiment is needed to isolate causality. Cache correctness must be validated before enabling it on the farm; earlier investigation of Qwen-style recurrent caching must not be generalized blindly to every model.

No preemption warnings were present in the inspected Gemma logs. Low active concurrency and nearly full KV cache are direct observations; do not describe this as confirmed repeated preemption or recomputation.

Gemma also reveals a routing-state problem: the scheduler retained its high remote rate from the old v5p farm after moving to a slower v4 endpoint. Its successful-completion estimate had not yet incorporated a current-v4 completion. A single “remote” speed estimate is inappropriate across farms with different hardware and settings.

**Interpretation:** the v5p result supports using larger-memory hardware for Gemma. Current v4 serving needs a KV-capacity/prefix-sharing investigation. More nominal engines or a higher max-sequence setting alone will not fix the memory bottleneck.

## What farm assistance does and does not establish

All local inference engines in this comparison are TP4 engines on the inference hosts of a v4-64 training slice; the v4 farms also use four TP4 engines. A v4-64 slice is not one 64-chip inference engine. The useful comparison is four local inference engines plus four optional external engines, with model-dependent effective capacity.

Farm generation does not change the intended reward/loss recipe: whole rollout groups enter the same downstream pipeline. It does change routing, concurrency, failure exposure, and observed generation behavior. Matching compatibility hashes is evidence of the selected model/runtime identity contract, not exact output equivalence.

Successful request timings are not full training speedups. Grading, long-tail groups, initialization, adapter upload, recompilation, checkpointing, and spot recovery also affect step time. The logs establish overlapping local/farm generation, not a universal 2x reduction in optimizer-step wall time.

## Operational weaknesses and recommended order

1. **Fix routing accounting across all models.** Track rates per farm identity and configuration, age stale observations, periodically probe excluded routes, and consider queued groups when predicting finish time. Gemma needs a fresh estimate after a v5p-to-v4 change; Muse needs a way to escape permanent exclusion.
2. **Release healthy but persistently unused leases.** Heartbeat success is not useful utilization. Preserve deliberate short holds across optimizer updates, but distinguish those from hours of no remote dispatch.
3. **Add generation-progress detection.** Existing lease/health checks can invalidate a failed farm and retry affected requests locally. A healthy control plane does not prove decoding progress. These profiles use a 21,600-second generation HTTP timeout; a silent stall can wait a long time. The HTTP timeout is not an independent token-progress watchdog.
4. **Muse:** compare one isolated engine with the working local max-sequences=16 configuration, including memory stability; then refresh scheduling estimates. The existing max-sequences=8 configuration is a measured throughput disadvantage.
5. **Gemma:** prioritize adequate KV capacity. Compare safe prefix reuse and memory allocation on v4, or use v5p capacity whose benefit is already observed. Do not promise the former v5p speed from the current v4 farm.
6. **Qwen:** retain assistance and validate the systematic output-length difference with identical prompt and adapter replay. Its engine throughput is already healthy.

No changes from this recommendation list were deployed by this analysis.

## Evidence and reproducibility

The compact numerical artifact is [summary.json](tpu/science/results/farm-local-comparison-20260922/summary.json). It includes route counts, time windows, matched versions, latency percentiles, output lengths, per-engine throughput, KV occupancy, and actual launch settings.

Private raw evidence is under `.science/routing-relaunch-20260921/farm-local-analysis-20260922/`: `head-<job>.json`, `engines-<job>-<rank>.json`, `gemma-previous-farm-discovery.json`, `gemma-previous-farm-job.json`, and `gemma-kv-diagnosis.json`. Read-only collection and aggregation helpers are `analyze_farm_local.py`, `collect_engine_comparison.py`, `inspect_gemma_kv.py`, and `summarize_farm_comparison.py` in the parent evidence directory. All 36 engine-host probes succeeded. The collected raw files remain on disk; no opaque credentials or lease tokens are included in the public summary.
