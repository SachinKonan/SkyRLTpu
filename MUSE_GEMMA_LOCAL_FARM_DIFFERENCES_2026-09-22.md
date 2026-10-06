# Muse and Gemma: local versus farm inference, September 22, 2026

## Scope and conclusion

This records all differences established by the inspected launch contracts, engine startup logs, runtime attestations, and generation measurements. It is not a claim to have exhaustively compared native binaries, hardware counters, every process environment variable, or identical-prompt outputs. Investigation was read-only on TPU workers; no jobs, leases, or engines were changed.

**Muse has a newly verified attention-kernel mismatch.** All four local engines use default RPA, while all four farm engines use experimental batched RPA. The farm also has half the sequence concurrency. These are concrete configuration differences and strong performance suspects; their individual contributions require a controlled replay.

**Gemma's observed difference is effective concurrency under KV-cache pressure.** Local inference uses prefix caching; the current v4 farm disables it. Local engines reuse shared prompts and fit more active sequences. The previous fast Gemma farm used v5p hardware and must not be confused with this replacement v4 farm.

## Muse: configuration and derived runtime differences

| Item | Local inference | Farm inference | Interpretation |
|---|---|---|---|
| Job / worker | Qubit PWC 1482 / v4-64 worker 727 | Farm 1490 / v4-32 worker 182 | Separate slices; both have four TP4 inference engines |
| Hosts in entire slice | 8, split between training and inference | 4 inference hosts | Each inference engine still uses four v4 chips |
| `USE_BATCHED_RPA_KERNEL` | `0` | `1` | Verified in launch contracts on all four engines per side |
| Loaded attention implementation | `ragged_paged_attention.v3.kernel` | `experimental.batched_rpa.wrapper` | Confirmed by startup logs and installed source |
| KV page/block size | 16 tokens | 256 tokens | Installed TPU platform code raises block size to at least 256 for batched RPA |
| Maximum active sequences per engine | 16 | 8 | Farm processes a 32-rollout group with a lower concurrency ceiling |
| Maximum LoRA slots | 8 | 1 | Capacity setting; does not mean eight adapters participate in local generation |
| Memory-utilization setting | 0.75 | 0.70 | Allocation budget, not measured device utilization |
| KV token capacity per engine | 312,608 | 328,960 | Startup logs agree across all four engines on each side; farm has more KV tokens despite its lower total allocation budget |
| KV block count per layer | 19,538 | 1,285 | Different block sizes; block counts cannot be compared directly as capacity |
| KV tensor shape per block | `(16, 4, 2, 128)` | `(256, 4, 2, 128)` | Same head-related dimensions; different page size |
| HBM immediately after KV initialization, per chip | 23.06 / 30.75 GiB | 21.52 / 30.75 GiB | Startup snapshot; not peak execution memory |
| Reported concurrency at full 22,528-token length | 13.88 | 14.60 | Capacity estimate; separate from configured 16/8 sequence ceilings |
| Native source bundle directory | `native-7cabe6dd...` | `native-a0ef1035...` | Bundle identities differ, but checked server entrypoints and attested Python package contents match |
| Storage/cache namespace | `~/.cache/qubit-v4-muse-parallel2-pwc-rho05-20260921/` | `~/.cache/farm10-muse-1-20260921/` | Separate environments, model-cache paths, compilation caches, and adapter directories |
| Invocation path | Training coordinator to local Serve engines | Borrower to external farm Serve, with lease and adapter distribution | Extra control/transport layer; measured slow decoding occurs inside farm engines |
| Startup history visible on inspected hosts | Current engine startup around 01:02 UTC | Earlier startup around 00:07 UTC and replacement around 01:52 UTC | Do not combine historical engine instances with the later comparison window |

The differing path values in the launch environment are `JAX_COMPILATION_CACHE_DIR`, `VLLM_XLA_CACHE_PATH`, and `VLLM_LORA_RESOLVER_CACHE_DIR`. Commands also differ in Python executable path, native source path, model snapshot path, adapter directory, and download directory because each run has its own namespace. These path differences do not by themselves establish a performance cause.

### Muse: measured effects

The same-adapter mixed phase was 01:59:50–02:48:06 EDT, adapter `model_bd1b930e_000002`. Routed prompts and random generations were not identical.

| Measurement | Local | Farm |
|---|---:|---:|
| Completed groups in selected phase | 12 | 4 |
| Rollouts per group | 32 | 32 |
| Median group completion time | 13.37 min | 47.83 min |
| Median output tokens per rollout | 10,134 | 10,248 |
| Median full-occupancy decode throughput across engine medians | 462.3–463.9 tokens/s | 122.8–131.2 tokens/s |
| Active sequences at that occupancy | 16 | 8 |
| Approximate decode throughput per active sequence | 29 tokens/s | 15–16 tokens/s |
| Peak KV utilization across engines in compared window | 47.3–50.6% | 22.9–28.4% |

Both smaller batches and a different attention implementation are present. The farm continuously decoded during the slow groups; the gap is not explained by longer outputs or an observed HTTP-only stall. No compile pause or preemption warning was seen in this window. Low KV utilization does not establish safe peak HBM headroom for changing concurrency.

At the 12:37 p.m. EDT status check, the scheduler had four local requests active, four queued, and zero remote requests, while holding a ready lease for four farm engines. Its remote-rate estimate still reflected the earlier slow groups. This is a second problem: a healthy slow farm can remain leased but unused because the scheduler stops sending it work that could update its estimate.

### Muse: verified matches

- Model `meta-models/Muse-Glimmer-30B`, revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`; attested config/tokenizer/generation-config/index hashes match.
- Four TP4 engines, per-engine mesh `data=1, model=4`, four local devices, 30.75 GiB HBM per device, chip bounds `2,2,1`, process bounds `1,1,1`, visible chips `0,1,2,3`.
- Context length 22,528, batched-token limit 1,024, rank-32 LoRA support, bfloat16, prefix caching enabled in the current compared attempt, and chunked prefill enabled.
- `TPU_BACKEND_TYPE=jax`, `MODEL_IMPL_TYPE=vllm`, `SKIP_JAX_PRECOMPILE=1`, `USE_JAX_RAGGED_CONV1D=1`, `VLLM_USE_RAY_EXECUTOR=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- JAX compilation cache enabled, persistent cache minimum time and entry size both zero, runtime LoRA updating enabled.
- Attested Python-source fingerprints match for tpu-inference, transformers, torchax, jaxlib, jax, torch, and vllm-tpu. Versions include vllm-tpu 0.23.0, JAX/JAXLIB 0.10.1, transformers 5.15.1, torchax 0.0.11, torch 2.10.0. This does not hash every native binary.
- Both checked serving entrypoints have matching source hashes despite different bundle-directory hashes.

### Why the compatibility check did not catch Muse's mismatch

Both endpoints advertised compatibility hash `b85b17feece06d34ab0496fa83374bd45f8b0de7ea1bfdf4f06807cab7d99a1d`. Its contract includes model identity, selected model-file hashes, server code, Python package fingerprints, native-thinking mode, model implementation, context length, and LoRA rank. It does **not** include the batched-RPA flag, KV block size, memory fraction, sequence ceiling, or LoRA-slot count. Matching this hash therefore does not prove the same execution configuration.

The local qubit profile explicitly sets `batched_rpa_kernel: false`; the generic Muse preset sets it true. The farm launch explicitly exports `USE_BATCHED_RPA_KERNEL=1`. The exact farm-profile inheritance history has not been fully reconstructed, so inheritance is a plausible origin rather than a proven deployment history.

Installed source establishes the mechanism: `tpu_inference/layers/common/attention_interface.py:50` selects the experimental wrapper when the flag is true; `tpu_inference/platforms/tpu_platform.py:307` then imposes a minimum 256-token cache block. This proves that the flag changes the execution path and layout, not how much of the observed latency gap it causes.

## Gemma: differences and explanation

The comparison below concerns local qubit PWC job 1493 and replacement **v4** farm 1485. The earlier 9.27-minute remote-group result came from **v5p farm 1507**, now cancelled, and is a different hardware comparison.

| Item | Local v4 engines | Current v4 farm |
|---|---:|---:|
| Prefix caching | Enabled | Disabled |
| Observed prefix-cache hit rate | Approximately 97–98% | 0% |
| Maximum LoRA slots | 8 | 1 |
| KV capacity, two inspected engines per side | 32,512 tokens | 42,496 tokens |
| Full-length 22,528-token concurrency estimate | 1.44 | 1.89 |
| Configured maximum sequences | 16 | 16 |
| Median running sequences | 5–6 | 2–3 |
| Median decode throughput while active, per-engine range | 95.5–108.2 tokens/s | 41.6–60.6 tokens/s |
| Peak KV utilization | 99.9–100% | 99.1–100% |
| Memory-utilization setting | 0.8 | 0.8 |

The follow-up launch inspection checked two engines on each side: **both use `USE_BATCHED_RPA_KERNEL=0`**, so the Muse kernel mismatch is absent in those Gemma engines. Both also use `TPU_BACKEND_TYPE=torchax`, `SKIP_JAX_PRECOMPILE=0`, `USE_JAX_RAGGED_CONV1D=0`, the filesystem LoRA resolver, TP4, 1,024 batched tokens, rank-32 LoRA, and the same model revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. Audio/image/video input limits are zero and chunked multimodal input is disabled on both.

Gemma's inspected commands differ in prefix-cache flag, max-LoRA slots, and run-specific filesystem paths. The selected performance environment variables otherwise match after normalizing cache/adapter paths. The local namespace is `~/.cache/qubit-v4-gemma-parallel2-pwc-rho05-20260921-r2/`; the farm namespace is `~/.cache/farm10-gemma-2-20260921/`. Native bundle directories differ (`7cabe6dd...` versus `a0ef1035...`) but the inspected thinking-server source hashes match. Local worker 726 belongs to the v4-64 training slice; farm worker 185 belongs to the v4-32 pool. As with Muse, local dispatch and leased external dispatch have different coordination paths.

The performance/occupancy window is 10:27:41–11:01:23 EDT. These measurements are not a same-prompt prefix-cache A/B test. At the earlier collection cutoff, current-v4 farm groups were still unfinished and must not be replaced by success-only results from its predecessor.

Generation groups share a parent prompt. Prefix reuse can let their sequences reference already stored prompt blocks rather than each retaining separate copies. With only about 32k–42k KV tokens available, repeated long prompts materially affect how many sequences fit. Local engines demonstrably had higher prefix reuse and active concurrency despite a smaller KV cache. This strongly supports prefix reuse as the explanation for their advantage; the exact causal contribution still needs a matched test.

No preemption warnings were observed in the inspected logs. The evidence supports KV pressure and low occupancy, not a claim of confirmed repeated preemption/recomputation. The max-sequences flag is an upper bound, not a promise that 16 long sequences fit in memory.

Gemma also retained a scheduler remote-speed estimate learned on its old v5p farm after switching to v4. Its routing estimate can therefore be too optimistic, whereas Muse's can be persistently pessimistic.

## Qwen follow-up: no comparable farm throughput regression

Checked qubit PWC job 1483 and Qwen farm 1484. All four local and all four farm launch contracts set `USE_BATCHED_RPA_KERNEL=0`; inspected startup logs confirm default RPA. Both use TP4, 16 maximum sequences, 22,528 context, 1,024 batched tokens, memory fraction 0.8, `torchax`, `SKIP_JAX_PRECOMPILE=0`, and `USE_JAX_RAGGED_CONV1D=1`. Both point to model revision `fc05daec18b0a78c049392ed2e771dde82bdf654` and have matching inspected thinking-server source hashes.

The actual performance-setting differences are prefix caching **on locally / off remotely**, and maximum LoRA slots **8 / 1**. Namespace/source-bundle path differences follow the same pattern as Muse and Gemma; selected environment values otherwise match after path normalization. Representative startup logs report **569,254 / 719,906 KV tokens**, with full-length concurrency estimates **25.27 / 31.96**. These are much larger capacities than Gemma's; observed KV utilization stayed below 47% in the earlier comparison window.

The earlier same-adapter-phase measurements showed local full-occupancy decode medians **339.1–340.8 tokens/s per engine**, versus farm **352.0–353.6**. Three complete phases each used **8 local and 8 farm groups**. Thus there is no observed Muse-like decoding regression or Gemma-like KV-capacity bottleneck in this sample.

One unresolved behavioral difference remains: farm rollouts had median output length **10,024 tokens**, versus **6,995 locally** (about 43% longer). Group completion medians were consequently **19.87 versus 17.55 minutes**, despite healthy farm decode speed. Different routed parent prompts and stochastic outputs confound this comparison. It is not proof of a cache bug, nor proof of generation equivalence. Qwen's cache behavior across adapter versions warrants identical-prompt/adapter replay before enabling farm caching or declaring the routes equivalent.

Raw follow-up evidence: `qwen-detail-1483-{2,4,5,6}.json` and `qwen-detail-1484-{0,1,2,3}.json`; helper `diagnose_qwen_engine.py`, under the same private evidence paths as above. No Qwen settings or jobs were changed.

## Validation sequence, not deployed

1. For Muse, replay the same fixed adapter and prompts on an isolated engine, changing only batched RPA off/on at eight sequences. Measure warm steady decode, group latency, output lengths, and memory stability.
2. With default RPA, compare eight versus sixteen sequences; then assess the memory fraction and LoRA-slot setting separately if needed. Matching several settings at once can validate an operational fix but cannot isolate the kernel's contribution.
3. For Gemma, validate prefix-cache correctness across adapter/version reloads and replay identical shared-parent groups with caching off/on. Measure KV occupancy and actual simultaneous sequences.
4. Refresh scheduler estimates after a farm/kernel/configuration change. A faster replacement can otherwise inherit an unusable historical estimate.

## Evidence

- Prior measurements and methodology: [INFERENCE_FARM_LOCAL_ANALYSIS_2026-09-22.md](INFERENCE_FARM_LOCAL_ANALYSIS_2026-09-22.md).
- Numerical comparison: [summary.json](tpu/science/results/farm-local-comparison-20260922/summary.json).
- Private raw evidence: `.science/routing-relaunch-20260921/farm-local-analysis-20260922/`, including `muse-detail-1482-{3,4,6,7}.json`, `muse-detail-1490-{0,1,2,3}.json`, and `muse-runtime-source.json`.
- Gemma follow-up evidence: `gemma-detail-1493-{1,2}.json` and `gemma-detail-1485-{0,1}.json` in the same directory.
- Helpers: `diagnose_muse_engine.py` and `diagnose_gemma_engine.py` in the parent private evidence directory.

The earlier report's Muse configuration comparison was incomplete: it omitted `USE_BATCHED_RPA_KERNEL`. This document corrects that omission. The recorded timings remain valid; the earlier explanation cannot assign the unexplained gap solely to batch shape.
