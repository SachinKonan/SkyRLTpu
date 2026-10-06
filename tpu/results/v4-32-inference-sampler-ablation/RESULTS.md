# v4-32 Inference Sampler Ablation

Date: 2026-09-05

## Summary

This ablation measured TP4 inference throughput for Qwen3.5, Muse Glimmer,
and Gemma 4 on a v4-32 slice. It used both a short-decode synthetic workload
and a replay of the actual grouped improvement-generation workload.

The production recommendations from the completed measurements are:

| Model | Recommended `max_num_seqs` | HBM utilization | Basis |
| --- | ---: | ---: | --- |
| Qwen3.5-27B | 32 | 0.80 | Best stable short-decode result |
| Muse Glimmer | 16 | 0.95 | Best stable short-decode result; higher concurrency crossed a sharp knee |
| Gemma 4 31B | 16 | 0.90 | Full realistic replay; 18.0% faster than `max_num_seqs=128` |

The central finding is that a high sequence cap helps when generated
continuations are short, but does not imply high sustained concurrency for
long generations. Once the KV cache is full, `max_num_seqs` is only a ceiling.

## Workloads

### Synthetic short-decode workload

- TP4 inference on one v4-32 slice.
- Approximately 21.5K prompt tokens per request.
- Prefix caching enabled.
- 256 generated tokens per sequence with EOS ignored.
- Concurrency equal to `max_num_seqs`.
- Two measured rounds per result.
- Throughput is aggregate completion tokens divided by aggregate measured
  elapsed time. The per-round rates are included because some settings were
  highly unstable.

### Realistic improvement replay

- Fifteen real states: five each from Erdos, JSSP, and AC1.
- Group size 32, for 480 generated sequences per cycle.
- One sequence per HTTP request.
- Prompt lengths from 2,907 to 6,610 tokens for Gemma.
- Phase-one cap of 13,824 tokens and total context window of 22,528 tokens.
- Temperature 1.0, `logprobs=1`, and prefix caching enabled.
- A second continuation phase was used when a sequence exhausted its
  phase-one budget.

## Realistic Results

Cycle time excludes model setup and warmup.

| Model and configuration | Completion tokens | Cycle time | Completion tok/s | Mean length | Median length | Phase-two sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Muse `s16/u0.90` | 5,512,056 | 2h 27m 55s | **621.05** | 11,483 | 11,652 | 445 |
| Qwen `s16/u0.80` | 2,917,573 | 2h 7m 40s | **380.87** | 6,078 | 4,252 | 109 |
| Gemma `s16/u0.90` | 4,123,687 | 4h 44m 21s | **241.70** | 8,591 | 8,115 | 141 |
| Gemma `s128/u0.90` | 4,148,064 | 5h 37m 31s | **204.83** | 8,642 | 8,410 | 138 |

The two Gemma runs produced almost the same token volume: the `s128` run
produced only 0.59% more tokens. Nevertheless, `s16` completed the cycle 3,190
seconds sooner and delivered 18.0% more completion tokens per second. Both
managed jobs completed with zero recoveries:

- Job 199: Gemma `s16/u0.90`.
- Job 207: Gemma `s128/u0.90`.

### Gemma by problem

The `s16` advantage was present in every problem subset.

| Configuration | Problem | Completion tokens | Tok/s over complete cycle | Mean length | Phase-two sequences |
| --- | --- | ---: | ---: | ---: | ---: |
| `s16/u0.90` | Erdos | 1,280,644 | 75.06 | 8,004 | 8 |
| `s16/u0.90` | JSSP | 1,660,439 | 97.32 | 10,378 | 109 |
| `s16/u0.90` | AC1 | 1,182,604 | 69.32 | 7,391 | 24 |
| `s128/u0.90` | Erdos | 1,316,014 | 64.98 | 8,225 | 9 |
| `s128/u0.90` | JSSP | 1,635,020 | 80.74 | 10,219 | 104 |
| `s128/u0.90` | AC1 | 1,197,030 | 59.11 | 7,481 | 25 |

Cross-model throughput should not be interpreted as a model-speed ranking:
the models generated different output-length distributions and invoked the
second phase at very different rates. Comparisons within the two Gemma arms
are directly informative because workload shape and token volume were closely
matched.

## Synthetic Results

### Gemma 4 31B

| Configuration | Round 1 tok/s | Round 2 tok/s | Aggregate tok/s |
| --- | ---: | ---: | ---: |
| `s32/u0.90` | 207.43 | 209.03 | 208.23 |
| `s64/u0.90` | 329.93 | 326.88 | 328.40 |
| `s64/u0.95` | 327.27 | 329.45 | 328.36 |
| `s128/u0.90` | 473.91 | 473.86 | 473.89 |
| `s144/u0.90` | 496.20 | 495.76 | **495.98** |

Gemma scaled cleanly through `s144` in this short-decode workload. The
realistic replay demonstrates that this result must not be used to select the
long-generation production setting.

### Qwen3.5-27B

| Configuration | Round 1 tok/s | Round 2 tok/s | Aggregate tok/s | Assessment |
| --- | ---: | ---: | ---: | --- |
| `s16/u0.80` | 373.00 | 369.77 | 371.38 | Stable |
| `s32/u0.80` | 564.61 | 556.46 | **560.51** | Best stable result |
| `s32/u0.85` | 73.90 | 512.67 | 129.18 | Unstable |
| `s64/u0.85` | 124.25 | 129.49 | 126.82 | Stable but collapsed |

Qwen's stable short setting was `s32/u0.80`. Raising HBM utilization to 0.85
introduced severe instability, and `s64/u0.85` was consistently slow.

### Muse Glimmer

| Configuration | Round 1 tok/s | Round 2 tok/s | Aggregate tok/s | Assessment |
| --- | ---: | ---: | ---: | --- |
| `s8/u0.85` | 51.80 | 260.24 | 86.40 | Unstable |
| `s16/u0.90` | 601.26 | 557.99 | 578.82 | Usable |
| `s16/u0.92`, first run | 57.87 | 105.39 | 74.71 | Unstable |
| `s16/u0.92`, rerun | 617.60 | 556.13 | 585.26 | Usable |
| `s16/u0.95`, first run | 53.76 | 593.72 | 98.59 | Unstable |
| `s16/u0.95`, rerun | 611.03 | 587.34 | **598.95** | Best stable result |
| `s16/u0.98` | 89.85 | 580.32 | 155.61 | Unstable |
| `s20/u0.95` | 131.45 | 660.79 | 219.29 | Unstable knee |
| `s24/u0.95` | 158.78 | 157.87 | 158.33 | Stable collapse |
| `s32/u0.92` | 197.95 | 172.47 | 184.33 | Collapsed |

Muse has a sharp operating knee. `s16/u0.95` was the best stable rerun;
increasing either utilization to 0.98 or concurrency beyond 16 caused unstable
or consistently collapsed throughput.

## Why Short Decodes Scale

The synthetic prompt is long, but prefix caching allows its common blocks to
be reused. Each sequence contributes only 256 unique generated-token positions
to the KV cache. At concurrency 144 this is roughly 36,864 generated-token
positions, and the large batch improves TPU utilization while amortizing
scheduler, sampler, and kernel-launch overhead.

The realistic Gemma sequences averaged approximately 8,600 generated tokens
and reached as high as 15,193. Generated continuation KV state cannot be shared.
Even 16 average-length continuations represent roughly 137,600 unique
generated-token positions before prompts and phase-two work are considered.
During the realistic runs, KV utilization stayed near capacity and only a
single-digit to low-teens number of long generations generally remained active,
even when `max_num_seqs` was 128. The remaining requests waited for capacity.

Thus, short decoding is primarily helped by wider batching, while realistic
long decoding becomes KV-capacity and memory-traffic limited. A larger sequence
cap cannot create memory capacity and may add contention without increasing
the sustained batch.

## Applicability to v5p

The same mechanism applies to v5p: autoregressive continuations create unique,
growing KV state, and prefix caching only shares prompt blocks. The exact v4
threshold does not transfer, however. V5p has different memory capacity,
bandwidth, topology, and kernels, so its realistic optimum may be `s16`, `s32`,
or higher.

The v5p follow-up should replay the same 480-sequence workload at a minimum of
`s16` and `s32`, adding `s64` if actual running concurrency is not yet limited
by KV capacity. Selection should use full-cycle completion tok/s together with
actual running requests, capacity-waiting requests, KV utilization, and any
request preemption or recomputation. A 256-token synthetic sweep alone is not
sufficient to select the production setting.

## Result Artifacts

Synthetic results:

```text
gs://sk7524-tinker-tpu-us-central2/v4-smoke-results/sweeps/
```

Realistic results:

```text
gs://sk7524-tinker-tpu-us-central2/v4-smoke-results/realistic-improvement/
```

The realistic directory contains the completed Muse, Qwen, Gemma `s16`, and
Gemma `s128` JSON artifacts used in this report.

## Invalid Gemma Attempts

Two setup attempts did not produce benchmark results and are excluded from the
tables:

- Job 197 used a generic bundle with `max_num_batched_tokens=1024`; startup
  rejected Gemma's 2,496-token multimodal item limit.
- Job 198 raised the chunk size to 4,096 but reached an unsupported v4 path in
  the ragged-paged-attention v3 kernel.

Jobs 199 and 207 used the proven pinned v4 bundle and completed the corrected
one-sequence-per-request replay.
