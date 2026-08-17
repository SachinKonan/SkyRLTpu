# Muse-Glimmer-30B at 22k context: 1×TP=4 vs 2×TP=2 (torch/vLLM path)

Job **3714197**, 2026-08-17, one spot v5p-8 (QR `sk7524-museglimmer-tp4fix`,
us-east5-a, landed in 226 s), **2521 s active = 2.80 chip-hours** for both the
TP=4 fixed-build regression (VLLM-IMPL.md §11) and this benchmark. Submodule:
clean `git archive` of **afe0cb9e9** (the fixed build). Harness:
`vllm_tp4_bench_tpu.sh` + `mg_bench22k.py`; prompts are **real erdos rollout
prompts** — env-built token ids lifted verbatim from
`muse-rs2/manifests/erdos.json` (10 prompts, 631–3847 tokens).

Why 22528: the RL serving profile is ~64–96 concurrent sequences generating
~11k-token responses; `max_model_len=22528` holds prompt + response with
margin, and it is past the regime where the KV-capacity question bites.

## Configuration (boot-log truth)

| | 1×TP=4 | 2×TP=2 |
|---|---|---|
| HBM/chip, model resident | 13.09 / 95.74 GiB | 25.98 / 95.74 GiB |
| KV pool (tokens) | 3,024,576 | 2,504,832 × 2 = 5,009,664 |
| max concurrency @22528 | **134.3** | **111.2 × 2 = 222.4** (1.656×) |
| boot to first answer (warm weights, cold cache) | 105 s | 115 s + 88 s |
| `max-num-seqs` | 128 | 128 per engine |

The capacity ratio reproduces the §7/tp_compare 1.656× exactly — the split
pays one extra weight replica and gets back double KV density from unpadded
KV heads (`get_padded_num_heads(2, TP) = TP`).

## Decode throughput ladder

`steady` = steady-state decode rate by differencing two otherwise-identical
timed passes (256 vs 768 generated tokens per request): prefill, ramp and
client overhead cancel. `raw` = long-pass completion tokens / wall. Offered
concurrency is round-robined across both engines in the 2×TP=2 arm and both
run concurrently. Each bucket was preceded by an untimed warm pass at the same
concurrency (64 tokens/request).

| offered c | 1×TP=4 steady | 1×TP=4 raw | 2×TP=2 steady | 2×TP=2 raw |
|---|---|---|---|---|
| 1 | **92.5** | 92.6 | 67.9 | 67.8 |
| 8 | n/m † | 625.3 | n/m † | 494.0 |
| 32 | 1333.3 | 1277.5 | 1471.0 | 1447.7 |
| 64 | **1622.1** | 1546.7 | 823.8 ‡ | 1020.2 ‡ |
| 96 | **1696.9** | 1645.1 | **2569.8** | 2449.0 |
| 128 | 809.1 ‡ | 975.8 ‡ | **2735.6** | 2723.6 |

† c=8, both arms: the *short* pass absorbed a residual XLA compile (its wall
≈ the warm pass's, ~45 s; the long pass is clean at ~10–12 s), so the
difference is meaningless. The long-pass raw is the usable number.

‡ One long-pass stall per arm (1×TP=4 at c=128: 100.7 s; 2×TP=2 at c=64:
48.2 s), each 3–5× off the trend of its neighbours while the short pass at the
same concurrency is clean. The signature is a `(padded batch, kv-length
bucket)` XLA compile first crossed *inside* the 768-token pass — the warm and
short passes cover the batch shape but only reach 64/256 tokens of kv growth.
These two rows are floors, not measurements; the adjacent clean rows bound the
truth. (This is the same failure mode that inflated the old saturated
number — see below — just smaller, because everything else here is warmed.)

## Verdict for the RL serving profile (~64–96 concurrent, long generations)

- **2×TP=2 is the shape to deploy: ~2570–2736 tok/s per host** at 96–128
  offered concurrency (clean points), against the wide engine's best of
  ~1620–1700 at 64–96. **The split wins ~1.5–1.6× at the RL profile**, on top
  of its 1.656× KV-capacity headroom (222 vs 134 max sequences @22528).
- **The old 6.687× saturated-throughput figure is retired.** Measured with
  per-bucket warming and differencing, the split's true advantage at
  saturation is 1.5–1.6×. The prior number (E2E.md §7.2, one unreplicated
  point with `SKIP_JAX_PRECOMPILE=1` and no warm pass) paid wide-engine XLA
  compilation inside its timed window, exactly as suspected there.
- **Batch-1 latency still belongs to the wide engine**: 92.5 vs 67.9 tok/s
  (1.36× here; the JAX head-to-head measured 1.73× at 16384). Serve wide only
  if single-stream latency is the objective.

## Caveats, all honest

1. **Prefill was not measured cold.** The 3847-token and (synthetic
   concatenation) 7693-token probes returned ~0.03 s medians — that is vLLM's
   **prefix cache** answering, primed by the probe's own warm runs. Cold
   prefill needs `--no-enable-prefix-caching` or per-repeat unique prompts;
   next slice. (The decode ladder is unaffected: generated-token KV is
   per-sequence regardless of shared prompt blocks, and differencing removes
   the prefill term entirely.)
2. Requests cycle 10 unique prompts, so above c=10 the prompt KV is shared via
   prefix caching — decode cost is per-sequence and real, but prompt-KV
   pressure is lower than 128 truly-distinct prompts would produce. At these
   pool sizes (≤ ~123k unique KV tokens vs 3.0M pool) nothing was near a
   capacity cliff either way.
3. Single repeat per bucket. The two ‡ rows show exactly why medians of ≥3
   would be better; the budget went to covering two arms × six buckets
   instead. Generations are `ignore_eos` at fixed length, temperature 0.
4. Response length here is 768 tokens/request, not the RL profile's ~11k.
   Steady-state decode rate at a given *live batch size* is the right
   transferable quantity, but very long generations grow kv-length buckets
   (more compile points, slowly rising per-token cost) that this ladder only
   samples up to prompt+768.

## Reproducing

```bash
sbatch --export=ALL,ZONES=us-east5-a,ZONE_TRY_SEC=99999,LAND_SEC=5400 \
    tpu/muse_glimmer/vllm_tp4_bench_tpu.sbatch
```

Artefacts: `runs/muse_glimmer/res_bench_1xtp4.json`,
`res_bench_2xtp2.json`, `res_tp4_e2e.json`, `res_lora_tp4.json`,
`report-{torch4,lora4,bench4,bench2a,bench2b}.txt`, log
`tp4bench-3714197.log`. The prompt pack builder is inline in the session
notes; `runs/muse_glimmer/bench22k_prompts.json` is the shipped artefact.
