# Muse-Glimmer-30B on TPU

JAX-native, **text-only** support for `meta-models/Muse-Glimmer-30B` — serving through
vLLM/tpu-inference and training through MaxText/tunix, both matching the HF reference.
No vision tower (out of scope by construction).

Start here, then follow the document that matches what you are doing.

| document | what it holds |
|---|---|
| [`SPEC.md`](SPEC.md) | The contract: exact forward pass and 12 parity traps. **Corrected three times — if code and spec disagree, the HF reference wins.** |
| [`E2E.md`](E2E.md) | Real-weight results: parity, greedy decode, long context, sampling, LoRA, TP head-to-head. |
| [`MAXTEXT.md`](MAXTEXT.md) | The training half: fork, converter, launch spec. |
| [`REASONING-STRENGTH.md`](REASONING-STRENGTH.md) | `high` vs `xhigh`, 960 rollouts across three problems. |

## Status

**Serving — proven end to end on real weights.**

- Teacher-forced vs HF: **3949/3949 positions agree on argmax**; logits max-abs 1.43e-05
  (short prompts) to 1.19e-03 at 3609 tokens.
- Greedy decode on a spot v5p-8: **4/5 prompts token-for-token exact**. The fifth diverges
  on a genuine argmax tie — bit-identical logprobs (−1.4882723093032837), `top1_top2_gap = 0.0`.
- Paged KV under load: 5 concurrent requests spanning 6–3609 tokens returned ids identical
  to sequential; 8 concurrent × 64 tokens at 513.5 tok/s.
- **Max working context 32768.** A needle and a second planted fact both recovered at
  16,320 tokens — the ladder ran out, not the model.
- Temperature > 0: top-1 matches HF on all 5 prompts, top-16 overlap 16/16, no degeneration
  at 0.7–1.0. No `output_multiplier` / softcap bug.

**Training — LoRA smoke passes.** 56 adapter arrays / 191,692,800 params on
`self_attention` + `mlp`, `fsdp_applied=True` (12.97 GiB/device), loss −0.8057 → −0.9298,
grad norms finite. Real weights converted to orbax and staged (40.58 GiB).

**Tinker client.** `MuseGlimmerRenderer` implements the ATEM channel protocol
(`to=self` reasoning, `to=user` answer), byte-exact against the shipped
`chat_template.jinja`, with `rosetta_stone.py` translating responses between the
muse_glimmer / qwen3 / gemma4 / gpt_oss families. 57 tests. Lives in the `discover`
submodule (`ttt_discover/tinker_utils/`).

**Not proven:** vision, quantization, TP ≠ 4 / multi-host, sustained serving.

## Serving configuration: prefer 2×TP=2 for throughput

Muse-Glimmer has **`num_key_value_heads = 2`**, and `tpu_inference/utils.py:230
get_padded_num_heads` pads KV heads *up* to the TP size whenever heads < shards. So
per-token KV **doubles every time TP doubles** until TP reaches 2 — you pay to store the
same two heads replicated across more chips.

Measured on ONE v5p-8 (= 4 chips; a v5p host is 4 chips), two TP=2 engines coexisting on
chips `{0,1}` and `{2,3}` via `TPU_VISIBLE_CHIPS`, launcher untouched:

| | 1 × TP=4 | 2 × TP=2 |
|---|---|---|
| per-token KV bytes | 106,496 | **53,248** |
| KV pool (tokens) | 3,025,856 | **5,012,160** |
| max concurrent @ 16k | 184.68 | **305.92** (1.656×) |
| decode, batch 1 | **69.76 tok/s** | 40.28 tok/s (1.73× to wide) |
| HBM / chip | 13.06 / 95.74 GiB | 25.94 / 95.74 GiB |

Two weight replicas cost 25.94 GiB/chip against 95.74 available — comfortable. Capacity
rises **1.656×**, not 2×, because the split pays for that second replica; a closed-form
prediction from measured constants matched to **0.013%**.

**Split for throughput, stay wide for latency.** Batch-1 decode is 1.73× faster on the
single wide engine, since weights shard further and decode is HBM-bandwidth bound.

Two caveats. A saturated-throughput measurement came out 6.687× in favour of the split —
**do not bank it**: the bandwidth model predicts the opposite sign, it is one unreplicated
point, and `SKIP_JAX_PRECOMPILE=1` may have put XLA compilation inside the timed window.
And this is **model-specific**: Qwen3.5-27B (4 KV heads) and gemma-4-31b (16, plus 4
global) already sit at zero padding at TP=4, so splitting them buys nothing and costs a
replica.

## Things that cost real time

- **`uv` applies this repo's `[tool.uv] override-dependencies` transformers≤5.8.0 pin even
  to an explicit `transformers @ git+…@main` installed with `--no-deps
  --force-reinstall`**, silently yielding a transformers that cannot parse `muse_glimmer`.
  Use `UV_NO_CONFIG=1` for the HF reference environment.
- `num_key_value_heads=2 < TP=4` needs **`repeat_interleave`, not `tile`**, for KV
  replication. Invisible on tiny random weights, catastrophic on real ones.
- LoRA name collisions: vLLM matches `target_modules` on the **last dotted component**, so
  `self_attn.gate_proj` collided with the MLP's and was renamed `attn_gate_proj`. In
  MaxText the attribute must be `self_attention` — olmo-3's `attention` yields zero
  adapters *silently*.
- `lm_head` is **NOT tied** despite `_tied_weights_keys` (the tie is gated on the config
  flag). Measured `max|lm_head − embed| = 3.09`.
- `generation_config.json` lists **two** eos ids — `<|end_of_text|>` (200001) and `<|eot|>`
  (200008). Treating only one as a stop marks the others truncated, and truncated
  responses are dropped as format errors — losing rollouts silently.
- QR deletion needed **two re-issues** (the QR still described as `ACTIVE` ~2 min after the
  first request). Always re-issue and verify.

## Rejected: the hybrid sliding KV cache

Sliding layers only need `sliding_window` worth of KV, so giving the 39 sliding layers
their own `SlidingWindowSpec` looks like free capacity. It was implemented, measured, and
**reverted** (`runs/muse_glimmer/reverted-hybrid-kv.patch`):

- KV capacity went **3,025,664 → 2,890,369 tokens — a 4.5% regression.** The saving is
  bounded by `max_model_len / sliding_window`, only 2 at 4096.
- **It is incorrect under concurrency.** A 27-point boundary sweep (16 … 2046/2047/2048/
  2049/2050/2052/2056 … 3609) was clean, but with 5 concurrent prompts one diverged at
  generated token 45 of 80 — *inside* the window, so block-pool bookkeeping, not
  windowing. Baseline was 5/5 identical on the same slice.

Note vLLM groups layers by the **repeating pattern**, not by spec type: `[S,S,S,F]×13`
becomes **4 groups of 13 → 13 KVCacheTensors**, so a model whose forward indexes
`kv_caches[i]` by absolute layer index raises `IndexError` at `i=13`. Any retry must make
the forward group-aware first.

**Single-stream boundary testing cannot catch this class of bug. Test concurrent.**
